"""Strict integrity audit for Exp07 manifests and result cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.partitions import (
    DISCOVERY_SUBJECTS,
    HOLDOUT_SUBJECTS,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.audit import (
    _confusion_matrix,
    _validate_fold_metrics,
    _validate_preprocessing,
    _validate_split,
    _validate_training_trajectory,
)

from .catalog import all_candidates
from .config import Exp07Config
from .manifest import (
    Exp07Cell,
    Exp07Manifest,
    build_confirm_manifest,
    build_screen_manifest,
    manifest_from_payload,
)


def _close(left: Any, right: Any, *, atol: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=0.0))


def _validate_cell(
    payload: dict[str, Any],
    expected: Exp07Cell,
    manifest: Exp07Manifest,
    candidate: Any,
    *,
    allow_dirty: bool,
) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError(f"Cell {expected.cell_id} has an invalid schema")
    for key, value in {
        "status": "completed",
        "experiment": "exp07",
        "phase": expected.phase,
        "cell_id": expected.cell_id,
        "config_id": expected.config_id,
        "mechanism": expected.mechanism,
        "kind": expected.kind,
        "subject": expected.subject,
        "seed": expected.seed,
    }.items():
        if payload.get(key) != value:
            raise ValueError(f"Cell {expected.cell_id} has invalid {key}")
    if expected.subject in HOLDOUT_SUBJECTS or expected.subject not in DISCOVERY_SUBJECTS:
        raise ValueError(f"Cell {expected.cell_id} violates the holdout boundary")
    candidate_payload = dict(payload.get("candidate", {}))
    candidate_signature = str(candidate_payload.pop("sha256", ""))
    if candidate_signature != candidate.sha256 or candidate_signature != canonical_sha256(
        candidate_payload
    ):
        raise ValueError(f"Cell {expected.cell_id} candidate identity drifted")
    fingerprint = payload.get("fingerprint", {})
    required_fingerprint = {
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "manifest_sha256": manifest.sha256,
        "parent_decision_sha256": manifest.parent_decision_sha256,
        "screen_decision_sha256": manifest.screen_decision_sha256,
        "cell_config_sha256": expected.config_sha256,
        "candidate_sha256": candidate.sha256,
        "revision": manifest.revision,
    }
    for key, value in required_fingerprint.items():
        if fingerprint.get(key) != value:
            raise ValueError(f"Cell {expected.cell_id} fingerprint mismatch: {key}")
    if not allow_dirty and fingerprint.get("git_dirty") is not False:
        raise ValueError(f"Cell {expected.cell_id} was generated from a dirty tree")

    reports = payload.get("fold_reports")
    if not isinstance(reports, list) or len(reports) != len(expected.folds):
        raise ValueError(f"Cell {expected.cell_id} has incomplete fold reports")
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    expected_flags: set[str] = set()
    nested_split_hashes: list[str] = []
    outer_tests: list[set[int]] = []
    trial_universe: set[int] | None = None
    if tuple(int(report.get("outer_fold", -1)) for report in reports) != expected.folds:
        raise ValueError(f"Cell {expected.cell_id} fold order differs from its manifest")
    for split_index, report in enumerate(reports):
        _validate_split(report)
        _validate_preprocessing(report)
        _validate_training_trajectory(
            report,
            config_id=expected.config_id,
            cell_seed=int(expected.seed),
            split_index=split_index,
        )
        y_true, y_pred, y_score = _validate_fold_metrics(report)
        recomputed_flags = validation_quality_flags(y_true.tolist(), y_pred.tolist())
        if report.get("quality_flags") != sorted(set(recomputed_flags)):
            raise ValueError(f"Cell {expected.cell_id} fold quality flags drifted")
        expected_flags.update(recomputed_flags)
        labels.append(y_true)
        predictions.append(y_pred)
        scores.append(y_score)
        split = report["split"]
        outer_train = set(map(int, split["outer_train_indices"]))
        outer_test = set(map(int, split["outer_test_indices"]))
        fold_universe = outer_train | outer_test
        if trial_universe is None:
            trial_universe = fold_universe
        elif fold_universe != trial_universe:
            raise ValueError(f"Cell {expected.cell_id} folds use different trial universes")
        if any(outer_test & previous for previous in outer_tests):
            raise ValueError(f"Cell {expected.cell_id} outer test folds overlap")
        outer_tests.append(outer_test)
        nested_split_hashes.append(
            canonical_sha256(
                {
                    "session": report["session"],
                    "fold": int(report["outer_fold"]),
                    "inner_train_indices": split["inner_train_indices"],
                    "inner_validation_indices": split["inner_validation_indices"],
                    "outer_train_indices": split["outer_train_indices"],
                    "outer_test_indices": split["outer_test_indices"],
                    "outer_sha256": split["outer_sha256"],
                    "inner_sha256": split["inner_sha256"],
                }
            )
        )
        if candidate.params.get("grouped_windows"):
            diagnostics = report.get("model_diagnostics")
            if not isinstance(diagnostics, dict) or "window_weights" not in diagnostics:
                raise ValueError(f"Cell {expected.cell_id} is missing window weights")
            weights = np.asarray(diagnostics["window_weights"], dtype=float)
            if weights.ndim != 1 or len(weights) != int(candidate.params["max_windows"]):
                raise ValueError(f"Cell {expected.cell_id} has invalid window-weight shape")
            if not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError(f"Cell {expected.cell_id} has invalid window weights")
            if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                raise ValueError(f"Cell {expected.cell_id} window weights are not normalized")
    if trial_universe is None:
        raise ValueError(f"Cell {expected.cell_id} has no trial universe")
    if len(expected.folds) == 5 and set().union(*outer_tests) != trial_universe:
        raise ValueError(f"Cell {expected.cell_id} folds do not partition the trial universe")
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    y_score = np.concatenate(scores)
    if (
        payload.get("y_true") != y_true.astype(int).tolist()
        or payload.get("y_pred") != y_pred.astype(int).tolist()
    ):
        raise ValueError(f"Cell {expected.cell_id} aggregates differ from fold reports")
    if not np.array_equal(np.asarray(payload.get("y_score"), dtype=float), y_score):
        raise ValueError(f"Cell {expected.cell_id} scores differ from fold reports")
    for name, value in compute_metrics(y_true, y_pred, y_score).items():
        if not _close(payload.get("aggregate_metrics", {}).get(name), value):
            raise ValueError(f"Cell {expected.cell_id} metric mismatch: {name}")
    if payload.get("confusion_matrix") != _confusion_matrix(y_true, y_pred):
        raise ValueError(f"Cell {expected.cell_id} confusion matrix drifted")
    if payload.get("quality_flags") != sorted(expected_flags):
        raise ValueError(f"Cell {expected.cell_id} aggregate quality flags drifted")
    if fingerprint.get("split_sha256") != canonical_sha256(nested_split_hashes):
        raise ValueError(f"Cell {expected.cell_id} split fingerprint drifted")
    runtime = payload.get("runtime")
    environment = runtime.get("environment") if isinstance(runtime, dict) else None
    if not isinstance(environment, dict):
        raise ValueError(f"Cell {expected.cell_id} runtime environment is missing")
    if fingerprint.get("environment_sha256") != canonical_sha256(environment):
        raise ValueError(f"Cell {expected.cell_id} environment fingerprint drifted")
    if fingerprint.get("device") != environment.get("device"):
        raise ValueError(f"Cell {expected.cell_id} device fingerprint drifted")
    for name in ("code_sha256", "data_sha256"):
        value = fingerprint.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Cell {expected.cell_id} has invalid {name}")
    parameter_report = payload.get("parameter_report")
    if not isinstance(parameter_report, dict) or int(parameter_report.get("trainable", 0)) < 1:
        raise ValueError(f"Cell {expected.cell_id} parameter report is invalid")


def audit_phase(
    output_dir: Path,
    config: Exp07Config,
    catalog: tuple[Any, ...],
    *,
    phase: str,
    revision: str,
    parent_decision_sha256: str,
) -> dict[str, Any]:
    path = output_dir / "manifests" / f"{phase}-manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing Exp07 {phase} manifest")
    manifest = manifest_from_payload(json.loads(path.read_text(encoding="utf-8")))
    if phase == "screen":
        expected_manifest = build_screen_manifest(
            config,
            catalog,
            revision=revision,
            parent_decision_sha256=parent_decision_sha256,
        )
    else:
        decision = json.loads(
            (output_dir / "decisions" / "screen-promotion.json").read_text(encoding="utf-8")
        )
        decision_body = dict(decision)
        decision_signature = str(decision_body.pop("sha256", ""))
        if decision_signature != canonical_sha256(decision_body):
            raise ValueError("Exp07 screen decision signature is invalid")
        expected_manifest = build_confirm_manifest(
            config,
            catalog,
            promoted_backbones=tuple(map(str, decision["selected_backbones"])),
            revision=revision,
            parent_decision_sha256=parent_decision_sha256,
            screen_decision_sha256=decision_signature,
        )
    if manifest != expected_manifest:
        raise ValueError(f"Exp07 {phase} manifest differs from the frozen contract")
    paths = sorted((output_dir / f"{phase}_cells").glob("*.json"))
    if len(paths) != len(manifest.cells):
        raise ValueError(
            f"Exp07 {phase} incomplete: expected={len(manifest.cells)} found={len(paths)}"
        )
    candidates = {item.config_id: item for item in all_candidates()}
    code_hashes: set[str] = set()
    data_hashes: dict[str, set[str]] = {}
    for expected in manifest.cells:
        cell_path = output_dir / f"{phase}_cells" / f"{expected.cell_id}.json"
        if not cell_path.is_file():
            raise ValueError(f"Missing Exp07 cell: {cell_path}")
        payload = json.loads(cell_path.read_text(encoding="utf-8"))
        _validate_cell(
            payload,
            expected,
            manifest,
            candidates[expected.config_id],
            allow_dirty=config.profile != "full",
        )
        fingerprint = payload["fingerprint"]
        code_hashes.add(str(fingerprint["code_sha256"]))
        data_hashes.setdefault(expected.subject, set()).add(str(fingerprint["data_sha256"]))
    if len(code_hashes) != 1:
        raise ValueError(f"Exp07 {phase} cells were produced by different code surfaces")
    if any(len(values) != 1 for values in data_hashes.values()):
        raise ValueError(f"Exp07 {phase} subject data hashes drifted")
    if {cell.subject for cell in manifest.cells} & set(config.holdout_subjects):
        raise ValueError("Exp07 manifest opened the holdout")
    return {
        "phase": phase,
        "expected": len(manifest.cells),
        "found": len(paths),
        "manifest_sha256": manifest.sha256,
        "code_sha256": next(iter(code_hashes)),
        "data_sha256_by_subject": {
            subject: next(iter(values)) for subject, values in sorted(data_hashes.items())
        },
        "complete": True,
    }
