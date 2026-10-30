"""Strict integrity, convergence, and pair-identity audit for Exp08."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.partitions import DISCOVERY_SUBJECTS, HOLDOUT_SUBJECTS
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.audit import (
    _confusion_matrix,
    _validate_fold_metrics,
    _validate_preprocessing,
    _validate_split,
    _validate_training_trajectory,
)

from .catalog import all_candidates
from .config import Exp08Config
from .manifest import (
    Exp08Cell,
    Exp08Manifest,
    build_confirm_manifest,
    build_screen_manifest,
    manifest_from_payload,
)


def _validate_deep_fold(
    fold: dict[str, Any], config: Exp08Config, *, aggregation: str, seed: int
) -> None:
    _validate_split(fold)
    refit = fold.get("refit_preprocessing")
    if not isinstance(refit, dict) or refit.get("fit_partition") != "outer_train":
        raise ValueError("Exp08 preprocessing was not fit on outer train")
    if fold.get("selected_epoch") is not None:
        raise ValueError("Exp08 deep fold reintroduced epoch selection")
    if fold.get("selection_method") != "fixed_convergence_no_validation_selection":
        raise ValueError("Exp08 deep fold has the wrong convergence policy")
    if int(fold.get("backbone_epochs", 0)) != config.backbone_epochs:
        raise ValueError("Exp08 backbone did not run the fixed epoch budget")
    history = fold.get("backbone_history")
    if not isinstance(history, list) or len(history) != config.backbone_epochs:
        raise ValueError("Exp08 backbone history is incomplete")
    if [int(item.get("epoch", 0)) for item in history] != list(
        range(1, config.backbone_epochs + 1)
    ):
        raise ValueError("Exp08 backbone epochs are not sequential")
    previous_updates = 0
    for item in history:
        values = [
            float(item.get("train_loss", float("nan"))),
            float(item.get("learning_rate", float("nan"))),
        ]
        if not np.isfinite(values).all():
            raise ValueError("Exp08 backbone history contains non-finite values")
        updates = int(item.get("updates", 0))
        if updates <= previous_updates:
            raise ValueError("Exp08 backbone update count is not increasing")
        previous_updates = updates
    if int(fold.get("backbone_updates", 0)) != previous_updates:
        raise ValueError("Exp08 backbone update total is inconsistent")
    if int(fold.get("n_train_examples", 0)) != 6 * int(fold.get("n_train_trials", -1)):
        raise ValueError("Exp08 did not train crop-wise on six windows per trial")
    if int(fold.get("windows_per_test_trial", 0)) != 6:
        raise ValueError("Exp08 did not evaluate six windows per trial")
    if fold.get("fusion_fit_partition") != "outer_train_only":
        raise ValueError("Exp08 fusion saw a non-training partition")
    before = str(fold.get("backbone_state_sha256", ""))
    after = str(fold.get("backbone_state_after_fusion_sha256", ""))
    if len(before) != 64 or before != after:
        raise ValueError("Exp08 fusion modified its frozen backbone")
    trajectory = {
        "seed": seed,
        "epochs": config.backbone_epochs,
        "updates": previous_updates,
        "history": history,
        "state_sha256": before,
    }
    if fold.get("backbone_trajectory_sha256") != canonical_sha256(trajectory):
        raise ValueError("Exp08 backbone trajectory signature is invalid")
    diagnostics = fold.get("model_diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("backbone_frozen_during_fusion") is not True
    ):
        raise ValueError("Exp08 fold lacks the frozen-backbone diagnostic")
    weights = np.asarray(diagnostics.get("window_weights"), dtype=float)
    if weights.shape != (6,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Exp08 fold has invalid window weights")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("Exp08 window weights are not normalized")
    if fold.get("aggregation") != aggregation or diagnostics.get("aggregation") != aggregation:
        raise ValueError("Exp08 fold aggregation is mislabeled")
    fusion_history = fold.get("fusion_history")
    expected_parameters = 0 if aggregation == "uniform" else 6
    expected_epochs = 0 if aggregation == "uniform" else config.fusion_epochs
    if int(fold.get("fusion_trainable_parameters", -1)) != expected_parameters:
        raise ValueError("Exp08 fusion parameter count is invalid")
    if int(fold.get("fusion_epochs", -1)) != expected_epochs:
        raise ValueError("Exp08 fusion epoch count is invalid")
    if not isinstance(fusion_history, list) or len(fusion_history) != expected_epochs:
        raise ValueError("Exp08 fusion history length is invalid")
    logits = np.asarray(diagnostics.get("window_logits"), dtype=float)
    if logits.shape != (6,) or not np.isfinite(logits).all():
        raise ValueError("Exp08 window logits are invalid")
    if aggregation == "uniform":
        if not np.allclose(logits, 0.0) or not np.allclose(weights, 1.0 / 6.0):
            raise ValueError("Exp08 uniform aggregation is not exactly uniform")
    else:
        for index, item in enumerate(fusion_history, start=1):
            if int(item.get("epoch", 0)) != index:
                raise ValueError("Exp08 fusion epochs are not sequential")
            values = [
                float(item.get("train_loss", float("nan"))),
                float(item.get("learning_rate", float("nan"))),
            ]
            if not np.isfinite(values).all():
                raise ValueError("Exp08 fusion history contains non-finite values")


def _validate_cell(
    payload: dict[str, Any],
    expected: Exp08Cell,
    manifest: Exp08Manifest,
    candidate: Any,
    config: Exp08Config,
) -> None:
    signed_body = dict(payload)
    payload_signature = str(signed_body.pop("sha256", ""))
    if payload_signature != canonical_sha256(signed_body):
        raise ValueError(f"Cell {expected.cell_id} signature is invalid")
    required = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp08",
        "phase": expected.phase,
        "cell_id": expected.cell_id,
        "config_id": expected.config_id,
        "mechanism": expected.mechanism,
        "kind": expected.kind,
        "subject": expected.subject,
        "seed": expected.seed,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError(f"Cell {expected.cell_id} differs from its signed manifest")
    if expected.subject in HOLDOUT_SUBJECTS or expected.subject not in DISCOVERY_SUBJECTS:
        raise ValueError("Exp08 cell violates the holdout boundary")
    candidate_payload = dict(payload.get("candidate", {}))
    signature = str(candidate_payload.pop("sha256", ""))
    if signature != candidate.sha256 or signature != canonical_sha256(candidate_payload):
        raise ValueError("Exp08 candidate identity drifted")
    fingerprint = payload.get("fingerprint", {})
    for key, value in {
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "manifest_sha256": manifest.sha256,
        "parent_decision_sha256": manifest.parent_decision_sha256,
        "screen_decision_sha256": manifest.screen_decision_sha256,
        "cell_config_sha256": expected.config_sha256,
        "candidate_sha256": candidate.sha256,
        "revision": manifest.revision,
    }.items():
        if fingerprint.get(key) != value:
            raise ValueError(f"Exp08 fingerprint mismatch: {key}")
    if config.profile == "full" and fingerprint.get("git_dirty") is not False:
        raise ValueError("Full Exp08 result came from a dirty tree")
    for name in ("code_sha256", "data_sha256"):
        value = fingerprint.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Exp08 fingerprint has invalid {name}")
    runtime = payload.get("runtime")
    environment = runtime.get("environment") if isinstance(runtime, dict) else None
    if not isinstance(environment, dict):
        raise ValueError("Exp08 runtime environment is missing")
    if fingerprint.get("environment_sha256") != canonical_sha256(environment):
        raise ValueError("Exp08 environment fingerprint drifted")
    if fingerprint.get("device") != environment.get("device"):
        raise ValueError("Exp08 device fingerprint drifted")
    reports = payload.get("fold_reports")
    if not isinstance(reports, list) or len(reports) != len(expected.folds):
        raise ValueError("Exp08 cell has incomplete folds")
    if tuple(int(fold.get("outer_fold", -1)) for fold in reports) != expected.folds:
        raise ValueError("Exp08 fold order drifted")
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    flags: set[str] = set()
    split_hashes: list[str] = []
    outer_tests: list[set[int]] = []
    trial_universe: set[int] | None = None
    for index, fold in enumerate(reports):
        if candidate.backbone == "fbcnet":
            _validate_split(fold)
            _validate_preprocessing(fold)
            _validate_training_trajectory(
                fold,
                config_id=expected.config_id,
                cell_seed=int(expected.seed),
                split_index=index,
            )
        else:
            _validate_deep_fold(
                fold,
                config,
                aggregation=candidate.aggregation,
                seed=int(expected.seed) * 100_000 + index,
            )
        y_true, y_pred, y_score = _validate_fold_metrics(fold)
        recomputed = validation_quality_flags(y_true.tolist(), y_pred.tolist())
        if fold.get("quality_flags") != sorted(set(recomputed)):
            raise ValueError("Exp08 fold quality flags drifted")
        flags.update(recomputed)
        labels.append(y_true)
        predictions.append(y_pred)
        scores.append(y_score)
        split = fold["split"]
        outer_train = set(map(int, split["outer_train_indices"]))
        outer_test = set(map(int, split["outer_test_indices"]))
        universe = outer_train | outer_test
        if trial_universe is None:
            trial_universe = universe
        elif universe != trial_universe:
            raise ValueError("Exp08 folds use different trial universes")
        if any(outer_test & previous for previous in outer_tests):
            raise ValueError("Exp08 outer-test folds overlap")
        outer_tests.append(outer_test)
        split_hashes.append(
            canonical_sha256(
                {
                    "session": fold["session"],
                    "fold": int(fold["outer_fold"]),
                    "inner_train_indices": split["inner_train_indices"],
                    "inner_validation_indices": split["inner_validation_indices"],
                    "outer_train_indices": split["outer_train_indices"],
                    "outer_test_indices": split["outer_test_indices"],
                    "outer_sha256": split["outer_sha256"],
                    "inner_sha256": split["inner_sha256"],
                }
            )
        )
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    y_score = np.concatenate(scores)
    if payload.get("y_true") != y_true.astype(int).tolist():
        raise ValueError("Exp08 aggregate labels drifted")
    if payload.get("y_pred") != y_pred.astype(int).tolist():
        raise ValueError("Exp08 aggregate predictions drifted")
    if not np.array_equal(np.asarray(payload.get("y_score"), dtype=float), y_score):
        raise ValueError("Exp08 aggregate scores drifted")
    if payload.get("confusion_matrix") != _confusion_matrix(y_true, y_pred):
        raise ValueError("Exp08 aggregate confusion matrix drifted")
    for name, value in compute_metrics(y_true, y_pred, y_score).items():
        if not np.isclose(float(payload["aggregate_metrics"][name]), value, atol=1e-9):
            raise ValueError(f"Exp08 aggregate metric drifted: {name}")
    if payload.get("quality_flags") != sorted(flags):
        raise ValueError("Exp08 aggregate quality flags drifted")
    if trial_universe is None:
        raise ValueError("Exp08 cell has no trial universe")
    if len(expected.folds) == 5 and set().union(*outer_tests) != trial_universe:
        raise ValueError("Exp08 outer-test folds do not partition the trial universe")
    if fingerprint.get("split_sha256") != canonical_sha256(split_hashes):
        raise ValueError("Exp08 split fingerprint drifted")
    parameter_report = payload.get("parameter_report")
    if not isinstance(parameter_report, dict) or int(parameter_report.get("total", 0)) < 1:
        raise ValueError("Exp08 parameter report is invalid")
    if candidate.backbone != "fbcnet":
        expected_fusion = 0 if candidate.aggregation == "uniform" else 6
        if int(parameter_report.get("trainable_stage2_fusion", -1)) != expected_fusion:
            raise ValueError("Exp08 parameter report has the wrong fusion size")
        if int(parameter_report.get("trainable_at_inference", -1)) != 0:
            raise ValueError("Exp08 inference parameter state is mislabeled")


def audit_phase(
    output_dir: Path,
    config: Exp08Config,
    catalog: tuple[Any, ...],
    *,
    phase: str,
    revision: str,
    parent_decision_sha256: str,
) -> dict[str, Any]:
    path = output_dir / "manifests" / f"{phase}-manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing Exp08 {phase} manifest")
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
        body = dict(decision)
        signature = str(body.pop("sha256", ""))
        if signature != canonical_sha256(body):
            raise ValueError("Exp08 screen decision signature is invalid")
        expected_manifest = build_confirm_manifest(
            config,
            catalog,
            promoted_backbones=tuple(map(str, decision["selected_backbones"])),
            revision=revision,
            parent_decision_sha256=parent_decision_sha256,
            screen_decision_sha256=signature,
        )
    if manifest != expected_manifest:
        raise ValueError("Exp08 manifest differs from the frozen contract")
    paths = sorted((output_dir / f"{phase}_cells").glob("*.json"))
    if len(paths) != len(manifest.cells):
        raise ValueError(
            f"Exp08 {phase} incomplete: expected={len(manifest.cells)} found={len(paths)}"
        )
    candidates = {item.config_id: item for item in all_candidates()}
    pair_hashes: dict[tuple[str, int, str, int], dict[str, str]] = defaultdict(dict)
    code_hashes: set[str] = set()
    data_hashes: dict[str, set[str]] = defaultdict(set)
    environment_hashes: dict[str, set[str]] = defaultdict(set)
    for expected in manifest.cells:
        payload = json.loads(
            (output_dir / f"{phase}_cells" / f"{expected.cell_id}.json").read_text(
                encoding="utf-8"
            )
        )
        candidate = candidates[expected.config_id]
        _validate_cell(payload, expected, manifest, candidate, config)
        fingerprint = payload["fingerprint"]
        code_hashes.add(str(fingerprint["code_sha256"]))
        data_hashes[expected.subject].add(str(fingerprint["data_sha256"]))
        environment_hashes[expected.subject].add(str(fingerprint["environment_sha256"]))
        if candidate.backbone != "fbcnet":
            for fold in payload["fold_reports"]:
                key = (expected.subject, expected.seed, candidate.family, int(fold["outer_fold"]))
                pair_hashes[key][candidate.aggregation] = str(fold["backbone_state_sha256"])
    if len(code_hashes) != 1:
        raise ValueError("Exp08 cells were produced by different code surfaces")
    if any(len(values) != 1 for values in data_hashes.values()):
        raise ValueError("Exp08 subject data hashes drifted")
    if any(len(values) != 1 for values in environment_hashes.values()):
        raise ValueError("Exp08 subject environments drifted")
    for key, hashes in pair_hashes.items():
        if set(hashes) != {"uniform", "lpwf"} or len(set(hashes.values())) != 1:
            raise ValueError(f"Exp08 pair does not share one backbone: {key}")
    if {cell.subject for cell in manifest.cells} & set(config.holdout_subjects):
        raise ValueError("Exp08 manifest opened the holdout")
    return {
        "phase": phase,
        "expected": len(manifest.cells),
        "found": len(paths),
        "manifest_sha256": manifest.sha256,
        "code_sha256": next(iter(code_hashes)),
        "shared_backbone_pairs": len(pair_hashes),
        "complete": True,
    }

