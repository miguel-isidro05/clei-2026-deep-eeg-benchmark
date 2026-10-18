"""Portable integrity audit for Exp04 result cells."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.partitions import DISCOVERY_SUBJECTS, HOLDOUT_SUBJECTS

from .manifest import Exp04Manifest

FORBIDDEN_KEYS = {
    "y_test",
    "test_accuracy",
    "test_metrics",
    "outer_test_predictions",
    "outer_test_indices",
}
METRIC_NAMES = {
    "accuracy",
    "kappa",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
}
EXPECTED_STRESS_MASKS = {"missing_1": 15, "missing_2": 105}


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value] + [
            nested for child in value.values() for nested in _walk_keys(child)
        ]
    if isinstance(value, list):
        return [nested for child in value for nested in _walk_keys(child)]
    return []


def _validate_metrics(
    metrics: dict[str, float], y_true: list[int], y_pred: list[int], y_score: list[float]
) -> None:
    if set(metrics) != METRIC_NAMES:
        raise ValueError("Metrics do not contain the fixed binary metric set")
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if len(true) < 1 or not (len(true) == len(pred) == len(score)):
        raise ValueError("Prediction arrays have inconsistent lengths")
    if len(set(true.tolist())) != 2:
        raise ValueError("Ground truth must contain both classes")
    if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("Scores must be finite probabilities")
    recomputed = compute_metrics(true, pred, score)
    if any(
        not math.isclose(float(metrics[name]), float(recomputed[name]), abs_tol=1e-9, rel_tol=1e-9)
        for name in METRIC_NAMES
    ):
        raise ValueError("Stored metrics do not match predictions")


def _signed_payload(value: Any, label: str) -> str:
    if not isinstance(value, dict) or "sha256" not in value:
        raise ValueError(f"Cell is missing signed {label}")
    body = dict(value)
    stored = str(body.pop("sha256"))
    if stored != canonical_sha256(body):
        raise ValueError(f"Cell {label} signature is invalid")
    return stored


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_checkpoints(
    output_dir: Path, cell_id: str, payload: dict[str, Any]
) -> set[str]:
    expected_files: set[str] = set()
    candidate_sha256 = str(payload["candidate"]["sha256"])
    source_sha256 = str(payload["source"]["sha256"])
    for report in payload["fold_reports"]:
        expected_name = (
            f"{cell_id}__session-{report['session']}__fold{report['fold']}.pt"
        )
        if report.get("checkpoint_file") != expected_name:
            raise ValueError(f"Checkpoint filename mismatch in {cell_id}")
        path = output_dir / "checkpoints" / expected_name
        if not path.is_file():
            raise ValueError(f"Missing candidate checkpoint: {expected_name}")
        if _file_sha256(path) != report.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint digest mismatch: {expected_name}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("schema_version") != 4
            or checkpoint.get("cell_id") != cell_id
            or checkpoint.get("session") != report["session"]
            or checkpoint.get("fold") != report["fold"]
            or checkpoint.get("source_sha256") != source_sha256
            or checkpoint.get("candidate", {}).get("sha256") != candidate_sha256
            or checkpoint.get("checkpoint_policy") != "last_epoch"
            or checkpoint.get("selected_epoch") != report["selected_epoch"]
            or checkpoint.get("best_diagnostic_epoch") != report["best_diagnostic_epoch"]
            or checkpoint.get("trajectory_sha256") != report["trajectory_sha256"]
            or not isinstance(checkpoint.get("state_dict"), dict)
            or not checkpoint["state_dict"]
        ):
            raise ValueError(f"Checkpoint metadata mismatch: {expected_name}")
        expected_files.add(expected_name)
    return expected_files


def validate_result_cell(payload: dict[str, Any], *, allow_dirty: bool = False) -> None:
    leaked = FORBIDDEN_KEYS & set(_walk_keys(payload))
    if leaked:
        raise ValueError(f"Cell contains forbidden outer-test material: {sorted(leaked)}")
    required = {
        "schema_version",
        "status",
        "experiment",
        "cell_id",
        "kind",
        "model_id",
        "config_id",
        "subject",
        "seed",
        "folds",
        "validation_metrics",
        "validation_y_true",
        "validation_y_pred",
        "validation_y_score",
        "quality_flags",
        "fold_reports",
        "stress_results",
        "fingerprint",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Cell schema is missing keys: {sorted(missing)}")
    if payload["schema_version"] != 4 or payload["status"] != "completed":
        raise ValueError("Cell is not a completed Exp04 result")
    if payload["experiment"] != "exp04":
        raise ValueError("Cell experiment is invalid")
    if payload["subject"] in HOLDOUT_SUBJECTS or payload["subject"] not in DISCOVERY_SUBJECTS:
        raise ValueError("Cell subject violates the discovery/holdout boundary")
    if payload["quality_flags"] != sorted(set(payload["quality_flags"])):
        raise ValueError("Cell quality flags must be sorted and unique")
    _validate_metrics(
        payload["validation_metrics"],
        payload["validation_y_true"],
        payload["validation_y_pred"],
        payload["validation_y_score"],
    )
    expected_flags: set[str] = set()
    expected_stress_labels = {"missing_1": [], "missing_2": []}
    stress_fold_shapes = {"missing_1": [], "missing_2": []}
    offset = 0
    for report in payload["fold_reports"]:
        count = int(report.get("n_validation_trials", 0))
        if count < 1:
            raise ValueError("Fold validation count is invalid")
        fold_true = payload["validation_y_true"][offset : offset + count]
        fold_pred = payload["validation_y_pred"][offset : offset + count]
        flags = validation_quality_flags(fold_true, fold_pred)
        if report.get("quality_flags") != flags:
            raise ValueError("Fold quality flags do not match predictions")
        expected_flags.update(flags)
        if payload["kind"] == "candidate" and (
            report.get("checkpoint_policy") != "last_epoch"
            or report.get("selected_epoch") != report.get("last_epoch")
        ):
            raise ValueError("Candidate metrics must use the fixed last epoch")
        for condition in expected_stress_labels:
            mask_count = int(report.get("stress", {}).get(condition, {}).get("mask_count", 0))
            if mask_count < 1:
                raise ValueError(f"Fold report is missing {condition} mask coverage")
            expected_stress_labels[condition].extend(fold_true * mask_count)
            stress_fold_shapes[condition].append((count, mask_count))
        offset += count
    if offset != len(payload["validation_y_true"]):
        raise ValueError("Fold reports do not cover all predictions")
    if payload["quality_flags"] != sorted(expected_flags):
        raise ValueError("Cell quality flags do not match fold flags")
    for condition in ("missing_1", "missing_2"):
        stress = payload["stress_results"].get(condition)
        if not isinstance(stress, dict):
            raise ValueError(f"Cell is missing {condition} stress results")
        _validate_metrics(stress["metrics"], stress["y_true"], stress["y_pred"], stress["y_score"])
        if stress["y_true"] != expected_stress_labels[condition]:
            raise ValueError(f"Cell {condition} labels do not cover every declared mask")
        expected_mask_count = EXPECTED_STRESS_MASKS[condition]
        declared_counts = {mask_count for _, mask_count in stress_fold_shapes[condition]}
        if declared_counts != {expected_mask_count}:
            raise ValueError(
                f"Cell {condition} must cover exactly {expected_mask_count} masks per fold"
            )
        if int(stress.get("n_masks_per_fold", 0)) != expected_mask_count:
            raise ValueError(f"Cell {condition} mask counts are inconsistent")
        cursor = 0
        mask_accuracies: list[float] = []
        for count, mask_count in stress_fold_shapes[condition]:
            for _ in range(mask_count):
                true = np.asarray(stress["y_true"][cursor : cursor + count])
                pred = np.asarray(stress["y_pred"][cursor : cursor + count])
                mask_accuracies.append(float(np.mean(true == pred)))
                cursor += count
        if cursor != len(stress["y_true"]):
            raise ValueError(f"Cell {condition} stress blocks have invalid lengths")
        if not math.isclose(
            float(stress.get("worst_mask_accuracy", -1.0)),
            min(mask_accuracies),
            abs_tol=1e-12,
            rel_tol=1e-12,
        ):
            raise ValueError(f"Cell {condition} worst-mask accuracy is invalid")
    fingerprint = payload["fingerprint"]
    expected_split_hash = canonical_sha256(
        [str(report["search_split_sha256"]) for report in payload["fold_reports"]]
    )
    if fingerprint.get("split_sha256") != expected_split_hash:
        raise ValueError("Fingerprint split hash does not match fold reports")
    for key in (
        "config_sha256",
        "catalog_sha256",
        "manifest_sha256",
        "source_sha256",
        "code_sha256",
        "data_sha256",
        "split_sha256",
        "environment_sha256",
    ):
        if len(str(fingerprint.get(key, ""))) != 64:
            raise ValueError(f"Fingerprint {key} is invalid")
    source_hash = _signed_payload(payload.get("source"), "source")
    if fingerprint["source_sha256"] != source_hash:
        raise ValueError("Cell source signature differs from its fingerprint")
    if payload["kind"] == "candidate":
        candidate_hash = _signed_payload(payload.get("candidate"), "candidate")
        if fingerprint.get("cell_config_sha256") != candidate_hash:
            raise ValueError("Candidate signature differs from its fingerprint")
    if not allow_dirty and fingerprint.get("git_dirty") is not False:
        raise ValueError("Full result cell was produced from a dirty Git tree")


def audit_results(
    output_dir: Path, manifest: Exp04Manifest, *, allow_dirty: bool = False
) -> dict[str, Any]:
    paths = sorted((output_dir / "cells").glob("*.json"))
    expected = {cell.cell_id: cell for cell in manifest.cells}
    found = {path.stem: path for path in paths}
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(f"Result set mismatch: missing={len(missing)} extra={len(extra)}")
    code_hashes: set[str] = set()
    environment_hashes: set[str] = set()
    expected_checkpoints: set[str] = set()
    for cell_id, path in found.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_result_cell(payload, allow_dirty=allow_dirty)
        cell = expected[cell_id]
        if (
            payload["cell_id"] != cell.cell_id
            or payload["kind"] != cell.kind
            or payload["model_id"] != cell.model_id
            or payload["config_id"] != cell.config_id
            or payload["subject"] != cell.subject
            or payload["seed"] != cell.seed
            or tuple(payload["folds"]) != cell.folds
        ):
            raise ValueError(f"Cell identity mismatch in {cell_id}")
        fingerprint = payload["fingerprint"]
        required = {
            "config_sha256": manifest.config_sha256,
            "catalog_sha256": manifest.catalog_sha256,
            "manifest_sha256": manifest.sha256,
            "source_sha256": manifest.source_sha256,
            "revision": manifest.revision,
        }
        if any(fingerprint.get(key) != value for key, value in required.items()):
            raise ValueError(f"Run fingerprint mismatch in {cell_id}")
        if fingerprint.get("cell_config_sha256") != cell.config_sha256:
            raise ValueError(f"Cell config fingerprint mismatch in {cell_id}")
        if cell.kind == "candidate":
            expected_checkpoints.update(
                validate_candidate_checkpoints(output_dir, cell_id, payload)
            )
        code_hashes.add(str(fingerprint["code_sha256"]))
        environment_hashes.add(str(fingerprint["environment_sha256"]))
    if len(code_hashes) != 1:
        raise ValueError("Results mix multiple Exp04 code identities")
    if len(environment_hashes) != 1:
        raise ValueError("Results mix multiple Exp04 scientific environments")
    found_checkpoints = {
        path.name for path in (output_dir / "checkpoints").glob("*.pt")
    }
    if found_checkpoints != expected_checkpoints:
        raise ValueError(
            "Checkpoint set mismatch: "
            f"missing={len(expected_checkpoints - found_checkpoints)} "
            f"extra={len(found_checkpoints - expected_checkpoints)}"
        )
    return {
        "complete": True,
        "expected": len(expected),
        "found": len(found),
        "code_sha256": next(iter(code_hashes)),
        "environment_sha256": next(iter(environment_hashes)),
    }
