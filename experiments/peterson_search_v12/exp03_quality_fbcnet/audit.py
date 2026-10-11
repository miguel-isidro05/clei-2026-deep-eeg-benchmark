"""Portable integrity audit for exp03 recipe and quality result cells."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.partitions import DISCOVERY_SUBJECTS, HOLDOUT_SUBJECTS

from .manifest import Exp03Manifest

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


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _validate_metrics(payload: dict[str, Any]) -> None:
    metrics = payload["validation_metrics"]
    if set(metrics) != METRIC_NAMES:
        raise ValueError("Validation metrics do not contain the fixed metric set")
    y_true = np.asarray(payload["validation_y_true"], dtype=int)
    y_pred = np.asarray(payload["validation_y_pred"], dtype=int)
    y_score = np.asarray(payload["validation_y_score"], dtype=float)
    if len(y_true) < 1 or not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("Validation arrays have inconsistent lengths")
    if len(set(y_true.tolist())) != 2:
        raise ValueError("Validation ground truth must contain both classes")
    if not np.isfinite(y_score).all() or np.any((y_score < 0.0) | (y_score > 1.0)):
        raise ValueError("Validation scores must be finite probabilities")
    recomputed = compute_metrics(y_true, y_pred, y_score)
    if any(
        not math.isclose(float(metrics[name]), float(recomputed[name]), abs_tol=1e-9, rel_tol=1e-9)
        for name in METRIC_NAMES
    ):
        raise ValueError("Validation metrics do not match stored predictions")


def validate_result_cell(payload: dict[str, Any]) -> None:
    leaked = FORBIDDEN_KEYS & set(_walk_keys(payload))
    if leaked:
        raise ValueError(f"Cell contains forbidden outer-test material: {sorted(leaked)}")
    required = {
        "schema_version",
        "status",
        "experiment",
        "phase",
        "kind",
        "cell_id",
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
        "fingerprint",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Cell schema is missing keys: {sorted(missing)}")
    if payload["schema_version"] != 3 or payload["status"] != "completed":
        raise ValueError("Cell is not a completed exp03 result")
    if payload["experiment"] != "exp03" or payload["phase"] not in {"recipe", "quality"}:
        raise ValueError("Cell experiment or phase is invalid")
    if payload["subject"] in HOLDOUT_SUBJECTS or payload["subject"] not in DISCOVERY_SUBJECTS:
        raise ValueError("Cell subject violates the discovery/holdout boundary")
    if payload["quality_flags"] != sorted(set(payload["quality_flags"])):
        raise ValueError("Cell quality flags must be a sorted unique list")
    _validate_metrics(payload)
    y_true = list(payload["validation_y_true"])
    y_pred = list(payload["validation_y_pred"])
    reports = payload["fold_reports"]
    if len(reports) != len(payload["folds"]):
        raise ValueError("Cell must contain one report per fold")
    offset = 0
    expected_flags: set[str] = set()
    for report in reports:
        count = int(report.get("n_validation_trials", 0))
        if count < 1 or offset + count > len(y_true):
            raise ValueError("Fold validation counts are inconsistent")
        flags = validation_quality_flags(
            [int(value) for value in y_true[offset : offset + count]],
            [int(value) for value in y_pred[offset : offset + count]],
        )
        if report.get("quality_flags") != flags:
            raise ValueError("Fold quality flags do not match predictions")
        expected_flags.update(flags)
        if len(str(report.get("search_split_sha256", ""))) != 64:
            raise ValueError("Fold split hash is invalid")
        offset += count
    if offset != len(y_true):
        raise ValueError("Fold reports do not cover all predictions")
    if payload["quality_flags"] != sorted(expected_flags):
        raise ValueError("Cell quality flags do not match fold flags")
    fingerprint = payload["fingerprint"]
    for key in (
        "config_sha256",
        "catalog_sha256",
        "manifest_sha256",
        "code_sha256",
        "data_sha256",
        "split_sha256",
        "environment_sha256",
    ):
        if len(str(fingerprint.get(key, ""))) != 64:
            raise ValueError(f"Fingerprint {key} is invalid")
    if fingerprint.get("git_dirty") is not False:
        raise ValueError("Full result cell was produced from a dirty Git tree")


def audit_results(output_dir: Path, manifest: Exp03Manifest) -> dict[str, Any]:
    cells_dir = output_dir / f"{manifest.phase}_cells"
    paths = sorted(cells_dir.glob("*.json")) if cells_dir.is_dir() else []
    expected = {cell.cell_id: cell for cell in manifest.cells}
    found = {path.stem: path for path in paths}
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(f"Result set mismatch: missing={len(missing)} extra={len(extra)}")
    code_hashes: set[str] = set()
    for cell_id, path in found.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_result_cell(payload)
        expected_cell = expected[cell_id]
        if (
            payload["cell_id"] != cell_id
            or payload["subject"] != expected_cell.subject
            or payload["seed"] != expected_cell.seed
            or tuple(payload["folds"]) != expected_cell.folds
            or payload["config_id"] != expected_cell.config_id
        ):
            raise ValueError(f"Cell identity mismatch in {cell_id}")
        fingerprint = payload["fingerprint"]
        required = {
            "config_sha256": manifest.config_sha256,
            "catalog_sha256": manifest.catalog_sha256,
            "manifest_sha256": manifest.sha256,
            "revision": manifest.revision,
        }
        if any(fingerprint.get(key) != value for key, value in required.items()):
            raise ValueError(f"Run fingerprint mismatch in {cell_id}")
        code_hashes.add(str(fingerprint["code_sha256"]))
    if len(code_hashes) != 1:
        raise ValueError("Results mix multiple exp03 code identities")
    return {
        "complete": True,
        "phase": manifest.phase,
        "expected": len(expected),
        "found": len(found),
        "code_sha256": next(iter(code_hashes)),
    }

