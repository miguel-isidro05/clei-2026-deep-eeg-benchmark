"""Strict integrity and leakage audit for V12 screening outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from deepbench.metrics import compute_metrics

from .identity import canonical_sha256
from .manifests import SearchManifest
from .partitions import DISCOVERY_SUBJECTS, HOLDOUT_SUBJECTS

FORBIDDEN_KEYS = {"y_test", "test_accuracy", "test_metrics", "outer_test_predictions"}
VALIDATION_COLLAPSE_FLAG = "validation_class_collapse"
KNOWN_QUALITY_FLAGS = {VALIDATION_COLLAPSE_FLAG}


def validation_quality_flags(y_true: list[int], y_pred: list[int]) -> list[str]:
    if len(set(y_true)) != 2:
        raise ValueError("Validation fold ground truth must contain both classes")
    if len(set(y_pred)) != 2:
        return [VALIDATION_COLLAPSE_FLAG]
    return []


def _validated_quality_flags(flags: Any) -> list[str]:
    if not isinstance(flags, list) or flags != sorted(set(flags)):
        raise ValueError("Cell quality flags must be a sorted unique list")
    unknown = set(flags) - KNOWN_QUALITY_FLAGS
    if unknown:
        raise ValueError(f"Unknown cell quality flags: {sorted(unknown)}")
    return flags


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


def validate_cell_payload(payload: dict[str, Any], *, allow_collapse: bool = False) -> None:
    # Retained for compatibility with existing smoke callers; collapsed cells are
    # now represented explicitly by a quality flag in every profile.
    _ = allow_collapse
    leaked = FORBIDDEN_KEYS & set(_walk_keys(payload))
    if leaked:
        raise ValueError(f"Cell contains forbidden outer-test material: {sorted(leaked)}")
    required = {
        "schema_version",
        "status",
        "experiment",
        "cell_id",
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
    if payload["status"] != "completed" or payload["experiment"] != "exp01":
        raise ValueError("Cell is not a completed exp01 result")
    if payload["subject"] in HOLDOUT_SUBJECTS or payload["subject"] not in DISCOVERY_SUBJECTS:
        raise ValueError("Cell subject violates the discovery/holdout boundary")
    if tuple(payload["folds"]) != (0, 2) or payload["seed"] != 0:
        raise ValueError("Cell uses an unexpected fold or seed")
    metric_names = {
        "accuracy",
        "kappa",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
    }
    if set(payload["validation_metrics"]) != metric_names:
        raise ValueError("Validation metrics do not contain the fixed metric set")
    for metric in metric_names:
        value = float(payload["validation_metrics"][metric])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite validation metric: {metric}")
    flags = _validated_quality_flags(payload["quality_flags"])
    y_true, y_pred, y_score = (
        payload["validation_y_true"],
        payload["validation_y_pred"],
        payload["validation_y_score"],
    )
    if not (len(y_true) == len(y_pred) == len(y_score) and len(y_true) > 0):
        raise ValueError("Validation prediction arrays have inconsistent length")
    if len(set(y_true)) != 2:
        raise ValueError("Validation ground truth must contain both classes")
    if any(not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0 for score in y_score):
        raise ValueError("Validation scores must be finite probabilities")
    recomputed = compute_metrics(
        np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score, dtype=float)
    )
    if any(
        not math.isclose(
            float(payload["validation_metrics"][metric]),
            float(recomputed[metric]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for metric in metric_names
    ):
        raise ValueError("Validation metrics do not match the stored predictions")
    reports = payload["fold_reports"]
    if not isinstance(reports, list) or len(reports) != len(payload["folds"]):
        raise ValueError("Cell must contain one fold report per validation fold")
    offset = 0
    expected_flags: set[str] = set()
    for report in reports:
        count = int(report.get("n_validation_trials", 0))
        if count < 1 or offset + count > len(y_true):
            raise ValueError("Fold report validation counts are inconsistent")
        fold_flags = _validated_quality_flags(report.get("quality_flags"))
        recomputed_flags = validation_quality_flags(
            y_true[offset : offset + count], y_pred[offset : offset + count]
        )
        if fold_flags != recomputed_flags:
            raise ValueError("Fold collapse quality flag is inconsistent with the predictions")
        expected_flags.update(fold_flags)
        offset += count
    if offset != len(y_true):
        raise ValueError("Fold report validation counts do not cover all predictions")
    if flags != sorted(expected_flags):
        raise ValueError("Validation collapse quality flag is inconsistent with the folds")


def validate_verifiable_fingerprints(payload: dict[str, Any]) -> None:
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict) or "sha256" not in candidate:
        raise ValueError("Missing candidate identity")
    candidate_body = {key: value for key, value in candidate.items() if key != "sha256"}
    declared_candidate_hash = candidate["sha256"]
    if canonical_sha256(candidate_body) != declared_candidate_hash:
        raise ValueError("Candidate metadata hash mismatch")
    if payload.get("fingerprint", {}).get("candidate_sha256") != declared_candidate_hash:
        raise ValueError("Candidate fingerprint mismatch")
    reports = payload.get("fold_reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("Missing fold reports")
    split_hash = canonical_sha256([report.get("search_split_sha256") for report in reports])
    if payload["fingerprint"].get("split_sha256") != split_hash:
        raise ValueError("Split fingerprint mismatch")
    runtime_environment = payload.get("runtime", {}).get("environment")
    if not isinstance(runtime_environment, dict):
        raise ValueError("Missing runtime environment")
    if payload["fingerprint"].get("device") != runtime_environment.get("device"):
        raise ValueError("Runtime device mismatch")
    if payload["fingerprint"].get("environment_sha256") != canonical_sha256(runtime_environment):
        raise ValueError("Runtime environment mismatch")


def audit_results(output_dir: Path, manifest: SearchManifest) -> dict[str, Any]:
    cells_dir = output_dir / "cells"
    paths = sorted(cells_dir.glob("*.json")) if cells_dir.is_dir() else []
    expected = {cell.cell_id: cell for cell in manifest.cells}
    found = {path.stem: path for path in paths}
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(f"Result set mismatch: missing={len(missing)} extra={len(extra)}")
    code_hashes: set[str] = set()
    data_hashes: dict[str, set[str]] = {}
    for cell_id, path in found.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_cell_payload(payload)
        validate_verifiable_fingerprints(payload)
        expected_cell = expected[cell_id]
        if payload["cell_id"] != cell_id:
            raise ValueError(f"Cell identity mismatch in {cell_id}")
        if payload["config_id"] != expected_cell.config_id:
            raise ValueError(f"Config mismatch in {cell_id}")
        if payload["subject"] != expected_cell.subject:
            raise ValueError(f"Subject mismatch in {cell_id}")
        if payload["seed"] != expected_cell.seed or tuple(payload["folds"]) != expected_cell.folds:
            raise ValueError(f"Seed or fold mismatch in {cell_id}")
        if payload["fingerprint"]["candidate_sha256"] != expected_cell.candidate_sha256:
            raise ValueError(f"Candidate hash mismatch in {cell_id}")
        expected_fingerprint = {
            "config_sha256": manifest.config_sha256,
            "catalog_sha256": manifest.catalog_sha256,
            "manifest_sha256": manifest.sha256,
            "revision": manifest.revision,
        }
        if any(
            payload["fingerprint"].get(key) != value for key, value in expected_fingerprint.items()
        ):
            raise ValueError(f"Run fingerprint mismatch in {cell_id}")
        code_hash = str(payload["fingerprint"].get("code_sha256", ""))
        data_hash = str(payload["fingerprint"].get("data_sha256", ""))
        if len(code_hash) != 64 or len(data_hash) != 64:
            raise ValueError(f"Invalid code or data hash in {cell_id}")
        code_hashes.add(code_hash)
        if payload["fingerprint"].get("git_dirty") is not False:
            raise ValueError(f"Cell was produced from a dirty Git tree: {cell_id}")
        data_hashes.setdefault(payload["subject"], set()).add(data_hash)
        reports = payload["fold_reports"]
        if not reports or {report["fold"] for report in reports} != {0, 2}:
            raise ValueError(f"Incomplete fold reports in {cell_id}")
        split_hashes: set[str] = set()
        for report in reports:
            if not report.get("history") or report.get("n_validation_trials", 0) < 1:
                raise ValueError(f"Incomplete validation history in {cell_id}")
            for key in ("search_split_sha256", "outer_split_sha256", "outer_test_sha256"):
                if len(str(report.get(key, ""))) != 64:
                    raise ValueError(f"Invalid split fingerprint in {cell_id}")
            split_hashes.add(report["search_split_sha256"])
        if len(split_hashes) != len(reports):
            raise ValueError(f"Duplicate split report in {cell_id}")
    if len(code_hashes) != 1:
        raise ValueError("Results mix multiple code revisions")
    if any(len(hashes) != 1 for hashes in data_hashes.values()):
        raise ValueError("Results mix multiple data identities for one subject")
    return {"complete": True, "expected": len(expected), "found": len(found)}
