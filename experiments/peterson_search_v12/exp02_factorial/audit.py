"""Portable and strict integrity checks for exp02 outputs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
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

from .catalog import Exp02Candidate, InferenceRecipe
from .manifest import ExperimentManifest

FORBIDDEN_KEYS = {"y_test", "test_accuracy", "test_metrics", "outer_test_predictions"}


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


def _validate_metrics(result: dict[str, Any]) -> None:
    y_true = result.get("y_true")
    y_pred = result.get("y_pred")
    y_score = result.get("y_score")
    if not isinstance(y_true, list) or not (
        len(y_true) == len(y_pred) == len(y_score) and len(y_true) > 0
    ):
        raise ValueError("Inference prediction lengths are invalid")
    recomputed = compute_metrics(
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        np.asarray(y_score, dtype=float),
    )
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(recomputed):
        raise ValueError("Inference metrics do not match the fixed metric set")
    if any(
        not math.isclose(float(metrics[key]), value, rel_tol=1e-9, abs_tol=1e-9)
        for key, value in recomputed.items()
    ):
        raise ValueError("Inference metrics do not match stored predictions")


def validate_candidate_cell(payload: dict[str, Any], *, expected_inference_ids: set[str]) -> None:
    leaked = FORBIDDEN_KEYS & set(_walk_keys(payload))
    if leaked:
        raise ValueError(f"Cell contains forbidden outer-test material: {sorted(leaked)}")
    if payload.get("status") != "completed" or payload.get("experiment") != "exp02":
        raise ValueError("Cell is not a completed exp02 result")
    if payload.get("kind") != "candidate":
        raise ValueError("Candidate cell has the wrong kind")
    subject = payload.get("subject")
    if subject in HOLDOUT_SUBJECTS or subject not in DISCOVERY_SUBJECTS:
        raise ValueError("Candidate cell violates the discovery/holdout boundary")
    if tuple(payload.get("folds", ())) != (0, 2, 4) or payload.get("seed") != 0:
        raise ValueError("Candidate cell uses an unexpected fold or seed")
    reports = payload.get("fold_reports")
    if not isinstance(reports, list) or {report.get("fold") for report in reports} != {0, 2, 4}:
        raise ValueError("Candidate cell fold reports are incomplete")
    fold_sizes = [int(report.get("n_validation_trials", 0)) for report in reports]
    if any(size < 1 for size in fold_sizes):
        raise ValueError("Candidate fold validation counts are invalid")
    results = payload.get("inference_results")
    if not isinstance(results, dict) or set(results) != expected_inference_ids:
        raise ValueError("Candidate cell inference catalog mismatch")
    cell_flags: set[str] = set()
    for result in results.values():
        _validate_metrics(result)
        offset = 0
        expected_fold_flags: list[list[str]] = []
        for size in fold_sizes:
            expected_fold_flags.append(
                validation_quality_flags(
                    result["y_true"][offset : offset + size],
                    result["y_pred"][offset : offset + size],
                )
            )
            offset += size
        if offset != len(result["y_true"]):
            raise ValueError("Candidate fold counts do not cover all predictions")
        if result.get("fold_quality_flags") != expected_fold_flags:
            raise ValueError("Candidate fold quality flags are inconsistent")
        expected_result_flags = sorted({flag for flags in expected_fold_flags for flag in flags})
        if result.get("quality_flags") != expected_result_flags:
            raise ValueError("Candidate inference quality flags are inconsistent")
        cell_flags.update(expected_result_flags)
    if payload.get("quality_flags") != sorted(cell_flags):
        raise ValueError("Candidate cell quality flags are inconsistent")


def validate_baseline_cell(payload: dict[str, Any]) -> None:
    leaked = FORBIDDEN_KEYS & set(_walk_keys(payload))
    if leaked:
        raise ValueError(f"Cell contains forbidden outer-test material: {sorted(leaked)}")
    if payload.get("status") != "completed" or payload.get("kind") != "baseline":
        raise ValueError("Baseline cell is not completed")
    if payload.get("model_id") not in {"CSP+LDA", "FBCNet"}:
        raise ValueError("Unknown baseline")
    if payload.get("subject") not in DISCOVERY_SUBJECTS:
        raise ValueError("Baseline subject violates the discovery boundary")
    if tuple(payload.get("folds", ())) != (0, 2, 4):
        raise ValueError("Baseline cell uses unexpected folds")
    _validate_metrics(
        {
            "metrics": payload.get("validation_metrics"),
            "y_true": payload.get("validation_y_true"),
            "y_pred": payload.get("validation_y_pred"),
            "y_score": payload.get("validation_y_score"),
        }
    )
    reports = payload.get("fold_reports")
    if not isinstance(reports, list) or {report.get("fold") for report in reports} != {0, 2, 4}:
        raise ValueError("Baseline fold reports are incomplete")
    if payload.get("quality_flags") != sorted(
        {flag for report in reports for flag in report.get("quality_flags", [])}
    ):
        raise ValueError("Baseline quality flags are inconsistent")


def audit_results(
    output_dir: Path,
    manifest: ExperimentManifest,
    candidates: tuple[Exp02Candidate, ...],
    inference_catalog: tuple[InferenceRecipe, ...],
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    paths = sorted((output_dir / "cells").glob("*.json"))
    expected = {cell.cell_id: cell for cell in manifest.cells}
    found = {path.stem: path for path in paths}
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(f"Result set mismatch: missing={len(missing)} extra={len(extra)}")
    candidate_by_id = {candidate.config_id: candidate for candidate in candidates}
    inference_by_candidate: dict[str, set[str]] = {}
    for recipe in inference_catalog:
        inference_by_candidate.setdefault(recipe.config_id, set()).add(recipe.inference_id)
    code_hashes: set[str] = set()
    subject_data_hashes: dict[str, set[str]] = {}
    subject_split_hashes: dict[str, set[str]] = {}
    for cell_id, path in found.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell = expected[cell_id]
        if cell.kind == "candidate":
            if cell.config_id is None:
                raise ValueError("Candidate manifest cell has no config id")
            validate_candidate_cell(
                payload,
                expected_inference_ids=inference_by_candidate[cell.config_id],
            )
            candidate = candidate_by_id[cell.config_id]
            if canonical_sha256(payload.get("candidate")) != canonical_sha256(asdict(candidate)):
                raise ValueError(f"Candidate metadata mismatch in {cell_id}")
            if payload["fingerprint"].get("candidate_sha256") != cell.candidate_sha256:
                raise ValueError(f"Candidate fingerprint mismatch in {cell_id}")
        else:
            validate_baseline_cell(payload)
        if payload.get("cell_id") != cell_id or payload.get("subject") != cell.subject:
            raise ValueError(f"Cell identity mismatch in {cell_id}")
        fingerprint = payload.get("fingerprint", {})
        for key, expected_value in (
            ("manifest_sha256", manifest.sha256),
            ("config_sha256", manifest.config_sha256),
            ("catalog_sha256", manifest.catalog_sha256),
            ("revision", manifest.revision),
        ):
            if fingerprint.get(key) != expected_value:
                raise ValueError(f"Fingerprint mismatch in {cell_id}: {key}")
        if not allow_dirty and fingerprint.get("git_dirty") is not False:
            raise ValueError(f"Cell was produced from a dirty tree: {cell_id}")
        code_hash = str(fingerprint.get("code_sha256", ""))
        if len(code_hash) != 64:
            raise ValueError(f"Invalid code hash in {cell_id}")
        code_hashes.add(code_hash)
        reports = payload.get("fold_reports", [])
        split_hash = canonical_sha256([report.get("search_split_sha256") for report in reports])
        if fingerprint.get("split_sha256") != split_hash:
            raise ValueError(f"Split fingerprint mismatch in {cell_id}")
        if len(str(fingerprint.get("data_sha256", ""))) != 64:
            raise ValueError(f"Invalid data hash in {cell_id}")
        environment = payload.get("runtime", {}).get("environment")
        if not isinstance(environment, dict):
            raise ValueError(f"Missing runtime environment in {cell_id}")
        if fingerprint.get("device") != environment.get("device"):
            raise ValueError(f"Runtime device mismatch in {cell_id}")
        if fingerprint.get("environment_sha256") != canonical_sha256(environment):
            raise ValueError(f"Runtime environment mismatch in {cell_id}")
        subject = str(payload["subject"])
        subject_data_hashes.setdefault(subject, set()).add(fingerprint["data_sha256"])
        subject_split_hashes.setdefault(subject, set()).add(fingerprint["split_sha256"])
    if len(code_hashes) != 1:
        raise ValueError("Results mix multiple code identities")
    if any(len(values) != 1 for values in subject_data_hashes.values()):
        raise ValueError("Results mix multiple data identities for one subject")
    if any(len(values) != 1 for values in subject_split_hashes.values()):
        raise ValueError("Candidates and baselines do not share identical subject splits")
    return {
        "complete": True,
        "expected": len(expected),
        "found": len(found),
        "cells_sha256": canonical_sha256(
            [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        ),
        "code_sha256": next(iter(code_hashes)),
    }
