"""Strict integrity and leakage audit for V12 screening outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .manifests import SearchManifest
from .partitions import DISCOVERY_SUBJECTS, HOLDOUT_SUBJECTS

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


def validate_cell_payload(payload: dict[str, Any]) -> None:
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
    for metric in ("accuracy", "kappa"):
        value = float(payload["validation_metrics"][metric])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite validation metric: {metric}")


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
        expected_cell = expected[cell_id]
        if payload["config_id"] != expected_cell.config_id:
            raise ValueError(f"Config mismatch in {cell_id}")
        if payload["fingerprint"]["candidate_sha256"] != expected_cell.candidate_sha256:
            raise ValueError(f"Candidate hash mismatch in {cell_id}")
        code_hash = str(payload["fingerprint"].get("code_sha256", ""))
        data_hash = str(payload["fingerprint"].get("data_sha256", ""))
        if len(code_hash) != 64 or len(data_hash) != 64:
            raise ValueError(f"Invalid code or data hash in {cell_id}")
        code_hashes.add(code_hash)
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
