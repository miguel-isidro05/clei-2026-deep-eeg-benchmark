"""Strict completion audit for V11 result cells."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .manifests import CellSpec


def audit_cells(
    output_dir: Path,
    expected_cells: list[CellSpec],
    *,
    expected_config_sha256: str,
    expected_code_sha256: str | None = None,
) -> dict[str, object]:
    cell_dir = output_dir / "cells"
    expected = {cell.cell_id for cell in expected_cells}
    actual_paths = list(cell_dir.glob("*.json")) if cell_dir.exists() else []
    actual = {path.stem for path in actual_paths}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    errors: list[str] = []
    data_hashes: dict[str, set[str]] = {}
    if missing:
        errors.append(f"missing={len(missing)} first={missing[:3]}")
    if extra:
        errors.append(f"extra={len(extra)} first={extra[:3]}")
    for path in actual_paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_json={path.name}:{exc}")
            continue
        if payload.get("status") != "completed":
            errors.append(f"incomplete={path.name}")
        if payload.get("cell_id") != path.stem:
            errors.append(f"identity_mismatch={path.name}")
        if payload.get("config_sha256") != expected_config_sha256:
            errors.append(f"config_mismatch={path.name}")
        if expected_code_sha256 is not None and payload.get("code_sha256") != expected_code_sha256:
            errors.append(f"code_mismatch={path.name}")
        accuracy = payload.get("metrics", {}).get("accuracy")
        if accuracy is None or not np.isfinite(float(accuracy)) or not 0 <= float(accuracy) <= 1:
            errors.append(f"nonfinite_metric={path.name}")
        if len(payload.get("y_true", [])) != len(payload.get("y_pred", [])):
            errors.append(f"prediction_length_mismatch={path.name}")
        subject = str(payload.get("subject", ""))
        data_sha256 = str(payload.get("data_sha256", ""))
        data_hashes.setdefault(subject, set()).add(data_sha256)
        for fold in payload.get("fold_reports", []):
            split_keys = ("inner_train_indices", "validation_indices", "test_indices")
            if all(key in fold for key in split_keys):
                train = set(fold["inner_train_indices"])
                validation = set(fold["validation_indices"])
                test = set(fold["test_indices"])
                if train & validation or train & test or validation & test:
                    errors.append(f"split_leakage={path.name}:fold{fold.get('fold')}")
            windows_per_trial = fold.get("test_windows_per_trial")
            condition = payload.get("condition")
            expected_windows = 1 if condition == "full" else 6
            if windows_per_trial is not None and int(windows_per_trial) != expected_windows:
                errors.append(f"window_count_mismatch={path.name}:fold{fold.get('fold')}")
        if path.stem.startswith("wave01") and len(set(payload.get("y_pred", []))) < 2:
            errors.append(f"prediction_collapse={path.name}")
    for subject, hashes in data_hashes.items():
        if len(hashes) != 1 or "" in hashes:
            errors.append(f"data_hash_mismatch={subject}")
    if errors:
        raise RuntimeError("V11 audit failed: " + "; ".join(errors))
    return {"complete": True, "expected": len(expected), "found": len(actual)}
