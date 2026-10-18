"""Portable completeness and integrity checks for Exp04b."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.partitions import DISCOVERY_SUBJECTS, HOLDOUT_SUBJECTS

from .manifest import Exp04bManifest


def validate_cell(payload: dict[str, Any], *, allow_dirty: bool = False) -> None:
    if payload.get("schema_version") != 5 or payload.get("status") != "completed":
        raise ValueError("Exp04b cell is incomplete")
    if payload.get("experiment") != "exp04b":
        raise ValueError("Cell belongs to another experiment")
    subject = str(payload.get("subject"))
    if subject in HOLDOUT_SUBJECTS or subject not in DISCOVERY_SUBJECTS:
        raise ValueError("Cell violates the discovery/holdout boundary")
    y_true = np.asarray(payload["validation_y_true"], dtype=int)
    y_pred = np.asarray(payload["validation_y_pred"], dtype=int)
    y_score = np.asarray(payload["validation_y_score"], dtype=float)
    recomputed = compute_metrics(y_true, y_pred, y_score)
    for name, value in recomputed.items():
        if not math.isclose(value, payload["validation_metrics"][name], abs_tol=1e-9):
            raise ValueError(f"Stored {name} does not match predictions")
    for condition in ("missing_1", "missing_2"):
        floor = payload["physical_mask_floor"][condition]
        means = floor["mask_mean_accuracy"]
        if floor["worst_mask_id"] not in means:
            raise ValueError("Physical-mask identity is missing")
        if not math.isclose(floor["worst_mask_accuracy"], min(means.values()), abs_tol=1e-12):
            raise ValueError("Physical masks were not aggregated before selecting the floor")
    if not allow_dirty and payload["fingerprint"].get("git_dirty") is not False:
        raise ValueError("Full Exp04b cell was produced from a dirty tree")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_against_manifest(output_dir: Path, payload: dict[str, Any], expected: Any) -> None:
    if payload.get("cell_id") != expected.cell_id:
        raise ValueError("Cell identity differs from the manifest")
    if payload.get("config_id") != expected.config_id:
        raise ValueError("Cell configuration differs from the manifest")
    if payload["fingerprint"].get("cell_config_sha256") != expected.config_sha256:
        raise ValueError("Cell fingerprint differs from the signed configuration")
    candidate = payload.get("candidate")
    if candidate is not None:
        body = dict(candidate)
        signature = str(body.pop("sha256", ""))
        if signature != expected.config_sha256 or signature != canonical_sha256(body):
            raise ValueError("Candidate signature is invalid")
        for fold in payload["fold_reports"]:
            checkpoint = output_dir / fold["checkpoint_file"]
            if not checkpoint.is_file():
                raise ValueError(f"Missing checkpoint: {checkpoint}")
            if _file_sha256(checkpoint) != fold["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")


def audit_wave(
    output_dir: Path,
    manifest: Exp04bManifest,
    wave: str,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    expected = [cell for cell in manifest.cells if cell.wave == wave]
    expected_names = {f"{cell.cell_id}.json" for cell in expected}
    paths = sorted((output_dir / "cells" / wave).glob("*.json"))
    found = {path.name for path in paths}
    missing = sorted(expected_names - found)
    extras = sorted(found - expected_names)
    if missing or extras:
        raise ValueError(f"Exp04b {wave} incomplete: missing={missing[:5]} extras={extras[:5]}")
    expected_by_name = {f"{cell.cell_id}.json": cell for cell in expected}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_cell(payload, allow_dirty=allow_dirty)
        _validate_against_manifest(output_dir, payload, expected_by_name[path.name])
    return {"status": "completed", "wave": wave, "expected": len(expected), "found": len(paths)}
