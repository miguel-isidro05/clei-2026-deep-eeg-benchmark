from __future__ import annotations

import json

import pytest

from experiments.peterson_diffusion_v11.audit import audit_cells
from experiments.peterson_diffusion_v11.config import ExperimentConfig
from experiments.peterson_diffusion_v11.decision import decide_h1
from experiments.peterson_diffusion_v11.manifests import build_manifest


def test_audit_rejects_missing_cells(tmp_path) -> None:
    manifest = build_manifest(ExperimentConfig(wave=0))[:2]
    with pytest.raises(RuntimeError, match="missing"):
        audit_cells(tmp_path, manifest, expected_config_sha256="abc")


def test_audit_accepts_only_completed_matching_cells(tmp_path) -> None:
    config = ExperimentConfig(wave=0)
    manifest = build_manifest(config)[:1]
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / f"{manifest[0].cell_id}.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "cell_id": manifest[0].cell_id,
                "config_sha256": config.sha256,
                "data_sha256": "data-hash",
                "metrics": {"accuracy": 0.75},
                "y_true": [0, 1],
                "y_pred": [0, 1],
            }
        )
    )
    report = audit_cells(tmp_path, manifest, expected_config_sha256=config.sha256)
    assert report["complete"] is True


def test_h1_requires_effect_size_and_seven_subjects() -> None:
    center = {f"S{i:02d}": 0.70 for i in range(2, 12)}
    overlap = dict(center)
    for subject in list(overlap)[:7]:
        overlap[subject] += 0.03
    decision = decide_h1(center, overlap)
    assert decision["decision"] == "advance"
    overlap[list(overlap)[0]] = center[list(overlap)[0]]
    decision = decide_h1(center, overlap)
    assert decision["decision"] == "revise"
