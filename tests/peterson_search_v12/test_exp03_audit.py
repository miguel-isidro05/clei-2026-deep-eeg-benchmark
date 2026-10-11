from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp03_quality_fbcnet.audit import (
    audit_results,
    validate_result_cell,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.manifest import (
    Exp03Cell,
    Exp03Manifest,
)


def _cell() -> dict[str, object]:
    y_true = np.asarray([0, 1, 0, 1])
    y_pred = np.asarray([0, 1, 1, 1])
    y_score = np.asarray([0.1, 0.8, 0.7, 0.9])
    return {
        "schema_version": 3,
        "status": "completed",
        "experiment": "exp03",
        "phase": "quality",
        "kind": "quality",
        "cell_id": "quality__exp03-qcfg-001__seed0__S09",
        "model_id": "QualityFBCNet",
        "config_id": "exp03-qcfg-001",
        "subject": "S09",
        "seed": 0,
        "folds": [0],
        "validation_metrics": compute_metrics(y_true, y_pred, y_score),
        "validation_y_true": y_true.tolist(),
        "validation_y_pred": y_pred.tolist(),
        "validation_y_score": y_score.tolist(),
        "quality_flags": [],
        "fold_reports": [
            {
                "fold": 0,
                "n_validation_trials": 4,
                "quality_flags": [],
                "search_split_sha256": "a" * 64,
            }
        ],
        "fingerprint": {
            "config_sha256": "b" * 64,
            "catalog_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "cell_config_sha256": "3" * 64,
            "code_sha256": "e" * 64,
            "data_sha256": "f" * 64,
            "split_sha256": canonical_sha256(["a" * 64]),
            "environment_sha256": "2" * 64,
            "device": "cpu",
            "revision": "abc123",
            "git_dirty": False,
        },
    }


def test_exp03_audit_recomputes_metrics_and_rejects_outer_test_material():
    validate_result_cell(_cell())
    bad_metric = copy.deepcopy(_cell())
    bad_metric["validation_metrics"]["accuracy"] = 1.0
    with pytest.raises(ValueError, match="metrics"):
        validate_result_cell(bad_metric)
    leaked = copy.deepcopy(_cell())
    leaked["outer_test_predictions"] = [0, 1]
    with pytest.raises(ValueError, match="forbidden"):
        validate_result_cell(leaked)


def test_exp03_audit_binds_cell_config_and_split_hashes(tmp_path):
    payload = _cell()
    cell = Exp03Cell(
        cell_id=str(payload["cell_id"]),
        kind="quality",
        model_id="QualityFBCNet",
        config_id="exp03-qcfg-001",
        config_sha256="3" * 64,
        subject="S09",
        seed=0,
        folds=(0,),
    )
    manifest = Exp03Manifest(
        experiment="exp03",
        phase="quality",
        revision="abc123",
        config_sha256="b" * 64,
        catalog_sha256="c" * 64,
        parent_decision_sha256="4" * 64,
        cells=(cell,),
        sha256="d" * 64,
    )
    cells_dir = tmp_path / "quality_cells"
    cells_dir.mkdir()
    path = cells_dir / f"{cell.cell_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert audit_results(tmp_path, manifest)["complete"] is True

    altered_config = copy.deepcopy(payload)
    altered_config["fingerprint"]["cell_config_sha256"] = "5" * 64
    path.write_text(json.dumps(altered_config), encoding="utf-8")
    with pytest.raises(ValueError, match="cell config"):
        audit_results(tmp_path, manifest)

    altered_split = copy.deepcopy(payload)
    altered_split["fingerprint"]["split_sha256"] = "6" * 64
    path.write_text(json.dumps(altered_split), encoding="utf-8")
    with pytest.raises(ValueError, match="split hash"):
        audit_results(tmp_path, manifest)

