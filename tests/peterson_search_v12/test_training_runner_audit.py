from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.peterson_search_v12.common.audit import audit_results, validate_cell_payload
from experiments.peterson_search_v12.common.catalogs import build_exp01_catalog
from experiments.peterson_search_v12.common.config import SearchConfig
from experiments.peterson_search_v12.common.manifests import build_manifest
from experiments.peterson_search_v12.common.runner import is_resumable_cell, write_plan
from experiments.peterson_search_v12.common.training import predict_numpy, train_candidate


def test_training_restores_checkpoint_and_is_reproducible(tmp_path):
    config = SearchConfig(epochs=2, patience=2, batch_size=4)
    candidate = build_exp01_catalog(config)[0]
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 3, 64)).astype(np.float32)
    y = np.tile(np.array([0, 1]), 6)
    checkpoint = tmp_path / "checkpoint.pt"
    first = train_candidate(
        candidate,
        x[:8],
        y[:8],
        x[8:],
        y[8:],
        config=config,
        device="cpu",
        seed=7,
        checkpoint_path=checkpoint,
    )
    assert checkpoint.is_file()
    first_logits = predict_numpy(
        first.model, x[8:], candidate=candidate, config=config, device="cpu", seed=9
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["best_epoch"] == first.best_epoch
    second = train_candidate(
        candidate,
        x[:8],
        y[:8],
        x[8:],
        y[8:],
        config=config,
        device="cpu",
        seed=7,
    )
    second_logits = predict_numpy(
        second.model, x[8:], candidate=candidate, config=config, device="cpu", seed=9
    )
    assert np.array_equal(first_logits, second_logits)
    assert first.history == second.history


def test_plan_only_writes_36_candidates_and_216_cells(tmp_path):
    plan = write_plan(tmp_path, SearchConfig(), revision="abc123")
    assert plan["candidate_count"] == 36
    assert plan["cell_count"] == 216
    assert (tmp_path / "manifests" / "exp01.json").is_file()
    assert not (tmp_path / "cells").exists()


def test_cell_schema_rejects_outer_test_material():
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp01",
        "cell_id": "exp01-cfg-001__seed0__S09",
        "config_id": "exp01-cfg-001",
        "subject": "S09",
        "seed": 0,
        "folds": [0, 2],
        "validation_metrics": {"accuracy": 0.6, "kappa": 0.2},
        "fold_reports": [],
        "fingerprint": {"candidate_sha256": "a" * 64, "code_sha256": "b" * 64},
    }
    validate_cell_payload(payload)
    for key in ("y_test", "test_accuracy", "test_metrics", "outer_test_predictions"):
        leaked = dict(payload)
        leaked[key] = []
        with pytest.raises(ValueError, match="outer-test"):
            validate_cell_payload(leaked)


def test_audit_requires_exact_manifest(tmp_path):
    config = SearchConfig()
    catalog = build_exp01_catalog(config)
    manifest = build_manifest(config, catalog, revision="abc123")
    with pytest.raises(ValueError, match="missing"):
        audit_results(tmp_path, manifest)


def test_cell_schema_rejects_collapsed_or_inconsistent_validation():
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp01",
        "cell_id": "exp01-cfg-001__seed0__S09",
        "config_id": "exp01-cfg-001",
        "subject": "S09",
        "seed": 0,
        "folds": [0, 2],
        "validation_metrics": {"accuracy": 0.5, "kappa": 0.0},
        "validation_y_true": [0, 1],
        "validation_y_pred": [0, 0],
        "validation_y_score": [0.5, 0.5],
        "fold_reports": [],
        "fingerprint": {"candidate_sha256": "a" * 64, "code_sha256": "b" * 64},
    }
    with pytest.raises(ValueError, match="collapse"):
        validate_cell_payload(payload)
    payload["validation_y_pred"] = [0, 1]
    payload["validation_y_score"] = [0.2]
    with pytest.raises(ValueError, match="length"):
        validate_cell_payload(payload)


def test_resumption_requires_the_entire_fingerprint():
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp01",
        "cell_id": "exp01-cfg-001__seed0__S09",
        "config_id": "exp01-cfg-001",
        "subject": "S09",
        "seed": 0,
        "folds": [0, 2],
        "validation_metrics": {"accuracy": 0.6, "kappa": 0.2},
        "validation_y_true": [0, 1],
        "validation_y_pred": [0, 1],
        "validation_y_score": [0.2, 0.8],
        "fold_reports": [],
        "fingerprint": {
            "candidate_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "catalog_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "code_sha256": "e" * 64,
            "data_sha256": "f" * 64,
            "split_sha256": "1" * 64,
            "environment_sha256": "2" * 64,
            "device": "cuda:0",
            "revision": "3" * 40,
            "git_dirty": False,
        },
    }
    assert is_resumable_cell(payload, payload["fingerprint"])
    for key in ("config_sha256", "data_sha256", "split_sha256", "revision", "device"):
        changed = dict(payload["fingerprint"])
        changed[key] = "changed"
        assert not is_resumable_cell(payload, changed)
