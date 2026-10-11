from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from experiments.peterson_search_v12.common.audit import (
    audit_results,
    validate_cell_payload,
    validate_verifiable_fingerprints,
)
from experiments.peterson_search_v12.common.catalogs import build_exp01_catalog
from experiments.peterson_search_v12.common.config import SearchConfig
from experiments.peterson_search_v12.common.identity import canonical_sha256
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
    assert {
        "train_classification_loss",
        "train_denoising_loss",
        "train_ranking_loss",
        "train_consistency_loss",
        "validation_classification_loss",
        "validation_denoising_loss",
        "validation_ranking_loss",
        "validation_consistency_loss",
    } <= set(first.history[0])


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
        "validation_metrics": {
            "accuracy": 1.0,
            "kappa": 1.0,
            "precision_macro": 1.0,
            "recall_macro": 1.0,
            "f1_macro": 1.0,
            "roc_auc": 1.0,
        },
        "quality_flags": [],
        "validation_y_true": [0, 1, 0, 1],
        "validation_y_pred": [0, 1, 0, 1],
        "validation_y_score": [0.1, 0.9, 0.2, 0.8],
        "fold_reports": [
            {"n_validation_trials": 2, "quality_flags": []},
            {"n_validation_trials": 2, "quality_flags": []},
        ],
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


def test_cell_schema_records_collapsed_validation_without_aborting_the_search():
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp01",
        "cell_id": "exp01-cfg-001__seed0__S09",
        "config_id": "exp01-cfg-001",
        "subject": "S09",
        "seed": 0,
        "folds": [0, 2],
        "validation_metrics": {
            "accuracy": 0.5,
            "kappa": 0.0,
            "precision_macro": 0.25,
            "recall_macro": 0.5,
            "f1_macro": 1 / 3,
            "roc_auc": 0.5,
        },
        "quality_flags": [],
        "validation_y_true": [0, 1, 0, 1],
        "validation_y_pred": [0, 0, 0, 0],
        "validation_y_score": [0.5, 0.5, 0.5, 0.5],
        "fold_reports": [
            {"n_validation_trials": 2, "quality_flags": ["validation_class_collapse"]},
            {"n_validation_trials": 2, "quality_flags": ["validation_class_collapse"]},
        ],
        "fingerprint": {"candidate_sha256": "a" * 64, "code_sha256": "b" * 64},
    }
    with pytest.raises(ValueError, match="quality flag"):
        validate_cell_payload(payload)
    payload["quality_flags"] = ["validation_class_collapse"]
    validate_cell_payload(payload)
    payload["validation_y_pred"] = [0, 1, 0, 1]
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
        "validation_metrics": {
            "accuracy": 1.0,
            "kappa": 1.0,
            "precision_macro": 1.0,
            "recall_macro": 1.0,
            "f1_macro": 1.0,
            "roc_auc": 1.0,
        },
        "quality_flags": [],
        "validation_y_true": [0, 1, 0, 1],
        "validation_y_pred": [0, 1, 0, 1],
        "validation_y_score": [0.2, 0.8, 0.1, 0.9],
        "fold_reports": [
            {"n_validation_trials": 2, "quality_flags": []},
            {"n_validation_trials": 2, "quality_flags": []},
        ],
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


def test_completed_cell_cannot_omit_prediction_evidence():
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp01",
        "cell_id": "exp01-cfg-001__seed0__S09",
        "config_id": "exp01-cfg-001",
        "subject": "S09",
        "seed": 0,
        "folds": [0, 2],
        "validation_metrics": {
            "accuracy": 1.0,
            "kappa": 1.0,
            "precision_macro": 1.0,
            "recall_macro": 1.0,
            "f1_macro": 1.0,
            "roc_auc": 1.0,
        },
        "quality_flags": [],
        "fold_reports": [],
        "fingerprint": {"candidate_sha256": "a" * 64, "code_sha256": "b" * 64},
    }
    with pytest.raises(ValueError, match="missing keys"):
        validate_cell_payload(payload)


def test_opposite_fold_collapses_are_not_hidden_by_aggregation():
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp01",
        "cell_id": "exp01-cfg-001__seed0__S09",
        "config_id": "exp01-cfg-001",
        "subject": "S09",
        "seed": 0,
        "folds": [0, 2],
        "validation_metrics": {
            "accuracy": 0.5,
            "kappa": 0.0,
            "precision_macro": 0.5,
            "recall_macro": 0.5,
            "f1_macro": 0.5,
            "roc_auc": 0.5,
        },
        "quality_flags": ["validation_class_collapse"],
        "validation_y_true": [0, 1, 0, 1],
        "validation_y_pred": [0, 0, 1, 1],
        "validation_y_score": [0.4, 0.4, 0.6, 0.6],
        "fold_reports": [
            {"n_validation_trials": 2, "quality_flags": ["validation_class_collapse"]},
            {"n_validation_trials": 2, "quality_flags": ["validation_class_collapse"]},
        ],
        "fingerprint": {"candidate_sha256": "a" * 64, "code_sha256": "b" * 64},
    }
    validate_cell_payload(payload)
    payload["quality_flags"] = []
    with pytest.raises(ValueError, match="folds"):
        validate_cell_payload(payload)


def test_verifiable_fingerprints_reject_candidate_split_environment_and_device_tampering():
    candidate = asdict(build_exp01_catalog(SearchConfig())[0])
    reports = [
        {"search_split_sha256": "1" * 64},
        {"search_split_sha256": "2" * 64},
    ]
    environment = {"device": "cuda:0", "torch": "test"}
    payload = {
        "candidate": candidate,
        "fold_reports": reports,
        "runtime": {"environment": environment},
        "fingerprint": {
            "candidate_sha256": candidate["sha256"],
            "split_sha256": canonical_sha256([report["search_split_sha256"] for report in reports]),
            "environment_sha256": canonical_sha256(environment),
            "device": "cuda:0",
        },
    }
    validate_verifiable_fingerprints(payload)
    corruptions = (
        ("candidate", "backbone", "tampered"),
        ("fingerprint", "split_sha256", "0" * 64),
        ("fingerprint", "environment_sha256", "0" * 64),
        ("fingerprint", "device", "cuda:1"),
    )
    for section, key, value in corruptions:
        changed = json.loads(json.dumps(payload))
        changed[section][key] = value
        with pytest.raises(ValueError, match="mismatch"):
            validate_verifiable_fingerprints(changed)
