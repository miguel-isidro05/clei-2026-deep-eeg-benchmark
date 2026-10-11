from __future__ import annotations

import json

import pytest

from experiments.peterson_search_v12.exp02_factorial.analysis import analyze_payloads
from experiments.peterson_search_v12.exp02_factorial.audit import validate_candidate_cell
from experiments.peterson_search_v12.exp02_factorial.catalog import (
    build_candidate_catalog,
    build_inference_catalog,
)
from experiments.peterson_search_v12.exp02_factorial.config import Exp02Config


def _candidate_cell(config_id: str, subject: str, accuracy: float) -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "completed",
        "experiment": "exp02",
        "kind": "candidate",
        "cell_id": f"{config_id}__seed0__{subject}",
        "config_id": config_id,
        "subject": subject,
        "seed": 0,
        "folds": [0, 2, 4],
        "quality_flags": [],
        "inference_results": {
            f"{config_id}__alpha-1__k0": {
                "alpha": 1.0,
                "k": 0,
                "metrics": {"accuracy": accuracy, "kappa": accuracy - 0.5},
                "quality_flags": [],
                "fold_quality_flags": [[], [], []],
                "y_true": [0, 1, 0, 1, 0, 1],
                "y_pred": [0, 1, 0, 1, 0, 1],
                "y_score": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            }
        },
        "fold_reports": [
            {"fold": fold, "quality_flags": [], "n_validation_trials": 2} for fold in (0, 2, 4)
        ],
        "parameters": {"total": 1000, "trainable": 1000, "active": 900},
        "runtime": {"elapsed_seconds": 1.0, "inference_seconds_per_trial": 0.001},
        "fingerprint": {"candidate_sha256": "a" * 64},
    }


def test_candidate_audit_rejects_outer_test_material_and_wrong_folds():
    payload = _candidate_cell("exp02-cfg-001", "S09", 1.0)
    result = next(iter(payload["inference_results"].values()))
    result["metrics"] = {
        "accuracy": 1.0,
        "kappa": 1.0,
        "precision_macro": 1.0,
        "recall_macro": 1.0,
        "f1_macro": 1.0,
        "roc_auc": 1.0,
    }
    validate_candidate_cell(payload, expected_inference_ids=set(payload["inference_results"]))
    leaked = json.loads(json.dumps(payload))
    leaked["outer_test_predictions"] = []
    with pytest.raises(ValueError, match="outer-test"):
        validate_candidate_cell(leaked, expected_inference_ids=set(payload["inference_results"]))
    wrong = json.loads(json.dumps(payload))
    wrong["folds"] = [0, 2]
    with pytest.raises(ValueError, match="fold"):
        validate_candidate_cell(wrong, expected_inference_ids=set(payload["inference_results"]))


def test_analysis_requires_exact_baselines_and_applies_success_gate():
    config = Exp02Config()
    candidate = build_candidate_catalog(config)[0]
    inference = build_inference_catalog((candidate,))
    candidate_cells = [
        _candidate_cell(candidate.config_id, subject, 0.84) for subject in config.subjects
    ]
    baseline_cells = []
    for model_id, accuracy in (("CSP+LDA", 0.75), ("FBCNet", 0.82)):
        for subject in config.subjects:
            baseline_cells.append(
                {
                    "kind": "baseline",
                    "model_id": model_id,
                    "subject": subject,
                    "validation_metrics": {"accuracy": accuracy, "kappa": accuracy - 0.5},
                    "quality_flags": [],
                }
            )
    report = analyze_payloads(
        candidate_cells,
        baseline_cells,
        candidates=(candidate,),
        inference_catalog=inference,
        config=config,
    )
    assert report["status"] == "advance"
    assert report["promotion_decision"]["eligible_count"] == 1
    assert report["promotion_decision"]["promoted"][0]["config_id"] == candidate.config_id
