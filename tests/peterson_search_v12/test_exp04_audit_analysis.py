from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pytest

from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp04_qcr_ssnet.analysis import analyze_results
from experiments.peterson_search_v12.exp04_qcr_ssnet.audit import validate_result_cell
from experiments.peterson_search_v12.exp04_qcr_ssnet.catalog import build_model_catalog
from experiments.peterson_search_v12.exp04_qcr_ssnet.config import Exp04Config
from experiments.peterson_search_v12.exp04_qcr_ssnet.source import select_exp03_source


def _summary(
    config_id: str,
    *,
    dropout: float,
    filters: int,
    accuracy: float,
    stress_floor: float,
) -> dict[str, object]:
    return {
        "config_id": config_id,
        "gate_id": "G0",
        "channel_dropout": dropout,
        "n_filters_spat": filters,
        "stride_factor": 4,
        "mean_accuracy": accuracy,
        "lower_quartile": accuracy - 0.02,
        "mean_kappa": 2 * accuracy - 1,
        "collapse_count": 0,
        "wins_vs_control": 6,
        "stress_floor": stress_floor,
        "parameters": {"active": 2500},
    }


def _source():
    report = {
        "control": _summary("control", dropout=0.0, filters=32, accuracy=0.80, stress_floor=0.60),
        "ranking": [
            _summary("clean", dropout=0.0, filters=16, accuracy=0.83, stress_floor=0.61),
            _summary("robust", dropout=0.1, filters=48, accuracy=0.75, stress_floor=0.72),
        ],
    }
    return select_exp03_source(
        report,
        promotion_decision_sha256="a" * 64,
        quality_manifest_sha256="b" * 64,
    )


def _metrics(accuracy: float) -> tuple[dict[str, float], list[int], list[int], list[float]]:
    total = 100
    correct = int(round(accuracy * total))
    y_true = [index % 2 for index in range(total)]
    y_pred = [value if index < correct else 1 - value for index, value in enumerate(y_true)]
    y_score = [0.9 if value == 1 else 0.1 for value in y_pred]
    return (
        compute_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score)),
        y_true,
        y_pred,
        y_score,
    )


def _audit_cell() -> dict[str, object]:
    metrics, y_true, y_pred, y_score = _metrics(0.8)
    source = _source().payload()
    candidate = asdict(build_model_catalog(Exp04Config(), _source())[0])
    split = "c" * 64
    stress_counts = {"missing_1": 15, "missing_2": 105}
    stress_results = {}
    for name, count in stress_counts.items():
        stress_true = y_true * count
        stress_pred = y_pred * count
        stress_score = y_score * count
        stress_results[name] = {
            "metrics": compute_metrics(
                np.asarray(stress_true), np.asarray(stress_pred), np.asarray(stress_score)
            ),
            "y_true": stress_true,
            "y_pred": stress_pred,
            "y_score": stress_score,
            "n_masks_per_fold": count,
            "worst_mask_accuracy": metrics["accuracy"],
        }
    return {
        "schema_version": 4,
        "status": "completed",
        "experiment": "exp04",
        "cell_id": "candidate__test",
        "kind": "candidate",
        "model_id": "QCR-SSNet",
        "config_id": candidate["config_id"],
        "subject": "S09",
        "seed": 0,
        "folds": [0],
        "validation_metrics": metrics,
        "validation_y_true": y_true,
        "validation_y_pred": y_pred,
        "validation_y_score": y_score,
        "quality_flags": [],
        "fold_reports": [
            {
                "n_validation_trials": len(y_true),
                "search_split_sha256": split,
                "checkpoint_policy": "last_epoch",
                "selected_epoch": 300,
                "last_epoch": 300,
                "stress": {
                    "missing_1": {"mask_count": 15},
                    "missing_2": {"mask_count": 105},
                },
                "quality_flags": [],
            }
        ],
        "stress_results": stress_results,
        "source": source,
        "candidate": candidate,
        "fingerprint": {
            "cell_config_sha256": candidate["sha256"],
            "config_sha256": "d" * 64,
            "catalog_sha256": "e" * 64,
            "manifest_sha256": "f" * 64,
            "source_sha256": source["sha256"],
            "code_sha256": "1" * 64,
            "data_sha256": "2" * 64,
            "split_sha256": canonical_sha256([split]),
            "environment_sha256": "3" * 64,
            "git_dirty": False,
        },
    }


def test_exp04_cell_audit_recomputes_metrics_and_rejects_test_leakage():
    payload = _audit_cell()
    validate_result_cell(payload)
    tampered = copy.deepcopy(payload)
    tampered["validation_metrics"]["accuracy"] = 0.99
    with pytest.raises(ValueError, match="do not match"):
        validate_result_cell(tampered)
    leaked = copy.deepcopy(payload)
    leaked["outer_test_predictions"] = [0, 1]
    with pytest.raises(ValueError, match="forbidden"):
        validate_result_cell(leaked)
    selected_best = copy.deepcopy(payload)
    selected_best["fold_reports"][0]["checkpoint_policy"] = "best_val_ce"
    with pytest.raises(ValueError, match="last epoch"):
        validate_result_cell(selected_best)


def test_exp04_cell_audit_rejects_truncated_stress_mask_coverage():
    payload = _audit_cell()
    payload["fold_reports"][0]["stress"]["missing_2"]["mask_count"] = 1
    payload["stress_results"]["missing_2"]["n_masks_per_fold"] = 1
    payload["stress_results"]["missing_2"]["y_true"] = payload["validation_y_true"]
    payload["stress_results"]["missing_2"]["y_pred"] = payload["validation_y_pred"]
    payload["stress_results"]["missing_2"]["y_score"] = payload["validation_y_score"]
    payload["stress_results"]["missing_2"]["metrics"] = payload["validation_metrics"]
    with pytest.raises(ValueError, match="exactly 105"):
        validate_result_cell(payload)


def _analysis_cell(
    *,
    model_id: str,
    config_id: str | None,
    subject: str,
    seed: int,
    accuracy: float,
    stress: float,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "config_id": config_id,
        "subject": subject,
        "seed": seed,
        "validation_metrics": {"accuracy": accuracy, "kappa": 2 * accuracy - 1},
        "quality_flags": [],
        "stress_results": {
            "missing_1": {
                "metrics": {"accuracy": stress},
                "worst_mask_accuracy": stress,
            },
            "missing_2": {
                "metrics": {"accuracy": stress},
                "worst_mask_accuracy": stress,
            },
        },
        "parameters": {"active": 5000},
        "fold_reports": [{"clean_inference_seconds_per_trial": 0.001}],
    }


def test_exp04_analysis_promotes_only_predeclared_full_model():
    config = Exp04Config()
    source = _source()
    catalog = build_model_catalog(config, source)
    payloads: list[dict[str, object]] = []
    for subject in config.subjects:
        payloads.append(
            _analysis_cell(
                model_id="CSP+LDA",
                config_id=None,
                subject=subject,
                seed=0,
                accuracy=0.76,
                stress=0.55,
            )
        )
        for seed in config.seeds:
            payloads.append(
                _analysis_cell(
                    model_id="FBCNet-V10",
                    config_id=None,
                    subject=subject,
                    seed=seed,
                    accuracy=0.80,
                    stress=0.62,
                )
            )
            for candidate in catalog:
                is_full = candidate.role == "qcr_full"
                payloads.append(
                    _analysis_cell(
                        model_id="QCR-SSNet",
                        config_id=candidate.config_id,
                        subject=subject,
                        seed=seed,
                        accuracy=0.83 if is_full else 0.79,
                        stress=0.70 if is_full else 0.61,
                    )
                )
    report = analyze_results(payloads, catalog, config)
    assert report["status"] == "promoted"
    assert report["promotion_decision"]["selected_config_id"] == "exp04-qcr-full"
    assert report["qcr_full"]["primary_success"] is True
    assert report["qcr_full"]["robust_success"] is True
    assert report["csp_baseline"]["mean_accuracy"] == pytest.approx(0.76)
