from __future__ import annotations

import numpy as np
import pandas as pd

from deepbench.peterson_postprocessing import (
    all_metric_descriptives,
    friedman_omnibus,
    prediction_diagnostics,
    protocol_gap_tests,
    review_closure_table,
    training_diagnostics,
)


def _metric_frame() -> pd.DataFrame:
    rows = []
    models = ("CSP+LDA", "EEGNet", "FBCNet")
    for protocol in ("within_split", "within_session", "loso"):
        for subject_index in range(6):
            for seed in (0, 1):
                for model_index, model in enumerate(models):
                    for metric in ("accuracy", "kappa", "roc_auc"):
                        protocol_offset = {
                            "within_split": 0.04,
                            "within_session": 0.02,
                            "loso": 0.0,
                        }[protocol]
                        value = 0.55 + 0.03 * model_index + 0.005 * subject_index + protocol_offset
                        if metric == "kappa":
                            value = 2 * value - 1
                        rows.append(
                            {
                                "dataset": "MI-OpenBCI",
                                "task": "motor_imagery_vs_rest",
                                "protocol": protocol,
                                "condition": "overlap",
                                "ica_policy": "none",
                                "model": model,
                                "subject": f"S{subject_index + 2:02d}",
                                "seed": seed,
                                "metric": metric,
                                "value": value,
                                "source": "fixture.json",
                            }
                        )
    return pd.DataFrame(rows)


def test_all_metric_descriptives_do_not_drop_auc() -> None:
    table = all_metric_descriptives(_metric_frame())
    assert set(table["metric"]) == {"accuracy", "kappa", "roc_auc"}
    assert set(table["n_subjects"]) == {6}
    assert table["mean_within_subject_seed_sd"].fillna(0).eq(0).all()


def test_friedman_reports_kendall_w_and_exploratory_scope() -> None:
    table = friedman_omnibus(_metric_frame())
    row = table.query("protocol == 'loso' and metric == 'accuracy'").iloc[0]
    assert row["n_models"] == 3
    assert row["n_subjects"] == 6
    assert 0 <= row["kendall_w"] <= 1
    assert row["analysis_tier"] == "exploratory"


def test_protocol_gap_holm_family_spans_models_and_comparisons() -> None:
    table = protocol_gap_tests(_metric_frame())
    accuracy = table.loc[table["metric"] == "accuracy"]
    assert len(accuracy) == 6
    assert (accuracy["p_holm"] >= accuracy["p_raw"]).all()
    assert accuracy["p_holm"].max() >= 0.18


def test_prediction_and_training_diagnostics_are_explicit() -> None:
    payload = {
        "dataset": "MI-OpenBCI",
        "task": "motor_imagery_vs_rest",
        "protocol": "loso",
        "condition": "overlap",
        "ica_policy": "none",
        "model": "EEGNet",
        "subject": "S02",
        "seed": 0,
        "y_true": [0, 0, 1, 1],
        "y_pred": [0, 1, 1, 1],
        "y_score": [0.1, 0.7, 0.8, 0.9],
        "fold_reports": [
            {
                "training_history": [
                    {
                        "epoch": 1,
                        "train_loss": 1.0,
                        "duration_seconds": 0.2,
                        "train_batch_count": 2,
                    },
                    {
                        "epoch": 2,
                        "train_loss": 0.5,
                        "duration_seconds": 0.3,
                        "train_batch_count": 2,
                    },
                ]
            }
        ],
    }
    predictions = prediction_diagnostics([payload])
    assert predictions.loc[0, "sensitivity"] == 1.0
    assert predictions.loc[0, "specificity"] == 0.5
    assert predictions.loc[0, "predicted_positive_rate"] == 0.75
    assert np.isclose(predictions.loc[0, "brier_score"], 0.1375)
    training = training_diagnostics([payload])
    assert training.loc[0, "epochs_recorded"] == 2
    assert training.loc[0, "optimizer_updates"] == 4
    assert training.loc[0, "best_epoch"] == 2


def test_review_closure_preserves_single_dataset_limitation() -> None:
    table = review_closure_table()
    scope = table.loc[table["item_id"] == "external_low_cost_validation"].iloc[0]
    assert scope["status"] == "unresolved_by_design"
    assert "single" in scope["limitation"].lower()
    latency = table.loc[table["item_id"] == "latency"].iloc[0]
    assert latency["status"] == "pending_measurement"
