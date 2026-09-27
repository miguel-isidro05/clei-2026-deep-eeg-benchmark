from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deepbench.config import MODEL_NAMES
from deepbench.statistics import (
    aggregate_seeds,
    completeness_table,
    load_cells,
    mean_ci,
    paired_model_tests,
    paired_rank_biserial,
    sample_accounting_table,
    write_statistics,
)


def _frame() -> pd.DataFrame:
    rows = []
    for subject in range(8):
        for model_index, model in enumerate(MODEL_NAMES):
            for seed in range(5):
                rows.append(
                    {
                        "dataset": "D",
                        "task": "binary",
                        "protocol": "within_split",
                        "condition": "full",
                        "ica_policy": "kurtosis",
                        "model": model,
                        "subject": str(subject),
                        "seed": seed,
                        "metric": "accuracy",
                        "value": 0.6 + model_index * 0.02 + subject * 0.001 + seed * 0.0001,
                        "source": "synthetic",
                    }
                )
    return pd.DataFrame(rows)


def test_seed_repeats_are_averaged_before_inference() -> None:
    frame = _frame()
    aggregated = aggregate_seeds(frame)
    assert len(aggregated) == len(MODEL_NAMES) * 8
    tests = paired_model_tests(frame)
    assert len(tests) == 10
    assert set(tests["n_subjects"]) == {8}
    assert tests["p_holm"].between(0, 1).all()


def test_completeness_reports_missing_seed() -> None:
    frame = _frame()
    dropped = frame.drop(frame.index[0])
    table = completeness_table(dropped, (0, 1, 2, 3, 4))
    incomplete = table.loc[~table["complete"]]
    assert len(incomplete) == 1
    assert incomplete.iloc[0]["missing_seeds"] == "0"


def test_student_t_confidence_interval_contains_mean() -> None:
    mean, low, high = mean_ci(np.array([0.1, 0.2, 0.3, 0.4]))
    assert low < mean < high
    assert np.isclose(mean, 0.25)


def test_paired_rank_biserial_keeps_difference_direction() -> None:
    assert paired_rank_biserial(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert paired_rank_biserial(np.array([-1.0, -2.0, -3.0])) == pytest.approx(-1.0)
    assert paired_rank_biserial(np.zeros(3)) == pytest.approx(0.0)


def test_sample_accounting_exports_fold_class_counts() -> None:
    payloads = [
        {
            "dataset": "D",
            "task": "binary",
            "protocol": "within_split",
            "condition": "overlap",
            "ica_policy": "none",
            "model": "EEGNet",
            "subject": "1",
            "seed": 0,
            "fold_reports": [
                {
                    "session": "session_0",
                    "fold": 0,
                    "n_train_trials": 10,
                    "n_train_examples": 60,
                    "n_test_trials": 4,
                    "train_class_counts": {"0": 5, "1": 5},
                    "test_class_counts": {"0": 2, "1": 2},
                }
            ],
        }
    ]
    table = sample_accounting_table(payloads)
    assert len(table) == 1
    assert table.iloc[0]["n_train_examples"] == 60
    assert table.iloc[0]["train_class_0"] == 5
    assert table.iloc[0]["train_class_1"] == 5
    assert table.iloc[0]["test_class_0"] == 2
    assert table.iloc[0]["test_class_1"] == 2


def test_paired_models_require_identical_subject_cohorts() -> None:
    frame = _frame()
    mask = (frame["model"] == "EEGNet") & (frame["subject"] == "0")
    with pytest.raises(ValueError, match="cohort mismatch"):
        paired_model_tests(frame.loc[~mask])


def test_incomplete_seed_grid_blocks_inferential_outputs(tmp_path) -> None:
    cells = tmp_path / "cells"
    cells.mkdir()
    payload = {
        "dataset": "D",
        "task": "binary",
        "protocol": "within_split",
        "condition": "full",
        "ica_policy": "none",
        "model": "EEGNet",
        "subject": "1",
        "seed": 0,
        "metrics": {"accuracy": 0.5, "kappa": 0.0},
        "run_configuration": {
            "schema_version": 1,
            "code_sha256": "abc",
            "epochs": 300,
            "split_seed": 2026,
            "device_type": "cuda",
            "hardware": "GPU",
            "deterministic_policy": "torch_deterministic_warn_only_cudnn_deterministic",
            "recipe": {"optimizer": "adamw"},
        },
    }
    import json

    (cells / "cell.json").write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "statistics"
    with pytest.raises(RuntimeError, match="Incomplete seed grid"):
        write_statistics(cells, output, (0, 1))
    assert (output / "completeness.csv").exists()
    assert not (output / "paired_wilcoxon_holm.csv").exists()
    exploratory = tmp_path / "exploratory"
    write_statistics(cells, exploratory, (0, 1), allow_incomplete=True)
    assert (exploratory / "INCOMPLETE_EXPLORATORY_ONLY.txt").exists()
    assert not (exploratory / "paired_wilcoxon_holm.csv").exists()


def test_missing_expected_group_forces_exploratory_outputs(tmp_path) -> None:
    import json

    cells = tmp_path / "cells"
    cells.mkdir()
    for seed in range(5):
        payload = {
            "dataset": "D",
            "task": "binary",
            "protocol": "within_split",
            "condition": "full",
            "ica_policy": "none",
            "model": "EEGNet",
            "subject": "1",
            "seed": seed,
            "metrics": {"accuracy": 0.5, "kappa": 0.0},
            "run_configuration": {
                "schema_version": 1,
                "code_sha256": "abc",
                "epochs": 300,
                "split_seed": 2026,
                "device_type": "cuda",
                "hardware": "GPU",
                "deterministic_policy": "torch_deterministic_warn_only_cudnn_deterministic",
                "recipe": {"optimizer": "adamw"},
            },
        }
        (cells / f"seed-{seed}.json").write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "forced-exploratory"
    write_statistics(
        cells,
        output,
        (0, 1, 2, 3, 4),
        allow_incomplete=True,
        force_exploratory=True,
    )
    assert (output / "INCOMPLETE_EXPLORATORY_ONLY.txt").exists()
    assert not (output / "paired_wilcoxon_holm.csv").exists()


def test_malformed_cell_is_not_silently_ignored(tmp_path) -> None:
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed result cell"):
        load_cells(cells)


def test_mixed_code_versions_across_subjects_block_analysis(tmp_path) -> None:
    import copy
    import json

    cells = tmp_path / "cells"
    cells.mkdir()
    base = {
        "dataset": "D",
        "task": "binary",
        "protocol": "within_split",
        "condition": "full",
        "ica_policy": "none",
        "model": "EEGNet",
        "seed": 0,
        "metrics": {"accuracy": 0.5, "kappa": 0.0},
        "run_configuration": {
            "schema_version": 1,
            "code_sha256": "code-a",
            "epochs": 300,
            "split_seed": 2026,
            "device_type": "cuda",
            "hardware": "GPU",
            "deterministic_policy": "torch_deterministic_warn_only_cudnn_deterministic",
            "recipe": {"optimizer": "adamw"},
        },
    }
    first = copy.deepcopy(base)
    first["subject"] = "1"
    second = copy.deepcopy(base)
    second["subject"] = "2"
    second["run_configuration"]["code_sha256"] = "code-b"
    (cells / "one.json").write_text(json.dumps(first), encoding="utf-8")
    (cells / "two.json").write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible result cells"):
        load_cells(cells)
