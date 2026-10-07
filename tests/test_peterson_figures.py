from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from deepbench.peterson_figures import (
    MODEL_COLORS,
    _forest_labels,
    _metric_color_limits,
    _performance,
    _primary_summary,
    _set_publication_style,
)
from deepbench.statistics import mean_ci


def test_forest_labels_include_protocol_for_augmentation_rows() -> None:
    table = pd.DataFrame(
        [
            {
                "protocol": "within_session",
                "model": "EEGNet",
                "comparison": "overlap-center_x6",
            }
        ]
    )

    assert _forest_labels(table) == [
        "EEGNet: overlap − center ×6 (Within-session)"
    ]


def test_publication_style_defines_a_stable_color_for_every_model() -> None:
    _set_publication_style()

    assert set(MODEL_COLORS) == {
        "CSP+LDA",
        "EEGNet",
        "FBCNet",
        "ShallowConvNet",
        "EEGConformer",
    }
    assert len(set(MODEL_COLORS.values())) == len(MODEL_COLORS)


def _primary_metric_frame() -> pd.DataFrame:
    rows = []
    for model_index, model in enumerate(MODEL_COLORS):
        for protocol_index, protocol in enumerate(("within_split", "within_session", "loso")):
            for subject_index, subject in enumerate(("S02", "S03", "S04")):
                for seed in (0, 1):
                    rows.append(
                        {
                            "dataset": "MI-OpenBCI",
                            "task": "motor_imagery_vs_rest",
                            "condition": "overlap",
                            "ica_policy": "none",
                            "metric": "accuracy",
                            "protocol": protocol,
                            "model": model,
                            "subject": subject,
                            "seed": seed,
                            "value": (
                                0.55
                                + 0.02 * model_index
                                + 0.01 * protocol_index
                                + 0.005 * subject_index
                                + 0.001 * seed
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_primary_summary_uses_participant_student_t_intervals() -> None:
    frame = _primary_metric_frame()

    summary = _primary_summary(frame, "accuracy")

    selected = frame.loc[
        (frame["model"] == "EEGNet") & (frame["protocol"] == "within_split")
    ]
    participant_values = selected.groupby("subject")["value"].mean().to_numpy()
    expected = mean_ci(participant_values)
    row = summary.loc[
        (summary["model"] == "EEGNet") & (summary["protocol"] == "within_split")
    ].iloc[0]
    assert np.allclose(
        [row["mean"], row["ci95_low"], row["ci95_high"]], expected
    )
    assert row["n_subjects"] == 3


def test_primary_performance_png_is_deterministic(tmp_path) -> None:
    frame = _primary_metric_frame()
    first = tmp_path / "first"
    second = tmp_path / "second"

    _set_publication_style()
    _performance(frame, first, "accuracy")
    _performance(frame, second, "accuracy")

    first_hash = hashlib.sha256((first / "primary_accuracy.png").read_bytes()).hexdigest()
    second_hash = hashlib.sha256((second / "primary_accuracy.png").read_bytes()).hexdigest()
    assert first_hash == second_hash


def test_metric_color_limits_do_not_clip_kappa_or_accuracy() -> None:
    kappa = np.asarray([-0.13, 0.19, 0.38, 0.72])
    accuracy = np.asarray([0.43, 0.51, 0.91])

    kappa_low, kappa_high = _metric_color_limits("kappa", kappa)
    accuracy_low, accuracy_high = _metric_color_limits("accuracy", accuracy)

    assert kappa_low <= kappa.min() < kappa.max() <= kappa_high
    assert kappa_low <= 0 <= kappa_high
    assert accuracy_low <= accuracy.min() < accuracy.max() <= accuracy_high
    assert 0 <= accuracy_low < accuracy_high <= 1
