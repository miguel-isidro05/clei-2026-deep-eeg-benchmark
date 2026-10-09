from __future__ import annotations

import numpy as np

from experiments.peterson_diffusion_v11.statistics import holm_adjust, paired_summary


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_paired_summary_uses_subject_differences() -> None:
    summary = paired_summary(
        {"S02": 0.70, "S03": 0.60, "S04": 0.80},
        {"S02": 0.75, "S03": 0.65, "S04": 0.85},
    )
    assert np.isclose(summary["mean_difference"], 0.05)
    assert summary["favorable_subjects"] == 3
    assert summary["n_subjects"] == 3
    assert -1 <= summary["rank_biserial"] <= 1
