from __future__ import annotations

import numpy as np

from deepbench.transfer_statistics import (
    CONTRASTS,
    _bootstrap_mean_interval,
    _paired_rank_biserial,
)


def test_transfer_statistics_include_all_predeclared_contrasts() -> None:
    assert CONTRASTS == (
        ("linear_probe", "scratch"),
        ("full_finetune", "scratch"),
        ("full_finetune", "linear_probe"),
    )


def test_rank_biserial_preserves_direction() -> None:
    assert _paired_rank_biserial(np.asarray([1.0, 2.0, 3.0])) == 1.0
    assert _paired_rank_biserial(np.asarray([-1.0, -2.0, -3.0])) == -1.0


def test_bootstrap_interval_is_reproducible() -> None:
    differences = np.asarray([0.01, 0.02, 0.03, 0.04, 0.05])

    first = _bootstrap_mean_interval(differences, seed=2026, replicates=1_000)
    second = _bootstrap_mean_interval(differences, seed=2026, replicates=1_000)

    assert first == second
    assert first[0] <= differences.mean() <= first[1]
