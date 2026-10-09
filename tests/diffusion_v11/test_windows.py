from __future__ import annotations

import numpy as np

from deepbench.preprocessing import augment_training, prepare_test_windows


def test_center_x6_is_compute_matched_but_not_diverse() -> None:
    x = np.arange(4 * 2 * 512, dtype=np.float32).reshape(4, 2, 512)
    y = np.array([0, 1, 0, 1])
    center_x6, center_y = augment_training(x, y, "center_x6")
    overlap, overlap_y = augment_training(x, y, "overlap")
    assert center_x6.shape == overlap.shape == (24, 2, 256)
    assert np.array_equal(center_y, overlap_y)
    for start in range(0, 24, 6):
        assert all(
            np.array_equal(center_x6[start], center_x6[start + offset])
            for offset in range(6)
        )
        assert any(
            not np.array_equal(overlap[start], overlap[start + offset])
            for offset in range(1, 6)
        )


def test_test_windows_retain_trial_indices() -> None:
    x = np.zeros((3, 2, 512), dtype=np.float32)
    windows, trial_indices = prepare_test_windows(x, "overlap")
    assert windows.shape[0] == 18
    assert np.array_equal(trial_indices, np.repeat(np.arange(3), 6))
