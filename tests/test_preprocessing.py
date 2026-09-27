from __future__ import annotations

import numpy as np

from deepbench.preprocessing import augment_training, prepare_test_input, preprocess_split


def test_training_augmentation_preserves_trial_labels() -> None:
    x = np.arange(3 * 2 * 512, dtype=np.float32).reshape(3, 2, 512)
    y = np.array([0, 1, 0])
    nonoverlap_x, nonoverlap_y = augment_training(x, y, "nonoverlap")
    overlap_x, overlap_y = augment_training(x, y, "overlap")
    assert nonoverlap_x.shape == (6, 2, 256)
    assert nonoverlap_y.tolist() == [0, 0, 1, 1, 0, 0]
    assert overlap_x.shape == (18, 2, 256)
    assert overlap_y.tolist() == [0] * 6 + [1] * 6 + [0] * 6
    assert prepare_test_input(x, "overlap").shape == (3, 2, 256)


def test_normalization_is_fitted_only_on_training_partition() -> None:
    rng = np.random.default_rng(4)
    x_train = rng.normal(0, 1, (12, 3, 512)).astype(np.float32)
    x_test = rng.normal(0, 10, (4, 3, 512)).astype(np.float32)
    result = preprocess_split(
        x_train,
        x_test,
        sfreq=128.0,
        ch_names=("C3", "Cz", "C4"),
        seed=0,
        ica_policy="none",
    )
    assert np.allclose(result.x_train.mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(result.x_train.std(axis=(0, 2)), 1.0, atol=1e-4)
    assert np.all(result.x_test.std(axis=(0, 2)) > 5.0)
    assert result.report["zscore_fit_partition"] == "train_only"
