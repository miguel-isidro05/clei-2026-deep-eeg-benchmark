from __future__ import annotations

import numpy as np

from deepbench.preprocessing import (
    aggregate_trial_probabilities,
    augment_training,
    prepare_test_input,
    prepare_test_windows,
    preprocess_split,
)


def test_training_augmentation_preserves_trial_labels() -> None:
    x = np.arange(3 * 2 * 512, dtype=np.float32).reshape(3, 2, 512)
    y = np.array([0, 1, 0])
    nonoverlap_x, nonoverlap_y = augment_training(x, y, "nonoverlap")
    overlap_x, overlap_y = augment_training(x, y, "overlap")
    center_x2, center_y2 = augment_training(x, y, "center_x2")
    center_x6, center_y6 = augment_training(x, y, "center_x6")
    assert nonoverlap_x.shape == (6, 2, 256)
    assert nonoverlap_y.tolist() == [0, 0, 1, 1, 0, 0]
    assert overlap_x.shape == (18, 2, 256)
    assert overlap_y.tolist() == [0] * 6 + [1] * 6 + [0] * 6
    assert center_x2.shape == nonoverlap_x.shape
    assert center_y2.tolist() == nonoverlap_y.tolist()
    assert center_x6.shape == overlap_x.shape
    assert center_y6.tolist() == overlap_y.tolist()
    assert prepare_test_input(x, "overlap").shape == (3, 2, 256)


def test_windowed_test_input_preserves_trial_mapping_and_aggregates_probabilities() -> None:
    x = np.arange(2 * 2 * 512, dtype=np.float32).reshape(2, 2, 512)

    windows, trial_indices = prepare_test_windows(x, "overlap")

    assert windows.shape == (12, 2, 256)
    assert trial_indices.tolist() == [0] * 6 + [1] * 6
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    aggregated = aggregate_trial_probabilities(probabilities, trial_indices, n_trials=2)
    assert np.allclose(aggregated, [0.35, 0.65])


def test_compute_matched_center_windows_repeat_one_crop_per_trial() -> None:
    x = np.arange(2 * 2 * 512, dtype=np.float32).reshape(2, 2, 512)

    windows, trial_indices = prepare_test_windows(x, "center_x6")

    assert windows.shape == (12, 2, 256)
    assert trial_indices.tolist() == [0] * 6 + [1] * 6
    for trial in range(2):
        selected = windows[trial_indices == trial]
        assert all(np.array_equal(selected[0], item) for item in selected[1:])


def test_nonoverlap_test_windows_cover_both_trial_halves() -> None:
    x = np.arange(2 * 1 * 512, dtype=np.float32).reshape(2, 1, 512)

    windows, trial_indices = prepare_test_windows(x, "nonoverlap")

    assert windows.shape == (4, 1, 256)
    assert trial_indices.tolist() == [0, 0, 1, 1]
    assert np.array_equal(windows[0], x[0, :, :256])
    assert np.array_equal(windows[1], x[0, :, 256:])


def test_souza_overlap_keeps_six_compute_matched_windows() -> None:
    x = np.arange(2 * 16 * 384, dtype=np.float32).reshape(2, 16, 384)
    y = np.array([0, 1])

    overlap_x, overlap_y = augment_training(x, y, "overlap")
    center_x6, center_y6 = augment_training(x, y, "center_x6")

    assert overlap_x.shape == (12, 16, 256)
    assert overlap_y.tolist() == [0] * 6 + [1] * 6
    assert center_x6.shape == overlap_x.shape
    assert center_y6.tolist() == overlap_y.tolist()


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
    assert result.report["initial_bandpass_source"] == "preprocess_split"
    assert result.report["initial_bandpass_applied"] is True
    assert len(str(result.report["post_preprocessing_train_sha256"])) == 64


def test_loader_bandpass_is_not_applied_twice(monkeypatch) -> None:
    import deepbench.preprocessing as preprocessing

    calls: list[tuple[float, float]] = []

    def fake_bandpass(x, sfreq, low=8.0, high=30.0):
        calls.append((low, high))
        return x

    monkeypatch.setattr(preprocessing, "_bandpass", fake_bandpass)
    x = np.ones((4, 3, 512), dtype=np.float32)
    result = preprocess_split(
        x,
        x,
        sfreq=128.0,
        ch_names=("C3", "Cz", "C4"),
        seed=0,
        loader_bandpass_hz=(1.0, 40.0),
    )
    assert calls == [(8.0, 30.0), (8.0, 30.0)]
    assert result.report["initial_bandpass_source"] == "dataset_loader"
    assert result.report["initial_bandpass_applied"] is False


def test_moabb_loader_bandpass_skips_duplicate_initial_filter() -> None:
    rng = np.random.default_rng(5)
    x_train = rng.normal(0, 1, (12, 3, 512)).astype(np.float32)
    x_test = rng.normal(0, 1, (4, 3, 512)).astype(np.float32)
    result = preprocess_split(
        x_train,
        x_test,
        sfreq=128.0,
        ch_names=("C3", "Cz", "C4"),
        seed=0,
        ica_policy="none",
        loader_bandpass_hz=(1.0, 40.0),
    )
    assert result.report["initial_bandpass_source"] == "dataset_loader"
    assert result.report["initial_bandpass_applied"] is False


def test_ica_keeps_multiple_components_when_one_pca_component_dominates() -> None:
    rng = np.random.default_rng(17)
    n_train, n_test, n_channels, n_times = 20, 6, 8, 384
    shared_train = rng.normal(size=(n_train, 1, n_times))
    shared_test = rng.normal(size=(n_test, 1, n_times))
    channel_weights = np.linspace(1.0, 2.0, n_channels)[None, :, None]
    x_train = (
        shared_train * channel_weights + 0.01 * rng.normal(size=(n_train, n_channels, n_times))
    ).astype(np.float32)
    x_test = (
        shared_test * channel_weights + 0.01 * rng.normal(size=(n_test, n_channels, n_times))
    ).astype(np.float32)

    result = preprocess_split(
        x_train,
        x_test,
        sfreq=128.0,
        ch_names=tuple(f"C{index}" for index in range(n_channels)),
        seed=2026,
        ica_policy="kurtosis",
    )

    assert result.report["ica"]["n_components"] > 1
    assert result.x_train.shape == x_train.shape
    assert result.x_test.shape == x_test.shape
