"""Leakage-safe filtering, automated ICA, normalization, and augmentation."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.signal
import scipy.stats
from sklearn.exceptions import ConvergenceWarning

from .config import AUGMENT_WINDOW_SAMPLES


@dataclass(frozen=True)
class PreprocessingResult:
    x_train: np.ndarray
    x_test: np.ndarray
    report: dict[str, object]


def _bandpass(x: np.ndarray, sfreq: float, low: float = 8.0, high: float = 30.0) -> np.ndarray:
    sos = scipy.signal.butter(5, (low, high), btype="bandpass", fs=sfreq, output="sos")
    return scipy.signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def _array_sha256(x: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(x, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _apply_train_fitted_ica(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    sfreq: float,
    ch_names: tuple[str, ...],
    seed: int,
    kurtosis_threshold: float = 10.0,
    max_excluded: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit FastICA on training epochs and select artifacts without labels or test data."""
    import mne
    from mne.preprocessing import ICA

    info = mne.create_info(list(ch_names), sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    train_epochs = mne.EpochsArray(np.asarray(x_train, np.float64), info, verbose="ERROR")
    test_epochs = mne.EpochsArray(np.asarray(x_test, np.float64), info, verbose="ERROR")
    ica = ICA(
        n_components=min(x_train.shape[1], len(ch_names)),
        method="fastica",
        random_state=seed,
        max_iter="auto",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        ica.fit(train_epochs, verbose="ERROR")
    convergence_warnings = [
        str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    if convergence_warnings:
        raise RuntimeError(
            "FastICA did not converge; this fold is not written and must be investigated: "
            + " | ".join(convergence_warnings)
        )
    sources = ica.get_sources(train_epochs).get_data(copy=True)
    flattened = np.transpose(sources, (1, 0, 2)).reshape(sources.shape[1], -1)
    kurtosis = scipy.stats.kurtosis(flattened, axis=1, fisher=False, bias=False)
    candidates = np.flatnonzero(kurtosis > kurtosis_threshold).tolist()
    excluded = sorted(candidates, key=lambda index: float(kurtosis[index]), reverse=True)[
        :max_excluded
    ]
    ica.exclude = [int(index) for index in excluded]
    cleaned_train = train_epochs.copy()
    cleaned_test = test_epochs.copy()
    ica.apply(cleaned_train, verbose="ERROR")
    ica.apply(cleaned_test, verbose="ERROR")
    report = {
        "policy": "kurtosis",
        "fit_partition": "train_only",
        "algorithm": "FastICA",
        "converged": True,
        "n_iterations": int(ica.n_iter_),
        "failure_policy": "raise_and_do_not_write_cell",
        "n_components": int(ica.n_components_),
        "n_components_parameter": int(min(x_train.shape[1], len(ch_names))),
        "pca_nonzero_variance_threshold": 0.999999,
        "kurtosis_threshold": float(kurtosis_threshold),
        "max_excluded": int(max_excluded),
        "component_kurtosis": [float(value) for value in kurtosis],
        "candidate_components": [int(index) for index in candidates],
        "excluded_components": [int(index) for index in excluded],
        "manual_selection": False,
    }
    return (
        cleaned_train.get_data(copy=True).astype(np.float32),
        cleaned_test.get_data(copy=True).astype(np.float32),
        report,
    )


def preprocess_split(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    sfreq: float,
    ch_names: tuple[str, ...],
    seed: int,
    ica_policy: str = "none",
    loader_bandpass_hz: tuple[float, float] | None = None,
) -> PreprocessingResult:
    """Apply a train-only preprocessing fit and return an auditable report."""
    train = np.ascontiguousarray(x_train, dtype=np.float32)
    test = np.ascontiguousarray(x_test, dtype=np.float32)
    if loader_bandpass_hz is None:
        train = _bandpass(train, sfreq, low=1.0, high=40.0)
        test = _bandpass(test, sfreq, low=1.0, high=40.0)
        initial_bandpass_source = "preprocess_split"
    else:
        if tuple(map(float, loader_bandpass_hz)) != (1.0, 40.0):
            raise ValueError(
                "Loaded epochs must use the frozen 1-40 Hz bandpass before final filtering"
            )
        initial_bandpass_source = "dataset_loader"
    if ica_policy == "kurtosis":
        train, test, ica_report = _apply_train_fitted_ica(
            train,
            test,
            sfreq=sfreq,
            ch_names=ch_names,
            seed=seed,
        )
    elif ica_policy == "none":
        ica_report = {
            "policy": "none",
            "fit_partition": None,
            "excluded_components": [],
            "manual_selection": False,
        }
    else:
        raise ValueError(f"Unsupported ICA policy: {ica_policy}")
    train = _bandpass(train, sfreq)
    test = _bandpass(test, sfreq)
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True) + 1e-6
    train = ((train - mean) / std).astype(np.float32)
    test = ((test - mean) / std).astype(np.float32)
    return PreprocessingResult(
        x_train=train,
        x_test=test,
        report={
            "ica": ica_report,
            "initial_bandpass_hz": [1.0, 40.0],
            "initial_bandpass_source": initial_bandpass_source,
            "initial_bandpass_applied": initial_bandpass_source == "preprocess_split",
            "loader_bandpass_hz": list(loader_bandpass_hz) if loader_bandpass_hz else None,
            "final_bandpass_hz": [8.0, 30.0],
            "zscore_fit_partition": "train_only",
            "post_preprocessing_train_sha256": _array_sha256(train),
            "post_preprocessing_test_sha256": _array_sha256(test),
            "n_train_trials": int(len(train)),
            "n_test_trials": int(len(test)),
        },
    )


def center_crop(x: np.ndarray, samples: int = AUGMENT_WINDOW_SAMPLES) -> np.ndarray:
    if samples > x.shape[-1]:
        raise ValueError(f"Cannot crop {samples} samples from epochs with {x.shape[-1]}")
    start = (x.shape[-1] - samples) // 2
    return np.ascontiguousarray(x[..., start : start + samples], dtype=np.float32)


def _window_starts(n_times: int, condition: str) -> tuple[int, ...]:
    """Return deterministic window starts for one trial-level condition."""
    if AUGMENT_WINDOW_SAMPLES > n_times:
        raise ValueError("Augmentation window exceeds trial length")
    if condition == "nonoverlap":
        return (0, n_times - AUGMENT_WINDOW_SAMPLES)
    if condition == "overlap":
        max_start = n_times - AUGMENT_WINDOW_SAMPLES
        return tuple(np.linspace(0, max_start, 6, dtype=int))
    raise ValueError(f"Unsupported window condition: {condition}")


def augment_training(x: np.ndarray, y: np.ndarray, condition: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply a model-independent training condition with no cross-trial mixing."""
    if condition == "full":
        return np.ascontiguousarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)
    if condition in {"center", "center_x2", "center_x6"}:
        repetitions = {"center": 1, "center_x2": 2, "center_x6": 6}[condition]
        cropped = center_crop(x)
        return (
            np.repeat(cropped, repetitions, axis=0).astype(np.float32, copy=False),
            np.repeat(np.asarray(y, dtype=np.int64), repetitions),
        )
    starts = _window_starts(x.shape[-1], condition)
    windows = [trial[:, start : start + AUGMENT_WINDOW_SAMPLES] for trial in x for start in starts]
    labels = np.repeat(np.asarray(y, dtype=np.int64), len(starts))
    return np.ascontiguousarray(windows, dtype=np.float32), labels


def prepare_test_input(x: np.ndarray, condition: str) -> np.ndarray:
    """Keep evaluation at one prediction per original trial."""
    if condition == "full":
        return np.ascontiguousarray(x, dtype=np.float32)
    if condition in {"center", "center_x2", "center_x6", "nonoverlap", "overlap"}:
        return center_crop(x)
    raise ValueError(f"Unsupported condition: {condition}")


def prepare_test_windows(x: np.ndarray, condition: str) -> tuple[np.ndarray, np.ndarray]:
    """Expand test trials while retaining the index needed for trial-level voting."""
    trials = np.ascontiguousarray(x, dtype=np.float32)
    if condition == "full":
        return trials, np.arange(len(trials), dtype=np.int64)
    if condition == "center":
        return center_crop(trials), np.arange(len(trials), dtype=np.int64)
    if condition in {"center_x2", "center_x6"}:
        repetitions = {"center_x2": 2, "center_x6": 6}[condition]
        cropped = center_crop(trials)
        return (
            np.repeat(cropped, repetitions, axis=0).astype(np.float32, copy=False),
            np.repeat(np.arange(len(trials), dtype=np.int64), repetitions),
        )
    starts = _window_starts(trials.shape[-1], condition)
    windows = [
        trial[:, start : start + AUGMENT_WINDOW_SAMPLES] for trial in trials for start in starts
    ]
    trial_indices = np.repeat(np.arange(len(trials), dtype=np.int64), len(starts))
    return np.ascontiguousarray(windows, dtype=np.float32), trial_indices


def aggregate_trial_probabilities(
    probabilities: np.ndarray,
    trial_indices: np.ndarray,
    *,
    n_trials: int,
) -> np.ndarray:
    """Average window probabilities into one score for each original trial."""
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    indices = np.asarray(trial_indices, dtype=np.int64).reshape(-1)
    if len(values) != len(indices):
        raise ValueError("Window probabilities and trial indices must have equal length")
    if n_trials < 1 or np.any(indices < 0) or np.any(indices >= n_trials):
        raise ValueError("Trial indices fall outside the declared trial count")
    counts = np.bincount(indices, minlength=n_trials)
    if np.any(counts == 0):
        raise ValueError("Every trial must contribute at least one window probability")
    totals = np.bincount(indices, weights=values, minlength=n_trials)
    return totals / counts
