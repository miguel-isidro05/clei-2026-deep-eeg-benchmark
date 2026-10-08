"""Non-destructive signal-quality summaries for loaded EEG epochs."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .types import SubjectRecording


def signal_quality_table(recording: SubjectRecording) -> pd.DataFrame:
    """Summarize every channel without excluding trials, channels, or subjects."""
    x = np.asarray(recording.x, dtype=np.float64)
    finite = np.isfinite(x)
    finite_fraction = finite.mean(axis=(0, 2))
    safe = np.where(finite, x, np.nan)
    channel_std = np.nanstd(safe, axis=(0, 2))
    channel_rms = np.sqrt(np.nanmean(np.square(safe), axis=(0, 2)))
    channel_p2p = np.nanmax(safe, axis=(0, 2)) - np.nanmin(safe, axis=(0, 2))
    median_rms = float(np.nanmedian(channel_rms))
    median_std = float(np.nanmedian(channel_std))
    amplitude_ratio = channel_rms / max(median_rms, np.finfo(float).eps)
    std_ratio = channel_std / max(median_std, np.finfo(float).eps)
    rows = []
    for index, channel in enumerate(recording.ch_names):
        rows.append(
            {
                "dataset": recording.dataset,
                "subject": recording.subject,
                "channel": channel,
                "n_trials": int(len(recording.y)),
                "n_samples_per_trial": int(recording.x.shape[-1]),
                "finite_fraction": float(finite_fraction[index]),
                "std_loaded_units": float(channel_std[index]),
                "variance_loaded_units": float(channel_std[index] ** 2),
                "rms_loaded_units": float(channel_rms[index]),
                "peak_to_peak_loaded_units": float(channel_p2p[index]),
                "rms_to_channel_median": float(amplitude_ratio[index]),
                "std_to_channel_median": float(std_ratio[index]),
                "flat_channel_flag": bool(
                    channel_std[index] <= np.finfo(np.float32).eps
                    or channel_p2p[index] <= np.finfo(np.float32).eps
                    or std_ratio[index] < 1e-3
                ),
                "robust_amplitude_flag": bool(amplitude_ratio[index] > 10.0),
                "automatic_exclusion": False,
            }
        )
    return pd.DataFrame(rows)


def signal_quality_metadata(recording: SubjectRecording) -> dict[str, object]:
    """Return auditable class/session counts and loader-level filter availability."""
    return {
        "dataset": recording.dataset,
        "subject": recording.subject,
        "shape": list(recording.x.shape),
        "sfreq": float(recording.sfreq),
        "class_counts": {str(key): int(value) for key, value in Counter(recording.y).items()},
        "session_counts": {
            str(key): int(value) for key, value in Counter(map(str, recording.sessions)).items()
        },
        "loaded_epoch_sha256": recording.data_sha256,
        "loader_bandpass_hz": list(recording.loader_bandpass_hz)
        if recording.loader_bandpass_hz
        else None,
        "line_noise_assessment": "unavailable_after_loader_bandpass"
        if recording.loader_bandpass_hz
        else "not_computed",
        "automatic_exclusion": False,
        "quality_thresholds": {
            "flat_std_or_peak_to_peak": float(np.finfo(np.float32).eps),
            "near_flat_std_to_channel_median": 1e-3,
            "robust_amplitude_rms_to_channel_median": 10.0,
        },
    }
