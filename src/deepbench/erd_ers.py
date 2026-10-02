"""Predeclared motor-imagery band-power plausibility audits."""

from __future__ import annotations

import numpy as np
import scipy.signal


def band_power(x: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    frequencies, psd = scipy.signal.welch(x, fs=sfreq, axis=-1, nperseg=min(256, x.shape[-1]))
    mask = (frequencies >= low) & (frequencies < high)
    return np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)


def peterson_mi_rest_change(
    x: np.ndarray, y: np.ndarray, ch_names: tuple[str, ...], sfreq: float
) -> dict[str, dict[str, float]]:
    """Return MI relative power change against rest at C3/Cz/C4."""
    result: dict[str, dict[str, float]] = {}
    for low, high, band in ((8.0, 13.0, "mu"), (13.0, 30.0, "beta")):
        power = band_power(x, sfreq, low, high)
        result[band] = {}
        for channel in ("C3", "Cz", "C4"):
            index = ch_names.index(channel)
            rest = float(np.mean(power[y == 0, index]))
            imagery = float(np.mean(power[y == 1, index]))
            result[band][channel] = (imagery - rest) / max(rest, np.finfo(float).eps)
    return result


def souza_lateralization(
    x: np.ndarray, y: np.ndarray, ch_names: tuple[str, ...], sfreq: float
) -> dict[str, dict[str, float]]:
    """Return log(C4/C3) lateralization separately for left and right imagery."""
    c3, c4 = ch_names.index("C3"), ch_names.index("C4")
    result: dict[str, dict[str, float]] = {}
    for low, high, band in ((8.0, 13.0, "mu"), (13.0, 30.0, "beta")):
        power = band_power(x, sfreq, low, high)
        lateral = np.log(np.maximum(power[:, c4], 1e-20)) - np.log(np.maximum(power[:, c3], 1e-20))
        result[band] = {
            "left": float(np.mean(lateral[y == 0])),
            "right": float(np.mean(lateral[y == 1])),
            "left_minus_right": float(np.mean(lateral[y == 0]) - np.mean(lateral[y == 1])),
        }
    return result
