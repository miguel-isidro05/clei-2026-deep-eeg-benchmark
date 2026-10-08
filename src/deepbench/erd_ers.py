"""Predeclared motor-imagery band-power plausibility audits."""

from __future__ import annotations

import numpy as np
import scipy.signal

BANDS = ((8.0, 13.0, "mu"), (13.0, 30.0, "beta"))


def band_power(x: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    frequencies, psd = scipy.signal.welch(
        x, fs=sfreq, axis=-1, nperseg=min(256, x.shape[-1])
    )
    mask = (frequencies >= low) & (frequencies < high)
    return np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)


def _mean_ci(values: np.ndarray, *, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _peterson_summary(
    x: np.ndarray, y: np.ndarray, ch_names: tuple[str, ...], sfreq: float
) -> dict[str, object]:
    result: dict[str, object] = {
        "trial_counts": {"rest": int(np.sum(y == 0)), "motor_imagery": int(np.sum(y == 1))}
    }
    for band_index, (low, high, band) in enumerate(BANDS):
        log_power = np.log(np.maximum(band_power(x, sfreq, low, high), 1e-20))
        band_result: dict[str, object] = {}
        for channel_index, channel in enumerate(("C3", "Cz", "C4")):
            index = ch_names.index(channel)
            rest = log_power[y == 0, index]
            imagery = log_power[y == 1, index]
            change = imagery.mean() - rest.mean()
            rng = np.random.default_rng(2026 + band_index * 10 + channel_index)
            boot = (
                rng.choice(imagery, size=(10_000, len(imagery)), replace=True).mean(axis=1)
                - rng.choice(rest, size=(10_000, len(rest)), replace=True).mean(axis=1)
            )
            band_result[channel] = float(change)
            band_result[f"{channel}_ci95"] = [
                float(value) for value in np.quantile(boot, (0.025, 0.975))
            ]
        result[band] = band_result
    return result


def peterson_mi_rest_change(
    x: np.ndarray,
    y: np.ndarray,
    ch_names: tuple[str, ...],
    sfreq: float,
    sessions: np.ndarray | None = None,
) -> dict[str, object]:
    """Return MI-minus-rest log-power changes with counts, intervals, and run summaries."""
    result = _peterson_summary(x, y, ch_names, sfreq)
    result["definition"] = "mean_log_power_motor_imagery_minus_rest"
    if sessions is not None:
        result["runs"] = {
            str(run): _peterson_summary(
                x[np.asarray(sessions) == run],
                y[np.asarray(sessions) == run],
                ch_names,
                sfreq,
            )
            for run in sorted(set(np.asarray(sessions).tolist()), key=str)
        }
    return result


def _souza_summary(
    x: np.ndarray, y: np.ndarray, ch_names: tuple[str, ...], sfreq: float
) -> dict[str, object]:
    c3, c4 = ch_names.index("C3"), ch_names.index("C4")
    result: dict[str, object] = {
        "trial_counts": {"left": int(np.sum(y == 0)), "right": int(np.sum(y == 1))}
    }
    for band_index, (low, high, band) in enumerate(BANDS):
        log_power = np.log(np.maximum(band_power(x, sfreq, low, high), 1e-20))
        c4_minus_c3 = log_power[:, c4] - log_power[:, c3]
        left = c4_minus_c3[y == 0]
        right = c4_minus_c3[y == 1]
        result[band] = {
            "left": float(left.mean()),
            "right": float(right.mean()),
            "left_minus_right": float(left.mean() - right.mean()),
            "left_contralateral_minus_ipsilateral": float(left.mean()),
            "right_contralateral_minus_ipsilateral": float((-right).mean()),
            "left_ci95": _mean_ci(left, seed=2026 + band_index * 10),
            "right_ci95": _mean_ci(right, seed=2027 + band_index * 10),
        }
    return result


def souza_lateralization(
    x: np.ndarray,
    y: np.ndarray,
    ch_names: tuple[str, ...],
    sfreq: float,
    sessions: np.ndarray | None = None,
) -> dict[str, object]:
    """Return C3/C4 lateralization with contralateral indices and run summaries."""
    result = _souza_summary(x, y, ch_names, sfreq)
    result["definition"] = "log_band_power_lateralization"
    if sessions is not None:
        result["runs"] = {
            str(run): _souza_summary(
                x[np.asarray(sessions) == run],
                y[np.asarray(sessions) == run],
                ch_names,
                sfreq,
            )
            for run in sorted(set(np.asarray(sessions).tolist()), key=str)
        }
    return result
