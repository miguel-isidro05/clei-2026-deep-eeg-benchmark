"""Participant-level paired statistics for V11 decisions."""

from __future__ import annotations

import numpy as np
import scipy.stats


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (len(values) - rank))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = scipy.stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_summary(
    reference: dict[str, float], candidate: dict[str, float]
) -> dict[str, float | int | list[float]]:
    subjects = sorted(set(reference) & set(candidate))
    if not subjects:
        raise ValueError("Paired comparison has no common subjects")
    differences = np.asarray([candidate[s] - reference[s] for s in subjects], dtype=float)
    if len(differences) > 1 and np.any(differences != 0):
        p_value = float(scipy.stats.wilcoxon(differences, alternative="two-sided").pvalue)
    else:
        p_value = 1.0
    rng = np.random.default_rng(2026)
    bootstrap = np.asarray(
        [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(10_000)]
    )
    return {
        "n_subjects": len(subjects),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "confidence_interval_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "favorable_subjects": int(np.sum(differences > 0)),
        "wilcoxon_p": p_value,
        "rank_biserial": float(_rank_biserial(differences)),
    }
