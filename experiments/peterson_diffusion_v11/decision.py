"""Predeclared hypothesis gates."""

from __future__ import annotations

import numpy as np


def decide_h1(
    center_x6: dict[str, float], overlap: dict[str, float]
) -> dict[str, object]:
    subjects = sorted(set(center_x6) & set(overlap))
    if len(subjects) != 10:
        raise ValueError("H1 requires the same ten Peterson subjects")
    differences = np.asarray([overlap[s] - center_x6[s] for s in subjects], dtype=float)
    mean_difference = float(differences.mean())
    favorable = int(np.sum(differences > 0))
    advance = mean_difference >= 0.02 and favorable >= 7
    return {
        "hypothesis": "H1",
        "decision": "advance" if advance else "revise",
        "mean_accuracy_difference": mean_difference,
        "favorable_subjects": favorable,
        "thresholds": {"mean_difference": 0.02, "favorable_subjects": 7},
        "primary_condition": "overlap" if advance else "full",
    }
