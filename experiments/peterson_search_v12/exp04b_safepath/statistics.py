"""Small-sample summaries with physical-mask aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def physical_mask_floor(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record["mask_id"])].append(float(record["accuracy"]))
    if not grouped:
        raise ValueError("Physical-mask floor requires at least one record")
    means = {mask_id: float(np.mean(values)) for mask_id, values in grouped.items()}
    worst = min(means, key=lambda item: (means[item], item))
    return {
        "worst_mask_id": worst,
        "worst_mask_accuracy": means[worst],
        "mask_mean_accuracy": means,
    }
