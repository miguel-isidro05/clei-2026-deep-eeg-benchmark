"""Predeclared exp01 ranking and diversity-constrained promotion."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .identity import canonical_sha256


def rank_and_promote(payloads: list[dict[str, Any]], *, expected_cells: int) -> dict[str, Any]:
    if len(payloads) != expected_cells:
        raise ValueError(
            f"Promotion requires complete results: expected={expected_cells} found={len(payloads)}"
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[payload["config_id"]].append(payload)
    if len(grouped) != 36 or any(len(items) != 6 for items in grouped.values()):
        raise ValueError("Promotion requires six discovery-subject results for all 36 candidates")
    ranking: list[dict[str, Any]] = []
    for config_id, items in grouped.items():
        subjects = {item["subject"] for item in items}
        if len(subjects) != 6:
            raise ValueError(f"Candidate {config_id} does not contain six unique subjects")
        accuracies = np.asarray(
            [item["validation_metrics"]["accuracy"] for item in items], dtype=float
        )
        candidate = items[0]["candidate"]
        ranking.append(
            {
                "config_id": config_id,
                "candidate_sha256": candidate["sha256"],
                "backbone": candidate["backbone"],
                "formulation": candidate["formulation"],
                "width": candidate["width"],
                "mean_accuracy": float(accuracies.mean()),
                "lower_quartile_accuracy": float(np.quantile(accuracies, 0.25)),
                "mean_kappa": float(
                    np.mean([item["validation_metrics"]["kappa"] for item in items])
                ),
                "parameters": int(items[0]["parameters"]["total"]),
                "mean_inference_seconds_per_trial": float(
                    np.mean([item["runtime"]["inference_seconds_per_trial"] for item in items])
                ),
            }
        )
    ranking.sort(
        key=lambda item: (
            -item["mean_accuracy"],
            -item["lower_quartile_accuracy"],
            item["parameters"],
            item["mean_inference_seconds_per_trial"],
            item["config_id"],
        )
    )
    promoted: list[dict[str, Any]] = []
    backbone_counts: Counter[str] = Counter()
    for item in ranking:
        if backbone_counts[item["backbone"]] >= 3:
            continue
        promoted.append(item)
        backbone_counts[item["backbone"]] += 1
        if len(promoted) == 12:
            break
    formulations = {item["formulation"] for item in promoted}
    if len(formulations) < 2:
        for remove_index in range(len(promoted) - 1, -1, -1):
            removed = promoted[remove_index]
            adjusted_counts = backbone_counts.copy()
            adjusted_counts[removed["backbone"]] -= 1
            alternative = next(
                (
                    item
                    for item in ranking
                    if item["formulation"] not in formulations
                    and item not in promoted
                    and adjusted_counts[item["backbone"]] < 3
                ),
                None,
            )
            if alternative is not None:
                not_completely_dominated = (
                    alternative["mean_accuracy"] >= removed["mean_accuracy"]
                    or alternative["parameters"] < removed["parameters"]
                    or alternative["mean_inference_seconds_per_trial"]
                    < removed["mean_inference_seconds_per_trial"]
                )
                if not_completely_dominated:
                    promoted[remove_index] = alternative
                    break
    if len(promoted) != 12:
        raise RuntimeError("Diversity constraints could not promote 12 candidates")
    base = {"ranking": ranking, "promoted": promoted, "expected_cells": expected_cells}
    return {**base, "decision_sha256": canonical_sha256(base), "status": "proposed"}
