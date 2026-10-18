"""Wave summaries, staged selection, and final Exp04b promotion decision."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.io import write_json_atomic

from .config import Exp04bConfig


def _read_cells(output_dir: Path, wave: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "cells" / wave).glob("*.json"))
    ]


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty metric list")
    return float(np.mean(values))


def _summarize_config(config_id: str, values: list[dict[str, Any]]) -> dict[str, Any]:
    subjects = sorted({str(value["subject"]) for value in values})
    subject_metrics: dict[str, dict[str, float]] = {}
    for subject in subjects:
        subject_cells = [value for value in values if value["subject"] == subject]
        subject_metrics[subject] = {
            "accuracy": _mean(
                [float(value["validation_metrics"]["accuracy"]) for value in subject_cells]
            ),
            "kappa": _mean(
                [float(value["validation_metrics"]["kappa"]) for value in subject_cells]
            ),
            "missing_1_floor": _mean(
                [
                    float(value["physical_mask_floor"]["missing_1"]["worst_mask_accuracy"])
                    for value in subject_cells
                ]
            ),
            "missing_2_floor": _mean(
                [
                    float(value["physical_mask_floor"]["missing_2"]["worst_mask_accuracy"])
                    for value in subject_cells
                ]
            ),
        }
    accuracies = [metrics["accuracy"] for metrics in subject_metrics.values()]
    return {
        "config_id": config_id,
        "model_id": str(values[0]["model_id"]),
        "role": str(values[0].get("candidate", {}).get("role", "classical_control")),
        "mean_accuracy": _mean(accuracies),
        "lower_quartile": float(np.quantile(accuracies, 0.25)),
        "mean_kappa": _mean([metrics["kappa"] for metrics in subject_metrics.values()]),
        "collapse_count": sum(bool(value["quality_flags"]) for value in values),
        "subject_metrics": subject_metrics,
        "missing_1_floor": _mean(
            [metrics["missing_1_floor"] for metrics in subject_metrics.values()]
        ),
        "missing_2_floor": _mean(
            [metrics["missing_2_floor"] for metrics in subject_metrics.values()]
        ),
    }


def _exact_sign_flip_pvalue(deltas: list[float]) -> float:
    values = np.asarray(deltas, dtype=float)
    if len(values) == 0:
        raise ValueError("Paired test requires at least one subject")
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        for signs in product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def _holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _write_w2_selection(output_dir: Path, report: dict[str, Any]) -> None:
    eligible = [item for item in report["ranking"] if item["collapse_count"] == 0]
    if len(eligible) < 3:
        raise RuntimeError("W2 did not produce three zero-collapse candidates")
    selected = tuple(item["config_id"] for item in eligible[:3])
    body = {
        "status": "selected",
        "wave": "w2",
        "selected_config_ids": list(selected),
        "selection_rule": "zero-collapse_then_clean_then_physical-mask-floor",
        "readout_sha256": canonical_sha256(report),
    }
    write_json_atomic(
        output_dir / "decisions" / "w2-selection.json",
        {**body, "sha256": canonical_sha256(body)},
    )


def build_final_decision(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen W3 rule without conflating the two stress endpoints."""

    config = Exp04bConfig()
    by_id = {item["config_id"]: item for item in report["ranking"]}
    baseline = by_id.get("exp04b-w3-fbc-control")
    if baseline is None:
        raise RuntimeError("W3 is missing the predeclared FBCNet control")
    promoted = [item for item in report["ranking"] if item["role"] == "promoted"]
    tests: list[dict[str, Any]] = []
    for candidate in promoted:
        common = sorted(set(candidate["subject_metrics"]) & set(baseline["subject_metrics"]))
        clean_deltas = [
            candidate["subject_metrics"][subject]["accuracy"]
            - baseline["subject_metrics"][subject]["accuracy"]
            for subject in common
        ]
        missing_1_deltas = [
            candidate["subject_metrics"][subject]["missing_1_floor"]
            - baseline["subject_metrics"][subject]["missing_1_floor"]
            for subject in common
        ]
        missing_2_deltas = [
            candidate["subject_metrics"][subject]["missing_2_floor"]
            - baseline["subject_metrics"][subject]["missing_2_floor"]
            for subject in common
        ]
        tests.append(
            {
                "config_id": candidate["config_id"],
                "subjects": common,
                "clean_mean_delta": _mean(clean_deltas),
                "missing_1_mean_delta": _mean(missing_1_deltas),
                "missing_2_mean_delta": _mean(missing_2_deltas),
                "clean_wins": sum(delta > 0 for delta in clean_deltas),
                "stress_wins": sum(
                    left > 0 and right > 0
                    for left, right in zip(missing_1_deltas, missing_2_deltas, strict=True)
                ),
                "clean_sign_flip_p": _exact_sign_flip_pvalue(clean_deltas),
                "missing_1_sign_flip_p": _exact_sign_flip_pvalue(missing_1_deltas),
                "missing_2_sign_flip_p": _exact_sign_flip_pvalue(missing_2_deltas),
                "collapse_count": candidate["collapse_count"],
            }
        )
    raw_pvalues = [
        float(item[field])
        for item in tests
        for field in (
            "clean_sign_flip_p",
            "missing_1_sign_flip_p",
            "missing_2_sign_flip_p",
        )
    ]
    adjusted = iter(_holm_adjust(raw_pvalues))
    for item in tests:
        item["clean_holm_p"] = next(adjusted)
        item["missing_1_holm_p"] = next(adjusted)
        item["missing_2_holm_p"] = next(adjusted)
        item["passes_predeclared_margins"] = bool(
            item["collapse_count"] == 0
            and item["clean_mean_delta"] >= config.clean_noninferiority_margin
            and item["missing_1_mean_delta"] >= config.stress_superiority_margin
            and item["missing_2_mean_delta"] >= config.stress_superiority_margin
            and item["stress_wins"] >= 4
        )
    eligible = [item for item in tests if item["passes_predeclared_margins"]]
    eligible.sort(
        key=lambda item: (
            -min(item["missing_1_mean_delta"], item["missing_2_mean_delta"]),
            -item["clean_mean_delta"],
            max(item["missing_1_holm_p"], item["missing_2_holm_p"]),
        )
    )
    return {
        "status": "promoted" if eligible else "no_promotion",
        "selected_config_id": eligible[0]["config_id"] if eligible else None,
        "baseline_config_id": baseline["config_id"],
        "clean_noninferiority_margin": config.clean_noninferiority_margin,
        "stress_superiority_margin": config.stress_superiority_margin,
        "tests": tests,
        "readout_sha256": canonical_sha256(report),
    }


def _write_final_decision(output_dir: Path, report: dict[str, Any]) -> None:
    body = build_final_decision(report)
    write_json_atomic(
        output_dir / "decisions" / "final-promotion.json",
        {**body, "sha256": canonical_sha256(body)},
    )


def summarize_wave(output_dir: Path, wave: str) -> dict[str, Any]:
    cells = _read_cells(output_dir, wave)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[str(cell["config_id"])].append(cell)
    ranking = [_summarize_config(config_id, values) for config_id, values in grouped.items()]
    ranking.sort(
        key=lambda item: (
            item["collapse_count"] > 0,
            -item["mean_accuracy"],
            -min(item["missing_1_floor"], item["missing_2_floor"]),
            -item["lower_quartile"],
        )
    )
    report = {"wave": wave, "n_cells": len(cells), "ranking": ranking}
    write_json_atomic(output_dir / "analysis" / f"{wave}-readout.json", report)
    if wave == "w2":
        _write_w2_selection(output_dir, report)
    if wave == "w3":
        _write_final_decision(output_dir, report)
    return report
