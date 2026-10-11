"""Predeclared exp02 ranking, paired contrasts and promotion decision."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .catalog import Exp02Candidate, InferenceRecipe
from .config import Exp02Config


def _exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    signs = np.asarray(
        [
            [1.0 if (mask >> index) & 1 else -1.0 for index in range(len(differences))]
            for mask in range(2 ** len(differences))
        ]
    )
    permuted = np.abs((signs * differences).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-12))


def _bootstrap_mean_ci(differences: np.ndarray, *, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(10_000, len(differences)))
    means = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _subject_values(
    baseline_cells: list[dict[str, Any]], model_id: str, metric: str
) -> dict[str, float]:
    selected = [cell for cell in baseline_cells if cell["model_id"] == model_id]
    values = {str(cell["subject"]): float(cell["validation_metrics"][metric]) for cell in selected}
    if len(values) != 6:
        raise ValueError(f"Baseline {model_id} is incomplete")
    return values


def _promote(ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    topology_counts: defaultdict[str, int] = defaultdict(int)
    objective_counts: defaultdict[str, int] = defaultdict(int)
    for item in ranking:
        if not item["eligible"] or not item["passes_success_gate"]:
            continue
        if topology_counts[item["topology_id"]] >= 2:
            continue
        if objective_counts[item["objective_id"]] >= 2:
            continue
        promoted.append(item)
        topology_counts[item["topology_id"]] += 1
        objective_counts[item["objective_id"]] += 1
        if len(promoted) == 8:
            break
    return promoted


def analyze_payloads(
    candidate_cells: list[dict[str, Any]],
    baseline_cells: list[dict[str, Any]],
    *,
    candidates: tuple[Exp02Candidate, ...],
    inference_catalog: tuple[InferenceRecipe, ...],
    config: Exp02Config,
) -> dict[str, Any]:
    _ = inference_catalog
    fbc_accuracy = _subject_values(baseline_cells, "FBCNet", "accuracy")
    csp_accuracy = _subject_values(baseline_cells, "CSP+LDA", "accuracy")
    fbc_kappa = _subject_values(baseline_cells, "FBCNet", "kappa")
    candidate_by_id = {candidate.config_id: candidate for candidate in candidates}
    grouped: defaultdict[tuple[str, str], list[tuple[str, dict[str, Any], list[str]]]] = (
        defaultdict(list)
    )
    for cell in candidate_cells:
        for inference_id, result in cell["inference_results"].items():
            grouped[(str(cell["config_id"]), inference_id)].append(
                (str(cell["subject"]), result, list(result.get("quality_flags", [])))
            )
    by_config: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for (config_id, inference_id), rows in grouped.items():
        if len({subject for subject, _, _ in rows}) != len(config.subjects):
            continue
        accuracies = {subject: float(result["metrics"]["accuracy"]) for subject, result, _ in rows}
        kappas = {subject: float(result["metrics"]["kappa"]) for subject, result, _ in rows}
        deltas = np.asarray(
            [accuracies[subject] - fbc_accuracy[subject] for subject in config.subjects],
            dtype=float,
        )
        csp_deltas = np.asarray(
            [accuracies[subject] - csp_accuracy[subject] for subject in config.subjects],
            dtype=float,
        )
        kappa_deltas = np.asarray(
            [kappas[subject] - fbc_kappa[subject] for subject in config.subjects],
            dtype=float,
        )
        quality_flags = sorted({flag for _, _, flags in rows for flag in flags})
        candidate = candidate_by_id[config_id]
        mean_accuracy = float(np.mean(list(accuracies.values())))
        fbc_delta = float(np.mean(deltas))
        lower_quartile_delta = float(
            np.quantile(list(accuracies.values()), 0.25)
            - np.quantile(list(fbc_accuracy.values()), 0.25)
        )
        eligible = not quality_flags and float(np.mean(kappa_deltas)) >= -0.02
        success = bool(
            eligible
            and fbc_delta >= 0.01 - 1e-12
            and int(np.sum(deltas > 0)) >= 4
            and lower_quartile_delta >= -0.01 - 1e-12
        )
        by_config[config_id].append(
            {
                "config_id": config_id,
                "inference_id": inference_id,
                "topology_id": candidate.topology_id,
                "objective_id": candidate.objective_id,
                "family": candidate.family,
                "mean_accuracy": mean_accuracy,
                "mean_kappa": float(np.mean(list(kappas.values()))),
                "delta_fbcnet": fbc_delta,
                "delta_csp_lda": float(np.mean(csp_deltas)),
                "fbcnet_difference_ci95": _bootstrap_mean_ci(deltas, seed=2026),
                "csp_lda_difference_ci95": _bootstrap_mean_ci(csp_deltas, seed=2027),
                "fbcnet_exact_sign_flip_p": _exact_sign_flip_p(deltas),
                "csp_lda_exact_sign_flip_p": _exact_sign_flip_p(csp_deltas),
                "wins_vs_fbcnet": int(np.sum(deltas > 0)),
                "lower_quartile_delta_fbcnet": lower_quartile_delta,
                "mean_kappa_delta_fbcnet": float(np.mean(kappa_deltas)),
                "quality_flags": quality_flags,
                "eligible": eligible,
                "passes_success_gate": success,
                "subject_accuracy": accuracies,
            }
        )
    ranking: list[dict[str, Any]] = []
    for variants in by_config.values():
        best = max(
            variants,
            key=lambda item: (
                item["eligible"],
                item["mean_accuracy"],
                item["mean_kappa"],
            ),
        )
        ranking.append(best)
    ranking.sort(
        key=lambda item: (
            item["eligible"],
            item["passes_success_gate"],
            item["mean_accuracy"],
            item["mean_kappa"],
        ),
        reverse=True,
    )
    promoted = _promote(ranking)
    status = "advance" if promoted else "needs_redesign"
    return {
        "status": status,
        "baseline_summary": {
            "CSP+LDA": float(np.mean(list(csp_accuracy.values()))),
            "FBCNet": float(np.mean(list(fbc_accuracy.values()))),
        },
        "ranking": ranking,
        "promotion_decision": {
            "status": status,
            "eligible_count": sum(item["eligible"] for item in ranking),
            "success_count": sum(item["passes_success_gate"] for item in ranking),
            "promoted": promoted,
        },
    }
