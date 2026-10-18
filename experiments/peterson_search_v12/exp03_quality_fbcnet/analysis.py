"""Predeclared recipe selection and quality-candidate promotion for exp03."""

from __future__ import annotations

import itertools
from dataclasses import asdict
from typing import Any

import numpy as np

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import QualityCandidate, RecipeSpec


def select_recipe(
    summaries: list[dict[str, Any]], *, practical_tie: float = 0.005
) -> dict[str, Any]:
    eligible = [item for item in summaries if int(item["collapse_count"]) == 0]
    if not eligible:
        raise ValueError("No collapse-free recipe is eligible")
    best_accuracy = max(float(item["mean_accuracy"]) for item in eligible)
    tied = [
        item
        for item in eligible
        if best_accuracy - float(item["mean_accuracy"]) <= practical_tie + 1e-12
    ]
    return max(
        tied,
        key=lambda item: (
            float(item["lower_quartile"]),
            float(item["mean_kappa"]),
            -float(item["seed_std"]),
            item["recipe_id"] == "R0-last-wd0",
        ),
    )


def _sign_flip_pvalue(differences: list[float]) -> float:
    values = np.asarray(differences, dtype=float)
    if len(values) == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    permutations = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(permutations) >= observed - 1e-12))


def _hierarchical_ci(
    rows: list[dict[str, Any]], *, value_key: str, seed: int = 2026, draws: int = 5000
) -> tuple[float, float]:
    by_subject: dict[str, list[float]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject"]), []).append(float(row[value_key]))
    subjects = sorted(by_subject)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(draws):
        selected_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        subject_means = [
            float(rng.choice(by_subject[str(subject)], size=len(by_subject[str(subject)])).mean())
            for subject in selected_subjects
        ]
        samples.append(float(np.mean(subject_means)))
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return float(lower), float(upper)


def _summary(payloads: list[dict[str, Any]], config_id: str) -> dict[str, Any]:
    selected = [payload for payload in payloads if payload["config_id"] == config_id]
    accuracies = [float(payload["validation_metrics"]["accuracy"]) for payload in selected]
    kappas = [float(payload["validation_metrics"]["kappa"]) for payload in selected]
    subject_accuracy = {
        subject: float(
            np.mean(
                [
                    payload["validation_metrics"]["accuracy"]
                    for payload in selected
                    if payload["subject"] == subject
                ]
            )
        )
        for subject in sorted({payload["subject"] for payload in selected})
    }
    seed_accuracy = {
        seed: float(
            np.mean(
                [
                    payload["validation_metrics"]["accuracy"]
                    for payload in selected
                    if payload["seed"] == seed
                ]
            )
        )
        for seed in sorted({payload["seed"] for payload in selected})
    }
    return {
        "recipe_id": config_id,
        "mean_accuracy": float(np.mean(accuracies)),
        "lower_quartile": float(np.quantile(list(subject_accuracy.values()), 0.25)),
        "mean_kappa": float(np.mean(kappas)),
        "seed_std": float(np.std(list(seed_accuracy.values()))),
        "collapse_count": sum(bool(payload["quality_flags"]) for payload in selected),
        "subject_accuracy": subject_accuracy,
        "seed_accuracy": seed_accuracy,
        "n_cells": len(selected),
    }


def _parity_report(
    manual: dict[tuple[str, int], float],
    reference: dict[tuple[str, int], float],
    *,
    mean_tolerance: float = 0.02,
    mean_absolute_tolerance: float = 0.03,
    max_subject_tolerance: float = 0.05,
) -> dict[str, Any]:
    shared = sorted(set(manual) & set(reference))
    deltas = {key: float(manual[key] - reference[key]) for key in shared}
    subject_delta = {
        subject: float(np.mean([delta for (name, _), delta in deltas.items() if name == subject]))
        for subject in sorted({subject for subject, _ in shared})
    }
    mean_delta = float(np.mean(list(deltas.values()))) if deltas else float("nan")
    mean_absolute_delta = (
        float(np.mean(np.abs(list(deltas.values())))) if deltas else float("nan")
    )
    max_absolute_subject_delta = (
        float(np.max(np.abs(list(subject_delta.values())))) if subject_delta else float("nan")
    )
    passed = bool(
        deltas
        and abs(mean_delta) <= mean_tolerance
        and mean_absolute_delta <= mean_absolute_tolerance
        and max_absolute_subject_delta <= max_subject_tolerance
    )
    return {
        "mean_accuracy_delta_manual_minus_v10": mean_delta,
        "mean_absolute_paired_delta": mean_absolute_delta,
        "max_absolute_subject_delta": max_absolute_subject_delta,
        "subject_delta": subject_delta,
        "mean_tolerance": mean_tolerance,
        "mean_absolute_tolerance": mean_absolute_tolerance,
        "max_subject_tolerance": max_subject_tolerance,
        "pass": passed,
        "paired_cells": len(shared),
    }


def _contrast_report(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"contrast": name, "status": "insufficient_pairs", "paired_rows": 0}
    subject_delta = {
        subject: float(np.mean([row["delta"] for row in rows if row["subject"] == subject]))
        for subject in sorted({row["subject"] for row in rows})
    }
    return {
        "contrast": name,
        "status": "descriptive_discovery",
        "mean_delta": float(np.mean([row["delta"] for row in rows])),
        "hierarchical_ci95": list(_hierarchical_ci(rows, value_key="delta")),
        "sign_flip_p": _sign_flip_pvalue(list(subject_delta.values())),
        "subject_delta": subject_delta,
        "paired_rows": len(rows),
    }


def _quality_factorial_contrasts(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def matched_rows(
        factor: str, left: Any, right: Any, matching: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        left_items = {
            tuple(item[key] for key in matching): item
            for item in summaries
            if item[factor] == left
        }
        right_items = {
            tuple(item[key] for key in matching): item
            for item in summaries
            if item[factor] == right
        }
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left_items) & set(right_items)):
            left_accuracy = left_items[key]["subject_accuracy"]
            right_accuracy = right_items[key]["subject_accuracy"]
            for subject in sorted(set(left_accuracy) & set(right_accuracy)):
                rows.append(
                    {
                        "subject": subject,
                        "delta": float(left_accuracy[subject] - right_accuracy[subject]),
                    }
                )
        return rows

    contrasts = [
        _contrast_report(
            "gate_G1_minus_G0",
            matched_rows(
                "gate_id",
                "G1",
                "G0",
                ("channel_dropout", "n_filters_spat", "stride_factor"),
            ),
        ),
        _contrast_report(
            "gate_G2_minus_G1",
            matched_rows(
                "gate_id",
                "G2",
                "G1",
                ("channel_dropout", "n_filters_spat", "stride_factor"),
            ),
        ),
    ]
    for left, right in ((0.1, 0.0), (0.2, 0.0), (0.2, 0.1)):
        contrasts.append(
            _contrast_report(
                f"channel_dropout_{left}_minus_{right}",
                matched_rows(
                    "channel_dropout",
                    left,
                    right,
                    ("gate_id", "n_filters_spat", "stride_factor"),
                ),
            )
        )
    indexed = {
        (
            item["gate_id"],
            item["channel_dropout"],
            item["n_filters_spat"],
            item["stride_factor"],
        ): item
        for item in summaries
    }
    for gate in ("G1", "G2"):
        for dropout in (0.1, 0.2):
            interaction_rows: list[dict[str, Any]] = []
            for n_filters_spat in (16, 32, 48):
                for stride_factor in (2, 4, 8):
                    keys = (
                        (gate, dropout, n_filters_spat, stride_factor),
                        (gate, 0.0, n_filters_spat, stride_factor),
                        ("G0", dropout, n_filters_spat, stride_factor),
                        ("G0", 0.0, n_filters_spat, stride_factor),
                    )
                    if not all(key in indexed for key in keys):
                        continue
                    items = [indexed[key]["subject_accuracy"] for key in keys]
                    for subject in sorted(set.intersection(*(set(item) for item in items))):
                        interaction_rows.append(
                            {
                                "subject": subject,
                                "delta": float(
                                    (items[0][subject] - items[1][subject])
                                    - (items[2][subject] - items[3][subject])
                                ),
                            }
                        )
            contrasts.append(
                _contrast_report(
                    f"interaction_{gate}_x_dropout_{dropout}", interaction_rows
                )
            )
    inferential = [item for item in contrasts if "sign_flip_p" in item]
    ordered = sorted(inferential, key=lambda item: float(item["sign_flip_p"]))
    running = 0.0
    total = len(ordered)
    for rank, item in enumerate(ordered):
        running = max(running, (total - rank) * float(item["sign_flip_p"]))
        item["holm_adjusted_p"] = min(1.0, running)
    return contrasts


def _rank_quality_summaries(
    summaries: list[dict[str, Any]], *, practical_tie: float = 0.005
) -> list[dict[str, Any]]:
    remaining = list(summaries)
    ranking: list[dict[str, Any]] = []
    while remaining:
        eligibility = max(bool(item["eligible"]) for item in remaining)
        eligible_group = [item for item in remaining if bool(item["eligible"]) == eligibility]
        best_accuracy = max(float(item["mean_accuracy"]) for item in eligible_group)
        tied = [
            item
            for item in eligible_group
            if best_accuracy - float(item["mean_accuracy"]) <= practical_tie + 1e-12
        ]
        selected = sorted(
            tied,
            key=lambda item: (
                -float(item["lower_quartile"]),
                -float(item["mean_kappa"]),
                -float(item["stress_floor"]),
                int(item["parameters"].get("active", 0)),
                float(item["mean_inference_seconds_per_trial"]),
                str(item["config_id"]),
            ),
        )[0]
        ranking.append(selected)
        remaining.remove(selected)
    return ranking


def analyze_recipe_payloads(
    payloads: list[dict[str, Any]], recipes: tuple[RecipeSpec, ...]
) -> dict[str, Any]:
    recipe_cells = [payload for payload in payloads if payload["kind"] == "recipe"]
    baseline_cells = [payload for payload in payloads if payload["kind"] == "baseline"]
    summaries = [_summary(recipe_cells, recipe.recipe_id) for recipe in recipes]
    selected_summary = select_recipe(summaries)
    selected_recipe = next(
        recipe for recipe in recipes if recipe.recipe_id == selected_summary["recipe_id"]
    )
    baseline_summary: dict[str, dict[str, float]] = {}
    for model_id in ("CSP+LDA", "FBCNet-V10"):
        cells = [cell for cell in baseline_cells if cell["model_id"] == model_id]
        baseline_summary[model_id] = {
            "mean_accuracy": float(
                np.mean([cell["validation_metrics"]["accuracy"] for cell in cells])
            ),
            "mean_kappa": float(np.mean([cell["validation_metrics"]["kappa"] for cell in cells])),
        }
    manual = {
        (cell["subject"], cell["seed"]): float(cell["validation_metrics"]["accuracy"])
        for cell in recipe_cells
        if cell["config_id"] == "R0-last-wd0"
    }
    v10 = {
        (cell["subject"], cell["seed"]): float(cell["validation_metrics"]["accuracy"])
        for cell in baseline_cells
        if cell["model_id"] == "FBCNet-V10"
    }
    parity = _parity_report(manual, v10)
    contrasts: list[dict[str, Any]] = []
    for left, right, name in (
        ("R1-best-wd0", "R0-last-wd0", "checkpoint_at_wd0"),
        ("R3-best-wd1e-4", "R2-last-wd1e-4", "checkpoint_at_wd1e-4"),
        ("R2-last-wd1e-4", "R0-last-wd0", "weight_decay_at_last"),
        ("R3-best-wd1e-4", "R1-best-wd0", "weight_decay_at_best"),
    ):
        left_cells = {
            (cell["subject"], cell["seed"]): cell
            for cell in recipe_cells
            if cell["config_id"] == left
        }
        right_cells = {
            (cell["subject"], cell["seed"]): cell
            for cell in recipe_cells
            if cell["config_id"] == right
        }
        pairs = sorted(set(left_cells) & set(right_cells))
        rows = [
            {
                "subject": subject,
                "seed": seed,
                "delta": float(
                    left_cells[(subject, seed)]["validation_metrics"]["accuracy"]
                    - right_cells[(subject, seed)]["validation_metrics"]["accuracy"]
                ),
            }
            for subject, seed in pairs
        ]
        subject_delta = {
            subject: float(np.mean([row["delta"] for row in rows if row["subject"] == subject]))
            for subject in sorted({row["subject"] for row in rows})
        }
        contrasts.append(
            {
                "contrast": name,
                "left": left,
                "right": right,
                "mean_delta": float(np.mean([row["delta"] for row in rows])),
                "hierarchical_ci95": list(_hierarchical_ci(rows, value_key="delta")),
                "sign_flip_p": _sign_flip_pvalue(list(subject_delta.values())),
                "subject_delta": subject_delta,
            }
        )
    decision_body = {
        "status": "selected" if parity["pass"] else "blocked_parity",
        "selected_recipe": asdict(selected_recipe),
        "selection_summary": selected_summary,
        "parity": parity,
    }
    decision = {**decision_body, "sha256": canonical_sha256(decision_body)}
    return {
        "status": decision["status"],
        "recipe_summaries": summaries,
        "baseline_summary": baseline_summary,
        "contrasts": contrasts,
        "recipe_decision": decision,
    }


def analyze_quality_payloads(
    payloads: list[dict[str, Any]], candidates: tuple[QualityCandidate, ...]
) -> dict[str, Any]:
    candidate_by_id = {candidate.config_id: candidate for candidate in candidates}
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        cells = [cell for cell in payloads if cell["config_id"] == candidate.config_id]
        subject_accuracy = {
            cell["subject"]: float(cell["validation_metrics"]["accuracy"]) for cell in cells
        }
        summaries.append(
            {
                "config_id": candidate.config_id,
                "gate_id": candidate.gate_id,
                "channel_dropout": candidate.channel_dropout,
                "n_filters_spat": candidate.n_filters_spat,
                "stride_factor": candidate.stride_factor,
                "mean_accuracy": float(np.mean(list(subject_accuracy.values()))),
                "lower_quartile": float(np.quantile(list(subject_accuracy.values()), 0.25)),
                "mean_kappa": float(
                    np.mean([cell["validation_metrics"]["kappa"] for cell in cells])
                ),
                "collapse_count": sum(bool(cell["quality_flags"]) for cell in cells),
                "subject_accuracy": subject_accuracy,
                "mean_stress_accuracy": {
                    key: float(
                        np.mean(
                            [cell["stress_results"][key]["metrics"]["accuracy"] for cell in cells]
                        )
                    )
                    for key in ("missing_1", "missing_2")
                },
                "mean_inference_seconds_per_trial": float(
                    np.mean(
                        [
                            report["clean_inference_seconds_per_trial"]
                            for cell in cells
                            for report in cell["fold_reports"]
                        ]
                    )
                ),
                "parameters": cells[0]["parameters"] if cells else {},
            }
        )
    control = next(
        item
        for item in summaries
        if item["gate_id"] == "G0"
        and item["channel_dropout"] == 0.0
        and item["n_filters_spat"] == 32
        and item["stride_factor"] == 4
    )
    for item in summaries:
        deltas = {
            subject: item["subject_accuracy"][subject] - control["subject_accuracy"][subject]
            for subject in control["subject_accuracy"]
        }
        item["delta_vs_control"] = float(np.mean(list(deltas.values())))
        item["wins_vs_control"] = sum(value > 0 for value in deltas.values())
        item["eligible"] = bool(
            item["collapse_count"] == 0
            and item["delta_vs_control"] >= 0.01
            and item["wins_vs_control"] >= 4
            and item["lower_quartile"] >= control["lower_quartile"] - 0.01
            and item["mean_kappa"] >= control["mean_kappa"] - 0.02
        )
        item["stress_floor"] = min(item["mean_stress_accuracy"].values())
    ranking = _rank_quality_summaries(summaries)
    promoted: list[dict[str, Any]] = []
    gate_counts: dict[str, int] = {}
    dropout_counts: dict[float, int] = {}
    for item in ranking:
        if not item["eligible"] or len(promoted) >= 8:
            continue
        gate = str(item["gate_id"])
        dropout = float(item["channel_dropout"])
        if gate_counts.get(gate, 0) >= 2 or dropout_counts.get(dropout, 0) >= 2:
            continue
        promoted.append(item)
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        dropout_counts[dropout] = dropout_counts.get(dropout, 0) + 1
    decision_body = {
        "status": "promoted" if promoted else "needs_redesign",
        "control_config_id": control["config_id"],
        "promoted": [candidate_by_id[item["config_id"]].sha256 for item in promoted],
        "promoted_config_ids": [item["config_id"] for item in promoted],
    }
    decision = {**decision_body, "sha256": canonical_sha256(decision_body)}
    return {
        "status": decision["status"],
        "control": control,
        "ranking": ranking,
        "factorial_contrasts": _quality_factorial_contrasts(summaries),
        "promotion_decision": decision,
    }
