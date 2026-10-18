"""Paired multi-seed analysis and promotion decision for Exp04."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Any

import numpy as np

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import QCRCandidate
from .config import Exp04Config


def _subject_seed(payloads: list[dict[str, Any]], metric: str) -> dict[tuple[str, int], float]:
    return {
        (str(payload["subject"]), int(payload["seed"])): float(
            payload["validation_metrics"][metric]
        )
        for payload in payloads
    }


def _sign_flip_pvalue(deltas: list[float]) -> float:
    values = np.asarray(deltas, dtype=float)
    if len(values) < 1:
        return 1.0
    observed = abs(float(values.mean()))
    distribution = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(distribution) >= observed - 1e-12))


def _summary(payloads: list[dict[str, Any]], config_id: str) -> dict[str, Any]:
    cells = [payload for payload in payloads if payload["config_id"] == config_id]
    by_subject = {
        subject: float(
            np.mean(
                [
                    cell["validation_metrics"]["accuracy"]
                    for cell in cells
                    if cell["subject"] == subject
                ]
            )
        )
        for subject in sorted({str(cell["subject"]) for cell in cells})
    }
    by_seed = {
        seed: float(
            np.mean(
                [cell["validation_metrics"]["accuracy"] for cell in cells if cell["seed"] == seed]
            )
        )
        for seed in sorted({int(cell["seed"]) for cell in cells})
    }
    stress = {
        condition: float(
            np.mean([cell["stress_results"][condition]["metrics"]["accuracy"] for cell in cells])
        )
        for condition in ("missing_1", "missing_2")
    }
    stress_worst = {
        condition: float(
            min(cell["stress_results"][condition]["worst_mask_accuracy"] for cell in cells)
        )
        for condition in ("missing_1", "missing_2")
    }
    return {
        "config_id": config_id,
        "mean_accuracy": float(np.mean(list(by_subject.values()))),
        "lower_quartile": float(np.quantile(list(by_subject.values()), 0.25)),
        "mean_kappa": float(np.mean([cell["validation_metrics"]["kappa"] for cell in cells])),
        "seed_std": float(np.std(list(by_seed.values()), ddof=0)),
        "collapse_count": sum(bool(cell["quality_flags"]) for cell in cells),
        "subject_accuracy": by_subject,
        "stress_accuracy": stress,
        "stress_worst_mask_accuracy": stress_worst,
        "stress_floor": min(stress_worst.values()),
        "parameters": cells[0]["parameters"],
        "mean_inference_seconds_per_trial": float(
            np.mean(
                [
                    report["clean_inference_seconds_per_trial"]
                    for cell in cells
                    for report in cell["fold_reports"]
                ]
            )
        ),
    }


def analyze_results(
    payloads: list[dict[str, Any]],
    catalog: tuple[QCRCandidate, ...],
    config: Exp04Config,
) -> dict[str, Any]:
    """Evaluate clean superiority and low-cost robustness as separate routes."""

    baseline_cells = [payload for payload in payloads if payload["model_id"] == "FBCNet-V10"]
    baseline = _summary(
        [{**cell, "config_id": "FBCNet-V10"} for cell in baseline_cells], "FBCNet-V10"
    )
    csp_cells = [payload for payload in payloads if payload["model_id"] == "CSP+LDA"]
    csp_baseline = (
        _summary([{**cell, "config_id": "CSP+LDA"} for cell in csp_cells], "CSP+LDA")
        if csp_cells
        else None
    )
    baseline_map = _subject_seed(baseline_cells, "accuracy")
    baseline_stress = {
        condition: {
            (str(cell["subject"]), int(cell["seed"])): float(
                cell["stress_results"][condition]["worst_mask_accuracy"]
            )
            for cell in baseline_cells
        }
        for condition in ("missing_1", "missing_2")
    }
    summaries: list[dict[str, Any]] = []
    for candidate in catalog:
        cells = [payload for payload in payloads if payload["config_id"] == candidate.config_id]
        summary = _summary(cells, candidate.config_id)
        candidate_map = _subject_seed(cells, "accuracy")
        pairs = sorted(set(candidate_map) & set(baseline_map))
        subject_deltas = {
            subject: float(
                np.mean(
                    [
                        candidate_map[pair] - baseline_map[pair]
                        for pair in pairs
                        if pair[0] == subject
                    ]
                )
            )
            for subject in sorted({pair[0] for pair in pairs})
        }
        stress_deltas = {}
        for condition in ("missing_1", "missing_2"):
            values = {
                (str(cell["subject"]), int(cell["seed"])): float(
                    cell["stress_results"][condition]["worst_mask_accuracy"]
                )
                for cell in cells
            }
            stress_deltas[condition] = float(
                np.mean([values[pair] - baseline_stress[condition][pair] for pair in pairs])
            )
        mean_delta = float(np.mean(list(subject_deltas.values())))
        mean_delta_vs_csp = (
            float(
                np.mean(
                    [
                        summary["subject_accuracy"][subject]
                        - csp_baseline["subject_accuracy"][subject]
                        for subject in summary["subject_accuracy"]
                    ]
                )
            )
            if csp_baseline is not None
            else None
        )
        clean_harm_rate = float(np.mean([value < -0.01 for value in subject_deltas.values()]))
        primary_success = bool(
            summary["collapse_count"] == 0
            and mean_delta >= config.clean_superiority_margin
            and sum(value > 0 for value in subject_deltas.values()) >= 4
            and summary["lower_quartile"] >= baseline["lower_quartile"] - 0.01
            and summary["mean_kappa"] >= baseline["mean_kappa"] - 0.02
        )
        robust_success = bool(
            summary["collapse_count"] == 0
            and mean_delta >= config.clean_noninferiority_margin
            and min(stress_deltas.values()) >= config.stress_superiority_margin
        )
        summaries.append(
            {
                **summary,
                "candidate": asdict(candidate),
                "mean_delta_vs_fbcnet": mean_delta,
                "mean_delta_vs_csp": mean_delta_vs_csp,
                "subject_delta_vs_fbcnet": subject_deltas,
                "sign_flip_p": _sign_flip_pvalue(list(subject_deltas.values())),
                "stress_delta_vs_fbcnet": stress_deltas,
                "clean_harm_rate": clean_harm_rate,
                "primary_success": primary_success,
                "robust_success": robust_success,
            }
        )
    ranking = sorted(
        summaries,
        key=lambda item: (
            not (item["primary_success"] or item["robust_success"]),
            -round(item["mean_accuracy"] / config.practical_tie),
            -item["lower_quartile"],
            -item["mean_kappa"],
            -item["stress_floor"],
            item["parameters"]["active"],
            item["mean_inference_seconds_per_trial"],
        ),
    )
    full = next(item for item in summaries if item["candidate"]["role"] == "qcr_full")
    status = "promoted" if full["primary_success"] or full["robust_success"] else "rejected"
    decision_body = {
        "status": status,
        "selected_config_id": full["config_id"] if status == "promoted" else None,
        "selected_config_sha256": (full["candidate"]["sha256"] if status == "promoted" else None),
        "primary_success": full["primary_success"],
        "robust_success": full["robust_success"],
        "mean_delta_vs_fbcnet": full["mean_delta_vs_fbcnet"],
        "stress_delta_vs_fbcnet": full["stress_delta_vs_fbcnet"],
    }
    return {
        "status": status,
        "baseline": baseline,
        "csp_baseline": csp_baseline,
        "ranking": ranking,
        "qcr_full": full,
        "promotion_decision": {
            **decision_body,
            "sha256": canonical_sha256(decision_body),
        },
    }
