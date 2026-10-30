"""Paired fixed-convergence analysis and signed decisions for Exp08."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.io import write_json_atomic

from .catalog import E11_REFERENCE_ID, FBC_CONTROL_ID, backbone_pairs
from .config import Exp08Config

_COLLAPSE_FLAG = "validation_class_collapse"


def collapse_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts = [
        sum(
            _COLLAPSE_FLAG in set(map(str, fold.get("quality_flags", [])))
            for fold in cell.get("fold_reports", [])
        )
        for cell in cells
    ]
    return {
        "collapsed_fold_count": int(sum(counts)),
        "affected_cell_count": int(sum(value > 0 for value in counts)),
    }


def _read_cells(output_dir: Path, phase: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / f"{phase}_cells").glob("*.json"))
    ]


def _manifest_sha256(output_dir: Path, phase: str) -> str:
    payload = json.loads(
        (output_dir / "manifests" / f"{phase}-manifest.json").read_text(encoding="utf-8")
    )
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError(f"Exp08 {phase} manifest signature is invalid")
    return signature


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize empty Exp08 values")
    return float(np.mean(np.asarray(values, dtype=float)))


def _sign_flip_pvalue(deltas: list[float]) -> float:
    values = np.asarray(deltas, dtype=float)
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def _summary(config_id: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    subjects = sorted({str(cell["subject"]) for cell in cells})
    seeds = sorted({int(cell["seed"]) for cell in cells})
    subject_metrics = {
        subject: {
            metric: _mean(
                [
                    float(cell["aggregate_metrics"][metric])
                    for cell in cells
                    if cell["subject"] == subject
                ]
            )
            for metric in ("accuracy", "kappa")
        }
        for subject in subjects
    }
    seed_means = [
        _mean(
            [
                float(cell["aggregate_metrics"]["accuracy"])
                for cell in cells
                if int(cell["seed"]) == seed
            ]
        )
        for seed in seeds
    ]
    accuracies = [values["accuracy"] for values in subject_metrics.values()]
    return {
        "config_id": config_id,
        "backbone": str(cells[0].get("candidate", {}).get("backbone", "fbcnet")),
        "aggregation": str(cells[0].get("candidate", {}).get("aggregation", "control")),
        "n_cells": len(cells),
        "n_subjects": len(subjects),
        "n_seeds": len(seeds),
        "mean_accuracy": _mean(accuracies),
        "mean_kappa": _mean([values["kappa"] for values in subject_metrics.values()]),
        "lower_quartile": float(np.quantile(np.asarray(accuracies), 0.25)),
        "seed_std": float(np.std(np.asarray(seed_means), ddof=0)),
        **collapse_counts(cells),
        "quality_flag_count": sum(len(cell.get("quality_flags", [])) for cell in cells),
        "subject_metrics": subject_metrics,
    }


def _deltas(item: dict[str, Any], reference: dict[str, Any]) -> list[float]:
    subjects = sorted(set(item["subject_metrics"]) & set(reference["subject_metrics"]))
    return [
        item["subject_metrics"][subject]["accuracy"]
        - reference["subject_metrics"][subject]["accuracy"]
        for subject in subjects
    ]


def _compare(lpwf: dict[str, Any], uniform: dict[str, Any], config: Exp08Config) -> dict[str, Any]:
    deltas = _deltas(lpwf, uniform)
    accuracy_delta = _mean(deltas)
    kappa_delta = lpwf["mean_kappa"] - uniform["mean_kappa"]
    quartile_delta = lpwf["lower_quartile"] - uniform["lower_quartile"]
    seed_std_increase = lpwf["seed_std"] - uniform["seed_std"]
    wins = sum(delta > 0 for delta in deltas)
    eligible = bool(
        lpwf["collapsed_fold_count"] == 0
        and lpwf["affected_cell_count"] == 0
        and accuracy_delta >= config.screen_min_pair_delta
        and wins >= config.min_subject_wins
        and kappa_delta >= -config.max_kappa_drop - 1e-12
        and quartile_delta >= -config.max_lower_quartile_drop
        and seed_std_increase <= config.max_seed_std_increase + 1e-12
    )
    return {
        "backbone": lpwf["backbone"],
        "uniform_config_id": uniform["config_id"],
        "lpwf_config_id": lpwf["config_id"],
        "uniform": uniform,
        "lpwf": lpwf,
        "within_pair_accuracy_delta": accuracy_delta,
        "within_pair_kappa_delta": kappa_delta,
        "within_pair_lower_quartile_delta": quartile_delta,
        "within_pair_seed_std_increase": seed_std_increase,
        "within_pair_subject_wins": wins,
        "descriptive_exact_sign_flip_p": _sign_flip_pvalue(deltas),
        "inferential_claim": False,
        "eligible": eligible,
    }


def _screen_ranking(cells: list[dict[str, Any]], config: Exp08Config) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[str(cell["config_id"])].append(cell)
    summaries = {key: _summary(key, value) for key, value in grouped.items()}
    if E11_REFERENCE_ID not in summaries or FBC_CONTROL_ID not in summaries:
        raise RuntimeError("Exp08 readout requires both frozen controls")
    ranking = [
        {**_compare(summaries[lpwf_id], summaries[uniform_id], config), "backbone": family}
        for family, (uniform_id, lpwf_id) in backbone_pairs().items()
        if uniform_id in summaries and lpwf_id in summaries
    ]
    ranking.sort(
        key=lambda item: (
            not item["eligible"],
            -item["within_pair_accuracy_delta"],
            -item["within_pair_subject_wins"],
            -item["within_pair_lower_quartile_delta"],
            item["within_pair_seed_std_increase"],
        )
    )
    return {
        "context_baseline": summaries[FBC_CONTROL_ID],
        "reference": summaries[E11_REFERENCE_ID],
        "ranking": ranking,
    }


def select_screen_backbones(readout: dict[str, Any], *, max_promoted: int) -> list[str]:
    return [
        str(item["backbone"])
        for item in readout.get("ranking", [])
        if item.get("eligible") is True
    ][:max_promoted]


def analyze_screen(output_dir: Path, config: Exp08Config) -> dict[str, Any]:
    ranked = _screen_ranking(_read_cells(output_dir, "screen"), config)
    selected = select_screen_backbones(ranked, max_promoted=config.max_promoted_backbones)
    report = {
        "experiment": "exp08",
        "phase": "screen",
        "training_policy": "cropwise_fixed300_then_frozen_fusion",
        "primary_estimand": "lpwf_minus_uniform_within_backbone",
        "statistical_scope": "discovery_descriptive_only",
        **ranked,
    }
    write_json_atomic(output_dir / "analysis" / "screen-readout.json", report)
    body = {
        "status": "selected" if selected else "no_promotion",
        "phase": "screen",
        "selected_backbones": selected,
        "max_promoted_backbones": config.max_promoted_backbones,
        "reference_config_id": E11_REFERENCE_ID,
        "confirmatory_p_value_used": False,
        "manifest_sha256": _manifest_sha256(output_dir, "screen"),
        "readout_sha256": canonical_sha256(report),
    }
    write_json_atomic(
        output_dir / "decisions" / "screen-promotion.json",
        {**body, "sha256": canonical_sha256(body)},
    )
    return report


def load_screen_decision(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "decisions" / "screen-promotion.json"
    if not path.is_file():
        raise ValueError(f"Missing signed Exp08 screen decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    readout = json.loads((output_dir / "analysis" / "screen-readout.json").read_text())
    if signature != canonical_sha256(body):
        raise ValueError("Exp08 screen decision signature is invalid")
    if payload.get("readout_sha256") != canonical_sha256(readout):
        raise ValueError("Exp08 screen decision is detached from its readout")
    expected = select_screen_backbones(
        readout, max_promoted=int(payload["max_promoted_backbones"])
    )
    if payload.get("selected_backbones") != expected:
        raise ValueError("Exp08 screen decision differs from the frozen ranking")
    if payload.get("manifest_sha256") != _manifest_sha256(output_dir, "screen"):
        raise ValueError("Exp08 screen decision is detached from its manifest")
    return payload


def _final_comparison(
    pair: dict[str, Any], e11: dict[str, Any], config: Exp08Config
) -> dict[str, Any]:
    lpwf = pair["lpwf"]
    deltas = _deltas(lpwf, e11)
    accuracy_delta = _mean(deltas)
    kappa_delta = lpwf["mean_kappa"] - e11["mean_kappa"]
    quartile_delta = lpwf["lower_quartile"] - e11["lower_quartile"]
    seed_std_increase = lpwf["seed_std"] - e11["seed_std"]
    wins = sum(delta > 0 for delta in deltas)
    eligible = bool(
        pair["eligible"]
        and accuracy_delta >= config.final_min_e11_delta
        and wins >= config.min_subject_wins
        and kappa_delta >= -config.max_kappa_drop - 1e-12
        and quartile_delta >= -config.max_lower_quartile_drop
        and seed_std_increase <= config.max_seed_std_increase + 1e-12
    )
    return {
        **pair,
        "accuracy_delta": accuracy_delta,
        "kappa_delta": kappa_delta,
        "lower_quartile_delta": quartile_delta,
        "seed_std_increase": seed_std_increase,
        "subject_wins": wins,
        "descriptive_exact_sign_flip_p_vs_e11": _sign_flip_pvalue(deltas),
        "required_accuracy_delta": config.final_min_e11_delta,
        "eligible": eligible,
    }


def analyze_confirm(output_dir: Path, config: Exp08Config) -> dict[str, Any]:
    decision = load_screen_decision(output_dir)
    selected = tuple(map(str, decision["selected_backbones"]))
    allowed = {FBC_CONTROL_ID, E11_REFERENCE_ID}
    for family in selected:
        allowed.update(backbone_pairs()[family])
    cells = _read_cells(output_dir, "screen") + _read_cells(output_dir, "confirm")
    ranked = _screen_ranking(
        [cell for cell in cells if str(cell["config_id"]) in allowed], config
    )
    ranking = [
        _final_comparison(item, ranked["reference"], config)
        for item in ranked["ranking"]
        if item["backbone"] in selected
    ]
    ranking.sort(key=lambda item: (not item["eligible"], -item["accuracy_delta"]))
    winner = next((item for item in ranking if item["eligible"]), None)
    report = {
        "experiment": "exp08",
        "phase": "confirm",
        "seeds": list(config.screen_seeds + config.confirm_seeds),
        "training_policy": "cropwise_fixed300_then_frozen_fusion",
        "primary_estimand": "lpwf_minus_uniform_within_backbone",
        "promotion_estimand": "best_lpwf_minus_e11",
        "statistical_scope": "internal_confirmation_holdout_still_closed",
        "context_baseline": ranked["context_baseline"],
        "reference": ranked["reference"],
        "ranking": ranking,
    }
    write_json_atomic(output_dir / "analysis" / "confirm-readout.json", report)
    body = {
        "status": "promoted" if winner else "retain_e11",
        "phase": "confirm",
        "selected_config_id": winner["lpwf_config_id"] if winner else E11_REFERENCE_ID,
        "reference_config_id": E11_REFERENCE_ID,
        "holdout_opened": False,
        "confirmatory_p_value_used": False,
        "screen_decision_sha256": str(decision["sha256"]),
        "manifest_sha256": _manifest_sha256(output_dir, "confirm"),
        "readout_sha256": canonical_sha256(report),
    }
    write_json_atomic(
        output_dir / "decisions" / "final-promotion.json",
        {**body, "sha256": canonical_sha256(body)},
    )
    return report


def load_final_decision(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "decisions" / "final-promotion.json"
    if not path.is_file():
        raise ValueError(f"Missing signed Exp08 final decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body) or payload.get("phase") != "confirm":
        raise ValueError("Exp08 final decision signature is invalid")
    if payload.get("holdout_opened") is not False:
        raise ValueError("Exp08 final decision opened the holdout")
    readout = json.loads((output_dir / "analysis" / "confirm-readout.json").read_text())
    if payload.get("readout_sha256") != canonical_sha256(readout):
        raise ValueError("Exp08 final decision is detached from its readout")
    screen = load_screen_decision(output_dir)
    if payload.get("screen_decision_sha256") != screen.get("sha256"):
        raise ValueError("Exp08 final decision is detached from screening")
    if payload.get("manifest_sha256") != _manifest_sha256(output_dir, "confirm"):
        raise ValueError("Exp08 final decision is detached from its manifest")
    winner = next((item for item in readout.get("ranking", []) if item.get("eligible")), None)
    expected = winner["lpwf_config_id"] if winner else E11_REFERENCE_ID
    status = "promoted" if winner else "retain_e11"
    if payload.get("selected_config_id") != expected or payload.get("status") != status:
        raise ValueError("Exp08 final decision differs from the frozen ranking")
    return payload

