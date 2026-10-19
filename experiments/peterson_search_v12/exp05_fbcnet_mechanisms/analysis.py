"""Discovery-only readouts and signed decisions for Peterson Exp05."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.io import write_json_atomic

from .config import BASELINE_CONFIG_ID, Exp05Config


def _read_cells(output_dir: Path, phase: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / f"{phase}_cells").glob("*.json"))
    ]


def _manifest_sha256(output_dir: Path, phase: str) -> str:
    path = output_dir / "manifests" / f"{phase}-manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing Exp05 {phase} manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError(f"Exp05 {phase} manifest signature is invalid")
    return signature


def _metric(cell: dict[str, Any], name: str) -> float:
    metrics = cell.get("aggregate_metrics")
    if not isinstance(metrics, dict) or name not in metrics:
        raise ValueError(f"Cell {cell.get('cell_id')} is missing aggregate metric {name}")
    return float(metrics[name])


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty metric list")
    return float(np.mean(np.asarray(values, dtype=float)))


def _exact_sign_flip_pvalue(deltas: list[float]) -> float:
    """Two-sided exact sign-flip p-value, reported as descriptive in discovery."""

    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        raise ValueError("Paired test requires at least one subject")
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        for signs in product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def _config_summary(config_id: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    subjects = sorted({str(cell["subject"]) for cell in cells})
    seeds = sorted({int(cell["seed"]) for cell in cells})
    subject_metrics: dict[str, dict[str, float]] = {}
    for subject in subjects:
        subject_cells = [cell for cell in cells if cell["subject"] == subject]
        subject_metrics[subject] = {
            "accuracy": _mean([_metric(cell, "accuracy") for cell in subject_cells]),
            "kappa": _mean([_metric(cell, "kappa") for cell in subject_cells]),
        }
    seed_means = [
        _mean([_metric(cell, "accuracy") for cell in cells if int(cell["seed"]) == seed])
        for seed in seeds
    ]
    accuracies = [item["accuracy"] for item in subject_metrics.values()]
    quality_flags = [flag for cell in cells for flag in cell.get("quality_flags", [])]
    fold_flags = [
        flag
        for cell in cells
        for fold in cell.get("fold_reports", [])
        for flag in fold.get("quality_flags", [])
    ]
    return {
        "config_id": config_id,
        "mechanism": str(cells[0].get("mechanism", "frozen_fbcnet")),
        "family": str(
            cells[0].get("candidate", {}).get(
                "family", "baseline" if config_id == BASELINE_CONFIG_ID else "unknown"
            )
        ),
        "kind": str(cells[0].get("kind", "candidate")),
        "n_cells": len(cells),
        "n_subjects": len(subjects),
        "n_seeds": len(seeds),
        "mean_accuracy": _mean(accuracies),
        "mean_kappa": _mean([item["kappa"] for item in subject_metrics.values()]),
        "lower_quartile": float(np.quantile(np.asarray(accuracies), 0.25)),
        "seed_std": float(np.std(np.asarray(seed_means), ddof=0)),
        "collapse_count": sum("class_collapse" in str(flag) for flag in quality_flags + fold_flags),
        "quality_flag_count": len(quality_flags) + len(fold_flags),
        "subject_metrics": subject_metrics,
    }


def _comparison(
    item: dict[str, Any], baseline: dict[str, Any], config: Exp05Config, *, final: bool
) -> dict[str, Any]:
    common = sorted(set(item["subject_metrics"]) & set(baseline["subject_metrics"]))
    deltas = [
        item["subject_metrics"][subject]["accuracy"]
        - baseline["subject_metrics"][subject]["accuracy"]
        for subject in common
    ]
    kappa_delta = item["mean_kappa"] - baseline["mean_kappa"]
    q1_delta = item["lower_quartile"] - baseline["lower_quartile"]
    seed_std_increase = item["seed_std"] - baseline["seed_std"]
    accuracy_delta = _mean(deltas)
    required_delta = (
        config.final_preferred_accuracy_delta if final else config.screen_min_accuracy_delta
    )
    eligible = bool(
        item["collapse_count"] == 0
        and accuracy_delta >= required_delta
        and sum(delta > 0 for delta in deltas) >= config.min_subject_wins
        and kappa_delta >= -config.max_kappa_drop
        and q1_delta >= -config.max_lower_quartile_drop
        and seed_std_increase <= config.max_seed_std_increase
    )
    return {
        **item,
        "accuracy_delta": accuracy_delta,
        "kappa_delta": kappa_delta,
        "lower_quartile_delta": q1_delta,
        "seed_std_increase": seed_std_increase,
        "subject_wins": sum(delta > 0 for delta in deltas),
        "subject_ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "descriptive_exact_sign_flip_p": _exact_sign_flip_pvalue(deltas),
        "inferential_claim": False,
        "required_accuracy_delta": required_delta,
        "eligible": eligible,
    }


def _rank(cells: list[dict[str, Any]], config: Exp05Config, *, final: bool) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[str(cell["config_id"])].append(cell)
    summaries = [_config_summary(config_id, values) for config_id, values in grouped.items()]
    baseline = next(
        (item for item in summaries if item["config_id"] == BASELINE_CONFIG_ID), None
    )
    if baseline is None:
        raise RuntimeError("Exp05 readout is missing the paired FBC control")
    comparisons = [
        _comparison(item, baseline, config, final=final)
        for item in summaries
        if item["config_id"] != BASELINE_CONFIG_ID
    ]
    comparisons.sort(
        key=lambda item: (
            not item["eligible"],
            -item["accuracy_delta"],
            -item["subject_wins"],
            -item["lower_quartile_delta"],
            item["seed_std_increase"],
        )
    )
    return {"baseline": baseline, "ranking": comparisons}


def select_screen_config_ids(
    readout: dict[str, Any], *, max_promoted: int
) -> list[str]:
    """Select the strongest eligible candidate from distinct families first."""

    eligible = [item for item in readout.get("ranking", []) if item.get("eligible") is True]
    selected: list[str] = []
    represented: set[str] = set()
    for item in eligible:
        family = str(item.get("family", "unknown"))
        if family in represented:
            continue
        selected.append(str(item["config_id"]))
        represented.add(family)
        if len(selected) == max_promoted:
            return selected
    for item in eligible:
        config_id = str(item["config_id"])
        if config_id not in selected:
            selected.append(config_id)
        if len(selected) == max_promoted:
            break
    return selected


def analyze_screen(output_dir: Path, config: Exp05Config) -> dict[str, Any]:
    """Rank the 20 mechanisms and sign a promotion of at most four."""

    cells = _read_cells(output_dir, "screen")
    ranked = _rank(cells, config, final=False)
    selected = select_screen_config_ids(ranked, max_promoted=config.max_promoted)
    report = {
        "experiment": "exp05",
        "phase": "screen",
        "n_cells": len(cells),
        "statistical_scope": "discovery_descriptive_only",
        **ranked,
    }
    write_json_atomic(output_dir / "analysis" / "screen-readout.json", report)
    body = {
        "status": "selected" if selected else "no_promotion",
        "phase": "screen",
        "selected_config_ids": selected,
        "max_promoted": config.max_promoted,
        "selection_rule": (
            "zero-flags_delta>=0.5pp_wins>=4of6_kappa_q1_seed-stability_then_effect"
        ),
        "confirmatory_p_value_used": False,
        "manifest_sha256": _manifest_sha256(output_dir, "screen"),
        "readout_sha256": canonical_sha256(report),
    }
    decision = {**body, "sha256": canonical_sha256(body)}
    write_json_atomic(output_dir / "decisions" / "screen-promotion.json", decision)
    return report


def load_screen_decision(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "decisions" / "screen-promotion.json"
    if not path.is_file():
        raise ValueError(f"Missing signed Exp05 screen decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError("Exp05 screen decision signature is invalid")
    selected = payload.get("selected_config_ids")
    if not isinstance(selected, list) or len(selected) > Exp05Config().max_promoted:
        raise ValueError("Exp05 screen decision exceeds the promotion budget")
    readout_path = output_dir / "analysis" / "screen-readout.json"
    if not readout_path.is_file():
        raise ValueError(f"Missing Exp05 screen readout: {readout_path}")
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    if payload.get("readout_sha256") != canonical_sha256(readout):
        raise ValueError("Exp05 screen decision is detached from its readout")
    expected = select_screen_config_ids(
        readout, max_promoted=int(payload["max_promoted"])
    )
    if selected != expected:
        raise ValueError("Exp05 screen decision differs from its frozen ranking")
    if payload.get("manifest_sha256") != _manifest_sha256(output_dir, "screen"):
        raise ValueError("Exp05 screen decision is detached from its manifest")
    return payload


def analyze_confirm(output_dir: Path, config: Exp05Config) -> dict[str, Any]:
    """Combine screen and confirmation seeds without opening the holdout."""

    decision = load_screen_decision(output_dir)
    selected = set(map(str, decision["selected_config_ids"]))
    cells = _read_cells(output_dir, "screen") + _read_cells(output_dir, "confirm")
    allowed = selected | {BASELINE_CONFIG_ID}
    cells = [cell for cell in cells if str(cell["config_id"]) in allowed]
    ranked = _rank(cells, config, final=True)
    eligible = [item for item in ranked["ranking"] if item["eligible"]]
    selected_config_id = eligible[0]["config_id"] if eligible else None
    report = {
        "experiment": "exp05",
        "phase": "confirm",
        "n_cells": len(cells),
        "seeds": list(config.screen_seeds + config.confirm_seeds),
        "statistical_scope": "internal_confirmation_holdout_still_closed",
        **ranked,
    }
    write_json_atomic(output_dir / "analysis" / "confirm-readout.json", report)
    body = {
        "status": "promoted" if selected_config_id is not None else "no_promotion",
        "phase": "confirm",
        "selected_config_id": selected_config_id,
        "preferred_accuracy_delta": config.final_preferred_accuracy_delta,
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
