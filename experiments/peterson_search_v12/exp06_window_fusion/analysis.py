"""Discovery-only ranking and signed decisions for Exp06."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.io import write_json_atomic

from .catalog import E11_REFERENCE_ID, FBC_CONTROL_ID
from .config import Exp06Config


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
        raise ValueError(f"Exp06 {phase} manifest signature is invalid")
    return signature


def _metric(cell: dict[str, Any], name: str) -> float:
    return float(cell["aggregate_metrics"][name])


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize empty Exp06 values")
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
            "accuracy": _mean(
                [_metric(cell, "accuracy") for cell in cells if cell["subject"] == subject]
            ),
            "kappa": _mean(
                [_metric(cell, "kappa") for cell in cells if cell["subject"] == subject]
            ),
        }
        for subject in subjects
    }
    seed_means = [
        _mean([_metric(cell, "accuracy") for cell in cells if int(cell["seed"]) == seed])
        for seed in seeds
    ]
    accuracies = [value["accuracy"] for value in subject_metrics.values()]
    all_flags = [
        flag
        for cell in cells
        for flag in (
            list(cell.get("quality_flags", []))
            + [
                nested
                for fold in cell.get("fold_reports", [])
                for nested in fold.get("quality_flags", [])
            ]
        )
    ]
    return {
        "config_id": config_id,
        "mechanism": str(cells[0].get("mechanism")),
        "family": str(cells[0].get("candidate", {}).get("family", "control")),
        "n_cells": len(cells),
        "n_subjects": len(subjects),
        "n_seeds": len(seeds),
        "mean_accuracy": _mean(accuracies),
        "mean_kappa": _mean([value["kappa"] for value in subject_metrics.values()]),
        "lower_quartile": float(np.quantile(np.asarray(accuracies), 0.25)),
        "seed_std": float(np.std(np.asarray(seed_means), ddof=0)),
        "collapse_count": sum("class_collapse" in str(flag) for flag in all_flags),
        "quality_flag_count": len(all_flags),
        "subject_metrics": subject_metrics,
    }


def _comparison(
    item: dict[str, Any], reference: dict[str, Any], config: Exp06Config, *, final: bool
) -> dict[str, Any]:
    subjects = sorted(set(item["subject_metrics"]) & set(reference["subject_metrics"]))
    deltas = [
        item["subject_metrics"][subject]["accuracy"]
        - reference["subject_metrics"][subject]["accuracy"]
        for subject in subjects
    ]
    accuracy_delta = _mean(deltas)
    required = config.final_preferred_accuracy_delta if final else config.screen_min_accuracy_delta
    kappa_delta = item["mean_kappa"] - reference["mean_kappa"]
    quartile_delta = item["lower_quartile"] - reference["lower_quartile"]
    seed_std_increase = item["seed_std"] - reference["seed_std"]
    wins = sum(delta > 0 for delta in deltas)
    eligible = bool(
        item["collapse_count"] == 0
        and accuracy_delta >= required
        and wins >= config.min_subject_wins
        and kappa_delta >= -config.max_kappa_drop
        and quartile_delta >= -config.max_lower_quartile_drop
        and seed_std_increase <= config.max_seed_std_increase
    )
    return {
        **item,
        "accuracy_delta": accuracy_delta,
        "kappa_delta": kappa_delta,
        "lower_quartile_delta": quartile_delta,
        "seed_std_increase": seed_std_increase,
        "subject_wins": wins,
        "subject_ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "descriptive_exact_sign_flip_p": _sign_flip_pvalue(deltas),
        "inferential_claim": False,
        "required_accuracy_delta": required,
        "eligible": eligible,
    }


def _rank(cells: list[dict[str, Any]], config: Exp06Config, *, final: bool) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[str(cell["config_id"])].append(cell)
    summaries = {key: _summary(key, value) for key, value in grouped.items()}
    if E11_REFERENCE_ID not in summaries or FBC_CONTROL_ID not in summaries:
        raise RuntimeError("Exp06 readout requires both frozen controls")
    reference = summaries[E11_REFERENCE_ID]
    comparisons = [
        _comparison(item, reference, config, final=final)
        for key, item in summaries.items()
        if key not in {E11_REFERENCE_ID, FBC_CONTROL_ID}
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
    return {
        "context_baseline": summaries[FBC_CONTROL_ID],
        "reference": reference,
        "ranking": comparisons,
    }


def select_screen_config_ids(readout: dict[str, Any], *, max_promoted: int) -> list[str]:
    eligible = [item for item in readout.get("ranking", []) if item.get("eligible") is True]
    selected: list[str] = []
    families: set[str] = set()
    for item in eligible:
        family = str(item["family"])
        if family not in families:
            selected.append(str(item["config_id"]))
            families.add(family)
        if len(selected) == max_promoted:
            return selected
    for item in eligible:
        config_id = str(item["config_id"])
        if config_id not in selected:
            selected.append(config_id)
        if len(selected) == max_promoted:
            break
    return selected


def analyze_screen(output_dir: Path, config: Exp06Config) -> dict[str, Any]:
    ranked = _rank(_read_cells(output_dir, "screen"), config, final=False)
    selected = select_screen_config_ids(ranked, max_promoted=config.max_promoted)
    report = {
        "experiment": "exp06",
        "phase": "screen",
        "statistical_scope": "discovery_descriptive_only",
        **ranked,
    }
    write_json_atomic(output_dir / "analysis" / "screen-readout.json", report)
    body = {
        "status": "selected" if selected else "no_promotion",
        "phase": "screen",
        "selected_config_ids": selected,
        "max_promoted": config.max_promoted,
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
        raise ValueError(f"Missing signed Exp06 screen decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError("Exp06 screen decision signature is invalid")
    readout = json.loads((output_dir / "analysis" / "screen-readout.json").read_text())
    if payload.get("readout_sha256") != canonical_sha256(readout):
        raise ValueError("Exp06 screen decision is detached from its readout")
    expected = select_screen_config_ids(readout, max_promoted=int(payload["max_promoted"]))
    if payload.get("selected_config_ids") != expected:
        raise ValueError("Exp06 screen decision differs from the frozen ranking")
    if payload.get("manifest_sha256") != _manifest_sha256(output_dir, "screen"):
        raise ValueError("Exp06 screen decision is detached from its manifest")
    return payload


def analyze_confirm(output_dir: Path, config: Exp06Config) -> dict[str, Any]:
    decision = load_screen_decision(output_dir)
    selected = set(map(str, decision["selected_config_ids"]))
    allowed = selected | {FBC_CONTROL_ID, E11_REFERENCE_ID}
    cells = _read_cells(output_dir, "screen") + _read_cells(output_dir, "confirm")
    cells = [cell for cell in cells if str(cell["config_id"]) in allowed]
    ranked = _rank(cells, config, final=True)
    eligible = [item for item in ranked["ranking"] if item["eligible"]]
    selected_config_id = eligible[0]["config_id"] if eligible else None
    report = {
        "experiment": "exp06",
        "phase": "confirm",
        "seeds": list(config.screen_seeds + config.confirm_seeds),
        "statistical_scope": "internal_confirmation_holdout_still_closed",
        **ranked,
    }
    write_json_atomic(output_dir / "analysis" / "confirm-readout.json", report)
    body = {
        "status": "promoted" if selected_config_id else "retain_e11",
        "phase": "confirm",
        "selected_config_id": selected_config_id or E11_REFERENCE_ID,
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
    """Load and verify the Exp06 final decision and its complete lineage."""

    path = output_dir / "decisions" / "final-promotion.json"
    if not path.is_file():
        raise ValueError(f"Missing signed Exp06 final decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body) or payload.get("phase") != "confirm":
        raise ValueError("Exp06 final decision signature is invalid")
    if payload.get("holdout_opened") is not False:
        raise ValueError("Exp06 final decision opened the holdout")
    if payload.get("confirmatory_p_value_used") is not False:
        raise ValueError("Exp06 discovery cannot make a confirmatory p-value claim")
    readout_path = output_dir / "analysis" / "confirm-readout.json"
    if not readout_path.is_file():
        raise ValueError(f"Missing Exp06 confirmation readout: {readout_path}")
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    if payload.get("readout_sha256") != canonical_sha256(readout):
        raise ValueError("Exp06 final decision is detached from its readout")
    screen = load_screen_decision(output_dir)
    if payload.get("screen_decision_sha256") != screen.get("sha256"):
        raise ValueError("Exp06 final decision is detached from screening")
    if payload.get("manifest_sha256") != _manifest_sha256(output_dir, "confirm"):
        raise ValueError("Exp06 final decision is detached from its manifest")
    expected = next(
        (
            str(item["config_id"])
            for item in readout.get("ranking", [])
            if item.get("eligible") is True
        ),
        E11_REFERENCE_ID,
    )
    if payload.get("selected_config_id") != expected:
        raise ValueError("Exp06 final decision differs from the frozen ranking")
    expected_status = "promoted" if expected != E11_REFERENCE_ID else "retain_e11"
    if payload.get("status") != expected_status:
        raise ValueError("Exp06 final decision status differs from the frozen ranking")
    return payload
