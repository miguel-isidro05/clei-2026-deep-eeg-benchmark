#!/usr/bin/env python3
"""Analyze a completed V11 wave using its predeclared decision rule."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.peterson_diffusion_v11.decision import decide_h1  # noqa: E402
from experiments.peterson_diffusion_v11.io import write_json_atomic  # noqa: E402
from experiments.peterson_diffusion_v11.statistics import (  # noqa: E402
    holm_adjust,
    paired_summary,
)

DEEP_MODELS = ("EEGNet", "FBCNet", "ShallowConvNet", "EEGConformer")
ALL_MODELS = ("CSP+LDA", *DEEP_MODELS)


def _mean_cells(paths: list[Path]) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        key = (str(payload["model"]), str(payload["subject"]))
        values[key].append(float(payload["metrics"]["accuracy"]))
    return {key: float(np.mean(items)) for key, items in values.items()}


def _load_index(paths: list[Path]) -> dict[tuple[str, str, int], dict[str, object]]:
    output: dict[tuple[str, str, int], dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        key = (str(payload["model"]), str(payload["subject"]), int(payload["seed"]))
        if key in output:
            raise RuntimeError(f"Duplicate result cell for {key}")
        output[key] = payload
    return output


def _training_updates(payload: dict[str, object]) -> int:
    if "training_updates" in payload:
        return int(payload["training_updates"])
    total = 0
    for fold in payload.get("fold_reports", []):
        for epoch in fold.get("training_history", []):
            total += int(epoch.get("train_batch_count", 0))
    return total


def _verify_compute_match(center_paths: list[Path], overlap_paths: list[Path]) -> dict[str, object]:
    center = _load_index(center_paths)
    overlap = _load_index(overlap_paths)
    if len(center) != 250 or len(overlap) != 250 or set(center) != set(overlap):
        raise RuntimeError(
            f"H1 requires matching 250-cell matrices; center={len(center)} overlap={len(overlap)}"
        )
    mismatches: list[str] = []
    for key in sorted(center):
        left = center[key]
        right = overlap[key]
        if int(left["n_train_examples"]) != int(right["n_train_examples"]):
            mismatches.append(f"examples:{key}")
        if _training_updates(left) != _training_updates(right):
            mismatches.append(f"updates:{key}")
    if mismatches:
        raise RuntimeError(f"H1 compute matching failed: {mismatches[:5]}")
    return {
        "matched": True,
        "cell_pairs": len(center),
        "matched_fields": ["n_train_examples", "training_updates"],
    }


def _subject_model_map(
    means: dict[tuple[str, str], float], model: str
) -> dict[str, float]:
    return {subject: value for (current, subject), value in means.items() if current == model}


def _deep_subject_average(means: dict[tuple[str, str], float]) -> dict[str, float]:
    subjects = sorted({subject for model, subject in means if model in DEEP_MODELS})
    output: dict[str, float] = {}
    for subject in subjects:
        values = [means[(model, subject)] for model in DEEP_MODELS if (model, subject) in means]
        if len(values) != len(DEEP_MODELS):
            raise RuntimeError(f"Incomplete deep-model evidence for {subject}")
        output[subject] = float(np.mean(values))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v10-results-dir", type=Path, required=True)
    args = parser.parse_args()
    complete = args.output_dir / "run_complete.json"
    if not complete.exists():
        raise SystemExit("Audit the V11 wave before analyzing it")
    center_paths = sorted((args.output_dir / "cells").glob("wave00__*.json"))
    overlap_paths = sorted(
        (args.v10_results_dir / "cells").glob(
            "MI-OpenBCI__within_session__overlap__ica-none__*.json"
        )
    )
    center = _mean_cells(center_paths)
    overlap = _mean_cells(overlap_paths)
    compute_match = _verify_compute_match(center_paths, overlap_paths)
    model_summaries: dict[str, dict[str, object]] = {}
    raw_p: list[float] = []
    for model in ALL_MODELS:
        summary = paired_summary(
            _subject_model_map(center, model), _subject_model_map(overlap, model)
        )
        model_summaries[model] = summary
        raw_p.append(float(summary["wilcoxon_p"]))
    for model, adjusted in zip(ALL_MODELS, holm_adjust(raw_p), strict=True):
        model_summaries[model]["holm_p"] = adjusted
    primary_center = _deep_subject_average(center)
    primary_overlap = _deep_subject_average(overlap)
    decision = decide_h1(primary_center, primary_overlap)
    payload = {
        "status": "completed",
        "wave": 0,
        "primary_estimand": (
            "seed_mean_then_mean_of_EEGNet_FBCNet_ShallowConvNet_EEGConformer_per_subject"
        ),
        "primary_statistics": paired_summary(primary_center, primary_overlap),
        "compute_match": compute_match,
        "model_sensitivities": model_summaries,
        "decision": decision,
        "center_x6_subject_values": primary_center,
        "overlap_subject_values": primary_overlap,
    }
    write_json_atomic(args.output_dir / "statistics" / "h1.json", payload)
    write_json_atomic(args.output_dir / "decisions" / "wave_00.json", decision)
    print(
        f"h1_decision={decision['decision']} primary_condition={decision['primary_condition']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
