#!/usr/bin/env python3
"""Run the predeclared paper profile in restartable phases."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.config import MODEL_NAMES, PAPER_EPOCHS, PAPER_SEEDS, RESULTS_DIR
from deepbench.datasets import available_subjects
from deepbench.io import write_json_atomic
from deepbench.paper_profile import paper_blocks, paper_profile_metadata
from deepbench.reproducibility import configure_determinism, get_device, write_manifest
from deepbench.runner import cell_path, run_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("peterson", "souza", "primary", "external", "ica-sensitivity", "all"),
        default="all",
    )
    parser.add_argument("--device", default=get_device())
    parser.add_argument("--epochs", type=int, default=PAPER_EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_determinism()
    if args.epochs != PAPER_EPOCHS:
        raise SystemExit(
            f"The confirmatory paper profile is frozen at {PAPER_EPOCHS} epochs; "
            "use scripts/run_experiments.py for pilots."
        )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.num_shards > 1 and args.phase in {"peterson", "souza", "primary", "all"}:
        raise SystemExit(
            "Subject sharding is disabled for the primary phase because LOSO requires the full "
            "training cohort. Shard only external or ica-sensitivity."
        )
    write_manifest(
        args.output_dir
        / "manifests"
        / (
            f"paper-profile-{args.phase}-epochs{args.epochs}-"
            f"shard{args.shard_index}-of-{args.num_shards}.json"
        ),
        config={
            "profile": "paper",
            "phase": args.phase,
            "models": list(MODEL_NAMES),
            "seeds": list(PAPER_SEEDS),
            "epochs": args.epochs,
            "device": args.device,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
        root=Path(__file__).resolve().parents[1],
    )
    common = {
        "models": list(MODEL_NAMES),
        "seeds": list(PAPER_SEEDS),
        "device": args.device,
        "epochs": args.epochs,
        "output_dir": args.output_dir,
        "overwrite": False,
        "save_weights": args.save_weights,
    }
    blocks = paper_blocks(args.phase)
    expected_cells = []
    selected_subjects: dict[str, list[str]] = {}
    all_subjects: dict[str, list[str]] = {}
    for block in blocks:
        dataset = str(block["dataset"])
        if dataset not in all_subjects:
            all_subjects[dataset] = available_subjects(dataset)
            selected_subjects[dataset] = all_subjects[dataset][args.shard_index :: args.num_shards]
        for protocol in block["protocols"]:
            for condition in block["conditions"]:
                for model in MODEL_NAMES:
                    for seed in PAPER_SEEDS:
                        for subject in all_subjects[dataset]:
                            expected_cells.append(
                                str(
                                    cell_path(
                                        args.output_dir,
                                        dataset,
                                        str(protocol),
                                        str(condition),
                                        model,
                                        seed,
                                        subject,
                                        str(block["ica_policy"]),
                                    ).relative_to(args.output_dir)
                                )
                            )
    write_json_atomic(
        args.output_dir
        / "manifests"
        / (f"paper-expected-{args.phase}-shard{args.shard_index}-of-{args.num_shards}.json"),
        {
            **paper_profile_metadata(args.phase),
            "phase": args.phase,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "expected_cells": sorted(set(expected_cells)),
        },
    )
    print(f"planned_cells={len(set(expected_cells))}")
    if args.plan_only:
        return
    for block in blocks:
        run_job(subjects=selected_subjects[str(block["dataset"])], **block, **common)


if __name__ == "__main__":
    main()
