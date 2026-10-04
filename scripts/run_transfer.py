#!/usr/bin/env python3
"""Run resumable Peterson source pretraining or Souza transfer cells."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.config import MODEL_NAMES, PAPER_SEEDS, SOUZA_SUBJECTS
from deepbench.io import read_json
from deepbench.runner import _code_fingerprint
from deepbench.transfer import TRANSFER_ARMS
from deepbench.transfer_profile import TRANSFER_PROFILE_VERSION
from deepbench.transfer_runner import (
    load_source_checkpoint,
    pretrain_peterson_source,
    run_souza_transfer_cell,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pretrain", "target"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--subjects", nargs="+", default=list(SOUZA_SUBJECTS))
    parser.add_argument("--arms", nargs="+", choices=TRANSFER_ARMS, default=list(TRANSFER_ARMS))
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("within_session", "cross_session"),
        default=["within_session", "cross_session"],
    )
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Invalid shard")
    jobs: list[tuple[object, ...]] = []
    if args.phase == "pretrain":
        jobs = [(model, seed) for model in args.models for seed in args.seeds]
    else:
        jobs = [
            (subject, model, seed, arm, protocol)
            for protocol in args.protocols
            for arm in args.arms
            for model in args.models
            for seed in args.seeds
            for subject in args.subjects
        ]
    jobs = jobs[args.shard_index :: args.num_shards]
    print(f"planned_jobs={len(jobs)} phase={args.phase}", flush=True)
    if args.plan_only:
        return
    for job in jobs:
        if args.phase == "pretrain":
            model, seed = job
            destination = args.output_dir / "source_checkpoints" / f"source__{model}__seed{seed}.pt"
            if destination.exists():
                load_source_checkpoint(destination, model=str(model), seed=int(seed))
                print(f"skipped {destination.name}", flush=True)
                continue
            pretrain_peterson_source(
                str(model),
                int(seed),
                device=args.device,
                output_dir=args.output_dir,
                max_epochs=args.max_epochs,
                min_epochs=args.min_epochs,
                patience=args.patience or 50,
            )
            print(f"completed {destination.name}", flush=True)
        else:
            subject, model, seed, arm, protocol = job
            destination = (
                args.output_dir
                / "transfer_cells"
                / (
                    f"Souza2023__transfer__{protocol}__overlap__{arm}__{model}__seed{seed}__{subject}.json"
                )
            )
            if destination.exists():
                previous = read_json(destination)
                expected = {
                    "profile": TRANSFER_PROFILE_VERSION,
                    "subject": subject,
                    "model": model,
                    "seed": seed,
                    "arm": arm,
                    "protocol": protocol,
                    "condition": "overlap",
                    "scientific_code_sha256": _code_fingerprint(),
                }
                mismatched = [key for key, value in expected.items() if previous.get(key) != value]
                if mismatched:
                    raise RuntimeError(f"Stale transfer cell {destination}: {mismatched}")
                print(f"skipped {destination.name}", flush=True)
                continue
            run_souza_transfer_cell(
                str(subject),
                str(model),
                int(seed),
                str(arm),
                str(protocol),
                device=args.device,
                output_dir=args.output_dir,
                max_epochs=args.max_epochs,
                min_epochs=args.min_epochs,
                patience=args.patience or 40,
            )
            print(f"completed {destination.name}", flush=True)


if __name__ == "__main__":
    main()
