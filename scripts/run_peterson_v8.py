#!/usr/bin/env python3
"""Run the frozen no-LOSO Peterson v8 primary matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.config import MODEL_NAMES, PAPER_EPOCHS, PAPER_SEEDS
from deepbench.runner import run_job

BLOCKS = (
    (["within_split", "within_session"], ["overlap"], "none"),
    (["within_split", "within_session"], ["full"], "none"),
    (["within_split"], ["center_x2", "nonoverlap", "center_x6"], "none"),
    (["within_split"], ["overlap"], "kurtosis"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Invalid shard")
    models_seeds = [(model, seed) for model in MODEL_NAMES for seed in PAPER_SEEDS]
    selected = models_seeds[args.shard_index :: args.num_shards]
    print(f"planned_cells={len(selected) * 10 * 8}", flush=True)
    if args.plan_only:
        return
    for model, seed in selected:
        for protocols, conditions, ica_policy in BLOCKS:
            run_job(
                dataset="MI-OpenBCI",
                models=[model],
                protocols=protocols,
                conditions=conditions,
                seeds=[seed],
                subjects=None,
                device=args.device,
                epochs=PAPER_EPOCHS,
                ica_policy=ica_policy,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
