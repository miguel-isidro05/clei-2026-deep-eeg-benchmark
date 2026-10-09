#!/usr/bin/env python3
"""Plan or execute one resumable Peterson diffusion V11 shard."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.peterson_diffusion_v11.config import load_wave_config  # noqa: E402
from experiments.peterson_diffusion_v11.runner import run_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--inference-k", type=int)
    parser.add_argument(
        "--objective",
        choices=("noise_only", "noise_rank", "noise_rank_consistency"),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = load_wave_config(args.wave)
    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "inference_k": args.inference_k,
            "objective": args.objective,
        }.items()
        if value is not None
    }
    config = replace(config, **overrides)
    report = run_plan(
        config,
        output_dir=args.output_dir,
        device=args.device,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        plan_only=args.plan_only,
    )
    print(" ".join(f"{key}={value}" for key, value in report.items()), flush=True)


if __name__ == "__main__":
    main()
