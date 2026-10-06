#!/usr/bin/env python3
"""Run the frozen Peterson-only journal matrix with resumable shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.config import PAPER_EPOCHS, PAPER_SEEDS, PETERSON_MODEL_NAMES
from deepbench.io import write_json_atomic
from deepbench.peterson_profile import PETERSON_BLOCKS, profile_metadata
from deepbench.runner import _code_fingerprint, run_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Invalid shard")
    metadata = profile_metadata()
    metadata["scientific_code_sha256"] = _code_fingerprint()
    write_json_atomic(args.output_dir / "manifests" / "peterson-journal-expected.json", metadata)
    pairs = [(model, seed) for model in PETERSON_MODEL_NAMES for seed in PAPER_SEEDS]
    selected = pairs[args.shard_index :: args.num_shards]
    cells_per_pair = int(metadata["expected_cell_count"]) // len(pairs)
    print(
        f"peterson_profile={metadata['profile_version']} "
        f"planned_total={metadata['expected_cell_count']} "
        f"planned_shard={len(selected) * cells_per_pair}",
        flush=True,
    )
    if args.plan_only:
        return
    for model, seed in selected:
        for block in PETERSON_BLOCKS:
            run_job(
                dataset="MI-OpenBCI",
                models=[model],
                protocols=list(block.protocols),
                conditions=list(block.conditions),
                seeds=[seed],
                subjects=None,
                device=args.device,
                epochs=PAPER_EPOCHS,
                ica_policy=block.ica_policy,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
