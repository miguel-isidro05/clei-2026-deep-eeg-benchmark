#!/usr/bin/env python3
"""Plan or execute the Peterson V12 exp01 search."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.peterson_search_v12.common.config import SearchConfig  # noqa: E402
from experiments.peterson_search_v12.common.identity import git_identity  # noqa: E402
from experiments.peterson_search_v12.common.runner import (  # noqa: E402
    run_search,
    write_plan,
    write_smoke_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-cells-per-shard", type=int)
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SearchConfig()
    if args.epochs_override is not None:
        if not args.smoke:
            raise SystemExit("--epochs-override is restricted to --smoke")
        config = replace(
            config,
            epochs=args.epochs_override,
            patience=min(args.epochs_override, config.patience),
        )
    root = ROOT
    if args.plan_only:
        revision = str(git_identity(root)["revision"] or "unknown")
        plan = (
            write_smoke_plan(args.output_dir, config, revision=revision)
            if args.smoke
            else write_plan(args.output_dir, config, revision=revision)
        )
        print(f"planned_candidates={plan['candidate_count']}")
        print(f"planned_cells={plan['cell_count']}")
        return
    summary = run_search(
        args.output_dir,
        config,
        device=args.device,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_cells_per_shard=args.max_cells_per_shard,
        smoke=args.smoke,
    )
    print(
        f"search_shard_complete={summary['completed']}/{summary['planned']} "
        f"manifest_cells={summary['manifest_cells']}"
    )


if __name__ == "__main__":
    main()
