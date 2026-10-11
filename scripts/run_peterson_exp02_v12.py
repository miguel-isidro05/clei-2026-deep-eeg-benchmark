#!/usr/bin/env python3
"""Plan or execute Peterson V12 exp02."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.common.identity import git_identity  # noqa: E402
from experiments.peterson_search_v12.exp02_factorial.config import Exp02Config  # noqa: E402
from experiments.peterson_search_v12.exp02_factorial.runner import (  # noqa: E402
    run_search,
    write_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-cells-per-shard", type=int)
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = Exp02Config()
    if args.epochs_override is not None:
        if not args.smoke:
            raise SystemExit("--epochs-override is restricted to --smoke")
        config = replace(
            config,
            epochs=args.epochs_override,
            patience=min(args.epochs_override, config.patience),
            baseline_epochs=args.epochs_override,
        )
    revision = str(git_identity(ROOT)["revision"] or "unknown")
    if args.plan_only:
        plan = write_plan(args.output_dir, config, revision=revision, smoke=args.smoke)
        print(f"planned_candidates={plan['candidate_count']}")
        print(f"planned_inference_recipes={plan['inference_recipe_count']}")
        print(f"planned_cells={plan['cell_count']}")
        return
    summary = run_search(
        args.output_dir,
        config,
        device=args.device,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        smoke=args.smoke,
        max_cells_per_shard=args.max_cells_per_shard,
    )
    print(
        f"exp02_shard_complete={summary['completed']}/{summary['planned']} "
        f"failed={summary['failed']} manifest_cells={summary['manifest_cells']}"
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
