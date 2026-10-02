#!/usr/bin/env python3
"""Require exact v8 completeness before statistics or export."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.transfer_profile import (
    expected_peterson_cells,
    expected_source_checkpoints,
    expected_transfer_cells,
)
from deepbench.transfer_runner import load_source_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("source", "all"), default="all")
    args = parser.parse_args()
    expected = [
        args.output_dir / "source_checkpoints" / name for name in expected_source_checkpoints()
    ]
    if args.phase == "all":
        expected += [args.output_dir / "cells" / name for name in expected_peterson_cells()]
        expected += [
            args.output_dir / "transfer_cells" / name for name in expected_transfer_cells()
        ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise SystemExit(f"v8 incomplete: missing={len(missing)} first={missing[:5]}")
    for model_seed in expected_source_checkpoints():
        stem = Path(model_seed).stem
        model = stem.split("__")[1]
        seed = int(stem.rsplit("seed", maxsplit=1)[1])
        load_source_checkpoint(
            args.output_dir / "source_checkpoints" / model_seed,
            model=model,
            seed=seed,
        )
    if args.phase == "all":
        expected_cells = set(expected_peterson_cells())
        actual_cells = {path.name for path in (args.output_dir / "cells").glob("*.json")}
        expected_transfer = set(expected_transfer_cells())
        actual_transfer = {
            path.name for path in (args.output_dir / "transfer_cells").glob("*.json")
        }
        unexpected = sorted((actual_cells - expected_cells) | (actual_transfer - expected_transfer))
        if unexpected:
            raise SystemExit(f"v8 contains unexpected cells: {unexpected[:5]}")
    print(f"v8_complete={len(expected)} phase={args.phase}")


if __name__ == "__main__":
    main()
