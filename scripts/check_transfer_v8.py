#!/usr/bin/env python3
"""Require exact v8 completeness before statistics or export."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.io import read_json
from deepbench.runner import _code_fingerprint
from deepbench.transfer_profile import (
    TRANSFER_PROFILE_VERSION,
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
        current_code = _code_fingerprint()
        environments: set[str] = set()
        for name in expected_cells:
            payload = read_json(args.output_dir / "cells" / name)
            configuration = payload.get("run_configuration", {})
            if configuration.get("code_sha256") != current_code:
                raise SystemExit(f"Stale Peterson cell: {name}")
            environment = configuration.get("environment_sha256")
            if not isinstance(environment, str) or len(environment) != 64:
                raise SystemExit(f"Invalid Peterson environment identity: {name}")
            environments.add(environment)
        for name in expected_transfer:
            payload = read_json(args.output_dir / "transfer_cells" / name)
            if payload.get("profile") != TRANSFER_PROFILE_VERSION:
                raise SystemExit(f"Mixed transfer profile: {name}")
            if payload.get("scientific_code_sha256") != current_code:
                raise SystemExit(f"Stale transfer cell: {name}")
            environment = payload.get("environment_sha256")
            if not isinstance(environment, str) or len(environment) != 64:
                raise SystemExit(f"Invalid transfer environment identity: {name}")
            environments.add(environment)
        if len(environments) != 1:
            raise SystemExit(f"Mixed numerical environments: {sorted(environments)}")
    print(f"transfer_complete={len(expected)} phase={args.phase}")


if __name__ == "__main__":
    main()
