#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.latency_merge import merge_latency_profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    merge_latency_profiles(args.inputs, args.output_dir)
    print(
        f"latency_merge=OK devices={len(args.inputs)} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
