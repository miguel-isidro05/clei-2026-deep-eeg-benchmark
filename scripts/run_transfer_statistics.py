#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.io import write_json_atomic
from deepbench.transfer_statistics import analyze_transfer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_transfer(args.output_dir)
    write_json_atomic(args.output_dir / "statistics" / "transfer_contrasts.json", result)
    print("transfer_statistics=OK", flush=True)


if __name__ == "__main__":
    main()
