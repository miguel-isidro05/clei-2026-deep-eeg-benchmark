#!/usr/bin/env python3
"""Analyze one completed Exp04b wave."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.exp04b_safepath.analysis import summarize_wave  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wave", choices=("w1", "w2", "w3"), required=True)
    args = parser.parse_args()
    report = summarize_wave(args.output_dir, args.wave)
    print(f"exp04b_analysis=OK wave={args.wave} cells={report['n_cells']}")


if __name__ == "__main__":
    main()
