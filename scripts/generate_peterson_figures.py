#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.peterson_figures import generate_peterson_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    files = generate_peterson_figures(args.results_dir)
    print(f"peterson_figures=OK files={len(files)} output={args.results_dir / 'figures'}")


if __name__ == "__main__":
    main()
