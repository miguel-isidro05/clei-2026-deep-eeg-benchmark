#!/usr/bin/env python3
"""Analyze one completed Peterson Exp05 phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.analysis import (  # noqa: E402
    analyze_confirm,
    analyze_screen,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.config import (  # noqa: E402
    Exp05Config,
    smoke_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = smoke_config() if args.smoke else Exp05Config()
    report = (
        analyze_screen(args.output_dir, config)
        if args.phase == "screen"
        else analyze_confirm(args.output_dir, config)
    )
    print(f"exp05_analysis=OK phase={args.phase} cells={report['n_cells']}")


if __name__ == "__main__":
    main()
