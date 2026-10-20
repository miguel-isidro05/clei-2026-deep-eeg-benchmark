#!/usr/bin/env python3
"""Create signed Exp06 screen or confirmation decisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.exp06_window_fusion.analysis import (  # noqa: E402
    analyze_confirm,
    analyze_screen,
)
from experiments.peterson_search_v12.exp06_window_fusion.config import (  # noqa: E402
    Exp06Config,
    smoke_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = smoke_config() if args.smoke else Exp06Config()
    report = (
        analyze_screen(args.output_dir, config)
        if args.phase == "screen"
        else analyze_confirm(args.output_dir, config)
    )
    print(f"exp06_analysis=OK phase={args.phase} candidates={len(report['ranking'])}")


if __name__ == "__main__":
    main()
