#!/usr/bin/env python3
"""Generate final descriptive and inferential tables from completed cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from deepbench.config import PAPER_SEEDS, RESULTS_DIR
from deepbench.statistics import write_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    (args.results_dir / "statistics").mkdir(parents=True, exist_ok=True)
    expected: set[str] = set()
    missing = 0
    for path in (args.results_dir / "manifests").glob("paper-expected-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected.update(payload.get("expected_cells", []))
    if expected:
        present = {
            str(path.relative_to(args.results_dir))
            for path in (args.results_dir / "cells").glob("*.json")
        }
        audit = pd.DataFrame(
            [{"cell": cell, "present": cell in present} for cell in sorted(expected)]
        )
        audit.to_csv(args.results_dir / "statistics" / "expected_cell_audit.csv", index=False)
        missing = int((~audit["present"]).sum())
        if missing and not args.allow_incomplete:
            raise SystemExit(
                f"{missing} expected paper cells are missing; see expected_cell_audit.csv"
            )
    write_statistics(
        args.results_dir / "cells",
        args.results_dir / "statistics",
        tuple(args.seeds),
        allow_incomplete=args.allow_incomplete,
        force_exploratory=missing > 0,
    )
    print(f"Statistics written to {args.results_dir / 'statistics'}")


if __name__ == "__main__":
    main()
