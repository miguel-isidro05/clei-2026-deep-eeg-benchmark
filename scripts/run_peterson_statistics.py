#!/usr/bin/env python3
"""Generate statistics only after the Peterson integrity gate passes."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.config import PAPER_SEEDS
from deepbench.peterson_audit import audit_peterson_results
from deepbench.statistics import write_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    audit, issues = audit_peterson_results(args.results_dir)
    output = args.results_dir / "statistics"
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "expected_cell_audit.csv", index=False)
    if issues:
        raise SystemExit("Peterson audit failed: " + "; ".join(issues[:20]))
    write_statistics(args.results_dir / "cells", output, PAPER_SEEDS)
    print(f"peterson_statistics=OK output={output}", flush=True)


if __name__ == "__main__":
    main()
