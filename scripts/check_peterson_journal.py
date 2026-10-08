#!/usr/bin/env python3
"""Fail unless the Peterson-only journal result grid is exact and valid."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.peterson_audit import audit_peterson_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    table, issues = audit_peterson_results(args.output_dir)
    destination = args.output_dir / "statistics" / "expected_cell_audit.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    if issues:
        raise SystemExit("Peterson audit failed: " + "; ".join(issues[:20]))
    print(f"peterson_complete={len(table)}", flush=True)


if __name__ == "__main__":
    main()
