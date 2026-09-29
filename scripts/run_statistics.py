#!/usr/bin/env python3
"""Generate final descriptive and inferential tables from completed cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepbench.config import PAPER_SEEDS, RESULTS_DIR
from deepbench.io import write_json_atomic
from deepbench.paper_audit import audit_expected_cells, result_cells_sha256
from deepbench.statistics import write_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    (args.results_dir / "statistics").mkdir(parents=True, exist_ok=True)
    audit, confirmatory, issues = audit_expected_cells(args.results_dir)
    audit.to_csv(args.results_dir / "statistics" / "expected_cell_audit.csv", index=False)
    if not confirmatory and not args.allow_incomplete:
        details = "; ".join(issues)
        raise SystemExit(f"Confirmatory integrity audit failed: {details}")
    write_statistics(
        args.results_dir / "cells",
        args.results_dir / "statistics",
        tuple(args.seeds),
        allow_incomplete=args.allow_incomplete,
        force_exploratory=not confirmatory,
    )
    expectation_manifests = sorted(
        (args.results_dir / "manifests").glob("paper-expected-*.json")
    )
    profile_hashes = sorted(
        {
            str(json.loads(path.read_text(encoding="utf-8")).get("profile_sha256"))
            for path in expectation_manifests
        }
    )
    write_json_atomic(
        args.results_dir / "statistics" / "statistics_manifest.json",
        {
            "result_cells_sha256": result_cells_sha256(args.results_dir),
            "profile_sha256_values": profile_hashes,
            "confirmatory": confirmatory,
            "expected_cell_count": int(len(audit)),
        },
    )
    print(f"Statistics written to {args.results_dir / 'statistics'}")


if __name__ == "__main__":
    main()
