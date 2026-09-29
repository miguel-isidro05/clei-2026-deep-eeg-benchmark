#!/usr/bin/env python3
"""Generate publication figures only after the confirmatory integrity gate passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepbench.config import RESULTS_DIR
from deepbench.figures import generate_all_figures
from deepbench.paper_audit import audit_expected_cells, result_cells_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    _audit, confirmatory, issues = audit_expected_cells(args.results_dir)
    if not confirmatory:
        raise SystemExit("Figure generation blocked: " + "; ".join(issues))
    manifest_path = args.results_dir / "statistics" / "statistics_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Figure generation blocked: statistics_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("result_cells_sha256") != result_cells_sha256(args.results_dir):
        raise SystemExit("Figure generation blocked: statistics are stale for the current cells")
    generate_all_figures(args.results_dir)
    print(f"Figures written to {args.results_dir / 'figures'}")


if __name__ == "__main__":
    main()
