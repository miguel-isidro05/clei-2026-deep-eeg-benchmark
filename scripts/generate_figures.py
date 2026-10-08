#!/usr/bin/env python3
"""Generate publication figures only after the confirmatory integrity gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deepbench.config import RESULTS_DIR
from deepbench.figures import figures_code_sha256, generate_all_figures
from deepbench.io import write_json_atomic
from deepbench.paper_audit import audit_expected_cells, result_cells_sha256
from deepbench.statistics import statistics_code_sha256


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
    if manifest.get("statistics_code_sha256") != statistics_code_sha256():
        raise SystemExit("Figure generation blocked: statistics code has changed")
    required_tables = {
        "all_seed_metrics.csv",
        "descriptive_subject_seed_variability.csv",
        "augmentation_wilcoxon_holm.csv",
    }
    recorded_tables = manifest.get("statistics_files", {})
    for name in required_tables:
        path = args.results_dir / "statistics" / name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual_hash is None or recorded_tables.get(name) != actual_hash:
            raise SystemExit(f"Figure generation blocked: invalid statistics table {name}")
    generate_all_figures(args.results_dir)
    figure_files = sorted((args.results_dir / "figures").glob("*.pdf"))
    write_json_atomic(
        args.results_dir / "figures" / "figures_manifest.json",
        {
            "result_cells_sha256": result_cells_sha256(args.results_dir),
            "figures_code_sha256": figures_code_sha256(),
            "figures": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in figure_files
            },
        },
    )
    print(f"Figures written to {args.results_dir / 'figures'}")


if __name__ == "__main__":
    main()
