#!/usr/bin/env python3
"""Generate statistics only after the Peterson integrity gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from deepbench.config import PAPER_SEEDS
from deepbench.peterson_audit import audit_peterson_results
from deepbench.peterson_postprocessing import (
    all_metric_descriptives,
    friedman_omnibus,
    ica_sensitivity_tests,
    model_rankings,
    prediction_diagnostics,
    protocol_gap_tests,
    sample_accounting_summary,
    subject_seed_summary,
    training_diagnostics,
)
from deepbench.statistics import load_cells, write_statistics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    frame = load_cells(args.results_dir / "cells")
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.results_dir / "cells").glob("*.json"))
    ]
    additions = {
        "descriptive_all_metrics.csv": all_metric_descriptives(frame),
        "subject_seed_summary.csv": subject_seed_summary(frame),
        "friedman_omnibus.csv": friedman_omnibus(frame),
        "model_rankings.csv": model_rankings(frame),
        "ica_sensitivity_wilcoxon_holm.csv": ica_sensitivity_tests(frame),
        "protocol_gap_wilcoxon_holm.csv": protocol_gap_tests(frame),
        "prediction_diagnostics.csv": prediction_diagnostics(payloads),
        "training_diagnostics.csv": training_diagnostics(payloads),
    }
    sample_accounting = pd.read_csv(output / "sample_accounting.csv")
    summary = sample_accounting_summary(sample_accounting)
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    additions["sample_accounting_summary.csv"] = summary
    for name, table in additions.items():
        table.to_csv(output / name, index=False)
    generated = sorted(output.glob("*.csv"))
    manifest = {
        "analysis_scope": "Peterson MI-vs-rest only",
        "inference_unit": "participant after within-participant seed averaging",
        "confirmatory_family": "overlap/no-ICA accuracy and kappa paired Wilcoxon-Holm",
        "exploratory_outputs": [
            "friedman_omnibus.csv",
            "ica_sensitivity_wilcoxon_holm.csv",
            "protocol_gap_wilcoxon_holm.csv",
        ],
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in generated
        ],
    }
    (output / "statistics_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"peterson_statistics=OK output={output}", flush=True)


if __name__ == "__main__":
    main()
