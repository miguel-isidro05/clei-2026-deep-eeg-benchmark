#!/usr/bin/env python3
"""Build a factual scientific readout and reviewer/claim ledgers."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from deepbench.peterson_postprocessing import review_closure_table, write_artifact_manifest


def _fmt(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    stats = args.results_dir / "statistics"
    descriptive = pd.read_csv(stats / "descriptive_all_metrics.csv")
    primary = descriptive.loc[
        (descriptive["condition"] == "overlap")
        & (descriptive["ica_policy"] == "none")
        & (descriptive["metric"].isin(["accuracy", "kappa"]))
    ]
    lines = [
        "# Peterson-only scientific readout",
        "",
        "## Evidence boundary",
        "",
        "This is a within-dataset benchmark of motor imagery versus rest on MI-OpenBCI: "
        "10 participants, five decoders, five seeds. It does not establish external "
        "generalization or a causal advantage of low-cost hardware.",
        "",
        "## Primary overlap/no-ICA results",
        "",
        "Participant means are computed after averaging seeds; uncertainty is across participants.",
        "",
        "| Protocol | Metric | Model | Mean | SD | 95% CI |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in primary.sort_values(
        ["protocol", "metric", "mean"], ascending=[True, True, False]
    ).itertuples():
        lines.append(
            f"| {row.protocol} | {row.metric} | {row.model} | {_fmt(row.mean)} | "
            f"{_fmt(row.sd_between_subjects)} | {_fmt(row.ci95_low)}–{_fmt(row.ci95_high)} |"
        )
    paired = pd.read_csv(stats / "paired_wilcoxon_holm.csv")
    significant = paired.loc[
        (paired["condition"] == "overlap")
        & (paired["ica_policy"] == "none")
        & paired["reject_holm_0_05"].astype(bool)
    ]
    lines += [
        "",
        "## Confirmatory paired inference",
        "",
        "Holm correction is applied within each protocol×metric family of planned model "
        "contrasts (three protocols × accuracy/kappa).",
        "",
    ]
    if significant.empty:
        lines.append("No overlap/no-ICA model contrast survived Holm correction.")
    else:
        for row in significant.itertuples():
            lines.append(
                f"- {row.protocol}, {row.metric}: {row.model_a} − {row.model_b} = "
                f"{row.mean_difference_a_minus_b:.4f}; Holm p={row.p_holm:.4g}; "
                f"paired rank-biserial={row.rank_biserial_a_minus_b:.3f}."
            )
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- Planned Wilcoxon-Holm overlap/no-ICA accuracy and kappa contrasts are confirmatory; "
        "the correction family is one protocol×metric block.",
        "- Friedman, protocol-gap, ICA, calibration, ERD/ERS, ranking, and "
        "convergence analyses are exploratory or diagnostic.",
        "- LOSO is reported separately and must not be pooled with within-subject protocols.",
        "- Model-forward latency is not end-to-end online BCI latency.",
        "- No claim should extend beyond this dataset and participant cohort.",
    ]
    (args.results_dir / "SCIENTIFIC_READOUT.md").write_text("\n".join(lines) + "\n")
    closure = review_closure_table(
        has_latency=(args.results_dir / "latency" / "latency.json").exists()
    )
    closure.to_csv(args.results_dir / "REVIEW_CLOSURE.csv", index=False)
    claims = [
        {
            "claim": "The benchmark grid is complete and internally consistent.",
            "status": "supported",
            "evidence": ["statistics/expected_cell_audit.csv", "statistics/completeness.csv"],
            "limitation": "Internal integrity does not imply external validity.",
        },
        {
            "claim": "Model performance differs by evaluation protocol.",
            "status": "supported_descriptively_exploratory_inferentially",
            "evidence": [
                "statistics/descriptive_all_metrics.csv",
                "statistics/protocol_gap_wilcoxon_holm.csv",
            ],
            "limitation": "Protocol-gap tests are exploratory.",
        },
        {
            "claim": "Overlapping windows universally improve generalization.",
            "status": "not_supported",
            "evidence": [
                "statistics/augmentation_wilcoxon_holm.csv",
                "figures/condition_accuracy_heatmaps.png",
            ],
            "limitation": "Effects are model- and protocol-dependent.",
        },
        {
            "claim": "Results generalize to other low-cost EEG datasets.",
            "status": "not_tested",
            "evidence": [],
            "limitation": "Single dataset, ten participants.",
        },
    ]
    (args.results_dir / "CLAIM_LEDGER.json").write_text(json.dumps(claims, indent=2) + "\n")
    bibliography_audit = Path("docs/PETERSON_BIBLIOGRAPHY_AUDIT.md")
    if bibliography_audit.exists():
        shutil.copy2(bibliography_audit, args.results_dir / "PETERSON_BIBLIOGRAPHY_AUDIT.md")
    write_artifact_manifest(
        args.results_dir, args.results_dir / "manifests" / "publication_artifacts.json"
    )
    print(f"peterson_readout=OK output={args.results_dir}")


if __name__ == "__main__":
    main()
