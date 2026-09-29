#!/usr/bin/env python3
"""Render validated result tables and figure includes into the active manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from deepbench.config import MODEL_NAMES, RESULTS_DIR
from deepbench.figures import figures_code_sha256
from deepbench.latency import latency_code_sha256
from deepbench.paper_audit import audit_expected_cells, result_cells_sha256
from deepbench.statistics import statistics_code_sha256


def _number(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def _performance_rows(table: pd.DataFrame, datasets: set[str]) -> list[str]:
    selected = table.loc[
        table["dataset"].isin(datasets)
        & (table["condition"] == "full")
        & (table["ica_policy"] == "none")
    ]
    indexed = selected.set_index(["dataset", "protocol", "model", "metric"])
    rows: list[str] = []
    groups = selected[["dataset", "protocol", "model"]].drop_duplicates()
    for group in groups.itertuples(index=False):
        accuracy = indexed.loc[(group.dataset, group.protocol, group.model, "accuracy")]
        kappa = indexed.loc[(group.dataset, group.protocol, group.model, "kappa")]
        rows.append(
            f"{_escape(group.dataset)} & {_escape(group.protocol)} & {_escape(group.model)} & "
            f"{_number(accuracy['mean'])} $\\pm$ {_number(accuracy['sd_between_subjects'])} & "
            f"{_number(kappa['mean'])} $\\pm$ {_number(kappa['sd_between_subjects'])} \\\\"
        )
    return rows


def _table_block(caption: str, label: str, rows: list[str]) -> list[str]:
    return [
        r"\begin{table*}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllcc}",
        r"\toprule",
        r"Dataset & Protocol & Model & Held-out test accuracy & Held-out test Cohen's $\kappa$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]


def _verify_file_hash(path: Path, expected: object, artifact: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    if actual is None or actual != expected:
        raise SystemExit(f"Manuscript generation blocked: invalid {artifact} {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/revision_v2/generated_results.tex"),
    )
    args = parser.parse_args()
    _audit, confirmatory, issues = audit_expected_cells(args.results_dir)
    if not confirmatory:
        raise SystemExit("Manuscript generation blocked: " + "; ".join(issues))
    statistics = args.results_dir / "statistics"
    statistics_manifest = json.loads(
        (statistics / "statistics_manifest.json").read_text(encoding="utf-8")
    )
    if statistics_manifest.get("result_cells_sha256") != result_cells_sha256(args.results_dir):
        raise SystemExit("Manuscript generation blocked: statistics are stale")
    if statistics_manifest.get("statistics_code_sha256") != statistics_code_sha256():
        raise SystemExit("Manuscript generation blocked: statistics code has changed")
    table_names = (
        "descriptive_subject_seed_variability.csv",
        "paired_wilcoxon_holm.csv",
        "augmentation_wilcoxon_holm.csv",
    )
    recorded_tables = statistics_manifest.get("statistics_files", {})
    for name in table_names:
        _verify_file_hash(statistics / name, recorded_tables.get(name), "statistics table")
    descriptive = pd.read_csv(statistics / table_names[0])
    paired = pd.read_csv(statistics / table_names[1])
    augmentation = pd.read_csv(statistics / table_names[2])
    latency_path = args.results_dir / "latency" / "latency.csv"
    latency_manifest_path = args.results_dir / "latency" / "latency_manifest.json"
    if not latency_manifest_path.exists():
        raise SystemExit("Manuscript generation blocked: latency_manifest.json is missing")
    latency_manifest = json.loads(latency_manifest_path.read_text(encoding="utf-8"))
    if latency_manifest.get("latency_code_sha256") != latency_code_sha256():
        raise SystemExit("Manuscript generation blocked: latency code has changed")
    _verify_file_hash(latency_path, latency_manifest.get("latency_csv_sha256"), "latency table")
    latency = pd.read_csv(latency_path)
    figure_dir = os.path.relpath(args.results_dir / "figures", args.output.parent)
    figures_manifest_path = args.results_dir / "figures" / "figures_manifest.json"
    if not figures_manifest_path.exists():
        raise SystemExit("Manuscript generation blocked: figures_manifest.json is missing")
    figures_manifest = json.loads(figures_manifest_path.read_text(encoding="utf-8"))
    if figures_manifest.get("result_cells_sha256") != result_cells_sha256(args.results_dir):
        raise SystemExit("Manuscript generation blocked: figures are stale")
    if figures_manifest.get("figures_code_sha256") != figures_code_sha256():
        raise SystemExit("Manuscript generation blocked: figure code has changed")
    lines = [
        r"\subsection{Decoder performance}",
        "",
        "Tables~\\ref{tab:peterson-results} and~\\ref{tab:souza-results} report "
        "participant-level mean $\\pm$ standard deviation after averaging the five optimization "
        "seeds within participant. The primary paired inference follows in "
        "Table~\\ref{tab:primary-paired-results}; all protocol-specific contrasts remain in "
        "the generated CSV artifacts.",
        "",
        *_table_block(
            "MI-OpenBCI full-trial performance. Values are participant-level mean $\\pm$ SD.",
            "tab:peterson-results",
            _performance_rows(descriptive, {"MI-OpenBCI"}),
        ),
        *_table_block(
            "Souza2023 full-trial performance. Values are participant-level mean $\\pm$ SD.",
            "tab:souza-results",
            _performance_rows(descriptive, {"Souza2023"}),
        ),
        r"\subsection{Primary paired model comparisons}",
        "",
        "Table~\\ref{tab:primary-paired-results} reports the prespecified held-out test "
        "accuracy contrasts for each low-cost dataset under the stratified five-fold "
        "within-session protocol. Each inferential family remains dataset-specific. Effect "
        "sizes are paired rank-biserial correlations with "
        "the sign of model A minus model B.",
        "",
        r"\begin{table*}[t]",
        r"\caption{Primary paired Wilcoxon comparisons with Holm correction.}",
        r"\label{tab:primary-paired-results}",
        r"\centering\small",
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        "Dataset & Model A & Model B & $n$ & Mean diff. & 95\\% CI low & 95\\% CI high & "
        "$r_{rb}$ & $p_{raw}$ & $p_{Holm}$ & Reject \\\\",
        r"\midrule",
    ]
    primary_paired = paired.loc[
        paired["dataset"].isin(["MI-OpenBCI", "Souza2023"])
        & (paired["protocol"] == "within_session")
        & (paired["condition"] == "full")
        & (paired["ica_policy"] == "none")
        & (paired["metric"] == "accuracy")
    ]
    if len(primary_paired) != 20:
        raise SystemExit(
            "Manuscript generation blocked: expected 20 low-cost pairwise model comparisons"
        )
    for row in primary_paired.itertuples(index=False):
        lines.append(
            f"{_escape(row.dataset)} & {_escape(row.model_a)} & {_escape(row.model_b)} & "
            f"{int(row.n_subjects)} & "
            f"{_number(row.mean_difference_a_minus_b)} & "
            f"{_number(row.difference_ci95_low)} & {_number(row.difference_ci95_high)} & "
            f"{_number(row.rank_biserial_a_minus_b)} & {_number(row.p_raw, 4)} & "
            f"{_number(row.p_holm, 4)} & "
            f"{'yes' if row.reject_holm_0_05 else 'no'} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\subsection{Compute-matched augmentation}",
            "",
            r"\begin{table*}[t]",
            r"\caption{Compute-matched augmentation contrasts after averaging seeds "
            r"within participant.}",
            r"\label{tab:augmentation-results}",
            r"\centering\small",
            r"\begin{tabular}{llllrrrrrrrr}",
            r"\toprule",
            "Dataset & Metric & Model & Contrast & $n$ & Mean diff. & 95\\% CI low & "
            "95\\% CI high & "
            "$r_{rb}$ & $p_{raw}$ & $p_{Holm}$ & Reject \\\\",
            r"\midrule",
        ]
    )
    for row in augmentation.itertuples(index=False):
        lines.append(
            f"{_escape(row.dataset)} & {_escape(row.metric)} & {_escape(row.model)} & "
            f"{_escape(row.comparison)} & "
            f"{int(row.n_subjects)} & "
            f"{_number(row.mean_difference)} & {_number(row.difference_ci95_low)} & "
            f"{_number(row.difference_ci95_high)} & "
            f"{_number(row.rank_biserial_augmented_minus_center)} & "
            f"{_number(row.p_raw, 4)} & {_number(row.p_holm, 4)} & "
            f"{'yes' if row.reject_holm_0_05 else 'no'} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\subsection{Model-forward latency}",
            "",
            r"\begin{table}[t]",
            r"\caption{Batch-one model-forward latency on the recorded execution device.}",
            r"\label{tab:latency-results}",
            r"\centering\small",
            r"\begin{tabular}{llrr}",
            r"\toprule",
            r"Dataset & Model & Median (ms) & P95 (ms) \\",
            r"\midrule",
        ]
    )
    batch_one = latency.loc[latency["batch_size"] == 1].set_index(["dataset", "model"])
    for dataset in ("MI-OpenBCI", "Souza2023"):
        for model in MODEL_NAMES:
            row = batch_one.loc[(dataset, model)]
            lines.append(
                f"{_escape(dataset)} & {_escape(model)} & "
                f"{_number(row['median_batch_ms'])} & {_number(row['p95_batch_ms'])} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    for stem, caption, label in (
        (
            "performance_overview_accuracy",
            "Accuracy across datasets and protocols.",
            "fig:accuracy",
        ),
        (
            "performance_overview_kappa",
            "Cohen's $\\kappa$ across datasets and protocols.",
            "fig:kappa",
        ),
        ("augmentation_effects", "Compute-matched augmentation effects.", "fig:augmentation"),
        (
            "optimization_seed_variability",
            "Optimization variability across five seeds.",
            "fig:seeds",
        ),
        ("ica_sensitivity", "Exploratory train-only ICA sensitivity.", "fig:ica"),
        ("model_inference_latency", "Model-forward latency.", "fig:latency"),
        (
            "primary_confusion_matrices",
            "Primary MI-OpenBCI five-fold confusion matrices.",
            "fig:confusion",
        ),
        (
            "souza_accuracy",
            "Souza2023 participant-level accuracy by protocol.",
            "fig:souza-accuracy",
        ),
        (
            "souza_confusion_matrices",
            "Souza2023 five-fold left-versus-right confusion matrices.",
            "fig:souza-confusion",
        ),
    ):
        figure_path = args.results_dir / "figures" / f"{stem}.pdf"
        recorded_hash = figures_manifest.get("figures", {}).get(figure_path.name)
        _verify_file_hash(figure_path, recorded_hash, "figure")
        lines.extend(
            [
                r"\begin{figure*}[t]",
                r"\centering",
                f"\\includegraphics[width=0.95\\textwidth]{{{figure_dir}/{stem}.pdf}}",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                r"\end{figure*}",
                "",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Manuscript results written to {args.output}")


if __name__ == "__main__":
    main()
