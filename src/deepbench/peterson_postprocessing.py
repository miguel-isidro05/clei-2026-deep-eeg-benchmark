"""Publication-oriented, Peterson-only postprocessing.

All inferential units are participants. Seeds are averaged within participant
before hypothesis tests; seed dispersion is reported separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multitest import multipletests

from .statistics import (
    CELL_KEYS,
    FAMILY_KEYS,
    _wilcoxon_pvalue,
    aggregate_seeds,
    mean_ci,
    paired_rank_biserial,
)

IDENTITY_KEYS = CELL_KEYS


def all_metric_descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe every recorded metric without treating seeds as participants."""
    subject_means = aggregate_seeds(frame)
    keys = FAMILY_KEYS + ["model"]
    seed_sd = (
        frame.groupby(keys + ["subject"], observed=True)["value"]
        .std(ddof=1)
        .groupby(keys, observed=True)
        .mean()
    )
    rows: list[dict[str, object]] = []
    for group, values in subject_means.groupby(keys, observed=True):
        array = values["value"].to_numpy(float)
        mean, low, high = mean_ci(array)
        rows.append(
            {
                **dict(zip(keys, group, strict=True)),
                "n_subjects": len(array),
                "mean": mean,
                "sd_between_subjects": float(np.std(array, ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "median": float(np.median(array)),
                "minimum": float(np.min(array)),
                "maximum": float(np.max(array)),
                "mean_within_subject_seed_sd": float(seed_sd.get(group, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def subject_seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in CELL_KEYS if key != "seed"] + ["metric"]
    return (
        frame.groupby(keys, observed=True)["value"]
        .agg(n_seeds="count", mean="mean", sd="std", minimum="min", maximum="max")
        .reset_index()
    )


def friedman_omnibus(frame: pd.DataFrame) -> pd.DataFrame:
    """Exploratory model-family omnibus tests with Kendall's W."""
    subject_means = aggregate_seeds(frame)
    rows: list[dict[str, object]] = []
    for family, values in subject_means.groupby(FAMILY_KEYS, observed=True):
        pivot = values.pivot(index="subject", columns="model", values="value").dropna()
        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            continue
        statistic, p_raw = scipy.stats.friedmanchisquare(
            *(pivot[column].to_numpy(float) for column in pivot.columns)
        )
        rows.append(
            {
                **dict(zip(FAMILY_KEYS, family, strict=True)),
                "n_subjects": pivot.shape[0],
                "n_models": pivot.shape[1],
                "friedman_chi2": float(statistic),
                "kendall_w": float(statistic / (pivot.shape[0] * (pivot.shape[1] - 1))),
                "p_raw": float(p_raw),
                "analysis_tier": "exploratory",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    reject, adjusted, _, _ = multipletests(result["p_raw"], method="holm")
    result["p_holm_global"] = adjusted
    result["reject_holm_0_05"] = reject
    return result


def model_rankings(frame: pd.DataFrame) -> pd.DataFrame:
    subject_means = aggregate_seeds(frame)
    rows: list[dict[str, object]] = []
    for family, values in subject_means.groupby(FAMILY_KEYS, observed=True):
        pivot = values.pivot(index="subject", columns="model", values="value").dropna()
        ranks = pivot.rank(axis=1, ascending=False, method="average")
        for model in pivot.columns:
            rows.append(
                {
                    **dict(zip(FAMILY_KEYS, family, strict=True)),
                    "model": model,
                    "n_subjects": len(pivot),
                    "mean_metric": float(pivot[model].mean()),
                    "mean_rank": float(ranks[model].mean()),
                    "median_rank": float(ranks[model].median()),
                    "subject_wins": int((ranks[model] == 1).sum()),
                }
            )
    return pd.DataFrame(rows)


def _paired_condition_tests(
    frame: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...],
    *,
    column: str,
) -> pd.DataFrame:
    subject_means = aggregate_seeds(frame.loc[frame["metric"].isin(("accuracy", "kappa"))])
    family_keys = ["dataset", "task", "metric"]
    rows: list[dict[str, object]] = []
    for family, values in subject_means.groupby(family_keys, observed=True):
        family_rows: list[dict[str, object]] = []
        for model, model_values in values.groupby("model", observed=True):
            for left, right in pairs:
                a = model_values.loc[model_values[column] == left, ["subject", "value"]]
                b = model_values.loc[model_values[column] == right, ["subject", "value"]]
                if a.empty or b.empty or set(a["subject"]) != set(b["subject"]):
                    continue
                paired = a.merge(b, on="subject", suffixes=("_left", "_right"))
                diff = paired["value_left"].to_numpy() - paired["value_right"].to_numpy()
                mean, low, high = mean_ci(diff)
                family_rows.append(
                    {
                        **dict(zip(family_keys, family, strict=True)),
                        "model": model,
                        "comparison": f"{left}-{right}",
                        "n_subjects": len(diff),
                        "mean_difference": mean,
                        "difference_ci95_low": low,
                        "difference_ci95_high": high,
                        "rank_biserial": paired_rank_biserial(diff),
                        "p_raw": _wilcoxon_pvalue(diff),
                        "analysis_tier": "exploratory",
                    }
                )
        if family_rows:
            reject, adjusted, _, _ = multipletests(
                [row["p_raw"] for row in family_rows], method="holm"
            )
            for row, decision, p_holm in zip(family_rows, reject, adjusted, strict=True):
                row["p_holm"] = float(p_holm)
                row["reject_holm_0_05"] = bool(decision)
            rows.extend(family_rows)
    return pd.DataFrame(rows)


def protocol_gap_tests(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[(frame["condition"] == "overlap") & (frame["ica_policy"] == "none")].copy()
    return _paired_condition_tests(
        selected,
        (("within_split", "loso"), ("within_session", "loso")),
        column="protocol",
    )


def ica_sensitivity_tests(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[
        (frame["protocol"] == "within_split") & (frame["condition"] == "overlap")
    ].copy()
    return _paired_condition_tests(selected, (("kurtosis", "none"),), column="ica_policy")


def prediction_diagnostics(payloads: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        y_true = np.asarray(payload["y_true"], dtype=int)
        y_pred = np.asarray(payload["y_pred"], dtype=int)
        y_score = np.asarray(payload["y_score"], dtype=float)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                **{key: payload[key] for key in IDENTITY_KEYS},
                "n_predictions": len(y_true),
                "true_positive_rate": float(y_true.mean()),
                "predicted_positive_rate": float(y_pred.mean()),
                "mean_predicted_probability": float(y_score.mean()),
                "sensitivity": float(tp / (tp + fn)),
                "specificity": float(tn / (tn + fp)),
                "balanced_accuracy": float(0.5 * (tp / (tp + fn) + tn / (tn + fp))),
                "brier_score": float(np.mean((y_score - y_true) ** 2)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def training_diagnostics(payloads: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        identity = {key: payload[key] for key in IDENTITY_KEYS}
        for fold_index, report in enumerate(payload.get("fold_reports", [])):
            history = report.get("training_history", [])
            if not history:
                continue
            losses = np.asarray([epoch["train_loss"] for epoch in history], dtype=float)
            durations = np.asarray(
                [epoch.get("duration_seconds", np.nan) for epoch in history], dtype=float
            )
            tail = losses[-min(50, len(losses)) :]
            slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) > 1 else np.nan
            rows.append(
                {
                    **identity,
                    "fold_index": fold_index,
                    "epochs_recorded": len(history),
                    "initial_loss": float(losses[0]),
                    "final_loss": float(losses[-1]),
                    "best_loss": float(losses.min()),
                    "best_epoch": int(np.argmin(losses) + 1),
                    "last50_loss_slope": slope,
                    "median_epoch_seconds": float(np.nanmedian(durations)),
                    "total_training_seconds": float(np.nansum(durations)),
                    "optimizer_updates": int(
                        sum(int(epoch.get("train_batch_count", 0)) for epoch in history)
                    ),
                }
            )
    return pd.DataFrame(rows)


def sample_accounting_summary(sample_accounting: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "task", "protocol", "condition", "ica_policy", "model"]
    numeric = [
        "n_train_trials",
        "n_train_examples",
        "n_test_trials",
        "optimizer_updates",
    ]
    return (
        sample_accounting.groupby(keys, observed=True)[numeric]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )


def review_closure_table(*, has_latency: bool = False) -> pd.DataFrame:
    latency_row = (
        (
            "latency",
            "partially_resolved",
            "Model-forward latency is reported.",
            "It is not end-to-end online BCI latency.",
        )
        if has_latency
        else (
            "latency",
            "pending_measurement",
            "No latency artifact is present in this result directory.",
            "Run the finalizer on the target GPU with PETERSON_LATENCY_DEVICE set.",
        )
    )
    rows = [
        (
            "paired_statistics",
            "resolved",
            "Paired participant-level Wilcoxon tests with Holm correction.",
            "",
        ),
        ("multiseed", "resolved", "Five seeds; subject and seed variability are separated.", ""),
        (
            "temporal_support",
            "resolved",
            "Full, center, non-overlap, overlap, and compute-matched controls are audited.",
            "",
        ),
        (
            "augmentation_compute",
            "resolved",
            "Matched examples and optimizer updates are checked before inference.",
            "",
        ),
        (
            "ica_sensitivity",
            "resolved",
            "Kurtosis ICA is compared with no ICA on paired subjects.",
            "",
        ),
        (
            "trial_accounting",
            "resolved",
            "Fold-level trials, windows, class counts, and optimizer updates are exported.",
            "",
        ),
        (
            "roc_aggregation",
            "resolved",
            "ROC curves are macro-averaged over subjects, after seed averaging.",
            "",
        ),
        (
            "convergence",
            "resolved_by_design",
            "All deep folds use the frozen 300-epoch budget and export training-loss diagnostics.",
            "No validation curve or validation-based epoch selection was used; held-out partitions "
            "were never used for model selection.",
        ),
        latency_row,
        (
            "bibliography",
            "partially_resolved",
            "Primary architecture and analysis sources were DOI-verified in "
            "PETERSON_BIBLIOGRAPHY_AUDIT.md.",
            "The later manuscript pass must add missing identifiers and remove "
            "references to abandoned experiments.",
        ),
        (
            "external_low_cost_validation",
            "unresolved_by_design",
            "No second dataset is claimed in the Peterson-only paper.",
            "The evidence remains a single low-cost dataset with ten participants.",
        ),
        (
            "hardware_causality",
            "unresolved_by_design",
            "The study benchmarks decoding on low-cost recordings.",
            "It does not isolate hardware quality as a causal factor.",
        ),
    ]
    return pd.DataFrame(rows, columns=["item_id", "status", "evidence", "limitation"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_artifact_manifest(root: Path, output: Path) -> None:
    excluded = ("fold_cache/**", "logs/**")
    excluded_roots = {"fold_cache", "logs"}
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            path.is_file()
            and path != output
            and relative.parts[0] not in excluded_roots
        ):
            files.append(
                {
                    "path": str(relative),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "stable publication artifacts",
        "excluded": list(excluded),
        "files": files,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
