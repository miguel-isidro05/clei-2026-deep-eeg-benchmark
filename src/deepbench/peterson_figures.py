"""Peterson-only publication figures with participant-level aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from .config import PETERSON_MODEL_NAMES
from .statistics import aggregate_seeds

PROTOCOLS = ("within_split", "within_session", "loso")
MODEL_ORDER = list(PETERSON_MODEL_NAMES)
MODEL_COLORS = {
    "CSP+LDA": "#6B7280",
    "EEGNet": "#0072B2",
    "FBCNet": "#009E73",
    "ShallowConvNet": "#E69F00",
    "EEGConformer": "#CC79A7",
}
PROTOCOL_LABELS = {
    "within_split": "Within-split",
    "within_session": "Within-session",
    "loso": "LOSO",
}


def _set_publication_style() -> None:
    """Apply a compact, colorblind-safe style suitable for two-column papers."""
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "grid.color": "#D1D5DB",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
        }
    )


def _format_protocol_axis(ax: plt.Axes) -> None:
    labels = [
        PROTOCOL_LABELS.get(str(label.get_text()), label.get_text())
        for label in ax.get_xticklabels()
    ]
    ax.set_xticks(ax.get_xticks(), labels)


def _forest_labels(data: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for row in data.itertuples():
        protocol = PROTOCOL_LABELS.get(str(row.protocol), str(row.protocol))
        if hasattr(row, "comparison"):
            comparison = str(row.comparison).replace("-", " − ").replace("_x", " ×")
            labels.append(f"{row.model}: {comparison} ({protocol})")
        else:
            labels.append(f"{row.model_a} − {row.model_b} ({protocol})")
    return labels


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _primary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return aggregate_seeds(
        frame.loc[
            (frame["condition"] == "overlap")
            & (frame["ica_policy"] == "none")
            & (frame["metric"] == metric)
        ]
    )


def _performance(frame: pd.DataFrame, output_dir: Path, metric: str) -> None:
    data = _primary(frame, metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(
        data=data,
        x="protocol",
        y="value",
        hue="model",
        order=PROTOCOLS,
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        errorbar=("ci", 95),
        dodge=0.35,
        ax=ax,
    )
    ax.axhline(0.5 if metric == "accuracy" else 0.0, color="0.4", ls="--", lw=1)
    ax.set(xlabel="Protocol", ylabel=metric.replace("_", " ").title())
    _format_protocol_axis(ax)
    ax.legend(ncol=3, fontsize=8, title=None)
    _save(fig, output_dir, f"primary_{metric}")


def _subject_heatmap(frame: pd.DataFrame, output_dir: Path, metric: str) -> None:
    data = _primary(frame, metric)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, protocol in zip(axes, PROTOCOLS, strict=True):
        pivot = data.loc[data["protocol"] == protocol].pivot(
            index="subject", columns="model", values="value"
        )
        pivot = pivot.reindex(columns=MODEL_ORDER)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
        ax.set_title(PROTOCOL_LABELS[protocol])
        ax.set(xlabel="", ylabel="Participant" if ax is axes[0] else "")
    _save(fig, output_dir, f"subject_{metric}_heatmaps")


def _condition_heatmap(frame: pd.DataFrame, output_dir: Path, metric: str) -> None:
    data = aggregate_seeds(frame.loc[(frame["ica_policy"] == "none") & (frame["metric"] == metric)])
    summary = data.groupby(["protocol", "condition", "model"], observed=True)["value"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, protocol in zip(axes, PROTOCOLS, strict=True):
        pivot = summary.loc[protocol].unstack("model").reindex(columns=MODEL_ORDER)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako", vmin=0.45, vmax=0.9, ax=ax)
        ax.set_title(PROTOCOL_LABELS[protocol])
        ax.set(xlabel="", ylabel="Temporal condition")
    _save(fig, output_dir, f"condition_{metric}_heatmaps")


def _forest(table: pd.DataFrame, output_dir: Path, metric: str, stem: str) -> None:
    data = table.loc[table["metric"] == metric].reset_index(drop=True)
    if data.empty:
        return
    labels = _forest_labels(data)
    mean_col = "mean_difference" if "mean_difference" in data else "mean_difference_a_minus_b"
    low_col = "difference_ci95_low"
    high_col = "difference_ci95_high"
    y = np.arange(len(data))
    means = data[mean_col].to_numpy()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(data))))
    significant = (
        data["reject_holm_0_05"].astype(bool).to_numpy()
        if "reject_holm_0_05" in data
        else np.zeros(len(data), dtype=bool)
    )
    errors = np.vstack((means - data[low_col], data[high_col] - means))
    for is_significant, color, label in (
        (False, "#0072B2", "Not significant"),
        (True, "#D55E00", "Holm-adjusted p < 0.05"),
    ):
        mask = significant == is_significant
        if mask.any():
            ax.errorbar(
                means[mask],
                y[mask],
                xerr=errors[:, mask],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2.5,
                label=label,
            )
    ax.axvline(0, color="0.3", ls="--")
    ax.set(yticks=y, yticklabels=labels, xlabel=f"Paired difference in {metric}")
    ax.legend(loc="lower right")
    _save(fig, output_dir, f"{stem}_{metric}_forest")


def _confusion(payloads: list[dict[str, object]], output_dir: Path, protocol: str) -> None:
    rows = []
    for payload in payloads:
        if (
            payload["protocol"] != protocol
            or payload["condition"] != "overlap"
            or payload["ica_policy"] != "none"
        ):
            continue
        y_true = np.asarray(payload["y_true"], int)
        y_pred = np.asarray(payload["y_pred"], int)
        for true_class in (0, 1):
            mask = y_true == true_class
            for predicted_class in (0, 1):
                rows.append(
                    {
                        "model": payload["model"],
                        "subject": payload["subject"],
                        "seed": payload["seed"],
                        "true": true_class,
                        "predicted": predicted_class,
                        "rate": float(np.mean(y_pred[mask] == predicted_class)),
                    }
                )
    data = (
        pd.DataFrame(rows)
        .groupby(["model", "subject", "true", "predicted"], observed=True)["rate"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        matrix = (
            data.loc[data["model"] == model].groupby(["true", "predicted"])["rate"].mean().unstack()
        )
        sns.heatmap(matrix, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", cbar=False, ax=ax)
        ax.set_title(model)
    _save(fig, output_dir, f"confusion_{protocol}")


def _seed_averaged_predictions(
    payloads: list[dict[str, object]], protocol: str, model: str, subject: str
) -> tuple[np.ndarray, np.ndarray] | None:
    selected = [
        payload
        for payload in payloads
        if payload["protocol"] == protocol
        and payload["condition"] == "overlap"
        and payload["ica_policy"] == "none"
        and payload["model"] == model
        and payload["subject"] == subject
    ]
    if not selected:
        return None
    truths = [np.asarray(payload["y_true"], dtype=int) for payload in selected]
    if any(not np.array_equal(truths[0], truth) for truth in truths[1:]):
        raise ValueError(f"Seed prediction order differs for {protocol}/{model}/{subject}")
    scores = np.mean([np.asarray(payload["y_score"], dtype=float) for payload in selected], axis=0)
    return truths[0], scores


def _roc(payloads: list[dict[str, object]], output_dir: Path, protocol: str) -> None:
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(6, 5))
    for model in MODEL_ORDER:
        subject_curves = []
        for subject in sorted({str(p["subject"]) for p in payloads}):
            predictions = _seed_averaged_predictions(payloads, protocol, model, subject)
            if predictions is not None:
                y_true, y_score = predictions
                fpr, tpr, _ = roc_curve(y_true, y_score)
                subject_curves.append(np.interp(grid, fpr, tpr))
        curves = np.asarray(subject_curves)
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
        ax.plot(grid, mean, label=model, color=MODEL_COLORS[model])
        ax.fill_between(
            grid,
            np.clip(mean - 1.96 * sem, 0, 1),
            np.clip(mean + 1.96 * sem, 0, 1),
            color=MODEL_COLORS[model],
            alpha=0.12,
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title=f"Subject-macro ROC: {PROTOCOL_LABELS[protocol]}",
    )
    ax.legend(fontsize=8)
    _save(fig, output_dir, f"roc_subject_macro_{protocol}")


def _calibration(payloads: list[dict[str, object]], output_dir: Path) -> None:
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(6, 5))
    for model in MODEL_ORDER:
        subject_x, subject_y = [], []
        for subject in sorted({str(p["subject"]) for p in payloads}):
            predictions = _seed_averaged_predictions(payloads, "loso", model, subject)
            if predictions is None:
                continue
            y_true, y_score = predictions
            bin_index = np.clip(np.digitize(y_score, bins) - 1, 0, len(bins) - 2)
            x_values = np.full(len(bins) - 1, np.nan)
            y_values = np.full(len(bins) - 1, np.nan)
            for index in range(len(bins) - 1):
                mask = bin_index == index
                if mask.any():
                    x_values[index] = y_score[mask].mean()
                    y_values[index] = y_true[mask].mean()
            subject_x.append(x_values)
            subject_y.append(y_values)
        x = np.nanmean(subject_x, axis=0)
        y = np.nanmean(subject_y, axis=0)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.plot(x[valid], y[valid], "o-", ms=4, label=model, color=MODEL_COLORS[model])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="LOSO calibration diagnostic",
    )
    ax.legend(fontsize=8)
    _save(fig, output_dir, "loso_calibration")


def _ica_sensitivity(statistics: Path, output_dir: Path) -> None:
    data = pd.read_csv(statistics / "ica_sensitivity_wilcoxon_holm.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric in zip(axes, ("accuracy", "kappa"), strict=True):
        selected = data.loc[data["metric"] == metric].set_index("model").reindex(MODEL_ORDER)
        mean = selected["mean_difference"].to_numpy()
        ax.errorbar(
            mean,
            np.arange(len(selected)),
            xerr=np.vstack(
                (mean - selected["difference_ci95_low"], selected["difference_ci95_high"] - mean)
            ),
            fmt="o",
            capsize=3,
        )
        ax.axvline(0, color="0.3", ls="--")
        ax.set(title=metric.title(), xlabel="Kurtosis ICA minus no ICA")
        ax.set_yticks(np.arange(len(selected)), selected.index)
    _save(fig, output_dir, "ica_sensitivity")


def _rankings(statistics: Path, output_dir: Path) -> None:
    data = pd.read_csv(statistics / "model_rankings.csv")
    data = data.loc[
        (data["condition"] == "overlap")
        & (data["ica_policy"] == "none")
        & (data["metric"] == "accuracy")
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=data,
        x="protocol",
        y="mean_rank",
        hue="model",
        order=PROTOCOLS,
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.set(ylabel="Mean participant rank (1 = best)", xlabel="Protocol")
    _format_protocol_axis(ax)
    ax.legend(ncol=3, fontsize=8, title=None)
    _save(fig, output_dir, "model_rankings")


def _sample_accounting(statistics: Path, output_dir: Path) -> None:
    data = pd.read_csv(statistics / "sample_accounting.csv")
    data = data.loc[(data["condition"] == "overlap") & (data["ica_policy"] == "none")]
    summary = (
        data.groupby(["protocol", "model"], observed=True)["n_train_examples"].mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=summary,
        x="protocol",
        y="n_train_examples",
        hue="model",
        order=PROTOCOLS,
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.set(ylabel="Mean training examples per fold", xlabel="Protocol")
    _format_protocol_axis(ax)
    ax.legend(ncol=3, fontsize=8, title=None)
    _save(fig, output_dir, "sample_accounting")


def _seed_variability(frame: pd.DataFrame, output_dir: Path) -> None:
    data = frame.loc[
        (frame["condition"] == "overlap")
        & (frame["ica_policy"] == "none")
        & (frame["metric"] == "accuracy")
    ]
    seed_sd = (
        data.groupby(["protocol", "model", "subject"], observed=True)["value"]
        .std()
        .reset_index(name="seed_sd")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=seed_sd,
        x="model",
        y="seed_sd",
        hue="protocol",
        order=MODEL_ORDER,
        palette=("#0072B2", "#009E73", "#E69F00"),
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=25)
    ax.set(xlabel="", ylabel="Within-participant SD across seeds")
    _save(fig, output_dir, "seed_variability")


def _training_curves(payloads: list[dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, protocol in zip(axes, PROTOCOLS, strict=True):
        for model in [m for m in MODEL_ORDER if m != "CSP+LDA"]:
            curves = []
            for payload in payloads:
                if (
                    payload["protocol"] == protocol
                    and payload["condition"] == "overlap"
                    and payload["ica_policy"] == "none"
                    and payload["model"] == model
                ):
                    for report in payload["fold_reports"]:
                        curves.append([e["train_loss"] for e in report["training_history"]])
            values = np.asarray(curves)
            ax.plot(
                np.arange(1, values.shape[1] + 1),
                np.median(values, axis=0),
                label=model,
                color=MODEL_COLORS[model],
            )
        ax.set(title=PROTOCOL_LABELS[protocol], xlabel="Epoch")
    axes[0].set_ylabel("Median training loss")
    axes[-1].legend(fontsize=8)
    _save(fig, output_dir, "training_convergence")


def _quality(results_dir: Path, output_dir: Path) -> None:
    channel_path = results_dir / "quality" / "signal_quality_channels.csv"
    if channel_path.exists():
        data = pd.read_csv(channel_path)
        pivot = data.pivot(index="subject", columns="channel", values="rms_to_channel_median")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(pivot, cmap="coolwarm", center=1, ax=ax)
        ax.set(
            title="Channel RMS relative to participant median",
            xlabel="Channel",
            ylabel="Participant",
        )
        _save(fig, output_dir, "signal_quality_rms")
    erd_path = results_dir / "quality" / "erd_ers.json"
    if erd_path.exists():
        payload = json.loads(erd_path.read_text())["peterson"]
        rows = []
        for subject, subject_data in payload.items():
            for band in ("mu", "beta"):
                for channel in ("C3", "Cz", "C4"):
                    rows.append(
                        {
                            "subject": subject,
                            "feature": f"{band}-{channel}",
                            "value": subject_data[band][channel],
                        }
                    )
        pivot = pd.DataFrame(rows).pivot(index="subject", columns="feature", values="value")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
        ax.set(title="Motor imagery minus rest log power", xlabel="", ylabel="Participant")
        _save(fig, output_dir, "erd_ers")


def _latency(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "latency" / "latency.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    data = data.loc[data["batch_size"] == 1].set_index("model").reindex(MODEL_ORDER[1:])
    data = data.dropna(subset=["median_batch_ms"])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(data))
    values = data["median_batch_ms"].to_numpy(float)
    ax.bar(x, values, color=[MODEL_COLORS[model] for model in data.index])
    spread_column = "median_batch_ms_between_gpu_sd"
    if spread_column in data:
        ax.errorbar(
            x,
            values,
            yerr=data[spread_column].fillna(0).to_numpy(float),
            fmt="none",
            color="black",
            capsize=3,
        )
    ax.set(
        xticks=x,
        xticklabels=data.index,
        ylabel="Median model-forward latency (ms)",
        title="Batch-one latency; median across GPUs",
    )
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output_dir, "model_forward_latency")


def generate_peterson_figures(results_dir: Path) -> list[Path]:
    statistics = results_dir / "statistics"
    output_dir = results_dir / "figures"
    frame = pd.read_csv(statistics / "all_seed_metrics.csv")
    payloads = [
        json.loads(path.read_text()) for path in sorted((results_dir / "cells").glob("*.json"))
    ]
    _set_publication_style()
    for metric in ("accuracy", "kappa"):
        _performance(frame, output_dir, metric)
        _subject_heatmap(frame, output_dir, metric)
        _condition_heatmap(frame, output_dir, metric)
    augmentation = pd.read_csv(statistics / "augmentation_wilcoxon_holm.csv")
    paired = pd.read_csv(statistics / "paired_wilcoxon_holm.csv")
    for metric in ("accuracy", "kappa"):
        _forest(augmentation, output_dir, metric, "augmentation")
        primary_pairs = paired.loc[
            (paired["condition"] == "overlap") & (paired["ica_policy"] == "none")
        ]
        _forest(primary_pairs, output_dir, metric, "model_comparison")
    _seed_variability(frame, output_dir)
    for protocol in PROTOCOLS:
        _confusion(payloads, output_dir, protocol)
        _roc(payloads, output_dir, protocol)
    _calibration(payloads, output_dir)
    _training_curves(payloads, output_dir)
    _ica_sensitivity(statistics, output_dir)
    _rankings(statistics, output_dir)
    _sample_accounting(statistics, output_dir)
    _quality(results_dir, output_dir)
    _latency(results_dir, output_dir)
    return sorted(output_dir.glob("*"))
