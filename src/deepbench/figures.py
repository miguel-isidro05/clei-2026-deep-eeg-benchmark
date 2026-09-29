"""Publication figure generation from integrity-checked result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import MODEL_NAMES
from .statistics import mean_ci


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_primary_accuracy(statistics_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(statistics_dir / "descriptive_subject_seed_variability.csv")
    selected = table.loc[
        (table["dataset"] == "MI-OpenBCI")
        & (table["condition"] == "full")
        & (table["ica_policy"] == "none")
        & (table["metric"] == "accuracy")
    ].copy()
    protocols = [
        value
        for value in ("within_split", "within_session", "loso")
        if value in set(selected["protocol"])
    ]
    figure, axes = plt.subplots(1, len(protocols), figsize=(4.5 * len(protocols), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, protocol in zip(axes, protocols, strict=True):
        values = (
            selected.loc[selected["protocol"] == protocol].set_index("model").reindex(MODEL_NAMES)
        )
        means = values["mean"].to_numpy(float)
        lower = means - values["ci95_low"].to_numpy(float)
        upper = values["ci95_high"].to_numpy(float) - means
        axis.bar(
            np.arange(len(MODEL_NAMES)),
            means,
            color="#4472C4",
            yerr=np.vstack([lower, upper]),
            capsize=3,
        )
        axis.set_title(protocol.replace("_", " "))
        axis.set_xticks(np.arange(len(MODEL_NAMES)), MODEL_NAMES, rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Accuracy")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("MI-OpenBCI: subject-level mean accuracy and 95% t confidence interval")
    _save_figure(figure, output_dir, "primary_accuracy")


def plot_augmentation_effects(statistics_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(statistics_dir / "augmentation_wilcoxon_holm.csv")
    selected = table.loc[(table["metric"] == "accuracy") & (table["dataset"] == "MI-OpenBCI")]
    figure, axis = plt.subplots(figsize=(9, 5))
    labels = [f"{row.model}\n{row.comparison}" for row in selected.itertuples()]
    means = selected["mean_difference"].to_numpy(float)
    lower = means - selected["difference_ci95_low"].to_numpy(float)
    upper = selected["difference_ci95_high"].to_numpy(float) - means
    axis.errorbar(np.arange(len(means)), means, yerr=np.vstack([lower, upper]), fmt="o", capsize=3)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_ylabel("Paired accuracy difference")
    axis.set_title("Compute-matched window augmentation effects")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir, "augmentation_effects")


def plot_performance_overview(statistics_dir: Path, output_dir: Path, metric: str) -> None:
    """Show subject points and seed-averaged means for all main protocols."""
    table = pd.read_csv(statistics_dir / "all_seed_metrics.csv")
    selected = table.loc[
        (table["condition"] == "full")
        & (table["ica_policy"] == "none")
        & (table["metric"] == metric)
    ]
    subjects = (
        selected.groupby(
            ["dataset", "protocol", "model", "subject"], as_index=False, observed=True
        )["value"]
        .mean()
        .sort_values(["dataset", "protocol", "model", "subject"])
    )
    groups = list(subjects[["dataset", "protocol"]].drop_duplicates().itertuples(index=False))
    columns = 3
    rows = int(np.ceil(len(groups) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows), sharey=True)
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, group in zip(flat_axes, groups, strict=False):
        values = subjects.loc[
            (subjects["dataset"] == group.dataset)
            & (subjects["protocol"] == group.protocol)
        ]
        for model_index, model in enumerate(MODEL_NAMES):
            model_values = values.loc[values["model"] == model, "value"].to_numpy(float)
            jitter = np.linspace(-0.08, 0.08, len(model_values)) if len(model_values) > 1 else [0]
            axis.scatter(
                model_index + np.asarray(jitter),
                model_values,
                color="#4472C4",
                alpha=0.55,
                s=18,
            )
            mean, low, high = mean_ci(model_values)
            axis.errorbar(
                model_index,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                color="black",
                capsize=3,
            )
        axis.set_title(f"{group.dataset}: {group.protocol.replace('_', ' ')}")
        axis.set_xticks(range(len(MODEL_NAMES)), MODEL_NAMES, rotation=35, ha="right")
        axis.set_ylim(-0.05 if metric == "kappa" else 0, 1)
        axis.set_ylabel(metric.capitalize())
        axis.grid(axis="y", alpha=0.25)
    for axis in flat_axes[len(groups) :]:
        axis.set_visible(False)
    figure.suptitle(f"Subject-level {metric}: points, mean, and 95% t confidence interval")
    _save_figure(figure, output_dir, f"performance_overview_{metric}")


def plot_seed_variability(statistics_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(statistics_dir / "descriptive_subject_seed_variability.csv")
    selected = table.loc[(table["condition"] == "full") & (table["ica_policy"] == "none")]
    groups = list(
        selected[["dataset", "protocol", "metric"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    figure, axis = plt.subplots(figsize=(12, max(5, len(groups) * 0.35)))
    for index, group in enumerate(groups):
        values = selected.loc[
            (selected["dataset"] == group.dataset)
            & (selected["protocol"] == group.protocol)
            & (selected["metric"] == group.metric)
        ].set_index("model")
        for model_index, model in enumerate(MODEL_NAMES):
            axis.scatter(
                float(values.loc[model, "mean_within_subject_seed_sd"]),
                index + (model_index - 2) * 0.08,
                label=model if index == 0 else None,
            )
    axis.set_yticks(
        range(len(groups)),
        [f"{g.dataset} | {g.protocol} | {g.metric}" for g in groups],
    )
    axis.set_xlabel("Mean within-subject standard deviation across seeds")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    _save_figure(figure, output_dir, "optimization_seed_variability")


def plot_ica_sensitivity(statistics_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(statistics_dir / "all_seed_metrics.csv")
    selected = table.loc[
        (table["dataset"] == "MI-OpenBCI")
        & (table["protocol"] == "within_split")
        & (table["condition"] == "full")
    ]
    subject_means = selected.groupby(
        ["metric", "model", "subject", "ica_policy"], observed=True
    )["value"].mean()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for axis, metric in zip(axes, ("accuracy", "kappa"), strict=True):
        metric_values = subject_means.loc[metric].unstack("ica_policy")
        for model_index, model in enumerate(MODEL_NAMES):
            differences = (
                metric_values.loc[model, "kurtosis"] - metric_values.loc[model, "none"]
            ).to_numpy(float)
            jitter = np.linspace(-0.08, 0.08, len(differences))
            axis.scatter(model_index + jitter, differences, alpha=0.55, s=18)
            mean, low, high = mean_ci(differences)
            axis.errorbar(
                model_index,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                color="black",
                capsize=3,
            )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(metric.capitalize())
        axis.set_xticks(range(len(MODEL_NAMES)), MODEL_NAMES, rotation=35, ha="right")
        axis.set_ylabel("Kurtosis ICA minus no ICA")
        axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir, "ica_sensitivity")


def plot_latency(results_dir: Path, output_dir: Path) -> None:
    table = pd.read_csv(results_dir / "latency" / "latency.csv")
    batch_one = table.loc[table["batch_size"] == 1].set_index("model").reindex(MODEL_NAMES)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(MODEL_NAMES, batch_one["median_batch_ms"], color="#70AD47")
    axis.scatter(MODEL_NAMES, batch_one["p95_batch_ms"], color="black", label="p95", zorder=3)
    axis.set_ylabel("Forward-pass latency (ms)")
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir, "model_inference_latency")


def plot_primary_confusion_matrices(cells_dir: Path, output_dir: Path) -> None:
    by_model: dict[str, list[np.ndarray]] = {model: [] for model in MODEL_NAMES}
    by_subject_seed: dict[tuple[str, str, int], list[np.ndarray]] = {}
    for path in sorted(cells_dir.glob("MI-OpenBCI__within_split__full__ica-none__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (str(payload["model"]), str(payload["subject"]), int(payload["seed"]))
        matrix = confusion_matrix(
            payload["y_true"], payload["y_pred"], labels=[0, 1], normalize="true"
        )
        by_subject_seed.setdefault(key, []).append(matrix)
    by_subject: dict[tuple[str, str], list[np.ndarray]] = {}
    for (model, subject, _seed), matrices in by_subject_seed.items():
        by_subject.setdefault((model, subject), []).append(np.mean(matrices, axis=0))
    for (model, _subject), matrices in by_subject.items():
        by_model[model].append(np.mean(matrices, axis=0))
    figure, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(15, 3), sharex=True, sharey=True)
    image = None
    for axis, model in zip(axes, MODEL_NAMES, strict=True):
        matrix = np.mean(by_model[model], axis=0)
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(model)
        axis.set_xticks([0, 1], ["Rest", "MI"])
        axis.set_yticks([0, 1], ["Rest", "MI"])
        axis.set_xlabel("Predicted")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    axes[0].set_ylabel("True")
    if image is not None:
        figure.colorbar(image, ax=axes.tolist(), fraction=0.02, pad=0.02)
    _save_figure(figure, output_dir, "primary_confusion_matrices")


def generate_all_figures(results_dir: Path) -> None:
    output_dir = results_dir / "figures"
    statistics_dir = results_dir / "statistics"
    plot_primary_accuracy(statistics_dir, output_dir)
    plot_performance_overview(statistics_dir, output_dir, "accuracy")
    plot_performance_overview(statistics_dir, output_dir, "kappa")
    plot_augmentation_effects(statistics_dir, output_dir)
    plot_seed_variability(statistics_dir, output_dir)
    plot_ica_sensitivity(statistics_dir, output_dir)
    plot_latency(results_dir, output_dir)
    plot_primary_confusion_matrices(results_dir / "cells", output_dir)
