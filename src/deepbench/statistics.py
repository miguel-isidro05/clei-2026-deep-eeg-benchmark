"""Subject-level inference with paired Wilcoxon tests and Holm correction."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests

from .config import PAPER_EPOCHS, is_classical_model

CELL_KEYS = [
    "dataset",
    "task",
    "protocol",
    "condition",
    "ica_policy",
    "model",
    "subject",
    "seed",
]
FAMILY_KEYS = ["dataset", "task", "protocol", "condition", "ica_policy", "metric"]
PRIMARY_METRICS = ("accuracy", "kappa")
MIN_PAIRED_SUBJECTS = 5
GLOBAL_COMPATIBILITY_KEYS = (
    "schema_version",
    "code_sha256",
    "split_seed",
    "environment_sha256",
)


def statistics_code_sha256() -> str:
    """Fingerprint the implementation that produces inferential tables."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_run_configurations(payloads: list[dict[str, object]]) -> None:
    global_signatures: set[tuple[object, ...]] = set()
    recipes_by_model: dict[str, set[tuple[object, ...]]] = {}
    data_by_subject: dict[tuple[str, str, str], set[str]] = {}
    for payload in payloads:
        configuration = payload.get("run_configuration")
        if not isinstance(configuration, dict):
            raise ValueError("Every result cell must contain run_configuration")
        missing = [key for key in GLOBAL_COMPATIBILITY_KEYS if key not in configuration]
        if missing:
            raise ValueError(f"run_configuration is missing compatibility fields: {missing}")
        if not configuration.get("data_sha256"):
            raise ValueError("run_configuration is missing data_sha256")
        if not isinstance(configuration.get("environment_versions"), dict):
            raise ValueError("run_configuration is missing environment_versions")
        global_signatures.add(tuple(configuration[key] for key in GLOBAL_COMPATIBILITY_KEYS))
        model = str(payload["model"])
        model_signature = (
            configuration.get("epochs"),
            configuration.get("device_type"),
            configuration.get("hardware"),
            configuration.get("deterministic_policy"),
            json.dumps(configuration.get("recipe"), sort_keys=True),
        )
        recipes_by_model.setdefault(model, set()).add(model_signature)
        data_key = (str(payload["dataset"]), str(payload["protocol"]), str(payload["subject"]))
        data_by_subject.setdefault(data_key, set()).add(str(configuration.get("data_sha256")))
    if len(global_signatures) != 1:
        raise ValueError("Incompatible result cells: code, split seed, or environment differs")
    inconsistent_models = [model for model, recipes in recipes_by_model.items() if len(recipes) > 1]
    if inconsistent_models:
        raise ValueError(f"Incompatible training recipes for models: {inconsistent_models}")
    inconsistent_data = [key for key, identities in data_by_subject.items() if len(identities) > 1]
    if inconsistent_data:
        raise ValueError(f"Incompatible data identities for cells: {inconsistent_data}")


def load_cells(cells_dir: Path) -> pd.DataFrame:
    """Read cell JSON files into one tidy table and reject duplicate cells."""
    rows: list[dict[str, object]] = []
    payloads: list[dict[str, object]] = []
    for path in sorted(cells_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not all(key in payload for key in CELL_KEYS) or "metrics" not in payload:
            raise ValueError(f"Malformed result cell: {path}")
        payloads.append(payload)
        for metric, value in payload["metrics"].items():
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError(f"Non-finite metric {metric!r} in result cell: {path}")
            rows.append(
                {
                    **{key: payload[key] for key in CELL_KEYS},
                    "metric": str(metric),
                    "value": numeric_value,
                    "source": str(path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No experiment cell JSON files found in {cells_dir}")
    _validate_run_configurations(payloads)
    frame = pd.DataFrame(rows)
    duplicate_keys = CELL_KEYS + ["metric"]
    duplicates = frame.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, duplicate_keys].head().to_dict("records")
        raise ValueError(f"Duplicate result cells detected: {sample}")
    return frame


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return mean and two-sided Student-t CI; no resampling is performed."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(clean.mean())
    if clean.size == 1:
        return mean, np.nan, np.nan
    sem = scipy.stats.sem(clean)
    half_width = float(scipy.stats.t.ppf((1 + confidence) / 2, clean.size - 1) * sem)
    return mean, mean - half_width, mean + half_width


def _wilcoxon_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Non-finite metric difference passed to paired Wilcoxon test")
    if np.allclose(differences, 0.0):
        return 1.0
    p_value = float(
        scipy.stats.wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )
    if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
        raise ValueError(f"Invalid Wilcoxon p-value: {p_value}")
    return p_value


def _holm_adjust(records: list[dict[str, object]]) -> None:
    p_values = np.asarray([record["p_raw"] for record in records], dtype=float)
    if not np.isfinite(p_values).all() or np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError(f"Invalid p-values before Holm correction: {p_values.tolist()}")
    adjusted = multipletests(p_values, method="holm")
    for record, reject, p_adjusted in zip(records, adjusted[0], adjusted[1], strict=True):
        if not np.isfinite(p_adjusted):
            raise ValueError(f"Invalid Holm-adjusted p-value: {p_adjusted}")
        record["p_holm"] = float(p_adjusted)
        record["reject_holm_0_05"] = bool(reject)


def paired_rank_biserial(differences: np.ndarray) -> float:
    """Return paired rank-biserial correlation with the sign of the stated difference."""
    clean = np.asarray(differences, dtype=float)
    clean = clean[np.isfinite(clean) & ~np.isclose(clean, 0.0)]
    if clean.size == 0:
        return 0.0
    ranks = scipy.stats.rankdata(np.abs(clean), method="average")
    positive = float(ranks[clean > 0].sum())
    negative = float(ranks[clean < 0].sum())
    return (positive - negative) / (positive + negative)


def aggregate_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """Average optimization repeats inside each subject before any inference."""
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite metric value found before seed aggregation")
    keys = [key for key in CELL_KEYS if key != "seed"] + ["metric"]
    return frame.groupby(keys, as_index=False, observed=True)["value"].mean()


def completeness_table(frame: pd.DataFrame, expected_seeds: tuple[int, ...]) -> pd.DataFrame:
    keys = [key for key in CELL_KEYS if key != "seed"] + ["metric"]
    expected = set(expected_seeds)
    records = []
    for group, values in frame.groupby(keys, observed=True):
        present = set(int(value) for value in values["seed"])
        records.append(
            {
                **dict(zip(keys, group, strict=True)),
                "n_seeds": len(present),
                "present_seeds": ",".join(map(str, sorted(present))),
                "missing_seeds": ",".join(map(str, sorted(expected - present))),
                "complete": present == expected,
            }
        )
    return pd.DataFrame(records)


def descriptive_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate subject dispersion from within-subject seed dispersion."""
    subject_means = aggregate_seeds(frame.loc[frame["metric"].isin(PRIMARY_METRICS)])
    subject_keys = FAMILY_KEYS + ["model"]
    seed_sd = (
        frame.groupby(subject_keys + ["subject"], observed=True)["value"]
        .std(ddof=1)
        .groupby(subject_keys, observed=True)
        .mean()
        .rename("mean_within_subject_seed_sd")
    )
    records = []
    for group, values in subject_means.groupby(subject_keys, observed=True):
        array = values["value"].to_numpy(float)
        mean, low, high = mean_ci(array)
        key = dict(zip(subject_keys, group, strict=True))
        records.append(
            {
                **key,
                "n_subjects": len(array),
                "mean": mean,
                "sd_between_subjects": float(np.std(array, ddof=1)) if len(array) > 1 else np.nan,
                "ci95_low": low,
                "ci95_high": high,
                "mean_within_subject_seed_sd": float(seed_sd.get(group, np.nan)),
            }
        )
    return pd.DataFrame(records)


def paired_model_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare every model pair within each predeclared statistical family."""
    subject_means = aggregate_seeds(frame.loc[frame["metric"].isin(PRIMARY_METRICS)])
    records: list[dict[str, object]] = []
    for family, values in subject_means.groupby(FAMILY_KEYS, observed=True):
        models = sorted(values["model"].unique())
        family_records = []
        for model_a, model_b in itertools.combinations(models, 2):
            a = values.loc[values["model"] == model_a, ["subject", "value"]]
            b = values.loc[values["model"] == model_b, ["subject", "value"]]
            if set(a["subject"]) != set(b["subject"]):
                raise ValueError(f"Paired cohort mismatch for {model_a} and {model_b} in {family}")
            paired = a.merge(b, on="subject", suffixes=("_a", "_b"), validate="one_to_one")
            if paired.empty:
                raise ValueError(f"No shared subjects for {model_a} and {model_b} in {family}")
            if len(paired) < MIN_PAIRED_SUBJECTS:
                raise ValueError(
                    f"Only {len(paired)} paired subjects for {model_a}/{model_b}; "
                    f"at least {MIN_PAIRED_SUBJECTS} are required"
                )
            differences = paired["value_a"].to_numpy() - paired["value_b"].to_numpy()
            mean, low, high = mean_ci(differences)
            family_records.append(
                {
                    **dict(zip(FAMILY_KEYS, family, strict=True)),
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_subjects": len(paired),
                    "mean_difference_a_minus_b": mean,
                    "median_difference_a_minus_b": float(np.median(differences)),
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                    "rank_biserial_a_minus_b": paired_rank_biserial(differences),
                    "p_raw": _wilcoxon_pvalue(differences),
                }
            )
        if family_records:
            _holm_adjust(family_records)
            records.extend(family_records)
    return pd.DataFrame(records)


def augmentation_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """Test augmentations against compute-matched repeated center-crop controls."""
    subject_means = aggregate_seeds(frame.loc[frame["metric"].isin(PRIMARY_METRICS)])
    family_keys = ["dataset", "task", "protocol", "ica_policy", "metric"]
    records: list[dict[str, object]] = []
    for family, values in subject_means.groupby(family_keys, observed=True):
        family_records = []
        for model in sorted(values["model"].unique()):
            model_values = values.loc[values["model"] == model]
            for condition, control in (
                ("overlap", "center"),
                ("nonoverlap", "center_x2"),
                ("overlap", "center_x6"),
            ):
                center = model_values.loc[
                    model_values["condition"] == control, ["subject", "value"]
                ]
                candidate = model_values.loc[
                    model_values["condition"] == condition, ["subject", "value"]
                ]
                if center.empty or candidate.empty:
                    continue
                if set(candidate["subject"]) != set(center["subject"]):
                    raise ValueError(
                        f"Augmentation cohort mismatch for {model}/{condition} in {family}"
                    )
                paired = candidate.merge(
                    center, on="subject", suffixes=("_augmented", "_center"), validate="one_to_one"
                )
                if len(paired) < MIN_PAIRED_SUBJECTS:
                    raise ValueError(
                        f"Only {len(paired)} paired subjects for {model}/{condition}; "
                        f"at least {MIN_PAIRED_SUBJECTS} are required"
                    )
                differences = paired["value_augmented"] - paired["value_center"]
                mean, low, high = mean_ci(differences.to_numpy())
                family_records.append(
                    {
                        **dict(zip(family_keys, family, strict=True)),
                        "model": model,
                        "comparison": f"{condition}-{control}",
                        "n_subjects": len(paired),
                        "mean_difference": mean,
                        "difference_ci95_low": low,
                        "difference_ci95_high": high,
                        "rank_biserial_augmented_minus_center": paired_rank_biserial(
                            differences.to_numpy()
                        ),
                        "p_raw": _wilcoxon_pvalue(differences.to_numpy()),
                    }
                )
        if family_records:
            _holm_adjust(family_records)
            records.extend(family_records)
    return pd.DataFrame(records)


def sample_accounting_table(payloads: list[dict[str, object]]) -> pd.DataFrame:
    """Flatten fold-level trial, example, and class counts for audit and reporting."""
    records: list[dict[str, object]] = []
    identity_keys = [
        "dataset",
        "task",
        "protocol",
        "condition",
        "ica_policy",
        "model",
        "subject",
        "seed",
    ]
    for payload in payloads:
        identity = {key: payload[key] for key in identity_keys}
        for fold_index, report in enumerate(payload.get("fold_reports", [])):
            train_counts = report.get("train_class_counts", {})
            test_counts = report.get("test_class_counts", {})
            training_history = report.get("training_history", [])
            records.append(
                {
                    **identity,
                    "fold_index": fold_index,
                    "session": report.get("session"),
                    "held_session": report.get("held_session"),
                    "n_train_trials": report.get("n_train_trials"),
                    "n_train_examples": report.get("n_train_examples"),
                    "n_test_trials": report.get("n_test_trials"),
                    "optimizer_updates": sum(
                        int(epoch.get("train_batch_count", 0)) for epoch in training_history
                    ),
                    "train_class_0": train_counts.get("0"),
                    "train_class_1": train_counts.get("1"),
                    "test_class_0": test_counts.get("0"),
                    "test_class_1": test_counts.get("1"),
                }
            )
    return pd.DataFrame(records)


def validate_matched_augmentation_compute(
    payloads: list[dict[str, object]], *, strict: bool = False
) -> None:
    """Reject augmentation pairs with unequal examples or optimizer updates."""
    matched: dict[tuple[object, ...], dict[str, tuple[int, tuple[int, ...]]]] = {}
    conditions = {"center_x2", "nonoverlap", "center_x6", "overlap"}
    for payload in payloads:
        condition = str(payload.get("condition"))
        if (
            payload.get("protocol") != "within_split"
            or condition not in conditions
            or payload.get("ica_policy") != "none"
        ):
            continue
        base = tuple(
            payload.get(key)
            for key in ("dataset", "protocol", "ica_policy", "model", "subject", "seed")
        )
        for fold_index, report in enumerate(payload.get("fold_reports", [])):
            batches_per_epoch = tuple(
                int(epoch.get("train_batch_count", 0))
                for epoch in report.get("training_history", [])
            )
            is_classical = is_classical_model(str(payload.get("model")))
            if strict and not is_classical and len(batches_per_epoch) != PAPER_EPOCHS:
                raise ValueError(
                    f"Compute audit for {(*base, fold_index)} requires "
                    f"{PAPER_EPOCHS} recorded epochs"
                )
            matched.setdefault((*base, fold_index), {})[condition] = (
                int(report["n_train_examples"]),
                batches_per_epoch,
            )
    for key, values in matched.items():
        for control, candidate in (
            ("center_x2", "nonoverlap"),
            ("center_x6", "overlap"),
        ):
            present = {control, candidate} & set(values)
            if strict and present and present != {control, candidate}:
                missing = {control, candidate} - present
                raise ValueError(f"Compute audit for {key} is missing {sorted(missing)}")
            if present == {control, candidate} and values[control] != values[candidate]:
                raise ValueError(
                    f"Compute mismatch for {key}: {control}={values[control]} "
                    f"and {candidate}={values[candidate]}"
                )


def write_statistics(
    cells_dir: Path,
    output_dir: Path,
    expected_seeds: tuple[int, ...],
    *,
    allow_incomplete: bool = False,
    force_exploratory: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_cells(cells_dir)
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(cells_dir.glob("*.json"))
    ]
    validate_matched_augmentation_compute(payloads, strict=not force_exploratory)
    completeness = completeness_table(frame, expected_seeds)
    completeness.to_csv(output_dir / "completeness.csv", index=False)
    incomplete = force_exploratory or not bool(completeness["complete"].all())
    if incomplete and not allow_incomplete:
        raise RuntimeError("Incomplete seed grid; see completeness.csv")
    outputs = {
        "all_seed_metrics.csv": frame,
        "descriptive_subject_seed_variability.csv": descriptive_table(frame),
        "sample_accounting.csv": sample_accounting_table(payloads),
    }
    incomplete_marker = output_dir / "INCOMPLETE_EXPLORATORY_ONLY.txt"
    inferential_names = (
        "paired_wilcoxon_holm.csv",
        "augmentation_wilcoxon_holm.csv",
    )
    if incomplete:
        incomplete_marker.write_text(
            "Seed grid incomplete. Descriptive outputs are exploratory; no inferential tables "
            "were generated.\n",
            encoding="utf-8",
        )
        for name in inferential_names:
            (output_dir / name).unlink(missing_ok=True)
    else:
        incomplete_marker.unlink(missing_ok=True)
        outputs["paired_wilcoxon_holm.csv"] = paired_model_tests(frame)
        outputs["augmentation_wilcoxon_holm.csv"] = augmentation_tests(frame)
    for name, table in outputs.items():
        table.to_csv(output_dir / name, index=False)
