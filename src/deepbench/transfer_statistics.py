"""Subject-level transfer contrasts with seeds treated as repeated fits."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

from .config import MODEL_NAMES, PAPER_SEEDS, SOUZA_SUBJECTS
from .io import read_json
from .transfer_profile import TRANSFER_PROFILE_VERSION

CONTRASTS = (
    ("linear_probe", "scratch"),
    ("full_finetune", "scratch"),
    ("full_finetune", "linear_probe"),
)


def _exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    null = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def _paired_rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    return float(
        (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    )


def _bootstrap_mean_interval(
    differences: np.ndarray, *, seed: int, replicates: int = 100_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(
        differences, size=(replicates, len(differences)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def analyze_transfer(output_dir: Path) -> dict[str, object]:
    subject_metrics: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for protocol in ("within_session", "cross_session"):
        for model in MODEL_NAMES:
            for subject in SOUZA_SUBJECTS:
                for arm in ("scratch", "linear_probe", "full_finetune"):
                    accuracy: list[float] = []
                    kappa: list[float] = []
                    for seed in PAPER_SEEDS:
                        path = (
                            output_dir
                            / "transfer_cells"
                            / (
                                f"Souza2023__transfer__{protocol}__overlap__{arm}__"
                                f"{model}__seed{seed}__{subject}.json"
                            )
                        )
                        if not path.exists():
                            raise RuntimeError(f"Incomplete transfer profile: missing {path.name}")
                        payload = read_json(path)
                        if payload.get("profile") != TRANSFER_PROFILE_VERSION:
                            raise RuntimeError(f"Mixed transfer profile in {path.name}")
                        accuracy.append(float(payload["metrics"]["accuracy"]))
                        kappa.append(float(payload["metrics"]["kappa"]))
                    subject_metrics[(protocol, model, subject, arm)] = {
                        "accuracy": float(np.mean(accuracy)),
                        "kappa": float(np.mean(kappa)),
                    }

    rows: list[dict[str, object]] = []
    pvalues: list[float] = []
    for protocol in ("within_session", "cross_session"):
        for model in MODEL_NAMES:
            for treatment, control in CONTRASTS:
                treatment_accuracy = np.asarray(
                    [
                        subject_metrics[(protocol, model, subject, treatment)]["accuracy"]
                        for subject in SOUZA_SUBJECTS
                    ]
                )
                control_accuracy = np.asarray(
                    [
                        subject_metrics[(protocol, model, subject, control)]["accuracy"]
                        for subject in SOUZA_SUBJECTS
                    ]
                )
                treatment_kappa = np.asarray(
                    [
                        subject_metrics[(protocol, model, subject, treatment)]["kappa"]
                        for subject in SOUZA_SUBJECTS
                    ]
                )
                control_kappa = np.asarray(
                    [
                        subject_metrics[(protocol, model, subject, control)]["kappa"]
                        for subject in SOUZA_SUBJECTS
                    ]
                )
                differences = treatment_accuracy - control_accuracy
                kappa_differences = treatment_kappa - control_kappa
                pvalue = _exact_sign_flip_pvalue(differences)
                interval = _bootstrap_mean_interval(differences, seed=len(rows) + 2026)
                pvalues.append(pvalue)
                rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "treatment": treatment,
                        "control": control,
                        "comparison": f"{treatment}-{control}",
                        "subject_unit": True,
                        "seed_aggregation": "mean_before_inference",
                        "mean_control_accuracy": float(np.mean(control_accuracy)),
                        "mean_treatment_accuracy": float(np.mean(treatment_accuracy)),
                        "mean_difference": float(np.mean(differences)),
                        "mean_difference_ci95": list(interval),
                        "median_difference": float(np.median(differences)),
                        "mean_kappa_difference": float(np.mean(kappa_differences)),
                        "positive_subjects": int(np.sum(differences > 0)),
                        "effect_size_dz": float(
                            np.mean(differences) / np.std(differences, ddof=1)
                        )
                        if np.std(differences, ddof=1) > 0
                        else 0.0,
                        "paired_rank_biserial": _paired_rank_biserial(differences),
                        "exact_sign_flip_p": pvalue,
                        "minimum_attainable_two_sided_p": 0.0625,
                        "subject_values": [
                            {
                                "subject": subject,
                                "control_accuracy": float(control_accuracy[index]),
                                "treatment_accuracy": float(treatment_accuracy[index]),
                                "accuracy_difference": float(differences[index]),
                                "kappa_difference": float(kappa_differences[index]),
                            }
                            for index, subject in enumerate(SOUZA_SUBJECTS)
                        ],
                    }
                )

    adjusted = multipletests(pvalues, method="holm")[1]
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p"] = float(value)
        counterpart = next(
            candidate
            for candidate in rows
            if candidate["model"] == row["model"]
            and candidate["treatment"] == row["treatment"]
            and candidate["control"] == row["control"]
            and candidate["protocol"] != row["protocol"]
        )
        row["compatible_direction_other_protocol"] = bool(
            float(counterpart["mean_difference"]) >= 0.0
        )
        row["practical_success"] = bool(
            row["control"] == "scratch"
            and float(row["mean_difference"]) >= 0.05
            and int(row["positive_subjects"]) >= 4
            and float(row["mean_kappa_difference"]) >= -0.01
            and row["compatible_direction_other_protocol"]
        )
    return {
        "profile": TRANSFER_PROFILE_VERSION,
        "inference_unit": "subject",
        "n_subjects": len(SOUZA_SUBJECTS),
        "multiplicity_family_size": len(rows),
        "interval_method": "subject_bootstrap_percentile_100000",
        "rows": rows,
    }
