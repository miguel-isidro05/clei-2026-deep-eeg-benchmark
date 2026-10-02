"""Subject-level transfer contrasts with seeds treated as repeated fits."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

from .config import MODEL_NAMES, PAPER_SEEDS, SOUZA_SUBJECTS
from .io import read_json


def _exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    null = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def analyze_transfer(output_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    pvalues: list[float] = []
    for protocol in ("within_session", "cross_session"):
        for model in MODEL_NAMES:
            scratch: list[float] = []
            scratch_kappa: list[float] = []
            arms = {"linear_probe": [], "full_finetune": []}
            arm_kappa = {"linear_probe": [], "full_finetune": []}
            for subject in SOUZA_SUBJECTS:
                values: dict[str, list[float]] = {arm: [] for arm in ("scratch", *arms)}
                kappas: dict[str, list[float]] = {arm: [] for arm in ("scratch", *arms)}
                for arm in values:
                    for seed in PAPER_SEEDS:
                        path = (
                            output_dir
                            / "transfer_cells"
                            / (
                                f"Souza2023__transfer__{protocol}__overlap__{arm}__{model}__seed{seed}__{subject}.json"
                            )
                        )
                        if not path.exists():
                            raise RuntimeError(f"Incomplete transfer profile: missing {path.name}")
                        payload = read_json(path)
                        if payload.get("profile") != "clei2026-peterson-souza-transfer-v8":
                            raise RuntimeError(f"Mixed transfer profile in {path.name}")
                        values[arm].append(float(payload["metrics"]["accuracy"]))
                        kappas[arm].append(float(payload["metrics"]["kappa"]))
                scratch.append(float(np.mean(values["scratch"])))
                scratch_kappa.append(float(np.mean(kappas["scratch"])))
                for arm in arms:
                    arms[arm].append(float(np.mean(values[arm])))
                    arm_kappa[arm].append(float(np.mean(kappas[arm])))
            for arm, scores in arms.items():
                differences = np.asarray(scores) - np.asarray(scratch)
                pvalue = _exact_sign_flip_pvalue(differences)
                pvalues.append(pvalue)
                rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "arm": arm,
                        "subject_unit": True,
                        "seed_aggregation": "mean_before_inference",
                        "mean_scratch_accuracy": float(np.mean(scratch)),
                        "mean_transfer_accuracy": float(np.mean(scores)),
                        "mean_difference": float(np.mean(differences)),
                        "mean_kappa_difference": float(
                            np.mean(np.asarray(arm_kappa[arm]) - np.asarray(scratch_kappa))
                        ),
                        "positive_subjects": int(np.sum(differences > 0)),
                        "effect_size_dz": float(np.mean(differences) / np.std(differences, ddof=1))
                        if np.std(differences, ddof=1) > 0
                        else 0.0,
                        "exact_sign_flip_p": pvalue,
                        "minimum_attainable_two_sided_p": 0.0625,
                        "practical_success": bool(
                            np.mean(differences) >= 0.05
                            and np.sum(differences > 0) >= 4
                            and np.mean(np.asarray(arm_kappa[arm]) - np.asarray(scratch_kappa))
                            >= -0.01
                        ),
                    }
                )
    adjusted = multipletests(pvalues, method="holm")[1]
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p"] = float(value)
        counterpart = next(
            candidate
            for candidate in rows
            if candidate["model"] == row["model"]
            and candidate["arm"] == row["arm"]
            and candidate["protocol"] != row["protocol"]
        )
        row["compatible_direction_other_protocol"] = bool(
            float(counterpart["mean_difference"]) >= 0.0
        )
        row["practical_success"] = bool(
            row["practical_success"] and row["compatible_direction_other_protocol"]
        )
    return {
        "profile": "clei2026-peterson-souza-transfer-v8",
        "inference_unit": "subject",
        "n_subjects": 5,
        "rows": rows,
    }
