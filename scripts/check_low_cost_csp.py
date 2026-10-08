#!/usr/bin/env python3
"""Run a fast, non-confirmatory CSP sanity check for one low-cost dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from deepbench.config import (
    MI_CHANNELS,
    MI_SUBJECTS,
    SOUZA_CHANNELS,
    SOUZA_SUBJECTS,
    SOUZA_TRIAL_SAMPLES,
    SPLIT_SEED,
    TRIAL_SAMPLES,
)
from deepbench.datasets import load_subject
from deepbench.preprocessing import center_crop, preprocess_split

DATASET_SETTINGS = {
    "MI-OpenBCI": {
        "subjects": MI_SUBJECTS,
        "n_channels": len(MI_CHANNELS),
        "n_times": TRIAL_SAMPLES,
        "minimum_mean_accuracy": 0.65,
    },
    "Souza2023": {
        "subjects": SOUZA_SUBJECTS,
        "n_channels": len(SOUZA_CHANNELS),
        "n_times": SOUZA_TRIAL_SAMPLES,
        "minimum_mean_accuracy": 0.50,
    },
}


def make_csp_lda() -> Pipeline:
    return Pipeline(
        [
            ("covariances", Covariances(estimator="oas")),
            ("csp", CSP(nfilter=6, log=True)),
            ("lda", LinearDiscriminantAnalysis(solver="svd")),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_SETTINGS, required=True)
    parser.add_argument("--condition", choices=("full", "center"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = DATASET_SETTINGS[args.dataset]
    subject_accuracy: dict[str, float] = {}
    for subject in settings["subjects"]:
        recording = load_subject(args.dataset, str(subject))
        expected_shape = (
            len(recording.y),
            int(settings["n_channels"]),
            int(settings["n_times"]),
        )
        if recording.x.shape != expected_shape:
            raise SystemExit(
                f"Unexpected {args.dataset} shape for {subject}: "
                f"{recording.x.shape}; expected {expected_shape}"
            )
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        for session in np.unique(recording.sessions):
            session_indices = np.flatnonzero(recording.sessions == session)
            session_y = recording.y[session_indices]
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
            for fold, (train_local, test_local) in enumerate(
                splitter.split(session_indices, session_y)
            ):
                train = session_indices[train_local]
                test = session_indices[test_local]
                processed = preprocess_split(
                    recording.x[train],
                    recording.x[test],
                    sfreq=recording.sfreq,
                    ch_names=recording.ch_names,
                    seed=SPLIT_SEED * 1000 + fold,
                    ica_policy="none",
                )
                x_train = processed.x_train
                x_test = processed.x_test
                if args.condition == "center":
                    x_train = center_crop(x_train)
                    x_test = center_crop(x_test)
                classifier = make_csp_lda()
                classifier.fit(x_train, recording.y[train])
                y_true.append(recording.y[test])
                y_pred.append(classifier.predict(x_test))
        accuracy = float(np.mean(np.concatenate(y_true) == np.concatenate(y_pred)))
        subject_accuracy[str(subject)] = accuracy
        print(f"csp_sanity dataset={args.dataset} subject={subject} accuracy={accuracy:.4f}")

    values = np.asarray(list(subject_accuracy.values()), dtype=float)
    mean_accuracy = float(values.mean())
    minimum = float(settings["minimum_mean_accuracy"])
    status = "pass" if mean_accuracy >= minimum else "failed"
    payload = {
        "analysis_role": "diagnostic_not_confirmatory",
        "dataset": args.dataset,
        "status": status,
        "protocol": "within_session_5fold",
        "condition": args.condition,
        "ica_policy": "none",
        "split_seed": SPLIT_SEED,
        "mean_accuracy": mean_accuracy,
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum_required_mean_accuracy": minimum,
        "subject_accuracy": subject_accuracy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"csp_sanity={status} dataset={args.dataset} mean_accuracy={mean_accuracy:.4f} "
        f"output={args.output}"
    )
    if status != "pass":
        raise SystemExit(
            f"{args.dataset} CSP sanity check failed; deep training was not started. "
            "Inspect loader, labels, filtering, and split construction."
        )


if __name__ == "__main__":
    main()
