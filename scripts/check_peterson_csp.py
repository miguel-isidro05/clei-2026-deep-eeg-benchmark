#!/usr/bin/env python3
"""Run a fast, non-confirmatory CSP sanity check for MI-OpenBCI."""

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

from deepbench.config import MI_SUBJECTS, SPLIT_SEED
from deepbench.datasets import load_mi_openbci_subject
from deepbench.preprocessing import preprocess_split


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
    parser.add_argument("--subjects", nargs="+", default=list(MI_SUBJECTS))
    parser.add_argument("--min-mean-accuracy", type=float, default=0.65)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    subject_accuracy: dict[str, float] = {}
    for subject in args.subjects:
        recording = load_mi_openbci_subject(subject)
        expected_shape = (len(recording.y), 15, 512)
        if recording.x.shape != expected_shape:
            raise SystemExit(
                f"Unexpected MI-OpenBCI shape for {subject}: "
                f"{recording.x.shape}; expected {expected_shape}"
            )
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        for fold, (train, test) in enumerate(splitter.split(recording.x, recording.y)):
            processed = preprocess_split(
                recording.x[train],
                recording.x[test],
                sfreq=recording.sfreq,
                ch_names=recording.ch_names,
                seed=SPLIT_SEED * 1000 + fold,
                ica_policy="none",
            )
            classifier = make_csp_lda()
            classifier.fit(processed.x_train, recording.y[train])
            y_true.append(recording.y[test])
            y_pred.append(classifier.predict(processed.x_test))
        accuracy = float(np.mean(np.concatenate(y_true) == np.concatenate(y_pred)))
        subject_accuracy[subject] = accuracy
        print(f"peterson_csp subject={subject} accuracy={accuracy:.4f}")

    values = np.asarray(list(subject_accuracy.values()), dtype=float)
    mean_accuracy = float(values.mean())
    status = "pass" if mean_accuracy >= args.min_mean_accuracy else "failed"
    payload = {
        "analysis_role": "diagnostic_not_confirmatory",
        "status": status,
        "protocol": "within_session_5fold",
        "ica_policy": "none",
        "split_seed": SPLIT_SEED,
        "mean_accuracy": mean_accuracy,
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum_required_mean_accuracy": float(args.min_mean_accuracy),
        "subject_accuracy": subject_accuracy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"peterson_csp_sanity={status} mean_accuracy={mean_accuracy:.4f} "
        f"output={args.output}"
    )
    if status != "pass":
        raise SystemExit(
            "Peterson CSP sanity check failed; deep validation was not started. "
            "Inspect loader, labels, filtering, and split construction."
        )


if __name__ == "__main__":
    main()
