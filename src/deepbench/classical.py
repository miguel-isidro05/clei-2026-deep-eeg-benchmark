"""Frozen CSP+LDA baseline evaluated with the neural-model data contract."""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from .preprocessing import aggregate_trial_probabilities, prepare_test_windows

CSP_RECIPE: dict[str, object] = {
    "estimator": "Covariances(OAS)+CSP+LDA",
    "covariance_estimator": "oas",
    "csp_n_filters": 6,
    "csp_log": True,
    "lda_solver": "svd",
    "optimization_stochastic": False,
    "selection_policy": "predeclared_no_test_tuning",
}


def make_csp_lda() -> Pipeline:
    return Pipeline(
        [
            ("covariances", Covariances(estimator="oas")),
            ("csp", CSP(nfilter=6, log=True)),
            ("lda", LinearDiscriminantAnalysis(solver="svd")),
        ]
    )


def predict_trial_scores(
    classifier: Pipeline, x: np.ndarray, condition: str
) -> tuple[np.ndarray, np.ndarray, int]:
    windows, trial_indices = prepare_test_windows(x, condition)
    probabilities = np.asarray(classifier.predict_proba(windows), dtype=float)
    class_one = int(np.flatnonzero(np.asarray(classifier.classes_) == 1)[0])
    scores = aggregate_trial_probabilities(
        probabilities[:, class_one], trial_indices, n_trials=len(x)
    )
    counts = np.bincount(trial_indices, minlength=len(x))
    if len(set(counts.tolist())) != 1:
        raise RuntimeError("Every test trial must contribute the same number of windows")
    return (scores >= 0.5).astype(np.int64), scores, int(counts[0])
