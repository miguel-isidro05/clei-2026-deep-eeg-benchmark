from __future__ import annotations

import numpy as np

from deepbench.classical import make_csp_lda, predict_trial_scores


def test_csp_pipeline_has_frozen_steps() -> None:
    pipeline = make_csp_lda()
    assert list(pipeline.named_steps) == ["covariances", "csp", "lda"]
    assert pipeline.named_steps["csp"].nfilter == 6


def test_classical_window_scores_are_aggregated_per_original_trial() -> None:
    class Classifier:
        classes_ = np.array([0, 1])

        def predict_proba(self, windows: np.ndarray) -> np.ndarray:
            probabilities = np.array([0.2] * 6 + [0.8] * 6)
            return np.column_stack([1.0 - probabilities, probabilities])

    x = np.zeros((2, 3, 512), dtype=np.float32)
    predictions, scores, windows_per_trial = predict_trial_scores(Classifier(), x, "overlap")

    assert predictions.tolist() == [0, 1]
    assert np.allclose(scores, [0.2, 0.8])
    assert windows_per_trial == 6
