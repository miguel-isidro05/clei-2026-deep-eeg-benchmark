from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.peterson_diffusion_v11.config import ExperimentConfig
from experiments.peterson_diffusion_v11.training import predict_probabilities, train_model


@pytest.mark.parametrize("formulation", ["discriminative", "diffusion"])
def test_training_smoke_restores_validation_selected_model(formulation: str) -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(8, 3, 32)).astype(np.float32)
    y_train = np.array([0, 1] * 4, dtype=np.int64)
    x_validation = rng.normal(size=(4, 3, 32)).astype(np.float32)
    y_validation = np.array([0, 1, 0, 1], dtype=np.int64)
    config = replace(
        ExperimentConfig(wave=1),
        epochs=2,
        patience=2,
        batch_size=4,
        hidden=8,
        diffusion_steps=5,
        inference_k=1,
        objective="noise_rank",
    )
    outcome = train_model(
        "res",
        formulation,
        x_train,
        y_train,
        x_validation,
        y_validation,
        config=config,
        device="cpu",
        seed=4,
    )
    probabilities = predict_probabilities(
        outcome.model,
        x_validation,
        formulation=formulation,
        config=config,
        device="cpu",
        seed=44,
    )
    assert len(outcome.history) == 2
    assert 1 <= outcome.best_epoch <= 2
    assert outcome.updates == 4
    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()
    assert np.all((0 <= probabilities) & (probabilities <= 1))
