from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from experiments.peterson_search_v12.exp08_fixed_convergence.config import (
    Exp08Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp08_fixed_convergence.training import (
    aggregate_grouped_logits,
    fit_frozen_lpwf,
    freeze_backbone,
)


def test_full_config_freezes_new_seeds_and_fixed_convergence() -> None:
    config = Exp08Config()
    assert config.screen_seeds == (15, 16, 17)
    assert config.confirm_seeds == (18, 19)
    assert config.backbone_epochs == 300
    assert config.expected_screen_cells == 216
    assert config.holdout_subjects == ("S07", "S04", "S05", "S06")


def test_smoke_config_keeps_a_minimal_fixed_epoch_budget() -> None:
    config = smoke_config()
    assert config.backbone_epochs == 1
    assert config.fusion_epochs == 2
    assert config.subjects == ("S09", "S03")


def test_training_surface_does_not_reintroduce_epoch_selection() -> None:
    from experiments.peterson_search_v12.exp08_fixed_convergence import training

    source = inspect.getsource(training)
    assert "select_epoch" not in source
    assert "early_stopping" not in source


def test_frozen_lpwf_changes_only_six_weights_and_not_backbone() -> None:
    torch.manual_seed(7)
    backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 2))
    freeze_backbone(backbone)
    before = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
    logits = np.random.default_rng(7).normal(size=(12, 6, 2)).astype(np.float32)
    labels = np.asarray([0, 1] * 6, dtype=np.int64)

    artifact = fit_frozen_lpwf(
        backbone,
        logits,
        labels,
        device="cpu",
        seed=7,
        epochs=3,
        learning_rate=0.05,
    )

    assert artifact.window_logits.shape == (6,)
    assert np.isclose(np.asarray(artifact.window_weights).sum(), 1.0)
    assert artifact.trainable_parameters == 6
    assert artifact.backbone_state_before == artifact.backbone_state_after
    assert all(torch.equal(before[key], value) for key, value in backbone.state_dict().items())


def test_uniform_and_lpwf_aggregation_return_trial_probabilities() -> None:
    logits = np.zeros((4, 6, 2), dtype=np.float32)
    logits[:, :, 1] = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float32)
    uniform = aggregate_grouped_logits(logits, aggregation="uniform")
    weighted = aggregate_grouped_logits(
        logits,
        aggregation="lpwf",
        window_logits=np.asarray([-5, -5, -5, -5, -5, 5], dtype=np.float32),
    )
    assert uniform.shape == (4,)
    assert weighted.shape == (4,)
    assert np.all(weighted > uniform)


def test_invalid_grouped_logits_are_rejected() -> None:
    with pytest.raises(ValueError, match="six windows"):
        aggregate_grouped_logits(np.zeros((3, 5, 2)), aggregation="uniform")

