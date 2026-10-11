from __future__ import annotations

import numpy as np
import torch
from torch import nn

from experiments.peterson_search_v12.exp03_quality_fbcnet.training import (
    train_checkpoint_trajectory,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(12, 2)

    def forward(self, x: torch.Tensor, *, dropout_seed: int | None = None) -> torch.Tensor:
        _ = dropout_seed
        return self.linear(self.flatten(x))


def test_last_and_best_checkpoints_come_from_one_trajectory():
    rng = np.random.default_rng(23)
    x_train = rng.normal(size=(12, 3, 4)).astype(np.float32)
    y_train = np.asarray([0, 1] * 6, dtype=np.int64)
    x_validation = rng.normal(size=(6, 3, 4)).astype(np.float32)
    y_validation = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    outcome = train_checkpoint_trajectory(
        TinyClassifier(),
        x_train,
        y_train,
        x_validation,
        y_validation,
        device="cpu",
        seed=31,
        epochs=3,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=None,
    )
    assert outcome.last.trajectory_sha256 == outcome.best.trajectory_sha256
    assert outcome.last.epoch == 3
    assert 1 <= outcome.best.epoch <= 3
    assert len(outcome.history) == 3
    assert outcome.last.state_dict.keys() == outcome.best.state_dict.keys()
