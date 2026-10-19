from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from deepbench.types import SubjectRecording
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.splits import build_nested_folds
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.training import (
    refit_fixed_epochs,
    select_epoch,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2 * 16, 2)

    def forward(
        self, x: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = x.flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def test_nested_folds_keep_outer_test_out_of_epoch_selection() -> None:
    rng = np.random.default_rng(3)
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 50)
    recording = SubjectRecording(
        dataset="MI-OpenBCI",
        subject="S09",
        x=rng.normal(size=(100, 2, 32)).astype(np.float32),
        y=labels,
        sessions=np.asarray(["session_0"] * 100),
        sfreq=128.0,
        ch_names=("C3", "C4"),
        task="motor_imagery_vs_rest",
    )
    folds = build_nested_folds(recording, split_seed=2026)
    assert len(folds) == 5
    for fold in folds:
        fold.assert_disjoint()
        assert set(fold.inner_train_indices) | set(fold.inner_validation_indices) == set(
            fold.outer_train_indices
        )
        assert not (
            set(fold.inner_train_indices) | set(fold.inner_validation_indices)
        ) & set(fold.outer_test_indices)


def test_epoch_selection_returns_only_epoch_then_refits_from_scratch() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(32, 2, 16)).astype(np.float32)
    y = np.tile(np.asarray([0, 1], dtype=np.int64), 16)
    candidate = SimpleNamespace(
        config_id="exp05-fbc-control",
        mechanism="baseline",
        regularization=None,
        augmentation=None,
        params={},
    )
    selection = select_epoch(
        TinyClassifier,
        candidate,
        x[:24],
        y[:24],
        x[24:],
        y[24:],
        device="cpu",
        seed=11,
        max_epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
    )
    assert 1 <= selection.selected_epoch <= 2
    refit = refit_fixed_epochs(
        TinyClassifier,
        candidate,
        x,
        y,
        device="cpu",
        seed=11,
        epochs=selection.selected_epoch,
        scheduler_t_max=2,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    assert refit.epochs == selection.selected_epoch
    assert refit.updates > 0
    assert len(refit.state_sha256) == 64
    assert [item["learning_rate"] for item in refit.history] == [
        item["learning_rate"] for item in selection.history[: selection.selected_epoch]
    ]
