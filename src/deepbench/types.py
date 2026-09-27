"""Typed data containers shared by loaders, protocols, and reporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SubjectRecording:
    """One subject after dataset-level loading and common epoch standardization."""

    dataset: str
    subject: str
    x: np.ndarray
    y: np.ndarray
    sessions: np.ndarray
    sfreq: float
    ch_names: tuple[str, ...]
    task: str

    def validate(self) -> None:
        """Raise a descriptive error when a loader violates the common contract."""
        if self.x.ndim != 3:
            raise ValueError(f"x must have shape (trials, channels, time), got {self.x.shape}")
        n_trials, n_chans, _ = self.x.shape
        if len(self.y) != n_trials or len(self.sessions) != n_trials:
            raise ValueError("x, y, and sessions must contain the same number of trials")
        if len(self.ch_names) != n_chans:
            raise ValueError("ch_names does not match the channel dimension")
        if set(np.unique(self.y)) != {0, 1}:
            raise ValueError(f"binary labels {{0, 1}} required, got {np.unique(self.y)}")


@dataclass
class CellResult:
    """Predictions and metadata for one subject, model, seed, and protocol cell."""

    dataset: str
    task: str
    protocol: str
    condition: str
    ica_policy: str
    model: str
    subject: str
    seed: int
    split_seed: int | None
    metrics: dict[str, float]
    y_true: list[int]
    y_pred: list[int]
    y_score: list[float]
    n_train_trials: int
    n_train_examples: int
    n_test_trials: int
    fold_reports: list[dict[str, Any]] = field(default_factory=list)
