"""Frozen configuration for the Peterson V12 exp02 factorial search."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.peterson_search_v12.common.partitions import (
    DISCOVERY_SUBJECTS,
    HOLDOUT_SUBJECTS,
)


@dataclass(frozen=True)
class Exp02Config:
    experiment: str = "exp02"
    dataset: str = "MI-OpenBCI"
    task: str = "motor_imagery_vs_rest"
    protocol: str = "within_session"
    condition: str = "overlap"
    subjects: tuple[str, ...] = DISCOVERY_SUBJECTS
    folds: tuple[int, ...] = (0, 2, 4)
    seeds: tuple[int, ...] = (0,)
    split_seed: int = 2026
    epochs: int = 150
    patience: int = 35
    baseline_epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 6.25e-4
    weight_decay: float = 1e-4
    diffusion_steps: int = 1000
    rank_margin: float = 0.1
    inference_k_values: tuple[int, ...] = (1, 4, 8)
    inference_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __post_init__(self) -> None:
        if self.experiment != "exp02":
            raise ValueError("This configuration is restricted to exp02")
        if self.dataset != "MI-OpenBCI" or self.task != "motor_imagery_vs_rest":
            raise ValueError("Exp02 is restricted to Peterson MI versus rest")
        if self.protocol != "within_session" or self.condition != "overlap":
            raise ValueError("Exp02 requires within_session overlap")
        if self.folds != (0, 2, 4) or self.seeds != (0,):
            raise ValueError("Exp02 folds and seed are frozen")
        if self.diffusion_steps != 1000:
            raise ValueError("Exp02 uses the 1000-step DDPM schedule")
        if self.inference_k_values != (1, 4, 8):
            raise ValueError("Exp02 K values are frozen")
        if self.inference_alphas != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Exp02 fusion weights are frozen")
        if min(self.epochs, self.patience, self.baseline_epochs, self.batch_size) < 1:
            raise ValueError("Training budget values must be positive")
        unknown = set(self.subjects) - set(DISCOVERY_SUBJECTS) - set(HOLDOUT_SUBJECTS)
        if unknown:
            raise ValueError(f"Unknown Peterson subjects: {sorted(unknown)}")
        leaked = set(self.subjects) & set(HOLDOUT_SUBJECTS)
        if leaked:
            raise ValueError(f"exp02 cannot access holdout subjects: {sorted(leaked)}")
        if tuple(self.subjects) != DISCOVERY_SUBJECTS:
            raise ValueError("Exp02 must use the complete frozen discovery partition")
