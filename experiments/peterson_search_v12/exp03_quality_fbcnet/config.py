"""Frozen scientific scope for Peterson exp03."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.peterson_search_v12.common.partitions import (
    DISCOVERY_SUBJECTS,
    HOLDOUT_SUBJECTS,
)


@dataclass(frozen=True)
class Exp03Config:
    experiment: str = "exp03"
    profile: str = "full"
    dataset: str = "MI-OpenBCI"
    task: str = "motor_imagery_vs_rest"
    protocol: str = "within_session"
    condition: str = "overlap"
    subjects: tuple[str, ...] = DISCOVERY_SUBJECTS
    recipe_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    recipe_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    quality_folds: tuple[int, ...] = (0, 2, 4)
    quality_seeds: tuple[int, ...] = (0,)
    split_seed: int = 2026
    recipe_epochs: int = 300
    replay_epochs: int = 150
    replay_patience: int = 35
    batch_size: int = 64
    learning_rate: float = 6.25e-4
    quality_gate_hidden: int = 8
    quality_stress_masks: tuple[int, ...] = (1, 2)

    def __post_init__(self) -> None:
        if self.experiment != "exp03":
            raise ValueError("This configuration is restricted to exp03")
        if self.profile not in {"full", "smoke"}:
            raise ValueError("Exp03 profile must be full or smoke")
        if self.dataset != "MI-OpenBCI" or self.task != "motor_imagery_vs_rest":
            raise ValueError("Exp03 is restricted to Peterson MI versus rest")
        if self.protocol != "within_session" or self.condition != "overlap":
            raise ValueError("Exp03 requires within_session overlap")
        unknown = set(self.subjects) - set(DISCOVERY_SUBJECTS) - set(HOLDOUT_SUBJECTS)
        if unknown:
            raise ValueError(f"Unknown Peterson subjects: {sorted(unknown)}")
        leaked = set(self.subjects) & set(HOLDOUT_SUBJECTS)
        if leaked:
            raise ValueError(f"exp03 cannot access holdout subjects: {sorted(leaked)}")
        if self.profile == "full":
            if tuple(self.subjects) != DISCOVERY_SUBJECTS:
                raise ValueError("Exp03 must use the complete frozen discovery partition")
            if self.recipe_folds != (0, 1, 2, 3, 4) or self.recipe_seeds != (0, 1, 2, 3, 4):
                raise ValueError("Exp03 recipe folds and seeds are frozen")
            if self.quality_folds != (0, 2, 4) or self.quality_seeds != (0,):
                raise ValueError("Exp03 quality folds and seeds are frozen")
            if (self.recipe_epochs, self.replay_epochs, self.replay_patience) != (300, 150, 35):
                raise ValueError("Exp03 training budgets are frozen")
        if (
            min(
                self.recipe_epochs,
                self.replay_epochs,
                self.replay_patience,
                self.batch_size,
                self.quality_gate_hidden,
            )
            < 1
        ):
            raise ValueError("Exp03 training values must be positive")


def smoke_config() -> Exp03Config:
    return Exp03Config(
        profile="smoke",
        subjects=("S09", "S03"),
        recipe_folds=(0,),
        recipe_seeds=(0,),
        quality_folds=(0,),
        quality_seeds=(0,),
        recipe_epochs=1,
        replay_epochs=1,
        replay_patience=1,
        batch_size=8,
    )
