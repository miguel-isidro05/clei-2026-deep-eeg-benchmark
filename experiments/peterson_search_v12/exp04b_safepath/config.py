"""Frozen scope and budgets for Peterson Exp04b."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.peterson_search_v12.common.partitions import (
    DISCOVERY_SUBJECTS,
    HOLDOUT_SUBJECTS,
)


@dataclass(frozen=True)
class Exp04bConfig:
    """Discovery-only SafePath search contract."""

    experiment: str = "exp04b"
    profile: str = "full"
    dataset: str = "MI-OpenBCI"
    task: str = "motor_imagery_vs_rest"
    protocol: str = "within_session"
    condition: str = "overlap"
    subjects: tuple[str, ...] = DISCOVERY_SUBJECTS
    holdout_subjects: tuple[str, ...] = HOLDOUT_SUBJECTS
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    split_seed: int = 2026
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 6.25e-4
    weight_decay: float = 1e-4
    max_missing_channels: int = 2
    clean_noninferiority_margin: float = -0.005
    stress_superiority_margin: float = 0.03

    def __post_init__(self) -> None:
        if self.experiment != "exp04b":
            raise ValueError("This configuration is restricted to exp04b")
        if self.profile not in {"full", "smoke"}:
            raise ValueError("Exp04b profile must be full or smoke")
        if self.dataset != "MI-OpenBCI" or self.task != "motor_imagery_vs_rest":
            raise ValueError("Exp04b is restricted to Peterson MI versus rest")
        if self.protocol != "within_session" or self.condition != "overlap":
            raise ValueError("Exp04b requires within_session overlap")
        leaked = set(self.subjects) & set(self.holdout_subjects)
        if leaked:
            raise ValueError(f"Exp04b cannot access holdout subjects: {sorted(leaked)}")
        if not set(self.subjects) <= set(DISCOVERY_SUBJECTS):
            raise ValueError("Exp04b contains an unknown discovery subject")
        if self.profile == "full":
            if self.subjects != DISCOVERY_SUBJECTS:
                raise ValueError("Full Exp04b requires the frozen discovery partition")
            if self.folds != (0, 1, 2, 3, 4) or self.seeds != (0, 1, 2, 3, 4):
                raise ValueError("Full Exp04b folds and seeds are frozen")
            if self.epochs != 300:
                raise ValueError("Full Exp04b training is fixed at 300 epochs")
        if min(self.epochs, self.batch_size, self.max_missing_channels) < 1:
            raise ValueError("Exp04b training values must be positive")

    @property
    def wave_jobs(self) -> dict[str, int]:
        return {"w1": 144, "w2": 144, "w3": 186}

    @property
    def total_training_jobs(self) -> int:
        return sum(self.wave_jobs.values())


def smoke_config() -> Exp04bConfig:
    """Return a two-shard, one-fold executable smoke contract."""

    return Exp04bConfig(
        profile="smoke",
        subjects=("S09", "S03"),
        folds=(0,),
        seeds=(0,),
        epochs=1,
        batch_size=8,
    )
