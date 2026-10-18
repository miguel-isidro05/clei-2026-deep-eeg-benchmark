"""Frozen scientific scope for Peterson Exp04."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.peterson_search_v12.common.partitions import (
    DISCOVERY_SUBJECTS,
    HOLDOUT_SUBJECTS,
)


@dataclass(frozen=True)
class Exp04Config:
    """Predeclared QCR-SSNet discovery experiment."""

    experiment: str = "exp04"
    profile: str = "full"
    dataset: str = "MI-OpenBCI"
    task: str = "motor_imagery_vs_rest"
    protocol: str = "within_session"
    condition: str = "overlap"
    subjects: tuple[str, ...] = DISCOVERY_SUBJECTS
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    split_seed: int = 2026
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 6.25e-4
    weight_decay: float = 1e-4
    channel_corruption_probability: float = 0.5
    max_missing_channels: int = 2
    quality_gate_hidden: int = 8
    router_hidden: int = 16
    stress_masks: tuple[int, ...] = (1, 2)
    clean_superiority_margin: float = 0.01
    clean_noninferiority_margin: float = -0.005
    stress_superiority_margin: float = 0.03
    practical_tie: float = 0.005

    def __post_init__(self) -> None:
        if self.experiment != "exp04":
            raise ValueError("This configuration is restricted to exp04")
        if self.profile not in {"full", "smoke"}:
            raise ValueError("Exp04 profile must be full or smoke")
        if self.dataset != "MI-OpenBCI" or self.task != "motor_imagery_vs_rest":
            raise ValueError("Exp04 is restricted to Peterson MI versus rest")
        if self.protocol != "within_session" or self.condition != "overlap":
            raise ValueError("Exp04 requires within_session overlap")
        unknown = set(self.subjects) - set(DISCOVERY_SUBJECTS) - set(HOLDOUT_SUBJECTS)
        if unknown:
            raise ValueError(f"Unknown Peterson subjects: {sorted(unknown)}")
        leaked = set(self.subjects) & set(HOLDOUT_SUBJECTS)
        if leaked:
            raise ValueError(f"Exp04 cannot access holdout subjects: {sorted(leaked)}")
        if self.profile == "full":
            if self.subjects != DISCOVERY_SUBJECTS:
                raise ValueError("Exp04 must use the complete frozen discovery partition")
            if self.folds != (0, 1, 2, 3, 4) or self.seeds != (0, 1, 2, 3, 4):
                raise ValueError("Exp04 folds and seeds are frozen")
            if self.epochs != 300:
                raise ValueError("Exp04 training budget is frozen at 300 epochs")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.max_missing_channels,
                self.quality_gate_hidden,
                self.router_hidden,
            )
            < 1
        ):
            raise ValueError("Exp04 training values must be positive")
        if not 0.0 <= self.channel_corruption_probability <= 1.0:
            raise ValueError("Channel corruption probability must be in [0, 1]")


def smoke_config() -> Exp04Config:
    """Return the smallest executable Exp04 contract."""

    return Exp04Config(
        profile="smoke",
        subjects=("S09", "S03"),
        folds=(0,),
        seeds=(0,),
        epochs=1,
        batch_size=8,
    )
