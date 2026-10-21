"""Frozen scope and decision gates for Peterson Exp07."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.peterson_search_v12.common.partitions import (
    DISCOVERY_SUBJECTS,
    HOLDOUT_SUBJECTS,
)


@dataclass(frozen=True)
class Exp07Config:
    experiment: str = "exp07"
    profile: str = "full"
    dataset: str = "MI-OpenBCI"
    task: str = "motor_imagery_vs_rest"
    protocol: str = "within_session"
    condition: str = "overlap"
    subjects: tuple[str, ...] = DISCOVERY_SUBJECTS
    holdout_subjects: tuple[str, ...] = HOLDOUT_SUBJECTS
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    screen_seeds: tuple[int, ...] = (10, 11, 12)
    confirm_seeds: tuple[int, ...] = (13, 14)
    split_seed: int = 2026
    inner_validation_fraction: float = 0.2
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 6.25e-4
    weight_decay: float = 1e-4
    max_promoted_backbones: int = 2
    screen_min_pair_delta: float = 0.005
    final_min_e11_delta: float = 0.01
    min_subject_wins: int = 4
    max_kappa_drop: float = 0.0
    max_lower_quartile_drop: float = 0.005
    max_seed_std_increase: float = 0.005

    def __post_init__(self) -> None:
        if self.experiment != "exp07":
            raise ValueError("This configuration is restricted to exp07")
        if self.profile not in {"full", "smoke"}:
            raise ValueError("Exp07 profile must be full or smoke")
        if self.dataset != "MI-OpenBCI" or self.task != "motor_imagery_vs_rest":
            raise ValueError("Exp07 is restricted to Peterson MI versus rest")
        if self.protocol != "within_session" or self.condition != "overlap":
            raise ValueError("Exp07 requires within_session overlap")
        if set(self.subjects) & set(self.holdout_subjects):
            raise ValueError("Exp07 cannot access holdout subjects")
        if not set(self.subjects) <= set(DISCOVERY_SUBJECTS):
            raise ValueError("Exp07 contains an unknown discovery subject")
        if set(self.screen_seeds) & set(self.confirm_seeds):
            raise ValueError("Screen and confirmation seeds must be disjoint")
        if not 0 < self.inner_validation_fraction < 0.5:
            raise ValueError("Inner validation fraction must be between zero and 0.5")
        if min(self.epochs, self.batch_size, self.max_promoted_backbones) < 1:
            raise ValueError("Training budgets must be positive")
        if self.profile == "full":
            if self.subjects != DISCOVERY_SUBJECTS or self.folds != (0, 1, 2, 3, 4):
                raise ValueError("Full Exp07 requires the frozen discovery folds")
            if self.screen_seeds != (10, 11, 12) or self.confirm_seeds != (13, 14):
                raise ValueError("Full Exp07 requires the frozen new seed phases")
            if self.epochs != 300 or self.max_promoted_backbones != 2:
                raise ValueError("Full Exp07 budgets are frozen")

    @property
    def expected_screen_cells(self) -> int:
        return 12 * len(self.subjects) * len(self.screen_seeds)

    def expected_confirm_cells(self, n_promoted_backbones: int) -> int:
        if not 0 <= n_promoted_backbones <= self.max_promoted_backbones:
            raise ValueError("Confirmation exceeds the backbone promotion budget")
        return (2 + 2 * n_promoted_backbones) * len(self.subjects) * len(self.confirm_seeds)


def smoke_config() -> Exp07Config:
    return Exp07Config(
        profile="smoke",
        subjects=("S09", "S03"),
        folds=(0,),
        screen_seeds=(10,),
        confirm_seeds=(13,),
        epochs=1,
        batch_size=8,
        max_promoted_backbones=1,
        min_subject_wins=1,
    )

