"""Frozen scope and decision gates for Peterson Exp08."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.peterson_search_v12.common.partitions import (
    DISCOVERY_SUBJECTS,
    HOLDOUT_SUBJECTS,
)


@dataclass(frozen=True)
class Exp08Config:
    experiment: str = "exp08"
    profile: str = "full"
    dataset: str = "MI-OpenBCI"
    task: str = "motor_imagery_vs_rest"
    protocol: str = "within_session"
    condition: str = "overlap"
    subjects: tuple[str, ...] = DISCOVERY_SUBJECTS
    holdout_subjects: tuple[str, ...] = HOLDOUT_SUBJECTS
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    screen_seeds: tuple[int, ...] = (15, 16, 17)
    confirm_seeds: tuple[int, ...] = (18, 19)
    split_seed: int = 2026
    inner_validation_fraction: float = 0.2
    backbone_epochs: int = 300
    fusion_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 6.25e-4
    weight_decay: float = 1e-4
    fusion_learning_rate: float = 5e-2
    max_promoted_backbones: int = 2
    screen_min_pair_delta: float = 0.005
    final_min_e11_delta: float = 0.01
    min_subject_wins: int = 4
    max_kappa_drop: float = 0.0
    max_lower_quartile_drop: float = 0.005
    max_seed_std_increase: float = 0.005

    def __post_init__(self) -> None:
        if self.experiment != "exp08":
            raise ValueError("This configuration is restricted to exp08")
        if self.profile not in {"full", "smoke"}:
            raise ValueError("Exp08 profile must be full or smoke")
        if self.dataset != "MI-OpenBCI" or self.task != "motor_imagery_vs_rest":
            raise ValueError("Exp08 is restricted to Peterson MI versus rest")
        if self.protocol != "within_session" or self.condition != "overlap":
            raise ValueError("Exp08 requires within_session overlap")
        if set(self.subjects) & set(self.holdout_subjects):
            raise ValueError("Exp08 cannot access holdout subjects")
        if not set(self.subjects) <= set(DISCOVERY_SUBJECTS):
            raise ValueError("Exp08 contains an unknown discovery subject")
        if set(self.screen_seeds) & set(self.confirm_seeds):
            raise ValueError("Screen and confirmation seeds must be disjoint")
        if min(
            self.backbone_epochs,
            self.fusion_epochs,
            self.batch_size,
            self.max_promoted_backbones,
        ) < 1:
            raise ValueError("Training budgets must be positive")
        if self.profile == "full":
            if self.subjects != DISCOVERY_SUBJECTS or self.folds != (0, 1, 2, 3, 4):
                raise ValueError("Full Exp08 requires the frozen discovery folds")
            if self.screen_seeds != (15, 16, 17) or self.confirm_seeds != (18, 19):
                raise ValueError("Full Exp08 requires the frozen new seed phases")
            if self.backbone_epochs != 300 or self.fusion_epochs != 100:
                raise ValueError("Full Exp08 training budgets are frozen")

    @property
    def expected_screen_cells(self) -> int:
        return 12 * len(self.subjects) * len(self.screen_seeds)

    def expected_confirm_cells(self, n_promoted_backbones: int) -> int:
        if not 0 <= n_promoted_backbones <= self.max_promoted_backbones:
            raise ValueError("Confirmation exceeds the backbone promotion budget")
        return (2 + 2 * n_promoted_backbones) * len(self.subjects) * len(self.confirm_seeds)

    @property
    def epochs(self) -> int:
        """Compatibility budget for the unchanged FBCNet/e11 control recipe."""

        return self.backbone_epochs


def smoke_config() -> Exp08Config:
    return Exp08Config(
        profile="smoke",
        subjects=("S09", "S03"),
        folds=(0,),
        screen_seeds=(15,),
        confirm_seeds=(18,),
        backbone_epochs=1,
        fusion_epochs=2,
        batch_size=8,
        max_promoted_backbones=1,
        min_subject_wins=1,
    )

