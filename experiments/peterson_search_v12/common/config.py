"""Validated immutable configuration for the staged Peterson search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .partitions import DISCOVERY_SUBJECTS, HOLDOUT_SUBJECTS


@dataclass(frozen=True)
class SearchConfig:
    experiment: str = "exp01"
    dataset: str = "MI-OpenBCI"
    task: str = "motor_imagery_vs_rest"
    protocol: str = "within_session"
    condition: str = "overlap"
    subjects: tuple[str, ...] = DISCOVERY_SUBJECTS
    folds: tuple[int, ...] = (0, 2)
    seeds: tuple[int, ...] = (0,)
    split_seed: int = 2026
    epochs: int = 80
    patience: int = 20
    batch_size: int = 64
    learning_rate: float = 6.25e-4
    weight_decay: float = 1e-4
    diffusion_steps: int = 100
    inference_k: int = 4
    rank_margin: float = 0.1
    lambda_denoise: float = 1.0
    lambda_rank: float = 0.2
    lambda_consistency: float = 0.1
    backbones: tuple[str, ...] = (
        "residual",
        "tcn",
        "inception",
        "filterbank",
        "smooth_basis",
        "conformer_lite",
    )
    formulations: tuple[str, ...] = ("discriminative", "diffusion_energy", "hybrid")
    widths: tuple[str, ...] = ("compact", "wide")

    def __post_init__(self) -> None:
        if self.experiment != "exp01":
            raise ValueError("Only exp01 is implemented; later stages require a signed decision")
        if self.dataset != "MI-OpenBCI" or self.task != "motor_imagery_vs_rest":
            raise ValueError("V12 exp01 is restricted to Peterson MI versus rest")
        if self.protocol != "within_session" or self.condition != "overlap":
            raise ValueError("V12 exp01 requires within_session with overlap windows")
        if self.folds != (0, 2) or self.seeds != (0,):
            raise ValueError("V12 exp01 folds and seed are frozen")
        if min(self.epochs, self.patience, self.batch_size) < 1:
            raise ValueError("Training budget values must be positive")
        self.validate_subjects(self.subjects)

    def validate_subjects(self, subjects: tuple[str, ...]) -> None:
        unknown = set(subjects) - set(DISCOVERY_SUBJECTS) - set(HOLDOUT_SUBJECTS)
        if unknown:
            raise ValueError(f"Unknown Peterson subjects: {sorted(unknown)}")
        leaked = set(subjects) & set(HOLDOUT_SUBJECTS)
        if leaked:
            raise ValueError(f"exp01 cannot access holdout subjects: {sorted(leaked)}")


def load_stage_config(stage: str, *, decision_dir: Path | str) -> SearchConfig:
    if stage == "exp01":
        return SearchConfig()
    previous = f"exp{int(stage[-2:]) - 1:02d}" if stage.startswith("exp") else "unknown"
    decision = Path(decision_dir) / f"{previous}.json"
    if not decision.is_file():
        raise ValueError(f"{stage} is blocked until {previous} has a signed decision")
    payload = json.loads(decision.read_text(encoding="utf-8"))
    if payload.get("status") != "approved" or not payload.get("results_sha256"):
        raise ValueError(f"{stage} decision is not approved and content-addressed")
    raise NotImplementedError(f"{stage} has not been specified yet")

