"""Frozen configuration contracts for Peterson diffusion V11."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from deepbench.config import MI_SUBJECTS, PAPER_EPOCHS, SPLIT_SEED

from .identity import canonical_sha256

ARCHITECTURES = ("res", "filterbank", "smooth_basis", "conformer")
FORMULATIONS = ("diffusion", "discriminative")
CONDITIONS = ("full", "center_x6", "overlap")
OBJECTIVES = ("noise_only", "noise_rank", "noise_rank_consistency")
BASELINE_MODELS = ("CSP+LDA", "EEGNet", "FBCNet", "ShallowConvNet", "EEGConformer")
V10_BASE_REVISION = "6c116d5"


@dataclass(frozen=True)
class ExperimentConfig:
    """Scientific configuration whose canonical hash identifies every V11 cell."""

    wave: int = 0
    dataset: str = "MI-OpenBCI"
    task: str = "motor_imagery_vs_rest"
    protocol: str = "within_session"
    subjects: tuple[str, ...] = MI_SUBJECTS
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    split_seed: int = SPLIT_SEED
    epochs: int = PAPER_EPOCHS
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 75
    diffusion_steps: int = 100
    inference_k: int = 4
    hidden: int = 32
    objective: str = "noise_rank_consistency"
    rank_margin: float = 0.25
    lambda_rank: float = 1.0
    lambda_consistency: float = 0.1
    v10_base_revision: str = V10_BASE_REVISION

    def __post_init__(self) -> None:
        if self.dataset != "MI-OpenBCI":
            raise ValueError("V11 accepts only the Peterson MI-OpenBCI dataset")
        if self.task != "motor_imagery_vs_rest":
            raise ValueError("V11 preserves the Peterson motor_imagery_vs_rest task")
        if self.protocol == "loso":
            raise ValueError("LOSO is outside the V11 exploratory program")
        if self.protocol != "within_session":
            raise ValueError("V11 permits only within_session")
        if not 0 <= self.wave <= 5:
            raise ValueError("wave must be between 0 and 5")
        unknown_subjects = set(self.subjects) - set(MI_SUBJECTS)
        if unknown_subjects:
            raise ValueError(f"Unknown Peterson subject(s): {sorted(unknown_subjects)}")
        unknown_conditions = set(self.conditions) - set(CONDITIONS)
        if unknown_conditions:
            raise ValueError(f"Unknown V11 condition(s): {sorted(unknown_conditions)}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"Unknown diffusion objective: {self.objective}")
        if self.epochs < 1 or self.batch_size < 1 or self.diffusion_steps < 2:
            raise ValueError("epochs, batch_size and diffusion_steps must be positive")
        if self.inference_k < 1:
            raise ValueError("inference_k must be positive")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_wave_config(wave: int) -> ExperimentConfig:
    path = Path(__file__).with_name("configs") / f"wave_{wave:02d}.json"
    if not path.exists():
        raise ValueError(f"Wave {wave} has no frozen configuration: {path}")
    payload = json.loads(path.read_text())
    for field in ("subjects", "conditions", "seeds"):
        if field in payload:
            payload[field] = tuple(payload[field])
    return ExperimentConfig(**payload)
