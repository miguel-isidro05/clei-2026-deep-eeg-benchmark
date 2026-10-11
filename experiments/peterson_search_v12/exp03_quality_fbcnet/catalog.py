"""Signed recipe and quality-search catalogs for exp03."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .config import Exp03Config


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    trajectory_id: str
    checkpoint_policy: str
    weight_decay: float
    epochs: int
    patience: int | None
    learning_rate: float
    batch_size: int
    sha256: str


@dataclass(frozen=True)
class QualityCandidate:
    config_id: str
    gate_id: str
    descriptor_names: tuple[str, ...]
    channel_dropout: float
    n_filters_spat: int
    stride_factor: int
    sha256: str


def _signed(cls: type, payload: dict[str, object]):
    return cls(**payload, sha256=canonical_sha256(payload))


def build_recipe_catalog(config: Exp03Config) -> tuple[RecipeSpec, ...]:
    base = {
        "epochs": config.recipe_epochs,
        "patience": None,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
    }
    payloads = (
        {
            "recipe_id": "R0-last-wd0",
            "trajectory_id": "trajectory-wd0",
            "checkpoint_policy": "last_epoch",
            "weight_decay": 0.0,
            **base,
        },
        {
            "recipe_id": "R1-best-wd0",
            "trajectory_id": "trajectory-wd0",
            "checkpoint_policy": "best_val_ce",
            "weight_decay": 0.0,
            **base,
        },
        {
            "recipe_id": "R2-last-wd1e-4",
            "trajectory_id": "trajectory-wd1e-4",
            "checkpoint_policy": "last_epoch",
            "weight_decay": 1e-4,
            **base,
        },
        {
            "recipe_id": "R3-best-wd1e-4",
            "trajectory_id": "trajectory-wd1e-4",
            "checkpoint_policy": "best_val_ce",
            "weight_decay": 1e-4,
            **base,
        },
        {
            "recipe_id": "R4-exp02-replay",
            "trajectory_id": "trajectory-exp02-replay",
            "checkpoint_policy": "best_val_ce",
            "weight_decay": 1e-4,
            "epochs": config.replay_epochs,
            "patience": config.replay_patience,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        },
    )
    return tuple(_signed(RecipeSpec, payload) for payload in payloads)


GATE_DESCRIPTORS = {
    "G0": (),
    "G1": ("log_diff_rms", "flatline_fraction", "extreme_fraction"),
    "G2": ("log_rms", "log_diff_rms", "flatline_fraction", "extreme_fraction"),
}


def build_quality_catalog(config: Exp03Config) -> tuple[QualityCandidate, ...]:
    _ = config
    candidates: list[QualityCandidate] = []
    factors = product(GATE_DESCRIPTORS, (0.0, 0.1, 0.2), (16, 32, 48), (2, 4, 8))
    for index, (gate_id, channel_dropout, n_filters_spat, stride_factor) in enumerate(
        factors, start=1
    ):
        payload = {
            "config_id": f"exp03-qcfg-{index:03d}",
            "gate_id": gate_id,
            "descriptor_names": GATE_DESCRIPTORS[gate_id],
            "channel_dropout": channel_dropout,
            "n_filters_spat": n_filters_spat,
            "stride_factor": stride_factor,
        }
        candidates.append(_signed(QualityCandidate, payload))
    return tuple(candidates)


def catalog_payload(items: tuple[RecipeSpec, ...] | tuple[QualityCandidate, ...]):
    return [asdict(item) for item in items]

