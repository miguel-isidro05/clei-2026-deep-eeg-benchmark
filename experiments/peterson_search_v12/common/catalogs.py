"""Deterministic exp01 candidate catalog."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_diffusion_v11.identity import canonical_sha256

from .config import SearchConfig

WIDTH_HIDDEN = {"compact": 16, "wide": 32}


@dataclass(frozen=True)
class CandidateSpec:
    config_id: str
    backbone: str
    formulation: str
    width: str
    hidden: int
    learning_rate: float
    weight_decay: float
    sha256: str


def build_exp01_catalog(config: SearchConfig) -> tuple[CandidateSpec, ...]:
    candidates: list[CandidateSpec] = []
    index = 0
    for backbone in config.backbones:
        for formulation in config.formulations:
            for width in config.widths:
                index += 1
                payload = {
                    "config_id": f"exp01-cfg-{index:03d}",
                    "backbone": backbone,
                    "formulation": formulation,
                    "width": width,
                    "hidden": WIDTH_HIDDEN[width],
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                }
                candidates.append(CandidateSpec(**payload, sha256=canonical_sha256(payload)))
    return tuple(candidates)


def catalog_payload(catalog: tuple[CandidateSpec, ...]) -> list[dict[str, object]]:
    return [asdict(candidate) for candidate in catalog]
