"""Deterministic candidate and inference catalogs for exp02."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .config import Exp02Config


@dataclass(frozen=True)
class Topology:
    topology_id: str
    family: str
    hidden: int
    kernels: tuple[int, ...]
    branch_type: str
    pooling: str
    fbc_filters: int | None = None


@dataclass(frozen=True)
class Objective:
    objective_id: str
    lambda_denoise: float
    lambda_rank: float
    lambda_consistency: float
    checkpoint_metric: str


@dataclass(frozen=True)
class Exp02Candidate:
    config_id: str
    topology_id: str
    objective_id: str
    family: str
    hidden: int
    kernels: tuple[int, ...]
    branch_type: str
    pooling: str
    fbc_filters: int | None
    lambda_denoise: float
    lambda_rank: float
    lambda_consistency: float
    checkpoint_metric: str
    learning_rate: float
    weight_decay: float
    sha256: str


@dataclass(frozen=True)
class InferenceRecipe:
    inference_id: str
    config_id: str
    objective_id: str
    alpha: float
    k: int
    sha256: str


TOPOLOGIES = (
    Topology("T0", "inception", 16, (9, 17, 33, 65), "standard", "mean"),
    Topology("T1", "inception", 32, (9, 17, 33, 65), "standard", "mean"),
    Topology("T2", "inception", 16, (17, 33, 65, 129), "standard", "mean"),
    Topology("T3", "inception", 16, (9, 17, 33, 65), "depthwise", "mean"),
    Topology("T4", "inception", 16, (9, 17, 33, 65), "standard", "mean_std"),
    Topology("T5", "inception", 16, (9, 17, 33, 65), "standard", "attention"),
    Topology("T6", "fbcnet", 96, (), "fbcnet", "logvar", fbc_filters=16),
    Topology("T7", "fbcnet", 192, (), "fbcnet", "logvar", fbc_filters=32),
)

OBJECTIVES = (
    Objective("O0", 0.0, 0.0, 0.0, "classification"),
    Objective("O1", 0.25, 0.0, 0.0, "classification"),
    Objective("O2", 0.25, 0.10, 0.0, "classification"),
    Objective("O3", 1.0, 0.20, 0.10, "total"),
    Objective("O4", 1.0, 0.20, 0.10, "classification"),
    Objective("O5", 0.25, 0.10, 0.05, "classification"),
)


def build_candidate_catalog(config: Exp02Config) -> tuple[Exp02Candidate, ...]:
    candidates: list[Exp02Candidate] = []
    index = 0
    for topology in TOPOLOGIES:
        for objective in OBJECTIVES:
            index += 1
            payload = {
                "config_id": f"exp02-cfg-{index:03d}",
                **asdict(topology),
                **asdict(objective),
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
            candidates.append(Exp02Candidate(**payload, sha256=canonical_sha256(payload)))
    return tuple(candidates)


def _alpha_label(alpha: float) -> str:
    return f"{alpha:g}"


def build_inference_catalog(
    candidates: tuple[Exp02Candidate, ...],
) -> tuple[InferenceRecipe, ...]:
    recipes: list[InferenceRecipe] = []
    for candidate in candidates:
        combinations = (
            ((1.0, 0),)
            if candidate.objective_id == "O0"
            else tuple((alpha, k) for alpha in (0.0, 0.25, 0.5, 0.75) for k in (1, 4, 8))
            + ((1.0, 0),)
        )
        for alpha, k in combinations:
            payload = {
                "inference_id": (f"{candidate.config_id}__alpha-{_alpha_label(alpha)}__k{k}"),
                "config_id": candidate.config_id,
                "objective_id": candidate.objective_id,
                "alpha": alpha,
                "k": k,
            }
            recipes.append(InferenceRecipe(**payload, sha256=canonical_sha256(payload)))
    return tuple(recipes)


def catalog_payload(catalog: tuple[Exp02Candidate, ...]) -> list[dict[str, object]]:
    return [asdict(candidate) for candidate in catalog]


def inference_catalog_payload(
    catalog: tuple[InferenceRecipe, ...],
) -> list[dict[str, object]]:
    return [asdict(recipe) for recipe in catalog]
