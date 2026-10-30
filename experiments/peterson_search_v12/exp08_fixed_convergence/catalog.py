"""Frozen backbone catalogue for Peterson Exp08."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.catalog import BASELINE_BANDS

FBC_CONTROL_ID = "exp08-fbc-control"
E11_REFERENCE_ID = "exp08-e11-reference"


@dataclass(frozen=True)
class Exp08Candidate:
    config_id: str
    backbone: str
    aggregation: str
    mechanism: str
    family: str
    description: str
    params: dict[str, object]
    regularization: str | None
    augmentation: str | None
    fit_hook: str | None
    kind: str
    sha256: str


def _candidate(
    *,
    config_id: str,
    backbone: str,
    aggregation: str,
    mechanism: str,
    family: str,
    description: str,
    params: dict[str, object],
    kind: str = "candidate",
) -> Exp08Candidate:
    payload: dict[str, object] = {
        "config_id": config_id,
        "backbone": backbone,
        "aggregation": aggregation,
        "mechanism": mechanism,
        "family": family,
        "description": description,
        "params": params,
        "regularization": None,
        "augmentation": None,
        "fit_hook": None,
        "kind": kind,
    }
    return Exp08Candidate(**payload, sha256=canonical_sha256(payload))


def control_candidates() -> tuple[Exp08Candidate, Exp08Candidate]:
    common = {
        "bands": BASELINE_BANDS,
        "n_filters_spat": 32,
        "stride_factor": 4,
        "max_windows": 6,
    }
    return (
        _candidate(
            config_id=FBC_CONTROL_ID,
            backbone="fbcnet",
            aggregation="crop_mean",
            mechanism="fbc_control",
            family="control",
            description="Exact FBCNet control inherited without modification.",
            params={**common, "grouped_windows": False},
            kind="context_control",
        ),
        _candidate(
            config_id=E11_REFERENCE_ID,
            backbone="fbcnet",
            aggregation="lpwf",
            mechanism="global_window_weights",
            family="control",
            description="Exact e11 reference: FBCNet plus six global positional weights.",
            params={**common, "grouped_windows": True},
            kind="promoted_reference",
        ),
    )


_BACKBONES: tuple[tuple[str, str, dict[str, object], str], ...] = (
    (
        "eegnet42-k32",
        "eegnet",
        {"F1": 4, "D": 2, "F2": 8, "kernel_length": 32, "drop_prob": 0.5},
        "EEGNet-4,2 with a 32-sample temporal kernel",
    ),
    (
        "eegnet82-k32",
        "eegnet",
        {"F1": 8, "D": 2, "F2": 16, "kernel_length": 32, "drop_prob": 0.5},
        "EEGNet-8,2 with a 32-sample temporal kernel",
    ),
    (
        "eegnet82-k64",
        "eegnet",
        {"F1": 8, "D": 2, "F2": 16, "kernel_length": 64, "drop_prob": 0.5},
        "EEGNet-8,2 with a 64-sample temporal kernel",
    ),
    (
        "eegtcnet",
        "eegtcnet",
        {
            "depth_multiplier": 2,
            "filter_1": 8,
            "kern_length": 64,
            "drop_prob": 0.5,
            "depth": 2,
            "kernel_size": 4,
            "filters": 12,
        },
        "Published compact EEG-TCNet configuration",
    ),
    (
        "eegnex-8-32",
        "eegnex",
        {
            "depth_multiplier": 2,
            "filter_1": 8,
            "filter_2": 32,
            "drop_prob": 0.5,
            "kernel_block_1_2": 64,
            "kernel_block_4": 16,
            "dilation_block_4": 2,
            "avg_pool_block4": 4,
            "kernel_block_5": 16,
            "dilation_block_5": 4,
            "avg_pool_block5": 8,
        },
        "Published EEGNeX-8,32 configuration",
    ),
)


def experimental_candidates() -> tuple[Exp08Candidate, ...]:
    items: list[Exp08Candidate] = []
    for family, backbone, backbone_params, description in _BACKBONES:
        for aggregation in ("uniform", "lpwf"):
            items.append(
                _candidate(
                    config_id=f"exp08-{family}-{aggregation}",
                    backbone=backbone,
                    aggregation=aggregation,
                    mechanism=f"fixed300_{backbone}_{aggregation}",
                    family=family,
                    description=(
                        f"{description}; fixed crop-wise convergence then frozen "
                        f"{aggregation} aggregation."
                    ),
                    params={
                        **backbone_params,
                        "max_windows": 6,
                        "grouped_windows": True,
                        "training_policy": "cropwise_fixed300_frozen_fusion",
                    },
                )
            )
    return tuple(items)


def backbone_pairs() -> dict[str, tuple[str, str]]:
    pairs: dict[str, list[str]] = {}
    for item in experimental_candidates():
        pairs.setdefault(item.family, []).append(item.config_id)
    return {family: (ids[0], ids[1]) for family, ids in pairs.items()}


def all_candidates() -> tuple[Exp08Candidate, ...]:
    return (*control_candidates(), *experimental_candidates())


def candidate_by_id(config_id: str) -> Exp08Candidate:
    matches = tuple(item for item in all_candidates() if item.config_id == config_id)
    if len(matches) != 1:
        raise KeyError(f"Unknown Exp08 config_id: {config_id}")
    return matches[0]


def catalog_payload() -> list[dict[str, object]]:
    return [asdict(item) for item in all_candidates()]

