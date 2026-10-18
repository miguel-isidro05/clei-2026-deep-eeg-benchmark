"""Signed Exp04 controls and QCR-SSNet ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .config import Exp04Config
from .source import Exp04Source


@dataclass(frozen=True)
class QCRCandidate:
    """One predeclared architecture/loss configuration."""

    config_id: str
    role: str
    architecture: str
    clean_n_filters_spat: int
    robust_n_filters_spat: int
    clean_stride_factor: int
    robust_stride_factor: int
    channel_dropout: float
    router_mode: str
    use_quality_gate: bool
    lambda_corrupt: float
    lambda_consistency: float
    lambda_router: float
    lambda_specialization: float
    sha256: str


def _candidate(**payload: object) -> QCRCandidate:
    return QCRCandidate(**payload, sha256=canonical_sha256(payload))


def build_model_catalog(config: Exp04Config, source: Exp04Source) -> tuple[QCRCandidate, ...]:
    """Build controls followed by causal QCR ablations."""

    common = {
        "clean_n_filters_spat": source.clean_n_filters_spat,
        "robust_n_filters_spat": source.robust_n_filters_spat,
        "clean_stride_factor": source.clean_stride_factor,
        "robust_stride_factor": source.robust_stride_factor,
    }
    payloads = (
        {
            "config_id": "exp04-c00-clean",
            "role": "clean_control",
            "architecture": "clean_only",
            "channel_dropout": 0.0,
            "router_mode": "none",
            "use_quality_gate": False,
            "lambda_corrupt": 0.0,
            "lambda_consistency": 0.0,
            "lambda_router": 0.0,
            "lambda_specialization": 0.0,
        },
        {
            "config_id": "exp04-c01-robust",
            "role": "robust_control",
            "architecture": "robust_only",
            "channel_dropout": source.robust_channel_dropout,
            "router_mode": "none",
            "use_quality_gate": False,
            "lambda_corrupt": 1.0,
            "lambda_consistency": 0.0,
            "lambda_router": 0.0,
            "lambda_specialization": 0.0,
        },
        {
            "config_id": "exp04-a00-equal",
            "role": "qcr_ablation",
            "architecture": "qcr",
            "channel_dropout": config.channel_corruption_probability,
            "router_mode": "fixed_half",
            "use_quality_gate": False,
            "lambda_corrupt": 1.0,
            "lambda_consistency": 0.0,
            "lambda_router": 0.0,
            "lambda_specialization": 0.5,
        },
        {
            "config_id": "exp04-a01-oracle",
            "role": "qcr_ablation",
            "architecture": "qcr",
            "channel_dropout": config.channel_corruption_probability,
            "router_mode": "oracle",
            "use_quality_gate": False,
            "lambda_corrupt": 1.0,
            "lambda_consistency": 0.0,
            "lambda_router": 0.0,
            "lambda_specialization": 0.5,
        },
        {
            "config_id": "exp04-a02-router",
            "role": "qcr_ablation",
            "architecture": "qcr",
            "channel_dropout": config.channel_corruption_probability,
            "router_mode": "learned",
            "use_quality_gate": False,
            "lambda_corrupt": 1.0,
            "lambda_consistency": 0.0,
            "lambda_router": 0.2,
            "lambda_specialization": 0.5,
        },
        {
            "config_id": "exp04-a03-consistency",
            "role": "qcr_ablation",
            "architecture": "qcr",
            "channel_dropout": config.channel_corruption_probability,
            "router_mode": "learned",
            "use_quality_gate": False,
            "lambda_corrupt": 1.0,
            "lambda_consistency": 0.1,
            "lambda_router": 0.2,
            "lambda_specialization": 0.5,
        },
        {
            "config_id": "exp04-a04-gate",
            "role": "qcr_ablation",
            "architecture": "qcr",
            "channel_dropout": config.channel_corruption_probability,
            "router_mode": "learned",
            "use_quality_gate": True,
            "lambda_corrupt": 1.0,
            "lambda_consistency": 0.0,
            "lambda_router": 0.2,
            "lambda_specialization": 0.5,
        },
        {
            "config_id": "exp04-qcr-full",
            "role": "qcr_full",
            "architecture": "qcr",
            "channel_dropout": config.channel_corruption_probability,
            "router_mode": "learned",
            "use_quality_gate": True,
            "lambda_corrupt": 1.0,
            "lambda_consistency": 0.1,
            "lambda_router": 0.2,
            "lambda_specialization": 0.5,
        },
        {
            "config_id": "exp04-a05-no-router-loss",
            "role": "qcr_ablation",
            "architecture": "qcr",
            "channel_dropout": config.channel_corruption_probability,
            "router_mode": "learned",
            "use_quality_gate": True,
            "lambda_corrupt": 1.0,
            "lambda_consistency": 0.1,
            "lambda_router": 0.0,
            "lambda_specialization": 0.5,
        },
        {
            "config_id": "exp04-a06-no-consistency",
            "role": "qcr_ablation",
            "architecture": "qcr",
            "channel_dropout": config.channel_corruption_probability,
            "router_mode": "learned",
            "use_quality_gate": True,
            "lambda_corrupt": 1.0,
            "lambda_consistency": 0.0,
            "lambda_router": 0.2,
            "lambda_specialization": 0.5,
        },
    )
    catalog = tuple(_candidate(**common, **payload) for payload in payloads)
    if config.profile == "smoke":
        selected = {
            "exp04-c00-clean",
            "exp04-c01-robust",
            "exp04-a02-router",
            "exp04-qcr-full",
        }
        return tuple(candidate for candidate in catalog if candidate.config_id in selected)
    return catalog


def catalog_payload(catalog: tuple[QCRCandidate, ...]) -> list[dict[str, object]]:
    return [asdict(candidate) for candidate in catalog]
