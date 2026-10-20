"""Signed Exp06 catalogue derived from the promoted Exp05-e11 recipe."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.catalog import BASELINE_BANDS

FBC_CONTROL_ID = "exp06-fbc-control"
E11_REFERENCE_ID = "exp06-e11-reference"


@dataclass(frozen=True)
class WindowFusionCandidate:
    """One frozen Exp06 control or candidate."""

    config_id: str
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
    config_id: str,
    mechanism: str,
    family: str,
    description: str,
    *,
    params: dict[str, object] | None = None,
    regularization: str | None = None,
    fit_hook: str | None = None,
    kind: str = "candidate",
) -> WindowFusionCandidate:
    payload: dict[str, object] = {
        "config_id": config_id,
        "mechanism": mechanism,
        "family": family,
        "description": description,
        "params": {
            "bands": BASELINE_BANDS,
            "n_filters_spat": 32,
            "stride_factor": 4,
            "max_windows": 6,
            "grouped_windows": mechanism != "fbc_control",
            **(params or {}),
        },
        "regularization": regularization,
        "augmentation": None,
        "fit_hook": fit_hook,
        "kind": kind,
    }
    return WindowFusionCandidate(**payload, sha256=canonical_sha256(payload))


def control_candidates() -> tuple[WindowFusionCandidate, WindowFusionCandidate]:
    """Return the immutable FBCNet and promoted-e11 controls."""

    return (
        _candidate(
            FBC_CONTROL_ID,
            "fbc_control",
            "control",
            "Exact Exp05 FBCNet control with ordinary crop averaging.",
            kind="context_control",
        ),
        _candidate(
            E11_REFERENCE_ID,
            "global_window_weights",
            "control",
            "Exact promoted Exp05-e11 global softmax window weighting.",
            kind="promoted_reference",
        ),
    )


def build_catalog() -> tuple[WindowFusionCandidate, ...]:
    """Return ten pre-specified, low-capacity extensions of Exp05-e11."""

    return (
        _candidate(
            "exp06-e01",
            "temperature_window_weights",
            "weight_parameterization",
            "Global softmax window weights with learned positive temperature.",
            params={"initial_temperature": 1.0},
        ),
        _candidate(
            "exp06-e02",
            "sparsemax_window_weights",
            "weight_parameterization",
            "Exact-sparse global positional window weights.",
        ),
        _candidate(
            "exp06-e03",
            "gaussian_window_kernel",
            "weight_parameterization",
            "Contiguous Gaussian positional window kernel with learned center and width.",
            params={"initial_center": 2.5, "initial_scale": 1.5},
        ),
        _candidate(
            "exp06-e04",
            "residual_uniform_window_weights",
            "shrinkage",
            "Learned convex shrinkage from global weights to uniform aggregation.",
            params={"initial_blend": 0.5},
        ),
        _candidate(
            "exp06-e05",
            "class_conditional_window_weights",
            "class_specific",
            "Separate input-independent temporal weights for MI and rest logits.",
        ),
        _candidate(
            "exp06-e06",
            "residual_temporal_logit_conv",
            "local_temporal",
            "Zero-initialized depthwise width-three residual convolution over window logits.",
            params={"kernel_size": 3},
        ),
        _candidate(
            "exp06-e07",
            "csp_init_finetune_window_weights",
            "spatial_prior",
            "Train-only CSP initialization with low-LR spatial fine-tuning and global weights.",
            params={"csp_mode": "finetune", "spatial_lr_multiplier": 0.1},
            fit_hook="csp",
        ),
        _candidate(
            "exp06-e08",
            "center_loss_window_weights",
            "objective",
            "Global window weighting with weak class-center compactness.",
            params={"center_loss_weight": 1e-3},
            regularization="center_loss",
        ),
        _candidate(
            "exp06-e09",
            "quality_weighted_window_weights",
            "quality_weighting",
            "Global window weighting with bounded train-fitted artifact sample weights.",
            params={"weight_min": 0.5, "weight_max": 1.5},
            fit_hook="quality_profile",
        ),
        _candidate(
            "exp06-e10",
            "window_dropout_weights",
            "regularized_fusion",
            "Global weighting with deterministic train-only window dropout.",
            params={"window_dropout_probability": 1.0 / 6.0},
        ),
    )


def all_candidates() -> tuple[WindowFusionCandidate, ...]:
    return (*control_candidates(), *build_catalog())


def candidate_by_id(config_id: str) -> WindowFusionCandidate:
    matches = tuple(item for item in all_candidates() if item.config_id == config_id)
    if len(matches) != 1:
        raise KeyError(f"Unknown Exp06 config_id: {config_id}")
    return matches[0]


def catalog_payload() -> list[dict[str, object]]:
    return [asdict(item) for item in all_candidates()]
