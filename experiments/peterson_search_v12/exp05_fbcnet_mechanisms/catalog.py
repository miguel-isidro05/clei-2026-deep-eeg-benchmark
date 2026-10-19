"""Frozen catalogue for the Exp05 FBCNet mechanism screen.

Each entry changes one scientifically interpretable component of the local
FBCNet reference (six 8--30 Hz bands, 32 spatial filters and stride four).
The catalogue deliberately excludes routers, diffusion and channel dropout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_search_v12.common.identity import canonical_sha256

BASELINE_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 28.0), (28.0, 30.0))
FINE_BANDS = tuple((float(low), float(low + 4)) for low in range(8, 27, 2))


@dataclass(frozen=True)
class MechanismCandidate:
    """One isolated Exp05 intervention and its signed scientific identity."""

    config_id: str
    mechanism: str
    family: str
    description: str
    params: dict[str, object]
    regularization: str | None
    augmentation: str | None
    fit_hook: str | None
    sha256: str


def _candidate(
    index: int,
    mechanism: str,
    family: str,
    description: str,
    *,
    params: dict[str, object] | None = None,
    regularization: str | None = None,
    augmentation: str | None = None,
    fit_hook: str | None = None,
) -> MechanismCandidate:
    payload: dict[str, object] = {
        "config_id": f"exp05-e{index:02d}",
        "mechanism": mechanism,
        "family": family,
        "description": description,
        "params": {
            "bands": BASELINE_BANDS,
            "n_filters_spat": 32,
            "stride_factor": 4,
            **(params or {}),
        },
        "regularization": regularization,
        "augmentation": augmentation,
        "fit_hook": fit_hook,
    }
    return MechanismCandidate(**payload, sha256=canonical_sha256(payload))


def build_catalog() -> tuple[MechanismCandidate, ...]:
    """Return the frozen twenty-candidate screen in experiment order."""

    return (
        _candidate(
            1,
            "global_band_weights",
            "spectral",
            "Learn one normalized reliability weight per band.",
        ),
        _candidate(
            2, "trial_band_attention", "spectral", "Infer normalized band weights from each trial."
        ),
        _candidate(
            3,
            "regularized_band_attention",
            "spectral",
            "Trial band attention with a weak KL-to-uniform penalty.",
            regularization="attention_kl",
            params={"attention_kl_weight": 1e-3},
        ),
        _candidate(
            4,
            "mu_centered_bands",
            "spectral",
            "Recenter the first band on a train-fitted subject-specific mu peak.",
            fit_hook="mu_peak",
        ),
        _candidate(
            5,
            "constrained_sinc_filters",
            "spectral",
            "Replace the fixed bank with trainable, ordered Sinc band-pass filters.",
            params={"sinc_kernel_size": 63},
        ),
        _candidate(
            6,
            "fine_overlapping_bank",
            "spectral",
            "Use ten overlapping four-Hz bands at a two-Hz step.",
            params={"bands": FINE_BANDS},
        ),
        _candidate(
            7,
            "fbmsnet_temporal",
            "temporal",
            "Use the published mixed-scale temporal block before spatial filtering.",
            params={"fbms_kernels": (15, 31, 63, 125)},
        ),
        _candidate(
            8,
            "multiscale_logvar",
            "temporal",
            "Concatenate log-variance features at stride factors two, four and eight.",
            params={"temporal_scales": (2, 4, 8)},
        ),
        _candidate(
            9,
            "robust_clipped_logvar",
            "pooling",
            "Winsorize temporal activations before log-variance pooling.",
            params={"clip_value": 3.0},
        ),
        _candidate(
            10,
            "median_of_means_power",
            "pooling",
            "Aggregate robust power as the median of four temporal block means.",
            params={"mom_blocks": 4},
        ),
        _candidate(
            11,
            "global_window_weights",
            "window_aggregation",
            "Learn six normalized global weights for grouped sliding windows.",
            params={"max_windows": 6},
        ),
        _candidate(
            12,
            "content_window_attention",
            "window_aggregation",
            "Aggregate grouped sliding windows with content-aware attention.",
            params={"max_windows": 6},
        ),
        _candidate(
            13,
            "csp_init_frozen",
            "spatial",
            "Initialize spatial filters from train-only CSP and keep them frozen.",
            fit_hook="csp",
            params={"csp_mode": "frozen"},
        ),
        _candidate(
            14,
            "csp_init_finetune",
            "spatial",
            "Initialize from train-only CSP and fine-tune spatial filters at lower LR.",
            fit_hook="csp",
            params={"csp_mode": "finetune", "spatial_lr_multiplier": 0.1},
        ),
        _candidate(
            15,
            "spatial_orthogonality",
            "spatial",
            "Penalize redundant spatial filters independently within each band.",
            regularization="spatial_orthogonality",
            params={"orthogonality_weight": 1e-3},
        ),
        _candidate(
            16,
            "covariance_spd_head",
            "geometry",
            "Classify log-Euclidean covariance features from spatial activations.",
            params={"spd_projection_dim": 24},
        ),
        _candidate(
            17,
            "center_loss",
            "objective",
            "Add class-center compactness to the ordinary FBCNet features.",
            regularization="center_loss",
            params={"center_loss_weight": 1e-3},
        ),
        _candidate(
            18,
            "supervised_contrastive",
            "objective",
            "Add a supervised contrastive objective on normalized FBCNet features.",
            regularization="supervised_contrastive",
            params={"supcon_weight": 1e-2, "temperature": 0.1},
        ),
        _candidate(
            19,
            "ftsurrogate",
            "augmentation",
            "Apply coherent Fourier phase surrogate augmentation on train only.",
            augmentation="ftsurrogate",
            params={"augmentation_probability": 0.5, "phase_scale": 0.35},
        ),
        _candidate(
            20,
            "quality_sample_weight",
            "quality_weighting",
            "Use bounded train-fitted artifact-quality sample weights.",
            fit_hook="quality_profile",
            params={"weight_min": 0.5, "weight_max": 1.5},
        ),
    )


def baseline_candidate() -> MechanismCandidate:
    """Return the frozen local FBCNet control (not one of the 20 interventions)."""

    payload: dict[str, object] = {
        "config_id": "exp05-fbc-control",
        "mechanism": "baseline",
        "family": "baseline",
        "description": "Local FBCNet control with the frozen six-band recipe.",
        "params": {
            "bands": BASELINE_BANDS,
            "n_filters_spat": 32,
            "stride_factor": 4,
        },
        "regularization": None,
        "augmentation": None,
        "fit_hook": None,
    }
    return MechanismCandidate(**payload, sha256=canonical_sha256(payload))


def candidate_by_id(config_id: str) -> MechanismCandidate:
    """Resolve a candidate without accepting ambiguous aliases."""

    matches = tuple(
        item for item in (*build_catalog(), baseline_candidate()) if item.config_id == config_id
    )
    if len(matches) != 1:
        raise KeyError(f"Unknown Exp05 config_id: {config_id}")
    return matches[0]


def catalog_payload(items: tuple[MechanismCandidate, ...] | None = None) -> list[dict[str, object]]:
    """Return a JSON-ready, order-preserving catalogue payload."""

    return [asdict(item) for item in (build_catalog() if items is None else items)]
