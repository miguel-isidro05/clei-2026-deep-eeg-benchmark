"""Compact FBCNet mechanism models and training hooks for Exp05."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from braindecode.models import FBCNet, FBMSNet
from torch import nn
from torch.nn import functional as F

from .catalog import MechanismCandidate, baseline_candidate
from .mechanisms import (
    EPSILON,
    ConstrainedSincFilterBank,
    GlobalBandWeights,
    TrialBandAttention,
    clipped_log_variance,
    fourier_surrogate,
    median_of_means_log_power,
    quality_descriptors,
    segmented_log_variance,
)


def _bands(candidate: MechanismCandidate, fit_context: dict[str, object] | None):
    if fit_context is not None and "bands" in fit_context:
        return tuple(tuple(float(edge) for edge in band) for band in fit_context["bands"])
    return tuple(tuple(float(edge) for edge in band) for band in candidate.params["bands"])


class FBCMechanismModel(nn.Module):
    """Official FBCNet stem with one optional, isolated Exp05 intervention."""

    def __init__(
        self,
        candidate: MechanismCandidate,
        *,
        n_chans: int,
        n_times: int,
        sfreq: float,
        fit_context: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.bands = _bands(candidate, fit_context)
        self.n_filters_spat = int(candidate.params["n_filters_spat"])
        self.stride_factor = int(candidate.params["stride_factor"])
        self.base = FBCNet(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
            sfreq=sfreq,
            n_bands=list(self.bands),
            n_filters_spat=self.n_filters_spat,
            temporal_layer="LogVarLayer",
            stride_factor=self.stride_factor,
        )
        if candidate.mechanism == "constrained_sinc_filters":
            self.base.spectral_filtering = ConstrainedSincFilterBank(
                self.bands,
                sfreq=sfreq,
                kernel_size=int(candidate.params["sinc_kernel_size"]),
            )
        n_bands = len(self.bands)
        self.band_transform: nn.Module | None = None
        if candidate.mechanism == "global_band_weights":
            self.band_transform = GlobalBandWeights(n_bands)
        elif candidate.mechanism in {"trial_band_attention", "regularized_band_attention"}:
            self.band_transform = TrialBandAttention(n_bands)

        base_features = n_bands * self.n_filters_spat * self.stride_factor
        if candidate.mechanism == "multiscale_logvar":
            self.feature_dim = (
                n_bands
                * self.n_filters_spat
                * sum(int(value) for value in candidate.params["temporal_scales"])
            )
        elif candidate.mechanism == "median_of_means_power":
            self.feature_dim = n_bands * self.n_filters_spat
        else:
            self.feature_dim = base_features

        self.spd_projection: nn.Conv1d | None = None
        self.spd_indices: torch.Tensor | None = None
        if candidate.mechanism == "covariance_spd_head":
            projection_dim = int(candidate.params["spd_projection_dim"])
            self.spd_projection = nn.Conv1d(n_bands * self.n_filters_spat, projection_dim, 1)
            indices = torch.triu_indices(projection_dim, projection_dim)
            self.register_buffer("_spd_indices", indices)
            self.feature_dim = projection_dim * (projection_dim + 1) // 2

        if self.feature_dim == base_features and candidate.mechanism != "covariance_spd_head":
            self.classifier = self.base.final_layer
        else:
            self.base.final_layer = nn.Identity()
            self.classifier = nn.Linear(self.feature_dim, 2)

        self.window_logits: nn.Parameter | None = None
        self.window_attention: nn.Linear | None = None
        if candidate.mechanism == "global_window_weights":
            self.window_logits = nn.Parameter(torch.zeros(int(candidate.params["max_windows"])))
        elif candidate.mechanism == "content_window_attention":
            self.window_attention = nn.Linear(self.feature_dim, 1)
            nn.init.zeros_(self.window_attention.weight)
            nn.init.zeros_(self.window_attention.bias)

        self.centers: nn.Parameter | None = None
        if candidate.regularization == "center_loss":
            self.centers = nn.Parameter(torch.zeros(2, self.feature_dim))

        self.last_band_weights: torch.Tensor | None = None
        self.last_window_weights: torch.Tensor | None = None
        if candidate.fit_hook == "csp":
            self._initialize_csp(fit_context)

    def _initialize_csp(self, fit_context: dict[str, object] | None) -> None:
        if fit_context is None or "csp_filters" not in fit_context:
            raise ValueError("CSP candidates require a train-fitted csp_filters context")
        filters = torch.as_tensor(fit_context["csp_filters"], dtype=torch.float32)
        expected = (len(self.bands), self.n_filters_spat, self.base.n_chans)
        if tuple(filters.shape) != expected:
            raise ValueError(f"CSP filter shape {tuple(filters.shape)} != {expected}")
        convolution = self.base.spatial_conv[0]
        original_weight = convolution.parametrizations.weight.original
        with torch.no_grad():
            original_weight.copy_(filters.reshape(-1, 1, self.base.n_chans, 1))
        if self.candidate.params["csp_mode"] == "frozen":
            original_weight.requires_grad_(False)

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        filtered = self.base.spectral_filtering(x)
        if self.band_transform is not None:
            filtered, weights = self.band_transform(filtered)
            self.last_band_weights = weights
        else:
            self.last_band_weights = None
        return self.base.spatial_conv(filtered).squeeze(2)

    def _spd_features(self, spatial: torch.Tensor) -> torch.Tensor:
        if self.spd_projection is None:
            raise RuntimeError("SPD projection is unavailable")
        projected = self.spd_projection(spatial)
        centered = projected - projected.mean(dim=-1, keepdim=True)
        covariance = centered @ centered.transpose(1, 2) / max(1, projected.shape[-1] - 1)
        identity = torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)
        stabilized = covariance + EPSILON * identity
        # ``eigh`` is not implemented by PyTorch's MPS backend.  Keep CUDA on
        # device for Cayetano and use a differentiable CPU transfer only for
        # the local Mac smoke path.
        eig_input = stabilized.cpu() if stabilized.device.type == "mps" else stabilized
        eigenvalues, eigenvectors = torch.linalg.eigh(eig_input)
        if stabilized.device.type == "mps":
            eigenvalues = eigenvalues.to(stabilized.device)
            eigenvectors = eigenvectors.to(stabilized.device)
        log_covariance = eigenvectors * eigenvalues.clamp_min(EPSILON).log().unsqueeze(1)
        log_covariance = log_covariance @ eigenvectors.transpose(1, 2)
        indices = self._spd_indices
        if indices is None:
            raise RuntimeError("SPD triangular indices are unavailable")
        return log_covariance[:, indices[0], indices[1]]

    def _official_logvar_features(self, spatial: torch.Tensor) -> torch.Tensor:
        padded = self.base.padding_layer(spatial)
        reshaped = padded.view(
            len(padded),
            padded.shape[1],
            self.stride_factor,
            self.base.n_times_padded // self.stride_factor,
        )
        return self.base.flatten_layer(self.base.temporal_layer(reshaped))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self._spatial(x)
        if self.candidate.mechanism == "multiscale_logvar":
            return torch.cat(
                [
                    segmented_log_variance(spatial, int(scale))
                    for scale in self.candidate.params["temporal_scales"]
                ],
                dim=1,
            )
        if self.candidate.mechanism == "robust_clipped_logvar":
            return clipped_log_variance(
                spatial,
                stride=self.stride_factor,
                clip_value=float(self.candidate.params["clip_value"]),
            )
        if self.candidate.mechanism == "median_of_means_power":
            return median_of_means_log_power(
                spatial, blocks=int(self.candidate.params["mom_blocks"])
            )
        if self.candidate.mechanism == "covariance_spd_head":
            return self._spd_features(spatial)
        return self._official_logvar_features(spatial)

    def _single_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(x)
        return self.classifier(features), features

    def _grouped_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, windows, channels, n_times = x.shape
        logits, features = self._single_forward(x.reshape(batch * windows, channels, n_times))
        logits = logits.view(batch, windows, -1)
        features = features.view(batch, windows, -1)
        if self.window_logits is not None:
            if windows > len(self.window_logits):
                raise ValueError("Grouped input exceeds configured max_windows")
            weights = torch.softmax(self.window_logits[:windows], dim=0).expand(batch, -1)
        elif self.window_attention is not None:
            weights = torch.softmax(self.window_attention(features).squeeze(-1), dim=1)
        else:
            weights = torch.full((batch, windows), 1.0 / windows, device=x.device, dtype=x.dtype)
        self.last_window_weights = weights
        return (logits * weights.unsqueeze(-1)).sum(dim=1), (features * weights.unsqueeze(-1)).sum(
            dim=1
        )

    def forward(
        self, x: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 4:
            logits, features = self._grouped_forward(x)
        elif x.ndim == 3:
            logits, features = self._single_forward(x)
        else:
            raise ValueError("Exp05 models expect (B,C,T) or grouped (B,W,C,T) input")
        return (logits, features) if return_features else logits

    def spatial_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.base.spatial_conv[0].parameters()

    def parameter_report(self) -> dict[str, int]:
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel() for parameter in self.parameters() if not parameter.requires_grad
        )
        return {
            "total": int(trainable + frozen),
            "trainable": int(trainable),
            "frozen": int(frozen),
        }


class FBMSMechanismModel(nn.Module):
    """Braindecode FBMSNet exposed through the common Exp05 feature API."""

    def __init__(
        self, candidate: MechanismCandidate, *, n_chans: int, n_times: int, sfreq: float
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.base = FBMSNet(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
            sfreq=sfreq,
            n_bands=list(_bands(candidate, None)),
            n_filters_spat=32,
            stride_factor=4,
            kernels_weights=tuple(int(value) for value in candidate.params["fbms_kernels"]),
        )
        self.feature_dim = self.base.out_channels_spatial * self.base.stride_factor

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.base.spectral_filtering(x)
        features = self.base.mix_conv(features)
        features = self.base.spatial_conv(features)
        features = self.base.padding_layer(features)
        features = features.view(
            len(features), self.base.out_channels_spatial, self.base.stride_factor, -1
        )
        features = self.base.flatten_layer(self.base.temporal_layer(features))
        return self.base.final_layer(features), features

    def forward(
        self, x: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("FBMS candidate expects (B,C,T) input")
        logits, features = self._forward(x)
        return (logits, features) if return_features else logits

    def parameter_report(self) -> dict[str, int]:
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel() for parameter in self.parameters() if not parameter.requires_grad
        )
        return {
            "total": int(trainable + frozen),
            "trainable": int(trainable),
            "frozen": int(frozen),
        }


def build_model(
    candidate: MechanismCandidate,
    *,
    n_chans: int,
    n_times: int,
    sfreq: float,
    fit_context: dict[str, object] | None = None,
) -> nn.Module:
    """Construct one candidate; the caller owns global RNG seeding."""

    if candidate.mechanism == "fbmsnet_temporal":
        return FBMSMechanismModel(candidate, n_chans=n_chans, n_times=n_times, sfreq=sfreq)
    return FBCMechanismModel(
        candidate,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=sfreq,
        fit_context=fit_context,
    )


def build_baseline_model(*, n_chans: int, n_times: int, sfreq: float) -> nn.Module:
    return build_model(baseline_candidate(), n_chans=n_chans, n_times=n_times, sfreq=sfreq)


def optimizer_parameter_groups(
    model: nn.Module, candidate: MechanismCandidate, *, base_lr: float
) -> list[dict[str, object]]:
    """Return disjoint optimizer groups, including the CSP low-LR contract."""

    if candidate.mechanism != "csp_init_finetune" or not isinstance(model, FBCMechanismModel):
        return [{"params": list(model.parameters()), "lr": float(base_lr)}]
    spatial_ids = {id(parameter) for parameter in model.spatial_parameters()}
    spatial = [parameter for parameter in model.parameters() if id(parameter) in spatial_ids]
    remaining = [parameter for parameter in model.parameters() if id(parameter) not in spatial_ids]
    return [
        {"params": remaining, "lr": float(base_lr)},
        {
            "params": spatial,
            "lr": float(base_lr) * float(candidate.params["spatial_lr_multiplier"]),
        },
    ]


def auxiliary_loss(
    model: nn.Module,
    logits: torch.Tensor,
    features: torch.Tensor,
    y: torch.Tensor,
    candidate: MechanismCandidate,
) -> torch.Tensor:
    """Candidate-specific auxiliary objective; classification CE remains external."""

    _ = logits
    zero = features.sum() * 0.0
    if candidate.regularization is None:
        return zero
    if not isinstance(model, FBCMechanismModel):
        raise TypeError("Auxiliary Exp05 losses require FBCMechanismModel")
    if candidate.regularization == "attention_kl":
        weights = model.last_band_weights
        if weights is None:
            raise RuntimeError("Attention weights were not produced before auxiliary_loss")
        probability = weights / weights.shape[1]
        kl = (
            (probability * (probability.clamp_min(EPSILON).log() + math.log(weights.shape[1])))
            .sum(dim=1)
            .mean()
        )
        return kl * float(candidate.params["attention_kl_weight"])
    if candidate.regularization == "spatial_orthogonality":
        weight = model.base.spatial_conv[0].weight.squeeze(1).squeeze(-1)
        weight = weight.view(len(model.bands), model.n_filters_spat, -1)
        normalized = F.normalize(weight, dim=-1)
        gram = normalized @ normalized.transpose(1, 2)
        identity = torch.eye(model.n_filters_spat, device=gram.device, dtype=gram.dtype)
        penalty = (gram - identity).square().mean()
        return penalty * float(candidate.params["orthogonality_weight"])
    if candidate.regularization == "center_loss":
        if model.centers is None:
            raise RuntimeError("Center-loss parameters are unavailable")
        penalty = (features - model.centers[y]).square().sum(dim=1).mean()
        return penalty * float(candidate.params["center_loss_weight"])
    if candidate.regularization == "supervised_contrastive":
        normalized = F.normalize(features, dim=1)
        similarity = normalized @ normalized.T / float(candidate.params["temperature"])
        identity = torch.eye(len(y), device=y.device, dtype=torch.bool)
        positives = y[:, None].eq(y[None, :]) & ~identity
        valid = positives.any(dim=1)
        if not valid.any():
            return zero
        masked = similarity.masked_fill(identity, float("-inf"))
        log_probability = masked - torch.logsumexp(masked, dim=1, keepdim=True)
        per_example = -(
            log_probability.masked_fill(~positives, 0.0).sum(dim=1)
            / positives.sum(dim=1).clamp_min(1)
        )
        return per_example[valid].mean() * float(candidate.params["supcon_weight"])
    raise ValueError(f"Unsupported regularization: {candidate.regularization}")


def augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    candidate: MechanismCandidate,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply deterministic train-only augmentation selected by the catalogue."""

    if candidate.augmentation is None:
        return x, y
    if candidate.augmentation != "ftsurrogate":
        raise ValueError(f"Unsupported augmentation: {candidate.augmentation}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected = torch.rand(len(x), generator=generator) < float(
        candidate.params["augmentation_probability"]
    )
    if not selected.any():
        return x, y
    augmented = x.clone()
    selected_device = selected.to(x.device)
    augmented[selected_device] = fourier_surrogate(
        x[selected_device],
        phase_scale=float(candidate.params["phase_scale"]),
        seed=seed + 1,
    )
    return augmented, y


def sample_weights(
    x: np.ndarray,
    candidate: MechanismCandidate,
    fit_context: dict[str, object] | None,
) -> np.ndarray:
    """Return bounded mean-one weights from a train-fitted quality profile."""

    if candidate.fit_hook != "quality_profile":
        return np.ones(len(x), dtype=np.float32)
    if fit_context is None or fit_context.get("fit_partition") != "train_only":
        raise ValueError("Quality weighting requires a train-only fit context")
    descriptors = quality_descriptors(x)
    median = np.asarray(fit_context["descriptor_median"], dtype=np.float64)
    mad = np.asarray(fit_context["descriptor_mad"], dtype=np.float64)
    artifact_score = np.mean(np.abs((descriptors - median) / np.maximum(mad, EPSILON)), axis=1)
    minimum = float(candidate.params["weight_min"])
    maximum = float(candidate.params["weight_max"])
    log_weight = -0.25 * artifact_score
    lower, upper = -20.0, 20.0
    for _ in range(64):
        offset = 0.5 * (lower + upper)
        trial = np.clip(np.exp(log_weight + offset), minimum, maximum)
        if trial.mean() < 1.0:
            lower = offset
        else:
            upper = offset
    weights = np.clip(np.exp(log_weight + 0.5 * (lower + upper)), minimum, maximum)
    return weights.astype(np.float32)
