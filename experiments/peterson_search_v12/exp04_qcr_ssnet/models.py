"""QCR-SSNet architecture and deterministic channel corruption."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from braindecode.models import FBCNet
from torch import nn

from experiments.peterson_search_v12.exp03_quality_fbcnet.models import (
    EPSILON,
    FBC_BANDS,
    QualityProfile,
)

from .catalog import QCRCandidate


@dataclass
class QCRForward:
    logits: torch.Tensor
    clean_logits: torch.Tensor
    robust_logits: torch.Tensor
    router_weight: torch.Tensor
    channel_weights: torch.Tensor


def corrupt_channels(
    x: torch.Tensor,
    *,
    probability: float,
    max_missing: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero one or more whole channels using a deterministic CPU generator."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Corruption probability must be in [0, 1]")
    if not 1 <= max_missing < x.shape[1]:
        raise ValueError("max_missing must leave at least one channel available")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mask = torch.ones((len(x), x.shape[1]), dtype=x.dtype)
    severity = torch.zeros(len(x), dtype=x.dtype)
    selected = torch.rand(len(x), generator=generator) < probability
    for row in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        missing = int(torch.randint(1, max_missing + 1, (1,), generator=generator).item())
        indices = torch.randperm(x.shape[1], generator=generator)[:missing]
        mask[row, indices] = 0.0
        severity[row] = missing / max_missing
    mask = mask.to(device=x.device)
    severity = severity.to(device=x.device)
    return x * mask.unsqueeze(-1), mask, severity


class _FBCExpert(nn.Module):
    def __init__(self, template: FBCNet) -> None:
        super().__init__()
        self.spatial_conv = template.spatial_conv
        self.padding_layer = template.padding_layer
        self.temporal_layer = template.temporal_layer
        self.flatten_layer = template.flatten_layer
        self.final_layer = template.final_layer
        self.stride_factor = template.stride_factor
        self.n_times_padded = template.n_times_padded

    def forward(self, filtered: torch.Tensor) -> torch.Tensor:
        values = self.spatial_conv(filtered)
        batch_size, channels, _, _ = values.shape
        values = self.padding_layer(values)
        values = values.view(
            batch_size,
            channels,
            self.stride_factor,
            self.n_times_padded // self.stride_factor,
        )
        values = self.temporal_layer(values)
        values = self.flatten_layer(values)
        return self.final_layer(values)


def _template(
    *, n_chans: int, n_times: int, sfreq: float, n_filters_spat: int, stride_factor: int
) -> FBCNet:
    return FBCNet(
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        sfreq=sfreq,
        n_bands=list(FBC_BANDS),
        n_filters_spat=n_filters_spat,
        temporal_layer="LogVarLayer",
        stride_factor=stride_factor,
    )


class QCRSSNet(nn.Module):
    """One network with quality-routed clean and corruption-robust experts."""

    def __init__(
        self,
        candidate: QCRCandidate,
        *,
        profile: QualityProfile,
        n_chans: int,
        n_times: int,
        sfreq: float,
        gate_hidden: int = 8,
        router_hidden: int = 16,
    ) -> None:
        super().__init__()
        if len(profile.channel_median) != n_chans:
            raise ValueError("Quality profile channel count does not match the model")
        self.candidate = candidate
        primary_filters = (
            candidate.robust_n_filters_spat
            if candidate.architecture == "robust_only"
            else candidate.clean_n_filters_spat
        )
        primary_stride = (
            candidate.robust_stride_factor
            if candidate.architecture == "robust_only"
            else candidate.clean_stride_factor
        )
        clean_template = _template(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            n_filters_spat=primary_filters,
            stride_factor=primary_stride,
        )
        self.spectral_filtering = clean_template.spectral_filtering
        self.clean_expert = _FBCExpert(clean_template)
        self.robust_expert: _FBCExpert | None = None
        if candidate.architecture == "qcr":
            robust_template = _template(
                n_chans=n_chans,
                n_times=n_times,
                sfreq=sfreq,
                n_filters_spat=candidate.robust_n_filters_spat,
                stride_factor=candidate.robust_stride_factor,
            )
            self.robust_expert = _FBCExpert(robust_template)
        self.register_buffer(
            "quality_channel_median", torch.tensor(profile.channel_median, dtype=torch.float32)
        )
        self.register_buffer(
            "quality_channel_mad", torch.tensor(profile.channel_mad, dtype=torch.float32)
        )
        self.register_buffer(
            "quality_diff_threshold", torch.tensor(profile.diff_threshold, dtype=torch.float32)
        )
        self.register_buffer(
            "quality_descriptor_mean", torch.tensor(profile.descriptor_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "quality_descriptor_std", torch.tensor(profile.descriptor_std, dtype=torch.float32)
        )
        descriptor_count = self.quality_descriptor_mean.shape[-1]
        self.quality_gate = (
            nn.Sequential(
                nn.Linear(descriptor_count, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            if candidate.use_quality_gate
            else None
        )
        if self.quality_gate is not None:
            nn.init.zeros_(self.quality_gate[-1].weight)
            nn.init.zeros_(self.quality_gate[-1].bias)
        self.router = (
            nn.Sequential(
                nn.Linear(descriptor_count * 4, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, 1),
            )
            if candidate.router_mode == "learned"
            else None
        )
        if self.router is not None:
            nn.init.zeros_(self.router[-1].weight)
            nn.init.constant_(self.router[-1].bias, -4.0)

    def _descriptors(self, x: torch.Tensor) -> torch.Tensor:
        differences = x.diff(dim=-1)
        log_rms = torch.log(torch.sqrt(torch.mean(x.square(), dim=-1)) + EPSILON)
        log_diff_rms = torch.log(torch.sqrt(torch.mean(differences.square(), dim=-1)) + EPSILON)
        flatline = (
            (differences.abs() <= self.quality_diff_threshold[None, :, None])
            .to(x.dtype)
            .mean(dim=-1)
        )
        robust_distance = (x - self.quality_channel_median[None, :, None]).abs()
        extreme = (
            (robust_distance > (6.0 * self.quality_channel_mad[None, :, None] + EPSILON))
            .to(x.dtype)
            .mean(dim=-1)
        )
        descriptors = torch.stack((log_rms, log_diff_rms, flatline, extreme), dim=-1)
        return (descriptors - self.quality_descriptor_mean[None]) / self.quality_descriptor_std[
            None
        ]

    def _router_weight(
        self, descriptors: torch.Tensor, oracle_severity: torch.Tensor | None
    ) -> torch.Tensor:
        batch = descriptors.shape[0]
        if self.candidate.architecture == "clean_only":
            return torch.zeros(batch, device=descriptors.device, dtype=descriptors.dtype)
        if self.candidate.architecture == "robust_only":
            return torch.ones(batch, device=descriptors.device, dtype=descriptors.dtype)
        if self.candidate.router_mode == "fixed_half":
            return torch.full((batch,), 0.5, device=descriptors.device, dtype=descriptors.dtype)
        if self.candidate.router_mode == "oracle":
            if oracle_severity is None:
                raise ValueError("Oracle routing requires a declared corruption severity")
            return oracle_severity.to(device=descriptors.device, dtype=descriptors.dtype)
        if self.candidate.router_mode == "learned" and self.router is not None:
            pooled = torch.cat(
                (
                    descriptors.mean(dim=1),
                    descriptors.std(dim=1, unbiased=False),
                    descriptors.amin(dim=1),
                    descriptors.amax(dim=1),
                ),
                dim=1,
            )
            return torch.sigmoid(self.router(pooled).squeeze(-1))
        raise ValueError(f"Unsupported router mode: {self.candidate.router_mode}")

    def forward(
        self, x: torch.Tensor, *, oracle_severity: torch.Tensor | None = None
    ) -> QCRForward:
        descriptors = self._descriptors(x)
        channel_weights = torch.ones((len(x), x.shape[1]), device=x.device, dtype=x.dtype)
        robust_input = x
        if self.quality_gate is not None:
            delta = 0.25 * torch.tanh(self.quality_gate(descriptors).squeeze(-1))
            channel_weights = 1.0 + delta
            channel_weights = channel_weights / channel_weights.mean(dim=1, keepdim=True)
            robust_input = x * channel_weights.unsqueeze(-1)
        clean_logits = self.clean_expert(self.spectral_filtering(x))
        if self.robust_expert is None:
            robust_logits = clean_logits
        else:
            robust_logits = self.robust_expert(self.spectral_filtering(robust_input))
        router_weight = self._router_weight(descriptors, oracle_severity)
        logits = (1.0 - router_weight[:, None]) * clean_logits + router_weight[
            :, None
        ] * robust_logits
        return QCRForward(
            logits=logits,
            clean_logits=clean_logits,
            robust_logits=robust_logits,
            router_weight=router_weight,
            channel_weights=channel_weights,
        )

    def parameter_report(self) -> dict[str, int]:
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel() for parameter in self.parameters() if not parameter.requires_grad
        )
        auxiliary_modules = [module for module in (self.quality_gate, self.router) if module]
        auxiliary = sum(
            parameter.numel()
            for module in auxiliary_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        return {
            "total": int(trainable),
            "active": int(trainable),
            "auxiliary": int(auxiliary),
            "frozen": int(frozen),
        }
