"""FBCNet with train-fitted channel-quality gating and channel dropout."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from braindecode.models import FBCNet
from torch import nn

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import QualityCandidate

FBC_BANDS = ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30))
EPSILON = 1e-6


def _raw_descriptors(
    x: np.ndarray,
    *,
    channel_median: np.ndarray,
    channel_mad: np.ndarray,
    diff_threshold: np.ndarray,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    differences = np.diff(values, axis=-1)
    log_rms = np.log(np.sqrt(np.mean(values**2, axis=-1)) + EPSILON)
    log_diff_rms = np.log(np.sqrt(np.mean(differences**2, axis=-1)) + EPSILON)
    flatline = np.mean(np.abs(differences) <= diff_threshold[None, :, None], axis=-1)
    robust_distance = np.abs(values - channel_median[None, :, None])
    extreme = np.mean(
        robust_distance > (6.0 * channel_mad[None, :, None] + EPSILON), axis=-1
    )
    return np.stack((log_rms, log_diff_rms, flatline, extreme), axis=-1).astype(np.float32)


@dataclass(frozen=True)
class QualityProfile:
    channel_median: tuple[float, ...]
    channel_mad: tuple[float, ...]
    diff_threshold: tuple[float, ...]
    descriptor_mean: tuple[tuple[float, ...], ...]
    descriptor_std: tuple[tuple[float, ...], ...]
    fit_partition: str
    n_train_windows: int
    sha256: str

    def transform(self, x: np.ndarray, *, include_amplitude: bool) -> np.ndarray:
        descriptors = _raw_descriptors(
            x,
            channel_median=np.asarray(self.channel_median, dtype=np.float32),
            channel_mad=np.asarray(self.channel_mad, dtype=np.float32),
            diff_threshold=np.asarray(self.diff_threshold, dtype=np.float32),
        )
        mean = np.asarray(self.descriptor_mean, dtype=np.float32)
        std = np.asarray(self.descriptor_std, dtype=np.float32)
        standardized = (descriptors - mean[None]) / std[None]
        return standardized if include_amplitude else standardized[..., 1:]

    def payload(self) -> dict[str, object]:
        return asdict(self)


def fit_quality_profile(x_train: np.ndarray) -> QualityProfile:
    values = np.asarray(x_train, dtype=np.float32)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("Quality profile expects non-empty (windows, channels, time) data")
    channel_median = np.median(values, axis=(0, 2)).astype(np.float32)
    channel_mad = np.median(
        np.abs(values - channel_median[None, :, None]), axis=(0, 2)
    ).astype(np.float32)
    differences = np.diff(values, axis=-1)
    diff_median = np.median(differences, axis=(0, 2))
    diff_mad = np.median(
        np.abs(differences - diff_median[None, :, None]), axis=(0, 2)
    ).astype(np.float32)
    diff_threshold = np.maximum(diff_mad * 0.01, EPSILON).astype(np.float32)
    descriptors = _raw_descriptors(
        values,
        channel_median=channel_median,
        channel_mad=channel_mad,
        diff_threshold=diff_threshold,
    )
    descriptor_mean = descriptors.mean(axis=0).astype(np.float32)
    descriptor_std = np.maximum(descriptors.std(axis=0), EPSILON).astype(np.float32)
    body = {
        "channel_median": tuple(float(value) for value in channel_median),
        "channel_mad": tuple(float(value) for value in channel_mad),
        "diff_threshold": tuple(float(value) for value in diff_threshold),
        "descriptor_mean": tuple(
            tuple(float(value) for value in row) for row in descriptor_mean
        ),
        "descriptor_std": tuple(tuple(float(value) for value in row) for row in descriptor_std),
        "fit_partition": "train_only",
        "n_train_windows": int(len(values)),
    }
    return QualityProfile(**body, sha256=canonical_sha256(body))


class ChannelDropout(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Channel dropout probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(
        self, x: torch.Tensor, *, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones((len(x), x.shape[1]), dtype=x.dtype, device=x.device)
        if not self.training or self.probability == 0.0:
            return x, mask
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        keep = torch.rand((len(x), x.shape[1]), generator=generator) >= self.probability
        empty = ~keep.any(dim=1)
        if empty.any():
            keep[empty, 0] = True
        mask = keep.to(device=x.device, dtype=x.dtype)
        return x * mask.unsqueeze(-1), mask


class ExactFBCNet(nn.Module):
    """Official FBCNet with a training interface shared by exp03 candidates."""

    def __init__(self, *, n_chans: int, n_times: int, sfreq: float) -> None:
        super().__init__()
        self.base = FBCNet(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
            sfreq=sfreq,
            n_bands=list(FBC_BANDS),
            n_filters_spat=32,
            temporal_layer="LogVarLayer",
            stride_factor=4,
        )

    def forward(self, x: torch.Tensor, *, dropout_seed: int | None = None) -> torch.Tensor:
        _ = dropout_seed
        return self.base(x)

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"total": int(total), "active": int(total), "auxiliary": 0}


class QualityConditionedFBCNet(nn.Module):
    """FBCNet whose optional channel gate starts as an exact identity."""

    def __init__(
        self,
        candidate: QualityCandidate,
        *,
        profile: QualityProfile,
        n_chans: int,
        n_times: int,
        sfreq: float,
        gate_hidden: int = 8,
    ) -> None:
        super().__init__()
        if len(profile.channel_median) != n_chans:
            raise ValueError("Quality profile channel count does not match the model")
        self.candidate = candidate
        self.channel_dropout = ChannelDropout(candidate.channel_dropout)
        self.base = FBCNet(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
            sfreq=sfreq,
            n_bands=list(FBC_BANDS),
            n_filters_spat=candidate.n_filters_spat,
            temporal_layer="LogVarLayer",
            stride_factor=candidate.stride_factor,
        )
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
        descriptor_count = len(candidate.descriptor_names)
        self.gate = (
            None
            if descriptor_count == 0
            else nn.Sequential(
                nn.Linear(descriptor_count, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
        )
        if self.gate is not None:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def _descriptors(self, x: torch.Tensor) -> torch.Tensor:
        differences = x.diff(dim=-1)
        log_rms = torch.log(torch.sqrt(torch.mean(x.square(), dim=-1)) + EPSILON)
        log_diff_rms = torch.log(
            torch.sqrt(torch.mean(differences.square(), dim=-1)) + EPSILON
        )
        flatline = (
            differences.abs() <= self.quality_diff_threshold[None, :, None]
        ).to(x.dtype).mean(dim=-1)
        robust_distance = (x - self.quality_channel_median[None, :, None]).abs()
        extreme = (
            robust_distance
            > (6.0 * self.quality_channel_mad[None, :, None] + EPSILON)
        ).to(x.dtype).mean(dim=-1)
        descriptors = torch.stack((log_rms, log_diff_rms, flatline, extreme), dim=-1)
        standardized = (
            descriptors - self.quality_descriptor_mean[None]
        ) / self.quality_descriptor_std[None]
        return standardized if self.candidate.gate_id == "G2" else standardized[..., 1:]

    def apply_quality_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gate is None:
            weights = torch.ones((len(x), x.shape[1]), dtype=x.dtype, device=x.device)
            return x, weights
        raw = 0.5 + 0.5 * torch.sigmoid(self.gate(self._descriptors(x)).squeeze(-1))
        weights = raw / raw.mean(dim=1, keepdim=True)
        return x * weights.unsqueeze(-1), weights

    def forward(self, x: torch.Tensor, *, dropout_seed: int | None = None) -> torch.Tensor:
        gated, _ = self.apply_quality_gate(x)
        dropped, _ = self.channel_dropout(gated, seed=dropout_seed)
        return self.base(dropped)

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        auxiliary = 0 if self.gate is None else sum(p.numel() for p in self.gate.parameters())
        return {"total": int(total), "active": int(total), "auxiliary": int(auxiliary)}

