"""Exp02 Inception and FBCNet-capable candidate models."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from braindecode.models import FBCNet
from torch import nn
from torch.nn import functional as F

from .catalog import Exp02Candidate

FBC_BANDS = ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30))


def _groups(channels: int) -> int:
    return 4 if channels % 4 == 0 else 1


def _time_embedding(timestep: torch.Tensor, hidden: int) -> torch.Tensor:
    half = hidden // 2
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return F.pad(torch.cat((angles.sin(), angles.cos()), dim=1), (0, hidden - 2 * half))


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(hidden), hidden)
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=7,
            dilation=dilation,
            padding=3 * dilation,
            groups=hidden,
        )
        self.pointwise = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.gelu(self.norm(x))
        return x + self.pointwise(F.gelu(self.depthwise(residual)))


class InceptionStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        kernels: tuple[int, ...],
        branch_type: str,
    ) -> None:
        super().__init__()
        branch_channels = max(2, hidden // len(kernels))
        branches: list[nn.Module] = []
        for kernel in kernels:
            if branch_type == "standard":
                branch: nn.Module = nn.Conv1d(
                    in_channels, branch_channels, kernel, padding=kernel // 2
                )
            elif branch_type == "depthwise":
                branch = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        in_channels,
                        kernel,
                        padding=kernel // 2,
                        groups=in_channels,
                    ),
                    nn.Conv1d(in_channels, branch_channels, 1),
                )
            else:
                raise ValueError(f"Unsupported Inception branch type: {branch_type}")
            branches.append(branch)
        self.branches = nn.ModuleList(branches)
        self.project = nn.Conv1d(branch_channels * len(kernels), hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class Exp02Model(nn.Module):
    """Common classifier/denoiser interface used by the factorial runner."""

    def classify(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        *,
        timestep: torch.Tensor,
        label: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def auxiliary_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError


class InceptionAuxModel(Exp02Model):
    def __init__(self, candidate: Exp02Candidate, n_chans: int) -> None:
        super().__init__()
        hidden = candidate.hidden
        self.hidden = hidden
        self.n_chans = n_chans
        self.pooling = candidate.pooling
        self.stem = InceptionStem(
            n_chans + hidden,
            hidden,
            candidate.kernels,
            candidate.branch_type,
        )
        self.sequence = nn.Sequential(ResidualBlock(hidden, 1), ResidualBlock(hidden, 2))
        classifier_features = hidden * 2 if candidate.pooling == "mean_std" else hidden
        self.attention = nn.Conv1d(hidden, 1, 1) if candidate.pooling == "attention" else None
        self.classifier = nn.Linear(classifier_features, 2)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.class_embedding = nn.Embedding(2, hidden)
        self.noise_head = nn.Conv1d(hidden, n_chans, 1)

    def _encode(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor,
        conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        observed = x * channel_mask.unsqueeze(-1)
        if conditioning is None:
            conditioning = x.new_zeros((len(x), self.hidden))
        expanded = conditioning.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return self.sequence(self.stem(torch.cat((observed, expanded), dim=1)))

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden.mean(dim=-1)
        if self.pooling == "mean_std":
            return torch.cat((hidden.mean(dim=-1), hidden.std(dim=-1, unbiased=False)), dim=1)
        if self.pooling == "attention" and self.attention is not None:
            weights = torch.softmax(self.attention(hidden), dim=-1)
            return (hidden * weights).sum(dim=-1)
        raise RuntimeError(f"Unsupported pooling contract: {self.pooling}")

    def classify(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._pool(self._encode(x, channel_mask, None)))

    def forward(
        self,
        x: torch.Tensor,
        *,
        timestep: torch.Tensor,
        label: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        conditioning = self.time_mlp(_time_embedding(timestep, self.hidden))
        conditioning = conditioning + self.class_embedding(label)
        return self.noise_head(self._encode(x, channel_mask, conditioning))

    def auxiliary_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.time_mlp.parameters()
        yield from self.class_embedding.parameters()
        yield from self.noise_head.parameters()


class FBCNetAuxModel(Exp02Model):
    """Official FBCNet classifier with an auxiliary pre-pooling denoising head."""

    def __init__(
        self,
        candidate: Exp02Candidate,
        *,
        n_chans: int,
        n_times: int,
        sfreq: float,
    ) -> None:
        super().__init__()
        if candidate.fbc_filters is None:
            raise ValueError("FBCNet topology requires fbc_filters")
        self.base = FBCNet(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
            sfreq=sfreq,
            n_bands=list(FBC_BANDS),
            n_filters_spat=candidate.fbc_filters,
            temporal_layer="LogVarLayer",
            stride_factor=4,
        )
        hidden = len(FBC_BANDS) * candidate.fbc_filters
        self.hidden = hidden
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.class_embedding = nn.Embedding(2, hidden)
        self.noise_head = nn.Conv1d(hidden, n_chans, 1)

    def classify(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        return self.base(x * channel_mask.unsqueeze(-1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        timestep: torch.Tensor,
        label: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        observed = x * channel_mask.unsqueeze(-1)
        filtered = self.base.spectral_filtering(observed)
        temporal = self.base.spatial_conv(filtered).squeeze(2)
        conditioning = self.time_mlp(_time_embedding(timestep, self.hidden))
        conditioning = conditioning + self.class_embedding(label)
        return self.noise_head(temporal + conditioning.unsqueeze(-1))

    def auxiliary_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.time_mlp.parameters()
        yield from self.class_embedding.parameters()
        yield from self.noise_head.parameters()


def make_model(
    candidate: Exp02Candidate,
    *,
    n_chans: int,
    n_times: int,
    sfreq: float,
) -> Exp02Model:
    if candidate.family == "inception":
        return InceptionAuxModel(candidate, n_chans)
    if candidate.family == "fbcnet":
        return FBCNetAuxModel(
            candidate,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
        )
    raise ValueError(f"Unknown exp02 family: {candidate.family}")


def parameter_report(model: Exp02Model, candidate: Exp02Candidate) -> dict[str, int]:
    auxiliary_ids = {id(parameter) for parameter in model.auxiliary_parameters()}
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    auxiliary = sum(
        parameter.numel() for parameter in model.parameters() if id(parameter) in auxiliary_ids
    )
    active = trainable if candidate.objective_id != "O0" else trainable - auxiliary
    return {
        "total": int(total),
        "trainable": int(trainable),
        "active": int(active),
        "auxiliary": int(auxiliary),
    }
