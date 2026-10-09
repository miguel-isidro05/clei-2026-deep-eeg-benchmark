"""Compact full-resolution EEG backbones with generative and discriminative heads."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _sinusoidal_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    scale = math.log(10_000) / max(1, half - 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=timestep.device, dtype=torch.float32)
    )
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return F.pad(embedding, (0, dimension - embedding.shape[1]))


class SmoothBasisConv1d(nn.Module):
    """Temporal convolution whose kernels are learned combinations of smooth bases."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 33) -> None:
        super().__init__()
        positions = torch.linspace(-1.0, 1.0, kernel_size)
        basis = []
        for frequency in range(4):
            envelope = torch.exp(-positions.square() * (1.5 + frequency))
            basis.append(envelope * torch.cos(math.pi * frequency * positions))
            basis.append(envelope * torch.sin(math.pi * (frequency + 1) * positions))
        normalized = torch.stack(basis)
        normalized = normalized / normalized.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("basis", normalized)
        self.coefficients = nn.Parameter(torch.empty(out_channels, in_channels, len(basis)))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = kernel_size // 2
        nn.init.xavier_uniform_(self.coefficients)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.einsum("oib,bk->oik", self.coefficients, self.basis)
        return F.conv1d(x, weight, self.bias, padding=self.padding)


class MultiKernelStem(nn.Module):
    def __init__(self, n_chans: int, hidden: int) -> None:
        super().__init__()
        widths = (9, 17, 33)
        branch_channels = max(1, hidden // len(widths))
        self.branches = nn.ModuleList(
            nn.Conv1d(n_chans, branch_channels, width, padding=width // 2)
            for width in widths
        )
        self.project = nn.Conv1d(branch_channels * len(widths), hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(4 if hidden >= 4 else 1, hidden)
        self.depthwise = nn.Conv1d(
            hidden, hidden, 7, padding=3 * dilation, dilation=dilation, groups=hidden
        )
        self.mix = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.mix(F.silu(self.depthwise(F.silu(self.norm(x)))))
        return x + residual


class FullResolutionBackbone(nn.Module):
    def __init__(self, architecture: str, n_chans: int, hidden: int) -> None:
        super().__init__()
        if architecture == "res":
            self.stem: nn.Module = nn.Conv1d(n_chans, hidden, 9, padding=4)
        elif architecture == "filterbank":
            self.stem = MultiKernelStem(n_chans, hidden)
        elif architecture == "smooth_basis":
            self.stem = SmoothBasisConv1d(n_chans, hidden)
        elif architecture == "conformer":
            self.stem = nn.Conv1d(n_chans, hidden, 9, padding=4)
        else:
            raise ValueError(f"Unknown V11 architecture: {architecture}")
        self.architecture = architecture
        self.n_chans = n_chans
        self.hidden = hidden
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.class_embedding = nn.Embedding(2, hidden)
        self.mask_projection = nn.Linear(n_chans, hidden)
        if architecture == "conformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=False,
            )
            self.sequence: nn.Module = nn.TransformerEncoder(layer, num_layers=1)
        else:
            self.sequence = nn.Sequential(
                ResidualTemporalBlock(hidden, 1), ResidualTemporalBlock(hidden, 2)
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        timestep: torch.Tensor,
        label: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        observed = x * channel_mask.unsqueeze(-1)
        hidden = self.stem(observed)
        conditioning = (
            self.time_mlp(_sinusoidal_embedding(timestep, self.hidden))
            + self.class_embedding(label)
            + self.mask_projection(channel_mask.float())
        )
        hidden = hidden + conditioning.unsqueeze(-1)
        if self.architecture == "conformer":
            hidden = self.sequence(hidden.transpose(1, 2)).transpose(1, 2)
        else:
            hidden = self.sequence(hidden)
        return hidden


class DiffusionEEGModel(nn.Module):
    def __init__(self, backbone: FullResolutionBackbone, n_chans: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv1d(backbone.hidden, n_chans, 1)

    def forward(self, x, *, timestep, label, channel_mask):
        return self.head(
            self.backbone(x, timestep=timestep, label=label, channel_mask=channel_mask)
        )


class DiscriminativeEEGModel(nn.Module):
    def __init__(self, backbone: FullResolutionBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.hidden, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        mask = channel_mask if channel_mask is not None else x.new_ones((batch, x.shape[1]))
        times = (
            timestep
            if timestep is not None
            else torch.zeros(batch, dtype=torch.long, device=x.device)
        )
        labels = (
            label
            if label is not None
            else torch.zeros(batch, dtype=torch.long, device=x.device)
        )
        hidden = self.backbone(x, timestep=times, label=labels, channel_mask=mask)
        return self.head(hidden.mean(dim=-1))


def make_model(
    architecture: str, formulation: str, *, n_chans: int, hidden: int = 32
) -> nn.Module:
    backbone = FullResolutionBackbone(architecture, n_chans, hidden)
    if formulation == "diffusion":
        return DiffusionEEGModel(backbone, n_chans)
    if formulation == "discriminative":
        return DiscriminativeEEGModel(backbone)
    raise ValueError(f"Unknown formulation: {formulation}")


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def parameter_report(
    architecture: str, *, n_chans: int, hidden: int = 32
) -> dict[str, float | int]:
    diffusion = make_model(architecture, "diffusion", n_chans=n_chans, hidden=hidden)
    discriminative = make_model(
        architecture, "discriminative", n_chans=n_chans, hidden=hidden
    )
    diffusion_backbone = _count(diffusion.backbone)
    discriminative_backbone = _count(discriminative.backbone)
    return {
        "diffusion_total": _count(diffusion),
        "discriminative_total": _count(discriminative),
        "diffusion_backbone": diffusion_backbone,
        "discriminative_backbone": discriminative_backbone,
        "backbone_relative_difference": abs(diffusion_backbone - discriminative_backbone)
        / max(1, discriminative_backbone),
    }
