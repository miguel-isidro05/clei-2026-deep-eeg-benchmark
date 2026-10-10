"""Compact low-cost EEG backbones and paired V12 heads."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.peterson_diffusion_v11.models.networks import SmoothBasisConv1d

from .catalogs import CandidateSpec


def _group_count(channels: int) -> int:
    return 4 if channels % 4 == 0 else 1


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(hidden), hidden)
        self.depthwise = nn.Conv1d(
            hidden, hidden, 7, padding=3 * dilation, dilation=dilation, groups=hidden
        )
        self.pointwise = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(F.silu(self.depthwise(F.silu(self.norm(x)))))


class InceptionStem(nn.Module):
    def __init__(self, n_chans: int, hidden: int) -> None:
        super().__init__()
        branch = max(2, hidden // 4)
        self.branches = nn.ModuleList(
            nn.Conv1d(n_chans, branch, kernel, padding=kernel // 2) for kernel in (9, 17, 33, 65)
        )
        self.project = nn.Conv1d(branch * 4, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class FilterBankStem(nn.Module):
    def __init__(self, n_chans: int, hidden: int) -> None:
        super().__init__()
        branch = max(2, hidden // 3)
        self.temporal = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(n_chans, n_chans, kernel, padding=kernel // 2, groups=n_chans),
                nn.Conv1d(n_chans, branch, 1),
            )
            for kernel in (17, 33, 65)
        )
        self.project = nn.Conv1d(branch * 3, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.temporal], dim=1))


class EEGBackbone(nn.Module):
    def __init__(self, name: str, n_chans: int, hidden: int) -> None:
        super().__init__()
        if name == "inception":
            self.stem: nn.Module = InceptionStem(n_chans, hidden)
        elif name == "filterbank":
            self.stem = FilterBankStem(n_chans, hidden)
        elif name == "smooth_basis":
            self.stem = SmoothBasisConv1d(n_chans, hidden, kernel_size=33)
        else:
            self.stem = nn.Conv1d(n_chans, hidden, 9, padding=4)
        if name == "tcn":
            self.sequence: nn.Module = nn.Sequential(
                ResidualBlock(hidden, dilation=1),
                ResidualBlock(hidden, dilation=2),
                ResidualBlock(hidden, dilation=4),
            )
        elif name == "conformer_lite":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=0.0,
                batch_first=True,
            )
            self.sequence = nn.TransformerEncoder(layer, num_layers=1)
        else:
            self.sequence = nn.Sequential(ResidualBlock(hidden), ResidualBlock(hidden, dilation=2))
        self.name = name
        self.hidden = hidden

    def encode(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x * channel_mask.unsqueeze(-1))
        if self.name == "conformer_lite":
            return self.sequence(hidden.transpose(1, 2)).transpose(1, 2)
        return self.sequence(hidden)


def _time_embedding(timestep: torch.Tensor, hidden: int) -> torch.Tensor:
    half = hidden // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timestep.device) / max(1, half - 1)
    )
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return F.pad(torch.cat((angles.sin(), angles.cos()), dim=1), (0, hidden - 2 * half))


class CandidateModel(nn.Module):
    def __init__(self, backbone: EEGBackbone, formulation: str, n_chans: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.formulation = formulation
        self.classifier = nn.Linear(backbone.hidden, 2)
        self.time_mlp = nn.Sequential(
            nn.Linear(backbone.hidden, backbone.hidden),
            nn.SiLU(),
            nn.Linear(backbone.hidden, backbone.hidden),
        )
        self.class_embedding = nn.Embedding(2, backbone.hidden)
        self.noise_head = nn.Conv1d(backbone.hidden, n_chans, 1)

    def classify(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone.encode(x, channel_mask).mean(dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        timestep: torch.Tensor,
        label: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone.encode(x, channel_mask)
        conditioning = self.time_mlp(_time_embedding(timestep, self.backbone.hidden))
        conditioning = conditioning + self.class_embedding(label)
        return self.noise_head(hidden + conditioning.unsqueeze(-1))


def make_candidate_model(candidate: CandidateSpec, *, n_chans: int) -> CandidateModel:
    if candidate.backbone not in {
        "residual",
        "tcn",
        "inception",
        "filterbank",
        "smooth_basis",
        "conformer_lite",
    }:
        raise ValueError(f"Unknown backbone: {candidate.backbone}")
    if candidate.formulation not in {"discriminative", "diffusion_energy", "hybrid"}:
        raise ValueError(f"Unknown formulation: {candidate.formulation}")
    if candidate.width not in {"compact", "wide"}:
        raise ValueError(f"Unknown width: {candidate.width}")
    return CandidateModel(
        EEGBackbone(candidate.backbone, n_chans=n_chans, hidden=candidate.hidden),
        candidate.formulation,
        n_chans,
    )


def parameter_report(candidate: CandidateSpec, *, n_chans: int) -> dict[str, int]:
    model = make_candidate_model(candidate, n_chans=n_chans)
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "backbone": sum(parameter.numel() for parameter in model.backbone.parameters()),
    }
