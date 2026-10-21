"""EEGNet-family backbones with controlled six-window aggregation."""

from __future__ import annotations

from typing import Any

import torch
from braindecode.models import EEGNet, EEGNeX, EEGTCNet
from torch import nn

from experiments.peterson_search_v12.exp06_window_fusion.models import WindowFusionModel

from .catalog import Exp07Candidate


def _backbone(candidate: Exp07Candidate, *, n_chans: int, n_times: int, sfreq: float) -> nn.Module:
    params = candidate.params
    common = {"n_chans": n_chans, "n_outputs": 2, "n_times": n_times, "sfreq": sfreq}
    if candidate.backbone == "eegnet":
        return EEGNet(
            **common,
            F1=int(params["F1"]),
            D=int(params["D"]),
            F2=int(params["F2"]),
            kernel_length=int(params["kernel_length"]),
            drop_prob=float(params["drop_prob"]),
        )
    if candidate.backbone == "eegtcnet":
        return EEGTCNet(
            **common,
            depth_multiplier=int(params["depth_multiplier"]),
            filter_1=int(params["filter_1"]),
            kern_length=int(params["kern_length"]),
            drop_prob=float(params["drop_prob"]),
            depth=int(params["depth"]),
            kernel_size=int(params["kernel_size"]),
            filters=int(params["filters"]),
        )
    if candidate.backbone == "eegnex":
        return EEGNeX(
            **common,
            depth_multiplier=int(params["depth_multiplier"]),
            filter_1=int(params["filter_1"]),
            filter_2=int(params["filter_2"]),
            drop_prob=float(params["drop_prob"]),
            kernel_block_1_2=int(params["kernel_block_1_2"]),
            kernel_block_4=int(params["kernel_block_4"]),
            dilation_block_4=int(params["dilation_block_4"]),
            avg_pool_block4=int(params["avg_pool_block4"]),
            kernel_block_5=int(params["kernel_block_5"]),
            dilation_block_5=int(params["dilation_block_5"]),
            avg_pool_block5=int(params["avg_pool_block5"]),
        )
    raise ValueError(f"Unsupported Exp07 backbone: {candidate.backbone}")


class EEGFamilyFusionModel(nn.Module):
    """One official backbone with either uniform or learned positional fusion."""

    def __init__(
        self,
        candidate: Exp07Candidate,
        *,
        n_chans: int,
        n_times: int,
        sfreq: float,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.base = _backbone(candidate, n_chans=n_chans, n_times=n_times, sfreq=sfreq)
        self.window_logits: nn.Parameter | None = None
        if candidate.aggregation == "lpwf":
            self.window_logits = nn.Parameter(torch.zeros(int(candidate.params["max_windows"])))
        elif candidate.aggregation != "uniform":
            raise ValueError(f"Unsupported Exp07 aggregation: {candidate.aggregation}")
        self.last_window_weights: torch.Tensor | None = None

    def _single_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.base(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(f"Exp07 backbone returned invalid logits: {tuple(logits.shape)}")
        return logits, logits

    def _grouped_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, windows, channels, n_times = x.shape
        if windows > int(self.candidate.params["max_windows"]):
            raise ValueError("Grouped input exceeds configured max_windows")
        logits, features = self._single_forward(x.reshape(batch * windows, channels, n_times))
        logits = logits.view(batch, windows, -1)
        features = features.view(batch, windows, -1)
        if self.window_logits is None:
            base_weights = torch.full(
                (windows,), 1.0 / windows, device=x.device, dtype=logits.dtype
            )
        else:
            base_weights = torch.softmax(self.window_logits[:windows], dim=0)
        weights = base_weights.unsqueeze(0).expand(batch, -1)
        self.last_window_weights = weights
        return (logits * weights.unsqueeze(-1)).sum(dim=1), (
            features * weights.unsqueeze(-1)
        ).sum(dim=1)

    def forward(
        self, x: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 4:
            logits, features = self._grouped_forward(x)
        elif x.ndim == 3:
            logits, features = self._single_forward(x)
        else:
            raise ValueError("Exp07 models expect (B,C,T) or grouped (B,W,C,T) input")
        return (logits, features) if return_features else logits

    def diagnostic_report(self) -> dict[str, Any]:
        windows = int(self.candidate.params["max_windows"])
        parameter = next(self.parameters())
        if self.window_logits is None:
            weights = torch.full(
                (windows,), 1.0 / windows, device=parameter.device, dtype=parameter.dtype
            )
        else:
            weights = torch.softmax(self.window_logits[:windows], dim=0)
        return {
            "backbone": self.candidate.backbone,
            "aggregation": self.candidate.aggregation,
            "window_weights": weights.detach().cpu().tolist(),
        }

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
    candidate: Exp07Candidate,
    *,
    n_chans: int,
    n_times: int,
    sfreq: float,
    fit_context: dict[str, object] | None = None,
) -> nn.Module:
    """Build an exact inherited control or one EEGNet-family candidate."""

    del fit_context
    if candidate.backbone == "fbcnet":
        return WindowFusionModel(
            candidate,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
        )
    return EEGFamilyFusionModel(
        candidate,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=sfreq,
    )

