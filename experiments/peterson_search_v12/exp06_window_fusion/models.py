"""Low-capacity Exp06 temporal-fusion models."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.models import (
    FBCMechanismModel,
)

from .catalog import WindowFusionCandidate


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Euclidean projection onto the probability simplex."""

    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits = torch.sort(shifted, dim=dim, descending=True).values
    cumulative = sorted_logits.cumsum(dim)
    size = logits.shape[dim]
    ranks_shape = [1] * logits.ndim
    ranks_shape[dim] = size
    ranks = torch.arange(1, size + 1, device=logits.device, dtype=logits.dtype).view(ranks_shape)
    support = 1 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumulative.gather(dim, support_size - 1) - 1) / support_size.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class WindowFusionModel(FBCMechanismModel):
    """FBCNet with a compact, input-independent temporal fusion rule."""

    def __init__(
        self,
        candidate: WindowFusionCandidate,
        *,
        n_chans: int,
        n_times: int,
        sfreq: float,
        fit_context: dict[str, object] | None = None,
    ) -> None:
        base_mechanism = (
            "csp_init_finetune"
            if candidate.fit_hook == "csp"
            else "global_window_weights"
            if candidate.mechanism == "global_window_weights"
            else "baseline"
        )
        base_candidate = replace(candidate, mechanism=base_mechanism)
        super().__init__(
            base_candidate,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            fit_context=fit_context,
        )
        self.candidate = candidate
        windows = int(candidate.params["max_windows"])
        if self.window_logits is None and candidate.mechanism != "fbc_control":
            self.window_logits = nn.Parameter(torch.zeros(windows))

        self.log_temperature: nn.Parameter | None = None
        self.gaussian_center: nn.Parameter | None = None
        self.gaussian_raw_scale: nn.Parameter | None = None
        self.uniform_blend_logit: nn.Parameter | None = None
        self.class_window_logits: nn.Parameter | None = None
        self.temporal_logit_conv: nn.Conv1d | None = None

        if candidate.mechanism == "temperature_window_weights":
            initial = float(candidate.params["initial_temperature"])
            self.log_temperature = nn.Parameter(torch.tensor(_inverse_softplus(initial)))
        elif candidate.mechanism == "gaussian_window_kernel":
            self.gaussian_center = nn.Parameter(
                torch.tensor(float(candidate.params["initial_center"]))
            )
            scale = float(candidate.params["initial_scale"])
            self.gaussian_raw_scale = nn.Parameter(torch.tensor(_inverse_softplus(scale)))
            self.register_parameter("window_logits", None)
        elif candidate.mechanism == "residual_uniform_window_weights":
            blend = float(candidate.params["initial_blend"])
            self.uniform_blend_logit = nn.Parameter(torch.tensor(math.log(blend / (1 - blend))))
        elif candidate.mechanism == "class_conditional_window_weights":
            self.class_window_logits = nn.Parameter(torch.zeros(windows, 2))
            self.register_parameter("window_logits", None)
        elif candidate.mechanism == "residual_temporal_logit_conv":
            kernel_size = int(candidate.params["kernel_size"])
            self.temporal_logit_conv = nn.Conv1d(
                2,
                2,
                kernel_size,
                padding=kernel_size // 2,
                groups=2,
                bias=False,
            )
            nn.init.zeros_(self.temporal_logit_conv.weight)

    def _base_weights(
        self, windows: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        mechanism = self.candidate.mechanism
        if mechanism == "gaussian_window_kernel":
            if self.gaussian_center is None or self.gaussian_raw_scale is None:
                raise RuntimeError("Gaussian window parameters are unavailable")
            positions = torch.arange(windows, device=device, dtype=dtype)
            scale = F.softplus(self.gaussian_raw_scale).clamp_min(0.25)
            scores = -0.5 * ((positions - self.gaussian_center) / scale).square()
            return torch.softmax(scores, dim=0)
        if mechanism == "class_conditional_window_weights":
            if self.class_window_logits is None:
                raise RuntimeError("Class-conditional window parameters are unavailable")
            return torch.softmax(self.class_window_logits[:windows], dim=0)
        if self.window_logits is None:
            return torch.full((windows,), 1.0 / windows, device=device, dtype=dtype)
        logits = self.window_logits[:windows]
        if mechanism == "temperature_window_weights":
            if self.log_temperature is None:
                raise RuntimeError("Window temperature is unavailable")
            temperature = F.softplus(self.log_temperature).clamp_min(0.05)
            return torch.softmax(logits / temperature, dim=0)
        if mechanism == "sparsemax_window_weights":
            return sparsemax(logits, dim=0)
        learned = torch.softmax(logits, dim=0)
        if mechanism == "residual_uniform_window_weights":
            if self.uniform_blend_logit is None:
                raise RuntimeError("Uniform blend parameter is unavailable")
            blend = torch.sigmoid(self.uniform_blend_logit)
            uniform = torch.full_like(learned, 1.0 / windows)
            return blend * learned + (1 - blend) * uniform
        return learned

    def _drop_windows(self, weights: torch.Tensor) -> torch.Tensor:
        probability = float(self.candidate.params["window_dropout_probability"])
        keep = torch.rand_like(weights) >= probability
        empty = ~keep.any(dim=1)
        if empty.any():
            keep[empty, weights[empty].argmax(dim=1)] = True
        dropped = weights * keep.to(weights.dtype)
        return dropped / dropped.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _grouped_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, windows, channels, n_times = x.shape
        logits, features = self._single_forward(x.reshape(batch * windows, channels, n_times))
        logits = logits.view(batch, windows, -1)
        features = features.view(batch, windows, -1)
        if self.temporal_logit_conv is not None:
            residual = self.temporal_logit_conv(logits.transpose(1, 2)).transpose(1, 2)
            logits = logits + residual

        base_weights = self._base_weights(windows, device=x.device, dtype=x.dtype)
        if base_weights.ndim == 2:
            weights = base_weights.unsqueeze(0).expand(batch, -1, -1)
            self.last_window_weights = weights
            trial_logits = (logits * weights).sum(dim=1)
            feature_weights = weights.mean(dim=2)
            trial_features = (features * feature_weights.unsqueeze(-1)).sum(dim=1)
            return trial_logits, trial_features

        weights = base_weights.unsqueeze(0).expand(batch, -1)
        if self.candidate.mechanism == "window_dropout_weights" and self.training:
            weights = self._drop_windows(weights)
        self.last_window_weights = weights
        return (logits * weights.unsqueeze(-1)).sum(dim=1), (features * weights.unsqueeze(-1)).sum(
            dim=1
        )

    def diagnostic_report(self) -> dict[str, Any]:
        """Return learned fusion parameters without depending on the final batch."""

        with torch.no_grad():
            parameter = next(self.parameters())
            weights = self._base_weights(
                int(self.candidate.params["max_windows"]),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            report: dict[str, Any] = {
                "window_weights": weights.detach().cpu().tolist(),
                "mechanism": self.candidate.mechanism,
            }
            if self.log_temperature is not None:
                report["temperature"] = float(F.softplus(self.log_temperature).detach().cpu())
            if self.gaussian_center is not None and self.gaussian_raw_scale is not None:
                report["gaussian_center"] = float(self.gaussian_center.detach().cpu())
                report["gaussian_scale"] = float(F.softplus(self.gaussian_raw_scale).detach().cpu())
            if self.uniform_blend_logit is not None:
                report["learned_blend"] = float(
                    torch.sigmoid(self.uniform_blend_logit).detach().cpu()
                )
            if self.temporal_logit_conv is not None:
                report["temporal_kernel"] = self.temporal_logit_conv.weight.detach().cpu().tolist()
            return report


def build_model(
    candidate: WindowFusionCandidate,
    *,
    n_chans: int,
    n_times: int,
    sfreq: float,
    fit_context: dict[str, object] | None = None,
) -> WindowFusionModel:
    return WindowFusionModel(
        candidate,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=sfreq,
        fit_context=fit_context,
    )
