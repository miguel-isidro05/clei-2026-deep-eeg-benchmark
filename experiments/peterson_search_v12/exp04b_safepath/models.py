"""SafePath-QCR with an exact clean bypass."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SafePathOutput:
    logits: torch.Tensor
    clean_logits: torch.Tensor
    robust_logits: torch.Tensor
    router_weight: torch.Tensor


class SafePathQCR(nn.Module):
    """Compose clean and robust experts while preserving the clean path exactly."""

    def __init__(
        self,
        clean_expert: nn.Module,
        robust_expert: nn.Module,
        *,
        router_mode: str,
        fusion_mode: str,
        freeze_clean: bool,
        hard_clean_bypass: bool = True,
        router_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.clean_expert = clean_expert
        self.robust_expert = robust_expert
        self.router_mode = router_mode
        self.fusion_mode = fusion_mode
        self.hard_clean_bypass = hard_clean_bypass
        self._clean_accepts_dropout_seed = (
            "dropout_seed" in inspect.signature(clean_expert.forward).parameters
        )
        self._robust_accepts_dropout_seed = (
            "dropout_seed" in inspect.signature(robust_expert.forward).parameters
        )
        if freeze_clean:
            for parameter in self.clean_expert.parameters():
                parameter.requires_grad_(False)
        self.router = (
            nn.Sequential(nn.Linear(8, router_hidden), nn.GELU(), nn.Linear(router_hidden, 1))
            if router_mode in {"learned_detached", "learned_joint", "hybrid"}
            else None
        )
        if self.router is not None:
            nn.init.zeros_(self.router[-1].weight)
            nn.init.constant_(self.router[-1].bias, -4.0)
        self.logit_adapter = nn.Linear(2, 2, bias=False) if fusion_mode == "affine_logit" else None
        if self.logit_adapter is not None:
            nn.init.eye_(self.logit_adapter.weight)
        self.film_scale = nn.Parameter(torch.ones(2)) if fusion_mode == "quality_film" else None
        self.film_bias = nn.Parameter(torch.zeros(2)) if fusion_mode == "quality_film" else None

    @staticmethod
    def _quality_features(x: torch.Tensor) -> torch.Tensor:
        diff = x.diff(dim=-1)
        rms = x.square().mean(dim=-1).sqrt().clamp_min(1e-6)
        diff_rms = diff.square().mean(dim=-1).sqrt().clamp_min(1e-6)
        flat = (diff.abs() < 1e-6).to(x.dtype).mean(dim=-1)
        extreme = (x.abs() > 6.0 * rms.unsqueeze(-1)).to(x.dtype).mean(dim=-1)
        descriptors = torch.stack((rms.log(), diff_rms.log(), flat, extreme), dim=-1)
        return torch.cat((descriptors.mean(dim=1), descriptors.amax(dim=1)), dim=1)

    def _route(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None,
        corruption_score: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = len(x)
        zero = torch.zeros(batch, dtype=x.dtype, device=x.device)
        one = torch.ones_like(zero)
        if self.router_mode == "none":
            return zero
        if self.router_mode == "always":
            return one
        explicit = None
        if channel_mask is not None:
            explicit = 1.0 - channel_mask.to(x).mean(dim=1)
        if self.router_mode == "oracle":
            if corruption_score is not None:
                return corruption_score.to(x).clamp(0.0, 1.0)
            if explicit is None:
                raise ValueError("Oracle routing requires corruption metadata")
            return explicit.clamp(0.0, 1.0)
        if self.router_mode == "hard_zero":
            detected = (x.abs().amax(dim=-1) <= 1e-8).any(dim=1)
            weight = detected.to(x.dtype)
        elif self.router_mode == "hard_quality":
            quality = self._quality_features(x)
            detected = (quality[:, 6] > 0.5) | (quality[:, 7] > 0.01)
            weight = detected.to(x.dtype)
        elif self.router_mode in {"learned_detached", "learned_joint", "hybrid"}:
            if self.router is None:
                raise RuntimeError("Learned router was not constructed")
            features = self._quality_features(x)
            if self.router_mode in {"learned_detached", "hybrid"}:
                features = features.detach()
            learned = torch.sigmoid(self.router(features).squeeze(-1))
            if self.router_mode == "hybrid":
                detected = (features[:, 6] > 0.5) | (features[:, 7] > 0.01)
                weight = torch.maximum(learned, detected.to(x.dtype))
            else:
                weight = learned
        else:
            raise ValueError(f"Unsupported router mode: {self.router_mode}")
        if explicit is not None:
            weight = torch.maximum(weight, explicit.clamp(0.0, 1.0))
            if self.hard_clean_bypass:
                weight = torch.where(explicit == 0, zero, weight)
        return weight

    def _fuse(
        self, clean: torch.Tensor, robust: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        expanded = weight[:, None]
        if self.fusion_mode == "logit_residual":
            return clean + expanded * (robust - clean)
        if self.fusion_mode == "probability_mixture":
            mixed = (1.0 - expanded) * clean.softmax(dim=1) + expanded * robust.softmax(dim=1)
            return mixed.clamp_min(1e-7).log()
        if self.fusion_mode == "affine_logit":
            if self.logit_adapter is None:
                raise RuntimeError("Logit adapter was not constructed")
            return clean + expanded * self.logit_adapter(robust - clean)
        if self.fusion_mode == "quality_film":
            if self.film_scale is None or self.film_bias is None:
                raise RuntimeError("FiLM parameters were not constructed")
            adjusted = robust * self.film_scale + self.film_bias
            return clean + expanded * (adjusted - clean)
        raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor | None = None,
        corruption_score: torch.Tensor | None = None,
        dropout_seed: int | None = None,
    ) -> SafePathOutput:
        clean_logits = (
            self.clean_expert(x, dropout_seed=dropout_seed)
            if self._clean_accepts_dropout_seed
            else self.clean_expert(x)
        )
        robust_logits = (
            self.robust_expert(x, dropout_seed=dropout_seed)
            if self._robust_accepts_dropout_seed
            else self.robust_expert(x)
        )
        weight = self._route(x, channel_mask, corruption_score)
        logits = self._fuse(clean_logits, robust_logits, weight)
        if self.hard_clean_bypass and channel_mask is not None:
            clean_rows = channel_mask.to(x).amin(dim=1) == 1
            logits = torch.where(clean_rows[:, None], clean_logits, logits)
            weight = torch.where(clean_rows, torch.zeros_like(weight), weight)
        return SafePathOutput(logits, clean_logits, robust_logits, weight)

    def parameter_report(self) -> dict[str, int]:
        active = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel() for parameter in self.parameters() if not parameter.requires_grad
        )
        auxiliary_modules = [module for module in (self.router, self.logit_adapter) if module]
        auxiliary = sum(
            parameter.numel() for module in auxiliary_modules for parameter in module.parameters()
        )
        if self.film_scale is not None and self.film_bias is not None:
            auxiliary += self.film_scale.numel() + self.film_bias.numel()
        return {
            "total": active + frozen,
            "active": active,
            "frozen": frozen,
            "auxiliary": auxiliary,
        }
