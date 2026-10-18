"""Tail-aware objectives for Exp04b."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def tail_risk(losses: torch.Tensor, objective: str) -> torch.Tensor:
    if losses.ndim != 1 or len(losses) < 1 or not torch.isfinite(losses).all():
        raise ValueError("Tail risk expects a finite non-empty loss vector")
    if objective == "mean_ce":
        return losses.mean()
    if objective == "cvar25":
        count = max(1, math.ceil(0.25 * len(losses)))
        return torch.topk(losses, count).values.mean()
    if objective == "group_dro":
        temperature = 0.25
        return temperature * torch.logsumexp(losses / temperature, dim=0)
    if objective == "softmax_max":
        weights = torch.softmax(5.0 * losses.detach(), dim=0)
        return torch.sum(weights * losses)
    raise ValueError(f"Unsupported tail objective: {objective}")


def safepath_loss(
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str,
    router_weight: torch.Tensor | None = None,
    corruption_score: torch.Tensor | None = None,
) -> torch.Tensor:
    clean = F.cross_entropy(clean_logits, labels, reduction="none")
    corrupt = F.cross_entropy(corrupt_logits, labels, reduction="none")
    classification = 0.25 * clean.mean() + tail_risk(corrupt, objective)
    if router_weight is None or corruption_score is None:
        return classification
    router = F.binary_cross_entropy(router_weight, corruption_score)
    return classification + 0.2 * router
