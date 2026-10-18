"""Predeclared paired clean/corrupt objective for QCR-SSNet."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .catalog import QCRCandidate
from .models import QCRForward


@dataclass(frozen=True)
class QCRLossTerms:
    total: torch.Tensor
    clean_classification: torch.Tensor
    corrupt_classification: torch.Tensor
    consistency: torch.Tensor
    router_supervision: torch.Tensor
    specialization: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "clean_classification": float(self.clean_classification.detach().cpu()),
            "corrupt_classification": float(self.corrupt_classification.detach().cpu()),
            "consistency": float(self.consistency.detach().cpu()),
            "router_supervision": float(self.router_supervision.detach().cpu()),
            "specialization": float(self.specialization.detach().cpu()),
        }


def _js_divergence(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    left = torch.softmax(left_logits, dim=1)
    right = torch.softmax(right_logits, dim=1)
    midpoint = 0.5 * (left + right)
    return 0.5 * (
        F.kl_div(torch.log(midpoint.clamp_min(1e-7)), left, reduction="batchmean")
        + F.kl_div(torch.log(midpoint.clamp_min(1e-7)), right, reduction="batchmean")
    )


def qcr_loss(
    clean: QCRForward,
    corrupt: QCRForward,
    labels: torch.Tensor,
    severity: torch.Tensor,
    candidate: QCRCandidate,
) -> QCRLossTerms:
    """Compute all terms even when an ablation assigns a zero coefficient."""

    tensors = (
        clean.logits,
        corrupt.logits,
        clean.clean_logits,
        corrupt.robust_logits,
        clean.router_weight,
        corrupt.router_weight,
        severity,
    )
    if any(not torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("QCR objective received non-finite values")
    clean_classification = F.cross_entropy(clean.logits, labels)
    corrupt_classification = F.cross_entropy(corrupt.logits, labels)
    consistency = _js_divergence(clean.logits, corrupt.logits)
    if candidate.router_mode == "learned":
        router_supervision = 0.5 * (
            F.binary_cross_entropy(clean.router_weight, torch.zeros_like(severity))
            + F.binary_cross_entropy(corrupt.router_weight, severity)
        )
    else:
        router_supervision = clean_classification.new_zeros(())
    specialization = 0.5 * (
        F.cross_entropy(clean.clean_logits, labels) + F.cross_entropy(corrupt.robust_logits, labels)
    )
    total = (
        clean_classification
        + candidate.lambda_corrupt * corrupt_classification
        + candidate.lambda_consistency * consistency
        + candidate.lambda_router * router_supervision
        + candidate.lambda_specialization * specialization
    )
    if not torch.isfinite(total):
        raise RuntimeError("QCR objective produced a non-finite loss")
    return QCRLossTerms(
        total=total,
        clean_classification=clean_classification,
        corrupt_classification=corrupt_classification,
        consistency=consistency,
        router_supervision=router_supervision,
        specialization=specialization,
    )
