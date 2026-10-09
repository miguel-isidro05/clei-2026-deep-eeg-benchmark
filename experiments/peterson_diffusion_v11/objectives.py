"""Predeclared diffusion-loss ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .diffusion import DiffusionSchedule, sample_timesteps_and_noise
from .energy import masked_mse


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    noise: torch.Tensor
    rank: torch.Tensor
    consistency: torch.Tensor


def _two_class_energies(
    model,
    noisy: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = []
    for candidate in (0, 1):
        labels = torch.full_like(timestep, candidate)
        prediction = model(noisy, timestep=timestep, label=labels, channel_mask=mask)
        values.append(masked_mse(prediction, noise, mask))
    return torch.stack(values, dim=1)


def diffusion_objective(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    channel_mask: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    objective: str,
    generator: torch.Generator | None = None,
    margin: float = 0.25,
    lambda_rank: float = 1.0,
    lambda_consistency: float = 0.1,
) -> LossTerms:
    if objective not in {"noise_only", "noise_rank", "noise_rank_consistency"}:
        raise ValueError(f"Unknown objective: {objective}")
    observed = x * channel_mask.unsqueeze(-1)
    timestep, noise = sample_timesteps_and_noise(
        x, steps=schedule.steps, generator=generator
    )
    noisy = schedule.q_sample(observed, timestep, noise)
    energies = _two_class_energies(model, noisy, noise, timestep, channel_mask)
    noise_loss = energies.gather(1, y[:, None]).mean()
    rank_loss = x.new_zeros(())
    consistency_loss = x.new_zeros(())
    if objective in {"noise_rank", "noise_rank_consistency"}:
        true_energy = energies.gather(1, y[:, None]).squeeze(1)
        wrong_energy = energies.gather(1, (1 - y)[:, None]).squeeze(1)
        rank_loss = F.relu(margin + true_energy - wrong_energy).mean()
    if objective == "noise_rank_consistency":
        corrupt_mask = channel_mask.clone()
        indices = torch.arange(len(x), device=x.device) % x.shape[1]
        corrupt_mask[torch.arange(len(x), device=x.device), indices] = 0.0
        corrupt_noisy = schedule.q_sample(observed * corrupt_mask.unsqueeze(-1), timestep, noise)
        corrupt_energies = _two_class_energies(
            model, corrupt_noisy, noise, timestep, corrupt_mask
        )
        clean_log = F.log_softmax(-energies, dim=1)
        corrupt_log = F.log_softmax(-corrupt_energies, dim=1)
        midpoint = 0.5 * (clean_log.exp() + corrupt_log.exp())
        consistency_loss = 0.5 * (
            F.kl_div(clean_log, midpoint, reduction="batchmean")
            + F.kl_div(corrupt_log, midpoint, reduction="batchmean")
        )
    total = noise_loss + lambda_rank * rank_loss + lambda_consistency * consistency_loss
    return LossTerms(total=total, noise=noise_loss, rank=rank_loss, consistency=consistency_loss)
