"""Monte Carlo denoising energies with common random numbers across classes."""

from __future__ import annotations

import torch

from .diffusion import DiffusionSchedule, sample_timesteps_and_noise


def masked_mse(
    predicted: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor
) -> torch.Tensor:
    weights = channel_mask.unsqueeze(-1).expand_as(predicted)
    squared = (predicted - target).square() * weights
    denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
    return squared.sum(dim=(1, 2)) / denominator


def class_energies(
    model: torch.nn.Module,
    x: torch.Tensor,
    channel_mask: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if k < 1:
        raise ValueError("k must be positive")
    observed = x * channel_mask.unsqueeze(-1)
    totals = x.new_zeros((len(x), 2))
    for _ in range(k):
        timestep, noise = sample_timesteps_and_noise(
            x, steps=schedule.steps, generator=generator
        )
        noisy = schedule.q_sample(observed, timestep, noise)
        for candidate in (0, 1):
            labels = torch.full_like(timestep, candidate)
            predicted = model(
                noisy, timestep=timestep, label=labels, channel_mask=channel_mask
            )
            totals[:, candidate] += masked_mse(predicted, noise, channel_mask)
    return totals / float(k)


def energy_probabilities(energies: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(-energies / temperature, dim=1)
