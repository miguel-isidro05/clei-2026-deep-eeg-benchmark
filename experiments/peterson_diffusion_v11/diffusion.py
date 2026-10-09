"""Minimal forward-diffusion process used by the V11 energy classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _generator_device(generator: torch.Generator | None) -> torch.device | None:
    if generator is None:
        return None
    return torch.device(generator.device)


def sample_timesteps_and_noise(
    x: torch.Tensor,
    *,
    steps: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample reproducibly even when a backend cannot own a Torch generator."""
    generator_device = _generator_device(generator)
    sampling_device = x.device if generator_device is None else generator_device
    timestep = torch.randint(
        0, steps, (len(x),), device=sampling_device, generator=generator
    ).to(x.device)
    noise = torch.randn(
        x.shape, device=sampling_device, dtype=x.dtype, generator=generator
    ).to(x.device)
    return timestep, noise


@dataclass(frozen=True)
class DiffusionSchedule:
    steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self) -> None:
        if self.steps < 2:
            raise ValueError("Diffusion schedule needs at least two steps")
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("Invalid beta range")

    def alpha_bars(self, device: torch.device) -> torch.Tensor:
        betas = torch.linspace(self.beta_start, self.beta_end, self.steps, device=device)
        return torch.cumprod(1.0 - betas, dim=0)

    def q_sample(
        self, x: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars(x.device)[timestep].view(-1, 1, 1)
        return alpha_bar.sqrt() * x + (1.0 - alpha_bar).sqrt() * noise
