"""Paired discriminative, diffusion-energy and hybrid objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from experiments.peterson_diffusion_v11.diffusion import DiffusionSchedule, sample_timesteps_and_noise
from experiments.peterson_diffusion_v11.energy import class_energies, energy_probabilities, masked_mse

from .catalogs import CandidateSpec
from .models import CandidateModel


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    classification: torch.Tensor
    denoising: torch.Tensor
    ranking: torch.Tensor
    consistency: torch.Tensor


def _generator(device: torch.device, seed: int) -> torch.Generator:
    target = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    return torch.Generator(device=target).manual_seed(seed)


def candidate_loss(
    model: CandidateModel,
    x: torch.Tensor,
    y: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    candidate: CandidateSpec,
    seed: int,
    diffusion_steps: int = 100,
    rank_margin: float = 0.1,
    lambda_denoise: float = 1.0,
    lambda_rank: float = 0.2,
    lambda_consistency: float = 0.1,
) -> LossTerms:
    logits = model.classify(x, channel_mask)
    classification = F.cross_entropy(logits, y)
    zero = classification.new_zeros(())
    if candidate.formulation == "discriminative":
        return LossTerms(classification, classification, zero, zero, zero)

    schedule = DiffusionSchedule(diffusion_steps)
    generator = _generator(x.device, seed)
    timestep, noise = sample_timesteps_and_noise(x, steps=diffusion_steps, generator=generator)
    noisy = schedule.q_sample(x * channel_mask.unsqueeze(-1), timestep, noise)
    predicted_true = model(noisy, timestep=timestep, label=y, channel_mask=channel_mask)
    wrong = 1 - y
    predicted_wrong = model(noisy, timestep=timestep, label=wrong, channel_mask=channel_mask)
    true_energy = masked_mse(predicted_true, noise, channel_mask)
    wrong_energy = masked_mse(predicted_wrong, noise, channel_mask)
    denoising = true_energy.mean()
    ranking = F.relu(rank_margin + true_energy - wrong_energy).mean()
    energy_probs = torch.softmax(torch.stack((-true_energy, -wrong_energy), dim=1), dim=1)
    true_first = torch.where(y[:, None] == 0, energy_probs, energy_probs.flip(1))
    consistency = F.mse_loss(torch.softmax(logits, dim=1), true_first.detach())
    if candidate.formulation == "diffusion_energy":
        total = lambda_denoise * denoising + lambda_rank * ranking
    else:
        total = (
            classification
            + lambda_denoise * denoising
            + lambda_rank * ranking
            + lambda_consistency * consistency
        )
    return LossTerms(total, classification, denoising, ranking, consistency)


def predict_probabilities(
    model: CandidateModel,
    x: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    candidate: CandidateSpec,
    seed: int,
    inference_k: int,
    diffusion_steps: int = 100,
) -> torch.Tensor:
    if candidate.formulation == "discriminative":
        return torch.softmax(model.classify(x, channel_mask), dim=1)
    energies = class_energies(
        model,
        x,
        channel_mask,
        DiffusionSchedule(diffusion_steps),
        k=inference_k,
        generator=_generator(x.device, seed),
    )
    energy_probs = energy_probabilities(energies)
    if candidate.formulation == "diffusion_energy":
        return energy_probs
    classifier_probs = torch.softmax(model.classify(x, channel_mask), dim=1)
    return (classifier_probs + energy_probs) / 2.0
