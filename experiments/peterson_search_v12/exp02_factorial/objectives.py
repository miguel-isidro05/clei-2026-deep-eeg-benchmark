"""Loss and predeclared nested Monte Carlo inference for exp02."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from experiments.peterson_diffusion_v11.diffusion import (
    DiffusionSchedule,
    sample_timesteps_and_noise,
)
from experiments.peterson_diffusion_v11.energy import energy_probabilities, masked_mse

from .catalog import Exp02Candidate
from .models import Exp02Model


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
    model: Exp02Model,
    x: torch.Tensor,
    y: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    candidate: Exp02Candidate,
    seed: int,
    diffusion_steps: int = 1000,
    rank_margin: float = 0.1,
) -> LossTerms:
    logits = model.classify(x, channel_mask)
    classification = F.cross_entropy(logits, y)
    zero = classification.new_zeros(())
    if candidate.objective_id == "O0":
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
    ordered_energy = torch.stack((true_energy, wrong_energy), dim=1)
    energy_probs = torch.softmax(-ordered_energy, dim=1)
    class_ordered = torch.where(y[:, None] == 0, energy_probs, energy_probs.flip(1))
    consistency = F.mse_loss(torch.softmax(logits, dim=1), class_ordered.detach())
    total = (
        classification
        + candidate.lambda_denoise * denoising
        + candidate.lambda_rank * ranking
        + candidate.lambda_consistency * consistency
    )
    return LossTerms(total, classification, denoising, ranking, consistency)


def _inference_id(candidate: Exp02Candidate, alpha: float, k: int) -> str:
    return f"{candidate.config_id}__alpha-{alpha:g}__k{k}"


def inference_probabilities(
    model: Exp02Model,
    x: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    candidate: Exp02Candidate,
    seed: int,
    diffusion_steps: int = 1000,
) -> dict[str, torch.Tensor]:
    classifier = torch.softmax(model.classify(x, channel_mask), dim=1)
    classifier_id = _inference_id(candidate, 1.0, 0)
    if candidate.objective_id == "O0":
        return {classifier_id: classifier}

    generator = _generator(x.device, seed)
    schedule = DiffusionSchedule(diffusion_steps)
    observed = x * channel_mask.unsqueeze(-1)
    totals = x.new_zeros((len(x), 2))
    energy_by_k: dict[int, torch.Tensor] = {}
    for draw in range(1, 9):
        timestep, noise = sample_timesteps_and_noise(x, steps=diffusion_steps, generator=generator)
        noisy = schedule.q_sample(observed, timestep, noise)
        for label in (0, 1):
            labels = torch.full_like(timestep, label)
            predicted = model(
                noisy,
                timestep=timestep,
                label=labels,
                channel_mask=channel_mask,
            )
            totals[:, label] += masked_mse(predicted, noise, channel_mask)
        if draw in {1, 4, 8}:
            energy_by_k[draw] = energy_probabilities(totals / float(draw))

    outputs: dict[str, torch.Tensor] = {classifier_id: classifier}
    for alpha in (0.0, 0.25, 0.5, 0.75):
        for k in (1, 4, 8):
            outputs[_inference_id(candidate, alpha, k)] = (
                alpha * classifier + (1.0 - alpha) * energy_by_k[k]
            )
    return outputs
