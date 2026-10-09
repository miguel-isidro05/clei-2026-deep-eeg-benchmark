"""Leakage-safe training and prediction for the V11 architecture screen."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepbench.reproducibility import set_seeds

from .config import ExperimentConfig
from .diffusion import DiffusionSchedule
from .energy import class_energies, energy_probabilities
from .models import make_model
from .objectives import diffusion_objective


@dataclass
class TrainingOutcome:
    model: torch.nn.Module
    history: list[dict[str, float | int]]
    best_epoch: int
    updates: int


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def _as_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def _validation_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    formulation: str,
    schedule: DiffusionSchedule,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    losses: list[float] = []
    generator = _device_generator(device, seed)
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask = torch.ones((len(x_batch), x_batch.shape[1]), device=device)
            if formulation == "discriminative":
                loss = F.cross_entropy(model(x_batch, channel_mask=mask), y_batch)
            else:
                energies = class_energies(
                    model, x_batch, mask, schedule, k=1, generator=generator
                )
                log_probabilities = torch.log(
                    energy_probabilities(energies).clamp_min(1e-8)
                )
                loss = F.nll_loss(log_probabilities, y_batch)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_model(
    architecture: str,
    formulation: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    config: ExperimentConfig,
    device: str,
    seed: int,
) -> TrainingOutcome:
    """Train with validation-only selection and restore the best state."""
    set_seeds(seed)
    target = torch.device(device)
    model = make_model(
        architecture, formulation, n_chans=x_train.shape[1], hidden=config.hidden
    ).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs - 1)
    )
    train_loader = _as_loader(
        x_train,
        y_train,
        batch_size=config.batch_size,
        seed=seed,
        shuffle=True,
    )
    validation_loader = _as_loader(
        x_validation,
        y_validation,
        batch_size=config.batch_size,
        seed=seed,
        shuffle=False,
    )
    schedule = DiffusionSchedule(config.diffusion_steps)
    noise_generator = _device_generator(target, seed + 10_000)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    updates = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(target)
            y_batch = y_batch.to(target)
            mask = torch.ones((len(x_batch), x_batch.shape[1]), device=target)
            optimizer.zero_grad(set_to_none=True)
            if formulation == "discriminative":
                loss = F.cross_entropy(model(x_batch, channel_mask=mask), y_batch)
            else:
                terms = diffusion_objective(
                    model,
                    x_batch,
                    y_batch,
                    mask,
                    schedule,
                    objective=config.objective,
                    generator=noise_generator,
                    margin=config.rank_margin,
                    lambda_rank=config.lambda_rank,
                    lambda_consistency=config.lambda_consistency,
                )
                loss = terms.total
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("Training produced a non-finite gradient")
            optimizer.step()
            updates += 1
            epoch_losses.append(float(loss.detach().cpu()))
        validation_loss = _validation_loss(
            model,
            validation_loader,
            formulation=formulation,
            schedule=schedule,
            device=target,
            seed=seed + epoch,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "updates": updates,
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid validation state")
    model.load_state_dict(best_state)
    return TrainingOutcome(model=model, history=history, best_epoch=best_epoch, updates=updates)


def predict_probabilities(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    formulation: str,
    config: ExperimentConfig,
    device: str,
    seed: int,
) -> np.ndarray:
    target = torch.device(device)
    loader = _as_loader(
        x, np.zeros(len(x), dtype=np.int64), batch_size=config.batch_size, seed=seed, shuffle=False
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    schedule = DiffusionSchedule(config.diffusion_steps)
    generator = _device_generator(target, seed)
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(target)
            mask = torch.ones((len(x_batch), x_batch.shape[1]), device=target)
            if formulation == "discriminative":
                scores = torch.softmax(model(x_batch, channel_mask=mask), dim=1)
            else:
                energies = class_energies(
                    model,
                    x_batch,
                    mask,
                    schedule,
                    k=config.inference_k,
                    generator=generator,
                )
                scores = energy_probabilities(energies)
            probabilities.append(scores[:, 1].cpu().numpy())
    return np.concatenate(probabilities)
