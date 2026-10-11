"""Deterministic validation-only training for V12 candidates."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepbench.reproducibility import set_seeds

from .catalogs import CandidateSpec
from .config import SearchConfig
from .io import write_torch_atomic
from .models import CandidateModel, make_candidate_model
from .objectives import LossTerms, candidate_loss, predict_probabilities


@dataclass
class TrainingOutcome:
    model: CandidateModel
    history: list[dict[str, float | int]]
    best_epoch: int
    updates: int


def _loader(
    x: np.ndarray, y: np.ndarray, *, batch_size: int, seed: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


_LOSS_NAMES = ("total", "classification", "denoising", "ranking", "consistency")


def _loss_values(terms: LossTerms) -> dict[str, float]:
    values = torch.stack([getattr(terms, name) for name in _LOSS_NAMES]).detach().cpu().tolist()
    return dict(zip(_LOSS_NAMES, (float(value) for value in values), strict=True))


def _mean_loss(
    model: CandidateModel,
    loader: DataLoader,
    candidate: CandidateSpec,
    config: SearchConfig,
    target: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = {name: [] for name in _LOSS_NAMES}
    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(target), y_batch.to(target)
            mask = torch.ones((len(x_batch), x_batch.shape[1]), device=target)
            terms = candidate_loss(
                model,
                x_batch,
                y_batch,
                mask,
                candidate=candidate,
                seed=seed + batch_index,
                diffusion_steps=config.diffusion_steps,
                rank_margin=config.rank_margin,
                lambda_denoise=config.lambda_denoise,
                lambda_rank=config.lambda_rank,
                lambda_consistency=config.lambda_consistency,
            )
            for name, value in _loss_values(terms).items():
                losses[name].append(value)
    return {name: float(np.mean(values)) for name, values in losses.items()}


def train_candidate(
    candidate: CandidateSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    config: SearchConfig,
    device: str,
    seed: int,
    checkpoint_path: Path | None = None,
) -> TrainingOutcome:
    set_seeds(seed)
    target = torch.device(device)
    model = make_candidate_model(candidate, n_chans=x_train.shape[1]).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate, weight_decay=candidate.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs - 1)
    )
    train_loader = _loader(x_train, y_train, batch_size=config.batch_size, seed=seed, shuffle=True)
    validation_loader = _loader(
        x_validation, y_validation, batch_size=config.batch_size, seed=seed, shuffle=False
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    updates = 0
    for epoch in range(config.epochs):
        model.train()
        batch_losses: dict[str, list[float]] = {name: [] for name in _LOSS_NAMES}
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(target), y_batch.to(target)
            mask = torch.ones((len(x_batch), x_batch.shape[1]), device=target)
            optimizer.zero_grad(set_to_none=True)
            terms = candidate_loss(
                model,
                x_batch,
                y_batch,
                mask,
                candidate=candidate,
                seed=seed + epoch * 100_000 + batch_index,
                diffusion_steps=config.diffusion_steps,
                rank_margin=config.rank_margin,
                lambda_denoise=config.lambda_denoise,
                lambda_rank=config.lambda_rank,
                lambda_consistency=config.lambda_consistency,
            )
            if not torch.isfinite(terms.total):
                raise RuntimeError("Training produced a non-finite loss")
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("Training produced a non-finite gradient")
            optimizer.step()
            updates += 1
            for name, value in _loss_values(terms).items():
                batch_losses[name].append(value)
        validation_losses = _mean_loss(
            model, validation_loader, candidate, config, target, seed + epoch * 10_000
        )
        epoch_report: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(batch_losses["total"])),
            "validation_loss": validation_losses["total"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "updates": updates,
        }
        for name in _LOSS_NAMES[1:]:
            epoch_report[f"train_{name}_loss"] = float(np.mean(batch_losses[name]))
            epoch_report[f"validation_{name}_loss"] = validation_losses[name]
        history.append(epoch_report)
        validation_loss = validation_losses["total"]
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        write_torch_atomic(
            checkpoint_path,
            {
                "schema_version": 1,
                "config_id": candidate.config_id,
                "candidate_sha256": candidate.sha256,
                "best_epoch": best_epoch,
                "updates": updates,
                "state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
            },
        )
    return TrainingOutcome(model=model, history=history, best_epoch=best_epoch, updates=updates)


def predict_numpy(
    model: CandidateModel,
    x: np.ndarray,
    *,
    candidate: CandidateSpec,
    config: SearchConfig,
    device: str,
    seed: int,
) -> np.ndarray:
    target = torch.device(device)
    loader = _loader(
        x, np.zeros(len(x), dtype=np.int64), batch_size=config.batch_size, seed=seed, shuffle=False
    )
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, (x_batch, _) in enumerate(loader):
            x_batch = x_batch.to(target)
            mask = torch.ones((len(x_batch), x_batch.shape[1]), device=target)
            probabilities = predict_probabilities(
                model,
                x_batch,
                mask,
                candidate=candidate,
                seed=seed + batch_index,
                inference_k=config.inference_k,
                diffusion_steps=config.diffusion_steps,
            )
            batches.append(probabilities.cpu().numpy())
    return np.concatenate(batches)
