"""Deterministic candidate training and multi-recipe inference for exp02."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepbench.reproducibility import set_seeds
from experiments.peterson_search_v12.common.io import write_torch_atomic

from .catalog import Exp02Candidate
from .config import Exp02Config
from .models import Exp02Model, make_model
from .objectives import LossTerms, candidate_loss, inference_probabilities


@dataclass
class TrainingOutcome:
    model: Exp02Model
    history: list[dict[str, float | int]]
    best_epoch: int
    updates: int


LOSS_NAMES = ("total", "classification", "denoising", "ranking", "consistency")


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
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


def _loss_values(terms: LossTerms) -> dict[str, float]:
    return {name: float(getattr(terms, name).detach().cpu()) for name in LOSS_NAMES}


def _mean_losses(
    model: Exp02Model,
    loader: DataLoader,
    candidate: Exp02Candidate,
    config: Exp02Config,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    values: dict[str, list[float]] = {name: [] for name in LOSS_NAMES}
    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            mask = torch.ones((len(x_batch), x_batch.shape[1]), device=device)
            terms = candidate_loss(
                model,
                x_batch,
                y_batch,
                mask,
                candidate=candidate,
                seed=seed + batch_index,
                diffusion_steps=config.diffusion_steps,
                rank_margin=config.rank_margin,
            )
            for name, value in _loss_values(terms).items():
                values[name].append(value)
    return {name: float(np.mean(items)) for name, items in values.items()}


def train_candidate(
    candidate: Exp02Candidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    sfreq: float,
    config: Exp02Config,
    device: str,
    seed: int,
    checkpoint_path: Path | None = None,
) -> TrainingOutcome:
    set_seeds(seed)
    target = torch.device(device)
    model = make_model(
        candidate,
        n_chans=x_train.shape[1],
        n_times=x_train.shape[-1],
        sfreq=sfreq,
    ).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs - 1)
    )
    train_loader = _loader(x_train, y_train, batch_size=config.batch_size, seed=seed, shuffle=True)
    validation_loader = _loader(
        x_validation,
        y_validation,
        batch_size=config.batch_size,
        seed=seed,
        shuffle=False,
    )
    best_value = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    updates = 0
    for epoch in range(config.epochs):
        model.train()
        train_values: dict[str, list[float]] = {name: [] for name in LOSS_NAMES}
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
                train_values[name].append(value)
        validation = _mean_losses(
            model,
            validation_loader,
            candidate,
            config,
            target,
            seed + epoch * 10_000,
        )
        report: dict[str, float | int] = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "updates": updates,
        }
        for name in LOSS_NAMES:
            report[f"train_{name}_loss"] = float(np.mean(train_values[name]))
            report[f"validation_{name}_loss"] = validation[name]
        history.append(report)
        monitored = validation[candidate.checkpoint_metric]
        if monitored < best_value - 1e-8:
            best_value = monitored
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
                "schema_version": 2,
                "config_id": candidate.config_id,
                "candidate_sha256": candidate.sha256,
                "best_epoch": best_epoch,
                "updates": updates,
                "state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
            },
        )
    return TrainingOutcome(model, history, best_epoch, updates)


def predict_all(
    model: Exp02Model,
    x: np.ndarray,
    *,
    candidate: Exp02Candidate,
    config: Exp02Config,
    device: str,
    seed: int,
) -> dict[str, np.ndarray]:
    target = torch.device(device)
    loader = _loader(
        x,
        np.zeros(len(x), dtype=np.int64),
        batch_size=config.batch_size,
        seed=seed,
        shuffle=False,
    )
    outputs: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for batch_index, (x_batch, _) in enumerate(loader):
            x_batch = x_batch.to(target)
            mask = torch.ones((len(x_batch), x_batch.shape[1]), device=target)
            probabilities = inference_probabilities(
                model,
                x_batch,
                mask,
                candidate=candidate,
                seed=seed + batch_index,
                diffusion_steps=config.diffusion_steps,
            )
            for inference_id, values in probabilities.items():
                outputs.setdefault(inference_id, []).append(values.detach().cpu().numpy())
    return {key: np.concatenate(values) for key, values in outputs.items()}
