"""Deterministic exp03 training with paired last/best checkpoints."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepbench.reproducibility import set_seeds
from experiments.peterson_search_v12.common.identity import canonical_sha256


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    validation_loss: float
    trajectory_sha256: str
    state_dict: dict[str, torch.Tensor]


@dataclass
class TrajectoryOutcome:
    model: nn.Module
    last: CheckpointState
    best: CheckpointState
    history: list[dict[str, float | int]]
    updates: int


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


def _forward(model: nn.Module, x: torch.Tensor, dropout_seed: int | None) -> torch.Tensor:
    return model(x, dropout_seed=dropout_seed)


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(values.shape)).encode("utf-8"))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _validation_loss(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            values.append(float(criterion(_forward(model, x_batch, None), y_batch).cpu()))
    return float(np.mean(values))


def train_checkpoint_trajectory(
    model_factory: Callable[[], nn.Module],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int | None,
) -> TrajectoryOutcome:
    if min(epochs, batch_size) < 1:
        raise ValueError("Epochs and batch size must be positive")
    set_seeds(seed)
    target = torch.device(device)
    model = model_factory().to(target)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 1))
    train_loader = _loader(x_train, y_train, batch_size=batch_size, seed=seed, shuffle=True)
    validation_loader = _loader(
        x_validation, y_validation, batch_size=batch_size, seed=seed, shuffle=False
    )
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    updates = 0
    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(target), y_batch.to(target)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(
                model,
                x_batch,
                seed * 1_000_000 + epoch * 10_000 + batch_index,
            )
            loss = criterion(logits, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            updates += 1
            train_losses.append(float(loss.detach().cpu()))
        validation_loss = _validation_loss(model, validation_loader, criterion, target)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "updates": updates,
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(_state_copy(model))
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if patience is not None and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    last_state = _state_copy(model)
    trajectory_body = {
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "history": history,
        "last_state_sha256": _state_sha256(last_state),
        "best_state_sha256": _state_sha256(best_state),
    }
    trajectory_sha256 = canonical_sha256(trajectory_body)
    return TrajectoryOutcome(
        model=model,
        last=CheckpointState(
            epoch=len(history),
            validation_loss=float(history[-1]["validation_loss"]),
            trajectory_sha256=trajectory_sha256,
            state_dict=last_state,
        ),
        best=CheckpointState(
            epoch=best_epoch,
            validation_loss=best_loss,
            trajectory_sha256=trajectory_sha256,
            state_dict=best_state,
        ),
        history=history,
        updates=updates,
    )
