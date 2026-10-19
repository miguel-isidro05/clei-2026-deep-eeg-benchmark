"""Leakage-safe epoch selection and fixed-epoch refit for Exp05."""

from __future__ import annotations

import copy
import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepbench.reproducibility import set_seeds
from experiments.peterson_search_v12.common.identity import canonical_sha256


@dataclass(frozen=True)
class EpochSelection:
    selected_epoch: int
    best_validation_loss: float
    history: tuple[dict[str, float | int], ...]
    trajectory_sha256: str


@dataclass
class RefitOutcome:
    model: nn.Module
    epochs: int
    history: tuple[dict[str, float | int], ...]
    state_sha256: str
    updates: int


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_sha256(model_or_state: nn.Module | dict[str, torch.Tensor]) -> str:
    state = _state_copy(model_or_state) if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    sample_weights: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> DataLoader:
    weights = (
        np.ones(len(y), dtype=np.float32)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float32)
    )
    position_values = (
        np.full(len(y), -1, dtype=np.int64)
        if positions is None
        else np.asarray(positions, dtype=np.int64)
    )
    if len(x) != len(y) or len(weights) != len(y) or len(position_values) != len(y):
        raise ValueError("Exp05 loader arrays must have equal length")
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)),
            torch.from_numpy(weights),
            torch.from_numpy(position_values),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def _forward(
    model: nn.Module,
    x: torch.Tensor,
    *,
    positions: torch.Tensor | None,
    return_features: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {}
    if "position" in parameters and positions is not None:
        kwargs["position"] = positions
    if "positions" in parameters and positions is not None:
        kwargs["positions"] = positions
    if "return_features" in parameters:
        kwargs["return_features"] = return_features
    output = model(x, **kwargs)
    if isinstance(output, tuple):
        logits, features = output[0], output[1]
    else:
        logits, features = output, None
    return logits, features


def _optional_mechanisms() -> Any:
    from . import models as mechanisms

    return mechanisms


def _candidate_id(candidate: Any) -> str:
    return str(getattr(candidate, "config_id", getattr(candidate, "candidate_id", "unknown")))


def _optimizer(model: nn.Module, candidate: Any, *, learning_rate: float, weight_decay: float):
    mechanisms = _optional_mechanisms()
    if hasattr(mechanisms, "optimizer_parameter_groups"):
        groups = mechanisms.optimizer_parameter_groups(
            model,
            candidate,
            base_lr=learning_rate,
        )
    elif hasattr(model, "parameter_groups"):
        groups = model.parameter_groups(learning_rate)
    else:
        groups = model.parameters()
    return torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)


def _batch_objective(
    model: nn.Module,
    candidate: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mechanisms = _optional_mechanisms()
    if hasattr(mechanisms, "augment_batch"):
        x, y = mechanisms.augment_batch(x, y, candidate, seed=seed)
    logits, features = _forward(
        model,
        x,
        positions=None if bool(torch.all(positions < 0)) else positions,
        return_features=True,
    )
    losses = nn.functional.cross_entropy(logits, y, reduction="none")
    loss = torch.sum(losses * weights) / torch.clamp(weights.sum(), min=1e-6)
    if hasattr(mechanisms, "auxiliary_loss"):
        loss = loss + mechanisms.auxiliary_loss(
            model=model,
            logits=logits,
            features=features,
            y=y,
            candidate=candidate,
        )
    return loss, logits


def _validation_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y, _weights, positions in loader:
            x, y, positions = x.to(device), y.to(device), positions.to(device)
            logits, _ = _forward(
                model,
                x,
                positions=None if bool(torch.all(positions < 0)) else positions,
                return_features=False,
            )
            total += float(nn.functional.cross_entropy(logits, y, reduction="sum").cpu())
            count += int(len(y))
    if count == 0:
        raise RuntimeError("Empty inner-validation loader")
    return total / count


def select_epoch(
    model_factory: Callable[[], nn.Module],
    candidate: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    device: str,
    seed: int,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    train_weights: np.ndarray | None = None,
    train_positions: np.ndarray | None = None,
    validation_positions: np.ndarray | None = None,
) -> EpochSelection:
    """Select an epoch using only an inner split; never returns trained weights."""
    if min(max_epochs, batch_size, patience) < 1:
        raise ValueError("Exp05 training budgets must be positive")
    set_seeds(seed)
    target = torch.device(device)
    model = model_factory().to(target)
    optimizer = _optimizer(
        model,
        candidate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_epochs - 1)
    )
    train_loader = _loader(
        x_train,
        y_train,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        sample_weights=train_weights,
        positions=train_positions,
    )
    validation_loader = _loader(
        x_validation,
        y_validation,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        positions=validation_positions,
    )
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    updates = 0
    for epoch in range(max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch_index, (x, y, weights, positions) in enumerate(train_loader):
            x, y = x.to(target), y.to(target)
            weights, positions = weights.to(target), positions.to(target)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_objective(
                model,
                candidate,
                x,
                y,
                weights,
                positions,
                seed=seed * 10_000_000 + epoch * 10_000 + batch_index,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for {_candidate_id(candidate)}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            updates += 1
            epoch_losses.append(float(loss.detach().cpu()))
        validation_loss = _validation_loss(model, validation_loader, device=target)
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
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            break
    if best_epoch < 1:
        raise RuntimeError("Epoch selection produced no finite checkpoint")
    body = {
        "candidate": _candidate_id(candidate),
        "seed": seed,
        "max_epochs": max_epochs,
        "patience": patience,
        "history": history,
        "selected_epoch": best_epoch,
    }
    return EpochSelection(
        selected_epoch=best_epoch,
        best_validation_loss=best_loss,
        history=tuple(history),
        trajectory_sha256=canonical_sha256(body),
    )


def refit_fixed_epochs(
    model_factory: Callable[[], nn.Module],
    candidate: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: str,
    seed: int,
    epochs: int,
    scheduler_t_max: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    train_weights: np.ndarray | None = None,
    train_positions: np.ndarray | None = None,
) -> RefitOutcome:
    """Initialize from scratch and train on all outer-train data for a fixed epoch count."""
    if min(epochs, scheduler_t_max, batch_size) < 1:
        raise ValueError("Exp05 refit budgets must be positive")
    if epochs > scheduler_t_max:
        raise ValueError("Refit epochs cannot exceed the selection scheduler horizon")
    set_seeds(seed)
    target = torch.device(device)
    model = model_factory().to(target)
    initial_state = copy.deepcopy(_state_copy(model))
    optimizer = _optimizer(
        model,
        candidate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, scheduler_t_max - 1)
    )
    loader = _loader(
        x_train,
        y_train,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        sample_weights=train_weights,
        positions=train_positions,
    )
    history: list[dict[str, float | int]] = []
    updates = 0
    for epoch in range(epochs):
        model.train()
        values: list[float] = []
        for batch_index, (x, y, weights, positions) in enumerate(loader):
            x, y = x.to(target), y.to(target)
            weights, positions = weights.to(target), positions.to(target)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_objective(
                model,
                candidate,
                x,
                y,
                weights,
                positions,
                seed=seed * 10_000_000 + epoch * 10_000 + batch_index,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite refit loss for {_candidate_id(candidate)}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            updates += 1
            values.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(values)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "updates": updates,
            }
        )
        scheduler.step()
    final_hash = state_sha256(model)
    if final_hash == state_sha256(initial_state):
        raise RuntimeError("Refit did not change model state")
    return RefitOutcome(
        model=model,
        epochs=epochs,
        history=tuple(history),
        state_sha256=final_hash,
        updates=updates,
    )


def predict_window_scores(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: str,
    batch_size: int,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    target = torch.device(device)
    values: list[np.ndarray] = []
    position_values = (
        np.full(len(x), -1, dtype=np.int64)
        if positions is None
        else np.asarray(positions, dtype=np.int64)
    )
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(x[start : start + batch_size], dtype=np.float32)
            ).to(target)
            batch_positions = torch.from_numpy(position_values[start : start + batch_size]).to(
                target
            )
            logits, _ = _forward(
                model,
                batch,
                positions=None if bool(torch.all(batch_positions < 0)) else batch_positions,
                return_features=False,
            )
            values.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(values)
