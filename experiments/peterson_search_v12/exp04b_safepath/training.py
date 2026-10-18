"""Deterministic anchor caching and SafePath training."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepbench.reproducibility import set_seeds
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.io import write_torch_atomic

from .catalog import SafePathCandidate
from .corruptions import training_corruption
from .models import SafePathQCR
from .objectives import safepath_loss


@dataclass(frozen=True)
class ModelCheckpoint:
    epoch: int
    objective: float
    trajectory_sha256: str
    state_dict: dict[str, torch.Tensor]


@dataclass
class SafePathOutcome:
    model: SafePathQCR
    best: ModelCheckpoint
    last: ModelCheckpoint
    history: list[dict[str, float | int]]
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


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_or_train_anchor(
    checkpoint_path: Path,
    model_factory: Callable[[], nn.Module],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    identity: dict[str, object],
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[nn.Module, dict[str, object]]:
    """Reuse a signed fold-specific clean anchor or train it once."""

    expected = canonical_sha256(identity)
    model = model_factory()
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("identity_sha256") != expected:
            raise RuntimeError(f"Anchor identity mismatch: {checkpoint_path.name}")
        model.load_state_dict(payload["state_dict"])
        return model, payload

    set_seeds(seed)
    target = torch.device(device)
    model = model.to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 1))
    train_loader = _loader(x_train, y_train, batch_size=batch_size, seed=seed, shuffle=True)
    validation_loader = _loader(
        x_validation, y_validation, batch_size=batch_size, seed=seed, shuffle=False
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(target), y_batch.to(target)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x_batch), y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Anchor training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        values: list[float] = []
        with torch.no_grad():
            for x_batch, y_batch in validation_loader:
                x_batch, y_batch = x_batch.to(target), y_batch.to(target)
                values.append(float(nn.functional.cross_entropy(model(x_batch), y_batch).cpu()))
        validation_loss = float(np.mean(values))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(_state_copy(model))
        scheduler.step()
    if best_state is None:
        raise RuntimeError("Anchor training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    payload: dict[str, object] = {
        "schema_version": 1,
        "identity": identity,
        "identity_sha256": expected,
        "selected_epoch": best_epoch,
        "validation_loss": best_loss,
        "trajectory_sha256": canonical_sha256(
            {"identity_sha256": expected, "history": history, "state": _state_sha256(best_state)}
        ),
        "state_dict": best_state,
    }
    write_torch_atomic(checkpoint_path, payload)
    return model, payload


def _validation_objective(
    model: SafePathQCR,
    loader: DataLoader,
    candidate: SafePathCandidate,
    *,
    device: torch.device,
    max_missing: int,
    seed: int,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            corrupted, mask, severity = training_corruption(
                x_batch,
                family=candidate.corruption_family,
                max_missing=max_missing,
                seed=seed + batch_index,
            )
            clean_mask = torch.ones_like(mask)
            batch_seed = seed + batch_index * 2
            clean = model(
                x_batch,
                channel_mask=clean_mask,
                dropout_seed=batch_seed,
            )
            corrupt_mask = None if bool(torch.all(mask == 1)) else mask
            corrupt = model(
                corrupted,
                channel_mask=corrupt_mask,
                corruption_score=severity,
                dropout_seed=batch_seed + 1,
            )
            values.append(
                float(
                    safepath_loss(
                        clean.logits,
                        corrupt.logits,
                        y_batch,
                        objective=candidate.objective,
                        router_weight=corrupt.router_weight,
                        corruption_score=severity,
                    ).cpu()
                )
            )
    return float(np.mean(values))


def train_safepath(
    model: SafePathQCR,
    candidate: SafePathCandidate,
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
    max_missing: int,
) -> SafePathOutcome:
    set_seeds(seed)
    target = torch.device(device)
    model = model.to(target)
    if candidate.training_mode == "joint":
        for parameter in model.clean_expert.parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        state = _state_copy(model)
        digest = canonical_sha256({"candidate": candidate.sha256, "state": _state_sha256(state)})
        checkpoint = ModelCheckpoint(0, 0.0, digest, state)
        return SafePathOutcome(model, checkpoint, checkpoint, [], 0)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 1))
    train_loader = _loader(x_train, y_train, batch_size=batch_size, seed=seed, shuffle=True)
    validation_loader = _loader(
        x_validation, y_validation, batch_size=batch_size, seed=seed, shuffle=False
    )
    history: list[dict[str, float | int]] = []
    best_value = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    updates = 0
    for epoch in range(epochs):
        model.train()
        if candidate.training_mode == "two_stage" and model.router is not None:
            train_router = epoch >= epochs // 2
            for parameter in model.router.parameters():
                parameter.requires_grad_(train_router)
        losses: list[float] = []
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(target), y_batch.to(target)
            corrupted, mask, severity = training_corruption(
                x_batch,
                family=candidate.corruption_family,
                max_missing=max_missing,
                seed=seed * 1_000_000 + epoch * 10_000 + batch_index,
            )
            clean_mask = torch.ones_like(mask)
            optimizer.zero_grad(set_to_none=True)
            batch_seed = seed * 1_000_000 + epoch * 10_000 + batch_index * 2
            clean = model(
                x_batch,
                channel_mask=clean_mask,
                dropout_seed=batch_seed,
            )
            corrupt_mask = None if bool(torch.all(mask == 1)) else mask
            corrupt = model(
                corrupted,
                channel_mask=corrupt_mask,
                corruption_score=severity,
                dropout_seed=batch_seed + 1,
            )
            loss = safepath_loss(
                clean.logits,
                corrupt.logits,
                y_batch,
                objective=candidate.objective,
                router_weight=corrupt.router_weight,
                corruption_score=severity,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("SafePath training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            updates += 1
        validation = _validation_objective(
            model,
            validation_loader,
            candidate,
            device=target,
            max_missing=max_missing,
            seed=seed * 10_000_000 + epoch * 10_000,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_objective": validation,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "updates": updates,
            }
        )
        if validation < best_value - 1e-8:
            best_value = validation
            best_epoch = epoch + 1
            best_state = copy.deepcopy(_state_copy(model))
        scheduler.step()
    if best_state is None:
        raise RuntimeError("SafePath training did not produce a finite checkpoint")
    last_state = _state_copy(model)
    trajectory = canonical_sha256(
        {
            "candidate": candidate.sha256,
            "history": history,
            "best": _state_sha256(best_state),
            "last": _state_sha256(last_state),
        }
    )
    model.load_state_dict(best_state)
    return SafePathOutcome(
        model=model,
        best=ModelCheckpoint(best_epoch, best_value, trajectory, best_state),
        last=ModelCheckpoint(
            epochs,
            float(history[-1]["validation_objective"]),
            trajectory,
            last_state,
        ),
        history=history,
        updates=updates,
    )
