"""Deterministic paired-view training for QCR-SSNet."""

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

from .catalog import QCRCandidate
from .models import QCRSSNet, corrupt_channels
from .objectives import qcr_loss


@dataclass(frozen=True)
class QCRCheckpoint:
    epoch: int
    validation_objective: float
    trajectory_sha256: str
    state_dict: dict[str, torch.Tensor]


@dataclass
class QCROutcome:
    model: QCRSSNet
    best: QCRCheckpoint
    last: QCRCheckpoint
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


def _paired_outputs(
    model: QCRSSNet,
    x: torch.Tensor,
    *,
    probability: float,
    max_missing: int,
    seed: int,
):
    corrupted, _, severity = corrupt_channels(
        x,
        probability=probability,
        max_missing=max_missing,
        seed=seed,
    )
    zero = torch.zeros_like(severity)
    clean_output = model(x, oracle_severity=zero)
    corrupt_output = model(corrupted, oracle_severity=severity)
    return clean_output, corrupt_output, severity


def _validation_objective(
    model: QCRSSNet,
    loader: DataLoader,
    candidate: QCRCandidate,
    *,
    device: torch.device,
    probability: float,
    max_missing: int,
    seed: int,
) -> tuple[float, float, float]:
    model.eval()
    clean_values: list[float] = []
    corrupt_values: list[float] = []
    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            clean, corrupt, _ = _paired_outputs(
                model,
                x_batch,
                probability=probability,
                max_missing=max_missing,
                seed=seed + batch_index,
            )
            clean_values.append(float(nn.functional.cross_entropy(clean.logits, y_batch).cpu()))
            corrupt_values.append(float(nn.functional.cross_entropy(corrupt.logits, y_batch).cpu()))
    clean_loss = float(np.mean(clean_values))
    corrupt_loss = float(np.mean(corrupt_values))
    objective = clean_loss + 0.25 * candidate.lambda_corrupt * corrupt_loss
    return objective, clean_loss, corrupt_loss


def train_qcr(
    model_factory: Callable[[], QCRSSNet],
    candidate: QCRCandidate,
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
    corruption_probability: float,
    max_missing_channels: int,
) -> QCROutcome:
    """Train one candidate and retain clean-primary best and last checkpoints."""

    if min(epochs, batch_size, max_missing_channels) < 1:
        raise ValueError("Training values must be positive")
    set_seeds(seed)
    target = torch.device(device)
    model = model_factory().to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 1))
    train_loader = _loader(x_train, y_train, batch_size=batch_size, seed=seed, shuffle=True)
    validation_loader = _loader(
        x_validation,
        y_validation,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )
    history: list[dict[str, float | int]] = []
    best_objective = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    updates = 0
    for epoch in range(epochs):
        model.train()
        totals: list[float] = []
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(target), y_batch.to(target)
            optimizer.zero_grad(set_to_none=True)
            clean, corrupt, severity = _paired_outputs(
                model,
                x_batch,
                probability=corruption_probability,
                max_missing=max_missing_channels,
                seed=seed * 1_000_000 + epoch * 10_000 + batch_index,
            )
            terms = qcr_loss(clean, corrupt, y_batch, severity, candidate)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals.append(float(terms.total.detach().cpu()))
            updates += 1
        validation_objective, clean_loss, corrupt_loss = _validation_objective(
            model,
            validation_loader,
            candidate,
            device=target,
            probability=1.0,
            max_missing=max_missing_channels,
            seed=seed * 10_000_000 + epoch * 10_000,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_total": float(np.mean(totals)),
                "validation_objective": validation_objective,
                "validation_clean_ce": clean_loss,
                "validation_corrupt_ce": corrupt_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "updates": updates,
            }
        )
        if validation_objective < best_objective - 1e-8:
            best_objective = validation_objective
            best_state = copy.deepcopy(_state_copy(model))
            best_epoch = epoch + 1
        scheduler.step()
    if best_state is None:
        raise RuntimeError("No finite Exp04 validation checkpoint was produced")
    last_state = _state_copy(model)
    trajectory_body = {
        "candidate_sha256": candidate.sha256,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "corruption_probability": corruption_probability,
        "max_missing_channels": max_missing_channels,
        "history": history,
        "best_state_sha256": _state_sha256(best_state),
        "last_state_sha256": _state_sha256(last_state),
    }
    trajectory_sha256 = canonical_sha256(trajectory_body)
    return QCROutcome(
        model=model,
        best=QCRCheckpoint(
            epoch=best_epoch,
            validation_objective=best_objective,
            trajectory_sha256=trajectory_sha256,
            state_dict=best_state,
        ),
        last=QCRCheckpoint(
            epoch=epochs,
            validation_objective=float(history[-1]["validation_objective"]),
            trajectory_sha256=trajectory_sha256,
            state_dict=last_state,
        ),
        history=history,
        updates=updates,
    )
