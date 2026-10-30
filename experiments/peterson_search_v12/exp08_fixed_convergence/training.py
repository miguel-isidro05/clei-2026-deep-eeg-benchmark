"""Fixed-convergence crop training and frozen positional fusion for Exp08."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from deepbench.reproducibility import set_seeds
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.training import state_sha256


@dataclass(frozen=True)
class FusionArtifact:
    window_logits: np.ndarray
    window_weights: np.ndarray
    epochs: int
    history: tuple[dict[str, float | int], ...]
    trainable_parameters: int
    backbone_state_before: str
    backbone_state_after: str


def freeze_backbone(backbone: nn.Module) -> str:
    """Freeze parameters and force evaluation mode, including BatchNorm buffers."""

    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return state_sha256(backbone)


def _validate_logits(grouped_logits: np.ndarray) -> np.ndarray:
    values = np.asarray(grouped_logits, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (6, 2):
        raise ValueError("Exp08 grouped logits require exactly six windows and two classes")
    if not np.isfinite(values).all():
        raise ValueError("Exp08 grouped logits must be finite")
    return values


def aggregate_grouped_logits(
    grouped_logits: np.ndarray,
    *,
    aggregation: str,
    window_logits: np.ndarray | None = None,
) -> np.ndarray:
    values = _validate_logits(grouped_logits)
    if aggregation == "uniform":
        weights = np.full(6, 1.0 / 6.0, dtype=np.float32)
    elif aggregation == "lpwf":
        if window_logits is None or np.asarray(window_logits).shape != (6,):
            raise ValueError("LPWF requires exactly six window logits")
        raw = np.asarray(window_logits, dtype=np.float64)
        weights = np.exp(raw - raw.max())
        weights = (weights / weights.sum()).astype(np.float32)
    else:
        raise ValueError(f"Unsupported Exp08 aggregation: {aggregation}")
    trial_logits = np.sum(values * weights[None, :, None], axis=1)
    shifted = trial_logits - trial_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1].astype(float)


def fit_frozen_lpwf(
    backbone: nn.Module,
    grouped_logits: np.ndarray,
    labels: np.ndarray,
    *,
    device: str,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> FusionArtifact:
    values = _validate_logits(grouped_logits)
    y = np.asarray(labels, dtype=np.int64)
    if len(y) != len(values) or set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Exp08 fusion labels are invalid")
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("Exp08 fusion budget must be positive")
    before = freeze_backbone(backbone)
    set_seeds(seed)
    target = torch.device(device)
    logits = torch.from_numpy(values).to(target)
    targets = torch.from_numpy(y).to(target)
    positional_logits = nn.Parameter(torch.zeros(6, device=target))
    optimizer = torch.optim.AdamW([positional_logits], lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - 1)
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(positional_logits, dim=0)
        trial_logits = torch.sum(logits * weights[None, :, None], dim=1)
        loss = nn.functional.cross_entropy(trial_logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Exp08 fusion loss")
        loss.backward()
        optimizer.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(loss.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        scheduler.step()
    after = state_sha256(backbone)
    if before != after:
        raise RuntimeError("Exp08 fusion modified the frozen backbone")
    raw = positional_logits.detach().cpu()
    weights = torch.softmax(raw, dim=0)
    return FusionArtifact(
        window_logits=np.asarray(raw.tolist(), dtype=float),
        window_weights=np.asarray(weights.tolist(), dtype=float),
        epochs=epochs,
        history=tuple(history),
        trainable_parameters=6,
        backbone_state_before=before,
        backbone_state_after=after,
    )


def predict_grouped_logits(
    backbone: nn.Module,
    grouped_windows: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Return immutable per-window logits with shape trial x six x class."""

    values = np.asarray(grouped_windows, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != 6:
        raise ValueError("Exp08 expects trial x six x channel x time windows")
    freeze_backbone(backbone)
    target = torch.device(device)
    backbone.to(target)
    flat = values.reshape(-1, values.shape[2], values.shape[3])
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(flat[start : start + batch_size])).to(
                target
            )
            logits = backbone(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError("Exp08 backbone returned invalid logits")
            outputs.append(logits.detach().cpu().numpy())
    result = np.concatenate(outputs).reshape(len(values), 6, 2)
    return _validate_logits(result)

