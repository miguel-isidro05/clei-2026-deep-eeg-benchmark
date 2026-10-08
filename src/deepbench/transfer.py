"""Cross-dataset Peterson-to-Souza transfer utilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from .config import MI_SUBJECTS
from .models import make_module
from .reproducibility import set_seeds

TRANSFER_ARMS = ("scratch", "linear_probe", "full_finetune")
HEAD_PREFIX = "final_layer."
SOURCE_VALIDATION_SUBJECTS = MI_SUBJECTS[:5]


def source_validation_subject(seed: int) -> str:
    """Return the predeclared whole-subject validation rotation for source training."""
    return SOURCE_VALIDATION_SUBJECTS[int(seed) % len(SOURCE_VALIDATION_SUBJECTS)]


def inner_validation_split(
    outer_train: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    *,
    protocol: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split only the outer-training partition for epoch selection."""
    indices = np.asarray(outer_train, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Outer-training partition is empty")
    if protocol == "within_session":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_local, validation_local = next(splitter.split(indices, labels[indices]))
        return indices[train_local], indices[validation_local]
    if protocol == "cross_session":
        runs = sorted({str(value) for value in sessions[indices]})
        if len(runs) < 2:
            raise ValueError("Cross-session inner validation requires at least two training runs")
        held_run = runs[int(seed) % len(runs)]
        validation_mask = np.asarray([str(value) == held_run for value in sessions[indices]])
        return indices[~validation_mask], indices[validation_mask]
    raise ValueError(f"Unsupported transfer protocol: {protocol}")


def trainable_parameter_names(module: torch.nn.Module) -> list[str]:
    """Return trainable parameter names in stable module order."""
    return [name for name, parameter in module.named_parameters() if parameter.requires_grad]


def _load_backbone(
    module: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> None:
    target_state = module.state_dict()
    backbone_state = {
        name: value for name, value in source_state.items() if not name.startswith(HEAD_PREFIX)
    }
    if not backbone_state:
        raise ValueError("Source checkpoint does not contain transferable backbone weights")
    unknown = sorted(set(backbone_state) - set(target_state))
    if unknown:
        raise ValueError(f"Source checkpoint has unknown backbone keys: {unknown[:5]}")
    mismatched = sorted(
        name
        for name, value in backbone_state.items()
        if tuple(value.shape) != tuple(target_state[name].shape)
    )
    if mismatched:
        raise ValueError(f"Source backbone shapes do not match target: {mismatched[:5]}")
    incompatible = module.load_state_dict(backbone_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_head_missing = [
        name for name in incompatible.missing_keys if not name.startswith(HEAD_PREFIX)
    ]
    if unexpected or non_head_missing:
        raise ValueError(
            f"Incomplete source backbone load: unexpected={unexpected} missing={non_head_missing}"
        )


def make_target_module(
    model_name: str,
    *,
    arm: str,
    seed: int,
    source_state: Mapping[str, torch.Tensor] | None,
    n_chans: int,
    n_times: int,
    sfreq: float,
) -> torch.nn.Module:
    """Create a target module with a fresh task head and an optional source backbone."""
    if arm not in TRANSFER_ARMS:
        raise ValueError(f"Unknown transfer arm {arm!r}; expected one of {TRANSFER_ARMS}")
    if arm == "scratch" and source_state is not None:
        raise ValueError("scratch arm must not receive a source state")
    if arm != "scratch" and source_state is None:
        raise ValueError(f"{arm} requires a Peterson source state")
    set_seeds(seed)
    module = make_module(
        model_name,
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        sfreq=sfreq,
    )
    if source_state is not None:
        _load_backbone(module, source_state)
    if arm == "linear_probe":
        for name, parameter in module.named_parameters():
            parameter.requires_grad = name.startswith(HEAD_PREFIX)
    return module


def optimizer_parameter_groups(
    module: torch.nn.Module,
    *,
    arm: str,
    base_lr: float,
) -> list[dict[str, object]]:
    """Return explicit parameter groups for scratch, probe, and full fine-tuning."""
    if arm not in TRANSFER_ARMS:
        raise ValueError(f"Unknown transfer arm {arm!r}")
    head = [
        parameter
        for name, parameter in module.named_parameters()
        if name.startswith(HEAD_PREFIX) and parameter.requires_grad
    ]
    backbone = [
        parameter
        for name, parameter in module.named_parameters()
        if not name.startswith(HEAD_PREFIX) and parameter.requires_grad
    ]
    if not head:
        raise ValueError("Target module exposes no trainable final_layer parameters")
    if arm == "full_finetune":
        if not backbone:
            raise ValueError("Full fine-tuning requires trainable backbone parameters")
        return [
            {"params": backbone, "lr": float(base_lr) * 0.1},
            {"params": head, "lr": float(base_lr)},
        ]
    parameters = backbone + head
    return [{"params": parameters, "lr": float(base_lr)}]
