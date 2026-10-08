"""Leakage-safe source pretraining and Souza target transfer evaluation."""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import MI_CHANNELS, MI_SUBJECTS, SPLIT_SEED
from .datasets import align_recording_channels, load_subject
from .evaluation import _index_hash, _protocol_splits
from .io import write_json_atomic
from .metrics import compute_metrics
from .models import TRAINING_RECIPES, make_module, recipe_dict
from .preprocessing import (
    _window_starts,
    aggregate_trial_probabilities,
    augment_training,
    prepare_test_windows,
    preprocess_split,
)
from .reproducibility import set_seeds
from .runner import _code_fingerprint, _environment_identity, _git_revision
from .transfer import (
    inner_validation_split,
    make_target_module,
    optimizer_parameter_groups,
    source_validation_subject,
    trainable_parameter_names,
)
from .transfer_profile import TRANSFER_PROFILE_VERSION

# Commit 4645dd8 produced scientifically valid checkpoints whose manifests contain
# NumPy int64 window starts. The restricted PyTorch loader rejects that metadata.
# This exact fingerprint is accepted only so those checkpoints can be resumed.
LEGACY_SOURCE_CODE_SHA256 = {
    "d49f099e0ce458c03980c580426e3a5bf7ee02827e7bc37939a1b938ad8d19b0",
}


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _forward_logits(module: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output = module(inputs)
    return output[0] if isinstance(output, tuple) else output


def _make_seeded_source_module(model: str, seed: int) -> torch.nn.Module:
    """Construct a source model only after fixing every initialization RNG."""
    set_seeds(seed)
    return make_module(model, n_chans=15, n_outputs=2, n_times=256, sfreq=128.0)


def _optimizer(module: torch.nn.Module, model: str, arm: str) -> torch.optim.Optimizer:
    recipe = TRAINING_RECIPES[model]
    optimizer_class = torch.optim.AdamW if recipe.optimizer == "adamw" else torch.optim.Adam
    groups = optimizer_parameter_groups(module, arm=arm, base_lr=recipe.lr)
    return optimizer_class(
        groups,
        weight_decay=recipe.weight_decay,
        betas=recipe.betas,
    )


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def select_epoch(
    module: torch.nn.Module,
    model: str,
    arm: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    device: str,
    seed: int,
    max_epochs: int = 300,
    min_epochs: int = 30,
    patience: int = 50,
    threshold: float = 1e-4,
) -> tuple[int, list[dict[str, float]]]:
    """Select an epoch using validation loss only; never sees an outer-test trial."""
    if not 1 <= min_epochs <= max_epochs or patience < 1:
        raise ValueError("Require 1 <= min_epochs <= max_epochs and patience >= 1")
    set_seeds(seed)
    module.to(device)
    optimizer = _optimizer(module, model, arm)
    criterion = torch.nn.CrossEntropyLoss()
    loader = _loader(x_train, y_train, TRAINING_RECIPES[model].batch_size, seed)
    validation_x = torch.from_numpy(np.ascontiguousarray(x_validation, dtype=np.float32)).to(device)
    validation_y = torch.from_numpy(np.asarray(y_validation, dtype=np.int64)).to(device)
    best_loss = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        module.train()
        train_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_forward_logits(module, batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(batch_y)
            seen += len(batch_y)
        module.eval()
        with torch.no_grad():
            validation_loss = float(criterion(_forward_logits(module, validation_x), validation_y))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss / seen,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - threshold:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in module.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Epoch selection completed without a checkpoint")
    module.load_state_dict(best_state)
    return best_epoch, history


def fit_fixed_epochs(
    module: torch.nn.Module,
    model: str,
    arm: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: str,
    seed: int,
    epochs: int,
) -> torch.nn.Module:
    """Refit on the complete outer-training partition for the selected epoch count."""
    set_seeds(seed)
    module.to(device)
    optimizer = _optimizer(module, model, arm)
    criterion = torch.nn.CrossEntropyLoss()
    loader = _loader(x, y, TRAINING_RECIPES[model].batch_size, seed)
    for _ in range(epochs):
        module.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_forward_logits(module, batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return module


def _predict_trials(
    module: torch.nn.Module, x: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray]:
    windows, trial_indices = prepare_test_windows(x, "overlap")
    module.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), 256):
            batch = torch.from_numpy(windows[start : start + 256]).to(device)
            scores.append(torch.softmax(_forward_logits(module, batch), dim=1)[:, 1].cpu().numpy())
    trial_scores = aggregate_trial_probabilities(
        np.concatenate(scores), trial_indices, n_trials=len(x)
    )
    return (trial_scores >= 0.5).astype(np.int64), trial_scores


def _weights_only_safe(value: Any) -> Any:
    """Convert checkpoint metadata to types accepted by PyTorch's restricted loader."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return _weights_only_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _weights_only_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_weights_only_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_weights_only_safe(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    torch.save(_weights_only_safe(payload), temporary)
    temporary.replace(destination)


def pretrain_peterson_source(
    model: str,
    seed: int,
    *,
    device: str,
    output_dir: Path,
    max_epochs: int = 300,
    min_epochs: int = 30,
    patience: int = 50,
) -> Path:
    """Pretrain one model/seed on Peterson with whole-subject validation and refit."""
    validation_subject = source_validation_subject(seed)
    recordings = {subject: load_subject("MI-OpenBCI", subject) for subject in MI_SUBJECTS}
    train_subjects = [subject for subject in MI_SUBJECTS if subject != validation_subject]
    x_train_raw = np.concatenate([recordings[subject].x for subject in train_subjects])
    y_train_raw = np.concatenate([recordings[subject].y for subject in train_subjects])
    x_validation_raw = recordings[validation_subject].x
    y_validation_raw = recordings[validation_subject].y
    selected = preprocess_split(
        x_train_raw,
        x_validation_raw,
        sfreq=128.0,
        ch_names=MI_CHANNELS,
        seed=SPLIT_SEED,
        ica_policy="none",
    )
    x_train, y_train = augment_training(selected.x_train, y_train_raw, "overlap")
    x_validation, y_validation = augment_training(selected.x_test, y_validation_raw, "overlap")
    candidate = _make_seeded_source_module(model, seed)
    best_epoch, history = select_epoch(
        candidate,
        model,
        "scratch",
        x_train,
        y_train,
        x_validation,
        y_validation,
        device=device,
        seed=seed,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        patience=patience,
    )
    validation_prediction, validation_score = _predict_trials(
        candidate, selected.x_test, device
    )
    validation_metrics = compute_metrics(
        y_validation_raw, validation_prediction, validation_score
    )
    x_all_raw = np.concatenate([x_train_raw, x_validation_raw])
    y_all_raw = np.concatenate([y_train_raw, y_validation_raw])
    refit_processed = preprocess_split(
        x_all_raw,
        x_all_raw[:1],
        sfreq=128.0,
        ch_names=MI_CHANNELS,
        seed=SPLIT_SEED,
        ica_policy="none",
    )
    x_all, y_all = augment_training(refit_processed.x_train, y_all_raw, "overlap")
    refit = _make_seeded_source_module(model, seed)
    fit_fixed_epochs(
        refit, model, "scratch", x_all, y_all, device=device, seed=seed, epochs=best_epoch
    )
    state = {name: value.detach().cpu() for name, value in refit.state_dict().items()}
    environment_sha256, environment_versions = _environment_identity()
    manifest = {
        "schema_version": 1,
        "profile": TRANSFER_PROFILE_VERSION,
        "dataset": "MI-OpenBCI",
        "task": "motor_imagery_vs_rest",
        "model": model,
        "seed": seed,
        "condition": "overlap",
        "channels": list(MI_CHANNELS),
        "n_times": 256,
        "sfreq": 128.0,
        "window_policy": {
            "name": "six_overlap_complete_trial",
            "window_samples": 256,
            "source_trial_samples": int(x_train_raw.shape[-1]),
            "starts": list(_window_starts(int(x_train_raw.shape[-1]), "overlap")),
        },
        "validation_subject": validation_subject,
        "train_subjects": train_subjects,
        "selected_epoch": best_epoch,
        "recipe": recipe_dict(model, best_epoch),
        "history": history,
        "source_validation_metrics": validation_metrics,
        "data_sha256": {subject: recordings[subject].data_sha256 for subject in MI_SUBJECTS},
        "scientific_code_sha256": _code_fingerprint(),
        "git_revision": _git_revision(),
        "environment_sha256": environment_sha256,
        "environment_versions": environment_versions,
        "state_sha256": _state_sha256(state),
    }
    destination = output_dir / "source_checkpoints" / f"source__{model}__seed{seed}.pt"
    _atomic_torch_save({"state_dict": state, "manifest": manifest}, destination)
    write_json_atomic(destination.with_suffix(".json"), manifest)
    return destination


def load_source_checkpoint(path: Path, *, model: str, seed: int) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Compatibility path for checkpoints written by 4645dd8. It keeps the
        # restricted loader and allowlists only the NumPy scalar types we wrote.
        numpy_scalar = np._core.multiarray.scalar
        numpy_int64_dtype = type(np.dtype(np.int64))
        with torch.serialization.safe_globals(
            [numpy_scalar, np.dtype, numpy_int64_dtype]
        ):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = payload.get("manifest", {})
    required = {
        "profile": TRANSFER_PROFILE_VERSION,
        "dataset": "MI-OpenBCI",
        "task": "motor_imagery_vs_rest",
        "model": model,
        "seed": seed,
        "condition": "overlap",
        "channels": list(MI_CHANNELS),
        "n_times": 256,
        "sfreq": 128.0,
    }
    mismatched = [key for key, value in required.items() if manifest.get(key) != value]
    accepted_code = {_code_fingerprint(), *LEGACY_SOURCE_CODE_SHA256}
    if manifest.get("scientific_code_sha256") not in accepted_code:
        mismatched.append("scientific_code_sha256")
    if not isinstance(manifest.get("environment_sha256"), str) or len(
        manifest["environment_sha256"]
    ) != 64:
        mismatched.append("environment_sha256")
    state = payload.get("state_dict")
    if (
        mismatched
        or not isinstance(state, dict)
        or manifest.get("state_sha256") != _state_sha256(state)
    ):
        raise RuntimeError(f"Incompatible or stale source checkpoint {path}: {mismatched}")
    return state


def run_souza_transfer_cell(
    subject: str,
    model: str,
    seed: int,
    arm: str,
    protocol: str,
    *,
    device: str,
    output_dir: Path,
    max_epochs: int = 300,
    min_epochs: int = 30,
    patience: int = 40,
) -> Path:
    """Run one complete target cell with nested epoch selection and outer-test isolation."""
    recording = align_recording_channels(load_subject("Souza2023", subject), MI_CHANNELS)
    source_path = output_dir / "source_checkpoints" / f"source__{model}__seed{seed}.pt"
    source_state = (
        None if arm == "scratch" else load_source_checkpoint(source_path, model=model, seed=seed)
    )
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_score_all: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for fold, (outer_train, outer_test, split_report) in enumerate(
        _protocol_splits(recording, protocol, SPLIT_SEED)
    ):
        inner_train, inner_validation = inner_validation_split(
            outer_train,
            recording.y,
            recording.sessions,
            protocol=protocol,
            seed=SPLIT_SEED + fold,
        )
        selected = preprocess_split(
            recording.x[inner_train],
            recording.x[inner_validation],
            sfreq=recording.sfreq,
            ch_names=recording.ch_names,
            seed=SPLIT_SEED + fold,
            ica_policy="none",
        )
        x_inner, y_inner = augment_training(selected.x_train, recording.y[inner_train], "overlap")
        x_validation, y_validation = augment_training(
            selected.x_test, recording.y[inner_validation], "overlap"
        )
        candidate = make_target_module(
            model,
            arm=arm,
            seed=seed * 1000 + fold,
            source_state=source_state,
            n_chans=15,
            n_times=256,
            sfreq=128.0,
        )
        best_epoch, history = select_epoch(
            candidate,
            model,
            arm,
            x_inner,
            y_inner,
            x_validation,
            y_validation,
            device=device,
            seed=seed * 1000 + fold,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            patience=patience,
        )
        outer = preprocess_split(
            recording.x[outer_train],
            recording.x[outer_test],
            sfreq=recording.sfreq,
            ch_names=recording.ch_names,
            seed=SPLIT_SEED + fold,
            ica_policy="none",
        )
        x_outer, y_outer = augment_training(outer.x_train, recording.y[outer_train], "overlap")
        refit = make_target_module(
            model,
            arm=arm,
            seed=seed * 1000 + fold,
            source_state=source_state,
            n_chans=15,
            n_times=256,
            sfreq=128.0,
        )
        fit_fixed_epochs(
            refit,
            model,
            arm,
            x_outer,
            y_outer,
            device=device,
            seed=seed * 1000 + fold,
            epochs=best_epoch,
        )
        prediction, score = _predict_trials(refit, outer.x_test, device)
        truth = recording.y[outer_test]
        y_true_all.append(truth)
        y_pred_all.append(prediction)
        y_score_all.append(score)
        reports.append(
            {
                **split_report,
                "outer_train_sha256": _index_hash(outer_train),
                "outer_test_sha256": _index_hash(outer_test),
                "inner_train_sha256": _index_hash(inner_train),
                "inner_validation_sha256": _index_hash(inner_validation),
                "selected_epoch": best_epoch,
                "selection_history": history,
                "trainable_parameters": trainable_parameter_names(refit),
                "n_trainable_parameters": int(
                    sum(
                        parameter.numel()
                        for parameter in refit.parameters()
                        if parameter.requires_grad
                    )
                ),
                "preprocessing": outer.report,
            }
        )
    truth = np.concatenate(y_true_all)
    prediction = np.concatenate(y_pred_all)
    score = np.concatenate(y_score_all)
    environment_sha256, environment_versions = _environment_identity()
    payload = {
        "schema_version": 1,
        "profile": TRANSFER_PROFILE_VERSION,
        "dataset": "Souza2023",
        "source_dataset": "MI-OpenBCI" if arm != "scratch" else None,
        "source_task": "motor_imagery_vs_rest" if arm != "scratch" else None,
        "target_task": "left_hand_vs_right_hand",
        "subject": subject,
        "model": model,
        "seed": seed,
        "arm": arm,
        "protocol": protocol,
        "condition": "overlap",
        "channels": list(MI_CHANNELS),
        "window_policy": {
            "name": "six_overlap_complete_trial",
            "window_samples": 256,
            "source_trial_samples": int(recording.x.shape[-1]),
            "starts": list(_window_starts(int(recording.x.shape[-1]), "overlap")),
            "test_aggregation": "mean_class_1_probability_by_trial",
        },
        "head_policy": "fresh_binary_head",
        "target_data_sha256": recording.data_sha256,
        "source_state_sha256": _state_sha256(source_state) if source_state is not None else None,
        "scientific_code_sha256": _code_fingerprint(),
        "git_revision": _git_revision(),
        "environment_sha256": environment_sha256,
        "environment_versions": environment_versions,
        "metrics": compute_metrics(truth, prediction, score),
        "y_true": truth.tolist(),
        "y_pred": prediction.tolist(),
        "y_score": score.tolist(),
        "fold_reports": reports,
    }
    destination = (
        output_dir
        / "transfer_cells"
        / (f"Souza2023__transfer__{protocol}__overlap__{arm}__{model}__seed{seed}__{subject}.json")
    )
    write_json_atomic(destination, payload)
    return destination
