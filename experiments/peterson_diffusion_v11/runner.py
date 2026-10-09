"""Resumable execution of Peterson V11 wave cells."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from deepbench.config import is_classical_model
from deepbench.datasets import load_subject
from deepbench.evaluation import run_subject_cell
from deepbench.metrics import compute_metrics
from deepbench.preprocessing import (
    aggregate_window_scores,
    augment_training,
    prepare_test_windows,
    preprocess_split,
)

from .config import ExperimentConfig
from .environment import runtime_identity
from .identity import canonical_sha256, code_sha256, git_identity
from .io import write_json_atomic, write_torch_atomic
from .manifests import CellSpec, build_manifest, manifest_payload, shard_cells
from .splits import train_validation_indices, within_session_splits
from .training import predict_probabilities, train_model


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_fingerprint(
    config: ExperimentConfig,
    cell: CellSpec,
    data_sha256: str,
    runtime: dict[str, object],
) -> str:
    return canonical_sha256(
        {
            "config_sha256": config.sha256,
            "cell": cell.to_dict(),
            "data_sha256": data_sha256,
            "code_sha256": code_sha256(_root()),
            "environment_sha256": runtime["environment_sha256"],
            "device_type": runtime["device_type"],
            "hardware": runtime["hardware"],
        }
    )


def cell_is_reusable(
    path: Path, cell: CellSpec, config_sha256: str, run_fingerprint: str
) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("cell_id") == cell.cell_id
        and payload.get("config_sha256") == config_sha256
        and payload.get("run_fingerprint") == run_fingerprint
    )


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    metrics = compute_metrics(y_true, y_pred, y_score)
    clipped = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    metrics["nll"] = float(
        -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))
    )
    metrics["brier"] = float(np.mean((clipped - y_true) ** 2))
    return metrics


def _run_wave0_cell(
    cell: CellSpec,
    recording,
    *,
    config: ExperimentConfig,
    device: str,
    output_dir: Path,
) -> dict[str, object]:
    result = run_subject_cell(
        recording,
        cell.model,
        protocol="within_session",
        condition="center_x6",
        device="cpu" if is_classical_model(cell.model) else device,
        seed=cell.seed,
        epochs=config.epochs,
        ica_policy="none",
        checkpoint_dir=None,
        fold_cache_dir=output_dir / "fold_cache" / cell.cell_id,
    )
    payload = asdict(result)
    updates = 0
    for fold in result.fold_reports:
        history = fold.get("training_history", [])
        if history:
            updates += sum(int(epoch.get("train_batch_count", 0)) for epoch in history)
    payload["training_updates"] = updates
    return payload


def _run_wave1_cell(
    cell: CellSpec,
    recording,
    *,
    config: ExperimentConfig,
    device: str,
    output_dir: Path,
) -> dict[str, object]:
    y_true_all: list[np.ndarray] = []
    y_score_all: list[np.ndarray] = []
    fold_reports: list[dict[str, object]] = []
    total_updates = 0
    for split in within_session_splits(recording, seed=config.split_seed):
        outer_train_y = recording.y[split.train_indices]
        inner_train_local, validation_local = train_validation_indices(
            outer_train_y, seed=config.split_seed * 100 + split.fold
        )
        inner_train_indices = split.train_indices[inner_train_local]
        validation_indices = split.train_indices[validation_local]
        combined_evaluation = np.concatenate((validation_indices, split.test_indices))
        processed = preprocess_split(
            recording.x[inner_train_indices],
            recording.x[combined_evaluation],
            sfreq=recording.sfreq,
            ch_names=recording.ch_names,
            seed=config.split_seed * 1000 + split.fold,
            ica_policy="none",
            loader_bandpass_hz=recording.loader_bandpass_hz,
        )
        validation_count = len(validation_indices)
        x_validation_trials = processed.x_test[:validation_count]
        x_test_trials = processed.x_test[validation_count:]
        x_train, y_train = augment_training(
            processed.x_train, recording.y[inner_train_indices], cell.condition
        )
        x_validation, validation_trial_indices = prepare_test_windows(
            x_validation_trials, cell.condition
        )
        y_validation = recording.y[validation_indices][validation_trial_indices]
        x_test, test_trial_indices = prepare_test_windows(x_test_trials, cell.condition)
        fold_seed = cell.seed * 1000 + split.fold
        outcome = train_model(
            cell.model,
            cell.formulation,
            x_train,
            y_train,
            x_validation,
            y_validation,
            config=config,
            device=device,
            seed=fold_seed,
        )
        window_scores = predict_probabilities(
            outcome.model,
            x_test,
            formulation=cell.formulation,
            config=config,
            device=device,
            seed=fold_seed + 50_000,
        )
        _, trial_scores, windows_per_trial = aggregate_window_scores(
            window_scores, test_trial_indices, n_trials=len(split.test_indices)
        )
        y_true_all.append(recording.y[split.test_indices])
        y_score_all.append(trial_scores)
        total_updates += outcome.updates
        checkpoint_path = output_dir / "checkpoints" / f"{cell.cell_id}__fold{split.fold}.pt"
        write_torch_atomic(
            checkpoint_path,
            {
                "state_dict": outcome.model.state_dict(),
                "cell": cell.to_dict(),
                "config": config.to_dict(),
                "config_sha256": config.sha256,
                "split_sha256": split.sha256,
                "fold": split.fold,
                "n_chans": int(recording.x.shape[1]),
                "best_epoch": outcome.best_epoch,
            },
        )
        fold_reports.append(
            {
                "session": split.session,
                "fold": split.fold,
                "split_sha256": split.sha256,
                "inner_train_indices": inner_train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
                "test_indices": split.test_indices.tolist(),
                "n_train_trials": len(inner_train_indices),
                "n_validation_trials": len(validation_indices),
                "n_test_trials": len(split.test_indices),
                "n_train_examples": len(x_train),
                "n_validation_windows": len(x_validation),
                "n_test_windows": len(x_test),
                "test_windows_per_trial": windows_per_trial,
                "best_epoch": outcome.best_epoch,
                "updates": outcome.updates,
                "training_history": outcome.history,
                "preprocessing": processed.report,
                "checkpoint": str(checkpoint_path.relative_to(output_dir)),
            }
        )
        del outcome
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()
    y_true = np.concatenate(y_true_all).astype(np.int64)
    y_score = np.concatenate(y_score_all).astype(float)
    y_pred = (y_score >= 0.5).astype(np.int64)
    if not np.isfinite(y_score).all() or len(np.unique(y_pred)) < 2:
        raise RuntimeError(f"Prediction collapse or non-finite scores in {cell.cell_id}")
    return {
        "dataset": recording.dataset,
        "task": recording.task,
        "protocol": "within_session",
        "condition": cell.condition,
        "model": cell.model,
        "formulation": cell.formulation,
        "objective": cell.objective,
        "subject": cell.subject,
        "seed": cell.seed,
        "metrics": _classification_metrics(y_true, y_pred, y_score),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_score": y_score.tolist(),
        "n_test_trials": len(y_true),
        "training_updates": total_updates,
        "fold_reports": fold_reports,
    }


def run_plan(
    config: ExperimentConfig,
    *,
    output_dir: Path,
    device: str,
    num_shards: int = 1,
    shard_index: int = 0,
    max_cells_per_shard: int | None = None,
    plan_only: bool = False,
) -> dict[str, int]:
    all_cells = build_manifest(config)
    selected_cells = shard_cells(all_cells, num_shards=num_shards, shard_index=shard_index)
    if max_cells_per_shard is not None:
        if max_cells_per_shard < 1:
            raise ValueError("max_cells_per_shard must be at least 1")
        selected_cells = selected_cells[:max_cells_per_shard]
    manifest = manifest_payload(config, all_cells)
    manifest["code_sha256"] = code_sha256(_root())
    manifest["git"] = git_identity(_root())
    write_json_atomic(output_dir / "manifests" / "expected.json", manifest)
    if max_cells_per_shard is not None:
        smoke_cells = [
            cell
            for index in range(num_shards)
            for cell in shard_cells(all_cells, num_shards=num_shards, shard_index=index)[
                :max_cells_per_shard
            ]
        ]
        smoke_manifest = manifest_payload(config, smoke_cells)
        smoke_manifest["code_sha256"] = manifest["code_sha256"]
        smoke_manifest["git"] = manifest["git"]
        write_json_atomic(
            output_dir / "manifests" / "smoke_expected.json", smoke_manifest
        )
    report = {"planned_total": len(all_cells), "planned_shard": len(selected_cells)}
    if plan_only:
        return report
    completed = 0
    skipped = 0
    recordings: dict[str, object] = {}
    for cell in selected_cells:
        if cell.subject not in recordings:
            recordings[cell.subject] = load_subject("MI-OpenBCI", cell.subject)
        recording = recordings[cell.subject]
        if recording.data_sha256 is None:
            raise ValueError(f"Peterson/{cell.subject} is missing a data fingerprint")
        effective_device = "cpu" if config.wave == 0 and is_classical_model(cell.model) else device
        runtime = runtime_identity(effective_device)
        fingerprint = _run_fingerprint(config, cell, recording.data_sha256, runtime)
        destination = output_dir / "cells" / f"{cell.cell_id}.json"
        if cell_is_reusable(destination, cell, config.sha256, fingerprint):
            skipped += 1
            continue
        if config.wave == 0:
            payload = _run_wave0_cell(
                cell, recording, config=config, device=device, output_dir=output_dir
            )
        elif config.wave == 1:
            payload = _run_wave1_cell(
                cell, recording, config=config, device=device, output_dir=output_dir
            )
        else:
            raise ValueError(f"Wave {config.wave} is not implemented without a signed decision")
        payload.update(
            {
                "status": "completed",
                "cell_id": cell.cell_id,
                "cell_spec": cell.to_dict(),
                "config_sha256": config.sha256,
                "run_fingerprint": fingerprint,
                "code_sha256": code_sha256(_root()),
                "data_sha256": recording.data_sha256,
                "runtime": runtime,
                "git": git_identity(_root()),
            }
        )
        if not math.isfinite(float(payload["metrics"]["accuracy"])):
            raise RuntimeError(f"Non-finite accuracy in {cell.cell_id}")
        write_json_atomic(destination, payload)
        completed += 1
        print(f"completed {destination.name}", flush=True)
    return report | {"completed": completed, "skipped": skipped}
