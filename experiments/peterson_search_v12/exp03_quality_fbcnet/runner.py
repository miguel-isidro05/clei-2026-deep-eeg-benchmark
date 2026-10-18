"""Resumable two-phase execution for Peterson exp03."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepbench.classical import make_csp_lda
from deepbench.classical import predict_trial_scores as predict_csp_trials
from deepbench.datasets import load_subject
from deepbench.metrics import compute_metrics
from deepbench.models import make_classifier, parameter_count, predict_scores, recipe_dict
from deepbench.preprocessing import (
    aggregate_window_scores,
    augment_training,
    prepare_test_windows,
    preprocess_split,
)
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.identity import (
    canonical_sha256,
    exp03_code_sha256,
    git_identity,
)
from experiments.peterson_search_v12.common.io import write_json_atomic, write_torch_atomic
from experiments.peterson_search_v12.common.runner import runtime_environment
from experiments.peterson_search_v12.common.splits import build_search_splits

from .audit import validate_result_cell
from .catalog import QualityCandidate, RecipeSpec
from .config import Exp03Config
from .manifest import Exp03Cell, Exp03Manifest
from .models import ExactFBCNet, QualityConditionedFBCNet, fit_quality_profile
from .training import CheckpointState, train_checkpoint_trajectory


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_execution_units(
    cells: tuple[Exp03Cell, ...], recipes: tuple[RecipeSpec, ...]
) -> tuple[tuple[Exp03Cell, ...], ...]:
    recipe_by_id = {recipe.recipe_id: recipe for recipe in recipes}
    grouped: dict[tuple[object, ...], list[Exp03Cell]] = {}
    for cell in cells:
        if cell.kind == "recipe":
            if cell.config_id not in recipe_by_id:
                raise ValueError(f"Unknown recipe cell: {cell.config_id}")
            recipe = recipe_by_id[cell.config_id]
            key = ("trajectory", recipe.trajectory_id, cell.subject, cell.seed)
        else:
            key = ("baseline", cell.cell_id)
        grouped.setdefault(key, []).append(cell)
    return tuple(tuple(unit) for unit in grouped.values())


def deterministic_stress_masks(
    *, n_chans: int, missing_channels: int
) -> tuple[tuple[int, ...], ...]:
    if n_chans < 2 or not 1 <= missing_channels < n_chans:
        raise ValueError("Invalid channel stress-mask dimensions")
    masks: list[tuple[int, ...]] = []
    for start in range(n_chans):
        mask = [1] * n_chans
        for offset in range(missing_channels):
            mask[(start + offset) % n_chans] = 0
        masks.append(tuple(mask))
    return tuple(masks)


def _clean_preprocessing_report(report: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(report)
    cleaned["post_preprocessing_validation_sha256"] = cleaned.pop("post_preprocessing_test_sha256")
    cleaned["n_validation_trials"] = cleaned.pop("n_test_trials")
    return cleaned


def _data_fragment(output_dir: Path, subject: str, recording: Any) -> None:
    raw_path = Path(os.environ["CLEI_DATA_DIR"]) / f"{subject}.mat"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing Peterson source file: {raw_path}")
    write_json_atomic(
        output_dir / "manifests" / "data" / f"{subject}.json",
        {
            "dataset": "MI-OpenBCI",
            "subject": subject,
            "source_name": raw_path.name,
            "source_sha256": _file_sha256(raw_path),
            "standardized_data_sha256": recording.data_sha256,
            "n_trials": int(len(recording.y)),
            "n_channels": int(recording.x.shape[1]),
            "n_times": int(recording.x.shape[2]),
            "sfreq": float(recording.sfreq),
        },
    )


def _split_fragment(output_dir: Path, phase: str, subject: str, splits: Any) -> None:
    write_json_atomic(
        output_dir / "manifests" / "splits" / f"{phase}-{subject}.json",
        {
            "phase": phase,
            "subject": subject,
            "folds": [
                {
                    "session": split.session,
                    "fold": split.fold,
                    "train_indices": list(split.train_indices),
                    "validation_indices": list(split.validation_indices),
                    "outer_split_sha256": split.outer_split_sha256,
                    "outer_test_sha256": split.outer_test_sha256,
                    "sha256": split.sha256,
                }
                for split in splits
            ],
            "split_sha256": canonical_sha256([split.sha256 for split in splits]),
        },
    )


def _fingerprint(
    manifest: Exp03Manifest,
    cell: Exp03Cell,
    *,
    code_hash: str,
    data_hash: str,
    split_hash: str,
    environment_hash: str,
    dirty: bool,
) -> dict[str, Any]:
    return {
        "cell_config_sha256": cell.config_sha256,
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "manifest_sha256": manifest.sha256,
        "parent_decision_sha256": manifest.parent_decision_sha256,
        "code_sha256": code_hash,
        "data_sha256": data_hash,
        "split_sha256": split_hash,
        "environment_sha256": environment_hash,
        "revision": manifest.revision,
        "git_dirty": dirty,
    }


def _is_resumable(path: Path, fingerprint: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_result_cell(payload)
        return payload.get("fingerprint") == fingerprint
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _checkpoint_for_policy(outcome: Any, checkpoint_policy: str) -> CheckpointState:
    if checkpoint_policy == "last_epoch":
        return outcome.last
    if checkpoint_policy == "best_val_ce":
        return outcome.best
    raise ValueError(f"Unknown checkpoint policy: {checkpoint_policy}")


def _training_seed(seed: int, split_index: int) -> int:
    return seed * 1_000 + split_index


def _predict_model(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: str,
    batch_size: int,
    channel_mask: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    target = torch.device(device)
    model.eval()
    scores: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            x_batch = torch.from_numpy(
                np.ascontiguousarray(x[start : start + batch_size], dtype=np.float32)
            ).to(target)
            if channel_mask is not None:
                mask = torch.tensor(channel_mask, dtype=x_batch.dtype, device=target)
                x_batch = x_batch * mask[None, :, None]
            if isinstance(model, QualityConditionedFBCNet):
                logits, weights = model.forward_with_gate(x_batch, dropout_seed=None)
                gates.append(weights.cpu().numpy())
            else:
                logits = model(x_batch, dropout_seed=None)
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    gate_values = np.concatenate(gates) if gates else None
    return np.concatenate(scores), gate_values


def _synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _profile_clean_inference(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: str,
    batch_size: int,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    if n_trials < 1:
        raise ValueError("Inference profiling requires at least one trial")
    warmup = x[: min(len(x), batch_size)]
    _predict_model(model, warmup, device=device, batch_size=batch_size)
    _synchronize_device(device)
    started = time.perf_counter()
    scores, gates = _predict_model(model, x, device=device, batch_size=batch_size)
    _synchronize_device(device)
    seconds_per_trial = (time.perf_counter() - started) / n_trials
    return scores, gates, float(seconds_per_trial)


def _base_payload(
    *,
    phase: str,
    kind: str,
    cell: Exp03Cell,
    fingerprint: dict[str, Any],
    metrics: dict[str, float],
    labels: list[int],
    predictions: list[int],
    scores: list[float],
    flags: set[str],
    reports: list[dict[str, Any]],
    parameters: dict[str, int],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "status": "completed",
        "experiment": "exp03",
        "phase": phase,
        "kind": kind,
        "cell_id": cell.cell_id,
        "model_id": cell.model_id,
        "config_id": cell.config_id,
        "subject": cell.subject,
        "seed": cell.seed,
        "folds": list(cell.folds),
        "validation_metrics": metrics,
        "validation_y_true": labels,
        "validation_y_pred": predictions,
        "validation_y_score": scores,
        "quality_flags": sorted(flags),
        "fold_reports": reports,
        "parameters": parameters,
        "runtime": runtime,
        "fingerprint": fingerprint,
    }


def _recipe_unit_results(
    output_dir: Path,
    config: Exp03Config,
    recipes: dict[str, RecipeSpec],
    unit: tuple[Exp03Cell, ...],
    *,
    device: str,
    fingerprints: dict[str, dict[str, Any]],
    recording: Any,
    splits: Any,
) -> dict[str, dict[str, Any]]:
    started = time.perf_counter()
    recipe_specs = [recipes[str(cell.config_id)] for cell in unit]
    trajectory = recipe_specs[0]
    if any(recipe.trajectory_id != trajectory.trajectory_id for recipe in recipe_specs):
        raise RuntimeError("Recipe execution unit mixes multiple trajectories")
    accumulators = {
        cell.cell_id: {"labels": [], "predictions": [], "scores": [], "flags": set(), "reports": []}
        for cell in unit
    }
    parameters: dict[str, int] = {}
    for split_index, split in enumerate(splits):
        train_indices = np.asarray(split.train_indices, dtype=np.int64)
        validation_indices = np.asarray(split.validation_indices, dtype=np.int64)
        prepared = preprocess_split(
            recording.x[train_indices],
            recording.x[validation_indices],
            sfreq=recording.sfreq,
            ch_names=recording.ch_names,
            seed=unit[0].seed + split.fold,
            ica_policy="none",
            loader_bandpass_hz=recording.loader_bandpass_hz,
        )
        x_train, y_train = augment_training(
            prepared.x_train, recording.y[train_indices], config.condition
        )
        x_validation, trial_indices = prepare_test_windows(prepared.x_test, config.condition)
        window_y = recording.y[validation_indices][trial_indices]
        outcome = train_checkpoint_trajectory(
            partial(
                ExactFBCNet,
                n_chans=x_train.shape[1],
                n_times=x_train.shape[-1],
                sfreq=recording.sfreq,
            ),
            x_train,
            y_train,
            x_validation,
            window_y,
            device=device,
            seed=_training_seed(unit[0].seed, split_index),
            epochs=trajectory.epochs,
            batch_size=trajectory.batch_size,
            learning_rate=trajectory.learning_rate,
            weight_decay=trajectory.weight_decay,
            patience=trajectory.patience,
        )
        parameters = outcome.model.parameter_report()
        write_torch_atomic(
            output_dir
            / "checkpoints"
            / "recipe"
            / (
                f"{trajectory.trajectory_id}__seed{unit[0].seed}__{unit[0].subject}"
                f"__session-{split.session}__fold{split.fold}.pt"
            ),
            {
                "schema_version": 3,
                "trajectory_id": trajectory.trajectory_id,
                "trajectory_sha256": outcome.last.trajectory_sha256,
                "last_epoch": outcome.last.epoch,
                "best_epoch": outcome.best.epoch,
                "last_state_dict": outcome.last.state_dict,
                "best_state_dict": outcome.best.state_dict,
            },
        )
        for cell, recipe in zip(unit, recipe_specs, strict=True):
            checkpoint = _checkpoint_for_policy(outcome, recipe.checkpoint_policy)
            outcome.model.load_state_dict(checkpoint.state_dict)
            window_scores, _ = _predict_model(
                outcome.model,
                x_validation,
                device=device,
                batch_size=config.batch_size,
            )
            fold_pred, fold_score, windows_per_trial = aggregate_window_scores(
                window_scores, trial_indices, n_trials=len(validation_indices)
            )
            fold_y = recording.y[validation_indices]
            flags = validation_quality_flags(fold_y.tolist(), fold_pred.tolist())
            bucket = accumulators[cell.cell_id]
            bucket["labels"].extend(int(value) for value in fold_y)
            bucket["predictions"].extend(int(value) for value in fold_pred)
            bucket["scores"].extend(float(value) for value in fold_score)
            bucket["flags"].update(flags)
            bucket["reports"].append(
                {
                    "session": split.session,
                    "fold": split.fold,
                    "search_split_sha256": split.sha256,
                    "outer_split_sha256": split.outer_split_sha256,
                    "outer_test_sha256": split.outer_test_sha256,
                    "n_train_trials": len(train_indices),
                    "n_validation_trials": len(validation_indices),
                    "n_train_windows": len(x_train),
                    "n_validation_windows": len(x_validation),
                    "windows_per_validation_trial": windows_per_trial,
                    "checkpoint_policy": recipe.checkpoint_policy,
                    "selected_epoch": checkpoint.epoch,
                    "best_epoch": outcome.best.epoch,
                    "last_epoch": outcome.last.epoch,
                    "trajectory_sha256": checkpoint.trajectory_sha256,
                    "history": outcome.history,
                    "quality_flags": flags,
                    "preprocessing": _clean_preprocessing_report(prepared.report),
                }
            )
    results: dict[str, dict[str, Any]] = {}
    for cell, recipe in zip(unit, recipe_specs, strict=True):
        bucket = accumulators[cell.cell_id]
        labels = bucket["labels"]
        predictions = bucket["predictions"]
        scores = bucket["scores"]
        payload = _base_payload(
            phase="recipe",
            kind="recipe",
            cell=cell,
            fingerprint=fingerprints[cell.cell_id],
            metrics=compute_metrics(
                np.asarray(labels), np.asarray(predictions), np.asarray(scores)
            ),
            labels=labels,
            predictions=predictions,
            scores=scores,
            flags=bucket["flags"],
            reports=bucket["reports"],
            parameters=parameters,
            runtime={
                "elapsed_seconds_shared_trajectory": time.perf_counter() - started,
                "environment": runtime_environment(device),
            },
        )
        payload["recipe"] = asdict(recipe)
        validate_result_cell(payload)
        results[cell.cell_id] = payload
    return results


def _baseline_result(
    output_dir: Path,
    config: Exp03Config,
    cell: Exp03Cell,
    *,
    device: str,
    fingerprint: dict[str, Any],
    recording: Any,
    splits: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    reports: list[dict[str, Any]] = []
    flags_all: set[str] = set()
    parameters = {"total": 0, "active": 0, "auxiliary": 0}
    for split_index, split in enumerate(splits):
        train_indices = np.asarray(split.train_indices, dtype=np.int64)
        validation_indices = np.asarray(split.validation_indices, dtype=np.int64)
        prepared = preprocess_split(
            recording.x[train_indices],
            recording.x[validation_indices],
            sfreq=recording.sfreq,
            ch_names=recording.ch_names,
            seed=cell.seed + split.fold,
            ica_policy="none",
            loader_bandpass_hz=recording.loader_bandpass_hz,
        )
        x_train, y_train = augment_training(
            prepared.x_train, recording.y[train_indices], config.condition
        )
        if cell.model_id == "CSP+LDA":
            classifier = make_csp_lda()
            classifier.fit(x_train, y_train)
            fold_pred, fold_score, windows_per_trial = predict_csp_trials(
                classifier, prepared.x_test, config.condition
            )
            history: list[dict[str, Any]] = []
            recipe = recipe_dict("CSP+LDA")
        else:
            classifier = make_classifier(
                "FBCNet",
                n_chans=x_train.shape[1],
                n_times=x_train.shape[-1],
                sfreq=recording.sfreq,
                device=device,
                seed=_training_seed(cell.seed, split_index),
                epochs=config.recipe_epochs,
            )
            classifier.fit(x_train, y_train)
            x_validation, trial_indices = prepare_test_windows(prepared.x_test, config.condition)
            _, window_scores = predict_scores(classifier, x_validation)
            fold_pred, fold_score, windows_per_trial = aggregate_window_scores(
                window_scores, trial_indices, n_trials=len(validation_indices)
            )
            count = parameter_count("FBCNet", x_train.shape[1], x_train.shape[-1], recording.sfreq)
            parameters = {"total": count, "active": count, "auxiliary": 0}
            history = [
                {"epoch": index + 1, "train_loss": float(row["train_loss"])}
                for index, row in enumerate(classifier.history)
            ]
            recipe = recipe_dict("FBCNet", config.recipe_epochs)
            write_torch_atomic(
                output_dir
                / "checkpoints"
                / "recipe"
                / f"{cell.cell_id}__session-{split.session}__fold{split.fold}.pt",
                {
                    "schema_version": 3,
                    "model": "FBCNet-V10",
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in classifier.module_.state_dict().items()
                    },
                },
            )
        fold_y = recording.y[validation_indices]
        flags = validation_quality_flags(fold_y.tolist(), fold_pred.tolist())
        flags_all.update(flags)
        labels.extend(int(value) for value in fold_y)
        predictions.extend(int(value) for value in fold_pred)
        scores.extend(float(value) for value in fold_score)
        reports.append(
            {
                "session": split.session,
                "fold": split.fold,
                "search_split_sha256": split.sha256,
                "outer_split_sha256": split.outer_split_sha256,
                "outer_test_sha256": split.outer_test_sha256,
                "n_train_trials": len(train_indices),
                "n_validation_trials": len(validation_indices),
                "n_train_windows": len(x_train),
                "windows_per_validation_trial": windows_per_trial,
                "history": history,
                "quality_flags": flags,
                "preprocessing": _clean_preprocessing_report(prepared.report),
            }
        )
    payload = _base_payload(
        phase="recipe",
        kind="baseline",
        cell=cell,
        fingerprint=fingerprint,
        metrics=compute_metrics(np.asarray(labels), np.asarray(predictions), np.asarray(scores)),
        labels=labels,
        predictions=predictions,
        scores=scores,
        flags=flags_all,
        reports=reports,
        parameters=parameters,
        runtime={
            "elapsed_seconds": time.perf_counter() - started,
            "environment": runtime_environment(device),
        },
    )
    payload["recipe"] = recipe
    validate_result_cell(payload)
    return payload


def _quality_result(
    output_dir: Path,
    config: Exp03Config,
    candidate: QualityCandidate,
    selected_recipe: RecipeSpec,
    cell: Exp03Cell,
    *,
    device: str,
    fingerprint: dict[str, Any],
    recording: Any,
    splits: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    reports: list[dict[str, Any]] = []
    flags_all: set[str] = set()
    stress: dict[int, dict[str, list[float] | list[int]]] = {
        missing: {"y_true": [], "y_pred": [], "y_score": []}
        for missing in config.quality_stress_masks
    }
    parameters: dict[str, int] = {}
    for split_index, split in enumerate(splits):
        train_indices = np.asarray(split.train_indices, dtype=np.int64)
        validation_indices = np.asarray(split.validation_indices, dtype=np.int64)
        prepared = preprocess_split(
            recording.x[train_indices],
            recording.x[validation_indices],
            sfreq=recording.sfreq,
            ch_names=recording.ch_names,
            seed=cell.seed + split.fold,
            ica_policy="none",
            loader_bandpass_hz=recording.loader_bandpass_hz,
        )
        x_train, y_train = augment_training(
            prepared.x_train, recording.y[train_indices], config.condition
        )
        x_validation, trial_indices = prepare_test_windows(prepared.x_test, config.condition)
        window_y = recording.y[validation_indices][trial_indices]
        profile = fit_quality_profile(x_train)
        outcome = train_checkpoint_trajectory(
            partial(
                QualityConditionedFBCNet,
                candidate,
                profile=profile,
                n_chans=x_train.shape[1],
                n_times=x_train.shape[-1],
                sfreq=recording.sfreq,
                gate_hidden=config.quality_gate_hidden,
            ),
            x_train,
            y_train,
            x_validation,
            window_y,
            device=device,
            seed=_training_seed(cell.seed, split_index),
            epochs=selected_recipe.epochs,
            batch_size=selected_recipe.batch_size,
            learning_rate=selected_recipe.learning_rate,
            weight_decay=selected_recipe.weight_decay,
            patience=selected_recipe.patience,
        )
        checkpoint = _checkpoint_for_policy(outcome, selected_recipe.checkpoint_policy)
        outcome.model.load_state_dict(checkpoint.state_dict)
        parameters = outcome.model.parameter_report()
        window_scores, gate_values, inference_seconds_per_trial = _profile_clean_inference(
            outcome.model,
            x_validation,
            device=device,
            batch_size=config.batch_size,
            n_trials=len(validation_indices),
        )
        fold_pred, fold_score, windows_per_trial = aggregate_window_scores(
            window_scores, trial_indices, n_trials=len(validation_indices)
        )
        fold_y = recording.y[validation_indices]
        flags = validation_quality_flags(fold_y.tolist(), fold_pred.tolist())
        flags_all.update(flags)
        labels.extend(int(value) for value in fold_y)
        predictions.extend(int(value) for value in fold_pred)
        scores.extend(float(value) for value in fold_score)
        fold_stress: dict[str, dict[str, Any]] = {}
        for missing in config.quality_stress_masks:
            repeated_y: list[int] = []
            repeated_pred: list[int] = []
            repeated_score: list[float] = []
            for mask in deterministic_stress_masks(
                n_chans=x_validation.shape[1], missing_channels=missing
            ):
                masked_scores, _ = _predict_model(
                    outcome.model,
                    x_validation,
                    device=device,
                    batch_size=config.batch_size,
                    channel_mask=mask,
                )
                mask_pred, mask_score, _ = aggregate_window_scores(
                    masked_scores, trial_indices, n_trials=len(validation_indices)
                )
                repeated_y.extend(int(value) for value in fold_y)
                repeated_pred.extend(int(value) for value in mask_pred)
                repeated_score.extend(float(value) for value in mask_score)
            stress[missing]["y_true"].extend(repeated_y)
            stress[missing]["y_pred"].extend(repeated_pred)
            stress[missing]["y_score"].extend(repeated_score)
            fold_stress[f"missing_{missing}"] = {
                "mask_count": x_validation.shape[1],
                "metrics": compute_metrics(
                    np.asarray(repeated_y), np.asarray(repeated_pred), np.asarray(repeated_score)
                ),
            }
        reports.append(
            {
                "session": split.session,
                "fold": split.fold,
                "search_split_sha256": split.sha256,
                "outer_split_sha256": split.outer_split_sha256,
                "outer_test_sha256": split.outer_test_sha256,
                "n_train_trials": len(train_indices),
                "n_validation_trials": len(validation_indices),
                "n_train_windows": len(x_train),
                "n_validation_windows": len(x_validation),
                "windows_per_validation_trial": windows_per_trial,
                "clean_inference_seconds_per_trial": inference_seconds_per_trial,
                "selected_epoch": checkpoint.epoch,
                "best_epoch": outcome.best.epoch,
                "last_epoch": outcome.last.epoch,
                "trajectory_sha256": checkpoint.trajectory_sha256,
                "history": outcome.history,
                "quality_profile": profile.payload(),
                "mean_channel_gate": (
                    gate_values.mean(axis=0).tolist()
                    if gate_values is not None
                    else [1.0] * x_validation.shape[1]
                ),
                "stress": fold_stress,
                "quality_flags": flags,
                "preprocessing": _clean_preprocessing_report(prepared.report),
            }
        )
        write_torch_atomic(
            output_dir
            / "checkpoints"
            / "quality"
            / f"{cell.cell_id}__session-{split.session}__fold{split.fold}.pt",
            {
                "schema_version": 3,
                "candidate": asdict(candidate),
                "selected_recipe": asdict(selected_recipe),
                "profile": profile.payload(),
                "selected_epoch": checkpoint.epoch,
                "trajectory_sha256": checkpoint.trajectory_sha256,
                "state_dict": checkpoint.state_dict,
            },
        )
    payload = _base_payload(
        phase="quality",
        kind="quality",
        cell=cell,
        fingerprint=fingerprint,
        metrics=compute_metrics(np.asarray(labels), np.asarray(predictions), np.asarray(scores)),
        labels=labels,
        predictions=predictions,
        scores=scores,
        flags=flags_all,
        reports=reports,
        parameters=parameters,
        runtime={
            "elapsed_seconds": time.perf_counter() - started,
            "environment": runtime_environment(device),
        },
    )
    payload["candidate"] = asdict(candidate)
    payload["selected_recipe"] = asdict(selected_recipe)
    payload["stress_results"] = {
        f"missing_{missing}": {
            "metrics": compute_metrics(
                np.asarray(values["y_true"]),
                np.asarray(values["y_pred"]),
                np.asarray(values["y_score"]),
            ),
            "n_masks_per_fold": recording.x.shape[1],
        }
        for missing, values in stress.items()
    }
    validate_result_cell(payload)
    return payload


def _run_context(
    output_dir: Path,
    config: Exp03Config,
    manifest: Exp03Manifest,
    *,
    device: str,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    root = _root()
    git = git_identity(root)
    if git["dirty"] is not False:
        raise RuntimeError("Exp03 execution requires a clean Git tree")
    if str(git["revision"] or "unknown") != manifest.revision:
        raise RuntimeError("Current Git revision differs from the signed exp03 manifest")
    code_hash = exp03_code_sha256(root)
    environment = runtime_environment(device)
    scientific_environment = {
        key: environment[key] for key in ("python", "platform", "packages")
    }
    environment_hash = canonical_sha256(scientific_environment)
    label = device.replace(":", "_")
    write_json_atomic(
        output_dir / "manifests" / f"environment-{manifest.phase}-{label}.json", environment
    )
    return git, code_hash, environment, environment_hash


def run_recipe_phase(
    output_dir: Path,
    config: Exp03Config,
    recipes: tuple[RecipeSpec, ...],
    manifest: Exp03Manifest,
    *,
    device: str,
    num_shards: int,
    shard_index: int,
) -> dict[str, int]:
    git, code_hash, environment, environment_hash = _run_context(
        output_dir, config, manifest, device=device
    )
    units = recipe_execution_units(manifest.cells, recipes)
    selected_units = tuple(
        unit for index, unit in enumerate(units) if index % num_shards == shard_index
    )
    recipe_by_id = {recipe.recipe_id: recipe for recipe in recipes}
    recording_cache: dict[str, Any] = {}
    split_cache: dict[str, Any] = {}
    completed = failed = planned = 0
    for unit in selected_units:
        cell = unit[0]
        if cell.subject not in recording_cache:
            recording = load_subject("MI-OpenBCI", cell.subject)
            splits = build_search_splits(
                recording, folds=config.recipe_folds, split_seed=config.split_seed
            )
            recording_cache[cell.subject] = recording
            split_cache[cell.subject] = splits
            _data_fragment(output_dir, cell.subject, recording)
            _split_fragment(output_dir, "recipe", cell.subject, splits)
        recording = recording_cache[cell.subject]
        splits = split_cache[cell.subject]
        split_hash = canonical_sha256([split.sha256 for split in splits])
        fingerprints = {
            item.cell_id: _fingerprint(
                manifest,
                item,
                code_hash=code_hash,
                data_hash=recording.data_sha256,
                split_hash=split_hash,
                environment_hash=environment_hash,
                dirty=bool(git["dirty"]),
            )
            for item in unit
        }
        paths = {
            item.cell_id: output_dir / "recipe_cells" / f"{item.cell_id}.json" for item in unit
        }
        planned += len(unit)
        if all(_is_resumable(paths[item.cell_id], fingerprints[item.cell_id]) for item in unit):
            completed += len(unit)
            print(f"skipped recipe unit {cell.subject} seed={cell.seed}", flush=True)
            continue
        try:
            if cell.kind == "recipe":
                results = _recipe_unit_results(
                    output_dir,
                    config,
                    recipe_by_id,
                    unit,
                    device=device,
                    fingerprints=fingerprints,
                    recording=recording,
                    splits=splits,
                )
            else:
                results = {
                    cell.cell_id: _baseline_result(
                        output_dir,
                        config,
                        cell,
                        device=device,
                        fingerprint=fingerprints[cell.cell_id],
                        recording=recording,
                        splits=splits,
                    )
                }
            for cell_id, payload in results.items():
                write_json_atomic(paths[cell_id], payload)
                print(f"completed {paths[cell_id].name}", flush=True)
            completed += len(results)
        except Exception as error:
            failed += len(unit)
            for item in unit:
                write_json_atomic(
                    output_dir / "failures" / "recipe" / f"{item.cell_id}.json",
                    {
                        "schema_version": 3,
                        "status": "failed",
                        "experiment": "exp03",
                        "phase": "recipe",
                        "cell_id": item.cell_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "fingerprint": fingerprints[item.cell_id],
                    },
                )
            print(f"failed recipe unit {cell.cell_id}: {type(error).__name__}: {error}", flush=True)
    _ = environment
    return {"planned": planned, "completed": completed, "failed": failed}


def run_quality_phase(
    output_dir: Path,
    config: Exp03Config,
    candidates: tuple[QualityCandidate, ...],
    manifest: Exp03Manifest,
    selected_recipe: RecipeSpec,
    *,
    device: str,
    num_shards: int,
    shard_index: int,
) -> dict[str, int]:
    git, code_hash, environment, environment_hash = _run_context(
        output_dir, config, manifest, device=device
    )
    candidate_by_id = {candidate.config_id: candidate for candidate in candidates}
    selected = tuple(
        cell for index, cell in enumerate(manifest.cells) if index % num_shards == shard_index
    )
    recording_cache: dict[str, Any] = {}
    split_cache: dict[str, Any] = {}
    completed = failed = 0
    for cell in selected:
        if cell.subject not in recording_cache:
            recording = load_subject("MI-OpenBCI", cell.subject)
            splits = build_search_splits(
                recording, folds=config.quality_folds, split_seed=config.split_seed
            )
            recording_cache[cell.subject] = recording
            split_cache[cell.subject] = splits
            _data_fragment(output_dir, cell.subject, recording)
            _split_fragment(output_dir, "quality", cell.subject, splits)
        recording = recording_cache[cell.subject]
        splits = split_cache[cell.subject]
        fingerprint = _fingerprint(
            manifest,
            cell,
            code_hash=code_hash,
            data_hash=recording.data_sha256,
            split_hash=canonical_sha256([split.sha256 for split in splits]),
            environment_hash=environment_hash,
            dirty=bool(git["dirty"]),
        )
        path = output_dir / "quality_cells" / f"{cell.cell_id}.json"
        if _is_resumable(path, fingerprint):
            completed += 1
            print(f"skipped {path.name}", flush=True)
            continue
        try:
            if cell.config_id is None:
                raise RuntimeError("Quality cell is missing a config ID")
            payload = _quality_result(
                output_dir,
                config,
                candidate_by_id[cell.config_id],
                selected_recipe,
                cell,
                device=device,
                fingerprint=fingerprint,
                recording=recording,
                splits=splits,
            )
            write_json_atomic(path, payload)
            completed += 1
            print(f"completed {path.name}", flush=True)
        except Exception as error:
            failed += 1
            write_json_atomic(
                output_dir / "failures" / "quality" / f"{cell.cell_id}.json",
                {
                    "schema_version": 3,
                    "status": "failed",
                    "experiment": "exp03",
                    "phase": "quality",
                    "cell_id": cell.cell_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "fingerprint": fingerprint,
                },
            )
            print(f"failed {cell.cell_id}: {type(error).__name__}: {error}", flush=True)
    _ = environment
    return {"planned": len(selected), "completed": completed, "failed": failed}
