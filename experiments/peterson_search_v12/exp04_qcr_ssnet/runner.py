"""Resumable discovery-only runner for Exp04 QCR-SSNet."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepbench.classical import make_csp_lda
from deepbench.classical import predict_trial_scores as predict_csp_trials
from deepbench.datasets import load_subject
from deepbench.metrics import compute_metrics
from deepbench.models import make_classifier, parameter_count, recipe_dict
from deepbench.preprocessing import (
    aggregate_window_scores,
    augment_training,
    prepare_test_windows,
    preprocess_split,
)
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.identity import (
    canonical_sha256,
    exp04_code_sha256,
    git_identity,
)
from experiments.peterson_search_v12.common.io import write_json_atomic, write_torch_atomic
from experiments.peterson_search_v12.common.runner import runtime_environment
from experiments.peterson_search_v12.common.splits import build_search_splits
from experiments.peterson_search_v12.exp03_quality_fbcnet.models import fit_quality_profile

from .audit import validate_candidate_checkpoints, validate_result_cell
from .catalog import QCRCandidate
from .config import Exp04Config
from .manifest import Exp04Cell, Exp04Manifest, shard_cells
from .models import QCRSSNet
from .source import Exp04Source
from .training import train_qcr


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_stress_masks(
    *, n_chans: int, missing_channels: int
) -> tuple[tuple[int, ...], ...]:
    if not 1 <= missing_channels < n_chans:
        raise ValueError("Invalid channel stress mask")
    masks: list[tuple[int, ...]] = []
    for missing_indices in combinations(range(n_chans), missing_channels):
        mask = [1] * n_chans
        for index in missing_indices:
            mask[index] = 0
        masks.append(tuple(mask))
    return tuple(masks)


def _clean_preprocessing_report(report: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(report)
    cleaned["post_preprocessing_validation_sha256"] = cleaned.pop("post_preprocessing_test_sha256")
    cleaned["n_validation_trials"] = cleaned.pop("n_test_trials")
    return cleaned


def _training_seed(seed: int, split_index: int) -> int:
    return seed * 1_000 + split_index


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


def _split_fragment(output_dir: Path, subject: str, splits: Any) -> None:
    write_json_atomic(
        output_dir / "manifests" / "splits" / f"discovery-{subject}.json",
        {
            "phase": "discovery",
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
    manifest: Exp04Manifest,
    cell: Exp04Cell,
    *,
    code_hash: str,
    data_hash: str,
    split_hash: str,
    environment_hash: str,
    dirty: bool,
) -> dict[str, Any]:
    return {
        "cell_config_sha256": cell.config_sha256
        or canonical_sha256({"model_id": cell.model_id, "protocol": "within_session_overlap"}),
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "manifest_sha256": manifest.sha256,
        "source_sha256": manifest.source_sha256,
        "code_sha256": code_hash,
        "data_sha256": data_hash,
        "split_sha256": split_hash,
        "environment_sha256": environment_hash,
        "revision": manifest.revision,
        "git_dirty": dirty,
    }


def _is_resumable(path: Path, fingerprint: dict[str, Any], *, smoke: bool) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_result_cell(payload, allow_dirty=smoke)
        if payload.get("kind") == "candidate":
            validate_candidate_checkpoints(path.parent.parent, path.stem, payload)
        return payload.get("fingerprint") == fingerprint
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _predict_qcr(
    model: QCRSSNet,
    x: np.ndarray,
    *,
    device: str,
    batch_size: int,
    channel_mask: tuple[int, ...] | None = None,
    oracle_severity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    scores: list[np.ndarray] = []
    routers: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(x[start : start + batch_size], dtype=np.float32)
            ).to(device)
            if channel_mask is not None:
                mask = torch.tensor(channel_mask, dtype=batch.dtype, device=batch.device)
                batch = batch * mask[None, :, None]
            severity = torch.full(
                (len(batch),), oracle_severity, dtype=batch.dtype, device=batch.device
            )
            output = model(batch, oracle_severity=severity)
            scores.append(torch.softmax(output.logits, dim=1)[:, 1].cpu().numpy())
            routers.append(output.router_weight.cpu().numpy())
            gates.append(output.channel_weights.cpu().numpy())
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return (
        np.concatenate(scores),
        np.concatenate(routers),
        np.concatenate(gates),
        float(elapsed),
    )


def _predict_fbc(
    classifier: Any,
    x: np.ndarray,
    *,
    channel_mask: tuple[int, ...] | None = None,
) -> np.ndarray:
    values = x
    if channel_mask is not None:
        values = values * np.asarray(channel_mask, dtype=np.float32)[None, :, None]
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    probabilities = np.asarray(classifier.predict_proba(contiguous), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("FBCNet must return probabilities for exactly two classes")
    return probabilities[:, 1]


def _stress_payload(
    *,
    predict: Any,
    x_validation: np.ndarray,
    trial_indices: np.ndarray,
    fold_y: np.ndarray,
    stress_masks: tuple[int, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[Any]]]]:
    fold_results: dict[str, dict[str, Any]] = {}
    values: dict[str, dict[str, list[Any]]] = {}
    for missing in stress_masks:
        labels: list[int] = []
        predictions: list[int] = []
        scores: list[float] = []
        mask_accuracies: list[float] = []
        router_values: list[np.ndarray] = []
        gate_values: list[np.ndarray] = []
        masks = deterministic_stress_masks(n_chans=x_validation.shape[1], missing_channels=missing)
        for mask in masks:
            predicted = predict(mask, missing)
            if isinstance(predicted, tuple):
                window_scores, router, gates = predicted
                router_values.append(router)
                gate_values.append(gates)
            else:
                window_scores = predicted
            trial_pred, trial_score, _ = aggregate_window_scores(
                window_scores, trial_indices, n_trials=len(fold_y)
            )
            labels.extend(int(value) for value in fold_y)
            predictions.extend(int(value) for value in trial_pred)
            scores.extend(float(value) for value in trial_score)
            mask_accuracies.append(float(np.mean(trial_pred == fold_y)))
        name = f"missing_{missing}"
        values[name] = {
            "y_true": labels,
            "y_pred": predictions,
            "y_score": scores,
            "mask_accuracies": mask_accuracies,
        }
        fold_results[name] = {
            "mask_count": len(masks),
            "worst_mask_accuracy": min(mask_accuracies),
            "metrics": compute_metrics(
                np.asarray(labels), np.asarray(predictions), np.asarray(scores)
            ),
        }
        if router_values:
            fold_results[name]["mean_router_weight"] = float(np.mean(np.concatenate(router_values)))
            fold_results[name]["mean_channel_gate"] = (
                np.concatenate(gate_values).mean(axis=0).tolist()
            )
    return fold_results, values


def _merge_stress(
    accumulator: dict[str, dict[str, list[Any]]],
    fold_values: dict[str, dict[str, list[Any]]],
) -> None:
    for condition, values in fold_values.items():
        bucket = accumulator.setdefault(
            condition,
            {"y_true": [], "y_pred": [], "y_score": [], "mask_accuracies": []},
        )
        for key in bucket:
            bucket[key].extend(values[key])


def _final_stress(
    values: dict[str, dict[str, list[Any]]], *, n_chans: int
) -> dict[str, dict[str, Any]]:
    return {
        condition: {
            "metrics": compute_metrics(
                np.asarray(bucket["y_true"]),
                np.asarray(bucket["y_pred"]),
                np.asarray(bucket["y_score"]),
            ),
            "y_true": bucket["y_true"],
            "y_pred": bucket["y_pred"],
            "y_score": bucket["y_score"],
            "n_masks_per_fold": len(
                deterministic_stress_masks(
                    n_chans=n_chans,
                    missing_channels=int(condition.rsplit("_", 1)[1]),
                )
            ),
            "worst_mask_accuracy": min(bucket["mask_accuracies"]),
        }
        for condition, bucket in values.items()
    }


def _base_payload(
    cell: Exp04Cell,
    source: Exp04Source,
    fingerprint: dict[str, Any],
    *,
    labels: list[int],
    predictions: list[int],
    scores: list[float],
    flags: set[str],
    reports: list[dict[str, Any]],
    stress: dict[str, dict[str, Any]],
    parameters: dict[str, int],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "status": "completed",
        "experiment": "exp04",
        "phase": "discovery",
        "kind": cell.kind,
        "cell_id": cell.cell_id,
        "model_id": cell.model_id,
        "config_id": cell.config_id,
        "subject": cell.subject,
        "seed": cell.seed,
        "folds": list(cell.folds),
        "validation_metrics": compute_metrics(
            np.asarray(labels), np.asarray(predictions), np.asarray(scores)
        ),
        "validation_y_true": labels,
        "validation_y_pred": predictions,
        "validation_y_score": scores,
        "quality_flags": sorted(flags),
        "fold_reports": reports,
        "stress_results": stress,
        "parameters": parameters,
        "source": source.payload(),
        "runtime": runtime,
        "fingerprint": fingerprint,
    }


def _candidate_result(
    output_dir: Path,
    config: Exp04Config,
    candidate: QCRCandidate,
    source: Exp04Source,
    cell: Exp04Cell,
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
    flags_all: set[str] = set()
    reports: list[dict[str, Any]] = []
    stress_all: dict[str, dict[str, list[Any]]] = {}
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
        outcome = train_qcr(
            partial(
                QCRSSNet,
                candidate,
                profile=profile,
                n_chans=x_train.shape[1],
                n_times=x_train.shape[-1],
                sfreq=recording.sfreq,
                gate_hidden=config.quality_gate_hidden,
                router_hidden=config.router_hidden,
            ),
            candidate,
            x_train,
            y_train,
            x_validation,
            window_y,
            device=device,
            seed=_training_seed(cell.seed, split_index),
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            corruption_probability=candidate.channel_dropout,
            max_missing_channels=config.max_missing_channels,
        )
        # Fixed last-epoch evaluation preserves parity with the frozen FBCNet-V10
        # baseline. The best checkpoint remains diagnostic and is never used for
        # discovery metrics or promotion.
        outcome.model.load_state_dict(outcome.last.state_dict)
        parameters = outcome.model.parameter_report()
        _predict_qcr(
            outcome.model,
            x_validation[: min(len(x_validation), config.batch_size)],
            device=device,
            batch_size=config.batch_size,
        )
        window_scores, routers, gates, elapsed = _predict_qcr(
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
        flags_all.update(flags)
        labels.extend(int(value) for value in fold_y)
        predictions.extend(int(value) for value in fold_pred)
        scores.extend(float(value) for value in fold_score)
        trained_model = outcome.model
        validation_windows = x_validation
        fold_stress, fold_values = _stress_payload(
            predict=lambda mask, missing, model=trained_model, windows=validation_windows: (
                _predict_qcr(
                    model,
                    windows,
                    device=device,
                    batch_size=config.batch_size,
                    channel_mask=mask,
                    oracle_severity=missing / config.max_missing_channels,
                )[:3]
            ),
            x_validation=x_validation,
            trial_indices=trial_indices,
            fold_y=fold_y,
            stress_masks=config.stress_masks,
        )
        _merge_stress(stress_all, fold_values)
        checkpoint_path = (
            output_dir
            / "checkpoints"
            / f"{cell.cell_id}__session-{split.session}__fold{split.fold}.pt"
        )
        checkpoint_payload = {
            "schema_version": 4,
            "cell_id": cell.cell_id,
            "session": split.session,
            "fold": split.fold,
            "candidate": asdict(candidate),
            "source_sha256": source.sha256,
            "checkpoint_policy": "last_epoch",
            "selected_epoch": outcome.last.epoch,
            "best_diagnostic_epoch": outcome.best.epoch,
            "trajectory_sha256": outcome.last.trajectory_sha256,
            "state_dict": outcome.last.state_dict,
        }
        write_torch_atomic(checkpoint_path, checkpoint_payload)
        checkpoint_sha256 = _file_sha256(checkpoint_path)
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
                "checkpoint_policy": "last_epoch",
                "selected_epoch": outcome.last.epoch,
                "best_diagnostic_epoch": outcome.best.epoch,
                "last_epoch": outcome.last.epoch,
                "trajectory_sha256": outcome.last.trajectory_sha256,
                "checkpoint_file": checkpoint_path.name,
                "checkpoint_sha256": checkpoint_sha256,
                "history": outcome.history,
                "quality_profile": profile.payload(),
                "mean_router_weight": float(np.mean(routers)),
                "mean_channel_gate": gates.mean(axis=0).tolist(),
                "clean_inference_seconds_per_trial": elapsed / len(validation_indices),
                "stress": fold_stress,
                "quality_flags": flags,
                "preprocessing": _clean_preprocessing_report(prepared.report),
            }
        )
    payload = _base_payload(
        cell,
        source,
        fingerprint,
        labels=labels,
        predictions=predictions,
        scores=scores,
        flags=flags_all,
        reports=reports,
        stress=_final_stress(stress_all, n_chans=recording.x.shape[1]),
        parameters=parameters,
        runtime={
            "elapsed_seconds": time.perf_counter() - started,
            "environment": runtime_environment(device),
        },
    )
    payload["candidate"] = asdict(candidate)
    validate_result_cell(payload, allow_dirty=config.profile == "smoke")
    return payload


def _baseline_result(
    config: Exp04Config,
    source: Exp04Source,
    cell: Exp04Cell,
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
    flags_all: set[str] = set()
    reports: list[dict[str, Any]] = []
    stress_all: dict[str, dict[str, list[Any]]] = {}
    parameters = {"total": 0, "active": 0, "auxiliary": 0, "frozen": 0}
    recipe: dict[str, Any] = {}
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
        if cell.model_id == "CSP+LDA":
            classifier = make_csp_lda()
            classifier.fit(x_train, y_train)
            predict_csp_trials(classifier, prepared.x_test, config.condition)
            started_inference = time.perf_counter()
            fold_pred, fold_score, windows_per_trial = predict_csp_trials(
                classifier, prepared.x_test, config.condition
            )
            inference_seconds = (time.perf_counter() - started_inference) / len(validation_indices)
            recipe = recipe_dict("CSP+LDA")

            # CSP returns trial scores, so stress is evaluated directly below.
            fold_stress: dict[str, dict[str, Any]] = {}
            fold_values: dict[str, dict[str, list[Any]]] = {}
            fold_y = recording.y[validation_indices]
            for missing in config.stress_masks:
                ys: list[int] = []
                ps: list[int] = []
                ss: list[float] = []
                mask_accuracies: list[float] = []
                masks = deterministic_stress_masks(
                    n_chans=prepared.x_test.shape[1], missing_channels=missing
                )
                for mask in masks:
                    masked = prepared.x_test * np.asarray(mask, dtype=np.float32)[None, :, None]
                    pred, score, _ = predict_csp_trials(classifier, masked, config.condition)
                    ys.extend(int(value) for value in fold_y)
                    ps.extend(int(value) for value in pred)
                    ss.extend(float(value) for value in score)
                    mask_accuracies.append(float(np.mean(pred == fold_y)))
                name = f"missing_{missing}"
                fold_values[name] = {
                    "y_true": ys,
                    "y_pred": ps,
                    "y_score": ss,
                    "mask_accuracies": mask_accuracies,
                }
                fold_stress[name] = {
                    "mask_count": len(masks),
                    "worst_mask_accuracy": min(mask_accuracies),
                    "metrics": compute_metrics(np.asarray(ys), np.asarray(ps), np.asarray(ss)),
                }
        else:
            classifier = make_classifier(
                "FBCNet",
                n_chans=x_train.shape[1],
                n_times=x_train.shape[-1],
                sfreq=recording.sfreq,
                device=device,
                seed=_training_seed(cell.seed, split_index),
                epochs=config.epochs,
            )
            classifier.fit(x_train, y_train)
            _predict_fbc(classifier, x_validation[: min(len(x_validation), config.batch_size)])
            _synchronize(device)
            started_inference = time.perf_counter()
            window_scores = _predict_fbc(classifier, x_validation)
            _synchronize(device)
            inference_seconds = (time.perf_counter() - started_inference) / len(validation_indices)
            fold_pred, fold_score, windows_per_trial = aggregate_window_scores(
                window_scores, trial_indices, n_trials=len(validation_indices)
            )
            count = parameter_count("FBCNet", x_train.shape[1], x_train.shape[-1], recording.sfreq)
            parameters = {"total": count, "active": count, "auxiliary": 0, "frozen": 0}
            recipe = recipe_dict("FBCNet", config.epochs)
            fold_y = recording.y[validation_indices]
            fitted_classifier = classifier
            validation_windows = x_validation
            fold_stress, fold_values = _stress_payload(
                predict=lambda mask,
                _missing,
                fitted=fitted_classifier,
                windows=(validation_windows): _predict_fbc(fitted, windows, channel_mask=mask),
                x_validation=x_validation,
                trial_indices=trial_indices,
                fold_y=fold_y,
                stress_masks=config.stress_masks,
            )
        flags = validation_quality_flags(fold_y.tolist(), fold_pred.tolist())
        flags_all.update(flags)
        labels.extend(int(value) for value in fold_y)
        predictions.extend(int(value) for value in fold_pred)
        scores.extend(float(value) for value in fold_score)
        _merge_stress(stress_all, fold_values)
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
                "clean_inference_seconds_per_trial": float(inference_seconds),
                "history": [],
                "stress": fold_stress,
                "quality_flags": flags,
                "preprocessing": _clean_preprocessing_report(prepared.report),
            }
        )
    payload = _base_payload(
        cell,
        source,
        fingerprint,
        labels=labels,
        predictions=predictions,
        scores=scores,
        flags=flags_all,
        reports=reports,
        stress=_final_stress(stress_all, n_chans=recording.x.shape[1]),
        parameters=parameters,
        runtime={
            "elapsed_seconds": time.perf_counter() - started,
            "environment": runtime_environment(device),
        },
    )
    payload["recipe"] = recipe
    validate_result_cell(payload, allow_dirty=config.profile == "smoke")
    return payload


def run_discovery(
    output_dir: Path,
    config: Exp04Config,
    catalog: tuple[QCRCandidate, ...],
    source: Exp04Source,
    manifest: Exp04Manifest,
    *,
    device: str,
    num_shards: int,
    shard_index: int,
) -> dict[str, int]:
    root = _root()
    git = git_identity(root)
    if config.profile == "full" and git["dirty"] is not False:
        raise RuntimeError("Full Exp04 execution requires a clean Git tree")
    if str(git["revision"] or "unknown") != manifest.revision:
        raise RuntimeError("Current Git revision differs from the signed Exp04 manifest")
    code_hash = exp04_code_sha256(root)
    environment = runtime_environment(device)
    environment_hash = canonical_sha256(
        {key: environment[key] for key in ("python", "platform", "packages")}
    )
    write_json_atomic(
        output_dir / "manifests" / f"environment-{device.replace(':', '_')}.json",
        environment,
    )
    candidates = {candidate.config_id: candidate for candidate in catalog}
    selected = shard_cells(manifest.cells, num_shards=num_shards, shard_index=shard_index)
    recording_cache: dict[str, Any] = {}
    split_cache: dict[str, Any] = {}
    completed = failed = 0
    for cell in selected:
        if cell.subject not in recording_cache:
            recording = load_subject("MI-OpenBCI", cell.subject)
            splits = build_search_splits(
                recording, folds=config.folds, split_seed=config.split_seed
            )
            recording_cache[cell.subject] = recording
            split_cache[cell.subject] = splits
            _data_fragment(output_dir, cell.subject, recording)
            _split_fragment(output_dir, cell.subject, splits)
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
        path = output_dir / "cells" / f"{cell.cell_id}.json"
        if _is_resumable(path, fingerprint, smoke=config.profile == "smoke"):
            completed += 1
            print(f"skipped {path.name}", flush=True)
            continue
        try:
            if cell.kind == "candidate":
                candidate = candidates[str(cell.config_id)]
                payload = _candidate_result(
                    output_dir,
                    config,
                    candidate,
                    source,
                    cell,
                    device=device,
                    fingerprint=fingerprint,
                    recording=recording,
                    splits=splits,
                )
            else:
                payload = _baseline_result(
                    config,
                    source,
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
                output_dir / "failures" / f"{cell.cell_id}.json",
                {
                    "schema_version": 4,
                    "status": "failed",
                    "experiment": "exp04",
                    "cell_id": cell.cell_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "fingerprint": fingerprint,
                },
            )
            print(f"failed {cell.cell_id}: {type(error).__name__}: {error}", flush=True)
    return {"planned": len(selected), "completed": completed, "failed": failed}
