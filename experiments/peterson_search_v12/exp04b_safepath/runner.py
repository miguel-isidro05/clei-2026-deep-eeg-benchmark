"""Resumable subject-sharded execution for Exp04b."""

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
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

from deepbench.datasets import load_subject
from deepbench.metrics import compute_metrics
from deepbench.preprocessing import (
    aggregate_window_scores,
    augment_training,
    prepare_test_windows,
    preprocess_split,
)
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.identity import canonical_sha256, git_identity
from experiments.peterson_search_v12.common.io import write_json_atomic, write_torch_atomic
from experiments.peterson_search_v12.common.runner import runtime_environment
from experiments.peterson_search_v12.common.splits import build_search_splits
from experiments.peterson_search_v12.exp03_quality_fbcnet.catalog import QualityCandidate
from experiments.peterson_search_v12.exp03_quality_fbcnet.models import (
    ExactFBCNet,
    QualityConditionedFBCNet,
    fit_quality_profile,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.source import Exp04Source

from .catalog import SafePathCandidate
from .config import Exp04bConfig
from .corruptions import CorruptionSpec, apply_corruption, build_stress_suite
from .identity import exp04b_code_sha256
from .manifest import Exp04bCell, Exp04bManifest, shard_cells_by_subject
from .models import SafePathQCR
from .statistics import physical_mask_floor
from .training import load_or_train_anchor, train_safepath


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_report(report: dict[str, Any]) -> dict[str, Any]:
    value = dict(report)
    value["post_preprocessing_validation_sha256"] = value.pop("post_preprocessing_test_sha256")
    value["n_validation_trials"] = value.pop("n_test_trials")
    return value


def _quality_candidate(
    *, config_id: str, gate_id: str, dropout: float, filters: int, stride: int
) -> QualityCandidate:
    descriptors = {
        "G0": (),
        "G2": ("log_rms", "log_diff_rms", "flatline_fraction", "extreme_fraction"),
    }[gate_id]
    body = {
        "config_id": config_id,
        "gate_id": gate_id,
        "descriptor_names": descriptors,
        "channel_dropout": dropout,
        "n_filters_spat": filters,
        "stride_factor": stride,
    }
    return QualityCandidate(**body, sha256=canonical_sha256(body))


def _anchor_factory(
    candidate: SafePathCandidate,
    source: Exp04Source,
    profile: Any,
    *,
    n_chans: int,
    n_times: int,
    sfreq: float,
):
    if candidate.clean_anchor == "fbc":
        return partial(ExactFBCNet, n_chans=n_chans, n_times=n_times, sfreq=sfreq)
    clean = _quality_candidate(
        config_id=source.clean_config_id,
        gate_id="G2",
        dropout=0.0,
        filters=source.clean_n_filters_spat,
        stride=source.clean_stride_factor,
    )
    return partial(
        QualityConditionedFBCNet,
        clean,
        profile=profile,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=sfreq,
    )


def _robust_expert(
    source: Exp04Source,
    profile: Any,
    *,
    dropout: float,
    n_chans: int,
    n_times: int,
    sfreq: float,
):
    recipe = _quality_candidate(
        config_id=source.robust_config_id,
        gate_id="G0",
        dropout=dropout,
        filters=source.robust_n_filters_spat,
        stride=source.robust_stride_factor,
    )
    return QualityConditionedFBCNet(
        recipe,
        profile=profile,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=sfreq,
    )


def _predict(
    model: SafePathQCR,
    x: np.ndarray,
    *,
    device: str,
    batch_size: int,
    spec: CorruptionSpec | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    routes: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(x), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(x[offset : offset + batch_size], dtype=np.float32)
            ).to(device)
            if spec is None:
                mask = torch.ones(
                    (len(batch), batch.shape[1]),
                    dtype=batch.dtype,
                    device=batch.device,
                )
                output = model(batch, channel_mask=mask, dropout_seed=seed + offset)
            else:
                batch, mask = apply_corruption(batch, spec, seed=seed + offset)
                explicit = mask if spec.family in {"missing", "flatline"} else None
                severity = torch.ones(len(batch), dtype=batch.dtype, device=batch.device)
                if spec.family == "missing":
                    severity.fill_(min(1.0, float(spec.severity) / 2.0))
                output = model(
                    batch,
                    channel_mask=explicit,
                    corruption_score=severity,
                    dropout_seed=seed + offset,
                )
            scores.append(torch.softmax(output.logits, dim=1)[:, 1].cpu().numpy())
            routes.append(output.router_weight.cpu().numpy())
    return np.concatenate(scores), np.concatenate(routes)


def _stress_specs(n_chans: int, *, exhaustive: bool) -> tuple[CorruptionSpec, ...]:
    suite = build_stress_suite(n_chans)
    if exhaustive:
        return suite
    return tuple(item for item in suite if item.family == "missing")


def _evaluate_stress(
    model: SafePathQCR,
    x_validation: np.ndarray,
    trial_indices: np.ndarray,
    fold_y: np.ndarray,
    *,
    fold: int,
    device: str,
    batch_size: int,
    exhaustive: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: dict[str, Any] = {}
    mask_records: list[dict[str, Any]] = []
    for index, spec in enumerate(_stress_specs(x_validation.shape[1], exhaustive=exhaustive)):
        scores, routes = _predict(
            model,
            x_validation,
            device=device,
            batch_size=batch_size,
            spec=spec,
            seed=fold * 100_000 + index,
        )
        pred, trial_score, _ = aggregate_window_scores(scores, trial_indices, n_trials=len(fold_y))
        metrics = compute_metrics(fold_y, pred, trial_score)
        summaries[spec.condition_id] = {
            "family": spec.family,
            "channels": list(spec.channels),
            "severity": spec.severity,
            "metrics": metrics,
            "mean_router_weight": float(np.mean(routes)),
        }
        if spec.family == "missing":
            mask_records.append(
                {
                    "condition": f"missing_{int(spec.severity)}",
                    "mask_id": "+".join(str(value) for value in spec.channels),
                    "fold": fold,
                    "accuracy": metrics["accuracy"],
                }
            )
    return summaries, mask_records


def _candidate_result(
    output_dir: Path,
    config: Exp04bConfig,
    candidate: SafePathCandidate,
    source: Exp04Source,
    cell: Exp04bCell,
    *,
    device: str,
    recording: Any,
    splits: Any,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    reports: list[dict[str, Any]] = []
    flags_all: set[str] = set()
    mask_records: list[dict[str, Any]] = []
    secondary: dict[str, list[float]] = {}
    parameters: dict[str, int] = {}
    started = time.perf_counter()
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
        y_window_validation = recording.y[validation_indices][trial_indices]
        profile = fit_quality_profile(x_train)
        factory = _anchor_factory(
            candidate,
            source,
            profile,
            n_chans=x_train.shape[1],
            n_times=x_train.shape[-1],
            sfreq=recording.sfreq,
        )
        anchor_identity = {
            "anchor": candidate.clean_anchor,
            "subject": cell.subject,
            "seed": cell.seed,
            "fold": split.fold,
            "split_sha256": split.sha256,
            "profile_sha256": profile.sha256,
            "preprocessing_sha256": canonical_sha256(prepared.report),
            "source_sha256": source.sha256,
            "code_sha256": fingerprint["code_sha256"],
            "epochs": config.epochs,
        }
        anchor_path = (
            output_dir
            / "anchors"
            / f"{candidate.clean_anchor}__seed{cell.seed}__{cell.subject}__fold{split.fold}.pt"
        )
        clean_expert, anchor_report = load_or_train_anchor(
            anchor_path,
            factory,
            x_train,
            y_train,
            x_validation,
            y_window_validation,
            identity=anchor_identity,
            device=device,
            seed=cell.seed * 1_000 + split_index,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        robust = _robust_expert(
            source,
            profile,
            dropout=candidate.robust_dropout,
            n_chans=x_train.shape[1],
            n_times=x_train.shape[-1],
            sfreq=recording.sfreq,
        )
        model = SafePathQCR(
            clean_expert,
            robust,
            router_mode=candidate.router_mode,
            fusion_mode=candidate.fusion_mode,
            freeze_clean=candidate.training_mode != "joint",
            hard_clean_bypass=candidate.hard_clean_bypass,
        )
        if candidate.router_mode == "none":
            for parameter in model.robust_expert.parameters():
                parameter.requires_grad_(False)
        outcome = train_safepath(
            model,
            candidate,
            x_train,
            y_train,
            x_validation,
            y_window_validation,
            device=device,
            seed=cell.seed * 1_000 + split_index,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_missing=config.max_missing_channels,
        )
        parameters = outcome.model.parameter_report()
        window_scores, routes = _predict(
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
        stress, fold_masks = _evaluate_stress(
            outcome.model,
            x_validation,
            trial_indices,
            fold_y,
            fold=split.fold,
            device=device,
            batch_size=config.batch_size,
            exhaustive=cell.wave == "w3",
        )
        mask_records.extend(fold_masks)
        for condition_id, report in stress.items():
            secondary.setdefault(condition_id, []).append(float(report["metrics"]["accuracy"]))
        checkpoint_path = (
            output_dir / "checkpoints" / cell.wave / f"{cell.cell_id}__fold{split.fold}.pt"
        )
        write_torch_atomic(
            checkpoint_path,
            {
                "schema_version": 5,
                "cell_id": cell.cell_id,
                "fold": split.fold,
                "candidate": asdict(candidate),
                "source_sha256": source.sha256,
                "checkpoint_policy": "best_inner_objective",
                "selected_epoch": outcome.best.epoch,
                "trajectory_sha256": outcome.best.trajectory_sha256,
                "anchor_identity_sha256": anchor_report["identity_sha256"],
                "state_dict": outcome.best.state_dict,
            },
        )
        reports.append(
            {
                "fold": split.fold,
                "search_split_sha256": split.sha256,
                "outer_split_sha256": split.outer_split_sha256,
                "outer_test_sha256": split.outer_test_sha256,
                "n_train_trials": len(train_indices),
                "n_validation_trials": len(validation_indices),
                "n_train_windows": len(x_train),
                "n_validation_windows": len(x_validation),
                "windows_per_validation_trial": windows_per_trial,
                "checkpoint_file": str(checkpoint_path.relative_to(output_dir)),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "selected_epoch": outcome.best.epoch,
                "history": outcome.history,
                "quality_flags": flags,
                "mean_clean_router_weight": float(np.mean(routes)),
                "preprocessing": _clean_report(prepared.report),
            }
        )
    floors = {
        condition: physical_mask_floor(
            [record for record in mask_records if record["condition"] == condition]
        )
        for condition in ("missing_1", "missing_2")
    }
    return {
        "schema_version": 5,
        "status": "completed",
        "experiment": "exp04b",
        "wave": cell.wave,
        "cell_id": cell.cell_id,
        "kind": cell.kind,
        "model_id": cell.model_id,
        "config_id": cell.config_id,
        "candidate": asdict(candidate),
        "subject": cell.subject,
        "seed": cell.seed,
        "folds": list(cell.folds),
        "validation_metrics": compute_metrics(
            np.asarray(labels), np.asarray(predictions), np.asarray(scores)
        ),
        "validation_y_true": labels,
        "validation_y_pred": predictions,
        "validation_y_score": scores,
        "quality_flags": sorted(flags_all),
        "physical_mask_floor": floors,
        "stress_condition_accuracy": {
            name: float(np.mean(values)) for name, values in secondary.items()
        },
        "fold_reports": reports,
        "parameters": parameters,
        "source": source.payload(),
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "environment": runtime_environment(device),
        },
        "fingerprint": fingerprint,
    }


def _conditional_reconstruct(
    x: np.ndarray, train_covariance: np.ndarray, channels: tuple[int, ...]
) -> np.ndarray:
    if not channels:
        return np.asarray(x, dtype=np.float32)
    missing = np.asarray(channels, dtype=int)
    observed = np.asarray(
        [index for index in range(x.shape[1]) if index not in set(channels)], dtype=int
    )
    sigma_mo = train_covariance[np.ix_(missing, observed)]
    sigma_oo = train_covariance[np.ix_(observed, observed)]
    weights = sigma_mo @ np.linalg.pinv(sigma_oo, hermitian=True)
    values = np.asarray(x, dtype=np.float32).copy()
    values[:, missing] = np.einsum("mo,bot->bmt", weights, values[:, observed])
    return values


def _classical_result(
    config: Exp04bConfig,
    source: Exp04Source,
    cell: Exp04bCell,
    *,
    recording: Any,
    splits: Any,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    reports: list[dict[str, Any]] = []
    flags_all: set[str] = set()
    mask_records: list[dict[str, Any]] = []
    secondary: dict[str, list[float]] = {}
    started = time.perf_counter()
    for split in splits:
        train_indices = np.asarray(split.train_indices, dtype=np.int64)
        validation_indices = np.asarray(split.validation_indices, dtype=np.int64)
        prepared = preprocess_split(
            recording.x[train_indices],
            recording.x[validation_indices],
            sfreq=recording.sfreq,
            ch_names=recording.ch_names,
            seed=split.fold,
            ica_policy="none",
            loader_bandpass_hz=recording.loader_bandpass_hz,
        )
        x_train, y_train = augment_training(
            prepared.x_train, recording.y[train_indices], config.condition
        )
        x_validation, trial_indices = prepare_test_windows(prepared.x_test, config.condition)
        covariance = Covariances(estimator="oas")
        classifier = MDM(metric="riemann")
        classifier.fit(covariance.fit_transform(x_train), y_train)
        train_samples = np.transpose(x_train, (1, 0, 2)).reshape(x_train.shape[1], -1)
        train_covariance = np.cov(train_samples)

        def predict_windows(
            values: np.ndarray,
            *,
            fitted_classifier: MDM = classifier,
            fitted_covariance: Covariances = covariance,
        ) -> np.ndarray:
            probabilities = fitted_classifier.predict_proba(fitted_covariance.transform(values))
            return np.asarray(probabilities[:, 1], dtype=float)

        clean_scores = predict_windows(x_validation)
        fold_pred, fold_score, windows_per_trial = aggregate_window_scores(
            clean_scores, trial_indices, n_trials=len(validation_indices)
        )
        fold_y = recording.y[validation_indices]
        flags = validation_quality_flags(fold_y.tolist(), fold_pred.tolist())
        flags_all.update(flags)
        labels.extend(int(value) for value in fold_y)
        predictions.extend(int(value) for value in fold_pred)
        scores.extend(float(value) for value in fold_score)
        for index, spec in enumerate(build_stress_suite(x_validation.shape[1])):
            tensor, _ = apply_corruption(
                torch.from_numpy(np.ascontiguousarray(x_validation, dtype=np.float32)),
                spec,
                seed=split.fold * 100_000 + index,
            )
            corrupted = tensor.numpy()
            if spec.family in {"missing", "flatline"}:
                corrupted = _conditional_reconstruct(corrupted, train_covariance, spec.channels)
            stress_scores = predict_windows(corrupted)
            stress_pred, stress_trial_score, _ = aggregate_window_scores(
                stress_scores, trial_indices, n_trials=len(validation_indices)
            )
            metrics = compute_metrics(fold_y, stress_pred, stress_trial_score)
            secondary.setdefault(spec.condition_id, []).append(metrics["accuracy"])
            if spec.family == "missing":
                mask_records.append(
                    {
                        "condition": f"missing_{int(spec.severity)}",
                        "mask_id": "+".join(str(value) for value in spec.channels),
                        "fold": split.fold,
                        "accuracy": metrics["accuracy"],
                    }
                )
        reports.append(
            {
                "fold": split.fold,
                "search_split_sha256": split.sha256,
                "outer_split_sha256": split.outer_split_sha256,
                "outer_test_sha256": split.outer_test_sha256,
                "n_train_trials": len(train_indices),
                "n_validation_trials": len(validation_indices),
                "n_train_windows": len(x_train),
                "n_validation_windows": len(x_validation),
                "windows_per_validation_trial": windows_per_trial,
                "quality_flags": flags,
                "preprocessing": _clean_report(prepared.report),
            }
        )
    floors = {
        condition: physical_mask_floor(
            [record for record in mask_records if record["condition"] == condition]
        )
        for condition in ("missing_1", "missing_2")
    }
    return {
        "schema_version": 5,
        "status": "completed",
        "experiment": "exp04b",
        "wave": cell.wave,
        "cell_id": cell.cell_id,
        "kind": cell.kind,
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
        "quality_flags": sorted(flags_all),
        "physical_mask_floor": floors,
        "stress_condition_accuracy": {
            name: float(np.mean(values)) for name, values in secondary.items()
        },
        "fold_reports": reports,
        "parameters": {"total": 0, "active": 0, "frozen": 0, "auxiliary": 0},
        "source": source.payload(),
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
        "fingerprint": fingerprint,
    }


def _write_subject_manifests(output_dir: Path, subject: str, recording: Any, splits: Any) -> None:
    raw = Path(os.environ["CLEI_DATA_DIR"]) / f"{subject}.mat"
    write_json_atomic(
        output_dir / "manifests" / "data" / f"{subject}.json",
        {
            "subject": subject,
            "source_sha256": _file_sha256(raw),
            "standardized_data_sha256": recording.data_sha256,
        },
    )
    write_json_atomic(
        output_dir / "manifests" / "splits" / f"{subject}.json",
        {
            "subject": subject,
            "folds": [
                {
                    "fold": split.fold,
                    "train_indices": list(split.train_indices),
                    "validation_indices": list(split.validation_indices),
                    "outer_test_sha256": split.outer_test_sha256,
                    "sha256": split.sha256,
                }
                for split in splits
            ],
        },
    )


def run_wave(
    output_dir: Path,
    config: Exp04bConfig,
    catalog: tuple[SafePathCandidate, ...],
    source: Exp04Source,
    manifest: Exp04bManifest,
    *,
    wave: str,
    device: str,
    num_shards: int,
    shard_index: int,
) -> dict[str, int]:
    root = _root()
    git = git_identity(root)
    if config.profile == "full" and git["dirty"] is not False:
        raise RuntimeError("Full Exp04b execution requires a clean Git tree")
    if str(git["revision"] or "unknown") != manifest.revision:
        raise RuntimeError("Current revision differs from the Exp04b manifest")
    environment = runtime_environment(device)
    environment_hash = canonical_sha256(
        {key: environment[key] for key in ("python", "platform", "packages")}
    )
    write_json_atomic(
        output_dir / "manifests" / f"environment-{device.replace(':', '_')}.json",
        environment,
    )
    code_hash = exp04b_code_sha256(root)
    candidates = {candidate.config_id: candidate for candidate in catalog}
    wave_cells = tuple(cell for cell in manifest.cells if cell.wave == wave)
    selected = shard_cells_by_subject(wave_cells, num_shards=num_shards, shard_index=shard_index)
    recordings: dict[str, Any] = {}
    split_cache: dict[str, Any] = {}
    completed = failed = 0
    for cell in selected:
        try:
            if cell.subject not in recordings:
                recording = load_subject("MI-OpenBCI", cell.subject)
                splits = build_search_splits(
                    recording, folds=config.folds, split_seed=config.split_seed
                )
                recordings[cell.subject] = recording
                split_cache[cell.subject] = splits
                _write_subject_manifests(output_dir, cell.subject, recording, splits)
            recording = recordings[cell.subject]
            splits = split_cache[cell.subject]
            wave_manifest_hash = canonical_sha256(
                [asdict(item) for item in manifest.cells if item.wave == wave]
            )
            fingerprint = {
                "cell_config_sha256": cell.config_sha256,
                "config_sha256": manifest.config_sha256,
                "catalog_sha256": canonical_sha256([asdict(candidate) for candidate in catalog]),
                "manifest_sha256": wave_manifest_hash,
                "source_sha256": source.sha256,
                "code_sha256": code_hash,
                "data_sha256": recording.data_sha256,
                "split_sha256": canonical_sha256([split.sha256 for split in splits]),
                "environment_sha256": environment_hash,
                "revision": manifest.revision,
                "git_dirty": bool(git["dirty"]),
            }
            path = output_dir / "cells" / wave / f"{cell.cell_id}.json"
            failure_path = output_dir / "failures" / wave / f"{cell.cell_id}.json"
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("fingerprint") == fingerprint:
                    failure_path.unlink(missing_ok=True)
                    completed += 1
                    print(f"skipped {path.name}", flush=True)
                    continue
            if cell.kind == "classical_control":
                payload = _classical_result(
                    config,
                    source,
                    cell,
                    recording=recording,
                    splits=splits,
                    fingerprint=fingerprint,
                )
            else:
                candidate = candidates[cell.config_id]
                payload = _candidate_result(
                    output_dir,
                    config,
                    candidate,
                    source,
                    cell,
                    device=device,
                    recording=recording,
                    splits=splits,
                    fingerprint=fingerprint,
                )
            write_json_atomic(path, payload)
            failure_path.unlink(missing_ok=True)
            completed += 1
            print(f"completed {path.name}", flush=True)
        except Exception as error:  # noqa: BLE001
            failed += 1
            write_json_atomic(
                output_dir / "failures" / wave / f"{cell.cell_id}.json",
                {
                    "cell_id": cell.cell_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"failed {cell.cell_id}: {error}", flush=True)
    return {"planned": len(selected), "completed": completed, "failed": failed}
