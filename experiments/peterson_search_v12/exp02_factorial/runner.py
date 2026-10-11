"""Resumable two-GPU execution for the exp02 factorial search."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict, replace
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
    exp02_code_sha256,
    git_identity,
)
from experiments.peterson_search_v12.common.io import (
    write_json_atomic,
    write_torch_atomic,
)
from experiments.peterson_search_v12.common.runner import runtime_environment
from experiments.peterson_search_v12.common.splits import build_search_splits

from .audit import validate_baseline_cell, validate_candidate_cell
from .catalog import (
    Exp02Candidate,
    build_candidate_catalog,
    build_inference_catalog,
    catalog_payload,
    inference_catalog_payload,
)
from .config import Exp02Config
from .manifest import (
    ExperimentCell,
    ExperimentManifest,
    build_experiment_manifest,
    manifest_payload,
    shard_cells,
)
from .models import parameter_report
from .training import predict_all, train_candidate


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoke_manifest(full: ExperimentManifest) -> ExperimentManifest:
    requested = {
        "exp02-cfg-001__seed0__S09",
        "exp02-cfg-004__seed0__S03",
        "exp02-cfg-037__seed0__S02",
        "exp02-cfg-040__seed0__S10",
        "baseline__CSP-LDA__seed0__S12",
        "baseline__FBCNet__seed0__S08",
    }
    cells = tuple(cell for cell in full.cells if cell.cell_id in requested)
    if len(cells) != len(requested):
        raise RuntimeError("Smoke manifest identities drifted")
    body = {
        "experiment": full.experiment,
        "profile": "smoke",
        "revision": full.revision,
        "config_sha256": full.config_sha256,
        "catalog_sha256": full.catalog_sha256,
        "parent_manifest_sha256": full.sha256,
        "cells": [asdict(cell) for cell in cells],
    }
    return replace(full, cells=cells, sha256=canonical_sha256(body))


def write_plan(
    output_dir: Path,
    config: Exp02Config,
    *,
    revision: str,
    smoke: bool = False,
) -> dict[str, Any]:
    candidates = build_candidate_catalog(config)
    inference = build_inference_catalog(candidates)
    full = build_experiment_manifest(config, candidates, revision=revision)
    manifest = _smoke_manifest(full) if smoke else full
    write_json_atomic(output_dir / "manifests" / "catalog-exp02.json", catalog_payload(candidates))
    write_json_atomic(
        output_dir / "manifests" / "inference-catalog-exp02.json",
        inference_catalog_payload(inference),
    )
    name = "smoke-exp02.json" if smoke else "experiment-exp02.json"
    write_json_atomic(output_dir / "manifests" / name, manifest_payload(manifest))
    plan = {
        "experiment": "exp02",
        "profile": "smoke" if smoke else "full",
        "candidate_count": len(candidates),
        "inference_recipe_count": len(inference),
        "cell_count": len(manifest.cells),
        "full_cell_count": len(full.cells),
        "subjects": list(config.subjects),
        "folds": list(config.folds),
        "manifest_sha256": manifest.sha256,
    }
    write_json_atomic(output_dir / ("smoke-plan.json" if smoke else "plan.json"), plan)
    return plan


def _clean_preprocessing_report(report: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(report)
    cleaned["post_preprocessing_validation_sha256"] = cleaned.pop("post_preprocessing_test_sha256")
    cleaned["n_validation_trials"] = cleaned.pop("n_test_trials")
    return cleaned


def _data_fragment(output_dir: Path, subject: str, recording: Any) -> None:
    data_root = Path(os.environ["CLEI_DATA_DIR"])
    raw_path = data_root / f"{subject}.mat"
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
        output_dir / "manifests" / "splits" / f"{subject}.json",
        {
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


def _candidate_result(
    output_dir: Path,
    config: Exp02Config,
    candidate: Exp02Candidate,
    cell: ExperimentCell,
    *,
    device: str,
    fingerprint: dict[str, Any],
    recording: Any,
    splits: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    expected_recipes = {recipe.inference_id for recipe in build_inference_catalog((candidate,))}
    accumulated = {
        inference_id: {"y_true": [], "y_pred": [], "y_score": [], "fold_flags": []}
        for inference_id in expected_recipes
    }
    fold_reports: list[dict[str, Any]] = []
    peak_memory = 0
    model_parameters: dict[str, int] | None = None
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
        checkpoint = (
            output_dir
            / "checkpoints"
            / f"{cell.cell_id}__session-{split.session}__fold{split.fold}.pt"
        )
        outcome = train_candidate(
            candidate,
            x_train,
            y_train,
            x_validation,
            window_y,
            sfreq=recording.sfreq,
            config=config,
            device=device,
            seed=cell.seed * 10_000 + split_index,
            checkpoint_path=checkpoint,
        )
        model_parameters = parameter_report(outcome.model, candidate)
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        inference_started = time.perf_counter()
        window_outputs = predict_all(
            outcome.model,
            x_validation,
            candidate=candidate,
            config=config,
            device=device,
            seed=cell.seed * 100 + split_index,
        )
        if set(window_outputs) != expected_recipes:
            raise RuntimeError("Runtime inference recipes do not match the signed catalog")
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        inference_seconds = time.perf_counter() - inference_started
        fold_inference_metrics: dict[str, dict[str, float]] = {}
        fold_inference_flags: dict[str, list[str]] = {}
        fold_y = recording.y[validation_indices]
        for inference_id, probabilities in window_outputs.items():
            fold_pred, fold_score, windows_per_trial = aggregate_window_scores(
                probabilities[:, 1], trial_indices, n_trials=len(validation_indices)
            )
            flags = validation_quality_flags(
                [int(value) for value in fold_y], [int(value) for value in fold_pred]
            )
            bucket = accumulated[inference_id]
            bucket["y_true"].extend(int(value) for value in fold_y)
            bucket["y_pred"].extend(int(value) for value in fold_pred)
            bucket["y_score"].extend(float(value) for value in fold_score)
            bucket["fold_flags"].append(flags)
            fold_inference_metrics[inference_id] = compute_metrics(fold_y, fold_pred, fold_score)
            fold_inference_flags[inference_id] = flags
        if device.startswith("cuda"):
            peak_memory = max(
                peak_memory,
                int(torch.cuda.max_memory_allocated(torch.device(device))),
            )
        fold_reports.append(
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
                "inference_seconds_per_trial_all_recipes": (
                    inference_seconds / len(validation_indices)
                ),
                "best_epoch": outcome.best_epoch,
                "updates": outcome.updates,
                "history": outcome.history,
                "inference_metrics": fold_inference_metrics,
                "inference_quality_flags": fold_inference_flags,
                "quality_flags": sorted(
                    {flag for flags in fold_inference_flags.values() for flag in flags}
                ),
                "preprocessing": _clean_preprocessing_report(prepared.report),
            }
        )
        del outcome.model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    inference_results: dict[str, Any] = {}
    cell_flags: set[str] = set()
    for inference_id, values in accumulated.items():
        y_true = np.asarray(values["y_true"], dtype=int)
        y_pred = np.asarray(values["y_pred"], dtype=int)
        y_score = np.asarray(values["y_score"], dtype=float)
        flags = sorted({flag for fold in values["fold_flags"] for flag in fold})
        cell_flags.update(flags)
        alpha_part, k_part = inference_id.rsplit("__", 2)[-2:]
        inference_results[inference_id] = {
            "alpha": float(alpha_part.removeprefix("alpha-")),
            "k": int(k_part.removeprefix("k")),
            "metrics": compute_metrics(y_true, y_pred, y_score),
            "quality_flags": flags,
            "fold_quality_flags": values["fold_flags"],
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_score": y_score.tolist(),
        }
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 2,
        "status": "completed",
        "experiment": "exp02",
        "kind": "candidate",
        "cell_id": cell.cell_id,
        "config_id": candidate.config_id,
        "candidate": asdict(candidate),
        "subject": cell.subject,
        "seed": cell.seed,
        "folds": list(cell.folds),
        "quality_flags": sorted(cell_flags),
        "inference_results": inference_results,
        "fold_reports": fold_reports,
        "parameters": model_parameters,
        "runtime": {
            "elapsed_seconds": elapsed,
            "inference_seconds_per_trial": float(
                np.mean(
                    [report["inference_seconds_per_trial_all_recipes"] for report in fold_reports]
                )
            ),
            "peak_device_memory_bytes": peak_memory,
            "environment": runtime_environment(device),
        },
        "fingerprint": fingerprint,
    }
    validate_candidate_cell(payload, expected_inference_ids=expected_recipes)
    return payload


def _baseline_result(
    output_dir: Path,
    config: Exp02Config,
    cell: ExperimentCell,
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
    quality_flags: set[str] = set()
    params = 0
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
            inference_started = time.perf_counter()
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
                seed=cell.seed * 1000 + split_index,
                epochs=config.baseline_epochs,
            )
            classifier.fit(x_train, y_train)
            windows, trial_indices = prepare_test_windows(prepared.x_test, config.condition)
            inference_started = time.perf_counter()
            _, window_scores = predict_scores(classifier, windows)
            fold_pred, fold_score, windows_per_trial = aggregate_window_scores(
                window_scores, trial_indices, n_trials=len(validation_indices)
            )
            params = parameter_count("FBCNet", x_train.shape[1], x_train.shape[-1], recording.sfreq)
            history = [
                {
                    "epoch": index + 1,
                    "train_loss": float(row["train_loss"]),
                }
                for index, row in enumerate(classifier.history)
            ]
            recipe = recipe_dict("FBCNet", config.baseline_epochs)
            write_torch_atomic(
                output_dir
                / "checkpoints"
                / f"{cell.cell_id}__session-{split.session}__fold{split.fold}.pt",
                {
                    "schema_version": 2,
                    "model": "FBCNet",
                    "recipe": recipe,
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in classifier.module_.state_dict().items()
                    },
                },
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        inference_seconds = time.perf_counter() - inference_started
        fold_y = recording.y[validation_indices]
        flags = validation_quality_flags(
            [int(value) for value in fold_y], [int(value) for value in fold_pred]
        )
        quality_flags.update(flags)
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
                "inference_seconds_per_trial": inference_seconds / len(validation_indices),
                "validation_metrics": compute_metrics(fold_y, fold_pred, fold_score),
                "quality_flags": flags,
                "history": history,
                "preprocessing": _clean_preprocessing_report(prepared.report),
            }
        )
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    payload = {
        "schema_version": 2,
        "status": "completed",
        "experiment": "exp02",
        "kind": "baseline",
        "cell_id": cell.cell_id,
        "model_id": cell.model_id,
        "subject": cell.subject,
        "seed": cell.seed,
        "folds": list(cell.folds),
        "validation_metrics": compute_metrics(y_true, y_pred, y_score),
        "quality_flags": sorted(quality_flags),
        "validation_y_true": y_true.tolist(),
        "validation_y_pred": y_pred.tolist(),
        "validation_y_score": y_score.tolist(),
        "fold_reports": reports,
        "parameters": {"total": params},
        "recipe": recipe,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "inference_seconds_per_trial": float(
                np.mean([report["inference_seconds_per_trial"] for report in reports])
            ),
            "environment": runtime_environment(device),
        },
        "fingerprint": fingerprint,
    }
    validate_baseline_cell(payload)
    return payload


def _is_resumable(path: Path, fingerprint: dict[str, Any], kind: str, recipes: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if kind == "candidate":
            validate_candidate_cell(payload, expected_inference_ids=recipes)
        else:
            validate_baseline_cell(payload)
        return payload.get("fingerprint") == fingerprint
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def run_search(
    output_dir: Path,
    config: Exp02Config,
    *,
    device: str,
    num_shards: int,
    shard_index: int,
    smoke: bool = False,
    max_cells_per_shard: int | None = None,
) -> dict[str, int]:
    root = _root()
    git = git_identity(root)
    revision = str(git["revision"] or "unknown")
    if git["dirty"] is not False and not smoke:
        raise RuntimeError("Full exp02 requires a clean Git tree")
    candidates = build_candidate_catalog(config)
    candidate_by_id = {candidate.config_id: candidate for candidate in candidates}
    inference = build_inference_catalog(candidates)
    recipes_by_candidate: dict[str, set[str]] = {}
    for recipe in inference:
        recipes_by_candidate.setdefault(recipe.config_id, set()).add(recipe.inference_id)
    full = build_experiment_manifest(config, candidates, revision=revision)
    manifest = _smoke_manifest(full) if smoke else full
    write_plan(output_dir, config, revision=revision, smoke=smoke)
    selected = shard_cells(manifest.cells, num_shards=num_shards, shard_index=shard_index)
    if max_cells_per_shard is not None:
        selected = selected[:max_cells_per_shard]
    code_hash = exp02_code_sha256(root)
    environment = runtime_environment(device)
    environment_hash = canonical_sha256(environment)
    device_label = device.replace(":", "_")
    write_json_atomic(output_dir / "manifests" / f"environment-{device_label}.json", environment)
    recording_cache: dict[str, Any] = {}
    split_cache: dict[str, Any] = {}
    completed = 0
    failed = 0
    for cell in selected:
        if cell.subject not in recording_cache:
            recording = load_subject("MI-OpenBCI", cell.subject)
            recording_cache[cell.subject] = recording
            split_cache[cell.subject] = build_search_splits(
                recording, folds=config.folds, split_seed=config.split_seed
            )
            _data_fragment(output_dir, cell.subject, recording)
            _split_fragment(output_dir, cell.subject, split_cache[cell.subject])
        recording = recording_cache[cell.subject]
        splits = split_cache[cell.subject]
        fingerprint = {
            "candidate_sha256": cell.candidate_sha256,
            "catalog_sha256": manifest.catalog_sha256,
            "config_sha256": manifest.config_sha256,
            "manifest_sha256": manifest.sha256,
            "code_sha256": code_hash,
            "data_sha256": recording.data_sha256,
            "split_sha256": canonical_sha256([split.sha256 for split in splits]),
            "environment_sha256": environment_hash,
            "device": device,
            "revision": revision,
            "git_dirty": bool(git["dirty"]),
        }
        path = output_dir / "cells" / f"{cell.cell_id}.json"
        recipes = recipes_by_candidate.get(cell.config_id or "", set())
        if _is_resumable(path, fingerprint, cell.kind, recipes):
            completed += 1
            print(f"skipped {path.name}", flush=True)
            continue
        try:
            if cell.kind == "candidate":
                if cell.config_id is None:
                    raise RuntimeError("Candidate cell is missing config_id")
                payload = _candidate_result(
                    output_dir,
                    config,
                    candidate_by_id[cell.config_id],
                    cell,
                    device=device,
                    fingerprint=fingerprint,
                    recording=recording,
                    splits=splits,
                )
            else:
                payload = _baseline_result(
                    output_dir,
                    config,
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
                    "schema_version": 2,
                    "status": "failed",
                    "experiment": "exp02",
                    "cell_id": cell.cell_id,
                    "kind": cell.kind,
                    "subject": cell.subject,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "fingerprint": fingerprint,
                },
            )
            print(f"failed {cell.cell_id}: {type(error).__name__}: {error}", flush=True)
    return {
        "planned": len(selected),
        "completed": completed,
        "failed": failed,
        "manifest_cells": len(manifest.cells),
    }
