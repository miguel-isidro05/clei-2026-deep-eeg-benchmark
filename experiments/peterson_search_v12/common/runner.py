"""Planning and execution of resumable V12 validation cells."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepbench.datasets import load_subject
from deepbench.metrics import compute_metrics
from deepbench.preprocessing import (
    aggregate_window_scores,
    augment_training,
    prepare_test_windows,
    preprocess_split,
)

from .audit import validate_cell_payload
from .catalogs import CandidateSpec, build_exp01_catalog, catalog_payload
from .config import SearchConfig
from .identity import code_sha256, git_identity
from .io import write_json_atomic
from .manifests import CellSpec, SearchManifest, build_manifest, shard_cells
from .models import parameter_report
from .splits import build_search_splits
from .training import predict_numpy, train_candidate


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _manifest_payload(manifest: SearchManifest) -> dict[str, Any]:
    return {
        "experiment": manifest.experiment,
        "revision": manifest.revision,
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "cells": [asdict(cell) for cell in manifest.cells],
        "sha256": manifest.sha256,
    }


def write_plan(output_dir: Path, config: SearchConfig, *, revision: str) -> dict[str, Any]:
    catalog = build_exp01_catalog(config)
    manifest = build_manifest(config, catalog, revision=revision)
    write_json_atomic(output_dir / "manifests" / "catalog-exp01.json", catalog_payload(catalog))
    write_json_atomic(output_dir / "manifests" / "exp01.json", _manifest_payload(manifest))
    plan = {
        "experiment": "exp01",
        "candidate_count": len(catalog),
        "cell_count": len(manifest.cells),
        "subjects": list(config.subjects),
        "folds": list(config.folds),
        "seed": 0,
        "manifest_sha256": manifest.sha256,
    }
    write_json_atomic(output_dir / "plan.json", plan)
    return plan


def _clean_preprocessing_report(report: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(report)
    cleaned["post_preprocessing_validation_sha256"] = cleaned.pop("post_preprocessing_test_sha256")
    cleaned["n_validation_trials"] = cleaned.pop("n_test_trials")
    return cleaned


def _environment(device: str) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("torch", "numpy", "scipy", "scikit-learn", "mne"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu_name = None
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.device(device))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": device,
        "gpu_name": gpu_name,
    }


def _run_cell(
    output_dir: Path,
    config: SearchConfig,
    candidate: CandidateSpec,
    cell: CellSpec,
    *,
    device: str,
    revision: str,
    code_hash: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    recording = load_subject("MI-OpenBCI", cell.subject)
    splits = build_search_splits(recording, folds=config.folds, split_seed=config.split_seed)
    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    peak_memory = 0
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
        window_y = np.repeat(recording.y[validation_indices], 6)
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
            config=config,
            device=device,
            seed=cell.seed * 10_000 + split_index,
            checkpoint_path=checkpoint,
        )
        window_probabilities = predict_numpy(
            outcome.model,
            x_validation,
            candidate=candidate,
            config=config,
            device=device,
            seed=cell.seed + split_index,
        )[:, 1]
        fold_pred, fold_score, windows_per_trial = aggregate_window_scores(
            window_probabilities, trial_indices, n_trials=len(validation_indices)
        )
        fold_y = recording.y[validation_indices]
        labels.extend(int(value) for value in fold_y)
        predictions.extend(int(value) for value in fold_pred)
        scores.extend(float(value) for value in fold_score)
        if device.startswith("cuda"):
            peak_memory = max(
                peak_memory, int(torch.cuda.max_memory_allocated(torch.device(device)))
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
                "best_epoch": outcome.best_epoch,
                "updates": outcome.updates,
                "history": outcome.history,
                "validation_metrics": compute_metrics(fold_y, fold_pred, fold_score),
                "preprocessing": _clean_preprocessing_report(prepared.report),
            }
        )
        del outcome.model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp01",
        "cell_id": cell.cell_id,
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
        "fold_reports": fold_reports,
        "parameters": parameter_report(candidate, n_chans=len(recording.ch_names)),
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_device_memory_bytes": peak_memory,
            "environment": _environment(device),
        },
        "fingerprint": {
            "candidate_sha256": candidate.sha256,
            "code_sha256": code_hash,
            "data_sha256": recording.data_sha256,
            "revision": revision,
        },
    }
    validate_cell_payload(payload)
    return payload


def run_search(
    output_dir: Path,
    config: SearchConfig,
    *,
    device: str,
    num_shards: int,
    shard_index: int,
    max_cells_per_shard: int | None = None,
) -> dict[str, int]:
    root = _root()
    git = git_identity(root)
    revision = str(git["revision"] or "unknown")
    plan = write_plan(output_dir, config, revision=revision)
    catalog = build_exp01_catalog(config)
    catalog_by_id = {candidate.config_id: candidate for candidate in catalog}
    manifest = build_manifest(config, catalog, revision=revision)
    cells = shard_cells(manifest.cells, num_shards=num_shards, shard_index=shard_index)
    if max_cells_per_shard is not None:
        cells = cells[:max_cells_per_shard]
    code_hash = code_sha256(root)
    completed = 0
    for cell in cells:
        path = output_dir / "cells" / f"{cell.cell_id}.json"
        if path.is_file():
            try:
                import json

                existing = json.loads(path.read_text(encoding="utf-8"))
                validate_cell_payload(existing)
                if (
                    existing["fingerprint"]["candidate_sha256"] == cell.candidate_sha256
                    and existing["fingerprint"]["code_sha256"] == code_hash
                ):
                    completed += 1
                    print(f"skipped {path.name}", flush=True)
                    continue
            except (ValueError, KeyError, TypeError):
                pass
        payload = _run_cell(
            output_dir,
            config,
            catalog_by_id[cell.config_id],
            cell,
            device=device,
            revision=revision,
            code_hash=code_hash,
        )
        write_json_atomic(path, payload)
        completed += 1
        print(f"completed {path.name}", flush=True)
    return {"planned": len(cells), "completed": completed, "manifest_cells": plan["cell_count"]}
