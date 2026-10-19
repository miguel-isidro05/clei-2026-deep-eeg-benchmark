"""Subject-sharded, resumable Exp05 execution with nested epoch selection."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, is_dataclass
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
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.identity import canonical_sha256, git_identity
from experiments.peterson_search_v12.common.io import write_json_atomic
from experiments.peterson_search_v12.common.runner import runtime_environment

from .catalog import candidate_by_id
from .models import build_model
from .splits import NestedFold, build_nested_folds
from .training import predict_window_scores, refit_fixed_epochs, select_epoch


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def exp05_code_sha256(root: Path) -> str:
    paths = list((root / "experiments" / "peterson_search_v12" / "common").rglob("*.py"))
    paths += list(
        (root / "experiments" / "peterson_search_v12" / "exp05_fbcnet_mechanisms").rglob(
            "*.py"
        )
    )
    paths += list((root / "scripts").glob("*peterson_exp05_v12*.py"))
    paths += list((root / "src" / "deepbench").glob("*.py"))
    paths += [root / "run_peterson_exp05_cayetano.sh"]
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    if is_dataclass(candidate):
        return asdict(candidate)
    if hasattr(candidate, "payload"):
        return dict(candidate.payload())
    return dict(vars(candidate))


def _manifest_payload(manifest: Any) -> dict[str, Any]:
    if is_dataclass(manifest):
        return asdict(manifest)
    return dict(manifest)


def _cell_path(output_dir: Path, phase: str, cell_id: str) -> Path:
    directory = "screen_cells" if phase == "screen" else "confirm_cells"
    return output_dir / directory / f"{cell_id}.json"


def _clean_inner_report(report: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(report)
    cleaned["post_preprocessing_validation_sha256"] = cleaned.pop(
        "post_preprocessing_test_sha256"
    )
    cleaned["n_validation_trials"] = cleaned.pop("n_test_trials")
    cleaned["fit_partition"] = "inner_train"
    return cleaned


def _outer_report(report: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(report)
    cleaned["fit_partition"] = "outer_train"
    return cleaned


def _positions(n_trials: int, *, windows_per_trial: int = 6) -> np.ndarray:
    return np.tile(np.arange(windows_per_trial, dtype=np.int64), n_trials)


def _uses_grouped_windows(candidate: Any) -> bool:
    return candidate.mechanism in {"global_window_weights", "content_window_attention"}


def _grouped_windows(x: np.ndarray) -> np.ndarray:
    windows, trial_indices = prepare_test_windows(x, "overlap")
    counts = np.bincount(trial_indices, minlength=len(x))
    if len(set(counts.tolist())) != 1 or int(counts[0]) != 6:
        raise RuntimeError("Exp05 grouped-window candidates require six windows per trial")
    return windows.reshape(len(x), 6, windows.shape[1], windows.shape[2])


def _mechanisms() -> Any:
    from . import mechanisms

    return mechanisms


def _fit_context(candidate: Any, x: np.ndarray, y: np.ndarray, sfreq: float) -> dict[str, Any]:
    module = _mechanisms()
    if hasattr(module, "prepare_fit_context"):
        context = module.prepare_fit_context(candidate, x, y, sfreq=sfreq)
        return {} if context is None else dict(context)
    return {}


def _context_report(context: dict[str, Any]) -> dict[str, Any]:
    module = _mechanisms()
    if hasattr(module, "fit_context_report"):
        return dict(module.fit_context_report(context))
    report: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, np.ndarray):
            report[f"{key}_shape"] = list(value.shape)
            report[f"{key}_sha256"] = canonical_sha256(value.tolist())
        elif isinstance(value, (str, int, float, bool)) or value is None:
            report[key] = value
        elif isinstance(value, (list, tuple)):
            report[key] = list(value)
    return report


def _sample_weights(candidate: Any, x: np.ndarray, context: dict[str, Any]) -> np.ndarray | None:
    from . import models as module
    if hasattr(module, "sample_weights"):
        values = module.sample_weights(x, candidate, context)
        if values is None:
            return None
        weights = np.asarray(values, dtype=np.float32)
        if len(weights) != len(x) or not np.isfinite(weights).all():
            raise ValueError("Invalid Exp05 sample weights")
        if np.any(weights < 0.5) or np.any(weights > 1.5):
            raise ValueError("Exp05 quality weights must remain in [0.5, 1.5]")
        return weights
    return None


def _parameter_report(model: torch.nn.Module) -> dict[str, int]:
    if hasattr(model, "parameter_report"):
        report = dict(model.parameter_report())
        if "trainable" not in report:
            report["trainable"] = int(report.get("active", 0))
        return report
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    return {
        "total": int(trainable + frozen),
        "active": int(trainable),
        "trainable": int(trainable),
        "auxiliary": 0,
        "frozen": int(frozen),
    }


def _aggregate(
    model: torch.nn.Module,
    probabilities: np.ndarray,
    trial_indices: np.ndarray,
    *,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if hasattr(model, "aggregate_window_probabilities"):
        scores = np.asarray(
            model.aggregate_window_probabilities(probabilities, trial_indices, n_trials=n_trials),
            dtype=float,
        )
        if scores.shape != (n_trials,) or not np.isfinite(scores).all():
            raise ValueError("Candidate returned invalid trial probabilities")
        counts = np.bincount(trial_indices, minlength=n_trials)
        if len(set(counts.tolist())) != 1:
            raise RuntimeError("Unequal window support across trials")
        return (scores >= 0.5).astype(np.int64), scores, int(counts[0])
    return aggregate_window_scores(probabilities, trial_indices, n_trials=n_trials)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    return [
        [
            int(np.sum((y_true == actual) & (y_pred == predicted)))
            for predicted in (0, 1)
        ]
        for actual in (0, 1)
    ]


def _run_fold(
    candidate: Any,
    cell: Any,
    recording: Any,
    split: NestedFold,
    *,
    config: Any,
    device: str,
    split_index: int,
) -> tuple[dict[str, Any], torch.nn.Module]:
    seed = int(cell.seed) * 100_000 + split_index
    inner_train_indices = np.asarray(split.inner_train_indices, dtype=np.int64)
    inner_validation_indices = np.asarray(split.inner_validation_indices, dtype=np.int64)
    outer_train_indices = np.asarray(split.outer_train_indices, dtype=np.int64)
    outer_test_indices = np.asarray(split.outer_test_indices, dtype=np.int64)

    inner = preprocess_split(
        recording.x[inner_train_indices],
        recording.x[inner_validation_indices],
        sfreq=recording.sfreq,
        ch_names=recording.ch_names,
        seed=seed,
        ica_policy="none",
        loader_bandpass_hz=recording.loader_bandpass_hz,
    )
    grouped = _uses_grouped_windows(candidate)
    if grouped:
        inner_x_train = _grouped_windows(inner.x_train)
        inner_y_train = recording.y[inner_train_indices]
        inner_x_validation = _grouped_windows(inner.x_test)
        inner_y_validation = recording.y[inner_validation_indices]
    else:
        inner_x_train, inner_y_train = augment_training(
            inner.x_train, recording.y[inner_train_indices], config.condition
        )
        inner_x_validation, _inner_validation_trial_indices = prepare_test_windows(
            inner.x_test, config.condition
        )
        inner_y_validation = np.repeat(recording.y[inner_validation_indices], 6)
    inner_context = _fit_context(
        candidate, inner_x_train, inner_y_train, sfreq=recording.sfreq
    )
    inner_weights = _sample_weights(candidate, inner_x_train, inner_context)
    inner_factory = lambda: build_model(  # noqa: E731
        candidate,
        n_chans=recording.x.shape[1],
        n_times=inner_x_train.shape[-1],
        sfreq=recording.sfreq,
        fit_context=inner_context,
    )
    selection = select_epoch(
        inner_factory,
        candidate,
        inner_x_train,
        inner_y_train,
        inner_x_validation,
        inner_y_validation,
        device=device,
        seed=seed,
        max_epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        patience=min(75, int(config.epochs)),
        train_weights=inner_weights,
        train_positions=None if grouped else _positions(len(inner_train_indices)),
        validation_positions=None if grouped else _positions(len(inner_validation_indices)),
    )

    outer = preprocess_split(
        recording.x[outer_train_indices],
        recording.x[outer_test_indices],
        sfreq=recording.sfreq,
        ch_names=recording.ch_names,
        seed=seed,
        ica_policy="none",
        loader_bandpass_hz=recording.loader_bandpass_hz,
    )
    if grouped:
        outer_x_train = _grouped_windows(outer.x_train)
        outer_y_train = recording.y[outer_train_indices]
        outer_x_test = _grouped_windows(outer.x_test)
        outer_test_trial_indices = np.arange(len(outer_test_indices), dtype=np.int64)
    else:
        outer_x_train, outer_y_train = augment_training(
            outer.x_train, recording.y[outer_train_indices], config.condition
        )
        outer_x_test, outer_test_trial_indices = prepare_test_windows(
            outer.x_test, config.condition
        )
    outer_context = _fit_context(
        candidate, outer_x_train, outer_y_train, sfreq=recording.sfreq
    )
    outer_weights = _sample_weights(candidate, outer_x_train, outer_context)
    outer_factory = lambda: build_model(  # noqa: E731
        candidate,
        n_chans=recording.x.shape[1],
        n_times=outer_x_train.shape[-1],
        sfreq=recording.sfreq,
        fit_context=outer_context,
    )
    refit = refit_fixed_epochs(
        outer_factory,
        candidate,
        outer_x_train,
        outer_y_train,
        device=device,
        seed=seed,
        epochs=selection.selected_epoch,
        scheduler_t_max=int(config.epochs),
        batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        train_weights=outer_weights,
        train_positions=None if grouped else _positions(len(outer_train_indices)),
    )
    window_probabilities = predict_window_scores(
        refit.model,
        outer_x_test,
        device=device,
        batch_size=int(config.batch_size),
        positions=None if grouped else _positions(len(outer_test_indices)),
    )
    if grouped:
        y_score = window_probabilities
        y_pred = (y_score >= 0.5).astype(np.int64)
        windows_per_trial = 6
    else:
        y_pred, y_score, windows_per_trial = _aggregate(
            refit.model,
            window_probabilities,
            outer_test_trial_indices,
            n_trials=len(outer_test_indices),
        )
    y_true = recording.y[outer_test_indices]
    flags = validation_quality_flags(y_true.tolist(), y_pred.tolist())
    selection_history = list(selection.history)
    selection_contract = {
        "candidate": str(candidate.config_id),
        "seed": seed,
        "max_epochs": int(config.epochs),
        "patience": min(75, int(config.epochs)),
        "scheduler_t_max": int(config.epochs),
    }
    selection_contract["sha256"] = canonical_sha256(selection_contract)
    refit_history = list(refit.history)
    refit_body = {
        "candidate": str(candidate.config_id),
        "seed": seed,
        "epochs": int(refit.epochs),
        "scheduler_t_max": int(config.epochs),
        "history": refit_history,
        "state_sha256": refit.state_sha256,
        "updates": int(refit.updates),
    }
    split_record = {
        "split_unit": "trial",
        "outer_train_indices": list(split.outer_train_indices),
        "inner_train_indices": list(split.inner_train_indices),
        "inner_validation_indices": list(split.inner_validation_indices),
        "outer_test_indices": list(split.outer_test_indices),
        "outer_sha256": split.outer_sha256,
        "inner_sha256": split.inner_sha256,
    }
    split_record["sha256"] = canonical_sha256(split_record)
    report = {
        "outer_fold": int(split.fold),
        "fold": int(split.fold),
        "session": split.session,
        "split": split_record,
        "selected_epoch": selection.selected_epoch,
        "best_inner_validation_loss": selection.best_validation_loss,
        "selection_trajectory_sha256": selection.trajectory_sha256,
        "selection_history": selection_history,
        "selection_contract": selection_contract,
        "selection_preprocessing": _clean_inner_report(inner.report),
        "refit_preprocessing": _outer_report(outer.report),
        "selection_fit_context": _context_report(inner_context),
        "refit_fit_context": _context_report(outer_context),
        "refit_state_sha256": refit.state_sha256,
        "refit_updates": refit.updates,
        "refit_epochs": refit.epochs,
        "refit_history": refit_history,
        "refit_trajectory_sha256": canonical_sha256(refit_body),
        "n_train_trials": int(len(outer_train_indices)),
        "n_train_examples": int(len(outer_x_train)),
        "n_test_trials": int(len(outer_test_indices)),
        "windows_per_test_trial": windows_per_trial,
        "y_true": y_true.astype(int).tolist(),
        "y_pred": y_pred.astype(int).tolist(),
        "y_score": y_score.astype(float).tolist(),
        "confusion_matrix": _confusion_matrix(y_true, y_pred),
        "metrics": compute_metrics(y_true, y_pred, y_score),
        "quality_flags": flags,
    }
    return report, refit.model


def _fingerprint(
    manifest: Any,
    cell: Any,
    candidate: Any,
    *,
    code_hash: str,
    data_hash: str,
    environment: dict[str, Any],
    git: dict[str, Any],
    split_hash: str,
) -> dict[str, Any]:
    return {
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "manifest_sha256": manifest.sha256,
        "cell_config_sha256": cell.config_sha256,
        "candidate_sha256": candidate.sha256,
        "screen_decision_sha256": getattr(manifest, "screen_decision_sha256", None),
        "code_sha256": code_hash,
        "data_sha256": data_hash,
        "environment_sha256": canonical_sha256(environment),
        "split_sha256": split_hash,
        "revision": git["revision"],
        "git_dirty": git["dirty"],
        "device": environment["device"],
    }


def _is_resumable(
    path: Path,
    fingerprint: dict[str, Any],
    *,
    cell: Any,
    manifest: Any,
    allow_dirty: bool,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return False
        from .audit import validate_cell

        validate_cell(payload, cell, manifest, allow_dirty=allow_dirty)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return True


def run_phase(
    output_dir: Path,
    config: Any,
    catalog: tuple[Any, ...],
    manifest: Any,
    *,
    phase: str,
    device: str,
    num_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    if phase not in {"screen", "confirm"}:
        raise ValueError("Exp05 phase must be screen or confirm")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid Exp05 shard")
    root = _root()
    git = git_identity(root)
    if config.profile == "full" and git["dirty"] is not False:
        raise RuntimeError("Exp05 scientific runs require a clean Git tree")
    if git["revision"] != manifest.revision:
        raise RuntimeError("Exp05 manifest revision does not match HEAD")
    environment = runtime_environment(device)
    code_hash = exp05_code_sha256(root)
    cells = tuple(cell for cell in manifest.cells if cell.phase == phase)
    subjects = tuple(config.subjects)
    owned_subjects = set(subjects[shard_index::num_shards])
    cells = tuple(cell for cell in cells if cell.subject in owned_subjects)
    if not cells:
        raise ValueError("Exp05 shard owns no cells")
    candidates = {candidate.config_id: candidate for candidate in catalog}
    started = time.perf_counter()
    completed = skipped = failed = 0
    for cell in cells:
        try:
            cell_started = time.perf_counter()
            recording = load_subject(config.dataset, cell.subject)
            splits = build_nested_folds(
                recording,
                split_seed=int(config.split_seed),
                folds=tuple(cell.folds),
                validation_fraction=float(config.inner_validation_fraction),
            )
            candidate = candidates.get(cell.config_id)
            if candidate is None:
                candidate = candidate_by_id(cell.config_id)
            split_hash = canonical_sha256([split.sha256 for split in splits])
            fingerprint = _fingerprint(
                manifest,
                cell,
                candidate,
                code_hash=code_hash,
                data_hash=str(recording.data_sha256),
                environment=environment,
                git=git,
                split_hash=split_hash,
            )
            path = _cell_path(output_dir, phase, cell.cell_id)
            if _is_resumable(
                path,
                fingerprint,
                cell=cell,
                manifest=manifest,
                allow_dirty=config.profile != "full",
            ):
                skipped += 1
                print(f"skipped {path.name}", flush=True)
                continue
            labels: list[int] = []
            predictions: list[int] = []
            scores: list[float] = []
            reports: list[dict[str, Any]] = []
            flags: set[str] = set()
            last_model: torch.nn.Module | None = None
            for split_index, split in enumerate(splits):
                report, last_model = _run_fold(
                    candidate,
                    cell,
                    recording,
                    split,
                    config=config,
                    device=device,
                    split_index=split_index,
                )
                reports.append(report)
                labels.extend(report["y_true"])
                predictions.extend(report["y_pred"])
                scores.extend(report["y_score"])
                flags.update(report["quality_flags"])
            if last_model is None:
                raise RuntimeError("Exp05 cell produced no fitted model")
            label_array = np.asarray(labels, dtype=np.int64)
            prediction_array = np.asarray(predictions, dtype=np.int64)
            score_array = np.asarray(scores, dtype=float)
            payload = {
                "schema_version": 1,
                "status": "completed",
                "experiment": "exp05",
                "phase": phase,
                "kind": cell.kind,
                "cell_id": cell.cell_id,
                "config_id": cell.config_id,
                "mechanism": cell.mechanism,
                "subject": cell.subject,
                "seed": int(cell.seed),
                "folds": list(cell.folds),
                "candidate": _candidate_payload(candidate),
                "aggregate_metrics": compute_metrics(
                    label_array,
                    prediction_array,
                    score_array,
                ),
                "y_true": labels,
                "y_pred": predictions,
                "y_score": scores,
                "confusion_matrix": _confusion_matrix(label_array, prediction_array),
                "quality_flags": sorted(flags),
                "fold_reports": reports,
                "parameter_report": _parameter_report(last_model),
                "fingerprint": fingerprint,
                "runtime": {
                    "seconds": float(time.perf_counter() - cell_started),
                    "environment": environment,
                },
            }
            write_json_atomic(path, payload)
            completed += 1
            print(f"completed {path.name}", flush=True)
        except Exception:
            failed += 1
            traceback.print_exc()
            raise
    summary = {
        "phase": phase,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "owned_subjects": sorted(owned_subjects),
        "planned": len(cells),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "manifest_sha256": manifest.sha256,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(output_dir / "logs" / f"{phase}-shard-{shard_index}-summary.json", summary)
    return summary


def write_manifest(output_dir: Path, manifest: Any, *, phase: str) -> None:
    write_json_atomic(output_dir / "manifests" / f"exp05-{phase}.json", _manifest_payload(manifest))


def write_run_failure(
    output_dir: Path, *, phase: str, shard_index: int, error: BaseException
) -> None:
    write_json_atomic(
        output_dir / "logs" / f"{phase}-shard-{shard_index}-failure.json",
        {
            "phase": phase,
            "shard_index": shard_index,
            "error": repr(error),
            "status": "failed",
        },
    )
