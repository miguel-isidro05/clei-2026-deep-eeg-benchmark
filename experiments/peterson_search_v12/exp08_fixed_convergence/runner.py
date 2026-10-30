"""Two-GPU Exp08 runner with shared fixed-convergence backbones per pair."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from deepbench.datasets import load_subject
from deepbench.metrics import compute_metrics
from deepbench.preprocessing import augment_training, prepare_test_windows, preprocess_split
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.identity import canonical_sha256, git_identity
from experiments.peterson_search_v12.common.io import write_json_atomic
from experiments.peterson_search_v12.common.runner import runtime_environment
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.runner import (
    _confusion_matrix,
    _parameter_report,
    _run_fold,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.splits import build_nested_folds
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.training import (
    refit_fixed_epochs,
    state_sha256,
)

from .catalog import all_candidates, backbone_pairs
from .models import build_model
from .training import (
    aggregate_grouped_logits,
    fit_frozen_lpwf,
    predict_grouped_logits,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def exp08_code_sha256(root: Path) -> str:
    paths = list((root / "experiments" / "peterson_search_v12" / "common").rglob("*.py"))
    for name in (
        "exp05_fbcnet_mechanisms",
        "exp06_window_fusion",
        "exp07_eegnet_fusion",
        "exp08_fixed_convergence",
    ):
        paths += list((root / "experiments" / "peterson_search_v12" / name).rglob("*.py"))
    paths += list((root / "scripts").glob("*peterson_exp08_v12*.py"))
    paths += list((root / "src" / "deepbench").glob("*.py"))
    paths += [root / "run_peterson_exp08_cayetano.sh"]
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cell_path(output_dir: Path, phase: str, cell_id: str) -> Path:
    return output_dir / f"{phase}_cells" / f"{cell_id}.json"


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
        "parent_decision_sha256": manifest.parent_decision_sha256,
        "screen_decision_sha256": manifest.screen_decision_sha256,
        "cell_config_sha256": cell.config_sha256,
        "candidate_sha256": candidate.sha256,
        "code_sha256": code_hash,
        "data_sha256": data_hash,
        "environment_sha256": canonical_sha256(environment),
        "split_sha256": split_hash,
        "revision": git["revision"],
        "git_dirty": git["dirty"],
        "device": environment["device"],
    }


def _resumable(path: Path, fingerprint: dict[str, Any], cell: Any) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("status") == "completed"
        and payload.get("cell_id") == cell.cell_id
        and payload.get("fingerprint") == fingerprint
    )


def _grouped(windows: np.ndarray, n_trials: int) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    if len(values) != n_trials * 6:
        raise RuntimeError("Exp08 requires exactly six crops per trial")
    return values.reshape(n_trials, 6, values.shape[1], values.shape[2])


def _split_record(split: Any) -> dict[str, Any]:
    body = {
        "split_unit": "trial",
        "outer_train_indices": list(split.outer_train_indices),
        "inner_train_indices": list(split.inner_train_indices),
        "inner_validation_indices": list(split.inner_validation_indices),
        "outer_test_indices": list(split.outer_test_indices),
        "outer_sha256": split.outer_sha256,
        "inner_sha256": split.inner_sha256,
    }
    return {**body, "sha256": canonical_sha256(body)}


def _deep_fold_pair(
    uniform_candidate: Any,
    lpwf_candidate: Any,
    cell: Any,
    recording: Any,
    split: Any,
    *,
    config: Any,
    device: str,
    split_index: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    seed = int(cell.seed) * 100_000 + split_index
    outer_train = np.asarray(split.outer_train_indices, dtype=np.int64)
    outer_test = np.asarray(split.outer_test_indices, dtype=np.int64)
    processed = preprocess_split(
        recording.x[outer_train],
        recording.x[outer_test],
        sfreq=recording.sfreq,
        ch_names=recording.ch_names,
        seed=seed,
        ica_policy="none",
        loader_bandpass_hz=recording.loader_bandpass_hz,
    )
    train_windows, train_labels = augment_training(
        processed.x_train, recording.y[outer_train], config.condition
    )
    test_windows, test_trial_indices = prepare_test_windows(
        processed.x_test, config.condition
    )
    expected_indices = np.repeat(np.arange(len(outer_test), dtype=np.int64), 6)
    if not np.array_equal(test_trial_indices, expected_indices):
        raise RuntimeError("Exp08 test crops are not grouped by trial")
    train_grouped = _grouped(train_windows, len(outer_train))
    test_grouped = _grouped(test_windows, len(outer_test))
    factory = lambda: build_model(  # noqa: E731
        uniform_candidate,
        n_chans=recording.x.shape[1],
        n_times=train_windows.shape[-1],
        sfreq=recording.sfreq,
    )
    outcome = refit_fixed_epochs(
        factory,
        uniform_candidate,
        train_windows,
        train_labels,
        device=device,
        seed=seed,
        epochs=int(config.backbone_epochs),
        scheduler_t_max=int(config.backbone_epochs),
        batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    base = outcome.model.base
    backbone_hash = state_sha256(base)
    train_logits = predict_grouped_logits(
        base, train_grouped, device=device, batch_size=int(config.batch_size)
    )
    test_logits = predict_grouped_logits(
        base, test_grouped, device=device, batch_size=int(config.batch_size)
    )
    fusion = fit_frozen_lpwf(
        base,
        train_logits,
        recording.y[outer_train],
        device=device,
        seed=seed + 50_000,
        epochs=int(config.fusion_epochs),
        learning_rate=float(config.fusion_learning_rate),
    )
    if (
        backbone_hash != fusion.backbone_state_before
        or backbone_hash != fusion.backbone_state_after
    ):
        raise RuntimeError("Exp08 shared backbone identity drifted during fusion")
    split_payload = _split_record(split)
    common = {
        "outer_fold": int(split.fold),
        "fold": int(split.fold),
        "session": split.session,
        "split": split_payload,
        "training_policy": "cropwise_fixed_then_frozen_fusion",
        "selected_epoch": None,
        "selection_method": "fixed_convergence_no_validation_selection",
        "backbone_epochs": int(outcome.epochs),
        "backbone_updates": int(outcome.updates),
        "backbone_history": list(outcome.history),
        "backbone_trajectory_sha256": canonical_sha256(
            {
                "seed": seed,
                "epochs": int(outcome.epochs),
                "updates": int(outcome.updates),
                "history": list(outcome.history),
                "state_sha256": backbone_hash,
            }
        ),
        "backbone_state_sha256": backbone_hash,
        "backbone_state_after_fusion_sha256": fusion.backbone_state_after,
        "refit_preprocessing": {**processed.report, "fit_partition": "outer_train"},
        "n_train_trials": int(len(outer_train)),
        "n_train_examples": int(len(train_windows)),
        "n_test_trials": int(len(outer_test)),
        "windows_per_test_trial": 6,
        "fusion_fit_partition": "outer_train_only",
        "fusion_epochs": int(config.fusion_epochs),
        "fusion_trainable_parameters": 6,
    }
    y_true = recording.y[outer_test]

    def report(aggregation: str) -> dict[str, Any]:
        if aggregation == "uniform":
            window_logits = np.zeros(6, dtype=float)
            window_weights = np.full(6, 1.0 / 6.0, dtype=float)
            fusion_history: list[dict[str, float | int]] = []
        else:
            window_logits = fusion.window_logits
            window_weights = fusion.window_weights
            fusion_history = list(fusion.history)
        y_score = aggregate_grouped_logits(
            test_logits, aggregation=aggregation, window_logits=window_logits
        )
        y_pred = (y_score >= 0.5).astype(np.int64)
        flags = validation_quality_flags(y_true.tolist(), y_pred.tolist())
        return {
            **common,
            "aggregation": aggregation,
            "model_diagnostics": {
                "aggregation": aggregation,
                "window_logits": np.asarray(window_logits, dtype=float).tolist(),
                "window_weights": np.asarray(window_weights, dtype=float).tolist(),
                "backbone_frozen_during_fusion": True,
            },
            "fusion_history": fusion_history,
            "y_true": y_true.astype(int).tolist(),
            "y_pred": y_pred.astype(int).tolist(),
            "y_score": y_score.astype(float).tolist(),
            "confusion_matrix": _confusion_matrix(y_true, y_pred),
            "metrics": compute_metrics(y_true, y_pred, y_score),
            "quality_flags": flags,
        }

    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    return report("uniform"), report("lpwf"), int(base_parameters)


def _payload(
    cell: Any,
    candidate: Any,
    reports: list[dict[str, Any]],
    *,
    fingerprint: dict[str, Any],
    environment: dict[str, Any],
    elapsed: float,
    parameter_report: dict[str, int],
) -> dict[str, Any]:
    labels = [value for report in reports for value in report["y_true"]]
    predictions = [value for report in reports for value in report["y_pred"]]
    scores = [value for report in reports for value in report["y_score"]]
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    y_score = np.asarray(scores, dtype=float)
    flags = sorted({flag for report in reports for flag in report["quality_flags"]})
    return {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp08",
        "phase": cell.phase,
        "kind": cell.kind,
        "cell_id": cell.cell_id,
        "config_id": cell.config_id,
        "mechanism": cell.mechanism,
        "subject": cell.subject,
        "seed": int(cell.seed),
        "folds": list(cell.folds),
        "candidate": asdict(candidate),
        "aggregate_metrics": compute_metrics(y_true, y_pred, y_score),
        "y_true": labels,
        "y_pred": predictions,
        "y_score": scores,
        "confusion_matrix": _confusion_matrix(y_true, y_pred),
        "quality_flags": flags,
        "fold_reports": reports,
        "parameter_report": parameter_report,
        "fingerprint": fingerprint,
        "runtime": {"seconds": float(elapsed), "environment": environment},
    }


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
    if phase not in {"screen", "confirm"} or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid Exp08 phase or shard")
    root = _root()
    git = git_identity(root)
    if config.profile == "full" and git["dirty"] is not False:
        raise RuntimeError("Exp08 scientific runs require a clean Git tree")
    if git["revision"] != manifest.revision:
        raise RuntimeError("Exp08 manifest revision does not match HEAD")
    environment = runtime_environment(device)
    code_hash = exp08_code_sha256(root)
    owned = set(tuple(config.subjects)[shard_index::num_shards])
    cells = [cell for cell in manifest.cells if cell.phase == phase and cell.subject in owned]
    if not cells:
        raise ValueError("Exp08 shard owns no cells")
    lookup = {candidate.config_id: candidate for candidate in all_candidates()}
    started = time.perf_counter()
    completed = skipped = failed = 0

    try:
        for subject in sorted(owned):
            recording = load_subject(config.dataset, subject)
            splits = build_nested_folds(
                recording,
                split_seed=int(config.split_seed),
                folds=tuple(config.folds),
                validation_fraction=float(config.inner_validation_fraction),
            )
            split_hash = canonical_sha256([split.sha256 for split in splits])
            subject_cells = [cell for cell in cells if cell.subject == subject]
            controls = [
                cell for cell in subject_cells if lookup[cell.config_id].backbone == "fbcnet"
            ]
            for cell in controls:
                candidate = lookup[cell.config_id]
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
                if _resumable(path, fingerprint, cell):
                    skipped += 1
                    continue
                cell_started = time.perf_counter()
                reports = []
                last_model = None
                for index, split in enumerate(splits):
                    fold, last_model = _run_fold(
                        candidate,
                        cell,
                        recording,
                        split,
                        config=config,
                        device=device,
                        split_index=index,
                        model_builder=build_model,
                    )
                    reports.append(fold)
                if last_model is None:
                    raise RuntimeError("Exp08 control produced no model")
                write_json_atomic(
                    path,
                    _payload(
                        cell,
                        candidate,
                        reports,
                        fingerprint=fingerprint,
                        environment=environment,
                        elapsed=time.perf_counter() - cell_started,
                        parameter_report=_parameter_report(last_model),
                    ),
                )
                completed += 1
                print(f"completed {path.name}", flush=True)

            pairs = backbone_pairs()
            for seed in sorted({cell.seed for cell in subject_cells}):
                for family, (uniform_id, lpwf_id) in pairs.items():
                    pair_cells = {
                        cell.config_id: cell
                        for cell in subject_cells
                        if cell.seed == seed and cell.config_id in {uniform_id, lpwf_id}
                    }
                    if not pair_cells:
                        continue
                    if set(pair_cells) != {uniform_id, lpwf_id}:
                        raise RuntimeError(f"Exp08 manifest split the {family} pair")
                    fingerprints = {
                        config_id: _fingerprint(
                            manifest,
                            cell,
                            lookup[config_id],
                            code_hash=code_hash,
                            data_hash=str(recording.data_sha256),
                            environment=environment,
                            git=git,
                            split_hash=split_hash,
                        )
                        for config_id, cell in pair_cells.items()
                    }
                    paths = {
                        config_id: _cell_path(output_dir, phase, cell.cell_id)
                        for config_id, cell in pair_cells.items()
                    }
                    if all(
                        _resumable(paths[key], fingerprints[key], pair_cells[key])
                        for key in pair_cells
                    ):
                        skipped += 2
                        continue
                    pair_started = time.perf_counter()
                    uniform_reports: list[dict[str, Any]] = []
                    lpwf_reports: list[dict[str, Any]] = []
                    base_parameters = 0
                    for index, split in enumerate(splits):
                        uniform, lpwf, base_parameters = _deep_fold_pair(
                            lookup[uniform_id],
                            lookup[lpwf_id],
                            pair_cells[uniform_id],
                            recording,
                            split,
                            config=config,
                            device=device,
                            split_index=index,
                        )
                        uniform_reports.append(uniform)
                        lpwf_reports.append(lpwf)
                    elapsed = time.perf_counter() - pair_started
                    for config_id, reports, extra in (
                        (uniform_id, uniform_reports, 0),
                        (lpwf_id, lpwf_reports, 6),
                    ):
                        write_json_atomic(
                            paths[config_id],
                            _payload(
                                pair_cells[config_id],
                                lookup[config_id],
                                reports,
                                fingerprint=fingerprints[config_id],
                                environment=environment,
                                elapsed=elapsed,
                                parameter_report={
                                    "total": base_parameters + extra,
                                    "trainable": base_parameters + extra,
                                    "backbone_stage_trainable": base_parameters,
                                    "fusion_stage_trainable": extra,
                                    "frozen_during_fusion": base_parameters,
                                },
                            ),
                        )
                        completed += 1
                        print(f"completed {paths[config_id].name}", flush=True)
    except Exception:
        failed += 1
        traceback.print_exc()
        raise
    summary = {
        "phase": phase,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "owned_subjects": sorted(owned),
        "planned": len(cells),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "manifest_sha256": manifest.sha256,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(output_dir / "logs" / f"{phase}-shard-{shard_index}-summary.json", summary)
    return summary


def write_run_failure(
    output_dir: Path, *, phase: str, shard_index: int, error: BaseException
) -> None:
    write_json_atomic(
        output_dir / "logs" / f"{phase}-shard-{shard_index}-failure.json",
        {"phase": phase, "shard_index": shard_index, "error": repr(error), "status": "failed"},
    )

