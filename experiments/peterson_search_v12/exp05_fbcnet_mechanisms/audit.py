"""Portable completeness, leakage, metric, and signature checks for Exp05."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.audit import validation_quality_flags
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.partitions import DISCOVERY_SUBJECTS, HOLDOUT_SUBJECTS

from .manifest import Exp05Cell, Exp05Manifest


def _as_int_set(value: Any, name: str) -> set[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an explicit list of trial indices")
    converted = [int(item) for item in value]
    if len(converted) != len(set(converted)):
        raise ValueError(f"{name} contains duplicate trial indices")
    return set(converted)


def _validate_split(fold: dict[str, Any]) -> None:
    split = fold.get("split")
    if not isinstance(split, dict):
        raise ValueError("Fold is missing the nested split record")
    body = dict(split)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError("Fold split signature is invalid")
    outer_train = _as_int_set(split.get("outer_train_indices"), "outer_train_indices")
    inner_train = _as_int_set(split.get("inner_train_indices"), "inner_train_indices")
    inner_validation = _as_int_set(
        split.get("inner_validation_indices"), "inner_validation_indices"
    )
    outer_test = _as_int_set(split.get("outer_test_indices"), "outer_test_indices")
    if not inner_train or not inner_validation or not outer_test:
        raise ValueError("Every nested split partition must be non-empty")
    if inner_train & inner_validation:
        raise ValueError("Inner train and validation overlap")
    if outer_train != inner_train | inner_validation:
        raise ValueError("Inner partitions do not exactly cover the outer train")
    if outer_train & outer_test:
        raise ValueError("Outer train and test overlap")
    if split.get("split_unit") != "trial":
        raise ValueError("Exp05 must split trials before creating windows")
    expected_inner_sha256 = canonical_sha256(
        {
            "inner_train_indices": split["inner_train_indices"],
            "inner_validation_indices": split["inner_validation_indices"],
        }
    )
    if split.get("inner_sha256") != expected_inner_sha256:
        raise ValueError("Inner split hash does not match its indices")
    expected_outer_sha256 = canonical_sha256(
        {
            "session": fold["session"],
            "fold": int(fold["outer_fold"]),
            "train_indices": split["outer_train_indices"],
            "test_indices": split["outer_test_indices"],
        }
    )
    if split.get("outer_sha256") != expected_outer_sha256:
        raise ValueError("Outer split hash does not match its indices")


def _validate_preprocessing(fold: dict[str, Any]) -> None:
    selection = fold.get("selection_preprocessing")
    refit = fold.get("refit_preprocessing")
    if not isinstance(selection, dict) or selection.get("fit_partition") != "inner_train":
        raise ValueError("Selection preprocessing was not fit on inner train only")
    if not isinstance(refit, dict) or refit.get("fit_partition") != "outer_train":
        raise ValueError("Refit preprocessing was not fit on outer train only")


def _validate_fold_metrics(fold: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(fold.get("y_true"), dtype=int)
    y_pred = np.asarray(fold.get("y_pred"), dtype=int)
    y_score = np.asarray(fold.get("y_score"), dtype=float)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or y_score.shape != y_true.shape:
        raise ValueError("Fold predictions have incompatible shapes")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Outer test fold does not contain both classes")
    if not set(np.unique(y_pred)) <= {0, 1}:
        raise ValueError("Fold predictions are not binary")
    if not np.isfinite(y_score).all() or np.any((y_score < 0.0) | (y_score > 1.0)):
        raise ValueError("Fold scores must be finite probabilities")
    recomputed = compute_metrics(y_true, y_pred, y_score)
    metrics = fold.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Fold is missing metrics")
    for name, value in recomputed.items():
        if not math.isclose(float(metrics[name]), value, abs_tol=1e-9):
            raise ValueError(f"Stored fold {name} does not match predictions")
    if fold.get("confusion_matrix") != _confusion_matrix(y_true, y_pred):
        raise ValueError("Stored fold confusion matrix does not match predictions")
    return y_true, y_pred, y_score


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    return [
        [
            int(np.sum((y_true == actual) & (y_pred == predicted)))
            for predicted in (0, 1)
        ]
        for actual in (0, 1)
    ]


def _validate_training_trajectory(
    fold: dict[str, Any], *, config_id: str, cell_seed: int, split_index: int
) -> None:
    contract = fold.get("selection_contract")
    if not isinstance(contract, dict):
        raise ValueError("Fold selection contract is missing")
    contract_body = dict(contract)
    contract_signature = str(contract_body.pop("sha256", ""))
    if contract_signature != canonical_sha256(contract_body):
        raise ValueError("Fold selection contract signature is invalid")
    expected_seed = int(cell_seed) * 100_000 + split_index
    if (
        contract_body.get("candidate") != config_id
        or int(contract_body.get("seed", -1)) != expected_seed
    ):
        raise ValueError("Fold selection contract has the wrong candidate or seed")
    max_epochs = int(contract_body.get("max_epochs", 0))
    patience = int(contract_body.get("patience", 0))
    if max_epochs < 1 or patience < 1 or int(contract_body.get("scheduler_t_max", 0)) != max_epochs:
        raise ValueError("Fold selection scheduler contract is invalid")
    history = fold.get("selection_history")
    if not isinstance(history, list) or not 1 <= len(history) <= max_epochs:
        raise ValueError("Fold selection history length is invalid")
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    previous_updates = 0
    for index, item in enumerate(history, start=1):
        if not isinstance(item, dict) or int(item.get("epoch", 0)) != index:
            raise ValueError("Fold selection epochs are not sequential")
        values = [
            float(item.get("train_loss", float("nan"))),
            float(item.get("validation_loss", float("nan"))),
            float(item.get("learning_rate", float("nan"))),
        ]
        if not np.isfinite(values).all():
            raise ValueError("Fold selection history contains non-finite values")
        updates = int(item.get("updates", 0))
        if updates <= previous_updates:
            raise ValueError("Fold selection update count is not increasing")
        previous_updates = updates
        validation_loss = values[1]
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = index
            stale = 0
        else:
            stale += 1
    if len(history) < max_epochs and stale < patience:
        raise ValueError("Fold selection stopped before its patience gate")
    selected_epoch = int(fold.get("selected_epoch", 0))
    if selected_epoch != best_epoch:
        raise ValueError("Selected epoch does not match the inner-validation trajectory")
    if not math.isclose(
        float(fold.get("best_inner_validation_loss", float("nan"))),
        best_loss,
        abs_tol=1e-12,
    ):
        raise ValueError("Best inner-validation loss does not match the trajectory")
    selection_body = {
        "candidate": config_id,
        "seed": expected_seed,
        "max_epochs": max_epochs,
        "patience": patience,
        "history": history,
        "selected_epoch": selected_epoch,
    }
    if fold.get("selection_trajectory_sha256") != canonical_sha256(selection_body):
        raise ValueError("Fold selection trajectory signature is invalid")

    refit_history = fold.get("refit_history")
    if not isinstance(refit_history, list) or len(refit_history) != selected_epoch:
        raise ValueError("Fold refit history does not match the selected epoch")
    refit_updates = int(fold.get("refit_updates", 0))
    previous_updates = 0
    for index, item in enumerate(refit_history, start=1):
        if not isinstance(item, dict) or int(item.get("epoch", 0)) != index:
            raise ValueError("Fold refit epochs are not sequential")
        values = [
            float(item.get("train_loss", float("nan"))),
            float(item.get("learning_rate", float("nan"))),
        ]
        if not np.isfinite(values).all():
            raise ValueError("Fold refit history contains non-finite values")
        if not math.isclose(
            values[1], float(history[index - 1]["learning_rate"]), abs_tol=1e-15
        ):
            raise ValueError("Fold refit changed the selected learning-rate trajectory")
        updates = int(item.get("updates", 0))
        if updates <= previous_updates:
            raise ValueError("Fold refit update count is not increasing")
        previous_updates = updates
    if int(fold.get("refit_epochs", 0)) != selected_epoch or refit_updates != previous_updates:
        raise ValueError("Fold refit budget does not match the selected epoch")
    state_sha256 = str(fold.get("refit_state_sha256", ""))
    if len(state_sha256) != 64:
        raise ValueError("Fold refit state signature is invalid")
    refit_body = {
        "candidate": config_id,
        "seed": expected_seed,
        "epochs": selected_epoch,
        "scheduler_t_max": max_epochs,
        "history": refit_history,
        "state_sha256": state_sha256,
        "updates": refit_updates,
    }
    if fold.get("refit_trajectory_sha256") != canonical_sha256(refit_body):
        raise ValueError("Fold refit trajectory signature is invalid")


def validate_cell(
    payload: dict[str, Any], expected: Exp05Cell, manifest: Exp05Manifest, *, allow_dirty: bool
) -> None:
    if payload.get("schema_version") != 1 or payload.get("status") != "completed":
        raise ValueError("Exp05 cell is incomplete")
    if payload.get("experiment") != "exp05" or payload.get("phase") != expected.phase:
        raise ValueError("Cell belongs to another experiment or phase")
    for name in ("cell_id", "kind", "config_id", "mechanism", "subject", "seed"):
        if payload.get(name) != getattr(expected, name):
            raise ValueError(f"Cell {name} differs from the signed manifest")
    subject = str(payload["subject"])
    if subject in HOLDOUT_SUBJECTS or subject not in DISCOVERY_SUBJECTS:
        raise ValueError("Cell violates the discovery/holdout boundary")
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("Cell fingerprint is missing")
    if fingerprint.get("cell_config_sha256") != expected.config_sha256:
        raise ValueError("Cell configuration hash differs from the manifest")
    if fingerprint.get("config_sha256") != manifest.config_sha256:
        raise ValueError("Cell experiment configuration differs from the manifest")
    if fingerprint.get("catalog_sha256") != manifest.catalog_sha256:
        raise ValueError("Cell catalog differs from the manifest")
    if fingerprint.get("manifest_sha256") != manifest.sha256:
        raise ValueError("Cell manifest hash differs from the signed manifest")
    if fingerprint.get("revision") != manifest.revision:
        raise ValueError("Cell revision differs from the signed manifest")
    if not allow_dirty and fingerprint.get("git_dirty") is not False:
        raise ValueError("Full Exp05 cell was produced from a dirty tree")
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Cell candidate payload is missing")
    candidate_body = dict(candidate)
    candidate_signature = str(candidate_body.pop("sha256", ""))
    if (
        candidate_signature != expected.config_sha256
        or candidate_signature != canonical_sha256(candidate_body)
    ):
        raise ValueError("Cell candidate signature differs from the signed manifest")
    if fingerprint.get("candidate_sha256") != candidate_signature:
        raise ValueError("Cell fingerprint candidate hash is invalid")
    reports = payload.get("fold_reports")
    if not isinstance(reports, list) or len(reports) != len(expected.folds):
        raise ValueError("Cell fold count differs from the manifest")
    if tuple(int(fold["outer_fold"]) for fold in reports) != expected.folds:
        raise ValueError("Cell outer folds differ from the manifest")
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_score: list[np.ndarray] = []
    expected_flags: set[str] = set()
    nested_split_hashes: list[str] = []
    outer_tests: list[set[int]] = []
    trial_universe: set[int] | None = None
    for split_index, fold in enumerate(reports):
        _validate_split(fold)
        _validate_preprocessing(fold)
        _validate_training_trajectory(
            fold,
            config_id=expected.config_id,
            cell_seed=int(expected.seed),
            split_index=split_index,
        )
        y_true, y_pred, y_score = _validate_fold_metrics(fold)
        recomputed_flags = validation_quality_flags(y_true.tolist(), y_pred.tolist())
        stored_flags = fold.get("quality_flags")
        if stored_flags != sorted(set(recomputed_flags)):
            raise ValueError("Fold collapse quality flags do not match predictions")
        expected_flags.update(recomputed_flags)
        split = fold["split"]
        outer_train = set(map(int, split["outer_train_indices"]))
        outer_test = set(map(int, split["outer_test_indices"]))
        fold_universe = outer_train | outer_test
        if trial_universe is None:
            trial_universe = fold_universe
        elif fold_universe != trial_universe:
            raise ValueError("Outer folds do not cover a common trial universe")
        if any(outer_test & previous for previous in outer_tests):
            raise ValueError("Outer test folds overlap")
        outer_tests.append(outer_test)
        nested_split_hashes.append(
            canonical_sha256(
                {
                    "session": fold["session"],
                    "fold": int(fold["outer_fold"]),
                    "inner_train_indices": split["inner_train_indices"],
                    "inner_validation_indices": split["inner_validation_indices"],
                    "outer_train_indices": split["outer_train_indices"],
                    "outer_test_indices": split["outer_test_indices"],
                    "outer_sha256": split["outer_sha256"],
                    "inner_sha256": split["inner_sha256"],
                }
            )
        )
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_score.append(y_score)
    recomputed = compute_metrics(
        np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_score)
    )
    aggregate = payload.get("aggregate_metrics")
    if not isinstance(aggregate, dict):
        raise ValueError("Cell aggregate metrics are missing")
    for name, value in recomputed.items():
        if not math.isclose(float(aggregate[name]), value, abs_tol=1e-9):
            raise ValueError(f"Stored aggregate {name} does not match predictions")
    concatenated_true = np.concatenate(all_true)
    concatenated_pred = np.concatenate(all_pred)
    concatenated_score = np.concatenate(all_score)
    if payload.get("y_true") != concatenated_true.astype(int).tolist():
        raise ValueError("Cell y_true differs from its fold reports")
    if payload.get("y_pred") != concatenated_pred.astype(int).tolist():
        raise ValueError("Cell y_pred differs from its fold reports")
    if not np.array_equal(
        np.asarray(payload.get("y_score"), dtype=float), concatenated_score
    ):
        raise ValueError("Cell y_score differs from its fold reports")
    if payload.get("confusion_matrix") != _confusion_matrix(
        concatenated_true, concatenated_pred
    ):
        raise ValueError("Cell confusion matrix does not match its fold reports")
    if payload.get("quality_flags") != sorted(expected_flags):
        raise ValueError("Cell quality flags do not equal the union of fold flags")
    if trial_universe is None:
        raise ValueError("Cell contains no trial universe")
    if len(expected.folds) == 5 and set().union(*outer_tests) != trial_universe:
        raise ValueError("Outer test folds do not partition the trial universe")
    if fingerprint.get("split_sha256") != canonical_sha256(nested_split_hashes):
        raise ValueError("Cell nested-split fingerprint is invalid")
    runtime = payload.get("runtime")
    environment = runtime.get("environment") if isinstance(runtime, dict) else None
    if not isinstance(environment, dict):
        raise ValueError("Cell runtime environment is missing")
    if fingerprint.get("environment_sha256") != canonical_sha256(environment):
        raise ValueError("Cell environment fingerprint is invalid")
    if fingerprint.get("device") != environment.get("device"):
        raise ValueError("Cell device fingerprint is invalid")
    for name in ("code_sha256", "data_sha256"):
        value = fingerprint.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Cell {name} is invalid")
    report = payload.get("parameter_report")
    if not isinstance(report, dict) or int(report.get("trainable", 0)) < 1:
        raise ValueError("Cell parameter report is invalid")


def audit_phase(
    output_dir: Path,
    manifest: Exp05Manifest,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    phase = manifest.phase
    expected_names = {f"{cell.cell_id}.json" for cell in manifest.cells}
    paths = sorted((output_dir / f"{phase}_cells").glob("*.json"))
    found = {path.name for path in paths}
    missing = sorted(expected_names - found)
    extras = sorted(found - expected_names)
    if missing or extras:
        raise ValueError(
            f"Exp05 {phase} incomplete: missing={missing[:5]} extras={extras[:5]}"
        )
    expected_by_name = {f"{cell.cell_id}.json": cell for cell in manifest.cells}
    code_hashes: set[str] = set()
    data_hashes: dict[str, set[str]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_cell(
            payload,
            expected_by_name[path.name],
            manifest,
            allow_dirty=allow_dirty,
        )
        fingerprint = payload["fingerprint"]
        code_hashes.add(str(fingerprint["code_sha256"]))
        data_hashes.setdefault(str(payload["subject"]), set()).add(
            str(fingerprint["data_sha256"])
        )
    if len(code_hashes) != 1:
        raise ValueError("Exp05 cells were produced from different code surfaces")
    if any(len(values) != 1 for values in data_hashes.values()):
        raise ValueError("Exp05 subject data hashes differ across cells")
    return {
        "status": "completed",
        "phase": phase,
        "expected": len(manifest.cells),
        "found": len(paths),
        "manifest_sha256": manifest.sha256,
        "code_sha256": next(iter(code_hashes)),
        "data_sha256_by_subject": {
            subject: next(iter(values)) for subject, values in sorted(data_hashes.items())
        },
    }


def validate_signed_decision(
    path: Path,
    *,
    phase: str,
    readout_path: Path | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing Exp05 {phase} decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body) or payload.get("phase") != phase:
        raise ValueError(f"Exp05 {phase} decision signature is invalid")
    if payload.get("confirmatory_p_value_used") is not False:
        raise ValueError("Discovery p-values cannot be used as confirmatory evidence")
    if readout_path is not None:
        if not readout_path.is_file():
            raise ValueError(f"Missing Exp05 {phase} readout: {readout_path}")
        readout = json.loads(readout_path.read_text(encoding="utf-8"))
        if payload.get("readout_sha256") != canonical_sha256(readout):
            raise ValueError(f"Exp05 {phase} decision is detached from its readout")
        if phase == "screen":
            from .analysis import select_screen_config_ids

            expected = select_screen_config_ids(
                readout, max_promoted=int(payload.get("max_promoted", 0))
            )
            if payload.get("selected_config_ids") != expected:
                raise ValueError("Exp05 screen decision differs from its frozen ranking")
        elif phase == "confirm":
            expected = next(
                (
                    str(item["config_id"])
                    for item in readout.get("ranking", [])
                    if item.get("eligible") is True
                ),
                None,
            )
            if payload.get("selected_config_id") != expected:
                raise ValueError("Exp05 final decision differs from its frozen ranking")
    if manifest_sha256 is not None and payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Exp05 {phase} decision is detached from its manifest")
    return payload
