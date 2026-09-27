"""Subject-level evaluation protocols with identical inputs across models."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .config import SPLIT_SEED
from .io import read_json, write_json_atomic
from .metrics import compute_metrics
from .models import make_classifier, predict_scores, recipe_dict
from .preprocessing import augment_training, prepare_test_input, preprocess_split
from .types import CellResult, SubjectRecording


def _index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {str(label): int(np.sum(labels == label)) for label in (0, 1)}


def _protocol_splits(
    recording: SubjectRecording, protocol: str, seed: int
) -> list[tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    splits: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
    all_indices = np.arange(len(recording.y))
    sessions = np.unique(recording.sessions)
    if protocol == "cross_session":
        if len(sessions) < 2:
            raise ValueError(f"{recording.dataset}/{recording.subject} has only one session")
        for held_session in sessions:
            test = all_indices[recording.sessions == held_session]
            train = all_indices[recording.sessions != held_session]
            splits.append((train, test, {"held_session": str(held_session)}))
        return splits
    for session in sessions:
        session_indices = all_indices[recording.sessions == session]
        session_y = recording.y[session_indices]
        if protocol == "within_split":
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
            train_local, test_local = next(splitter.split(session_indices, session_y))
            splits.append(
                (
                    session_indices[train_local],
                    session_indices[test_local],
                    {"session": str(session), "fold": 0},
                )
            )
        elif protocol == "within_session":
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (train_local, test_local) in enumerate(
                splitter.split(session_indices, session_y)
            ):
                splits.append(
                    (
                        session_indices[train_local],
                        session_indices[test_local],
                        {"session": str(session), "fold": fold},
                    )
                )
        else:
            raise ValueError(f"Unsupported subject protocol: {protocol}")
    return splits


def _save_checkpoint(
    classifier,
    path: Path,
    *,
    model: str,
    recording: SubjectRecording,
    condition: str,
    seed: int,
    epochs: int,
    n_times: int,
    ica_policy: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": classifier.module_.state_dict(),
            "model": model,
            "dataset": recording.dataset,
            "task": recording.task,
            "n_chans": int(recording.x.shape[1]),
            "n_times": int(n_times),
            "sfreq": float(recording.sfreq),
            "condition": condition,
            "ica_policy": ica_policy,
            "seed": int(seed),
            "recipe": recipe_dict(model, epochs),
        },
        path,
    )


def _fit_fold(
    recording: SubjectRecording,
    model: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    condition: str,
    device: str,
    model_seed: int,
    ica_seed: int,
    epochs: int,
    ica_policy: str,
    checkpoint_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, object]]:
    processed = preprocess_split(
        recording.x[train_indices],
        recording.x[test_indices],
        sfreq=recording.sfreq,
        ch_names=recording.ch_names,
        seed=ica_seed,
        ica_policy=ica_policy,
    )
    x_train, y_train = augment_training(processed.x_train, recording.y[train_indices], condition)
    x_test = prepare_test_input(processed.x_test, condition)
    classifier = make_classifier(
        model,
        n_chans=x_train.shape[1],
        n_times=x_train.shape[-1],
        sfreq=recording.sfreq,
        device=device,
        seed=model_seed,
        epochs=epochs,
    )
    classifier.fit(x_train, y_train)
    if not classifier.history or int(classifier.history[-1, "train_batch_count"]) < 1:
        raise RuntimeError("Training completed without processing any batches")
    losses = np.asarray([row["train_loss"] for row in classifier.history], dtype=float)
    if not np.isfinite(losses).all():
        raise RuntimeError("Training produced a non-finite loss")
    y_pred, y_score = predict_scores(classifier, x_test)
    training_history = [
        {
            "epoch": int(index + 1),
            "train_loss": float(row["train_loss"]),
            "duration_seconds": float(row["dur"]),
            "train_batch_count": int(row["train_batch_count"]),
            "learning_rate": float(row["event_lr"]),
        }
        for index, row in enumerate(classifier.history)
        if all(key in row for key in ("train_loss", "dur", "train_batch_count", "event_lr"))
    ]
    if checkpoint_path is not None:
        _save_checkpoint(
            classifier,
            checkpoint_path,
            model=model,
            recording=recording,
            condition=condition,
            seed=model_seed,
            epochs=epochs,
            n_times=x_train.shape[-1],
            ica_policy=ica_policy,
        )
    report = {
        **processed.report,
        "train_index_sha256": _index_hash(train_indices),
        "test_index_sha256": _index_hash(test_indices),
        "n_train_trials": int(len(train_indices)),
        "n_train_examples": int(len(x_train)),
        "n_test_trials": int(len(test_indices)),
        "train_class_counts": _class_counts(recording.y[train_indices]),
        "test_class_counts": _class_counts(recording.y[test_indices]),
        "training_history": training_history,
    }
    return recording.y[test_indices], y_pred, y_score, len(x_train), report


def run_subject_cell(
    recording: SubjectRecording,
    model: str,
    *,
    protocol: str,
    condition: str,
    device: str,
    seed: int,
    epochs: int,
    ica_policy: str,
    checkpoint_dir: Path | None = None,
    fold_cache_dir: Path | None = None,
) -> CellResult:
    """Run all folds for one subject and aggregate predictions at trial level."""
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_score_all: list[np.ndarray] = []
    reports: list[dict[str, object]] = []
    n_train_trials = 0
    n_train_examples = 0
    splits = _protocol_splits(recording, protocol, SPLIT_SEED)
    for fold_index, (train_indices, test_indices, split_report) in enumerate(splits):
        fold_model_seed = seed * 1000 + fold_index
        fold_ica_seed = SPLIT_SEED * 1000 + fold_index
        checkpoint = None
        if checkpoint_dir is not None:
            checkpoint = checkpoint_dir / (
                f"{recording.dataset}__{protocol}__{condition}__{model}__"
                f"ica-{ica_policy}__seed{seed}__{recording.subject}__fold{fold_index}.pt"
            )
        fold_cache = fold_cache_dir / f"fold-{fold_index}.json" if fold_cache_dir else None
        if fold_cache is not None and fold_cache.exists():
            cached = read_json(fold_cache)
            y_true = np.asarray(cached["y_true"], dtype=np.int64)
            y_pred = np.asarray(cached["y_pred"], dtype=np.int64)
            y_score = np.asarray(cached["y_score"], dtype=float)
            train_examples = int(cached["train_examples"])
            report = cached["report"]
        else:
            y_true, y_pred, y_score, train_examples, report = _fit_fold(
                recording,
                model,
                train_indices,
                test_indices,
                condition=condition,
                device=device,
                model_seed=fold_model_seed,
                ica_seed=fold_ica_seed,
                epochs=epochs,
                ica_policy=ica_policy,
                checkpoint_path=checkpoint,
            )
            if fold_cache is not None:
                write_json_atomic(
                    fold_cache,
                    {
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "y_score": y_score,
                        "train_examples": train_examples,
                        "report": report,
                    },
                )
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        y_score_all.append(y_score)
        n_train_trials += len(train_indices)
        n_train_examples += train_examples
        reports.append(
            {
                **split_report,
                **report,
                "model_seed": fold_model_seed,
                "ica_seed": fold_ica_seed,
                "split_seed": SPLIT_SEED,
            }
        )
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_score = np.concatenate(y_score_all)
    return CellResult(
        dataset=recording.dataset,
        task=recording.task,
        protocol=protocol,
        condition=condition,
        ica_policy=ica_policy,
        model=model,
        subject=recording.subject,
        seed=seed,
        split_seed=SPLIT_SEED,
        metrics=compute_metrics(y_true, y_pred, y_score),
        y_true=y_true.astype(int).tolist(),
        y_pred=y_pred.astype(int).tolist(),
        y_score=y_score.astype(float).tolist(),
        n_train_trials=int(n_train_trials),
        n_train_examples=int(n_train_examples),
        n_test_trials=int(len(y_true)),
        fold_reports=reports,
    )


def run_loso_cell(
    recordings: dict[str, SubjectRecording],
    held_subject: str,
    model: str,
    *,
    condition: str,
    device: str,
    seed: int,
    epochs: int,
    ica_policy: str,
    checkpoint_dir: Path | None = None,
) -> CellResult:
    """Train on all other subjects and evaluate one unseen subject."""
    held = recordings[held_subject]
    train_subjects = [subject for subject in recordings if subject != held_subject]
    for subject in train_subjects:
        current = recordings[subject]
        if current.ch_names != held.ch_names or current.x.shape[1] != held.x.shape[1]:
            raise ValueError("LOSO requires an identical channel montage across subjects")
    x_train = np.concatenate([recordings[subject].x for subject in train_subjects])
    y_train = np.concatenate([recordings[subject].y for subject in train_subjects])
    synthetic = SubjectRecording(
        dataset=held.dataset,
        subject=held.subject,
        x=np.concatenate([x_train, held.x]),
        y=np.concatenate([y_train, held.y]),
        sessions=np.concatenate([np.full(len(y_train), "train_pool", dtype=object), held.sessions]),
        sfreq=held.sfreq,
        ch_names=held.ch_names,
        task=held.task,
    )
    train_indices = np.arange(len(y_train))
    test_indices = np.arange(len(y_train), len(y_train) + len(held.y))
    checkpoint = None
    if checkpoint_dir is not None:
        checkpoint = checkpoint_dir / (
            f"{held.dataset}__loso__{condition}__{model}__ica-{ica_policy}__"
            f"seed{seed}__{held_subject}.pt"
        )
    y_true, y_pred, y_score, train_examples, report = _fit_fold(
        synthetic,
        model,
        train_indices,
        test_indices,
        condition=condition,
        device=device,
        model_seed=seed,
        ica_seed=SPLIT_SEED,
        epochs=epochs,
        ica_policy=ica_policy,
        checkpoint_path=checkpoint,
    )
    report["train_subjects"] = train_subjects
    report["held_subject"] = held_subject
    report["model_seed"] = seed
    report["ica_seed"] = SPLIT_SEED
    return CellResult(
        dataset=held.dataset,
        task=held.task,
        protocol="loso",
        condition=condition,
        ica_policy=ica_policy,
        model=model,
        subject=held_subject,
        seed=seed,
        split_seed=None,
        metrics=compute_metrics(y_true, y_pred, y_score),
        y_true=y_true.astype(int).tolist(),
        y_pred=y_pred.astype(int).tolist(),
        y_score=y_score.astype(float).tolist(),
        n_train_trials=int(len(y_train)),
        n_train_examples=int(train_examples),
        n_test_trials=int(len(y_true)),
        fold_reports=[report],
    )
