"""Public, hashed trial-level split contracts matching Peterson V10."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from deepbench.types import SubjectRecording

from .identity import canonical_sha256


@dataclass(frozen=True)
class FoldSplit:
    session: str
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    sha256: str


def _split_hash(
    session: str, fold: int, train_indices: np.ndarray, test_indices: np.ndarray
) -> str:
    return canonical_sha256(
        {
            "session": session,
            "fold": fold,
            "train_indices": np.asarray(train_indices, dtype=np.int64).tolist(),
            "test_indices": np.asarray(test_indices, dtype=np.int64).tolist(),
        }
    )


def within_session_splits(recording: SubjectRecording, *, seed: int) -> list[FoldSplit]:
    """Return V10-compatible five-fold splits without exposing a private V10 function."""
    splits: list[FoldSplit] = []
    all_indices = np.arange(len(recording.y), dtype=np.int64)
    for session_value in np.unique(recording.sessions):
        session = str(session_value)
        session_indices = all_indices[recording.sessions == session_value]
        labels = recording.y[session_indices]
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_local, test_local) in enumerate(splitter.split(session_indices, labels)):
            train_indices = session_indices[train_local]
            test_indices = session_indices[test_local]
            splits.append(
                FoldSplit(
                    session=session,
                    fold=fold,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    sha256=_split_hash(session, fold, train_indices, test_indices),
                )
            )
    return splits


def train_validation_indices(
    labels: np.ndarray, *, seed: int, validation_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Split an outer-training partition without inspecting its outer test fold."""
    indices = np.arange(len(labels), dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    train_indices, validation_indices = next(splitter.split(indices, labels))
    return indices[train_indices], indices[validation_indices]
