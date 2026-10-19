"""Nested trial-level splits for Exp05.

The outer test fold is never used for epoch selection.  Every transform that
learns from data is fitted once on the inner-train partition for selection and
again on the complete outer-train partition for the final refit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepbench.types import SubjectRecording
from experiments.peterson_diffusion_v11.splits import (
    train_validation_indices,
    within_session_splits,
)
from experiments.peterson_search_v12.common.identity import canonical_sha256


@dataclass(frozen=True)
class NestedFold:
    session: str
    fold: int
    inner_train_indices: tuple[int, ...]
    inner_validation_indices: tuple[int, ...]
    outer_train_indices: tuple[int, ...]
    outer_test_indices: tuple[int, ...]
    outer_sha256: str
    inner_sha256: str
    sha256: str

    def assert_disjoint(self) -> None:
        inner_train = set(self.inner_train_indices)
        inner_validation = set(self.inner_validation_indices)
        outer_train = set(self.outer_train_indices)
        outer_test = set(self.outer_test_indices)
        if inner_train & inner_validation:
            raise RuntimeError("Inner train and validation partitions overlap")
        if outer_train & outer_test:
            raise RuntimeError("Outer train and test partitions overlap")
        if inner_train | inner_validation != outer_train:
            raise RuntimeError("Inner partitions do not exhaust the outer train partition")
        if (inner_train | inner_validation) & outer_test:
            raise RuntimeError("Outer test leaked into epoch selection")


def build_nested_folds(
    recording: SubjectRecording,
    *,
    split_seed: int,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    validation_fraction: float = 0.2,
) -> tuple[NestedFold, ...]:
    if not folds or any(fold not in {0, 1, 2, 3, 4} for fold in folds):
        raise ValueError("Exp05 folds must be a non-empty subset of 0..4")
    nested: list[NestedFold] = []
    for outer in within_session_splits(recording, seed=split_seed):
        if outer.fold not in folds:
            continue
        inner_local_train, inner_local_validation = train_validation_indices(
            recording.y[outer.train_indices],
            seed=split_seed + outer.fold,
            validation_fraction=validation_fraction,
        )
        outer_train = tuple(int(value) for value in outer.train_indices)
        outer_test = tuple(int(value) for value in outer.test_indices)
        inner_train = tuple(int(value) for value in outer.train_indices[inner_local_train])
        inner_validation = tuple(
            int(value) for value in outer.train_indices[inner_local_validation]
        )
        inner_payload = {
            "inner_train_indices": inner_train,
            "inner_validation_indices": inner_validation,
        }
        payload = {
            "session": outer.session,
            "fold": int(outer.fold),
            "inner_train_indices": inner_train,
            "inner_validation_indices": inner_validation,
            "outer_train_indices": outer_train,
            "outer_test_indices": outer_test,
            "outer_sha256": outer.sha256,
            "inner_sha256": canonical_sha256(inner_payload),
        }
        item = NestedFold(**payload, sha256=canonical_sha256(payload))
        item.assert_disjoint()
        if set(np.unique(recording.y[list(inner_train)])) != {0, 1}:
            raise RuntimeError("Inner train partition lost a class")
        if set(np.unique(recording.y[list(inner_validation)])) != {0, 1}:
            raise RuntimeError("Inner validation partition lost a class")
        if set(np.unique(recording.y[list(outer_test)])) != {0, 1}:
            raise RuntimeError("Outer test partition lost a class")
        nested.append(item)
    if not nested:
        raise ValueError("No outer folds matched the requested Exp05 folds")
    return tuple(nested)
