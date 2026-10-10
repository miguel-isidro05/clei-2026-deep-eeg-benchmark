"""Inner-validation splits that keep outer test indices private."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepbench.types import SubjectRecording
from experiments.peterson_diffusion_v11.identity import canonical_sha256
from experiments.peterson_diffusion_v11.splits import train_validation_indices, within_session_splits


@dataclass(frozen=True)
class SearchSplit:
    session: str
    fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    outer_split_sha256: str
    outer_test_sha256: str
    sha256: str


def build_search_splits(
    recording: SubjectRecording, *, folds: tuple[int, ...], split_seed: int
) -> tuple[SearchSplit, ...]:
    if any(fold not in {0, 1, 2, 3, 4} for fold in folds):
        raise ValueError("Fold outside the five-fold Peterson protocol")
    selected: list[SearchSplit] = []
    for outer in within_session_splits(recording, seed=split_seed):
        if outer.fold not in folds:
            continue
        local_train, local_validation = train_validation_indices(
            recording.y[outer.train_indices],
            seed=split_seed + outer.fold,
        )
        train = tuple(int(value) for value in outer.train_indices[local_train])
        validation = tuple(int(value) for value in outer.train_indices[local_validation])
        test = tuple(int(value) for value in outer.test_indices)
        if set(train) & set(validation) or (set(train) | set(validation)) & set(test):
            raise RuntimeError("Search split partitions overlap")
        payload = {
            "session": outer.session,
            "fold": outer.fold,
            "train_indices": train,
            "validation_indices": validation,
            "outer_split_sha256": outer.sha256,
            "outer_test_sha256": canonical_sha256({"indices": test}),
        }
        selected.append(SearchSplit(**payload, sha256=canonical_sha256(payload)))
    if not selected:
        raise ValueError("No search splits matched the requested folds")
    return tuple(selected)

