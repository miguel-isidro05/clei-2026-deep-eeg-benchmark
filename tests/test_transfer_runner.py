from __future__ import annotations

import numpy as np
import torch

from deepbench.transfer import inner_validation_split, source_validation_subject
from deepbench.transfer_runner import _make_seeded_source_module


def test_source_validation_subject_rotates_predeclared_subjects() -> None:
    assert [source_validation_subject(seed) for seed in range(5)] == [
        "S02",
        "S03",
        "S04",
        "S05",
        "S06",
    ]


def test_within_session_inner_split_is_stratified_and_disjoint() -> None:
    outer_train = np.arange(40)
    labels = np.tile([0, 1], 20)
    sessions = np.full(40, "run_1", dtype=object)

    train, validation = inner_validation_split(
        outer_train,
        labels,
        sessions,
        protocol="within_session",
        seed=7,
    )

    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(outer_train)
    assert len(validation) == 8
    assert np.bincount(labels[validation], minlength=2).tolist() == [4, 4]


def test_cross_session_inner_split_holds_one_complete_training_run() -> None:
    outer_train = np.arange(60)
    labels = np.tile([0, 1], 30)
    sessions = np.repeat(["run_1", "run_2", "run_3"], 20).astype(object)

    train, validation = inner_validation_split(
        outer_train,
        labels,
        sessions,
        protocol="cross_session",
        seed=1,
    )

    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(outer_train)
    assert len(set(sessions[validation])) == 1
    assert set(sessions[validation]).isdisjoint(set(sessions[train]))


def test_fixed_inner_split_is_independent_of_optimization_seed() -> None:
    outer_train = np.arange(60)
    labels = np.tile([0, 1], 30)
    sessions = np.repeat(["run_1", "run_2", "run_3"], 20).astype(object)

    first = inner_validation_split(
        outer_train,
        labels,
        sessions,
        protocol="cross_session",
        seed=2026,
    )
    repeated = inner_validation_split(
        outer_train,
        labels,
        sessions,
        protocol="cross_session",
        seed=2026,
    )

    assert all(np.array_equal(left, right) for left, right in zip(first, repeated, strict=True))


def test_source_module_initialization_is_seeded() -> None:
    first = _make_seeded_source_module("EEGNet", 3)
    torch.rand(10)
    second = _make_seeded_source_module("EEGNet", 3)

    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
