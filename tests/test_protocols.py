from __future__ import annotations

import numpy as np

import deepbench.evaluation as evaluation
from deepbench.evaluation import _protocol_splits
from deepbench.types import SubjectRecording


def _recording() -> SubjectRecording:
    return SubjectRecording(
        dataset="synthetic",
        subject="1",
        x=np.zeros((40, 3, 64), dtype=np.float32),
        y=np.tile([0, 1], 20),
        sessions=np.repeat(["session_0", "session_1"], 20),
        sfreq=128.0,
        ch_names=("C3", "Cz", "C4"),
        task="binary",
    )


def test_cross_session_splits_never_mix_held_session() -> None:
    recording = _recording()
    splits = _protocol_splits(recording, "cross_session", seed=0)
    assert len(splits) == 2
    for train, test, report in splits:
        assert set(train).isdisjoint(set(test))
        assert set(recording.sessions[test]) == {report["held_session"]}
        assert report["held_session"] not in set(recording.sessions[train])


def test_within_session_predicts_each_trial_once_across_folds() -> None:
    recording = _recording()
    splits = _protocol_splits(recording, "within_session", seed=0)
    test_indices = np.concatenate([test for _, test, _ in splits])
    assert sorted(test_indices.tolist()) == list(range(40))
    for train, test, _ in splits:
        assert set(recording.sessions[train]) == set(recording.sessions[test])


def test_optimization_seed_does_not_change_split_or_ica_seed(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, int, int]] = []

    def fake_fit(recording, model, train_indices, test_indices, **kwargs):
        calls.append(
            (train_indices.copy(), test_indices.copy(), kwargs["model_seed"], kwargs["ica_seed"])
        )
        y_true = recording.y[test_indices]
        return y_true, y_true, y_true.astype(float), len(train_indices), {}

    monkeypatch.setattr(evaluation, "_fit_fold", fake_fit)
    recording = _recording()
    evaluation.run_subject_cell(
        recording,
        "EEGNet",
        protocol="within_split",
        condition="full",
        device="cpu",
        seed=0,
        epochs=1,
        ica_policy="none",
    )
    first = list(calls)
    calls.clear()
    evaluation.run_subject_cell(
        recording,
        "EEGNet",
        protocol="within_split",
        condition="full",
        device="cpu",
        seed=1,
        epochs=1,
        ica_policy="none",
    )
    second = list(calls)
    assert len(first) == len(second)
    for call_a, call_b in zip(first, second, strict=True):
        assert np.array_equal(call_a[0], call_b[0])
        assert np.array_equal(call_a[1], call_b[1])
        assert call_a[3] == call_b[3]
        assert call_a[2] != call_b[2]
