from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest

from deepbench.config import (
    DATASET_SPECS,
    SOUZA_CHANNELS,
    SOUZA_PUBLISHED_SUBJECTS,
    SOUZA_SUBJECTS,
)
from deepbench.datasets import (
    _load_mi_mat,
    _make_moabb_dataset,
    duplicate_file_groups,
    load_souza_subject,
    validate_dataset_dependencies,
)


@dataclass
class FakeAnnotations:
    onset: np.ndarray
    description: np.ndarray


class FakeSouzaRaw:
    def __init__(self, subject: str = "002") -> None:
        descriptions: list[str] = []
        onsets: list[float] = []
        for run in range(4):
            descriptions.append("NewRun")
            onsets.append(float(run * 200))
            for trial in range(40):
                descriptions.append("LeftExec" if trial % 2 == 0 else "RightExec")
                onsets.append(float(run * 200 + trial * 4 + 1))
        self.annotations = FakeAnnotations(
            onset=np.asarray(onsets),
            description=np.asarray(descriptions, dtype=object),
        )
        self.info = {"sfreq": 125.0, "subject_info": {"his_id": subject}}
        self.ch_names = list(SOUZA_CHANNELS)
        self.n_times = int((onsets[-1] + 4) * 125)

    def get_data(
        self, *, picks: list[int], start: int, stop: int
    ) -> np.ndarray:
        samples = np.arange(start, stop, dtype=np.float64)
        return np.stack([samples + channel for channel in picks])


def test_mi_openbci_loader_preserves_samples_channels_trials_axis_order(
    tmp_path, monkeypatch
) -> None:
    n_samples, n_channels, n_trials = 501, 15, 4
    source = np.arange(
        n_samples * n_channels * n_trials, dtype=np.float32
    ).reshape(n_samples, n_channels, n_trials)
    data_eeg = type(
        "FakeDataEEG",
        (),
        {
            "x": source,
            "y": np.asarray([1, 2, 1, 2]),
            "s": 125.0,
            "c": np.asarray([f"C{index}" for index in range(n_channels)]),
        },
    )()
    wrapper = type("FakeSubject", (), {"DataEEG": data_eeg})()
    monkeypatch.setattr(
        "scipy.io.loadmat",
        lambda *_args, **_kwargs: {"S02": wrapper},
    )

    trials, labels, sfreq, channels = _load_mi_mat(tmp_path / "S02.mat")

    assert trials.shape == (n_trials, n_channels, n_samples)
    assert np.array_equal(trials[2, 7], source[:, 7, 2])
    assert labels.tolist() == [1, 2, 1, 2]
    assert sfreq == 125.0
    assert channels == [f"C{index}" for index in range(n_channels)]


def test_tavakolan_is_exact_task_matched_and_multisession() -> None:
    spec = DATASET_SPECS["Tavakolan2017"]
    dataset = _make_moabb_dataset("Tavakolan2017")
    assert spec.events == ("rest", "right_hand")
    assert spec.task == "right_hand_vs_rest"
    assert spec.multi_session is True
    assert len(dataset.subject_list) == 12
    assert dataset.n_sessions == 4
    assert {"rest", "right_hand"}.issubset(dataset.event_id)


def test_tavakolan_dependency_check_fails_before_expensive_download(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="BCI2kReader"):
        validate_dataset_dependencies("Tavakolan2017")


def test_other_datasets_do_not_require_tavakolan_reader(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    validate_dataset_dependencies("Zhou2020")


def test_souza_spec_preserves_left_right_task_and_run_sessions() -> None:
    spec = DATASET_SPECS["Souza2023"]
    assert spec.events == ("left_hand", "right_hand")
    assert spec.task == "left_hand_vs_right_hand"
    assert spec.multi_session is True
    assert spec.trial_seconds == 3.0
    assert SOUZA_PUBLISHED_SUBJECTS == ("001", "002", "003", "004", "005", "006")
    assert SOUZA_SUBJECTS == ("002", "003", "004", "005", "006")


def test_souza_loader_uses_exec_events_and_new_run_sessions(tmp_path, monkeypatch) -> None:
    (tmp_path / "002.edf").write_bytes(b"synthetic-edf-placeholder")
    monkeypatch.setenv("SOUZA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("mne.io.read_raw_edf", lambda *_args, **_kwargs: FakeSouzaRaw())

    recording = load_souza_subject("002")

    assert recording.x.shape == (160, 16, 384)
    assert np.bincount(recording.y).tolist() == [80, 80]
    assert {
        session: int(np.sum(recording.sessions == session))
        for session in np.unique(recording.sessions)
    } == {"run_1": 40, "run_2": 40, "run_3": 40, "run_4": 40}
    assert recording.task == "left_hand_vs_right_hand"
    assert recording.loader_bandpass_hz is None


def test_souza_loader_rejects_excluded_subject_001(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SOUZA_DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="Unknown Souza2023 subject: 001"):
        load_souza_subject("001")


def test_duplicate_file_groups_reports_identical_subject_files(tmp_path) -> None:
    (tmp_path / "001.edf").write_bytes(b"duplicate")
    (tmp_path / "004.edf").write_bytes(b"duplicate")
    (tmp_path / "002.edf").write_bytes(b"unique")

    assert duplicate_file_groups(tmp_path, SOUZA_PUBLISHED_SUBJECTS) == [("001", "004")]
