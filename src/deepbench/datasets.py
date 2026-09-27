"""Dataset adapters for local MI-OpenBCI files and MOABB datasets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal

from .config import (
    DATASET_SPECS,
    MI_CHANNELS,
    MI_SUBJECTS,
    TARGET_SFREQ,
    resolve_mi_data_dir,
)
from .types import SubjectRecording


def _crop_or_pad(x: np.ndarray, target_samples: int) -> np.ndarray:
    """Center crop or edge-pad epochs to an exact temporal length."""
    n_times = x.shape[-1]
    if n_times == target_samples:
        return np.ascontiguousarray(x, dtype=np.float32)
    if n_times > target_samples:
        start = (n_times - target_samples) // 2
        return np.ascontiguousarray(x[..., start : start + target_samples], dtype=np.float32)
    left = (target_samples - n_times) // 2
    right = target_samples - n_times - left
    return np.pad(x, ((0, 0), (0, 0), (left, right)), mode="edge").astype(np.float32)


def _resample_epochs(x: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    """Polyphase-resample trialwise EEG without changing trial boundaries."""
    if np.isclose(source_sfreq, target_sfreq):
        return np.ascontiguousarray(x, dtype=np.float32)
    from fractions import Fraction

    ratio = Fraction(target_sfreq / source_sfreq).limit_denominator(1000)
    return scipy.signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=-1).astype(
        np.float32
    )


def _load_mi_mat(path: Path) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [key for key in mat if not key.startswith("__")]
    if not keys:
        raise ValueError(f"No subject variable found in {path}")
    obj = mat[keys[0]]
    if hasattr(obj, "DataEEG"):
        obj = obj.DataEEG
    x = np.asarray(obj.x, dtype=np.float32)
    y = np.asarray(obj.y, dtype=np.int64).reshape(-1)
    sfreq = float(np.asarray(obj.s).reshape(-1)[0])
    channels = [str(ch).strip() for ch in np.asarray(obj.c).reshape(-1)]
    if x.ndim != 3 or x.shape[2] != len(y):
        raise ValueError(f"Unexpected MI-OpenBCI arrays in {path}: x={x.shape}, y={y.shape}")
    return np.transpose(x, (2, 0, 1)), y, sfreq, channels


def load_mi_openbci_subject(subject: str) -> SubjectRecording:
    """Load one local MI-OpenBCI subject and map labels to Rest=0, MI=1."""
    if subject not in MI_SUBJECTS:
        raise ValueError(f"Unknown MI-OpenBCI subject: {subject}")
    x, raw_y, sfreq, source_channels = _load_mi_mat(resolve_mi_data_dir() / f"{subject}.mat")
    channel_index = {name: index for index, name in enumerate(source_channels)}
    missing = [name for name in MI_CHANNELS if name not in channel_index]
    if missing:
        raise ValueError(f"Missing MI-OpenBCI channels: {missing}")
    x = x[:, [channel_index[name] for name in MI_CHANNELS], :]
    x = _resample_epochs(x, sfreq, TARGET_SFREQ)
    x = _crop_or_pad(x, int(round(4.0 * TARGET_SFREQ)))
    y = np.where(raw_y == 1, 1, np.where(raw_y == 2, 0, -1)).astype(np.int64)
    if np.any(y < 0):
        raise ValueError(f"Unexpected MI-OpenBCI labels: {np.unique(raw_y)}")
    recording = SubjectRecording(
        dataset="MI-OpenBCI",
        subject=subject,
        x=x,
        y=y,
        sessions=np.full(len(y), "session_0", dtype=object),
        sfreq=TARGET_SFREQ,
        ch_names=MI_CHANNELS,
        task=DATASET_SPECS["MI-OpenBCI"].task,
    )
    recording.validate()
    return recording


def _make_moabb_dataset(name: str):
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    from moabb.datasets import BNCI2014_001, AlexMI, Zhou2020

    constructors = {
        "AlexMI": AlexMI,
        "BNCI2014_001": BNCI2014_001,
        "Zhou2020": Zhou2020,
    }
    try:
        return constructors[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported MOABB dataset: {name}") from exc


def available_subjects(dataset_name: str) -> list[str]:
    """Return stable subject identifiers without downloading data."""
    if dataset_name == "MI-OpenBCI":
        return list(MI_SUBJECTS)
    dataset = _make_moabb_dataset(dataset_name)
    return [str(subject) for subject in dataset.subject_list]


def load_moabb_subject(dataset_name: str, subject: str) -> SubjectRecording:
    """Load one binary MOABB subject with session metadata and common filtering."""
    if dataset_name not in DATASET_SPECS or dataset_name == "MI-OpenBCI":
        raise ValueError(f"Not a configured MOABB dataset: {dataset_name}")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    from moabb.paradigms import MotorImagery

    spec = DATASET_SPECS[dataset_name]
    dataset = _make_moabb_dataset(dataset_name)
    subject_value = next(
        (value for value in dataset.subject_list if str(value) == str(subject)), None
    )
    if subject_value is None:
        raise ValueError(f"Subject {subject} is not available in {dataset_name}")
    paradigm = MotorImagery(
        events=list(spec.events),
        n_classes=2,
        fmin=1.0,
        fmax=40.0,
        tmin=0.0,
        tmax=spec.trial_seconds,
        resample=TARGET_SFREQ,
    )
    epochs, labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=[subject_value],
        return_epochs=True,
    )
    x = epochs.get_data(copy=True).astype(np.float32)
    target_samples = int(round(spec.trial_seconds * TARGET_SFREQ))
    x = _crop_or_pad(x, target_samples)
    event_to_binary = {spec.events[0]: 0, spec.events[1]: 1}
    y = np.asarray([event_to_binary[str(label)] for label in labels], dtype=np.int64)
    sessions = metadata["session"].astype(str).to_numpy()
    recording = SubjectRecording(
        dataset=dataset_name,
        subject=str(subject),
        x=x,
        y=y,
        sessions=sessions,
        sfreq=TARGET_SFREQ,
        ch_names=tuple(epochs.ch_names),
        task=spec.task,
    )
    recording.validate()
    return recording


def load_subject(dataset_name: str, subject: str) -> SubjectRecording:
    """Dispatch to the local or MOABB dataset adapter."""
    if dataset_name == "MI-OpenBCI":
        return load_mi_openbci_subject(subject)
    return load_moabb_subject(dataset_name, subject)


def download_moabb_dataset(dataset_name: str, subjects: list[str] | None = None) -> None:
    """Download selected MOABB subjects through the official dataset API."""
    if dataset_name == "MI-OpenBCI":
        raise ValueError("MI-OpenBCI is local; set CLEI_DATA_DIR instead")
    dataset = _make_moabb_dataset(dataset_name)
    selected = dataset.subject_list
    if subjects:
        requested = set(map(str, subjects))
        selected = [subject for subject in selected if str(subject) in requested]
    dataset.download(subject_list=selected)
