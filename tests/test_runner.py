from __future__ import annotations

import pytest

import deepbench.runner as runner
from deepbench.runner import _environment_identity, _run_identity, run_job


def _identity(
    epochs: int, *, data_sha256: str = "data-a", environment_sha256: str = "env-a"
) -> str:
    fingerprint, _ = _run_identity(
        dataset="MI-OpenBCI",
        protocol="within_split",
        condition="full",
        model="EEGNet",
        seed=0,
        subject="S02",
        ica_policy="none",
        epochs=epochs,
        device="cpu",
        save_weights=False,
        data_sha256=data_sha256,
        environment_sha256=environment_sha256,
    )
    return fingerprint


def test_run_fingerprint_changes_with_training_budget() -> None:
    assert _identity(1) != _identity(300)


def test_run_fingerprint_changes_with_data_or_environment() -> None:
    baseline = _identity(300)
    assert baseline != _identity(300, data_sha256="data-b")
    assert baseline != _identity(300, environment_sha256="env-b")


def test_environment_identity_records_scientific_dependencies() -> None:
    digest, versions = _environment_identity()
    assert len(digest) == 64
    assert {
        "python",
        "torch",
        "braindecode",
        "moabb",
        "mne",
        "numpy",
        "scipy",
        "BCI2kReader",
    } <= set(versions)


def test_git_revision_is_metadata_not_part_of_operational_fingerprint(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_git_revision", lambda: "revision-a")
    fingerprint_a, configuration_a = runner._run_identity(
        dataset="MI-OpenBCI",
        protocol="within_split",
        condition="full",
        model="EEGNet",
        seed=0,
        subject="S02",
        ica_policy="none",
        epochs=300,
        device="cpu",
        save_weights=False,
        data_sha256="data-a",
        environment_sha256="env-a",
    )
    monkeypatch.setattr(runner, "_git_revision", lambda: "revision-b")
    fingerprint_b, configuration_b = runner._run_identity(
        dataset="MI-OpenBCI",
        protocol="within_split",
        condition="full",
        model="EEGNet",
        seed=0,
        subject="S02",
        ica_policy="none",
        epochs=300,
        device="cpu",
        save_weights=False,
        data_sha256="data-a",
        environment_sha256="env-a",
    )
    assert configuration_a["git_revision"] == "revision-a"
    assert configuration_b["git_revision"] == "revision-b"
    assert fingerprint_a == fingerprint_b


def test_loso_rejects_single_subject_before_loading_data(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least two subjects"):
        run_job(
            dataset="MI-OpenBCI",
            models=["EEGNet"],
            protocols=["loso"],
            conditions=["full"],
            seeds=[0],
            subjects=["S02"],
            device="cpu",
            epochs=1,
            ica_policy="none",
            output_dir=tmp_path,
        )
