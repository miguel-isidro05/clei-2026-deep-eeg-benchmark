from __future__ import annotations

import pytest

from deepbench.runner import _run_identity, run_job


def _identity(epochs: int) -> str:
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
    )
    return fingerprint


def test_run_fingerprint_changes_with_training_budget() -> None:
    assert _identity(1) != _identity(300)


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
