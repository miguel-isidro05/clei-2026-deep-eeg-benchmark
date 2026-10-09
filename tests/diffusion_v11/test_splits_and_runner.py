from __future__ import annotations

import numpy as np
import pytest

from deepbench.types import SubjectRecording
from experiments.peterson_diffusion_v11.config import ExperimentConfig
from experiments.peterson_diffusion_v11.manifests import build_manifest
from experiments.peterson_diffusion_v11.runner import cell_is_reusable, run_plan
from experiments.peterson_diffusion_v11.splits import within_session_splits


def _recording() -> SubjectRecording:
    rng = np.random.default_rng(1)
    recording = SubjectRecording(
        dataset="MI-OpenBCI",
        subject="S02",
        x=rng.normal(size=(20, 3, 512)).astype(np.float32),
        y=np.tile(np.array([0, 1]), 10),
        sessions=np.full(20, "session_0", dtype=object),
        sfreq=128.0,
        ch_names=("C3", "Cz", "C4"),
        task="motor_imagery_vs_rest",
        data_sha256="data-hash",
    )
    recording.validate()
    return recording


def test_splits_are_trial_level_disjoint_and_deterministic() -> None:
    first = within_session_splits(_recording(), seed=2026)
    second = within_session_splits(_recording(), seed=2026)
    assert len(first) == 5
    for left, right in zip(first, second, strict=True):
        assert set(left.train_indices).isdisjoint(left.test_indices)
        assert np.array_equal(left.train_indices, right.train_indices)
        assert np.array_equal(left.test_indices, right.test_indices)
        assert left.sha256 == right.sha256


def test_reusable_cell_requires_completed_matching_identity(tmp_path) -> None:
    config = ExperimentConfig(wave=1)
    cell = build_manifest(config)[0]
    destination = tmp_path / "cell.json"
    assert not cell_is_reusable(destination, cell, config.sha256, "run-hash")
    destination.write_text(
        '{"status":"completed","cell_id":"wrong","config_sha256":"x",'
        '"run_fingerprint":"y"}'
    )
    assert not cell_is_reusable(destination, cell, config.sha256, "run-hash")


def test_plan_only_does_not_load_data(tmp_path, monkeypatch) -> None:
    def fail_loader(*args, **kwargs):
        raise AssertionError("plan-only must not load Peterson files")

    monkeypatch.setattr("experiments.peterson_diffusion_v11.runner.load_subject", fail_loader)
    report = run_plan(
        ExperimentConfig(wave=1),
        output_dir=tmp_path,
        device="cpu",
        num_shards=2,
        shard_index=0,
        plan_only=True,
    )
    assert report == {"planned_total": 240, "planned_shard": 120}
    assert (tmp_path / "manifests" / "expected.json").exists()


def test_plan_only_can_limit_selected_cells_without_changing_manifest(tmp_path) -> None:
    report = run_plan(
        ExperimentConfig(wave=1),
        output_dir=tmp_path,
        device="cpu",
        num_shards=2,
        shard_index=0,
        max_cells_per_shard=1,
        plan_only=True,
    )
    assert report == {"planned_total": 240, "planned_shard": 1}
    manifest = (tmp_path / "manifests" / "expected.json").read_text()
    assert '"expected_cell_count": 240' in manifest


def test_plan_rejects_empty_smoke_shard(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_plan(
            ExperimentConfig(wave=1),
            output_dir=tmp_path,
            device="cpu",
            max_cells_per_shard=0,
            plan_only=True,
        )
