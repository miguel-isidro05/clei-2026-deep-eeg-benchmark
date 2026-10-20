from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.runner import _run_fold
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.splits import NestedFold
from experiments.peterson_search_v12.exp06_window_fusion.analysis import (
    analyze_confirm,
    analyze_screen,
)
from experiments.peterson_search_v12.exp06_window_fusion.audit import audit_phase
from experiments.peterson_search_v12.exp06_window_fusion.catalog import (
    all_candidates,
    build_catalog,
)
from experiments.peterson_search_v12.exp06_window_fusion.config import smoke_config
from experiments.peterson_search_v12.exp06_window_fusion.manifest import (
    build_confirm_manifest,
    build_screen_manifest,
)
from experiments.peterson_search_v12.exp06_window_fusion.models import build_model
from experiments.peterson_search_v12.exp06_window_fusion.runner import run_phase


@pytest.mark.parametrize("candidate", all_candidates(), ids=lambda item: item.config_id)
def test_every_exp06_candidate_completes_one_nested_cpu_fold(candidate) -> None:
    rng = np.random.default_rng(606)
    recording = SimpleNamespace(
        x=rng.normal(size=(20, 3, 500)).astype(np.float32),
        y=np.asarray([0, 1] * 10, dtype=np.int64),
        sfreq=125.0,
        ch_names=("C3", "Cz", "C4"),
        loader_bandpass_hz=(1.0, 40.0),
    )
    split = NestedFold(
        session="session_0",
        fold=0,
        inner_train_indices=tuple(range(12)),
        inner_validation_indices=tuple(range(12, 16)),
        outer_train_indices=tuple(range(16)),
        outer_test_indices=tuple(range(16, 20)),
        outer_sha256="a" * 64,
        inner_sha256="b" * 64,
        sha256="c" * 64,
    )
    cell = SimpleNamespace(seed=5)
    report, model = _run_fold(
        candidate,
        cell,
        recording,
        split,
        config=smoke_config(),
        device="cpu",
        split_index=0,
        model_builder=build_model,
    )
    assert len(report["y_true"]) == 4
    assert len(report["y_pred"]) == 4
    assert np.isfinite(np.asarray(report["y_score"])).all()
    if candidate.params["grouped_windows"]:
        diagnostics = report["model_diagnostics"]
        weights = np.asarray(diagnostics["window_weights"], dtype=float)
        assert np.isfinite(weights).all()
        assert np.allclose(weights.sum(axis=0), 1.0)
    assert model.parameter_report()["trainable"] > 0


def test_screen_runner_and_strict_auditor_complete_on_synthetic_recording(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(607)
    recording = SimpleNamespace(
        x=rng.normal(size=(20, 3, 500)).astype(np.float32),
        y=np.asarray([0, 1] * 10, dtype=np.int64),
        sfreq=125.0,
        ch_names=("C3", "Cz", "C4"),
        loader_bandpass_hz=(1.0, 40.0),
        sessions=np.asarray(["session_0"] * 20),
        data_sha256="d" * 64,
    )
    from experiments.peterson_search_v12.exp06_window_fusion import runner

    monkeypatch.setattr(runner, "load_subject", lambda *_args, **_kwargs: recording)
    monkeypatch.setattr(
        runner,
        "git_identity",
        lambda *_args, **_kwargs: {"revision": "test-revision", "dirty": False},
    )
    config = replace(smoke_config(), subjects=("S09",))
    catalog = build_catalog()
    parent_sha = "f" * 64
    manifest = build_screen_manifest(
        config,
        catalog,
        revision="test-revision",
        parent_decision_sha256=parent_sha,
    )
    from experiments.peterson_search_v12.common.io import write_json_atomic
    from experiments.peterson_search_v12.exp06_window_fusion.manifest import manifest_payload

    write_json_atomic(tmp_path / "manifests" / "screen-manifest.json", manifest_payload(manifest))
    summary = run_phase(
        tmp_path,
        config,
        catalog,
        manifest,
        phase="screen",
        device="cpu",
        num_shards=1,
        shard_index=0,
    )
    assert summary["completed"] == 12
    report = audit_phase(
        tmp_path,
        config,
        catalog,
        phase="screen",
        revision="test-revision",
        parent_decision_sha256=parent_sha,
    )
    assert report["complete"] is True
    assert report["found"] == 12

    analyze_screen(tmp_path, config)
    import json

    decision = json.loads((tmp_path / "decisions" / "screen-promotion.json").read_text())
    promoted = tuple(map(str, decision["selected_config_ids"]))
    confirm_manifest = build_confirm_manifest(
        config,
        catalog,
        promoted_ids=promoted,
        revision="test-revision",
        parent_decision_sha256=parent_sha,
        screen_decision_sha256=str(decision["sha256"]),
    )
    write_json_atomic(
        tmp_path / "manifests" / "confirm-manifest.json",
        manifest_payload(confirm_manifest),
    )
    confirm_summary = run_phase(
        tmp_path,
        config,
        catalog,
        confirm_manifest,
        phase="confirm",
        device="cpu",
        num_shards=1,
        shard_index=0,
    )
    assert confirm_summary["completed"] == 2 + len(promoted)
    confirm_audit = audit_phase(
        tmp_path,
        config,
        catalog,
        phase="confirm",
        revision="test-revision",
        parent_decision_sha256=parent_sha,
    )
    assert confirm_audit["complete"] is True
    final = analyze_confirm(tmp_path, config)
    assert final["reference"]["config_id"] == "exp06-e11-reference"
