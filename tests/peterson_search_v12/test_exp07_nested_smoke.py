from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from experiments.peterson_search_v12.common.io import write_json_atomic
from experiments.peterson_search_v12.exp07_eegnet_fusion.analysis import (
    analyze_confirm,
    analyze_screen,
)
from experiments.peterson_search_v12.exp07_eegnet_fusion.audit import audit_phase
from experiments.peterson_search_v12.exp07_eegnet_fusion.catalog import experimental_candidates
from experiments.peterson_search_v12.exp07_eegnet_fusion.config import smoke_config
from experiments.peterson_search_v12.exp07_eegnet_fusion.manifest import (
    build_confirm_manifest,
    build_screen_manifest,
    manifest_payload,
)
from experiments.peterson_search_v12.exp07_eegnet_fusion.runner import run_phase


def test_screen_runner_and_strict_auditor_complete_on_synthetic_recording(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(707)
    recording = SimpleNamespace(
        x=rng.normal(size=(20, 15, 512)).astype(np.float32),
        y=np.asarray([0, 1] * 10, dtype=np.int64),
        sfreq=128.0,
        ch_names=tuple(f"EEG-{index:02d}" for index in range(15)),
        loader_bandpass_hz=(1.0, 40.0),
        sessions=np.asarray(["session_0"] * 20),
        data_sha256="d" * 64,
    )
    from experiments.peterson_search_v12.exp07_eegnet_fusion import runner

    monkeypatch.setattr(runner, "load_subject", lambda *_args, **_kwargs: recording)
    monkeypatch.setattr(
        runner,
        "git_identity",
        lambda *_args, **_kwargs: {"revision": "test-revision", "dirty": False},
    )
    config = replace(smoke_config(), subjects=("S09",))
    catalog = experimental_candidates()
    parent_sha = "f" * 64
    manifest = build_screen_manifest(
        config,
        catalog,
        revision="test-revision",
        parent_decision_sha256=parent_sha,
    )
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
    promoted = tuple(map(str, decision["selected_backbones"]))
    confirm_manifest = build_confirm_manifest(
        config,
        catalog,
        promoted_backbones=promoted,
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
    assert confirm_summary["completed"] == 2 + 2 * len(promoted)
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
    assert final["reference"]["config_id"] == "exp07-e11-reference"

