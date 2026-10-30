from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.peterson_search_v12.common.io import write_json_atomic
from experiments.peterson_search_v12.exp08_fixed_convergence.audit import audit_phase
from experiments.peterson_search_v12.exp08_fixed_convergence.catalog import (
    experimental_candidates,
)
from experiments.peterson_search_v12.exp08_fixed_convergence.config import smoke_config
from experiments.peterson_search_v12.exp08_fixed_convergence.manifest import (
    build_screen_manifest,
    manifest_payload,
)
from experiments.peterson_search_v12.exp08_fixed_convergence.runner import run_phase


def test_screen_runner_shares_backbones_and_passes_strict_audit(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(808)
    recording = SimpleNamespace(
        x=rng.normal(size=(20, 15, 512)).astype(np.float32),
        y=np.asarray([0, 1] * 10, dtype=np.int64),
        sfreq=128.0,
        ch_names=tuple(f"EEG-{index:02d}" for index in range(15)),
        loader_bandpass_hz=(1.0, 40.0),
        sessions=np.asarray(["session_0"] * 20),
        data_sha256="d" * 64,
    )
    from experiments.peterson_search_v12.exp08_fixed_convergence import runner

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
    assert report["shared_backbone_pairs"] == 5

    uniform_path = next((tmp_path / "screen_cells").glob("*eegnet42-k32-uniform*.json"))
    lpwf_path = next((tmp_path / "screen_cells").glob("*eegnet42-k32-lpwf*.json"))
    uniform = json.loads(uniform_path.read_text())
    lpwf = json.loads(lpwf_path.read_text())
    assert uniform["fold_reports"][0]["fusion_trainable_parameters"] == 0
    assert lpwf["fold_reports"][0]["fusion_trainable_parameters"] == 6
    assert (
        uniform["fold_reports"][0]["backbone_state_sha256"]
        == lpwf["fold_reports"][0]["backbone_state_sha256"]
    )

    uniform["fold_reports"][0]["backbone_history"][0]["updates"] += 1
    body = {key: value for key, value in uniform.items() if key != "sha256"}
    from experiments.peterson_search_v12.common.identity import canonical_sha256

    uniform["sha256"] = canonical_sha256(body)
    uniform_path.write_text(json.dumps(uniform))
    with pytest.raises(ValueError, match="update total|trajectory"):
        audit_phase(
            tmp_path,
            config,
            catalog,
            phase="screen",
            revision="test-revision",
            parent_decision_sha256=parent_sha,
        )

