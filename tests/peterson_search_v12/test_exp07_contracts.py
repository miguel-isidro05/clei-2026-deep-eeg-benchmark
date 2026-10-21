from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp07_eegnet_fusion.analysis import (
    analyze_confirm,
    analyze_screen,
    load_final_decision,
)
from experiments.peterson_search_v12.exp07_eegnet_fusion.catalog import (
    E11_REFERENCE_ID,
    FBC_CONTROL_ID,
    backbone_pairs,
    experimental_candidates,
)
from experiments.peterson_search_v12.exp07_eegnet_fusion.config import (
    Exp07Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp07_eegnet_fusion.lineage import load_exp06_parent
from experiments.peterson_search_v12.exp07_eegnet_fusion.manifest import (
    build_confirm_manifest,
    build_screen_manifest,
    manifest_from_payload,
    manifest_payload,
    shard_cells_by_subject,
)


def _signed_exp06_parent(root: Path) -> str:
    (root / "analysis").mkdir(parents=True)
    (root / "decisions").mkdir(parents=True)
    screen_manifest = {"phase": "screen", "cells": []}
    screen_manifest["sha256"] = canonical_sha256(screen_manifest)
    confirm_manifest = {"phase": "confirm", "cells": []}
    confirm_manifest["sha256"] = canonical_sha256(confirm_manifest)
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "screen-manifest.json").write_text(json.dumps(screen_manifest))
    (root / "manifests" / "confirm-manifest.json").write_text(json.dumps(confirm_manifest))
    screen_readout = {"ranking": []}
    (root / "analysis" / "screen-readout.json").write_text(json.dumps(screen_readout))
    screen_body = {
        "status": "no_promotion",
        "phase": "screen",
        "selected_config_ids": [],
        "max_promoted": 3,
        "reference_config_id": "exp06-e11-reference",
        "confirmatory_p_value_used": False,
        "manifest_sha256": screen_manifest["sha256"],
        "readout_sha256": canonical_sha256(screen_readout),
    }
    screen = {**screen_body, "sha256": canonical_sha256(screen_body)}
    (root / "decisions" / "screen-promotion.json").write_text(json.dumps(screen))
    readout = {"ranking": []}
    (root / "analysis" / "confirm-readout.json").write_text(json.dumps(readout))
    decision_body = {
        "status": "retain_e11",
        "phase": "confirm",
        "selected_config_id": "exp06-e11-reference",
        "reference_config_id": "exp06-e11-reference",
        "holdout_opened": False,
        "confirmatory_p_value_used": False,
        "screen_decision_sha256": screen["sha256"],
        "manifest_sha256": confirm_manifest["sha256"],
        "readout_sha256": canonical_sha256(readout),
    }
    decision = {**decision_body, "sha256": canonical_sha256(decision_body)}
    (root / "decisions" / "final-promotion.json").write_text(json.dumps(decision))
    completion_body = {
        "status": "completed",
        "experiment": "exp06",
        "revision": "abc123",
        "parent_decision_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "cells_sha256": "c" * 64,
        "audits": [],
        "final_decision_sha256": decision["sha256"],
        "holdout_opened": False,
    }
    completion = {**completion_body, "sha256": canonical_sha256(completion_body)}
    (root / "run_complete.json").write_text(json.dumps(completion))
    return str(decision["sha256"])


def test_scope_budgets_seeds_and_holdout_are_frozen() -> None:
    config = Exp07Config()
    assert config.screen_seeds == (10, 11, 12)
    assert config.confirm_seeds == (13, 14)
    assert config.expected_screen_cells == 216
    assert config.expected_confirm_cells(2) == 72
    with pytest.raises(ValueError, match="holdout"):
        replace(config, subjects=("S07",))


def test_parent_requires_completed_exp06_that_retained_e11(tmp_path: Path) -> None:
    signature = _signed_exp06_parent(tmp_path)
    assert load_exp06_parent(tmp_path)["sha256"] == signature
    path = tmp_path / "decisions" / "final-promotion.json"
    payload = json.loads(path.read_text())
    payload["selected_config_id"] = "exp06-e09"
    body = {key: value for key, value in payload.items() if key != "sha256"}
    payload["sha256"] = canonical_sha256(body)
    path.write_text(json.dumps(payload))
    completion_path = tmp_path / "run_complete.json"
    completion = json.loads(completion_path.read_text())
    completion_body = {key: value for key, value in completion.items() if key != "sha256"}
    completion_body["final_decision_sha256"] = payload["sha256"]
    completion_path.write_text(
        json.dumps({**completion_body, "sha256": canonical_sha256(completion_body)})
    )
    with pytest.raises(ValueError, match="e11"):
        load_exp06_parent(tmp_path)


def test_manifests_screen_every_pair_and_confirm_at_most_two_complete_pairs(tmp_path: Path) -> None:
    config = Exp07Config()
    candidates = experimental_candidates()
    screen = build_screen_manifest(
        config, candidates, revision="abc123", parent_decision_sha256="f" * 64
    )
    assert len(screen.cells) == 216
    assert len({cell.config_id for cell in screen.cells}) == 12
    left = shard_cells_by_subject(screen.cells, num_shards=2, shard_index=0)
    right = shard_cells_by_subject(screen.cells, num_shards=2, shard_index=1)
    assert {cell.subject for cell in left}.isdisjoint(cell.subject for cell in right)
    assert manifest_from_payload(manifest_payload(screen)) == screen
    promoted = tuple(list(backbone_pairs())[:2])
    confirm = build_confirm_manifest(
        config,
        candidates,
        promoted_backbones=promoted,
        revision="abc123",
        parent_decision_sha256="f" * 64,
        screen_decision_sha256="e" * 64,
    )
    assert len(confirm.cells) == 72
    expected_ids = {FBC_CONTROL_ID, E11_REFERENCE_ID}
    for backbone in promoted:
        expected_ids.update(backbone_pairs()[backbone])
    assert {cell.config_id for cell in confirm.cells} == expected_ids


def _write_cell(
    root: Path,
    phase: str,
    config_id: str,
    subject: str,
    seed: int,
    accuracy: float,
    kappa: float,
) -> None:
    candidate = next(item for item in (*experimental_candidates(),) if item.config_id == config_id)
    folder = root / f"{phase}_cells"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": f"{phase}__{config_id}__seed{seed}__{subject}",
        "config_id": config_id,
        "mechanism": candidate.mechanism,
        "kind": candidate.kind,
        "subject": subject,
        "seed": seed,
        "candidate": {"backbone": candidate.backbone},
        "aggregate_metrics": {"accuracy": accuracy, "kappa": kappa},
        "fold_reports": [],
        "quality_flags": [],
    }
    (folder / f"{payload['cell_id']}.json").write_text(json.dumps(payload))


def _write_control(
    root: Path,
    phase: str,
    config_id: str,
    subject: str,
    seed: int,
    accuracy: float,
    kappa: float,
) -> None:
    folder = root / f"{phase}_cells"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": f"{phase}__{config_id}__seed{seed}__{subject}",
        "config_id": config_id,
        "mechanism": config_id,
        "kind": "control",
        "subject": subject,
        "seed": seed,
        "candidate": {"backbone": "fbcnet"},
        "aggregate_metrics": {"accuracy": accuracy, "kappa": kappa},
        "fold_reports": [],
        "quality_flags": [],
    }
    (folder / f"{payload['cell_id']}.json").write_text(json.dumps(payload))


def test_analysis_selects_pairs_then_requires_final_gain_over_e11(tmp_path: Path) -> None:
    config = smoke_config()
    candidates = experimental_candidates()
    screen = build_screen_manifest(
        config, candidates, revision="abc123", parent_decision_sha256="f" * 64
    )
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "screen-manifest.json").write_text(
        json.dumps(manifest_payload(screen))
    )
    pair_backbone = next(iter(backbone_pairs()))
    uniform_id, lpwf_id = backbone_pairs()[pair_backbone]
    for subject in config.subjects:
        _write_control(tmp_path, "screen", FBC_CONTROL_ID, subject, 10, 0.70, 0.40)
        _write_control(tmp_path, "screen", E11_REFERENCE_ID, subject, 10, 0.72, 0.44)
        _write_cell(tmp_path, "screen", uniform_id, subject, 10, 0.70, 0.40)
        _write_cell(tmp_path, "screen", lpwf_id, subject, 10, 0.74, 0.48)
    report = analyze_screen(tmp_path, config)
    assert report["ranking"][0]["backbone"] == pair_backbone
    decision = json.loads((tmp_path / "decisions" / "screen-promotion.json").read_text())
    assert decision["selected_backbones"] == [pair_backbone]

    confirm = build_confirm_manifest(
        config,
        candidates,
        promoted_backbones=(pair_backbone,),
        revision="abc123",
        parent_decision_sha256="f" * 64,
        screen_decision_sha256=decision["sha256"],
    )
    (tmp_path / "manifests" / "confirm-manifest.json").write_text(
        json.dumps(manifest_payload(confirm))
    )
    for subject in config.subjects:
        _write_control(tmp_path, "confirm", FBC_CONTROL_ID, subject, 13, 0.70, 0.40)
        _write_control(tmp_path, "confirm", E11_REFERENCE_ID, subject, 13, 0.72, 0.44)
        _write_cell(tmp_path, "confirm", uniform_id, subject, 13, 0.70, 0.40)
        _write_cell(tmp_path, "confirm", lpwf_id, subject, 13, 0.75, 0.50)
    final = analyze_confirm(tmp_path, config)
    assert final["ranking"][0]["eligible"] is True
    decision = load_final_decision(tmp_path)
    assert decision["selected_config_id"] == lpwf_id
    assert decision["holdout_opened"] is False

