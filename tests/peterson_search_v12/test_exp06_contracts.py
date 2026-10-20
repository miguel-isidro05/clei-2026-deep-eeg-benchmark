from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp06_window_fusion.analysis import (
    analyze_confirm,
    analyze_screen,
    load_final_decision,
)
from experiments.peterson_search_v12.exp06_window_fusion.catalog import (
    E11_REFERENCE_ID,
    FBC_CONTROL_ID,
    build_catalog,
)
from experiments.peterson_search_v12.exp06_window_fusion.config import (
    Exp06Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp06_window_fusion.lineage import (
    load_exp05_parent,
)
from experiments.peterson_search_v12.exp06_window_fusion.manifest import (
    build_confirm_manifest,
    build_screen_manifest,
    manifest_from_payload,
    manifest_payload,
    shard_cells_by_subject,
)


def _parent(root: Path) -> str:
    decisions = root / "decisions"
    decisions.mkdir(parents=True)
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    readout = {"ranking": [{"config_id": "exp05-e11", "eligible": True}]}
    (analysis / "confirm-readout.json").write_text(json.dumps(readout), encoding="utf-8")
    manifest_sha256 = "b" * 64
    body = {
        "status": "promoted",
        "phase": "confirm",
        "selected_config_id": "exp05-e11",
        "holdout_opened": False,
        "confirmatory_p_value_used": False,
        "manifest_sha256": manifest_sha256,
        "readout_sha256": canonical_sha256(readout),
    }
    payload = {**body, "sha256": canonical_sha256(body)}
    (decisions / "final-promotion.json").write_text(json.dumps(payload), encoding="utf-8")
    completion_body = {
        "status": "completed",
        "experiment": "exp05",
        "holdout_opened": False,
        "confirm_manifest_sha256": manifest_sha256,
        "final_decision_sha256": payload["sha256"],
    }
    completion = {**completion_body, "sha256": canonical_sha256(completion_body)}
    (root / "run_complete.json").write_text(json.dumps(completion), encoding="utf-8")
    return str(payload["sha256"])


def test_scope_budgets_new_seeds_and_holdout_are_frozen() -> None:
    config = Exp06Config()
    assert config.screen_seeds == (5, 6, 7)
    assert config.confirm_seeds == (8, 9)
    assert config.expected_screen_cells == 216
    assert config.expected_confirm_cells(3) == 60
    with pytest.raises(ValueError, match="holdout"):
        replace(config, subjects=("S07",))


def test_exp06_output_directories_are_ignored_by_git() -> None:
    root = Path(__file__).resolve().parents[2]
    patterns = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "results_peterson_exp06*/" in patterns


def test_parent_must_be_signed_exp05_e11_with_closed_holdout(tmp_path: Path) -> None:
    signature = _parent(tmp_path)
    assert load_exp05_parent(tmp_path)["sha256"] == signature
    payload_path = tmp_path / "decisions" / "final-promotion.json"
    payload = json.loads(payload_path.read_text())
    payload["selected_config_id"] = "exp05-e14"
    body = {key: value for key, value in payload.items() if key != "sha256"}
    payload["sha256"] = canonical_sha256(body)
    payload_path.write_text(json.dumps(payload))
    completion_path = tmp_path / "run_complete.json"
    completion = json.loads(completion_path.read_text())
    completion_body = {key: value for key, value in completion.items() if key != "sha256"}
    completion_body["final_decision_sha256"] = payload["sha256"]
    completion_path.write_text(
        json.dumps({**completion_body, "sha256": canonical_sha256(completion_body)})
    )
    with pytest.raises(ValueError, match="frozen ranking|exp05-e11"):
        load_exp05_parent(tmp_path)


def test_manifests_include_two_controls_and_are_subject_sharded(tmp_path: Path) -> None:
    parent_sha = _parent(tmp_path)
    config = Exp06Config()
    catalog = build_catalog()
    screen = build_screen_manifest(
        config, catalog, revision="abc123", parent_decision_sha256=parent_sha
    )
    assert len(screen.cells) == 216
    assert {cell.config_id for cell in screen.cells} == {
        FBC_CONTROL_ID,
        E11_REFERENCE_ID,
        *(item.config_id for item in catalog),
    }
    left = shard_cells_by_subject(screen.cells, num_shards=2, shard_index=0)
    right = shard_cells_by_subject(screen.cells, num_shards=2, shard_index=1)
    assert {cell.subject for cell in left}.isdisjoint(cell.subject for cell in right)
    assert len(left) + len(right) == 216
    assert manifest_from_payload(manifest_payload(screen)) == screen


def _write_cell(
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
        "kind": "control" if config_id in {FBC_CONTROL_ID, E11_REFERENCE_ID} else "candidate",
        "subject": subject,
        "seed": seed,
        "aggregate_metrics": {"accuracy": accuracy, "kappa": kappa},
        "fold_reports": [],
        "quality_flags": [],
    }
    (folder / f"{payload['cell_id']}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_analysis_ranks_against_e11_and_keeps_fbc_as_context(tmp_path: Path) -> None:
    config = smoke_config()
    parent_sha = _parent(tmp_path / "parent")
    catalog = build_catalog()
    screen = build_screen_manifest(
        config, catalog, revision="abc123", parent_decision_sha256=parent_sha
    )
    (tmp_path / "manifests").mkdir(parents=True)
    (tmp_path / "manifests" / "screen-manifest.json").write_text(
        json.dumps(manifest_payload(screen)), encoding="utf-8"
    )
    for subject in config.subjects:
        _write_cell(tmp_path, "screen", FBC_CONTROL_ID, subject, 5, 0.70, 0.40)
        _write_cell(tmp_path, "screen", E11_REFERENCE_ID, subject, 5, 0.72, 0.44)
        _write_cell(tmp_path, "screen", "exp06-e01", subject, 5, 0.74, 0.48)
    report = analyze_screen(tmp_path, config)
    assert report["reference"]["config_id"] == E11_REFERENCE_ID
    assert report["context_baseline"]["config_id"] == FBC_CONTROL_ID
    assert report["ranking"][0]["accuracy_delta"] == pytest.approx(0.02)
    decision = json.loads((tmp_path / "decisions" / "screen-promotion.json").read_text())
    assert decision["selected_config_ids"] == ["exp06-e01"]

    confirm = build_confirm_manifest(
        config,
        catalog,
        promoted_ids=("exp06-e01",),
        revision="abc123",
        parent_decision_sha256=parent_sha,
        screen_decision_sha256=str(decision["sha256"]),
    )
    (tmp_path / "manifests" / "confirm-manifest.json").write_text(
        json.dumps(manifest_payload(confirm)), encoding="utf-8"
    )
    for subject in config.subjects:
        _write_cell(tmp_path, "confirm", FBC_CONTROL_ID, subject, 8, 0.70, 0.40)
        _write_cell(tmp_path, "confirm", E11_REFERENCE_ID, subject, 8, 0.72, 0.44)
        _write_cell(tmp_path, "confirm", "exp06-e01", subject, 8, 0.74, 0.48)
    final = analyze_confirm(tmp_path, config)
    assert final["ranking"][0]["eligible"] is True
    final_decision = json.loads((tmp_path / "decisions" / "final-promotion.json").read_text())
    assert final_decision["selected_config_id"] == "exp06-e01"
    assert final_decision["holdout_opened"] is False
    assert load_final_decision(tmp_path)["sha256"] == final_decision["sha256"]

    readout_path = tmp_path / "analysis" / "confirm-readout.json"
    readout = json.loads(readout_path.read_text())
    readout["ranking"][0]["accuracy_delta"] = 99.0
    readout_path.write_text(json.dumps(readout))
    with pytest.raises(ValueError, match="detached from its readout"):
        load_final_decision(tmp_path)
