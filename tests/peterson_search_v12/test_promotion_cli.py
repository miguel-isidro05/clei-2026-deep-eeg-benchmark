from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.peterson_search_v12.common.catalogs import build_exp01_catalog
from experiments.peterson_search_v12.common.config import SearchConfig
from experiments.peterson_search_v12.common.promotion import rank_and_promote


def _payloads():
    config = SearchConfig()
    payloads = []
    for rank, candidate in enumerate(build_exp01_catalog(config)):
        for subject_index, subject in enumerate(config.subjects):
            accuracy = 0.95 - rank * 0.005 - subject_index * 0.001
            payloads.append(
                {
                    "config_id": candidate.config_id,
                    "candidate": {
                        "backbone": candidate.backbone,
                        "formulation": candidate.formulation,
                        "width": candidate.width,
                        "sha256": candidate.sha256,
                    },
                    "subject": subject,
                    "validation_metrics": {"accuracy": accuracy, "kappa": accuracy - 0.5},
                    "quality_flags": [],
                    "parameters": {"total": 1000 + rank},
                    "runtime": {
                        "elapsed_seconds": 10.0 + rank,
                        "inference_seconds_per_trial": 0.001 + rank * 1e-6,
                    },
                }
            )
    return payloads


def test_promotion_is_diverse_and_bounded():
    decision = rank_and_promote(_payloads(), expected_cells=216)
    promoted = decision["promoted"]
    assert len(promoted) == 12
    assert len({item["formulation"] for item in promoted}) >= 2
    assert all(
        sum(item["backbone"] == backbone for item in promoted) <= 3
        for backbone in SearchConfig().backbones
    )


def test_promotion_rejects_partial_results():
    with pytest.raises(ValueError, match="complete"):
        rank_and_promote(_payloads()[:-1], expected_cells=216)


def test_promotion_rejects_duplicate_subjects():
    payloads = _payloads()
    payloads[1]["subject"] = payloads[0]["subject"]
    with pytest.raises(ValueError, match="subjects"):
        rank_and_promote(payloads, expected_cells=216)


def test_promotion_records_but_excludes_collapsed_candidates():
    payloads = _payloads()
    collapsed_id = payloads[0]["config_id"]
    payloads[0]["quality_flags"] = ["validation_class_collapse"]
    decision = rank_and_promote(payloads, expected_cells=216)
    collapsed = next(item for item in decision["ranking"] if item["config_id"] == collapsed_id)
    assert collapsed["collapse_cells"] == 1
    assert collapsed["eligible_for_promotion"] is False
    assert collapsed_id not in {item["config_id"] for item in decision["promoted"]}


def test_cli_plan_only_does_not_load_data(tmp_path):
    result = subprocess.run(
        [
            "python",
            "scripts/run_peterson_search_v12.py",
            "--output-dir",
            str(tmp_path),
            "--plan-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "planned_candidates=36" in result.stdout
    assert "planned_cells=216" in result.stdout
    assert not (tmp_path / "cells").exists()


def test_cli_smoke_plan_covers_all_formulations_and_a_specialized_backbone(tmp_path):
    subprocess.run(
        [
            "python",
            "scripts/run_peterson_search_v12.py",
            "--output-dir",
            str(tmp_path),
            "--plan-only",
            "--smoke",
            "--epochs-override",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((tmp_path / "manifests" / "smoke-exp01.json").read_text(encoding="utf-8"))
    assert len(manifest["cells"]) == 4
    assert len({cell["cell_id"] for cell in manifest["cells"]}) == 4
    catalog = {
        item["config_id"]: item
        for item in json.loads(
            (tmp_path / "manifests" / "catalog-exp01.json").read_text(encoding="utf-8")
        )
    }
    selected = [catalog[cell["config_id"]] for cell in manifest["cells"]]
    assert {item["formulation"] for item in selected} == {
        "discriminative",
        "diffusion_energy",
        "hybrid",
    }
    assert "conformer_lite" in {item["backbone"] for item in selected}
    assert not (tmp_path / "manifests" / "exp01.json").exists()


def test_launcher_requires_exp01_and_two_devices():
    text = Path("run_peterson_search_cayetano.sh").read_text(encoding="utf-8")
    assert "PETERSON_SEARCH_EXPERIMENT:?Set PETERSON_SEARCH_EXPERIMENT=exp01" in text
    assert "PETERSON_SEARCH_DEVICES" in text
    assert "--num-shards 2 --shard-index 0" in text
    assert "--num-shards 2 --shard-index 1" in text
    assert "run_complete.json" not in text
