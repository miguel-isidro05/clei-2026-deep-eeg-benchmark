from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from experiments.peterson_search_v12.exp02_factorial.catalog import (
    build_candidate_catalog,
    build_inference_catalog,
)
from experiments.peterson_search_v12.exp02_factorial.config import Exp02Config
from experiments.peterson_search_v12.exp02_factorial.manifest import (
    build_experiment_manifest,
    shard_cells,
)


def test_exp02_scope_and_holdout_are_frozen():
    config = Exp02Config()
    assert config.experiment == "exp02"
    assert config.subjects == ("S09", "S03", "S02", "S10", "S12", "S08")
    assert config.folds == (0, 2, 4)
    assert config.epochs == 150
    assert config.patience == 35
    assert config.condition == "overlap"
    with pytest.raises(ValueError, match="holdout"):
        replace(config, subjects=("S07",))
    with pytest.raises(ValueError):
        replace(config, condition="center")


def test_exp02_catalogs_have_the_predeclared_factorial_size():
    config = Exp02Config()
    candidates = build_candidate_catalog(config)
    inference = build_inference_catalog(candidates)
    assert len(candidates) == 48
    assert len({candidate.config_id for candidate in candidates}) == 48
    assert {candidate.topology_id for candidate in candidates} == {
        f"T{index}" for index in range(8)
    }
    assert {candidate.objective_id for candidate in candidates} == {
        f"O{index}" for index in range(6)
    }
    assert len(inference) == 528
    assert len({recipe.inference_id for recipe in inference}) == 528
    assert all(
        recipe.alpha == 1.0 and recipe.k == 0 for recipe in inference if recipe.objective_id == "O0"
    )
    assert len([recipe for recipe in inference if recipe.config_id == "exp02-cfg-002"]) == 13


def test_exp02_manifest_contains_candidates_and_exact_split_baselines():
    config = Exp02Config()
    candidates = build_candidate_catalog(config)
    manifest = build_experiment_manifest(config, candidates, revision="abc123")
    assert len(manifest.cells) == 300
    assert sum(cell.kind == "candidate" for cell in manifest.cells) == 288
    assert sum(cell.kind == "baseline" for cell in manifest.cells) == 12
    assert {cell.model_id for cell in manifest.cells if cell.kind == "baseline"} == {
        "CSP+LDA",
        "FBCNet",
    }
    left = shard_cells(manifest.cells, num_shards=2, shard_index=0)
    right = shard_cells(manifest.cells, num_shards=2, shard_index=1)
    assert len(left) == len(right) == 150
    assert {cell.cell_id for cell in left}.isdisjoint(cell.cell_id for cell in right)


def test_exp02_cli_plan_only_is_data_free(tmp_path):
    result = subprocess.run(
        [
            "python",
            "scripts/run_peterson_exp02_v12.py",
            "--output-dir",
            str(tmp_path),
            "--plan-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "planned_candidates=48" in result.stdout
    assert "planned_inference_recipes=528" in result.stdout
    assert "planned_cells=300" in result.stdout
    manifest = json.loads(
        (tmp_path / "manifests" / "experiment-exp02.json").read_text(encoding="utf-8")
    )
    assert len(manifest["cells"]) == 300
    assert not (tmp_path / "cells").exists()
