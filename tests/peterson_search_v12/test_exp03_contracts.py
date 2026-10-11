from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from experiments.peterson_search_v12.exp03_quality_fbcnet.catalog import (
    build_quality_catalog,
    build_recipe_catalog,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.config import Exp03Config
from experiments.peterson_search_v12.exp03_quality_fbcnet.manifest import (
    build_quality_manifest,
    build_recipe_manifest,
    shard_cells,
)


def test_exp03_scope_and_holdout_are_frozen():
    config = Exp03Config()
    assert config.experiment == "exp03"
    assert config.subjects == ("S09", "S03", "S02", "S10", "S12", "S08")
    assert config.recipe_folds == (0, 1, 2, 3, 4)
    assert config.recipe_seeds == (0, 1, 2, 3, 4)
    assert config.quality_folds == (0, 2, 4)
    assert config.quality_seeds == (0,)
    assert config.condition == "overlap"
    with pytest.raises(ValueError, match="holdout"):
        replace(config, subjects=("S07",))
    with pytest.raises(ValueError, match="within_session overlap"):
        replace(config, condition="center")


def test_exp03_catalogs_match_the_predeclared_search_space():
    config = Exp03Config()
    recipes = build_recipe_catalog(config)
    quality = build_quality_catalog(config)
    assert len(recipes) == 5
    assert {recipe.recipe_id for recipe in recipes} == {
        "R0-last-wd0",
        "R1-best-wd0",
        "R2-last-wd1e-4",
        "R3-best-wd1e-4",
        "R4-exp02-replay",
    }
    assert recipes[0].trajectory_id == recipes[1].trajectory_id
    assert recipes[2].trajectory_id == recipes[3].trajectory_id
    assert len(quality) == 81
    assert len({candidate.config_id for candidate in quality}) == 81
    assert {candidate.gate_id for candidate in quality} == {"G0", "G1", "G2"}
    assert {candidate.channel_dropout for candidate in quality} == {0.0, 0.1, 0.2}
    assert {candidate.n_filters_spat for candidate in quality} == {16, 32, 48}
    assert {candidate.stride_factor for candidate in quality} == {2, 4, 8}


def test_exp03_manifests_have_exact_sizes_and_deterministic_shards():
    config = Exp03Config()
    recipes = build_recipe_catalog(config)
    recipe_manifest = build_recipe_manifest(config, recipes, revision="abc123")
    assert len(recipe_manifest.cells) == 186
    assert sum(cell.kind == "recipe" for cell in recipe_manifest.cells) == 150
    assert sum(cell.model_id == "CSP+LDA" for cell in recipe_manifest.cells) == 6
    assert sum(cell.model_id == "FBCNet-V10" for cell in recipe_manifest.cells) == 30

    decision_sha256 = "a" * 64
    candidates = build_quality_catalog(config)
    quality_manifest = build_quality_manifest(
        config,
        candidates,
        revision="abc123",
        recipe_decision_sha256=decision_sha256,
    )
    assert len(quality_manifest.cells) == 486
    assert all(cell.kind == "quality" for cell in quality_manifest.cells)
    assert quality_manifest.parent_decision_sha256 == decision_sha256
    left = shard_cells(quality_manifest.cells, num_shards=2, shard_index=0)
    right = shard_cells(quality_manifest.cells, num_shards=2, shard_index=1)
    assert len(left) == len(right) == 243
    assert {cell.cell_id for cell in left}.isdisjoint(cell.cell_id for cell in right)


def test_exp03_recipe_cli_plan_only_is_data_free(tmp_path):
    result = subprocess.run(
        [
            "python",
            "scripts/run_peterson_exp03_v12.py",
            "--output-dir",
            str(tmp_path),
            "--phase",
            "recipe",
            "--plan-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "planned_recipes=5" in result.stdout
    assert "planned_cells=186" in result.stdout
    manifest = json.loads(
        (tmp_path / "manifests" / "recipe-manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["cells"]) == 186
    assert not (tmp_path / "recipe_cells").exists()

