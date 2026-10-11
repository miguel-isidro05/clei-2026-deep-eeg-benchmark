from __future__ import annotations

from experiments.peterson_search_v12.exp03_quality_fbcnet.analysis import select_recipe
from experiments.peterson_search_v12.exp03_quality_fbcnet.catalog import build_recipe_catalog
from experiments.peterson_search_v12.exp03_quality_fbcnet.config import Exp03Config
from experiments.peterson_search_v12.exp03_quality_fbcnet.manifest import build_recipe_manifest
from experiments.peterson_search_v12.exp03_quality_fbcnet.runner import (
    deterministic_stress_masks,
    recipe_execution_units,
)


def test_recipe_execution_units_share_paired_checkpoint_trajectories():
    config = Exp03Config()
    recipes = build_recipe_catalog(config)
    manifest = build_recipe_manifest(config, recipes, revision="abc123")
    units = recipe_execution_units(manifest.cells, recipes)
    assert len(units) == 126
    paired = [
        unit
        for unit in units
        if {cell.config_id for cell in unit} == {"R0-last-wd0", "R1-best-wd0"}
    ]
    assert len(paired) == 30
    assert all(len(unit) == 2 for unit in paired)
    assert all(len({cell.subject for cell in unit}) == 1 for unit in paired)
    assert all(len({cell.seed for cell in unit}) == 1 for unit in paired)


def test_stress_masks_cover_every_channel_and_are_deterministic():
    one = deterministic_stress_masks(n_chans=4, missing_channels=1)
    two = deterministic_stress_masks(n_chans=4, missing_channels=2)
    assert one == deterministic_stress_masks(n_chans=4, missing_channels=1)
    assert two == deterministic_stress_masks(n_chans=4, missing_channels=2)
    assert len(one) == len(two) == 4
    assert all(len(mask) == 4 and sum(mask) == 3 for mask in one)
    assert all(len(mask) == 4 and sum(mask) == 2 for mask in two)
    assert {index for mask in one for index, value in enumerate(mask) if value == 0} == {
        0,
        1,
        2,
        3,
    }


def test_recipe_selection_applies_collapse_margin_and_tie_break_rules():
    summaries = [
        {
            "recipe_id": "R0-last-wd0",
            "mean_accuracy": 0.82,
            "lower_quartile": 0.79,
            "mean_kappa": 0.64,
            "seed_std": 0.02,
            "collapse_count": 0,
        },
        {
            "recipe_id": "R1-best-wd0",
            "mean_accuracy": 0.824,
            "lower_quartile": 0.80,
            "mean_kappa": 0.65,
            "seed_std": 0.01,
            "collapse_count": 0,
        },
        {
            "recipe_id": "R2-last-wd1e-4",
            "mean_accuracy": 0.90,
            "lower_quartile": 0.81,
            "mean_kappa": 0.70,
            "seed_std": 0.01,
            "collapse_count": 1,
        },
    ]
    selected = select_recipe(summaries, practical_tie=0.005)
    assert selected["recipe_id"] == "R1-best-wd0"
