from __future__ import annotations

import pytest

from experiments.peterson_search_v12.exp03_quality_fbcnet.analysis import (
    _parity_report,
    _quality_factorial_contrasts,
    _rank_quality_summaries,
    _summary,
    select_recipe,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.catalog import build_recipe_catalog
from experiments.peterson_search_v12.exp03_quality_fbcnet.config import Exp03Config
from experiments.peterson_search_v12.exp03_quality_fbcnet.manifest import build_recipe_manifest
from experiments.peterson_search_v12.exp03_quality_fbcnet.runner import (
    _training_seed,
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


def test_recipe_seed_std_measures_variation_between_seed_means():
    payloads = [
        {
            "config_id": "R0-last-wd0",
            "subject": subject,
            "seed": seed,
            "validation_metrics": {"accuracy": accuracy, "kappa": 0.5},
            "quality_flags": [],
        }
        for seed in (0, 1)
        for subject, accuracy in (("S02", 0.6), ("S03", 0.9))
    ]
    summary = _summary(payloads, "R0-last-wd0")
    assert summary["seed_std"] == 0.0
    assert summary["subject_accuracy"] == {"S02": 0.6, "S03": 0.9}


def test_training_seed_is_shared_by_manual_and_baseline_paths():
    assert [_training_seed(4, split) for split in range(3)] == [4000, 4001, 4002]


def test_parity_gate_rejects_signed_cancellation_between_subjects():
    manual = {("S02", 0): 0.9, ("S03", 0): 0.5}
    reference = {("S02", 0): 0.7, ("S03", 0): 0.7}
    parity = _parity_report(manual, reference)
    assert parity["mean_accuracy_delta_manual_minus_v10"] == pytest.approx(0.0)
    assert parity["mean_absolute_paired_delta"] == pytest.approx(0.2)
    assert parity["pass"] is False


def test_quality_analysis_reports_matched_gate_and_dropout_contrasts():
    summaries = [
        {
            "gate_id": gate,
            "channel_dropout": dropout,
            "n_filters_spat": 32,
            "stride_factor": 4,
            "subject_accuracy": {"S02": accuracy, "S03": accuracy},
        }
        for gate, accuracy in (("G0", 0.70), ("G1", 0.72))
        for dropout in (0.0, 0.1)
    ]
    contrasts = {
        item["contrast"]: item for item in _quality_factorial_contrasts(summaries)
    }
    assert contrasts["gate_G1_minus_G0"]["mean_delta"] == pytest.approx(0.02)
    assert contrasts["channel_dropout_0.1_minus_0.0"]["mean_delta"] == pytest.approx(0.0)
    assert contrasts["interaction_G1_x_dropout_0.1"]["mean_delta"] == pytest.approx(0.0)
    assert "holm_adjusted_p" in contrasts["gate_G1_minus_G0"]


def test_quality_ranking_uses_guardrails_within_practical_tie():
    def candidate(config_id: str, accuracy: float, lower_quartile: float):
        return {
            "config_id": config_id,
            "eligible": True,
            "mean_accuracy": accuracy,
            "lower_quartile": lower_quartile,
            "mean_kappa": 0.6,
            "stress_floor": 0.5,
            "parameters": {"active": 5000},
            "mean_inference_seconds_per_trial": 0.001,
        }

    higher_mean = candidate("higher-mean", 0.801, 0.70)
    better_guardrail = candidate("better-guardrail", 0.800, 0.75)
    ranking = _rank_quality_summaries([higher_mean, better_guardrail])
    assert ranking[0]["config_id"] == "better-guardrail"
