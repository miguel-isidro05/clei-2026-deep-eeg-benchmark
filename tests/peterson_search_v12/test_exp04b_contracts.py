from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from experiments.peterson_search_v12.exp04b_safepath.catalog import (
    build_w1_catalog,
    build_w2_catalog,
)
from experiments.peterson_search_v12.exp04b_safepath.config import (
    Exp04bConfig,
    smoke_config,
)
from experiments.peterson_search_v12.exp04b_safepath.corruptions import (
    apply_corruption,
    build_stress_suite,
)
from experiments.peterson_search_v12.exp04b_safepath.manifest import (
    build_search_manifest,
    shard_cells_by_subject,
)
from experiments.peterson_search_v12.exp04b_safepath.models import SafePathQCR
from experiments.peterson_search_v12.exp04b_safepath.objectives import tail_risk
from experiments.peterson_search_v12.exp04b_safepath.statistics import (
    physical_mask_floor,
)


def test_exp04b_scope_and_budget_are_frozen_before_holdout() -> None:
    config = Exp04bConfig()
    assert config.experiment == "exp04b"
    assert config.subjects == ("S09", "S03", "S02", "S10", "S12", "S08")
    assert config.holdout_subjects == ("S07", "S04", "S05", "S06")
    assert config.total_training_jobs == 474
    assert config.wave_jobs == {"w1": 144, "w2": 144, "w3": 186}
    with pytest.raises(ValueError, match="holdout"):
        replace(config, subjects=("S07",))
    with pytest.raises(ValueError, match="within_session overlap"):
        replace(config, condition="center")


def test_catalogs_cover_the_predeclared_mechanisms_without_duplicates() -> None:
    w1 = build_w1_catalog()
    w2 = build_w2_catalog()
    assert len(w1) == 24
    assert len(w2) == 12
    assert len({candidate.config_id for candidate in w1 + w2}) == 36
    assert len({candidate.sha256 for candidate in w1 + w2}) == 36
    assert {candidate.router_mode for candidate in w2} == {
        "hard_quality",
        "learned_detached",
        "hybrid",
    }
    assert {candidate.fusion_mode for candidate in w2} == {
        "logit_residual",
        "probability_mixture",
    }
    assert {candidate.objective for candidate in w2} == {"cvar25", "group_dro"}


def test_manifest_has_474_jobs_and_subject_stable_two_gpu_shards() -> None:
    config = Exp04bConfig()
    selected = tuple(candidate.config_id for candidate in build_w2_catalog()[:3])
    manifest = build_search_manifest(config, selected_w2=selected, revision="abc123")
    assert len(manifest.cells) == 474
    assert sum(cell.wave == "w1" for cell in manifest.cells) == 144
    assert sum(cell.wave == "w2" for cell in manifest.cells) == 144
    assert sum(cell.wave == "w3" for cell in manifest.cells) == 186
    left = shard_cells_by_subject(manifest.cells, num_shards=2, shard_index=0)
    right = shard_cells_by_subject(manifest.cells, num_shards=2, shard_index=1)
    assert {cell.cell_id for cell in left}.isdisjoint(cell.cell_id for cell in right)
    assert len(left) + len(right) == 474
    left_subjects = {cell.subject for cell in left}
    right_subjects = {cell.subject for cell in right}
    assert left_subjects.isdisjoint(right_subjects)
    assert left_subjects | right_subjects == set(config.subjects)


class _ToyExpert(torch.nn.Module):
    def __init__(self, bias: tuple[float, float]) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias[None].expand(len(x), -1)


def test_safepath_hard_clean_bypass_is_exact_and_clean_expert_is_frozen() -> None:
    clean = _ToyExpert((1.0, -1.0))
    robust = _ToyExpert((-2.0, 2.0))
    model = SafePathQCR(
        clean,
        robust,
        router_mode="hard_quality",
        fusion_mode="logit_residual",
        freeze_clean=True,
    )
    x = torch.randn(4, 5, 32)
    clean_mask = torch.ones(4, 5)
    output = model(x, channel_mask=clean_mask)
    assert torch.equal(output.logits, output.clean_logits)
    assert torch.count_nonzero(output.router_weight) == 0
    assert all(not parameter.requires_grad for parameter in model.clean_expert.parameters())
    corrupt_mask = clean_mask.clone()
    corrupt_mask[:, 0] = 0
    corrupted = model(x, channel_mask=corrupt_mask)
    assert torch.all(corrupted.router_weight > 0)
    assert not torch.equal(corrupted.logits, corrupted.clean_logits)


def test_stress_suite_is_exhaustive_for_missing_channels_and_deterministic() -> None:
    suite = build_stress_suite(n_chans=15)
    missing_one = [item for item in suite if item.family == "missing" and item.severity == 1]
    missing_two = [item for item in suite if item.family == "missing" and item.severity == 2]
    assert len(missing_one) == 15
    assert len(missing_two) == 105
    assert len({item.condition_id for item in suite}) == len(suite)
    x = torch.ones(3, 15, 64)
    first, first_mask = apply_corruption(x, missing_one[0], seed=5)
    second, second_mask = apply_corruption(x, missing_one[0], seed=5)
    assert torch.equal(first, second)
    assert torch.equal(first_mask, second_mask)


def test_tail_objectives_emphasize_bad_groups() -> None:
    losses = torch.tensor([0.1, 0.2, 0.9, 1.2], requires_grad=True)
    mean = losses.mean()
    cvar = tail_risk(losses, "cvar25")
    group_dro = tail_risk(losses, "group_dro")
    assert cvar > mean
    assert group_dro > mean
    (cvar + group_dro).backward()
    assert losses.grad is not None
    assert torch.isfinite(losses.grad).all()


def test_worst_physical_mask_is_aggregated_across_folds_before_minimum() -> None:
    records = [
        {"mask_id": "C3", "fold": 0, "accuracy": 0.20},
        {"mask_id": "C3", "fold": 1, "accuracy": 0.90},
        {"mask_id": "T6", "fold": 0, "accuracy": 0.50},
        {"mask_id": "T6", "fold": 1, "accuracy": 0.50},
    ]
    result = physical_mask_floor(records)
    assert result["worst_mask_id"] == "T6"
    assert result["worst_mask_accuracy"] == pytest.approx(0.50)


def test_smoke_contract_is_small_but_keeps_two_subject_shards() -> None:
    config = smoke_config()
    assert config.subjects == ("S09", "S03")
    assert config.folds == (0,)
    assert config.seeds == (0,)
