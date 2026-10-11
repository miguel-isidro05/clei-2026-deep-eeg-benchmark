from __future__ import annotations

import pytest
import torch

from experiments.peterson_search_v12.exp02_factorial.catalog import build_candidate_catalog
from experiments.peterson_search_v12.exp02_factorial.config import Exp02Config
from experiments.peterson_search_v12.exp02_factorial.models import (
    make_model,
    parameter_report,
)
from experiments.peterson_search_v12.exp02_factorial.objectives import (
    candidate_loss,
    inference_probabilities,
)


@pytest.mark.parametrize("topology_id", [f"T{index}" for index in range(8)])
def test_all_exp02_topologies_forward_backward(topology_id):
    candidate = next(
        item
        for item in build_candidate_catalog(Exp02Config())
        if item.topology_id == topology_id and item.objective_id == "O5"
    )
    torch.manual_seed(17)
    model = make_model(candidate, n_chans=3, n_times=128, sfreq=128.0)
    x = torch.randn(4, 3, 128)
    y = torch.tensor([0, 1, 0, 1])
    mask = torch.ones(4, 3)
    loss = candidate_loss(model, x, y, mask, candidate=candidate, seed=5)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert parameter_report(model, candidate)["total"] > 0


def test_o0_is_a_capacity_control_without_auxiliary_execution():
    candidate = next(
        item
        for item in build_candidate_catalog(Exp02Config())
        if item.topology_id == "T0" and item.objective_id == "O0"
    )
    model = make_model(candidate, n_chans=3, n_times=128, sfreq=128.0)
    x = torch.randn(4, 3, 128)
    y = torch.tensor([0, 1, 0, 1])
    mask = torch.ones(4, 3)
    loss = candidate_loss(model, x, y, mask, candidate=candidate, seed=3)
    assert torch.equal(loss.total, loss.classification)
    loss.total.backward()
    assert all(parameter.grad is None for parameter in model.auxiliary_parameters())


def test_nested_energy_prefixes_and_fusion_are_deterministic():
    candidate = next(
        item
        for item in build_candidate_catalog(Exp02Config())
        if item.topology_id == "T0" and item.objective_id == "O5"
    )
    torch.manual_seed(23)
    model = make_model(candidate, n_chans=3, n_times=128, sfreq=128.0).eval()
    x = torch.randn(3, 3, 128)
    mask = torch.ones(3, 3)
    with torch.no_grad():
        first = inference_probabilities(model, x, mask, candidate=candidate, seed=9)
        second = inference_probabilities(model, x, mask, candidate=candidate, seed=9)
    assert first.keys() == second.keys()
    assert len(first) == 13
    for key in first:
        assert torch.equal(first[key], second[key])
        assert first[key].shape == (3, 2)
        assert torch.allclose(first[key].sum(dim=1), torch.ones(3), atol=1e-5)
    classifier_keys = [key for key in first if "alpha-1" in key]
    assert len(classifier_keys) == 1


def test_fbcnet_routes_include_compact_and_exact_capacity_controls():
    catalog = build_candidate_catalog(Exp02Config())
    compact = next(
        item for item in catalog if item.topology_id == "T6" and item.objective_id == "O0"
    )
    exact = next(item for item in catalog if item.topology_id == "T7" and item.objective_id == "O0")
    compact_model = make_model(compact, n_chans=15, n_times=384, sfreq=128.0)
    exact_model = make_model(exact, n_chans=15, n_times=384, sfreq=128.0)
    assert compact.family == exact.family == "fbcnet"
    assert (
        parameter_report(compact_model, compact)["total"]
        < parameter_report(exact_model, exact)["total"]
    )
