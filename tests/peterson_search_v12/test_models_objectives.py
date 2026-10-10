from __future__ import annotations

import pytest
import torch

from experiments.peterson_search_v12.common.catalogs import build_exp01_catalog
from experiments.peterson_search_v12.common.config import SearchConfig
from experiments.peterson_search_v12.common.models import make_candidate_model, parameter_report
from experiments.peterson_search_v12.common.objectives import (
    candidate_loss,
    predict_probabilities,
)


@pytest.mark.parametrize("candidate", build_exp01_catalog(SearchConfig()))
def test_all_exp01_models_forward_backward_and_normalize(candidate):
    torch.manual_seed(7)
    model = make_candidate_model(candidate, n_chans=3)
    x = torch.randn(4, 3, 64)
    y = torch.tensor([0, 1, 0, 1])
    mask = torch.ones(4, 3)
    terms = candidate_loss(model, x, y, mask, candidate=candidate, seed=11)
    assert torch.isfinite(terms.total)
    terms.total.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    probabilities = predict_probabilities(
        model, x, mask, candidate=candidate, seed=11, inference_k=2
    )
    assert probabilities.shape == (4, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-5)


@pytest.mark.parametrize("backbone", SearchConfig().backbones)
def test_compact_is_smaller_than_wide_and_deterministic(backbone):
    config = SearchConfig()
    candidates = build_exp01_catalog(config)
    compact = next(
        item
        for item in candidates
        if item.backbone == backbone and item.formulation == "discriminative" and item.width == "compact"
    )
    wide = next(
        item
        for item in candidates
        if item.backbone == backbone and item.formulation == "discriminative" and item.width == "wide"
    )
    assert parameter_report(compact, n_chans=15)["total"] < parameter_report(wide, n_chans=15)["total"]
    assert parameter_report(wide, n_chans=15)["total"] < 1_000_000
    torch.manual_seed(19)
    model = make_candidate_model(compact, n_chans=3).eval()
    x = torch.randn(2, 3, 64)
    mask = torch.ones(2, 3)
    with torch.no_grad():
        first = predict_probabilities(model, x, mask, candidate=compact, seed=3, inference_k=2)
        second = predict_probabilities(model, x, mask, candidate=compact, seed=3, inference_k=2)
    assert torch.equal(first, second)


def test_hybrid_zero_weights_remove_terms_exactly():
    candidate = next(
        item for item in build_exp01_catalog(SearchConfig()) if item.formulation == "hybrid"
    )
    model = make_candidate_model(candidate, n_chans=3)
    x = torch.randn(4, 3, 64)
    y = torch.tensor([0, 1, 0, 1])
    mask = torch.ones(4, 3)
    terms = candidate_loss(
        model,
        x,
        y,
        mask,
        candidate=candidate,
        seed=4,
        lambda_denoise=0.0,
        lambda_rank=0.0,
        lambda_consistency=0.0,
    )
    assert torch.equal(terms.total, terms.classification)
