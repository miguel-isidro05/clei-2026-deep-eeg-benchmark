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
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
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
        if item.backbone == backbone
        and item.formulation == "discriminative"
        and item.width == "compact"
    )
    wide = next(
        item
        for item in candidates
        if item.backbone == backbone
        and item.formulation == "discriminative"
        and item.width == "wide"
    )
    assert (
        parameter_report(compact, n_chans=15)["total"] < parameter_report(wide, n_chans=15)["total"]
    )
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


def test_diffusion_conditioning_interacts_with_the_input_signal():
    candidate = next(
        item
        for item in build_exp01_catalog(SearchConfig())
        if item.backbone == "residual"
        and item.formulation == "diffusion_energy"
        and item.width == "compact"
    )
    torch.manual_seed(23)
    model = make_candidate_model(candidate, n_chans=3).eval()
    first = torch.randn(2, 3, 64)
    second = torch.randn(2, 3, 64)
    timestep = torch.tensor([10, 60])
    mask = torch.ones(2, 3)
    label_zero = torch.zeros(2, dtype=torch.long)
    label_one = torch.ones(2, dtype=torch.long)
    with torch.no_grad():
        first_effect = model(
            first, timestep=timestep, label=label_one, channel_mask=mask
        ) - model(first, timestep=timestep, label=label_zero, channel_mask=mask)
        second_effect = model(
            second, timestep=timestep, label=label_one, channel_mask=mask
        ) - model(second, timestep=timestep, label=label_zero, channel_mask=mask)
    assert not torch.allclose(first_effect, second_effect, atol=1e-6, rtol=1e-5)


def _stem_input_channels(model) -> int:
    stem = model.backbone.stem
    if isinstance(stem, torch.nn.Conv1d):
        return stem.in_channels
    if hasattr(stem, "branches"):
        return stem.branches[0].in_channels
    if hasattr(stem, "temporal"):
        return stem.temporal[0][0].in_channels
    return int(stem.coefficients.shape[1])


@pytest.mark.parametrize("backbone", SearchConfig().backbones)
def test_miccai_style_conditioning_enters_before_the_first_convolution(backbone):
    candidates = build_exp01_catalog(SearchConfig())
    discriminative = next(
        item
        for item in candidates
        if item.backbone == backbone
        and item.formulation == "discriminative"
        and item.width == "compact"
    )
    diffusion = next(
        item
        for item in candidates
        if item.backbone == backbone
        and item.formulation == "diffusion_energy"
        and item.width == "compact"
    )
    discriminative_model = make_candidate_model(discriminative, n_chans=3)
    diffusion_model = make_candidate_model(diffusion, n_chans=3)
    assert _stem_input_channels(discriminative_model) == 3
    assert _stem_input_channels(diffusion_model) == 3 + diffusion.hidden
    assert parameter_report(discriminative, n_chans=3)["total"] < parameter_report(
        diffusion, n_chans=3
    )["total"]


def test_formulations_only_construct_the_heads_they_use():
    candidates = build_exp01_catalog(SearchConfig())
    discriminative = next(item for item in candidates if item.formulation == "discriminative")
    diffusion = next(item for item in candidates if item.formulation == "diffusion_energy")
    hybrid = next(item for item in candidates if item.formulation == "hybrid")
    discriminative_model = make_candidate_model(discriminative, n_chans=3)
    diffusion_model = make_candidate_model(diffusion, n_chans=3)
    hybrid_model = make_candidate_model(hybrid, n_chans=3)
    assert discriminative_model.time_mlp is None
    assert discriminative_model.class_embedding is None
    assert discriminative_model.noise_head is None
    assert diffusion_model.classifier is None
    assert diffusion_model.time_mlp is not None
    assert diffusion_model.class_embedding is not None
    assert diffusion_model.noise_head is not None
    assert hybrid_model.classifier is not None
    assert hybrid_model.time_mlp is not None
    assert hybrid_model.class_embedding is not None
    assert hybrid_model.noise_head is not None
