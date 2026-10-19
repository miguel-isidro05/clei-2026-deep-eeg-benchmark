from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.catalog import (
    BASELINE_BANDS,
    baseline_candidate,
    build_catalog,
    candidate_by_id,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.mechanisms import (
    ConstrainedSincFilterBank,
    prepare_fit_context,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.models import (
    FBCMechanismModel,
    augment_batch,
    auxiliary_loss,
    build_baseline_model,
    build_model,
    optimizer_parameter_groups,
    sample_weights,
)


def _training_data():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(8, 3, 128)).astype(np.float32)
    y = np.asarray([0, 1] * 4, dtype=np.int64)
    return x, y


def _candidate(index: int):
    return candidate_by_id(f"exp05-e{index:02d}")


def test_catalog_has_twenty_signed_single_mechanism_candidates_and_a_control():
    catalog = build_catalog()
    assert len(catalog) == 20
    assert len({item.config_id for item in catalog}) == 20
    assert len({item.mechanism for item in catalog}) == 20
    assert all(item.params["n_filters_spat"] == 32 for item in catalog)
    assert all(item.params["stride_factor"] == 4 for item in catalog)
    assert not any(
        forbidden in item.mechanism
        for item in catalog
        for forbidden in ("router", "diffusion", "channel_dropout")
    )
    for item in catalog:
        payload = asdict(item)
        digest = payload.pop("sha256")
        assert digest == canonical_sha256(payload)
    control = baseline_candidate()
    assert candidate_by_id(control.config_id) == control
    assert control.params["bands"] == BASELINE_BANDS


def test_baseline_wrapper_is_numerically_exact_official_fbcnet():
    torch.manual_seed(3)
    model = build_baseline_model(n_chans=15, n_times=250, sfreq=125.0).eval()
    x = torch.randn(2, 15, 250)
    with torch.no_grad():
        wrapped = model(x)
        official = model.base(x)
    assert torch.equal(wrapped, official)
    assert model.parameter_report()["trainable"] == 4994


@pytest.mark.parametrize("candidate", build_catalog(), ids=lambda item: item.config_id)
def test_all_candidates_have_finite_common_forward_and_loss(candidate):
    x_numpy, y_numpy = _training_data()
    context = prepare_fit_context(candidate, x_numpy, y_numpy, 125.0)
    torch.manual_seed(7)
    model = build_model(
        candidate,
        n_chans=3,
        n_times=128,
        sfreq=125.0,
        fit_context=context,
    ).eval()
    x = torch.from_numpy(x_numpy[:4])
    y = torch.from_numpy(y_numpy[:4])
    logits, features = model(x, return_features=True)
    penalty = auxiliary_loss(model, logits, features, y, candidate)
    assert logits.shape == (4, 2)
    assert features.shape[0] == 4
    assert torch.isfinite(logits).all()
    assert torch.isfinite(penalty)


@pytest.mark.parametrize("index", (11, 12))
def test_window_models_aggregate_grouped_windows_to_trial_logits(index):
    candidate = _candidate(index)
    torch.manual_seed(11)
    model = build_model(candidate, n_chans=3, n_times=128, sfreq=125.0).eval()
    grouped = torch.randn(2, 6, 3, 128)
    with torch.no_grad():
        logits, features = model(grouped, return_features=True)
    assert logits.shape == (2, 2)
    assert features.shape[0] == 2
    assert model.last_window_weights is not None
    assert torch.allclose(model.last_window_weights.sum(dim=1), torch.ones(2))


def test_sinc_filters_remain_ordered_and_receive_gradients():
    candidate = _candidate(5)
    model = build_model(candidate, n_chans=3, n_times=128, sfreq=125.0)
    assert isinstance(model, FBCMechanismModel)
    bank = model.base.spectral_filtering
    assert isinstance(bank, ConstrainedSincFilterBank)
    low, high = bank.cutoffs()
    assert torch.all(low >= 4.0)
    assert torch.all(high <= 40.0)
    assert torch.all(high > low)
    assert torch.all(torch.diff(low) > 0.0)
    assert torch.all(torch.diff(high) > 0.0)
    with torch.no_grad():
        bank.raw_low.copy_(torch.linspace(20.0, -20.0, len(low)))
        bank.raw_high.copy_(torch.linspace(-20.0, 20.0, len(high)))
    perturbed_low, perturbed_high = bank.cutoffs()
    assert torch.all(torch.diff(perturbed_low) > 0.0)
    assert torch.all(torch.diff(perturbed_high) > 0.0)
    assert torch.all(perturbed_high - perturbed_low >= bank.min_bandwidth)
    model(torch.randn(2, 3, 128)).sum().backward()
    assert bank.raw_low.grad is not None
    assert bank.raw_high.grad is not None


@pytest.mark.parametrize("index", (8, 10, 16))
def test_alternative_heads_leave_no_trainable_off_path_classifier(index):
    candidate = _candidate(index)
    model = build_model(candidate, n_chans=3, n_times=128, sfreq=125.0)
    assert isinstance(model, FBCMechanismModel)
    assert isinstance(model.base.final_layer, torch.nn.Identity)
    x = torch.randn(2, 3, 128)
    model(x).sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_train_fitted_hooks_are_deterministic_and_marked_train_only():
    x, y = _training_data()
    for index in (4, 13, 14, 20):
        candidate = _candidate(index)
        first = prepare_fit_context(candidate, x, y, 125.0)
        second = prepare_fit_context(candidate, x, y, 125.0)
        assert first["fit_partition"] == "train_only"
        assert first.keys() == second.keys()
        for key in first:
            if isinstance(first[key], np.ndarray):
                assert np.array_equal(first[key], second[key])
            else:
                assert first[key] == second[key]


def test_csp_frozen_and_low_lr_contracts_are_distinct():
    x, y = _training_data()
    frozen_candidate = _candidate(13)
    frozen = build_model(
        frozen_candidate,
        n_chans=3,
        n_times=128,
        sfreq=125.0,
        fit_context=prepare_fit_context(frozen_candidate, x, y, 125.0),
    )
    assert isinstance(frozen, FBCMechanismModel)
    assert not frozen.base.spatial_conv[0].parametrizations.weight.original.requires_grad

    tune_candidate = _candidate(14)
    tune = build_model(
        tune_candidate,
        n_chans=3,
        n_times=128,
        sfreq=125.0,
        fit_context=prepare_fit_context(tune_candidate, x, y, 125.0),
    )
    groups = optimizer_parameter_groups(tune, tune_candidate, base_lr=1e-3)
    assert [group["lr"] for group in groups] == [1e-3, 1e-4]
    all_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(all_ids) == len(set(all_ids))


def test_ftsurrogate_is_deterministic_train_only_and_preserves_labels():
    candidate = _candidate(19)
    x = torch.randn(8, 3, 128)
    y = torch.tensor([0, 1] * 4)
    first, first_y = augment_batch(x, y, candidate, seed=31)
    second, second_y = augment_batch(x, y, candidate, seed=31)
    assert torch.equal(first, second)
    assert torch.equal(first_y, y)
    assert torch.equal(second_y, y)
    assert not torch.equal(first, x)


def test_quality_weights_are_fitted_on_train_bounded_and_mean_one():
    x, y = _training_data()
    candidate = _candidate(20)
    context = prepare_fit_context(candidate, x, y, 125.0)
    weights = sample_weights(x, candidate, context)
    assert weights.shape == (len(x),)
    assert weights.min() >= float(candidate.params["weight_min"]) - 1e-6
    assert weights.max() <= float(candidate.params["weight_max"]) + 1e-6
    assert np.isclose(weights.mean(), 1.0)
