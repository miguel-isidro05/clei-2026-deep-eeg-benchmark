from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.catalog import (
    candidate_by_id as exp05_candidate_by_id,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.mechanisms import (
    prepare_fit_context,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.models import (
    build_model as build_exp05_model,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.models import sample_weights
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.runner import (
    _collapse_grouped_sample_weights,
    _grouped_fit_arrays,
)
from experiments.peterson_search_v12.exp06_window_fusion.catalog import (
    E11_REFERENCE_ID,
    FBC_CONTROL_ID,
    build_catalog,
    candidate_by_id,
    control_candidates,
)
from experiments.peterson_search_v12.exp06_window_fusion.models import (
    WindowFusionModel,
    build_model,
    sparsemax,
)


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(61)
    x = rng.normal(size=(8, 3, 128)).astype(np.float32)
    y = np.asarray([0, 1] * 4, dtype=np.int64)
    return x, y


def test_catalog_has_ten_signed_candidates_and_two_frozen_controls() -> None:
    catalog = build_catalog()
    controls = control_candidates()
    assert len(catalog) == 10
    assert [item.config_id for item in controls] == [FBC_CONTROL_ID, E11_REFERENCE_ID]
    assert len({item.config_id for item in (*controls, *catalog)}) == 12
    assert all(item.params["grouped_windows"] is True for item in catalog)
    for item in (*controls, *catalog):
        payload = asdict(item)
        signature = payload.pop("sha256")
        assert signature == canonical_sha256(payload)


@pytest.mark.parametrize("candidate", build_catalog(), ids=lambda item: item.config_id)
def test_all_candidates_produce_finite_trial_logits_and_normalized_weights(candidate) -> None:
    x, y = _training_data()
    fit_context = None
    if candidate.fit_hook == "csp":
        from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.mechanisms import (
            prepare_fit_context,
        )

        fit_context = prepare_fit_context(candidate, x, y, sfreq=125.0)
    model = build_model(
        candidate,
        n_chans=3,
        n_times=128,
        sfreq=125.0,
        fit_context=fit_context,
    )
    grouped = torch.from_numpy(x[:2]).unsqueeze(1).repeat(1, 6, 1, 1)
    logits, features = model(grouped, return_features=True)
    assert logits.shape == (2, 2)
    assert features.shape[0] == 2
    assert torch.isfinite(logits).all()
    assert model.last_window_weights is not None
    weights = model.last_window_weights
    assert torch.isfinite(weights).all()
    if weights.ndim == 2:
        assert torch.allclose(weights.sum(dim=1), torch.ones(weights.shape[0]))
    else:
        assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights.sum(dim=1)))


def test_e11_reference_is_exact_uniform_initialized_global_weighting() -> None:
    candidate = candidate_by_id(E11_REFERENCE_ID)
    model = build_model(candidate, n_chans=15, n_times=128, sfreq=125.0).eval()
    assert isinstance(model, WindowFusionModel)
    grouped = torch.randn(2, 6, 15, 128)
    with torch.no_grad():
        model(grouped)
    expected = torch.full((2, 6), 1 / 6)
    assert torch.allclose(model.last_window_weights, expected)
    assert model.parameter_report()["trainable"] == 5000


@pytest.mark.parametrize(
    ("old_id", "new_id", "grouped"),
    [
        ("exp05-fbc-control", FBC_CONTROL_ID, False),
        ("exp05-e11", E11_REFERENCE_ID, True),
    ],
)
def test_exp06_controls_are_numerically_identical_to_exp05(
    old_id: str, new_id: str, grouped: bool
) -> None:
    torch.manual_seed(606)
    old_model = build_exp05_model(
        exp05_candidate_by_id(old_id), n_chans=3, n_times=128, sfreq=125.0
    ).eval()
    torch.manual_seed(999)
    new_model = build_model(candidate_by_id(new_id), n_chans=3, n_times=128, sfreq=125.0).eval()
    new_model.load_state_dict(old_model.state_dict(), strict=True)
    x = torch.randn(2, 6, 3, 128) if grouped else torch.randn(12, 3, 128)
    with torch.no_grad():
        old_logits, old_features = old_model(x, return_features=True)
        new_logits, new_features = new_model(x, return_features=True)
    assert torch.equal(new_logits, old_logits)
    assert torch.equal(new_features, old_features)


def test_sparsemax_is_normalized_sparse_and_differentiable() -> None:
    logits = torch.tensor([[4.0, 1.0, -2.0], [0.0, 0.0, 0.0]], requires_grad=True)
    weights = sparsemax(logits, dim=1)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert int((weights[0] == 0).sum()) >= 1
    (weights.square().sum()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_window_dropout_is_train_only_and_never_removes_all_windows() -> None:
    candidate = candidate_by_id("exp06-e10")
    model = build_model(candidate, n_chans=3, n_times=128, sfreq=125.0)
    grouped = torch.randn(32, 6, 3, 128)
    model.train()
    torch.manual_seed(17)
    model(grouped)
    train_weights = model.last_window_weights.clone()
    assert torch.allclose(train_weights.sum(dim=1), torch.ones(32))
    assert torch.all((train_weights > 0).sum(dim=1) >= 1)
    model.eval()
    with torch.no_grad():
        model(grouped)
    assert torch.all(model.last_window_weights > 0)


def test_class_conditional_weights_have_one_simplex_per_class() -> None:
    candidate = candidate_by_id("exp06-e05")
    model = build_model(candidate, n_chans=3, n_times=128, sfreq=125.0).eval()
    with torch.no_grad():
        model(torch.randn(2, 6, 3, 128))
    assert model.last_window_weights.shape == (2, 6, 2)
    assert torch.allclose(model.last_window_weights.sum(dim=1), torch.ones(2, 2))


def test_grouped_csp_and_quality_hooks_fit_only_flattened_training_windows() -> None:
    x, y = _training_data()
    grouped = np.repeat(x[:, None, :, :], 6, axis=1)
    fit_x, fit_y, windows = _grouped_fit_arrays(grouped, y, grouped=True)
    assert fit_x.shape == (48, 3, 128)
    assert np.array_equal(fit_y, np.repeat(y, 6))
    assert windows == 6

    quality_candidate = candidate_by_id("exp06-e09")
    context = prepare_fit_context(quality_candidate, fit_x, fit_y, sfreq=125.0)
    crop_weights = sample_weights(fit_x, quality_candidate, context)
    trial_weights = _collapse_grouped_sample_weights(
        crop_weights, n_trials=len(grouped), windows_per_trial=windows
    )
    assert trial_weights.shape == (len(grouped),)
    assert np.isfinite(trial_weights).all()
