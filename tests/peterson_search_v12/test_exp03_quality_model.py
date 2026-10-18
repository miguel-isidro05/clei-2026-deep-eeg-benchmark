from __future__ import annotations

import numpy as np
import torch

from experiments.peterson_search_v12.exp03_quality_fbcnet.catalog import (
    build_quality_catalog,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.config import Exp03Config
from experiments.peterson_search_v12.exp03_quality_fbcnet.models import (
    ChannelDropout,
    ExactFBCNet,
    QualityConditionedFBCNet,
    fit_quality_profile,
)


def _candidate(gate_id: str, dropout: float = 0.0):
    return next(
        item
        for item in build_quality_catalog(Exp03Config())
        if item.gate_id == gate_id
        and item.channel_dropout == dropout
        and item.n_filters_spat == 32
        and item.stride_factor == 4
    )


def test_quality_profile_is_fit_from_training_data_only():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(8, 3, 64)).astype(np.float32)
    first_test = rng.normal(size=(4, 3, 64)).astype(np.float32)
    shifted_test = first_test + 1000.0

    profile = fit_quality_profile(train)
    first = profile.transform(first_test, include_amplitude=True)
    shifted = profile.transform(shifted_test, include_amplitude=True)

    assert profile.fit_partition == "train_only"
    assert profile.n_train_windows == 8
    assert not np.allclose(first, shifted)
    assert profile.sha256 == fit_quality_profile(train).sha256


def test_g0_and_initialised_quality_gates_are_exact_identity():
    rng = np.random.default_rng(11)
    train = rng.normal(size=(8, 3, 128)).astype(np.float32)
    profile = fit_quality_profile(train)
    x = torch.from_numpy(train[:4])

    for gate_id in ("G0", "G1", "G2"):
        model = QualityConditionedFBCNet(
            _candidate(gate_id),
            profile=profile,
            n_chans=3,
            n_times=128,
            sfreq=128.0,
        ).eval()
        with torch.no_grad():
            gated, weights = model.apply_quality_gate(x)
        assert torch.equal(gated, x)
        assert torch.equal(weights, torch.ones_like(weights))


def test_channel_dropout_is_train_only_deterministic_and_keeps_a_channel():
    x = torch.ones(6, 4, 32)
    dropout = ChannelDropout(0.9)
    dropout.train()
    first, first_mask = dropout(x, seed=19)
    second, second_mask = dropout(x, seed=19)
    assert torch.equal(first, second)
    assert torch.equal(first_mask, second_mask)
    assert torch.all(first_mask.sum(dim=1) >= 1)
    assert torch.any(first_mask == 0)

    dropout.eval()
    evaluated, evaluated_mask = dropout(x, seed=19)
    assert torch.equal(evaluated, x)
    assert torch.equal(evaluated_mask, torch.ones_like(evaluated_mask))


def test_g0_model_contains_no_inactive_auxiliary_parameters():
    rng = np.random.default_rng(13)
    train = rng.normal(size=(8, 3, 128)).astype(np.float32)
    model = QualityConditionedFBCNet(
        _candidate("G0"),
        profile=fit_quality_profile(train),
        n_chans=3,
        n_times=128,
        sfreq=128.0,
    )
    report = model.parameter_report()
    assert report["auxiliary"] == 0
    assert report["active"] == report["total"]
    logits = model(torch.from_numpy(train[:4]), dropout_seed=3)
    assert logits.shape == (4, 2)


def test_exact_fbcnet_reports_only_trainable_parameters_as_active():
    report = ExactFBCNet(n_chans=15, n_times=375, sfreq=125.0).parameter_report()
    assert report == {"total": 4994, "active": 4994, "auxiliary": 0, "frozen": 662}


def test_channel_dropout_is_applied_before_quality_gate():
    rng = np.random.default_rng(17)
    train = rng.normal(size=(8, 3, 128)).astype(np.float32)
    model = QualityConditionedFBCNet(
        _candidate("G1", dropout=0.2),
        profile=fit_quality_profile(train),
        n_chans=3,
        n_times=128,
        sfreq=128.0,
    ).train()
    observed_gate_inputs: list[torch.Tensor] = []
    original_gate = model.apply_quality_gate

    def capture_gate_input(x: torch.Tensor):
        observed_gate_inputs.append(x.detach().clone())
        return original_gate(x)

    model.apply_quality_gate = capture_gate_input
    _ = model(torch.ones(64, 3, 128), dropout_seed=19)
    missing_channels = torch.all(observed_gate_inputs[0] == 0, dim=-1)
    assert torch.any(missing_channels)
