from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.peterson_search_v12.exp03_quality_fbcnet.models import fit_quality_profile
from experiments.peterson_search_v12.exp04_qcr_ssnet.catalog import QCRCandidate
from experiments.peterson_search_v12.exp04_qcr_ssnet.models import (
    QCRSSNet,
    corrupt_channels,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.objectives import qcr_loss


def _candidate(**overrides) -> QCRCandidate:
    payload = {
        "config_id": "qcr-test",
        "role": "qcr_full",
        "architecture": "qcr",
        "clean_n_filters_spat": 4,
        "robust_n_filters_spat": 6,
        "clean_stride_factor": 2,
        "robust_stride_factor": 4,
        "channel_dropout": 0.1,
        "router_mode": "learned",
        "use_quality_gate": True,
        "lambda_corrupt": 1.0,
        "lambda_consistency": 0.1,
        "lambda_router": 0.2,
        "lambda_specialization": 0.5,
        "sha256": "a" * 64,
    }
    payload.update(overrides)
    return QCRCandidate(**payload)


def test_channel_corruption_is_deterministic_and_never_drops_every_channel():
    x = torch.ones(8, 5, 32)
    first, first_mask, first_severity = corrupt_channels(x, probability=1.0, max_missing=2, seed=9)
    second, second_mask, second_severity = corrupt_channels(
        x, probability=1.0, max_missing=2, seed=9
    )
    assert torch.equal(first, second)
    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first_severity, second_severity)
    assert torch.all(first_mask.sum(dim=1) >= 3)
    assert torch.all((first_severity >= 0.5) & (first_severity <= 1.0))
    assert torch.equal(first, x * first_mask.unsqueeze(-1))


def test_qcr_forward_has_identity_gate_and_clean_router_initialization():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 5, 64)).astype(np.float32)
    profile = fit_quality_profile(x)
    model = QCRSSNet(
        _candidate(),
        profile=profile,
        n_chans=5,
        n_times=64,
        sfreq=128.0,
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(x[:3]))
    assert output.logits.shape == (3, 2)
    assert output.clean_logits.shape == output.robust_logits.shape == (3, 2)
    assert output.router_weight.shape == (3,)
    assert output.channel_weights.shape == (3, 5)
    assert torch.allclose(output.channel_weights, torch.ones_like(output.channel_weights))
    assert torch.all(output.router_weight < 0.05)
    report = model.parameter_report()
    assert 0 < report["active"] < 12_000
    assert report["frozen"] > 0


def test_oracle_router_uses_declared_severity():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(8, 5, 64)).astype(np.float32)
    profile = fit_quality_profile(x)
    model = QCRSSNet(
        _candidate(router_mode="oracle", use_quality_gate=False),
        profile=profile,
        n_chans=5,
        n_times=64,
        sfreq=128.0,
    )
    severity = torch.tensor([0.0, 0.5, 1.0])
    output = model(torch.from_numpy(x[:3]), oracle_severity=severity)
    assert torch.equal(output.router_weight, severity)
    assert torch.allclose(output.logits[0], output.clean_logits[0])
    assert torch.allclose(output.logits[2], output.robust_logits[2])


def test_qcr_loss_reports_every_predeclared_term_and_backpropagates():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(10, 5, 64)).astype(np.float32)
    profile = fit_quality_profile(x)
    candidate = _candidate()
    model = QCRSSNet(candidate, profile=profile, n_chans=5, n_times=64, sfreq=128.0)
    clean = torch.from_numpy(x[:4])
    corrupt, _, severity = corrupt_channels(clean, probability=1.0, max_missing=2, seed=3)
    labels = torch.tensor([0, 1, 0, 1])
    terms = qcr_loss(
        model(clean),
        model(corrupt),
        labels,
        severity,
        candidate,
    )
    assert set(terms.as_dict()) == {
        "total",
        "clean_classification",
        "corrupt_classification",
        "consistency",
        "router_supervision",
        "specialization",
    }
    assert torch.isfinite(terms.total)
    terms.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_qcr_loss_rejects_nonfinite_inputs():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(8, 5, 64)).astype(np.float32)
    profile = fit_quality_profile(x)
    candidate = _candidate()
    model = QCRSSNet(candidate, profile=profile, n_chans=5, n_times=64, sfreq=128.0)
    output = model(torch.from_numpy(x[:2]))
    output.logits[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        qcr_loss(
            output,
            model(torch.from_numpy(x[:2])),
            torch.tensor([0, 1]),
            torch.zeros(2),
            candidate,
        )
