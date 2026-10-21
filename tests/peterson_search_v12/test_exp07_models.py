from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp06_window_fusion.catalog import (
    E11_REFERENCE_ID as EXP06_E11_ID,
)
from experiments.peterson_search_v12.exp06_window_fusion.catalog import (
    FBC_CONTROL_ID as EXP06_FBC_ID,
)
from experiments.peterson_search_v12.exp06_window_fusion.catalog import (
    candidate_by_id as exp06_candidate_by_id,
)
from experiments.peterson_search_v12.exp06_window_fusion.models import (
    build_model as build_exp06_model,
)
from experiments.peterson_search_v12.exp07_eegnet_fusion.catalog import (
    E11_REFERENCE_ID,
    FBC_CONTROL_ID,
    all_candidates,
    backbone_pairs,
    candidate_by_id,
    experimental_candidates,
)
from experiments.peterson_search_v12.exp07_eegnet_fusion.models import build_model


def test_catalog_freezes_five_signed_uniform_lpwf_pairs_and_two_controls() -> None:
    candidates = experimental_candidates()
    pairs = backbone_pairs()
    assert len(candidates) == 10
    assert len(pairs) == 5
    assert len(all_candidates()) == 12
    assert {item.aggregation for item in candidates} == {"uniform", "lpwf"}
    assert all(len(pair) == 2 for pair in pairs.values())
    for item in all_candidates():
        payload = asdict(item)
        signature = payload.pop("sha256")
        assert signature == canonical_sha256(payload)


@pytest.mark.parametrize("candidate", experimental_candidates(), ids=lambda item: item.config_id)
def test_all_eegnet_family_candidates_produce_trial_logits_and_normalized_weights(
    candidate,
) -> None:
    torch.manual_seed(707)
    model = build_model(candidate, n_chans=15, n_times=256, sfreq=128.0).eval()
    grouped = torch.randn(2, 6, 15, 256)
    with torch.no_grad():
        logits, features = model(grouped, return_features=True)
    assert logits.shape == (2, 2)
    assert features.shape[0] == 2
    assert torch.isfinite(logits).all()
    assert model.last_window_weights is not None
    assert torch.allclose(model.last_window_weights.sum(dim=1), torch.ones(2))


@pytest.mark.parametrize("backbone, pair", backbone_pairs().items())
def test_lpwf_adds_exactly_six_parameters_and_starts_as_uniform(backbone, pair) -> None:
    uniform_id, lpwf_id = pair
    torch.manual_seed(17)
    uniform = build_model(
        candidate_by_id(uniform_id), n_chans=15, n_times=256, sfreq=128.0
    ).eval()
    torch.manual_seed(17)
    lpwf = build_model(candidate_by_id(lpwf_id), n_chans=15, n_times=256, sfreq=128.0).eval()
    assert lpwf.parameter_report()["trainable"] == uniform.parameter_report()["trainable"] + 6
    uniform_base = uniform.base.state_dict()
    lpwf_base = lpwf.base.state_dict()
    assert uniform_base.keys() == lpwf_base.keys()
    assert all(torch.equal(uniform_base[key], lpwf_base[key]) for key in uniform_base)
    grouped = torch.randn(2, 6, 15, 256)
    with torch.no_grad():
        uniform_logits = uniform(grouped)
        lpwf_logits = lpwf(grouped)
    assert torch.equal(uniform_logits, lpwf_logits)
    assert torch.allclose(lpwf.last_window_weights, torch.full((2, 6), 1 / 6))


@pytest.mark.parametrize(
    ("old_id", "new_id", "grouped"),
    [(EXP06_FBC_ID, FBC_CONTROL_ID, False), (EXP06_E11_ID, E11_REFERENCE_ID, True)],
)
def test_exp07_controls_are_exact_exp06_controls(old_id: str, new_id: str, grouped: bool) -> None:
    torch.manual_seed(71)
    old = build_exp06_model(
        exp06_candidate_by_id(old_id), n_chans=15, n_times=256, sfreq=128.0
    ).eval()
    torch.manual_seed(99)
    new = build_model(candidate_by_id(new_id), n_chans=15, n_times=256, sfreq=128.0).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(2, 6, 15, 256) if grouped else torch.randn(12, 15, 256)
    with torch.no_grad():
        assert torch.equal(old(x), new(x))

