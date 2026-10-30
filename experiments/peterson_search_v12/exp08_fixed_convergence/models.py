"""Exp08 backbone factory; aggregation is trained after the backbone is frozen."""

from __future__ import annotations

from torch import nn

from experiments.peterson_search_v12.exp06_window_fusion.models import WindowFusionModel
from experiments.peterson_search_v12.exp07_eegnet_fusion.models import EEGFamilyFusionModel

from .catalog import Exp08Candidate


def build_model(
    candidate: Exp08Candidate,
    *,
    n_chans: int,
    n_times: int,
    sfreq: float,
    fit_context: dict[str, object] | None = None,
) -> nn.Module:
    del fit_context
    if candidate.backbone == "fbcnet":
        return WindowFusionModel(candidate, n_chans=n_chans, n_times=n_times, sfreq=sfreq)
    return EEGFamilyFusionModel(candidate, n_chans=n_chans, n_times=n_times, sfreq=sfreq)

