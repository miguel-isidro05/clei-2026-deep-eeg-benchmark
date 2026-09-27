from __future__ import annotations

import importlib.util

import pytest

from deepbench.config import DATASET_SPECS
from deepbench.datasets import _make_moabb_dataset, validate_dataset_dependencies


def test_tavakolan_is_exact_task_matched_and_multisession() -> None:
    spec = DATASET_SPECS["Tavakolan2017"]
    dataset = _make_moabb_dataset("Tavakolan2017")
    assert spec.events == ("rest", "right_hand")
    assert spec.task == "right_hand_vs_rest"
    assert spec.multi_session is True
    assert len(dataset.subject_list) == 12
    assert dataset.n_sessions == 4
    assert {"rest", "right_hand"}.issubset(dataset.event_id)


def test_tavakolan_dependency_check_fails_before_expensive_download(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="BCI2kReader"):
        validate_dataset_dependencies("Tavakolan2017")


def test_other_datasets_do_not_require_tavakolan_reader(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    validate_dataset_dependencies("Zhou2020")
