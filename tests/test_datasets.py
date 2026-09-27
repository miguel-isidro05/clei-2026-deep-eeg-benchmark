from __future__ import annotations

from deepbench.config import DATASET_SPECS
from deepbench.datasets import _make_moabb_dataset


def test_tavakolan_is_exact_task_matched_and_multisession() -> None:
    spec = DATASET_SPECS["Tavakolan2017"]
    dataset = _make_moabb_dataset("Tavakolan2017")
    assert spec.events == ("rest", "right_hand")
    assert spec.task == "right_hand_vs_rest"
    assert spec.multi_session is True
    assert len(dataset.subject_list) == 12
    assert dataset.n_sessions == 4
    assert {"rest", "right_hand"}.issubset(dataset.event_id)
