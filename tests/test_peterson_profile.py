from __future__ import annotations

from deepbench.config import PAPER_SEEDS, PETERSON_MODEL_NAMES
from deepbench.peterson_profile import (
    PETERSON_BLOCKS,
    expected_cell_count,
    expected_cell_names,
)


def test_peterson_profile_closes_declared_experiment_grid() -> None:
    assert set(PETERSON_MODEL_NAMES) == {
        "CSP+LDA",
        "EEGNet",
        "FBCNet",
        "ShallowConvNet",
        "EEGConformer",
    }
    assert {protocol for block in PETERSON_BLOCKS for protocol in block.protocols} == {
        "within_split",
        "within_session",
        "loso",
    }
    assert expected_cell_count() == 3250
    names = expected_cell_names()
    assert len(names) == len(set(names)) == 3250
    assert len(PAPER_SEEDS) == 5


def test_peterson_profile_has_input_and_compute_matched_window_controls() -> None:
    conditions = {
        condition
        for block in PETERSON_BLOCKS
        if "within_split" in block.protocols and block.ica_policy == "none"
        for condition in block.conditions
    }
    assert {"full", "center", "overlap", "nonoverlap", "center_x2", "center_x6"} <= conditions
    assert any(block.ica_policy == "kurtosis" for block in PETERSON_BLOCKS)
