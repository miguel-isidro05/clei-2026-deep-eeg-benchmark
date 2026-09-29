from __future__ import annotations

import pytest

from deepbench.config import PAPER_EPOCHS
from deepbench.paper_profile import paper_blocks


def test_primary_profile_preserves_all_three_original_protocols() -> None:
    blocks = paper_blocks("primary")
    full = next(
        block
        for block in blocks
        if block["dataset"] == "MI-OpenBCI" and block["conditions"] == ["full"]
    )
    assert full["dataset"] == "MI-OpenBCI"
    assert full["protocols"] == ["within_split", "within_session", "loso"]


def test_souza_profile_adds_cross_run_without_merging_tasks() -> None:
    blocks = paper_blocks("souza")
    full = next(block for block in blocks if block["conditions"] == ["full"])
    augmentation = next(block for block in blocks if block["conditions"] != ["full"])
    assert full == {
        "dataset": "Souza2023",
        "protocols": ["within_split", "within_session", "cross_session", "loso"],
        "conditions": ["full"],
        "ica_policy": "none",
    }
    assert augmentation["dataset"] == "Souza2023"
    assert augmentation["protocols"] == ["within_split"]


def test_primary_combines_both_low_cost_datasets_as_separate_blocks() -> None:
    assert {block["dataset"] for block in paper_blocks("primary")} == {
        "MI-OpenBCI",
        "Souza2023",
    }


def test_all_profile_is_limited_to_the_two_requested_datasets() -> None:
    assert {block["dataset"] for block in paper_blocks("all")} == {
        "MI-OpenBCI",
        "Souza2023",
    }


def test_external_phase_is_not_part_of_the_frozen_profile() -> None:
    with pytest.raises(ValueError, match="Unsupported paper phase"):
        paper_blocks("external")


def test_augmentation_profile_uses_compute_matched_controls() -> None:
    augmentations = [
        block for block in paper_blocks("primary") if block["conditions"] != ["full"]
    ]
    assert {block["dataset"] for block in augmentations} == {"MI-OpenBCI", "Souza2023"}
    assert all(
        block["conditions"] == ["center_x2", "nonoverlap", "center_x6", "overlap"]
        for block in augmentations
    )
    assert PAPER_EPOCHS == 300
