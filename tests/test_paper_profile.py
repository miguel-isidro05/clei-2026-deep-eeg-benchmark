from __future__ import annotations

from deepbench.paper_profile import paper_blocks


def test_primary_profile_preserves_all_three_original_protocols() -> None:
    blocks = paper_blocks("primary")
    full = next(block for block in blocks if block["conditions"] == ["full"])
    assert full["dataset"] == "MI-OpenBCI"
    assert full["protocols"] == ["within_split", "within_session", "loso"]


def test_external_profile_uses_only_task_matched_multisession_datasets() -> None:
    blocks = paper_blocks("external")
    assert {block["dataset"] for block in blocks} == {"Zhou2020", "Tavakolan2017"}
    assert all(block["protocols"] == ["within_split", "cross_session"] for block in blocks)
