from deepbench.transfer_profile import (
    expected_peterson_cells,
    expected_source_checkpoints,
    expected_transfer_cells,
)


def test_v8_profile_has_exact_no_loso_scope() -> None:
    assert len(expected_source_checkpoints()) == 25
    assert len(expected_peterson_cells()) == 2000
    assert len(expected_transfer_cells()) == 750
    names = expected_peterson_cells() + expected_transfer_cells()
    assert all("__loso__" not in name for name in names)
    assert all("Souza2023__transfer__" in name for name in expected_transfer_cells())


def test_transfer_profile_uses_only_published_souza_files_002_to_006() -> None:
    cells = expected_transfer_cells()
    assert any(name.endswith("__002.json") for name in cells)
    assert not any(name.endswith("__001.json") for name in cells)
