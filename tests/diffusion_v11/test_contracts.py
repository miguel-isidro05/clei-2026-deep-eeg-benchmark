from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.peterson_diffusion_v11.config import ExperimentConfig, load_wave_config
from experiments.peterson_diffusion_v11.identity import canonical_sha256
from experiments.peterson_diffusion_v11.manifests import build_manifest, shard_cells


def test_config_rejects_non_peterson_and_loso() -> None:
    with pytest.raises(ValueError, match="Peterson"):
        ExperimentConfig(dataset="Souza2023")
    with pytest.raises(ValueError, match="LOSO"):
        ExperimentConfig(protocol="loso")


def test_config_rejects_unknown_subject() -> None:
    with pytest.raises(ValueError, match="subject"):
        ExperimentConfig(subjects=("S01",))


def test_canonical_hash_is_order_independent() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_wave_manifests_have_frozen_counts() -> None:
    assert len(build_manifest(ExperimentConfig(wave=0))) == 250
    assert len(build_manifest(ExperimentConfig(wave=1))) == 240


def test_two_shards_are_disjoint_and_complete() -> None:
    cells = build_manifest(ExperimentConfig(wave=1))
    shard_0 = shard_cells(cells, num_shards=2, shard_index=0)
    shard_1 = shard_cells(cells, num_shards=2, shard_index=1)
    ids_0 = {cell.cell_id for cell in shard_0}
    ids_1 = {cell.cell_id for cell in shard_1}
    assert ids_0.isdisjoint(ids_1)
    assert ids_0 | ids_1 == {cell.cell_id for cell in cells}


def test_manifest_identity_changes_with_scientific_configuration() -> None:
    base = ExperimentConfig(wave=1)
    changed = replace(base, epochs=base.epochs + 1)
    assert base.sha256 != changed.sha256


def test_frozen_wave_configs_load_with_expected_identity() -> None:
    assert load_wave_config(0).wave == 0
    assert load_wave_config(1).wave == 1
    with pytest.raises(ValueError, match="no frozen configuration"):
        load_wave_config(2)
