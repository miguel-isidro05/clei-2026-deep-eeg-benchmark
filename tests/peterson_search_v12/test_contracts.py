from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from deepbench.types import SubjectRecording
from experiments.peterson_search_v12.common.catalogs import build_exp01_catalog
from experiments.peterson_search_v12.common.config import SearchConfig, load_stage_config
from experiments.peterson_search_v12.common.manifests import build_manifest, shard_cells
from experiments.peterson_search_v12.common.partitions import (
    DISCOVERY_SUBJECTS,
    HOLDOUT_SUBJECTS,
    partition_subjects,
    validate_partition,
)
from experiments.peterson_search_v12.common.splits import build_search_splits


def _recording() -> SubjectRecording:
    labels = np.tile(np.array([0, 1], dtype=np.int64), 25)
    return SubjectRecording(
        dataset="MI-OpenBCI",
        subject="S02",
        x=np.zeros((50, 3, 64), dtype=np.float32),
        y=labels,
        sessions=np.array(["session_1"] * 50),
        sfreq=128.0,
        ch_names=("C3", "Cz", "C4"),
        task="motor_imagery_vs_rest",
    )


def test_exp01_contract_and_holdout_boundary(tmp_path):
    config = SearchConfig()
    assert config.dataset == "MI-OpenBCI"
    assert config.protocol == "within_session"
    assert config.condition == "overlap"
    config.validate_subjects(DISCOVERY_SUBJECTS)
    with pytest.raises(ValueError, match="holdout"):
        config.validate_subjects(("S07",))
    with pytest.raises(ValueError, match="exp02"):
        load_stage_config("exp02", decision_dir=tmp_path)
    for field, value in (
        ("dataset", "Souza2023"),
        ("protocol", "loso"),
        ("condition", "center"),
    ):
        with pytest.raises(ValueError):
            replace(config, **{field: value})


def test_subject_partition_is_frozen_and_complete():
    discovery, holdout = partition_subjects()
    assert discovery == DISCOVERY_SUBJECTS == ("S09", "S03", "S02", "S10", "S12", "S08")
    assert holdout == HOLDOUT_SUBJECTS == ("S07", "S04", "S05", "S06")
    validate_partition(discovery, holdout)
    with pytest.raises(ValueError):
        validate_partition(discovery, holdout + ("S09",))


def test_catalog_and_manifest_are_deterministic():
    config = SearchConfig()
    first = build_exp01_catalog(config)
    second = build_exp01_catalog(config)
    assert first == second
    assert len(first) == 36
    assert len({item.config_id for item in first}) == 36
    assert {(item.backbone, item.formulation, item.width) for item in first} == {
        (backbone, formulation, width)
        for backbone in config.backbones
        for formulation in config.formulations
        for width in config.widths
    }
    manifest = build_manifest(config, first, revision="abc123")
    assert len(manifest.cells) == 216
    left = shard_cells(manifest.cells, num_shards=2, shard_index=0)
    right = shard_cells(manifest.cells, num_shards=2, shard_index=1)
    assert len(left) == len(right) == 108
    assert not ({cell.cell_id for cell in left} & {cell.cell_id for cell in right})
    assert {cell.cell_id for cell in left + right} == {cell.cell_id for cell in manifest.cells}


def test_search_splits_hide_outer_test_values():
    recording = _recording()
    splits = build_search_splits(recording, folds=(0, 2), split_seed=2026)
    assert {split.fold for split in splits} == {0, 2}
    assert all(not hasattr(split, "test_indices") for split in splits)
    assert all(not hasattr(split, "outer_test_indices") for split in splits)
    assert all(set(split.train_indices).isdisjoint(split.validation_indices) for split in splits)
    assert splits == build_search_splits(recording, folds=(0, 2), split_seed=2026)
    assert all(len(split.outer_test_sha256) == 64 for split in splits)
