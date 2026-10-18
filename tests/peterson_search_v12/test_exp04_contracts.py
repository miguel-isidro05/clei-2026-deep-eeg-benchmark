from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.runner import runtime_environment
from experiments.peterson_search_v12.exp04_qcr_ssnet.catalog import build_model_catalog
from experiments.peterson_search_v12.exp04_qcr_ssnet.config import Exp04Config, smoke_config
from experiments.peterson_search_v12.exp04_qcr_ssnet.manifest import (
    build_manifest,
    shard_cells,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.runner import (
    _predict_fbc,
    deterministic_stress_masks,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.source import select_exp03_source


def _summary(
    config_id: str,
    *,
    dropout: float,
    filters: int,
    accuracy: float,
    stress_floor: float,
    wins: int = 4,
) -> dict[str, object]:
    return {
        "config_id": config_id,
        "gate_id": "G0",
        "channel_dropout": dropout,
        "n_filters_spat": filters,
        "stride_factor": 4,
        "mean_accuracy": accuracy,
        "lower_quartile": accuracy - 0.02,
        "mean_kappa": 2 * accuracy - 1,
        "collapse_count": 0,
        "wins_vs_control": wins,
        "stress_floor": stress_floor,
        "parameters": {"active": 2500},
    }


def _exp03_report() -> dict[str, object]:
    return {
        "control": _summary("control", dropout=0.0, filters=32, accuracy=0.82, stress_floor=0.61),
        "ranking": [
            _summary("clean", dropout=0.0, filters=16, accuracy=0.845, stress_floor=0.62),
            _summary("robust", dropout=0.1, filters=48, accuracy=0.775, stress_floor=0.74),
        ],
    }


def test_exp04_scope_is_frozen_and_holdout_is_inaccessible():
    config = Exp04Config()
    assert config.experiment == "exp04"
    assert config.subjects == ("S09", "S03", "S02", "S10", "S12", "S08")
    assert config.folds == (0, 1, 2, 3, 4)
    assert config.seeds == (0, 1, 2, 3, 4)
    assert config.condition == "overlap"
    with pytest.raises(ValueError, match="holdout"):
        replace(config, subjects=("S07",))
    with pytest.raises(ValueError, match="within_session overlap"):
        replace(config, condition="center")


def test_exp03_source_requires_a_reproduced_clean_robust_front():
    source = select_exp03_source(
        _exp03_report(),
        promotion_decision_sha256="a" * 64,
        quality_manifest_sha256="b" * 64,
    )
    assert source.status == "selected"
    assert source.clean_config_id == "clean"
    assert source.robust_config_id == "robust"
    assert source.clean_n_filters_spat == 16
    assert source.robust_n_filters_spat == 48
    assert len(source.sha256) == 64

    failed = _exp03_report()
    failed["ranking"][1]["stress_floor"] = 0.63
    with pytest.raises(ValueError, match="clean/robust front"):
        select_exp03_source(
            failed,
            promotion_decision_sha256="a" * 64,
            quality_manifest_sha256="b" * 64,
        )


def test_exp04_catalog_and_manifest_match_predeclared_budget():
    source = select_exp03_source(
        _exp03_report(),
        promotion_decision_sha256="a" * 64,
        quality_manifest_sha256="b" * 64,
    )
    config = Exp04Config()
    catalog = build_model_catalog(config, source)
    assert len(catalog) == 10
    assert {candidate.role for candidate in catalog} == {
        "clean_control",
        "robust_control",
        "qcr_ablation",
        "qcr_full",
    }
    assert sum(candidate.role == "qcr_full" for candidate in catalog) == 1
    assert len({candidate.config_id for candidate in catalog}) == len(catalog)

    manifest = build_manifest(config, catalog, source, revision="abc123")
    assert len(manifest.cells) == 336
    assert sum(cell.kind == "candidate" for cell in manifest.cells) == 300
    assert sum(cell.model_id == "FBCNet-V10" for cell in manifest.cells) == 30
    assert sum(cell.model_id == "CSP+LDA" for cell in manifest.cells) == 6
    left = shard_cells(manifest.cells, num_shards=2, shard_index=0)
    right = shard_cells(manifest.cells, num_shards=2, shard_index=1)
    assert len(left) == len(right) == 168
    assert {cell.cell_id for cell in left}.isdisjoint(cell.cell_id for cell in right)


def test_exp04_smoke_reduces_subjects_folds_seeds_and_catalog():
    source = select_exp03_source(
        _exp03_report(),
        promotion_decision_sha256="a" * 64,
        quality_manifest_sha256="b" * 64,
    )
    config = smoke_config()
    catalog = build_model_catalog(config, source)
    manifest = build_manifest(config, catalog, source, revision="abc123")
    assert config.subjects == ("S09", "S03")
    assert config.folds == (0,)
    assert config.seeds == (0,)
    assert len(catalog) == 4
    assert len(manifest.cells) == 10


def test_exp04_stress_enumerates_every_one_and_two_channel_combination():
    single = deterministic_stress_masks(n_chans=15, missing_channels=1)
    pairs = deterministic_stress_masks(n_chans=15, missing_channels=2)
    assert len(single) == len(set(single)) == 15
    assert len(pairs) == len(set(pairs)) == 105
    assert all(sum(value == 0 for value in mask) == 2 for mask in pairs)


def test_exp04_fbc_prediction_uses_exactly_one_forward_call():
    class Classifier:
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, windows: np.ndarray) -> np.ndarray:
            self.calls += 1
            assert windows.flags.c_contiguous
            return np.tile(np.asarray([[0.25, 0.75]]), (len(windows), 1))

        def predict(self, windows: np.ndarray) -> np.ndarray:
            raise AssertionError("_predict_fbc must not perform a second forward pass")

    classifier = Classifier()
    scores = _predict_fbc(classifier, np.ones((3, 15, 128), dtype=np.float64))
    assert classifier.calls == 1
    np.testing.assert_allclose(scores, [0.75, 0.75, 0.75])


def test_exp04_default_result_directory_is_ignored_by_git():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            "results_peterson_exp04_contract_probe/manifests/test.json",
        ],
        cwd=root,
        check=True,
    )


def test_scientific_environment_tracks_model_defining_dependencies():
    environment = runtime_environment("cpu")
    packages = environment["packages"]
    assert {"braindecode", "skorch", "torch_cuda", "cudnn"} <= set(packages)
    scientific = {
        key: environment[key] for key in ("python", "platform", "packages")
    }
    changed = {
        **scientific,
        "packages": {**packages, "braindecode": "different-version"},
    }
    assert canonical_sha256(scientific) != canonical_sha256(changed)
