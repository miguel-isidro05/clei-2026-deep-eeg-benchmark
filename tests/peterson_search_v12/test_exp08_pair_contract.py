from __future__ import annotations

from experiments.peterson_search_v12.exp08_fixed_convergence.catalog import (
    backbone_pairs,
    experimental_candidates,
)
from experiments.peterson_search_v12.exp08_fixed_convergence.config import Exp08Config
from experiments.peterson_search_v12.exp08_fixed_convergence.manifest import (
    build_screen_manifest,
)


def test_exp08_catalog_preserves_five_paired_backbones() -> None:
    catalog = experimental_candidates()
    pairs = backbone_pairs()
    assert len(catalog) == 10
    assert len(pairs) == 5
    assert all(len(pair) == 2 for pair in pairs.values())
    assert all(candidate.config_id.startswith("exp08-") for candidate in catalog)


def test_screen_manifest_has_frozen_budget_and_closed_holdout() -> None:
    config = Exp08Config()
    manifest = build_screen_manifest(
        config,
        experimental_candidates(),
        revision="a" * 40,
        parent_decision_sha256="b" * 64,
    )
    assert len(manifest.cells) == 216
    assert not ({cell.subject for cell in manifest.cells} & set(config.holdout_subjects))

