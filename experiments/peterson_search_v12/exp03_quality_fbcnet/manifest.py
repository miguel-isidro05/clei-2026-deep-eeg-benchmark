"""Content-addressed manifests for the two sequential exp03 phases."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import QualityCandidate, RecipeSpec, catalog_payload
from .config import Exp03Config


@dataclass(frozen=True)
class Exp03Cell:
    cell_id: str
    kind: str
    model_id: str
    config_id: str | None
    config_sha256: str | None
    subject: str
    seed: int
    folds: tuple[int, ...]


@dataclass(frozen=True)
class Exp03Manifest:
    experiment: str
    phase: str
    revision: str
    config_sha256: str
    catalog_sha256: str
    parent_decision_sha256: str | None
    cells: tuple[Exp03Cell, ...]
    sha256: str


def _manifest(
    config: Exp03Config,
    *,
    phase: str,
    revision: str,
    catalog: list[dict[str, object]],
    cells: tuple[Exp03Cell, ...],
    parent_decision_sha256: str | None,
) -> Exp03Manifest:
    config_hash = canonical_sha256(asdict(config))
    catalog_hash = canonical_sha256(catalog)
    body = {
        "experiment": config.experiment,
        "phase": phase,
        "revision": revision,
        "config_sha256": config_hash,
        "catalog_sha256": catalog_hash,
        "parent_decision_sha256": parent_decision_sha256,
        "cells": [asdict(cell) for cell in cells],
    }
    return Exp03Manifest(
        experiment=config.experiment,
        phase=phase,
        revision=revision,
        config_sha256=config_hash,
        catalog_sha256=catalog_hash,
        parent_decision_sha256=parent_decision_sha256,
        cells=cells,
        sha256=canonical_sha256(body),
    )


def build_recipe_manifest(
    config: Exp03Config,
    recipes: tuple[RecipeSpec, ...],
    *,
    revision: str,
) -> Exp03Manifest:
    recipe_cells = tuple(
        Exp03Cell(
            cell_id=f"recipe__{recipe.recipe_id}__seed{seed}__{subject}",
            kind="recipe",
            model_id="FBCNet",
            config_id=recipe.recipe_id,
            config_sha256=recipe.sha256,
            subject=subject,
            seed=seed,
            folds=config.recipe_folds,
        )
        for recipe in recipes
        for seed in config.recipe_seeds
        for subject in config.subjects
    )
    csp_cells = tuple(
        Exp03Cell(
            cell_id=f"baseline__CSP-LDA__seed0__{subject}",
            kind="baseline",
            model_id="CSP+LDA",
            config_id=None,
            config_sha256=None,
            subject=subject,
            seed=0,
            folds=config.recipe_folds,
        )
        for subject in config.subjects
    )
    fbc_cells = tuple(
        Exp03Cell(
            cell_id=f"baseline__FBCNet-V10__seed{seed}__{subject}",
            kind="baseline",
            model_id="FBCNet-V10",
            config_id=None,
            config_sha256=None,
            subject=subject,
            seed=seed,
            folds=config.recipe_folds,
        )
        for seed in config.recipe_seeds
        for subject in config.subjects
    )
    return _manifest(
        config,
        phase="recipe",
        revision=revision,
        catalog=catalog_payload(recipes),
        cells=recipe_cells + csp_cells + fbc_cells,
        parent_decision_sha256=None,
    )


def build_quality_manifest(
    config: Exp03Config,
    candidates: tuple[QualityCandidate, ...],
    *,
    revision: str,
    recipe_decision_sha256: str,
) -> Exp03Manifest:
    if len(recipe_decision_sha256) != 64:
        raise ValueError("Recipe decision SHA-256 must contain 64 hexadecimal characters")
    cells = tuple(
        Exp03Cell(
            cell_id=f"quality__{candidate.config_id}__seed{seed}__{subject}",
            kind="quality",
            model_id="QualityFBCNet",
            config_id=candidate.config_id,
            config_sha256=candidate.sha256,
            subject=subject,
            seed=seed,
            folds=config.quality_folds,
        )
        for candidate in candidates
        for seed in config.quality_seeds
        for subject in config.subjects
    )
    return _manifest(
        config,
        phase="quality",
        revision=revision,
        catalog=catalog_payload(candidates),
        cells=cells,
        parent_decision_sha256=recipe_decision_sha256,
    )


def shard_cells(
    cells: tuple[Exp03Cell, ...], *, num_shards: int, shard_index: int
) -> tuple[Exp03Cell, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard selection")
    return tuple(cell for index, cell in enumerate(cells) if index % num_shards == shard_index)


def manifest_payload(manifest: Exp03Manifest) -> dict[str, object]:
    return {
        "experiment": manifest.experiment,
        "phase": manifest.phase,
        "revision": manifest.revision,
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "parent_decision_sha256": manifest.parent_decision_sha256,
        "cells": [asdict(cell) for cell in manifest.cells],
        "sha256": manifest.sha256,
    }
