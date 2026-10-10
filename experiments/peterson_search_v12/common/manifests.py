"""Expected-cell manifests and deterministic sharding."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_diffusion_v11.identity import canonical_sha256

from .catalogs import CandidateSpec, catalog_payload
from .config import SearchConfig


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    config_id: str
    candidate_sha256: str
    subject: str
    seed: int
    folds: tuple[int, ...]


@dataclass(frozen=True)
class SearchManifest:
    experiment: str
    revision: str
    config_sha256: str
    catalog_sha256: str
    cells: tuple[CellSpec, ...]
    sha256: str


def build_manifest(
    config: SearchConfig, catalog: tuple[CandidateSpec, ...], *, revision: str
) -> SearchManifest:
    cells = tuple(
        CellSpec(
            cell_id=f"{candidate.config_id}__seed{seed}__{subject}",
            config_id=candidate.config_id,
            candidate_sha256=candidate.sha256,
            subject=subject,
            seed=seed,
            folds=config.folds,
        )
        for candidate in catalog
        for seed in config.seeds
        for subject in config.subjects
    )
    config_sha256 = canonical_sha256(asdict(config))
    catalog_sha256 = canonical_sha256(catalog_payload(catalog))
    base = {
        "experiment": config.experiment,
        "revision": revision,
        "config_sha256": config_sha256,
        "catalog_sha256": catalog_sha256,
        "cells": [asdict(cell) for cell in cells],
    }
    return SearchManifest(
        experiment=config.experiment,
        revision=revision,
        config_sha256=config_sha256,
        catalog_sha256=catalog_sha256,
        cells=cells,
        sha256=canonical_sha256(base),
    )


def shard_cells(
    cells: tuple[CellSpec, ...], *, num_shards: int, shard_index: int
) -> tuple[CellSpec, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard selection")
    return tuple(cell for index, cell in enumerate(cells) if index % num_shards == shard_index)


def build_smoke_manifest(
    config: SearchConfig,
    catalog: tuple[CandidateSpec, ...],
    *,
    revision: str,
    num_shards: int = 2,
) -> SearchManifest:
    full = build_manifest(config, catalog, revision=revision)
    cells = tuple(
        shard_cells(full.cells, num_shards=num_shards, shard_index=index)[0]
        for index in range(num_shards)
    )
    base = {
        "experiment": config.experiment,
        "profile": "smoke",
        "revision": revision,
        "config_sha256": full.config_sha256,
        "catalog_sha256": full.catalog_sha256,
        "parent_manifest_sha256": full.sha256,
        "cells": [asdict(cell) for cell in cells],
    }
    return SearchManifest(
        experiment=config.experiment,
        revision=revision,
        config_sha256=full.config_sha256,
        catalog_sha256=full.catalog_sha256,
        cells=cells,
        sha256=canonical_sha256(base),
    )
