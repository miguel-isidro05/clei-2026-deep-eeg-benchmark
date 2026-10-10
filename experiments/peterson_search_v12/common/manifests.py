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
    catalog_sha256 = canonical_sha256(catalog_payload(catalog))
    base = {
        "experiment": config.experiment,
        "revision": revision,
        "catalog_sha256": catalog_sha256,
        "cells": [asdict(cell) for cell in cells],
    }
    return SearchManifest(
        experiment=config.experiment,
        revision=revision,
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
