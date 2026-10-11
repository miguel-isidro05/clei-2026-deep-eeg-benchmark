"""Content-addressed exp02 manifest and deterministic sharding."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import Exp02Candidate, catalog_payload
from .config import Exp02Config


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    kind: str
    model_id: str
    config_id: str | None
    candidate_sha256: str | None
    subject: str
    seed: int
    folds: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentManifest:
    experiment: str
    revision: str
    config_sha256: str
    catalog_sha256: str
    cells: tuple[ExperimentCell, ...]
    sha256: str


def build_experiment_manifest(
    config: Exp02Config,
    candidates: tuple[Exp02Candidate, ...],
    *,
    revision: str,
) -> ExperimentManifest:
    candidate_cells = tuple(
        ExperimentCell(
            cell_id=f"{candidate.config_id}__seed{seed}__{subject}",
            kind="candidate",
            model_id=candidate.topology_id,
            config_id=candidate.config_id,
            candidate_sha256=candidate.sha256,
            subject=subject,
            seed=seed,
            folds=config.folds,
        )
        for candidate in candidates
        for seed in config.seeds
        for subject in config.subjects
    )
    baseline_cells = tuple(
        ExperimentCell(
            cell_id=f"baseline__{model.replace('+', '-')}__seed0__{subject}",
            kind="baseline",
            model_id=model,
            config_id=None,
            candidate_sha256=None,
            subject=subject,
            seed=0,
            folds=config.folds,
        )
        for model in ("CSP+LDA", "FBCNet")
        for subject in config.subjects
    )
    cells = candidate_cells + baseline_cells
    config_sha256 = canonical_sha256(asdict(config))
    catalog_sha256 = canonical_sha256(catalog_payload(candidates))
    body = {
        "experiment": config.experiment,
        "revision": revision,
        "config_sha256": config_sha256,
        "catalog_sha256": catalog_sha256,
        "cells": [asdict(cell) for cell in cells],
    }
    return ExperimentManifest(
        experiment=config.experiment,
        revision=revision,
        config_sha256=config_sha256,
        catalog_sha256=catalog_sha256,
        cells=cells,
        sha256=canonical_sha256(body),
    )


def shard_cells(
    cells: tuple[ExperimentCell, ...], *, num_shards: int, shard_index: int
) -> tuple[ExperimentCell, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard selection")
    return tuple(cell for index, cell in enumerate(cells) if index % num_shards == shard_index)


def manifest_payload(manifest: ExperimentManifest) -> dict[str, object]:
    return {
        "experiment": manifest.experiment,
        "revision": manifest.revision,
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "cells": [asdict(cell) for cell in manifest.cells],
        "sha256": manifest.sha256,
    }
