"""Content-addressed Exp04 manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import QCRCandidate, catalog_payload
from .config import Exp04Config
from .source import Exp04Source


@dataclass(frozen=True)
class Exp04Cell:
    cell_id: str
    kind: str
    model_id: str
    config_id: str | None
    config_sha256: str | None
    subject: str
    seed: int
    folds: tuple[int, ...]


@dataclass(frozen=True)
class Exp04Manifest:
    experiment: str
    revision: str
    config_sha256: str
    catalog_sha256: str
    source_sha256: str
    cells: tuple[Exp04Cell, ...]
    sha256: str


def build_manifest(
    config: Exp04Config,
    catalog: tuple[QCRCandidate, ...],
    source: Exp04Source,
    *,
    revision: str,
) -> Exp04Manifest:
    candidate_cells = tuple(
        Exp04Cell(
            cell_id=f"candidate__{candidate.config_id}__seed{seed}__{subject}",
            kind="candidate",
            model_id="QCR-SSNet" if candidate.architecture == "qcr" else candidate.architecture,
            config_id=candidate.config_id,
            config_sha256=candidate.sha256,
            subject=subject,
            seed=seed,
            folds=config.folds,
        )
        for candidate in catalog
        for seed in config.seeds
        for subject in config.subjects
    )
    fbc_cells = tuple(
        Exp04Cell(
            cell_id=f"baseline__FBCNet-V10__seed{seed}__{subject}",
            kind="baseline",
            model_id="FBCNet-V10",
            config_id=None,
            config_sha256=canonical_sha256(
                {"model_id": "FBCNet-V10", "protocol": "within_session_overlap"}
            ),
            subject=subject,
            seed=seed,
            folds=config.folds,
        )
        for seed in config.seeds
        for subject in config.subjects
    )
    csp_cells = (
        tuple()
        if config.profile == "smoke"
        else tuple(
            Exp04Cell(
                cell_id=f"baseline__CSP-LDA__seed0__{subject}",
                kind="baseline",
                model_id="CSP+LDA",
                config_id=None,
                config_sha256=canonical_sha256(
                    {"model_id": "CSP+LDA", "protocol": "within_session_overlap"}
                ),
                subject=subject,
                seed=0,
                folds=config.folds,
            )
            for subject in config.subjects
        )
    )
    config_hash = canonical_sha256(asdict(config))
    catalog_hash = canonical_sha256(catalog_payload(catalog))
    cells = candidate_cells + fbc_cells + csp_cells
    body = {
        "experiment": config.experiment,
        "revision": revision,
        "config_sha256": config_hash,
        "catalog_sha256": catalog_hash,
        "source_sha256": source.sha256,
        "cells": [asdict(cell) for cell in cells],
    }
    return Exp04Manifest(
        experiment=config.experiment,
        revision=revision,
        config_sha256=config_hash,
        catalog_sha256=catalog_hash,
        source_sha256=source.sha256,
        cells=cells,
        sha256=canonical_sha256(body),
    )


def shard_cells(
    cells: tuple[Exp04Cell, ...], *, num_shards: int, shard_index: int
) -> tuple[Exp04Cell, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard selection")
    return tuple(cell for index, cell in enumerate(cells) if index % num_shards == shard_index)


def manifest_payload(manifest: Exp04Manifest) -> dict[str, object]:
    return {
        "experiment": manifest.experiment,
        "revision": manifest.revision,
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "source_sha256": manifest.source_sha256,
        "cells": [asdict(cell) for cell in manifest.cells],
        "sha256": manifest.sha256,
    }
