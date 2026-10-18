"""Content-addressed staged manifest for Exp04b."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import (
    SafePathCandidate,
    build_w1_catalog,
    build_w2_catalog,
    build_w3_catalog,
    catalog_payload,
)
from .config import Exp04bConfig


@dataclass(frozen=True)
class Exp04bCell:
    cell_id: str
    wave: str
    kind: str
    model_id: str
    config_id: str
    config_sha256: str
    subject: str
    seed: int
    folds: tuple[int, ...]


@dataclass(frozen=True)
class Exp04bManifest:
    experiment: str
    revision: str
    config_sha256: str
    catalog_sha256: str
    cells: tuple[Exp04bCell, ...]
    sha256: str


def _candidate_cells(
    candidates: tuple[SafePathCandidate, ...],
    subjects: tuple[str, ...],
    seeds: tuple[int, ...],
    folds: tuple[int, ...],
) -> tuple[Exp04bCell, ...]:
    return tuple(
        Exp04bCell(
            cell_id=f"{candidate.wave}__{candidate.config_id}__seed{seed}__{subject}",
            wave=candidate.wave,
            kind="candidate",
            model_id="SafePath-QCR",
            config_id=candidate.config_id,
            config_sha256=candidate.sha256,
            subject=subject,
            seed=seed,
            folds=folds,
        )
        for candidate in candidates
        for seed in seeds
        for subject in subjects
    )


def build_search_manifest(
    config: Exp04bConfig,
    *,
    selected_w2: tuple[str, str, str],
    revision: str,
) -> Exp04bManifest:
    w1 = build_w1_catalog()
    w2 = build_w2_catalog()
    w3 = build_w3_catalog(selected_w2)
    if config.profile == "smoke":
        w1, w2 = w1[:2], w2[:3]
        w3 = (w3[0], w3[1], w3[3])
        cells = (
            _candidate_cells(w1, config.subjects, (0,), config.folds)
            + _candidate_cells(w2, config.subjects, (0,), config.folds)
            + _candidate_cells(w3, config.subjects, (0,), config.folds)
        )
    else:
        cells = (
            _candidate_cells(w1, config.subjects, (0,), config.folds)
            + _candidate_cells(w2, config.subjects, (0, 1), config.folds)
            + _candidate_cells(w3, config.subjects, config.seeds, config.folds)
            + tuple(
                Exp04bCell(
                    cell_id=f"w3__scm-mdrm__seed0__{subject}",
                    wave="w3",
                    kind="classical_control",
                    model_id="SCM-conditional+MDRM",
                    config_id="exp04b-w3-scm-mdrm",
                    config_sha256=canonical_sha256({"model_id": "SCM-conditional+MDRM", "seed": 0}),
                    subject=subject,
                    seed=0,
                    folds=config.folds,
                )
                for subject in config.subjects
            )
        )
    full_catalog = w1 + w2 + w3
    body = {
        "experiment": config.experiment,
        "revision": revision,
        "config_sha256": canonical_sha256(asdict(config)),
        "catalog_sha256": canonical_sha256(catalog_payload(full_catalog)),
        "cells": [asdict(cell) for cell in cells],
    }
    manifest = Exp04bManifest(
        experiment=config.experiment,
        revision=revision,
        config_sha256=str(body["config_sha256"]),
        catalog_sha256=str(body["catalog_sha256"]),
        cells=cells,
        sha256=canonical_sha256(body),
    )
    if config.profile == "full" and len(manifest.cells) != config.total_training_jobs:
        raise RuntimeError("Exp04b manifest budget drifted")
    return manifest


def shard_cells_by_subject(
    cells: tuple[Exp04bCell, ...], *, num_shards: int, shard_index: int
) -> tuple[Exp04bCell, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard selection")
    subjects = tuple(dict.fromkeys(cell.subject for cell in cells))
    ownership = {subject: index % num_shards for index, subject in enumerate(subjects)}
    return tuple(cell for cell in cells if ownership[cell.subject] == shard_index)


def manifest_payload(manifest: Exp04bManifest) -> dict[str, object]:
    return {**asdict(manifest), "cells": [asdict(cell) for cell in manifest.cells]}
