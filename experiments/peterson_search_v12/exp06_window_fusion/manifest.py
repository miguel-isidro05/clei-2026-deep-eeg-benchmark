"""Signed cell manifests for Exp06."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import E11_REFERENCE_ID, FBC_CONTROL_ID, control_candidates
from .config import Exp06Config


@dataclass(frozen=True)
class Exp06Cell:
    phase: str
    kind: str
    cell_id: str
    config_id: str
    mechanism: str
    config_sha256: str
    subject: str
    seed: int
    folds: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class Exp06Manifest:
    experiment: str
    phase: str
    revision: str
    config_sha256: str
    catalog_sha256: str
    parent_decision_sha256: str
    screen_decision_sha256: str | None
    cells: tuple[Exp06Cell, ...]
    sha256: str


def _cell(
    *,
    phase: str,
    kind: str,
    config_id: str,
    mechanism: str,
    config_sha256: str,
    subject: str,
    seed: int,
    folds: tuple[int, ...],
) -> Exp06Cell:
    body = {
        "phase": phase,
        "kind": kind,
        "cell_id": f"{phase}__{config_id}__seed{seed}__{subject}",
        "config_id": config_id,
        "mechanism": mechanism,
        "config_sha256": config_sha256,
        "subject": subject,
        "seed": int(seed),
        "folds": folds,
    }
    return Exp06Cell(**body, sha256=canonical_sha256(body))


def _entry(candidate: Any) -> tuple[str, str, str, str]:
    return (
        str(candidate.config_id),
        str(candidate.mechanism),
        str(candidate.sha256),
        str(candidate.kind),
    )


def _build(
    config: Exp06Config,
    catalog: tuple[Any, ...],
    *,
    phase: str,
    promoted_ids: tuple[str, ...],
    revision: str,
    parent_decision_sha256: str,
    screen_decision_sha256: str | None,
) -> Exp06Manifest:
    if phase not in {"screen", "confirm"}:
        raise ValueError("Exp06 phase must be screen or confirm")
    if len(catalog) != 10 or len({item.config_id for item in catalog}) != 10:
        raise ValueError("Exp06 requires exactly ten unique candidates")
    entries = {item.config_id: _entry(item) for item in catalog}
    controls = [_entry(item) for item in control_candidates()]
    if phase == "screen":
        selected = tuple(entries)
        seeds = config.screen_seeds
    else:
        if len(promoted_ids) > config.max_promoted or len(set(promoted_ids)) != len(promoted_ids):
            raise ValueError("Invalid Exp06 promotion budget")
        if set(promoted_ids) - set(entries):
            raise ValueError("Unknown promoted Exp06 configuration")
        if not screen_decision_sha256:
            raise ValueError("Confirmation requires a signed screen decision")
        selected = promoted_ids
        seeds = config.confirm_seeds
    specs = controls + [entries[item] for item in selected]
    cells = tuple(
        _cell(
            phase=phase,
            kind=kind,
            config_id=config_id,
            mechanism=mechanism,
            config_sha256=signature,
            subject=subject,
            seed=seed,
            folds=config.folds,
        )
        for config_id, mechanism, signature, kind in specs
        for seed in seeds
        for subject in config.subjects
    )
    body = {
        "experiment": config.experiment,
        "phase": phase,
        "revision": revision,
        "config_sha256": canonical_sha256(asdict(config)),
        "catalog_sha256": canonical_sha256(
            [asdict(item) for item in (*control_candidates(), *catalog)]
        ),
        "parent_decision_sha256": parent_decision_sha256,
        "screen_decision_sha256": screen_decision_sha256,
        "cells": [asdict(cell) for cell in cells],
    }
    return Exp06Manifest(
        experiment=config.experiment,
        phase=phase,
        revision=revision,
        config_sha256=str(body["config_sha256"]),
        catalog_sha256=str(body["catalog_sha256"]),
        parent_decision_sha256=parent_decision_sha256,
        screen_decision_sha256=screen_decision_sha256,
        cells=cells,
        sha256=canonical_sha256(body),
    )


def build_screen_manifest(
    config: Exp06Config, catalog: tuple[Any, ...], *, revision: str, parent_decision_sha256: str
) -> Exp06Manifest:
    manifest = _build(
        config,
        catalog,
        phase="screen",
        promoted_ids=(),
        revision=revision,
        parent_decision_sha256=parent_decision_sha256,
        screen_decision_sha256=None,
    )
    if len(manifest.cells) != config.expected_screen_cells:
        raise RuntimeError("Exp06 screen budget drifted")
    return manifest


def build_confirm_manifest(
    config: Exp06Config,
    catalog: tuple[Any, ...],
    *,
    promoted_ids: tuple[str, ...],
    revision: str,
    parent_decision_sha256: str,
    screen_decision_sha256: str,
) -> Exp06Manifest:
    manifest = _build(
        config,
        catalog,
        phase="confirm",
        promoted_ids=promoted_ids,
        revision=revision,
        parent_decision_sha256=parent_decision_sha256,
        screen_decision_sha256=screen_decision_sha256,
    )
    if len(manifest.cells) != config.expected_confirm_cells(len(promoted_ids)):
        raise RuntimeError("Exp06 confirmation budget drifted")
    return manifest


def shard_cells_by_subject(
    cells: tuple[Exp06Cell, ...], *, num_shards: int, shard_index: int
) -> tuple[Exp06Cell, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid Exp06 shard")
    subjects = tuple(dict.fromkeys(cell.subject for cell in cells))
    owner = {subject: index % num_shards for index, subject in enumerate(subjects)}
    return tuple(cell for cell in cells if owner[cell.subject] == shard_index)


def manifest_payload(manifest: Exp06Manifest) -> dict[str, Any]:
    return {**asdict(manifest), "cells": [asdict(cell) for cell in manifest.cells]}


def manifest_from_payload(payload: dict[str, Any]) -> Exp06Manifest:
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError("Exp06 manifest signature is invalid")
    cells = tuple(Exp06Cell(**{**cell, "folds": tuple(cell["folds"])}) for cell in payload["cells"])
    manifest = Exp06Manifest(
        experiment=str(payload["experiment"]),
        phase=str(payload["phase"]),
        revision=str(payload["revision"]),
        config_sha256=str(payload["config_sha256"]),
        catalog_sha256=str(payload["catalog_sha256"]),
        parent_decision_sha256=str(payload["parent_decision_sha256"]),
        screen_decision_sha256=payload.get("screen_decision_sha256"),
        cells=cells,
        sha256=signature,
    )
    if manifest.experiment != "exp06" or manifest.phase not in {"screen", "confirm"}:
        raise ValueError("Manifest belongs to another experiment")
    return manifest


__all__ = [
    "E11_REFERENCE_ID",
    "FBC_CONTROL_ID",
    "Exp06Cell",
    "Exp06Manifest",
    "build_confirm_manifest",
    "build_screen_manifest",
    "manifest_from_payload",
    "manifest_payload",
    "shard_cells_by_subject",
]
