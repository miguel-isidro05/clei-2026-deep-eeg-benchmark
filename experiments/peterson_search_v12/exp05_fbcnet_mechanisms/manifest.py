"""Signed, phase-specific manifests for Peterson Exp05."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .config import BASELINE_CONFIG_ID, Exp05Config


@dataclass(frozen=True)
class Exp05Cell:
    cell_id: str
    phase: str
    kind: str
    config_id: str
    mechanism: str
    config_sha256: str
    subject: str
    seed: int
    folds: tuple[int, ...]


@dataclass(frozen=True)
class Exp05Manifest:
    experiment: str
    phase: str
    revision: str
    config_sha256: str
    catalog_sha256: str
    screen_decision_sha256: str | None
    cells: tuple[Exp05Cell, ...]
    sha256: str


def _catalog_payload(catalog: Iterable[Any]) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in catalog]


def _entry(candidate: Any) -> tuple[str, str, str]:
    return (
        str(candidate.config_id),
        str(candidate.mechanism),
        str(candidate.sha256),
    )


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
) -> Exp05Cell:
    return Exp05Cell(
        cell_id=f"{phase}__{config_id}__seed{seed}__{subject}",
        phase=phase,
        kind=kind,
        config_id=config_id,
        mechanism=mechanism,
        config_sha256=config_sha256,
        subject=subject,
        seed=seed,
        folds=folds,
    )


def _baseline_signature(config: Exp05Config) -> str:
    del config  # The training recipe is already signed by config_sha256.
    from .catalog import baseline_candidate

    return str(baseline_candidate().sha256)


def _build_manifest(
    config: Exp05Config,
    catalog: tuple[Any, ...],
    *,
    phase: str,
    selected_ids: tuple[str, ...],
    revision: str,
    screen_decision_sha256: str | None,
) -> Exp05Manifest:
    if phase not in {"screen", "confirm"}:
        raise ValueError("Exp05 phase must be screen or confirm")
    if len(catalog) != 20:
        raise ValueError("Exp05 requires exactly 20 signed candidates")
    entries = {_entry(candidate)[0]: _entry(candidate) for candidate in catalog}
    if len(entries) != len(catalog):
        raise ValueError("Exp05 catalog contains duplicate config IDs")
    if phase == "screen":
        selected_ids = tuple(entries)
        seeds = config.screen_seeds
    else:
        if len(selected_ids) > config.max_promoted or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("Invalid Exp05 promotion selection or budget")
        unknown = set(selected_ids) - set(entries)
        if unknown:
            raise ValueError(f"Unknown promoted Exp05 configurations: {sorted(unknown)}")
        if screen_decision_sha256 is None:
            raise ValueError("Confirmation manifest requires the signed screen decision")
        seeds = config.confirm_seeds
    specs = [
        (BASELINE_CONFIG_ID, "frozen_fbcnet", _baseline_signature(config), "baseline")
    ]
    specs += [(*entries[config_id], "candidate") for config_id in selected_ids]
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
        "catalog_sha256": canonical_sha256(_catalog_payload(catalog)),
        "screen_decision_sha256": screen_decision_sha256,
        "cells": [asdict(cell) for cell in cells],
    }
    return Exp05Manifest(
        experiment=config.experiment,
        phase=phase,
        revision=revision,
        config_sha256=str(body["config_sha256"]),
        catalog_sha256=str(body["catalog_sha256"]),
        screen_decision_sha256=screen_decision_sha256,
        cells=cells,
        sha256=canonical_sha256(body),
    )


def build_screen_manifest(
    config: Exp05Config, catalog: tuple[Any, ...], *, revision: str
) -> Exp05Manifest:
    manifest = _build_manifest(
        config,
        catalog,
        phase="screen",
        selected_ids=(),
        revision=revision,
        screen_decision_sha256=None,
    )
    expected = config.expected_screen_cells
    if len(manifest.cells) != expected:
        raise RuntimeError(f"Exp05 screen budget drifted: {len(manifest.cells)} != {expected}")
    return manifest


def build_confirm_manifest(
    config: Exp05Config,
    catalog: tuple[Any, ...],
    *,
    promoted_ids: tuple[str, ...],
    revision: str,
    screen_decision_sha256: str,
) -> Exp05Manifest:
    manifest = _build_manifest(
        config,
        catalog,
        phase="confirm",
        selected_ids=promoted_ids,
        revision=revision,
        screen_decision_sha256=screen_decision_sha256,
    )
    expected = config.expected_confirm_cells(len(promoted_ids))
    if len(manifest.cells) != expected:
        raise RuntimeError(f"Exp05 confirm budget drifted: {len(manifest.cells)} != {expected}")
    return manifest


def shard_cells_by_subject(
    cells: tuple[Exp05Cell, ...], *, num_shards: int, shard_index: int
) -> tuple[Exp05Cell, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard selection")
    subjects = tuple(dict.fromkeys(cell.subject for cell in cells))
    ownership = {subject: index % num_shards for index, subject in enumerate(subjects)}
    return tuple(cell for cell in cells if ownership[cell.subject] == shard_index)


def manifest_payload(manifest: Exp05Manifest) -> dict[str, Any]:
    return {**asdict(manifest), "cells": [asdict(cell) for cell in manifest.cells]}


def manifest_from_payload(payload: dict[str, Any]) -> Exp05Manifest:
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError("Exp05 manifest signature is invalid")
    cells = tuple(
        Exp05Cell(**{**cell, "folds": tuple(cell["folds"])}) for cell in payload["cells"]
    )
    manifest = Exp05Manifest(
        experiment=str(payload["experiment"]),
        phase=str(payload["phase"]),
        revision=str(payload["revision"]),
        config_sha256=str(payload["config_sha256"]),
        catalog_sha256=str(payload["catalog_sha256"]),
        screen_decision_sha256=payload.get("screen_decision_sha256"),
        cells=cells,
        sha256=signature,
    )
    if manifest.experiment != "exp05" or manifest.phase not in {"screen", "confirm"}:
        raise ValueError("Manifest belongs to another experiment")
    return manifest
