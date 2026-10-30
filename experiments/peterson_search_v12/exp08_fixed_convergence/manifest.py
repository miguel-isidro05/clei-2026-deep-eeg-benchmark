"""Signed screen and confirmation manifests for Exp08."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import backbone_pairs, control_candidates
from .config import Exp08Config


@dataclass(frozen=True)
class Exp08Cell:
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
class Exp08Manifest:
    experiment: str
    phase: str
    revision: str
    config_sha256: str
    catalog_sha256: str
    parent_decision_sha256: str
    screen_decision_sha256: str | None
    cells: tuple[Exp08Cell, ...]
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
) -> Exp08Cell:
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
    return Exp08Cell(**body, sha256=canonical_sha256(body))


def _entry(candidate: Any) -> tuple[str, str, str, str]:
    return candidate.config_id, candidate.mechanism, candidate.sha256, candidate.kind


def _build(
    config: Exp08Config,
    catalog: tuple[Any, ...],
    *,
    phase: str,
    promoted_backbones: tuple[str, ...],
    revision: str,
    parent_decision_sha256: str,
    screen_decision_sha256: str | None,
) -> Exp08Manifest:
    if phase not in {"screen", "confirm"}:
        raise ValueError("Exp08 phase must be screen or confirm")
    if len(catalog) != 10 or len({item.config_id for item in catalog}) != 10:
        raise ValueError("Exp08 requires exactly ten unique experimental candidates")
    entries = {item.config_id: _entry(item) for item in catalog}
    pairs = backbone_pairs()
    if phase == "screen":
        selected = tuple(entries)
        seeds = config.screen_seeds
    else:
        if (
            len(promoted_backbones) > config.max_promoted_backbones
            or len(set(promoted_backbones)) != len(promoted_backbones)
        ):
            raise ValueError("Invalid Exp08 backbone promotion budget")
        if set(promoted_backbones) - set(pairs):
            raise ValueError("Unknown promoted Exp08 backbone")
        if not screen_decision_sha256:
            raise ValueError("Confirmation requires a signed screen decision")
        selected = tuple(
            config_id for backbone in promoted_backbones for config_id in pairs[backbone]
        )
        seeds = config.confirm_seeds
    specs = [_entry(item) for item in control_candidates()] + [entries[item] for item in selected]
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
    return Exp08Manifest(
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
    config: Exp08Config,
    catalog: tuple[Any, ...],
    *,
    revision: str,
    parent_decision_sha256: str,
) -> Exp08Manifest:
    manifest = _build(
        config,
        catalog,
        phase="screen",
        promoted_backbones=(),
        revision=revision,
        parent_decision_sha256=parent_decision_sha256,
        screen_decision_sha256=None,
    )
    if len(manifest.cells) != config.expected_screen_cells:
        raise RuntimeError("Exp08 screen budget drifted")
    return manifest


def build_confirm_manifest(
    config: Exp08Config,
    catalog: tuple[Any, ...],
    *,
    promoted_backbones: tuple[str, ...],
    revision: str,
    parent_decision_sha256: str,
    screen_decision_sha256: str,
) -> Exp08Manifest:
    manifest = _build(
        config,
        catalog,
        phase="confirm",
        promoted_backbones=promoted_backbones,
        revision=revision,
        parent_decision_sha256=parent_decision_sha256,
        screen_decision_sha256=screen_decision_sha256,
    )
    if len(manifest.cells) != config.expected_confirm_cells(len(promoted_backbones)):
        raise RuntimeError("Exp08 confirmation budget drifted")
    return manifest


def manifest_payload(manifest: Exp08Manifest) -> dict[str, Any]:
    return {**asdict(manifest), "cells": [asdict(cell) for cell in manifest.cells]}


def manifest_from_payload(payload: dict[str, Any]) -> Exp08Manifest:
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError("Exp08 manifest signature is invalid")
    cells = tuple(
        Exp08Cell(**{**cell, "folds": tuple(cell["folds"])}) for cell in payload["cells"]
    )
    manifest = Exp08Manifest(
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
    if manifest.experiment != "exp08" or manifest.phase not in {"screen", "confirm"}:
        raise ValueError("Manifest belongs to another experiment")
    return manifest

