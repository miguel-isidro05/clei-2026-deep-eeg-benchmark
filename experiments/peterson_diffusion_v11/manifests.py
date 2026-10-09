"""Predeclared cell matrices for V11 waves."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import (
    ARCHITECTURES,
    BASELINE_MODELS,
    CONDITIONS,
    FORMULATIONS,
    ExperimentConfig,
)
from .identity import canonical_sha256


@dataclass(frozen=True)
class CellSpec:
    wave: int
    subject: str
    seed: int
    condition: str
    model: str
    formulation: str
    objective: str

    @property
    def cell_id(self) -> str:
        fields = (
            f"wave{self.wave:02d}",
            self.condition,
            self.formulation,
            self.model,
            self.objective,
            f"seed{self.seed}",
            self.subject,
        )
        return "__".join(fields)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_manifest(config: ExperimentConfig) -> list[CellSpec]:
    cells: list[CellSpec] = []
    if config.wave == 0:
        for model in BASELINE_MODELS:
            for seed in (0, 1, 2, 3, 4):
                for subject in config.subjects:
                    cells.append(
                        CellSpec(0, subject, seed, "center_x6", model, "baseline", "none")
                    )
    elif config.wave == 1:
        for architecture in ARCHITECTURES:
            for formulation in FORMULATIONS:
                for condition in CONDITIONS:
                    for subject in config.subjects:
                        objective = (
                            config.objective
                            if formulation == "diffusion"
                            else "cross_entropy"
                        )
                        cells.append(
                            CellSpec(
                                1,
                                subject,
                                0,
                                condition,
                                architecture,
                                formulation,
                                objective,
                            )
                        )
    else:
        raise ValueError(
            f"Wave {config.wave} requires the signed decision from the previous wave"
        )
    identifiers = [cell.cell_id for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Manifest contains duplicate cell identities")
    return cells


def shard_cells(
    cells: list[CellSpec], *, num_shards: int, shard_index: int
) -> list[CellSpec]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard")
    return cells[shard_index::num_shards]


def manifest_payload(config: ExperimentConfig, cells: list[CellSpec]) -> dict[str, object]:
    serialized = [cell.to_dict() | {"cell_id": cell.cell_id} for cell in cells]
    return {
        "schema_version": 1,
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "expected_cell_count": len(cells),
        "cells": serialized,
        "cells_sha256": canonical_sha256(serialized),
    }
