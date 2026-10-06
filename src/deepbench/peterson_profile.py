"""Frozen Peterson-only confirmatory experiment profile."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    MI_SUBJECTS,
    PAPER_SEEDS,
    PETERSON_MODEL_NAMES,
    PETERSON_PROFILE_VERSION,
)


@dataclass(frozen=True)
class PetersonBlock:
    protocols: tuple[str, ...]
    conditions: tuple[str, ...]
    ica_policy: str


PETERSON_BLOCKS: tuple[PetersonBlock, ...] = (
    PetersonBlock(("within_split", "within_session", "loso"), ("overlap",), "none"),
    PetersonBlock(("within_split", "within_session", "loso"), ("full",), "none"),
    PetersonBlock(("within_split", "within_session", "loso"), ("center",), "none"),
    PetersonBlock(("within_split",), ("center_x2", "nonoverlap", "center_x6"), "none"),
    PetersonBlock(("within_split",), ("overlap",), "kurtosis"),
)


def expected_cell_names() -> list[str]:
    names: list[str] = []
    for block in PETERSON_BLOCKS:
        for protocol in block.protocols:
            for condition in block.conditions:
                for model in PETERSON_MODEL_NAMES:
                    for seed in PAPER_SEEDS:
                        for subject in MI_SUBJECTS:
                            names.append(
                                "cells/"
                                f"MI-OpenBCI__{protocol}__{condition}__"
                                f"ica-{block.ica_policy}__{model}__seed{seed}__{subject}.json"
                            )
    return names


def expected_cell_count() -> int:
    return len(expected_cell_names())


def profile_metadata() -> dict[str, object]:
    return {
        "profile_version": PETERSON_PROFILE_VERSION,
        "dataset": "MI-OpenBCI",
        "task": "motor_imagery_vs_rest",
        "subjects": list(MI_SUBJECTS),
        "models": list(PETERSON_MODEL_NAMES),
        "seeds": list(PAPER_SEEDS),
        "blocks": [
            {
                "protocols": list(block.protocols),
                "conditions": list(block.conditions),
                "ica_policy": block.ica_policy,
            }
            for block in PETERSON_BLOCKS
        ],
        "expected_cell_count": expected_cell_count(),
        "expected_cells": expected_cell_names(),
        "primary_condition": "overlap",
        "claim_scope": "within_dataset_low_cost_mi_vs_rest_benchmark",
        "known_limitation": "single_dataset_ten_subjects_no_hardware_causal_comparison",
    }
