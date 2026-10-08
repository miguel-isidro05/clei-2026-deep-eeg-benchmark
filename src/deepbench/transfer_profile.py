"""Frozen repaired Peterson primary and Peterson-to-Souza transfer scope."""

from __future__ import annotations

from .config import MI_SUBJECTS, MODEL_NAMES, PAPER_SEEDS, SOUZA_SUBJECTS
from .transfer import TRANSFER_ARMS

TRANSFER_PROFILE_VERSION = "clei2026-peterson-souza-transfer-v9-repair"


def expected_source_checkpoints() -> list[str]:
    return [f"source__{model}__seed{seed}.pt" for model in MODEL_NAMES for seed in PAPER_SEEDS]


def expected_peterson_cells() -> list[str]:
    blocks = [
        ("within_split", "overlap", "none"),
        ("within_session", "overlap", "none"),
        ("within_split", "full", "none"),
        ("within_session", "full", "none"),
        ("within_split", "center_x2", "none"),
        ("within_split", "nonoverlap", "none"),
        ("within_split", "center_x6", "none"),
        ("within_split", "overlap", "kurtosis"),
    ]
    return [
        f"MI-OpenBCI__{protocol}__{condition}__ica-{ica}__{model}__seed{seed}__{subject}.json"
        for protocol, condition, ica in blocks
        for model in MODEL_NAMES
        for seed in PAPER_SEEDS
        for subject in MI_SUBJECTS
    ]


def expected_transfer_cells() -> list[str]:
    return [
        f"Souza2023__transfer__{protocol}__overlap__{arm}__{model}__seed{seed}__{subject}.json"
        for protocol in ("within_session", "cross_session")
        for arm in TRANSFER_ARMS
        for model in MODEL_NAMES
        for seed in PAPER_SEEDS
        for subject in SOUZA_SUBJECTS
    ]
