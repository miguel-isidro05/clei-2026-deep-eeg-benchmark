"""Frozen experiment blocks used for the paper run."""

from __future__ import annotations

import hashlib
import json
from typing import TypedDict

from .config import MODEL_NAMES, PAPER_EPOCHS, PAPER_PROFILE_VERSION, PAPER_SEEDS
from .models import recipe_dict
from .runner import _code_fingerprint


class PaperBlock(TypedDict):
    dataset: str
    protocols: list[str]
    conditions: list[str]
    ica_policy: str


def paper_blocks(phase: str) -> list[PaperBlock]:
    """Return the predeclared blocks without inspecting any result."""
    if phase not in {
        "peterson",
        "souza",
        "primary",
        "ica-sensitivity",
        "no-loso",
        "all",
    }:
        raise ValueError(f"Unsupported paper phase: {phase}")
    blocks: list[PaperBlock] = []
    if phase in {"peterson", "primary", "no-loso", "all"}:
        blocks.extend(
            [
                {
                    "dataset": "MI-OpenBCI",
                    "protocols": (
                        ["within_split", "within_session"]
                        if phase == "no-loso"
                        else ["within_split", "within_session", "loso"]
                    ),
                    "conditions": ["full"],
                    "ica_policy": "none",
                },
                {
                    "dataset": "MI-OpenBCI",
                    "protocols": ["within_split"],
                    "conditions": ["center_x2", "nonoverlap", "center_x6", "overlap"],
                    "ica_policy": "none",
                },
            ]
        )
    if phase in {"souza", "primary", "no-loso", "all"}:
        blocks.extend(
            [
                {
                    "dataset": "Souza2023",
                    "protocols": (
                        ["within_split", "within_session", "cross_session"]
                        if phase == "no-loso"
                        else ["within_split", "within_session", "cross_session", "loso"]
                    ),
                    "conditions": ["full"],
                    "ica_policy": "none",
                },
                {
                    "dataset": "Souza2023",
                    "protocols": ["within_split"],
                    "conditions": ["center_x2", "nonoverlap", "center_x6", "overlap"],
                    "ica_policy": "none",
                },
            ]
        )
    if phase in {"ica-sensitivity", "no-loso", "all"}:
        blocks.extend(
            {
                "dataset": dataset,
                "protocols": ["within_split"],
                "conditions": ["full"],
                "ica_policy": "kurtosis",
            }
            for dataset in ("MI-OpenBCI", "Souza2023")
        )
    return blocks


def paper_profile_metadata(phase: str) -> dict[str, object]:
    """Return the frozen scientific profile embedded in expectation manifests."""
    profile: dict[str, object] = {
        "profile_version": PAPER_PROFILE_VERSION,
        "phase": phase,
        "models": list(MODEL_NAMES),
        "seeds": list(PAPER_SEEDS),
        "epochs": PAPER_EPOCHS,
        "blocks": paper_blocks(phase),
        "recipes": {model: recipe_dict(model, PAPER_EPOCHS) for model in MODEL_NAMES},
        "scientific_code_sha256": _code_fingerprint(),
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    profile["profile_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return profile
