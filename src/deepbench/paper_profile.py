"""Frozen experiment blocks used for the paper run."""

from __future__ import annotations

from typing import TypedDict


class PaperBlock(TypedDict):
    dataset: str
    protocols: list[str]
    conditions: list[str]
    ica_policy: str


def paper_blocks(phase: str) -> list[PaperBlock]:
    """Return the predeclared blocks without inspecting any result."""
    if phase not in {"primary", "external", "ica-sensitivity", "all"}:
        raise ValueError(f"Unsupported paper phase: {phase}")
    blocks: list[PaperBlock] = []
    if phase in {"primary", "all"}:
        blocks.extend(
            [
                {
                    "dataset": "MI-OpenBCI",
                    "protocols": ["within_split", "within_session", "loso"],
                    "conditions": ["full"],
                    "ica_policy": "none",
                },
                {
                    "dataset": "MI-OpenBCI",
                    "protocols": ["within_split"],
                    "conditions": ["center", "nonoverlap", "overlap"],
                    "ica_policy": "none",
                },
            ]
        )
    if phase in {"external", "all"}:
        blocks.extend(
            {
                "dataset": dataset,
                "protocols": ["within_split", "cross_session"],
                "conditions": ["full"],
                "ica_policy": "none",
            }
            for dataset in ("Zhou2020", "Tavakolan2017")
        )
    if phase in {"ica-sensitivity", "all"}:
        blocks.append(
            {
                "dataset": "MI-OpenBCI",
                "protocols": ["within_split"],
                "conditions": ["full"],
                "ica_policy": "kurtosis",
            }
        )
    return blocks
