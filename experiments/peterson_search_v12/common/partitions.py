"""Frozen discovery/holdout partition for Peterson V12."""

from __future__ import annotations

import hashlib

ALL_SUBJECTS = ("S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S12")
PARTITION_SALT = "peterson-v12-subject-partition-20261010"
DISCOVERY_SUBJECTS = ("S09", "S03", "S02", "S10", "S12", "S08")
HOLDOUT_SUBJECTS = ("S07", "S04", "S05", "S06")


def partition_subjects() -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(
        sorted(
            ALL_SUBJECTS,
            key=lambda subject: hashlib.sha256(
                f"{PARTITION_SALT}:{subject}".encode("utf-8")
            ).hexdigest(),
        )
    )
    discovery, holdout = ordered[:6], ordered[6:]
    validate_partition(discovery, holdout)
    if discovery != DISCOVERY_SUBJECTS or holdout != HOLDOUT_SUBJECTS:
        raise RuntimeError("Peterson V12 subject partition drifted")
    return discovery, holdout


def validate_partition(discovery: tuple[str, ...], holdout: tuple[str, ...]) -> None:
    if set(discovery) & set(holdout):
        raise ValueError("Discovery and holdout subjects overlap")
    if set(discovery) | set(holdout) != set(ALL_SUBJECTS):
        raise ValueError("Discovery and holdout must cover every Peterson subject exactly once")
    if len(discovery) != 6 or len(holdout) != 4:
        raise ValueError("Peterson V12 requires a 6/4 subject partition")

