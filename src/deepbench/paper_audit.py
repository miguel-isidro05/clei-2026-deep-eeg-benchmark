"""Integrity gate for confirmatory paper result directories."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_PHASES = {"primary", "external", "ica-sensitivity"}


def audit_expected_cells(results_dir: Path) -> tuple[pd.DataFrame, bool, list[str]]:
    """Compare the exact observed cell set with a complete paper-profile declaration."""
    manifest_paths = sorted((results_dir / "manifests").glob("paper-expected-*.json"))
    expected: set[str] = set()
    phases: set[str] = set()
    issues: list[str] = []
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected.update(map(str, payload.get("expected_cells", [])))
        if phase := payload.get("phase"):
            phases.add(str(phase))

    present = {
        str(path.relative_to(results_dir)) for path in (results_dir / "cells").glob("*.json")
    }
    if not manifest_paths:
        issues.append("No paper expectation manifest was found")
    elif "all" not in phases and not REQUIRED_PHASES.issubset(phases):
        issues.append(
            "Incomplete paper phase coverage: require phase='all' or primary, external, "
            "and ica-sensitivity manifests"
        )

    missing = expected - present
    unexpected = present - expected
    if missing:
        issues.append(f"{len(missing)} expected paper cells are missing")
    if unexpected:
        issues.append(f"{len(unexpected)} unexpected cells are outside the paper profile")

    audit = pd.DataFrame(
        [
            {"cell": cell, "expected": cell in expected, "present": cell in present}
            for cell in sorted(expected | present)
        ],
        columns=["cell", "expected", "present"],
    )
    return audit, not issues, issues
