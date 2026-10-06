"""Integrity gate for the frozen Peterson journal profile."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import PAPER_EPOCHS, PAPER_SEEDS, PETERSON_MODEL_NAMES
from .models import recipe_dict
from .peterson_profile import expected_cell_names


def audit_peterson_results(results_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    manifest_path = results_dir / "manifests" / "peterson-journal-expected.json"
    issues: list[str] = []
    if not manifest_path.exists():
        return pd.DataFrame(), ["missing Peterson expectation manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = set(expected_cell_names())
    declared = set(map(str, manifest.get("expected_cells", [])))
    if declared != expected:
        issues.append("expectation manifest does not match the frozen Peterson profile")
    present = {
        str(path.relative_to(results_dir)) for path in (results_dir / "cells").glob("*.json")
    }
    missing = expected - present
    unexpected = present - expected
    if missing:
        issues.append(f"{len(missing)} expected Peterson cells are missing")
    if unexpected:
        issues.append(f"{len(unexpected)} cells are outside the Peterson profile")
    code_sha256 = manifest.get("scientific_code_sha256")
    for relative in sorted(expected & present):
        payload = json.loads((results_dir / relative).read_text(encoding="utf-8"))
        configuration = payload.get("run_configuration", {})
        model = str(payload.get("model"))
        if payload.get("dataset") != "MI-OpenBCI":
            issues.append(f"{relative} is not MI-OpenBCI")
        if model not in PETERSON_MODEL_NAMES:
            issues.append(f"{relative} has unsupported model={model!r}")
            continue
        expected_epochs = 0 if model == "CSP+LDA" else PAPER_EPOCHS
        if configuration.get("epochs") != expected_epochs:
            issues.append(f"{relative} has epochs={configuration.get('epochs')}")
        if configuration.get("code_sha256") != code_sha256:
            issues.append(f"{relative} has a different scientific code fingerprint")
        if configuration.get("recipe") != recipe_dict(model, expected_epochs):
            issues.append(f"{relative} has a non-frozen recipe")
        if payload.get("seed") not in PAPER_SEEDS:
            issues.append(f"{relative} has a seed outside the frozen profile")
        reports = payload.get("fold_reports")
        if not isinstance(reports, list) or not reports:
            issues.append(f"{relative} has no fold reports")
            continue
        for fold_index, report in enumerate(reports):
            history = report.get("training_history", [])
            if model == "CSP+LDA":
                if history or report.get("estimator_family") != "classical":
                    issues.append(f"{relative} fold {fold_index} has invalid CSP metadata")
            elif len(history) != PAPER_EPOCHS or history[-1].get("epoch") != PAPER_EPOCHS:
                issues.append(f"{relative} fold {fold_index} did not reach {PAPER_EPOCHS}")
    table = pd.DataFrame(
        [
            {"cell": cell, "expected": cell in expected, "present": cell in present}
            for cell in sorted(expected | present)
        ]
    )
    return table, issues
