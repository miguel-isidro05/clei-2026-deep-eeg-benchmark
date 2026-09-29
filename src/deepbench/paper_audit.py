"""Integrity gate for confirmatory paper result directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import MODEL_NAMES, PAPER_EPOCHS, PAPER_SEEDS
from .models import recipe_dict
from .paper_profile import paper_profile_metadata
from .runner import _code_fingerprint, cell_path

REQUIRED_PHASES = {"primary", "external", "ica-sensitivity"}


def _json_canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def result_cells_sha256(results_dir: Path) -> str:
    """Hash cell names and contents so derived artifacts cannot silently go stale."""
    digest = hashlib.sha256()
    for path in sorted((results_dir / "cells").glob("*.json")):
        digest.update(str(path.relative_to(results_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_manifest_profile(payload: dict[str, object], path: Path) -> list[str]:
    phase = str(payload.get("phase", ""))
    if phase not in {*REQUIRED_PHASES, "all"}:
        return [f"{path.name} has unsupported phase={phase!r}"]
    expected = paper_profile_metadata(phase)
    keys = (
        "profile_version",
        "models",
        "seeds",
        "epochs",
        "blocks",
        "recipes",
        "scientific_code_sha256",
        "profile_sha256",
    )
    return [
        f"{path.name} does not match the frozen paper profile field {key!r}"
        for key in keys
        if _json_canonical(payload.get(key)) != _json_canonical(expected[key])
    ]


def _validate_present_cells(results_dir: Path, present: set[str]) -> list[str]:
    issues: list[str] = []
    expected_recipes = {model: recipe_dict(model, PAPER_EPOCHS) for model in MODEL_NAMES}
    for relative in sorted(present):
        path = results_dir / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"Unreadable result cell {relative}: {exc}")
            continue
        configuration = payload.get("run_configuration")
        if not isinstance(configuration, dict):
            issues.append(f"{relative} is missing run_configuration")
            continue
        model = str(payload.get("model", ""))
        if model not in MODEL_NAMES:
            issues.append(f"{relative} has model outside the frozen profile: {model!r}")
            continue
        if configuration.get("epochs") != PAPER_EPOCHS:
            issues.append(f"{relative} was not trained for exactly {PAPER_EPOCHS} epochs")
        if configuration.get("code_sha256") != _code_fingerprint():
            issues.append(f"{relative} was produced by a different scientific code version")
        if _json_canonical(configuration.get("recipe")) != _json_canonical(expected_recipes[model]):
            issues.append(f"{relative} has a non-frozen training recipe for {model}")
        if payload.get("seed") not in PAPER_SEEDS:
            issues.append(f"{relative} has a seed outside the frozen profile")
        identity_keys = (
            "dataset",
            "protocol",
            "condition",
            "model",
            "seed",
            "subject",
            "ica_policy",
        )
        mismatched = [key for key in identity_keys if configuration.get(key) != payload.get(key)]
        if mismatched:
            issues.append(f"{relative} has payload/config identity mismatches: {mismatched}")
        expected_relative = str(
            cell_path(
                results_dir,
                str(payload.get("dataset")),
                str(payload.get("protocol")),
                str(payload.get("condition")),
                model,
                int(payload.get("seed", -1)),
                str(payload.get("subject")),
                str(payload.get("ica_policy")),
            ).relative_to(results_dir)
        )
        if relative != expected_relative:
            issues.append(f"{relative} does not match its serialized cell identity")
        reports = payload.get("fold_reports")
        if not isinstance(reports, list) or not reports:
            issues.append(f"{relative} has no fold reports")
            continue
        for fold_index, report in enumerate(reports):
            history = report.get("training_history") if isinstance(report, dict) else None
            if not isinstance(history, list) or len(history) != PAPER_EPOCHS:
                issues.append(
                    f"{relative} fold {fold_index} does not contain {PAPER_EPOCHS} epochs"
                )
                continue
            if history[-1].get("epoch") != PAPER_EPOCHS:
                issues.append(f"{relative} fold {fold_index} did not reach epoch {PAPER_EPOCHS}")
    return issues


def audit_expected_cells(results_dir: Path) -> tuple[pd.DataFrame, bool, list[str]]:
    """Compare the exact observed cell set with a complete paper-profile declaration."""
    manifest_paths = sorted((results_dir / "manifests").glob("paper-expected-*.json"))
    expected: set[str] = set()
    phases: set[str] = set()
    issues: list[str] = []
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        issues.extend(_validate_manifest_profile(payload, path))
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
    issues.extend(_validate_present_cells(results_dir, present & expected))

    audit = pd.DataFrame(
        [
            {"cell": cell, "expected": cell in expected, "present": cell in present}
            for cell in sorted(expected | present)
        ],
        columns=["cell", "expected", "present"],
    )
    return audit, not issues, issues
