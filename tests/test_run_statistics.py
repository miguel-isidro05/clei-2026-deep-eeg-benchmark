from __future__ import annotations

import json

from deepbench.paper_audit import audit_expected_cells
from deepbench.paper_profile import paper_profile_metadata
from deepbench.runner import _code_fingerprint, cell_path


def _write_manifest(root, name: str, phase: str, cells: list[str]) -> None:
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / name).write_text(
        json.dumps({**paper_profile_metadata(phase), "expected_cells": cells}),
        encoding="utf-8",
    )


def _write_cell(root, name: str) -> None:
    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    (cells / name).write_text("{}", encoding="utf-8")


def test_confirmatory_audit_requires_a_complete_paper_profile_manifest(tmp_path) -> None:
    _write_cell(tmp_path, "observed.json")
    audit, confirmatory, issues = audit_expected_cells(tmp_path)
    assert confirmatory is False
    assert "No paper expectation manifest" in " ".join(issues)
    assert audit.iloc[0].to_dict() == {
        "cell": "cells/observed.json",
        "expected": False,
        "present": True,
    }


def test_confirmatory_audit_rejects_extra_cells(tmp_path) -> None:
    expected = "cells/expected.json"
    _write_manifest(tmp_path, "paper-expected-all-shard0-of-1.json", "all", [expected])
    _write_cell(tmp_path, "expected.json")
    _write_cell(tmp_path, "extra.json")
    audit, confirmatory, issues = audit_expected_cells(tmp_path)
    assert confirmatory is False
    assert "unexpected" in " ".join(issues).lower()
    extra = audit.loc[audit["cell"] == "cells/extra.json"].iloc[0]
    assert bool(extra["expected"]) is False
    assert bool(extra["present"]) is True


def test_confirmatory_audit_accepts_exact_all_profile(tmp_path) -> None:
    destination = cell_path(
        tmp_path,
        "MI-OpenBCI",
        "within_split",
        "full",
        "EEGNet",
        0,
        "S02",
        "none",
    )
    expected = str(destination.relative_to(tmp_path))
    _write_manifest(tmp_path, "paper-expected-all-shard0-of-1.json", "all", [expected])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "dataset": "MI-OpenBCI",
                "protocol": "within_split",
                "condition": "full",
                "ica_policy": "none",
                "model": "EEGNet",
                "subject": "S02",
                "seed": 0,
                "run_configuration": {
                    "dataset": "MI-OpenBCI",
                    "protocol": "within_split",
                    "condition": "full",
                    "ica_policy": "none",
                    "model": "EEGNet",
                    "subject": "S02",
                    "seed": 0,
                    "epochs": 300,
                    "code_sha256": _code_fingerprint(),
                    "recipe": paper_profile_metadata("all")["recipes"]["EEGNet"],
                },
                "fold_reports": [
                    {"training_history": [{"epoch": epoch} for epoch in range(1, 301)]}
                ],
            }
        ),
        encoding="utf-8",
    )
    audit, confirmatory, issues = audit_expected_cells(tmp_path)
    assert confirmatory is True
    assert issues == []
    assert audit.iloc[0].to_dict() == {
        "cell": expected,
        "expected": True,
        "present": True,
    }


def test_confirmatory_audit_accepts_exact_no_loso_profile(tmp_path) -> None:
    destination = cell_path(
        tmp_path,
        "MI-OpenBCI",
        "within_split",
        "full",
        "EEGNet",
        0,
        "S02",
        "none",
    )
    expected = str(destination.relative_to(tmp_path))
    _write_manifest(
        tmp_path,
        "paper-expected-no-loso-shard0-of-1.json",
        "no-loso",
        [expected],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "dataset": "MI-OpenBCI",
                "protocol": "within_split",
                "condition": "full",
                "ica_policy": "none",
                "model": "EEGNet",
                "subject": "S02",
                "seed": 0,
                "run_configuration": {
                    "dataset": "MI-OpenBCI",
                    "protocol": "within_split",
                    "condition": "full",
                    "ica_policy": "none",
                    "model": "EEGNet",
                    "subject": "S02",
                    "seed": 0,
                    "epochs": 300,
                    "code_sha256": _code_fingerprint(),
                    "recipe": paper_profile_metadata("no-loso")["recipes"]["EEGNet"],
                },
                "fold_reports": [
                    {"training_history": [{"epoch": epoch} for epoch in range(1, 301)]}
                ],
            }
        ),
        encoding="utf-8",
    )

    audit, confirmatory, issues = audit_expected_cells(tmp_path)

    assert confirmatory is True
    assert issues == []
    assert audit.iloc[0].to_dict() == {
        "cell": expected,
        "expected": True,
        "present": True,
    }


def test_confirmatory_audit_rejects_non_frozen_epoch_count(tmp_path) -> None:
    destination = cell_path(
        tmp_path, "MI-OpenBCI", "within_split", "full", "EEGNet", 0, "S02", "none"
    )
    expected = str(destination.relative_to(tmp_path))
    _write_manifest(tmp_path, "paper-expected-all-shard0-of-1.json", "all", [expected])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "dataset": "MI-OpenBCI",
                "protocol": "within_split",
                "condition": "full",
                "ica_policy": "none",
                "model": "EEGNet",
                "subject": "S02",
                "seed": 0,
                "run_configuration": {
                    "dataset": "MI-OpenBCI",
                    "protocol": "within_split",
                    "condition": "full",
                    "ica_policy": "none",
                    "model": "EEGNet",
                    "subject": "S02",
                    "seed": 0,
                    "epochs": 3,
                    "recipe": {},
                },
                "fold_reports": [{"training_history": [{"epoch": 1}]}],
            }
        ),
        encoding="utf-8",
    )
    _, confirmatory, issues = audit_expected_cells(tmp_path)
    assert confirmatory is False
    assert "exactly 300 epochs" in " ".join(issues)


def test_confirmatory_audit_rejects_cells_from_different_code(tmp_path) -> None:
    destination = cell_path(
        tmp_path, "MI-OpenBCI", "within_split", "full", "EEGNet", 0, "S02", "none"
    )
    expected = str(destination.relative_to(tmp_path))
    _write_manifest(tmp_path, "paper-expected-all-shard0-of-1.json", "all", [expected])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "dataset": "MI-OpenBCI",
                "protocol": "within_split",
                "condition": "full",
                "ica_policy": "none",
                "model": "EEGNet",
                "subject": "S02",
                "seed": 0,
                "run_configuration": {
                    "dataset": "MI-OpenBCI",
                    "protocol": "within_split",
                    "condition": "full",
                    "ica_policy": "none",
                    "model": "EEGNet",
                    "subject": "S02",
                    "seed": 0,
                    "epochs": 300,
                    "code_sha256": "stale-code",
                    "recipe": paper_profile_metadata("all")["recipes"]["EEGNet"],
                },
                "fold_reports": [
                    {"training_history": [{"epoch": epoch} for epoch in range(1, 301)]}
                ],
            }
        ),
        encoding="utf-8",
    )
    _, confirmatory, issues = audit_expected_cells(tmp_path)
    assert confirmatory is False
    assert "different scientific code version" in " ".join(issues)


def test_confirmatory_audit_rejects_partial_phase_coverage(tmp_path) -> None:
    expected = "cells/expected.json"
    _write_manifest(tmp_path, "paper-expected-primary-shard0-of-1.json", "primary", [expected])
    _write_cell(tmp_path, "expected.json")
    _, confirmatory, issues = audit_expected_cells(tmp_path)
    assert confirmatory is False
    assert "Incomplete paper phase coverage" in " ".join(issues)
