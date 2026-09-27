from __future__ import annotations

import json

from deepbench.paper_audit import audit_expected_cells


def _write_manifest(root, name: str, phase: str, cells: list[str]) -> None:
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / name).write_text(
        json.dumps({"phase": phase, "expected_cells": cells}), encoding="utf-8"
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
    expected = "cells/expected.json"
    _write_manifest(tmp_path, "paper-expected-all-shard0-of-1.json", "all", [expected])
    _write_cell(tmp_path, "expected.json")
    audit, confirmatory, issues = audit_expected_cells(tmp_path)
    assert confirmatory is True
    assert issues == []
    assert audit.iloc[0].to_dict() == {
        "cell": expected,
        "expected": True,
        "present": True,
    }


def test_confirmatory_audit_rejects_partial_phase_coverage(tmp_path) -> None:
    expected = "cells/expected.json"
    _write_manifest(
        tmp_path, "paper-expected-primary-shard0-of-1.json", "primary", [expected]
    )
    _write_cell(tmp_path, "expected.json")
    _, confirmatory, issues = audit_expected_cells(tmp_path)
    assert confirmatory is False
    assert "Incomplete paper phase coverage" in " ".join(issues)
