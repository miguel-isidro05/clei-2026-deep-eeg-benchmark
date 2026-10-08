from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from deepbench.peterson_profile import expected_cell_names, profile_metadata
from deepbench.runner import _code_fingerprint


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "write_run_report.py"
    spec = importlib.util.spec_from_file_location("write_run_report_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_run_report_uses_peterson_integrity_gate(tmp_path, monkeypatch) -> None:
    module = _load_script()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    metadata = profile_metadata()
    metadata["scientific_code_sha256"] = _code_fingerprint()
    metadata["expected_cells"] = expected_cell_names()
    (manifests / "peterson-journal-expected.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_run_report.py",
            "--results-dir",
            str(tmp_path),
            "--status",
            "failed",
        ],
    )
    module.main()

    report = (tmp_path / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    assert "peterson_journal_v10" in report
    assert "3250 expected Peterson cells are missing" in report
