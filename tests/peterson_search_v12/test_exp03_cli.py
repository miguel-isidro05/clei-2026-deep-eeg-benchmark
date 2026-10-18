from __future__ import annotations

import json

import pytest

from experiments.peterson_search_v12.common.identity import canonical_sha256
from scripts.run_peterson_exp03_v12 import _decision


def _write_decision(tmp_path, *, status: str = "selected"):
    body = {
        "status": status,
        "selected_recipe": {"recipe_id": "R0-last-wd0"},
        "selection_summary": {"mean_accuracy": 0.8},
        "parity": {"pass": status == "selected"},
    }
    payload = {**body, "sha256": canonical_sha256(body)}
    path = tmp_path / "decisions" / "recipe_decision.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_quality_phase_requires_an_authentic_selected_recipe_decision(tmp_path):
    path = _write_decision(tmp_path)
    assert _decision(tmp_path)["status"] == "selected"

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["selected_recipe"]["recipe_id"] = "R4-exp02-replay"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(SystemExit, match="signature"):
        _decision(tmp_path)

    _write_decision(tmp_path, status="blocked_parity")
    with pytest.raises(SystemExit, match="parity"):
        _decision(tmp_path)
