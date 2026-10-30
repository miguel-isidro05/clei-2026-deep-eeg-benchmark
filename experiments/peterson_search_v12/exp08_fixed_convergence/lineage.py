"""Verified Exp07 parent lineage for the isolated Exp08 repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp07_eegnet_fusion.analysis import load_final_decision
from experiments.peterson_search_v12.exp07_eegnet_fusion.catalog import E11_REFERENCE_ID


def load_exp07_parent(exp07_output_dir: Path) -> dict[str, Any]:
    completion_path = exp07_output_dir / "run_complete.json"
    if not completion_path.is_file():
        raise ValueError(f"Missing signed Exp07 completion marker: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    body = dict(completion)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError("Exp07 parent completion signature is invalid")
    if completion.get("status") != "completed" or completion.get("experiment") != "exp07":
        raise ValueError("Exp08 requires a completed Exp07 run")
    if completion.get("holdout_opened") is not False:
        raise ValueError("Exp07 parent opened the holdout")
    decision = load_final_decision(exp07_output_dir)
    if completion.get("final_decision_sha256") != decision.get("sha256"):
        raise ValueError("Exp07 completion is detached from its final decision")
    if decision.get("selected_config_id") != E11_REFERENCE_ID:
        raise ValueError("Exp08 requires Exp07 to retain e11")
    return decision

