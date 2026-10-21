"""Verified Exp06 parent lineage for Exp07."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp06_window_fusion.analysis import load_final_decision
from experiments.peterson_search_v12.exp06_window_fusion.catalog import E11_REFERENCE_ID


def load_exp06_parent(exp06_output_dir: Path) -> dict[str, Any]:
    completion_path = exp06_output_dir / "run_complete.json"
    if not completion_path.is_file():
        raise ValueError(f"Missing signed Exp06 completion marker: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion_body = dict(completion)
    completion_signature = str(completion_body.pop("sha256", ""))
    if completion_signature != canonical_sha256(completion_body):
        raise ValueError("Exp06 parent completion signature is invalid")
    if completion.get("status") != "completed" or completion.get("experiment") != "exp06":
        raise ValueError("Exp07 requires a completed Exp06 run")
    if completion.get("holdout_opened") is not False:
        raise ValueError("Exp06 parent completion opened the holdout")
    raw_decision = json.loads(
        (exp06_output_dir / "decisions" / "final-promotion.json").read_text(encoding="utf-8")
    )
    if raw_decision.get("selected_config_id") != E11_REFERENCE_ID:
        raise ValueError("Exp07 requires the verified Exp06 e11 reference")
    decision = load_final_decision(exp06_output_dir)
    if completion.get("final_decision_sha256") != decision.get("sha256"):
        raise ValueError("Exp06 completion is detached from its final decision")
    if decision.get("selected_config_id") != E11_REFERENCE_ID:
        raise ValueError("Exp07 requires Exp06 to retain e11")
    if decision.get("holdout_opened") is not False:
        raise ValueError("Exp07 parent must keep the holdout closed")
    return decision
