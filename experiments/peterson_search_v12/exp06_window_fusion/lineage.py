"""Verified parent-decision lineage for Exp06."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.audit import (
    validate_signed_decision,
)


def load_exp05_parent(exp05_output_dir: Path) -> dict[str, Any]:
    path = exp05_output_dir / "decisions" / "final-promotion.json"
    completion_path = exp05_output_dir / "run_complete.json"
    if not completion_path.is_file():
        raise ValueError(f"Missing signed Exp05 completion marker: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion_body = dict(completion)
    completion_signature = str(completion_body.pop("sha256", ""))
    if completion_signature != canonical_sha256(completion_body):
        raise ValueError("Exp05 parent completion signature is invalid")
    if completion.get("status") != "completed" or completion.get("experiment") != "exp05":
        raise ValueError("Exp05 parent is not a completed Exp05 run")
    if completion.get("holdout_opened") is not False:
        raise ValueError("Exp05 parent completion opened the holdout")
    payload = validate_signed_decision(
        path,
        phase="confirm",
        readout_path=exp05_output_dir / "analysis" / "confirm-readout.json",
        manifest_sha256=str(completion.get("confirm_manifest_sha256", "")),
    )
    if completion.get("final_decision_sha256") != payload.get("sha256"):
        raise ValueError("Exp05 parent completion is detached from its final decision")
    if payload.get("status") != "promoted" or payload.get("selected_config_id") != "exp05-e11":
        raise ValueError("Exp06 requires Exp05 to promote exp05-e11")
    if payload.get("holdout_opened") is not False:
        raise ValueError("Exp06 parent must keep the holdout closed")
    return payload
