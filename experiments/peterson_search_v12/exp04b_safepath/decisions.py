"""Fail-closed readers for staged Exp04b decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.peterson_search_v12.common.identity import canonical_sha256

from .catalog import build_w2_catalog


def _load_signed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing signed decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    signature = str(body.pop("sha256", ""))
    if signature != canonical_sha256(body):
        raise ValueError(f"Decision signature mismatch: {path}")
    return payload


def load_w2_selection(output_dir: Path) -> tuple[str, str, str]:
    payload = _load_signed(output_dir / "decisions" / "w2-selection.json")
    selected = tuple(str(value) for value in payload.get("selected_config_ids", []))
    valid = {candidate.config_id for candidate in build_w2_catalog()}
    if len(selected) != 3 or len(set(selected)) != 3 or not set(selected) <= valid:
        raise ValueError("W2 selection does not contain three distinct signed candidates")
    readout = output_dir / "analysis" / "w2-readout.json"
    if not readout.is_file():
        raise ValueError("W2 selection is missing its analysis readout")
    report = json.loads(readout.read_text(encoding="utf-8"))
    if payload.get("readout_sha256") != canonical_sha256(report):
        raise ValueError("W2 selection does not match its analysis readout")
    return selected  # type: ignore[return-value]


def load_final_decision(output_dir: Path) -> dict[str, Any]:
    payload = _load_signed(output_dir / "decisions" / "final-promotion.json")
    readout = output_dir / "analysis" / "w3-readout.json"
    if not readout.is_file():
        raise ValueError("Final promotion is missing its W3 analysis readout")
    report = json.loads(readout.read_text(encoding="utf-8"))
    if payload.get("readout_sha256") != canonical_sha256(report):
        raise ValueError("Final promotion does not match its W3 analysis readout")
    if payload.get("status") not in {"promoted", "no_promotion"}:
        raise ValueError("Final promotion has an unknown status")
    return payload
