#!/usr/bin/env python3
"""Audit Peterson Exp05 and write a portable completion marker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.common.identity import canonical_sha256  # noqa: E402
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.audit import (  # noqa: E402
    audit_phase,
    validate_signed_decision,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.catalog import (  # noqa: E402
    build_catalog,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.config import (  # noqa: E402
    Exp05Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.manifest import (  # noqa: E402
    build_confirm_manifest,
    build_screen_manifest,
    manifest_from_payload,
)


def _load(path: Path):  # type: ignore[no-untyped-def]
    if not path.is_file():
        raise ValueError(f"Missing Exp05 manifest: {path}")
    return manifest_from_payload(json.loads(path.read_text(encoding="utf-8")))


def _assert_current_contract(output_dir: Path, phase: str, config: Exp05Config):  # type: ignore[no-untyped-def]
    stored = _load(output_dir / "manifests" / f"{phase}-manifest.json")
    catalog = build_catalog()
    if phase == "screen":
        expected = build_screen_manifest(config, catalog, revision=stored.revision)
    else:
        screen = _load(output_dir / "manifests" / "screen-manifest.json")
        decision = validate_signed_decision(
            output_dir / "decisions" / "screen-promotion.json",
            phase="screen",
            readout_path=output_dir / "analysis" / "screen-readout.json",
            manifest_sha256=screen.sha256,
        )
        if (
            stored.revision != screen.revision
            or stored.config_sha256 != screen.config_sha256
            or stored.catalog_sha256 != screen.catalog_sha256
        ):
            raise ValueError("Exp05 screen and confirmation identities differ")
        expected = build_confirm_manifest(
            config,
            catalog,
            promoted_ids=tuple(map(str, decision["selected_config_ids"])),
            revision=stored.revision,
            screen_decision_sha256=str(decision["sha256"]),
        )
    if stored.sha256 != expected.sha256:
        raise ValueError(f"Stored Exp05 {phase} manifest differs from the current contract")
    return stored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm", "all"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = smoke_config() if args.smoke else Exp05Config()
    phases = ("screen", "confirm") if args.phase == "all" else (args.phase,)
    manifests = {
        phase: _assert_current_contract(args.output_dir, phase, config) for phase in phases
    }
    reports = [
        audit_phase(args.output_dir, manifests[phase], allow_dirty=args.smoke) for phase in phases
    ]
    if args.phase in {"confirm", "all"}:
        screen_decision = validate_signed_decision(
            args.output_dir / "decisions" / "screen-promotion.json",
            phase="screen",
            readout_path=args.output_dir / "analysis" / "screen-readout.json",
            manifest_sha256=(
                manifests["screen"].sha256
                if "screen" in manifests
                else _load(args.output_dir / "manifests" / "screen-manifest.json").sha256
            ),
        )
        if manifests["confirm"].screen_decision_sha256 != screen_decision["sha256"]:
            raise ValueError("Exp05 confirmation is detached from the screen decision")
    if args.phase == "all":
        final = validate_signed_decision(
            args.output_dir / "decisions" / "final-promotion.json",
            phase="confirm",
            readout_path=args.output_dir / "analysis" / "confirm-readout.json",
            manifest_sha256=manifests["confirm"].sha256,
        )
        if final.get("screen_decision_sha256") != screen_decision["sha256"]:
            raise ValueError("Exp05 final decision is detached from the screen decision")
        if reports[0]["code_sha256"] != reports[1]["code_sha256"]:
            raise ValueError("Exp05 screen and confirmation used different code surfaces")
        if reports[0]["data_sha256_by_subject"] != reports[1]["data_sha256_by_subject"]:
            raise ValueError("Exp05 screen and confirmation used different source data")
        body = {
            "status": "completed",
            "experiment": "exp05",
            "revision": manifests["screen"].revision,
            "holdout_opened": False,
            "screen_manifest_sha256": manifests["screen"].sha256,
            "confirm_manifest_sha256": manifests["confirm"].sha256,
            "final_decision_sha256": str(final["sha256"]),
            "audits": reports,
        }
        write_json_atomic(
            args.output_dir / "run_complete.json",
            {**body, "sha256": canonical_sha256(body)},
        )
    print(f"exp05_audit=OK phase={args.phase} reports={len(reports)}")


if __name__ == "__main__":
    main()
