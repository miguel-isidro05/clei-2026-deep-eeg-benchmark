#!/usr/bin/env python3
"""Audit full or bounded smoke outputs for Peterson V12."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.peterson_search_v12.common.audit import audit_results, validate_cell_payload
from experiments.peterson_search_v12.common.catalogs import build_exp01_catalog
from experiments.peterson_search_v12.common.config import SearchConfig
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.common.io import write_json_atomic
from experiments.peterson_search_v12.common.manifests import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-smoke-cells", type=int, default=2)
    args = parser.parse_args()
    manifest_payload = json.loads(
        (args.output_dir / "manifests" / "exp01.json").read_text(encoding="utf-8")
    )
    if args.smoke:
        paths = sorted((args.output_dir / "cells").glob("*.json"))
        if len(paths) != args.expected_smoke_cells:
            raise SystemExit(
                f"smoke incomplete: expected={args.expected_smoke_cells} found={len(paths)}"
            )
        for path in paths:
            validate_cell_payload(json.loads(path.read_text(encoding="utf-8")))
        report = {
            "status": "smoke_completed",
            "cells": len(paths),
            "manifest_sha256": manifest_payload["sha256"],
        }
        write_json_atomic(args.output_dir / "smoke_complete.json", report)
        print(f"peterson_search_smoke=OK cells={len(paths)}")
        return
    config = SearchConfig()
    manifest = build_manifest(
        config, build_exp01_catalog(config), revision=manifest_payload["revision"]
    )
    if manifest.sha256 != manifest_payload["sha256"]:
        raise SystemExit("manifest drift detected")
    report = audit_results(args.output_dir, manifest)
    report.update(
        {
            "status": "completed",
            "manifest_sha256": manifest.sha256,
            "cells_sha256": canonical_sha256(
                sorted(path.name for path in (args.output_dir / "cells").glob("*.json"))
            ),
        }
    )
    write_json_atomic(args.output_dir / "run_complete.json", report)
    print(f"peterson_search_audit=OK expected={report['expected']} found={report['found']}")


if __name__ == "__main__":
    main()
