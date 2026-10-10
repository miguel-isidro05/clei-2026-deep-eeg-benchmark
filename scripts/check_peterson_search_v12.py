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

from experiments.peterson_search_v12.common.audit import (  # noqa: E402
    audit_results,
    validate_cell_payload,
)
from experiments.peterson_search_v12.common.catalogs import (  # noqa: E402
    build_exp01_catalog,
)
from experiments.peterson_search_v12.common.config import SearchConfig  # noqa: E402
from experiments.peterson_search_v12.common.identity import (  # noqa: E402
    canonical_sha256,
    code_sha256,
)
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.common.manifests import build_manifest  # noqa: E402


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
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.output_dir / "cells").glob("*.json"))
    ]
    expected_code_hash = code_sha256(ROOT)
    if {payload["fingerprint"]["code_sha256"] for payload in payloads} != {expected_code_hash}:
        raise SystemExit("result code hash does not match the current V12 implementation")
    report.update(
        {
            "status": "completed",
            "manifest_sha256": manifest.sha256,
            "code_sha256": expected_code_hash,
            "cells_sha256": canonical_sha256(payloads),
        }
    )
    write_json_atomic(args.output_dir / "run_complete.json", report)
    print(f"peterson_search_audit=OK expected={report['expected']} found={report['found']}")


if __name__ == "__main__":
    main()
