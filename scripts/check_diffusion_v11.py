#!/usr/bin/env python3
"""Audit a completed Peterson diffusion V11 wave against its frozen manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.peterson_diffusion_v11.audit import audit_cells  # noqa: E402
from experiments.peterson_diffusion_v11.io import write_json_atomic  # noqa: E402
from experiments.peterson_diffusion_v11.manifests import CellSpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-complete", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifests" / "expected.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing V11 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    cells = [
        CellSpec(**{key: value for key, value in item.items() if key != "cell_id"})
        for item in manifest["cells"]
    ]
    report = audit_cells(
        args.output_dir,
        cells,
        expected_config_sha256=str(manifest["config_sha256"]),
        expected_code_sha256=str(manifest["code_sha256"]),
    )
    if args.write_complete:
        write_json_atomic(
            args.output_dir / "run_complete.json",
            {
                "status": "completed",
                "audit": report,
                "config_sha256": manifest["config_sha256"],
                "cells_sha256": manifest["cells_sha256"],
                "code_sha256": manifest["code_sha256"],
                "git": manifest["git"],
            },
        )
    print(
        f"diffusion_v11_audit=OK expected={report['expected']} found={report['found']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
