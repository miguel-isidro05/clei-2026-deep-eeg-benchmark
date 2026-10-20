#!/usr/bin/env python3
"""Audit Exp06 and write a signed completion marker."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.common.identity import (  # noqa: E402
    canonical_sha256,
    git_identity,
)
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.exp06_window_fusion.analysis import (  # noqa: E402
    load_final_decision,
)
from experiments.peterson_search_v12.exp06_window_fusion.audit import audit_phase  # noqa: E402
from experiments.peterson_search_v12.exp06_window_fusion.catalog import build_catalog  # noqa: E402
from experiments.peterson_search_v12.exp06_window_fusion.config import (  # noqa: E402
    Exp06Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp06_window_fusion.lineage import (  # noqa: E402
    load_exp05_parent,
)
from experiments.peterson_search_v12.exp06_window_fusion.runner import (  # noqa: E402
    exp06_code_sha256,
)


def _cells_sha256(output_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(output_dir.glob("*_cells/*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp05-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm", "all"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = smoke_config() if args.smoke else Exp06Config()
    catalog = build_catalog()
    revision = str(git_identity(ROOT)["revision"] or "unknown")
    parent_sha = str(load_exp05_parent(args.exp05_dir)["sha256"])
    phases = ("screen", "confirm") if args.phase == "all" else (args.phase,)
    reports = [
        audit_phase(
            args.output_dir,
            config,
            catalog,
            phase=phase,
            revision=revision,
            parent_decision_sha256=parent_sha,
        )
        for phase in phases
    ]
    if args.phase == "all":
        final = load_final_decision(args.output_dir)
        body = {
            "status": "completed",
            "experiment": "exp06",
            "revision": revision,
            "parent_decision_sha256": parent_sha,
            "code_sha256": exp06_code_sha256(ROOT),
            "cells_sha256": _cells_sha256(args.output_dir),
            "audits": reports,
            "final_decision_sha256": final["sha256"],
            "holdout_opened": False,
        }
        write_json_atomic(
            args.output_dir / "run_complete.json",
            {**body, "sha256": canonical_sha256(body)},
        )
    print(f"exp06_audit=OK phase={args.phase} reports={len(reports)}")


if __name__ == "__main__":
    main()
