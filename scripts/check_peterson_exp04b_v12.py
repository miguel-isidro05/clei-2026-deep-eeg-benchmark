#!/usr/bin/env python3
"""Audit Exp04b completeness and write the final completion marker."""

from __future__ import annotations

import argparse
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
from experiments.peterson_search_v12.exp04b_safepath.audit import audit_wave  # noqa: E402
from experiments.peterson_search_v12.exp04b_safepath.catalog import build_w2_catalog  # noqa: E402
from experiments.peterson_search_v12.exp04b_safepath.config import (  # noqa: E402
    Exp04bConfig,
    smoke_config,
)
from experiments.peterson_search_v12.exp04b_safepath.decisions import (  # noqa: E402
    load_final_decision,
    load_w2_selection,
)
from experiments.peterson_search_v12.exp04b_safepath.identity import (  # noqa: E402
    exp04b_code_sha256,
)
from experiments.peterson_search_v12.exp04b_safepath.manifest import (  # noqa: E402
    build_search_manifest,
)


def _selection(output_dir: Path, *, require: bool) -> tuple[str, str, str]:
    try:
        return load_w2_selection(output_dir)
    except ValueError:
        if require:
            raise
    return tuple(candidate.config_id for candidate in build_w2_catalog()[:3])  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wave", choices=("w1", "w2", "w3", "all"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = smoke_config() if args.smoke else Exp04bConfig()
    selected = _selection(args.output_dir, require=args.wave in {"w3", "all"})
    revision = str(git_identity(ROOT)["revision"] or "unknown")
    manifest = build_search_manifest(config, selected_w2=selected, revision=revision)
    waves = ("w1", "w2", "w3") if args.wave == "all" else (args.wave,)
    reports = [
        audit_wave(args.output_dir, manifest, wave, allow_dirty=args.smoke) for wave in waves
    ]
    if args.wave == "all":
        final_decision = load_final_decision(args.output_dir)
        body = {
            "status": "completed",
            "experiment": "exp04b",
            "revision": revision,
            "manifest_sha256": manifest.sha256,
            "code_sha256": exp04b_code_sha256(ROOT),
            "w2_selection_sha256": canonical_sha256(list(selected)),
            "final_decision_sha256": str(final_decision["sha256"]),
            "audits": reports,
        }
        write_json_atomic(
            args.output_dir / "run_complete.json",
            {**body, "sha256": canonical_sha256(body)},
        )
    print(f"exp04b_audit=OK wave={args.wave} reports={len(reports)}")


if __name__ == "__main__":
    main()
