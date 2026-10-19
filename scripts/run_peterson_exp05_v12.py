#!/usr/bin/env python3
"""Plan and run one leakage-safe Peterson Exp05 phase."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.common.identity import git_identity  # noqa: E402
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.analysis import (  # noqa: E402
    load_screen_decision,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.catalog import (  # noqa: E402
    build_catalog,
    catalog_payload,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.config import (  # noqa: E402
    Exp05Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.manifest import (  # noqa: E402
    build_confirm_manifest,
    build_screen_manifest,
    manifest_from_payload,
    manifest_payload,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.runner import (  # noqa: E402
    run_phase,
    write_run_failure,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = smoke_config() if args.smoke else Exp05Config()
    catalog = build_catalog()
    revision = str(git_identity(ROOT)["revision"] or "unknown")
    if args.phase == "screen":
        manifest = build_screen_manifest(config, catalog, revision=revision)
    else:
        decision = load_screen_decision(args.output_dir)
        screen_path = args.output_dir / "manifests" / "screen-manifest.json"
        screen = manifest_from_payload(json.loads(screen_path.read_text(encoding="utf-8")))
        expected_screen = build_screen_manifest(config, catalog, revision=revision)
        if screen.revision != revision or screen.sha256 != expected_screen.sha256:
            raise ValueError("Exp05 confirmation must use the same revision and screen contract")
        if decision.get("manifest_sha256") != screen.sha256:
            raise ValueError("Exp05 screen decision is detached from the screen manifest")
        promoted_ids = tuple(map(str, decision["selected_config_ids"]))
        manifest = build_confirm_manifest(
            config,
            catalog,
            promoted_ids=promoted_ids,
            revision=revision,
            screen_decision_sha256=str(decision["sha256"]),
        )
    write_json_atomic(args.output_dir / "manifests" / "catalog.json", catalog_payload(catalog))
    write_json_atomic(
        args.output_dir / "manifests" / f"{args.phase}-manifest.json",
        manifest_payload(manifest),
    )
    print(f"phase={args.phase} planned_cells={len(manifest.cells)}")
    if args.plan_only:
        return
    try:
        summary = run_phase(
            args.output_dir,
            config,
            catalog,
            manifest,
            phase=args.phase,
            device=args.device,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    except Exception as error:
        write_run_failure(
            args.output_dir,
            phase=args.phase,
            shard_index=args.shard_index,
            error=error,
        )
        raise
    print(
        f"exp05_{args.phase}_shard_complete={summary['completed']}/{summary['planned']} "
        f"failed={summary['failed']}"
    )
    if int(summary["failed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
