#!/usr/bin/env python3
"""Plan and execute one Exp08 fixed-convergence phase."""

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
from experiments.peterson_search_v12.exp08_fixed_convergence.analysis import (  # noqa: E402
    load_screen_decision,
)
from experiments.peterson_search_v12.exp08_fixed_convergence.catalog import (  # noqa: E402
    catalog_payload,
    experimental_candidates,
)
from experiments.peterson_search_v12.exp08_fixed_convergence.config import (  # noqa: E402
    Exp08Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp08_fixed_convergence.lineage import (  # noqa: E402
    load_exp07_parent,
)
from experiments.peterson_search_v12.exp08_fixed_convergence.manifest import (  # noqa: E402
    build_confirm_manifest,
    build_screen_manifest,
    manifest_from_payload,
    manifest_payload,
)
from experiments.peterson_search_v12.exp08_fixed_convergence.runner import (  # noqa: E402
    run_phase,
    write_run_failure,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp07-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = smoke_config() if args.smoke else Exp08Config()
    catalog = experimental_candidates()
    revision = str(git_identity(ROOT)["revision"] or "unknown")
    parent_sha = str(load_exp07_parent(args.exp07_dir)["sha256"])
    if args.phase == "screen":
        manifest = build_screen_manifest(
            config, catalog, revision=revision, parent_decision_sha256=parent_sha
        )
    else:
        decision = load_screen_decision(args.output_dir)
        screen = manifest_from_payload(
            json.loads(
                (args.output_dir / "manifests" / "screen-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        expected = build_screen_manifest(
            config, catalog, revision=revision, parent_decision_sha256=parent_sha
        )
        if screen != expected or decision.get("manifest_sha256") != screen.sha256:
            raise ValueError("Exp08 confirmation is detached from the frozen screen")
        manifest = build_confirm_manifest(
            config,
            catalog,
            promoted_backbones=tuple(map(str, decision["selected_backbones"])),
            revision=revision,
            parent_decision_sha256=parent_sha,
            screen_decision_sha256=str(decision["sha256"]),
        )
    write_json_atomic(args.output_dir / "manifests" / "catalog.json", catalog_payload())
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
            args.output_dir, phase=args.phase, shard_index=args.shard_index, error=error
        )
        raise
    print(
        f"exp08_{args.phase}_shard_complete={summary['completed']}/{summary['planned']} "
        f"failed={summary['failed']}"
    )


if __name__ == "__main__":
    main()

