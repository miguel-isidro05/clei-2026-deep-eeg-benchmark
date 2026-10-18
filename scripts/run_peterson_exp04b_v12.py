#!/usr/bin/env python3
"""Plan and execute one Peterson Exp04b wave."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.common.identity import git_identity  # noqa: E402
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.exp04b_safepath.catalog import (  # noqa: E402
    build_w1_catalog,
    build_w2_catalog,
    build_w3_catalog,
    catalog_payload,
)
from experiments.peterson_search_v12.exp04b_safepath.config import (  # noqa: E402
    Exp04bConfig,
    smoke_config,
)
from experiments.peterson_search_v12.exp04b_safepath.decisions import (  # noqa: E402
    load_w2_selection,
)
from experiments.peterson_search_v12.exp04b_safepath.manifest import (  # noqa: E402
    build_search_manifest,
    manifest_payload,
)
from experiments.peterson_search_v12.exp04b_safepath.runner import run_wave  # noqa: E402
from scripts.run_peterson_exp04_v12 import _load_source  # noqa: E402


def _selected(output_dir: Path, *, require: bool) -> tuple[str, str, str]:
    try:
        return load_w2_selection(output_dir)
    except ValueError:
        if require:
            raise
    return tuple(candidate.config_id for candidate in build_w2_catalog()[:3])  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp03-dir", type=Path, required=True)
    parser.add_argument("--wave", choices=("w1", "w2", "w3"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = smoke_config() if args.smoke else Exp04bConfig()
    selected = _selected(args.output_dir, require=args.wave == "w3")
    source = _load_source(args.exp03_dir, smoke=args.smoke)
    revision = str(git_identity(ROOT)["revision"] or "unknown")
    manifest = build_search_manifest(config, selected_w2=selected, revision=revision)
    catalogs = {
        "w1": build_w1_catalog(),
        "w2": build_w2_catalog(),
        "w3": build_w3_catalog(selected),
    }
    catalog = catalogs[args.wave]
    write_json_atomic(args.output_dir / "manifests" / "source.json", source.payload())
    write_json_atomic(
        args.output_dir / "manifests" / f"catalog-{args.wave}.json",
        catalog_payload(catalog),
    )
    write_json_atomic(
        args.output_dir / "manifests" / "search-manifest.json",
        manifest_payload(manifest),
    )
    planned = sum(cell.wave == args.wave for cell in manifest.cells)
    print(f"wave={args.wave} planned_cells={planned}")
    if args.plan_only:
        return
    summary = run_wave(
        args.output_dir,
        config,
        catalog,
        source,
        manifest,
        wave=args.wave,
        device=args.device,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print(
        f"exp04b_{args.wave}_shard_complete={summary['completed']}/{summary['planned']} "
        f"failed={summary['failed']}"
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
