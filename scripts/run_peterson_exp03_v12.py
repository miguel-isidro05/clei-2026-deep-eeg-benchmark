#!/usr/bin/env python3
"""Plan or execute Peterson V12 exp03."""

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
from experiments.peterson_search_v12.exp03_quality_fbcnet.catalog import (  # noqa: E402
    build_quality_catalog,
    build_recipe_catalog,
    catalog_payload,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.config import (  # noqa: E402
    Exp03Config,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.manifest import (  # noqa: E402
    build_quality_manifest,
    build_recipe_manifest,
    manifest_payload,
)


def _decision_sha256(output_dir: Path) -> str:
    path = output_dir / "decisions" / "recipe_decision.json"
    if not path.is_file():
        raise SystemExit("Quality phase requires decisions/recipe_decision.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = str(payload.get("sha256", ""))
    if len(value) != 64:
        raise SystemExit("Recipe decision is missing a valid SHA-256")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("recipe", "quality"), required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = Exp03Config()
    revision = str(git_identity(ROOT)["revision"] or "unknown")
    if args.phase == "recipe":
        catalog = build_recipe_catalog(config)
        manifest = build_recipe_manifest(config, catalog, revision=revision)
        write_json_atomic(
            args.output_dir / "manifests" / "recipe-catalog.json", catalog_payload(catalog)
        )
        write_json_atomic(
            args.output_dir / "manifests" / "recipe-manifest.json", manifest_payload(manifest)
        )
        print(f"planned_recipes={len(catalog)}")
    else:
        catalog = build_quality_catalog(config)
        manifest = build_quality_manifest(
            config,
            catalog,
            revision=revision,
            recipe_decision_sha256=_decision_sha256(args.output_dir),
        )
        write_json_atomic(
            args.output_dir / "manifests" / "quality-catalog.json", catalog_payload(catalog)
        )
        write_json_atomic(
            args.output_dir / "manifests" / "quality-manifest.json", manifest_payload(manifest)
        )
        print(f"planned_quality_configs={len(catalog)}")
    print(f"planned_cells={len(manifest.cells)}")
    if args.plan_only:
        return
    raise SystemExit("Execution runner is not implemented yet")


if __name__ == "__main__":
    main()

