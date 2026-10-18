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

from experiments.peterson_search_v12.common.identity import (  # noqa: E402
    canonical_sha256,
    git_identity,
)
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.exp03_quality_fbcnet.catalog import (  # noqa: E402
    build_quality_catalog,
    build_recipe_catalog,
    build_smoke_quality_catalog,
    catalog_payload,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.config import (  # noqa: E402
    Exp03Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.manifest import (  # noqa: E402
    build_quality_manifest,
    build_recipe_manifest,
    manifest_payload,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.runner import (  # noqa: E402
    run_quality_phase,
    run_recipe_phase,
)


def _decision(output_dir: Path) -> dict[str, object]:
    path = output_dir / "decisions" / "recipe_decision.json"
    if not path.is_file():
        raise SystemExit("Quality phase requires decisions/recipe_decision.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = str(payload.get("sha256", ""))
    body = {key: item for key, item in payload.items() if key != "sha256"}
    if len(value) != 64 or canonical_sha256(body) != value:
        raise SystemExit("Recipe decision signature is invalid")
    if payload.get("status") != "selected":
        raise SystemExit("Recipe decision did not pass the parity gate")
    return payload


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
    config = smoke_config() if args.smoke else Exp03Config()
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
        decision = _decision(args.output_dir)
        catalog = (
            build_smoke_quality_catalog(config) if args.smoke else build_quality_catalog(config)
        )
        manifest = build_quality_manifest(
            config,
            catalog,
            revision=revision,
            recipe_decision_sha256=str(decision["sha256"]),
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
    if args.phase == "recipe":
        summary = run_recipe_phase(
            args.output_dir,
            config,
            catalog,
            manifest,
            device=args.device,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    else:
        selected = decision.get("selected_recipe")
        if not isinstance(selected, dict):
            raise SystemExit("Recipe decision is missing selected_recipe")
        recipes = build_recipe_catalog(config)
        selected_recipe = next(
            (recipe for recipe in recipes if recipe.recipe_id == selected.get("recipe_id")), None
        )
        if selected_recipe is None or selected_recipe.sha256 != selected.get("sha256"):
            raise SystemExit("Selected recipe does not match the signed exp03 catalog")
        summary = run_quality_phase(
            args.output_dir,
            config,
            catalog,
            manifest,
            selected_recipe,
            device=args.device,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    print(
        f"exp03_{args.phase}_shard_complete={summary['completed']}/{summary['planned']} "
        f"failed={summary['failed']}"
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
