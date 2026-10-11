#!/usr/bin/env python3
"""Analyze exp03 recipe selection or quality-condition search."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.exp03_quality_fbcnet.analysis import (  # noqa: E402
    analyze_quality_payloads,
    analyze_recipe_payloads,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.catalog import (  # noqa: E402
    build_quality_catalog,
    build_recipe_catalog,
    build_smoke_quality_catalog,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.config import (  # noqa: E402
    Exp03Config,
    smoke_config,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(
        {key for row in rows for key, value in row.items() if not isinstance(value, dict)}
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("recipe", "quality"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not (args.output_dir / f"{args.phase}_audit.json").is_file():
        raise SystemExit(f"A complete {args.phase} audit is required before analysis")
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.output_dir / f"{args.phase}_cells").glob("*.json"))
    ]
    config = smoke_config() if args.smoke else Exp03Config()
    if args.phase == "recipe":
        report = analyze_recipe_payloads(payloads, build_recipe_catalog(config))
        decision = report["recipe_decision"]
        write_json_atomic(args.output_dir / "analysis" / "recipe-readout.json", report)
        write_json_atomic(args.output_dir / "decisions" / "recipe_decision.json", decision)
        _write_csv(args.output_dir / "tables" / "recipe-summary.csv", report["recipe_summaries"])
        print(
            f"peterson_exp03_recipe_analysis=OK status={report['status']} "
            f"selected={decision['selected_recipe']['recipe_id']}"
        )
        if not args.smoke and report["status"] == "blocked_parity":
            raise SystemExit("Manual FBCNet and V10 baseline failed the 2-point parity gate")
        return
    candidates = (
        build_smoke_quality_catalog(config) if args.smoke else build_quality_catalog(config)
    )
    report = analyze_quality_payloads(payloads, candidates)
    write_json_atomic(args.output_dir / "analysis" / "quality-readout.json", report)
    write_json_atomic(
        args.output_dir / "decisions" / "promotion_decision.json",
        report["promotion_decision"],
    )
    _write_csv(args.output_dir / "tables" / "quality-ranking.csv", report["ranking"])
    recipe_decision = json.loads(
        (args.output_dir / "decisions" / "recipe_decision.json").read_text(encoding="utf-8")
    )
    quality_audit = json.loads(
        (args.output_dir / "quality_audit.json").read_text(encoding="utf-8")
    )
    write_json_atomic(
        args.output_dir / "run_complete.json",
        {
            "status": "smoke_completed" if args.smoke else "completed",
            "experiment": "exp03",
            "recipe_decision_sha256": recipe_decision["sha256"],
            "promotion_decision_sha256": report["promotion_decision"]["sha256"],
            "quality_manifest_sha256": quality_audit["manifest_sha256"],
        },
    )
    print(
        f"peterson_exp03_quality_analysis=OK status={report['status']} "
        f"promoted={len(report['promotion_decision']['promoted_config_ids'])}"
    )


if __name__ == "__main__":
    main()

