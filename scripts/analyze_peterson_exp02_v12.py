#!/usr/bin/env python3
"""Generate exp02 rankings, contrasts and machine-readable tables."""

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
from experiments.peterson_search_v12.exp02_factorial.analysis import analyze_payloads  # noqa: E402
from experiments.peterson_search_v12.exp02_factorial.catalog import (  # noqa: E402
    build_candidate_catalog,
    build_inference_catalog,
)
from experiments.peterson_search_v12.exp02_factorial.config import Exp02Config  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row if not isinstance(row[key], dict)})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.output_dir / "run_complete.json").is_file():
        raise SystemExit("A complete exp02 audit is required before analysis")
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.output_dir / "cells").glob("*.json"))
    ]
    candidate_cells = [cell for cell in payloads if cell["kind"] == "candidate"]
    baseline_cells = [cell for cell in payloads if cell["kind"] == "baseline"]
    config = Exp02Config()
    candidates = build_candidate_catalog(config)
    inference = build_inference_catalog(candidates)
    report = analyze_payloads(
        candidate_cells,
        baseline_cells,
        candidates=candidates,
        inference_catalog=inference,
        config=config,
    )
    write_json_atomic(args.output_dir / "analysis" / "exp02-readout.json", report)
    write_json_atomic(args.output_dir / "decisions" / "exp02.json", report["promotion_decision"])
    write_json_atomic(args.output_dir / "analysis" / "ranking-exp02.json", report["ranking"])
    _write_csv(args.output_dir / "tables" / "inference-ranking.csv", report["ranking"])
    subject_rows: list[dict[str, object]] = []
    for item in report["ranking"]:
        for subject, accuracy in item["subject_accuracy"].items():
            subject_rows.append(
                {
                    "config_id": item["config_id"],
                    "inference_id": item["inference_id"],
                    "topology_id": item["topology_id"],
                    "objective_id": item["objective_id"],
                    "subject": subject,
                    "accuracy": accuracy,
                }
            )
    _write_csv(args.output_dir / "tables" / "subject-results.csv", subject_rows)
    print(
        f"peterson_exp02_analysis=OK status={report['status']} "
        f"promoted={len(report['promotion_decision']['promoted'])}"
    )


if __name__ == "__main__":
    main()
