#!/usr/bin/env python3
"""Generate Exp04 discovery readout, tables, and promotion decision."""

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

from experiments.peterson_search_v12.common.identity import canonical_sha256  # noqa: E402
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.exp04_qcr_ssnet.analysis import (  # noqa: E402
    analyze_results,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.catalog import (  # noqa: E402
    QCRCandidate,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.config import (  # noqa: E402
    Exp04Config,
    smoke_config,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: list[dict[str, object]] = []
    for row in rows:
        flattened.append(
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        )
    if not flattened:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    audit = json.loads((args.output_dir / "audit.json").read_text(encoding="utf-8"))
    expected_status = "smoke_completed" if args.smoke else "completed"
    if audit.get("status") != expected_status or audit.get("mode") != "strict":
        raise SystemExit("A strict complete Exp04 audit is required before analysis")
    catalog_payload = json.loads(
        (args.output_dir / "manifests" / "model-catalog.json").read_text(encoding="utf-8")
    )
    if canonical_sha256(catalog_payload) != audit.get("catalog_sha256"):
        raise SystemExit("Exp04 catalog changed after the strict audit")
    catalog = tuple(QCRCandidate(**candidate) for candidate in catalog_payload)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.output_dir / "cells").glob("*.json"))
    ]
    config = smoke_config() if args.smoke else Exp04Config()
    report = analyze_results(payloads, catalog, config)
    write_json_atomic(args.output_dir / "analysis" / "readout.json", report)
    write_json_atomic(
        args.output_dir / "decisions" / "promotion_decision.json",
        report["promotion_decision"],
    )
    _write_csv(args.output_dir / "tables" / "model-ranking.csv", report["ranking"])
    _write_csv(
        args.output_dir / "tables" / "subject-cell-metrics.csv",
        [
            {
                "cell_id": payload["cell_id"],
                "model_id": payload["model_id"],
                "config_id": payload["config_id"],
                "subject": payload["subject"],
                "seed": payload["seed"],
                **payload["validation_metrics"],
                "missing_1_accuracy": payload["stress_results"]["missing_1"]["metrics"]["accuracy"],
                "missing_2_accuracy": payload["stress_results"]["missing_2"]["metrics"]["accuracy"],
            }
            for payload in payloads
        ],
    )
    write_json_atomic(
        args.output_dir / "run_complete.json",
        {
            "status": expected_status,
            "experiment": "exp04",
            "manifest_sha256": audit["manifest_sha256"],
            "promotion_decision_sha256": report["promotion_decision"]["sha256"],
        },
    )
    print(
        f"peterson_exp04_analysis=OK status={report['status']} "
        f"selected={report['promotion_decision']['selected_config_id']}"
    )


if __name__ == "__main__":
    main()
