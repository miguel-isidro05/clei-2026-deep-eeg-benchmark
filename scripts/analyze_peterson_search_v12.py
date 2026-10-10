#!/usr/bin/env python3
"""Create the exp01 ranking and proposed promotion decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.peterson_search_v12.common.identity import canonical_sha256  # noqa: E402
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.common.promotion import rank_and_promote  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    completion = args.output_dir / "run_complete.json"
    if not completion.is_file():
        raise SystemExit("Full audit is required before exp01 analysis")
    paths = sorted((args.output_dir / "cells").glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    decision = rank_and_promote(payloads, expected_cells=216)
    decision.update(
        {
            "manifest_sha256": json.loads(completion.read_text(encoding="utf-8"))[
                "manifest_sha256"
            ],
            "results_sha256": canonical_sha256(payloads),
            "note": "Proposed only; exp02 remains blocked pending human approval.",
        }
    )
    write_json_atomic(args.output_dir / "decisions" / "exp01.json", decision)
    write_json_atomic(args.output_dir / "ranking-exp01.json", decision["ranking"])
    print("peterson_search_analysis=OK promoted=12 status=proposed")


if __name__ == "__main__":
    main()
