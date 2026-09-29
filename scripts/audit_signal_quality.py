#!/usr/bin/env python3
"""Write CPU-only signal-quality reports without modifying the data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deepbench.config import DATASET_SPECS, RESULTS_DIR
from deepbench.datasets import available_subjects, load_subject
from deepbench.io import write_json_atomic
from deepbench.signal_quality import signal_quality_metadata, signal_quality_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASET_SPECS), default=["MI-OpenBCI"]
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "quality")
    args = parser.parse_args()
    tables: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for dataset in args.datasets:
        for subject in available_subjects(dataset):
            recording = load_subject(dataset, subject)
            table = signal_quality_table(recording)
            if not bool((table["finite_fraction"] == 1.0).all()):
                raise SystemExit(f"Non-finite EEG samples detected in {dataset}/{subject}")
            tables.append(table)
            metadata.append(signal_quality_metadata(recording))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(
        args.output_dir / "signal_quality_channels.csv", index=False
    )
    write_json_atomic(args.output_dir / "signal_quality_metadata.json", metadata)
    write_json_atomic(
        args.output_dir / "signal_quality_status.json",
        {
            "status": "success",
            "datasets": args.datasets,
            "subjects": len(metadata),
            "nonfinite_detected": False,
            "automatic_exclusion": False,
        },
    )
    print(f"signal_quality=OK output={args.output_dir}")


if __name__ == "__main__":
    main()
