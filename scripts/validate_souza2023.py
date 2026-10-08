#!/usr/bin/env python3
"""Validate Souza2023 EDF identity, uniqueness, annotations, and trial counts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from deepbench.config import SOUZA_SUBJECTS, resolve_souza_data_dir
from deepbench.datasets import duplicate_file_groups, load_souza_subject
from deepbench.io import write_json_atomic
from deepbench.signal_quality import signal_quality_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.data_dir is not None:
        os.environ["SOUZA_DATA_DIR"] = str(args.data_dir.resolve())
    data_dir = resolve_souza_data_dir()
    present = [subject for subject in SOUZA_SUBJECTS if (data_dir / f"{subject}.edf").exists()]
    missing = [subject for subject in SOUZA_SUBJECTS if subject not in present]
    duplicates = duplicate_file_groups(data_dir, SOUZA_SUBJECTS)
    if duplicates:
        formatted = ", ".join("/".join(group) for group in duplicates)
        raise SystemExit(f"Souza2023 contains byte-identical subject files: {formatted}")
    if missing and not args.allow_incomplete:
        raise SystemExit(
            f"Souza2023 is incomplete in {data_dir}; missing valid subjects {missing}. "
            "The executable cohort is fixed to subjects 002-006."
        )
    if not present:
        raise SystemExit(f"No Souza2023 EDF files found in {data_dir}")
    metadata = [signal_quality_metadata(load_souza_subject(subject)) for subject in present]
    payload = {
        "status": "complete" if not missing else "incomplete",
        "data_dir": str(data_dir),
        "excluded_subjects": {
            "001": "public attachment 42 is byte-identical to subject 004"
        },
        "present_subjects": present,
        "missing_subjects": missing,
        "duplicate_groups": duplicates,
        "recordings": metadata,
    }
    output = args.output or data_dir / "validation.json"
    write_json_atomic(output, payload)
    print(f"souza_validation={payload['status']} subjects={len(present)} output={output}")


if __name__ == "__main__":
    main()
