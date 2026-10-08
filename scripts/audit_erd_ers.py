#!/usr/bin/env python3
"""Write ERD/ERS plausibility diagnostics without changing experiment inclusion."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.config import MI_SUBJECTS, SOUZA_SUBJECTS
from deepbench.datasets import load_subject
from deepbench.erd_ers import peterson_mi_rest_change, souza_lateralization
from deepbench.io import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("MI-OpenBCI", "Souza2023"),
        default=["MI-OpenBCI", "Souza2023"],
    )
    args = parser.parse_args()
    payload = {"policy": "audit_only_no_tuning_or_exclusion", "peterson": {}, "souza": {}}
    if "MI-OpenBCI" in args.datasets:
        for subject in MI_SUBJECTS:
            recording = load_subject("MI-OpenBCI", subject)
            payload["peterson"][subject] = peterson_mi_rest_change(
                recording.x,
                recording.y,
                recording.ch_names,
                recording.sfreq,
                recording.sessions,
            )
    if "Souza2023" in args.datasets:
        for subject in SOUZA_SUBJECTS:
            recording = load_subject("Souza2023", subject)
            payload["souza"][subject] = souza_lateralization(
                recording.x,
                recording.y,
                recording.ch_names,
                recording.sfreq,
                recording.sessions,
            )
    write_json_atomic(args.output_dir / "quality" / "erd_ers.json", payload)
    print("erd_ers=OK", flush=True)


if __name__ == "__main__":
    main()
