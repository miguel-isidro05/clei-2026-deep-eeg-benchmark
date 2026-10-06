#!/usr/bin/env python3
"""Profile all five models under identical device and input conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from deepbench.config import (
    MODEL_NAMES,
    RESULTS_DIR,
    SOUZA_TRIAL_SAMPLES,
    TARGET_SFREQ,
    TRIAL_SAMPLES,
)
from deepbench.io import write_json_atomic
from deepbench.latency import latency_code_sha256, latency_environment, profile_model
from deepbench.reproducibility import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=get_device())
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("MI-OpenBCI", "Souza2023"),
        default=["MI-OpenBCI", "Souza2023"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--windowed-transfer", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "latency")
    args = parser.parse_args()
    environment = latency_environment(args.device)
    input_shapes = {
        "MI-OpenBCI": (15, TRIAL_SAMPLES),
        "Souza2023": (16, SOUZA_TRIAL_SAMPLES),
    }
    if args.windowed_transfer:
        input_shapes = {dataset: (15, 256) for dataset in input_shapes}
    rows = []
    for dataset in args.datasets:
        n_chans, n_times = input_shapes[dataset]
        for batch_size in args.batch_sizes:
            for model in args.models:
                rows.append(
                    {
                        "dataset": dataset,
                        "n_chans": n_chans,
                        "n_times": n_times,
                        **environment,
                        **profile_model(
                            model,
                            n_chans=n_chans,
                            n_times=n_times,
                            sfreq=TARGET_SFREQ,
                            device=args.device,
                            batch_size=batch_size,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        ),
                    }
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "latency.json", rows)
    write_json_atomic(args.output_dir / "latency_environment.json", environment)
    with (args.output_dir / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(
        args.output_dir / "latency_manifest.json",
        {
            "latency_csv_sha256": hashlib.sha256(
                (args.output_dir / "latency.csv").read_bytes()
            ).hexdigest(),
            "environment": environment,
            "latency_code_sha256": latency_code_sha256(),
            "profile": {
                "device": args.device,
                "datasets": args.datasets,
                "models": args.models,
                "input_shapes": {dataset: input_shapes[dataset] for dataset in args.datasets},
                "batch_sizes": args.batch_sizes,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "windowed_transfer": args.windowed_transfer,
            },
        },
    )
    print(f"Latency written to {args.output_dir}")


if __name__ == "__main__":
    main()
