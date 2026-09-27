#!/usr/bin/env python3
"""Profile all five models under identical device and input conditions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from deepbench.config import MODEL_NAMES, RESULTS_DIR, TARGET_SFREQ, TRIAL_SAMPLES
from deepbench.io import write_json_atomic
from deepbench.latency import latency_environment, profile_model
from deepbench.reproducibility import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=get_device())
    parser.add_argument("--n-chans", type=int, default=15)
    parser.add_argument("--n-times", type=int, default=TRIAL_SAMPLES)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "latency")
    args = parser.parse_args()
    environment = latency_environment(args.device)
    rows = [
        {
            **environment,
            **profile_model(
                model,
                n_chans=args.n_chans,
                n_times=args.n_times,
                sfreq=TARGET_SFREQ,
                device=args.device,
                batch_size=batch_size,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
        }
        for batch_size in args.batch_sizes
        for model in MODEL_NAMES
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "latency.json", rows)
    write_json_atomic(args.output_dir / "latency_environment.json", environment)
    with (args.output_dir / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Latency written to {args.output_dir}")


if __name__ == "__main__":
    main()
