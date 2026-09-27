#!/usr/bin/env python3
"""Run a resumable block of dataset/model/seed experiment cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deepbench.config import DATASET_SPECS, MODEL_NAMES, PAPER_SEEDS, RESULTS_DIR
from deepbench.reproducibility import configure_determinism, get_device, write_manifest
from deepbench.runner import run_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_SPECS, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("within_split", "within_session", "cross_session", "loso"),
        required=True,
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("full", "center", "nonoverlap", "overlap"),
        default=["full"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default=get_device())
    parser.add_argument("--ica-policy", choices=("kurtosis", "none"), default="none")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_determinism()
    if "cross_session" in args.protocols and not DATASET_SPECS[args.dataset].multi_session:
        raise SystemExit(f"{args.dataset} does not support cross_session")
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    manifest_name = f"run-{args.dataset}-{config_hash}.json"
    write_manifest(
        args.output_dir / "manifests" / manifest_name,
        config=configuration,
        root=Path(__file__).resolve().parents[1],
    )
    counts = run_job(
        dataset=args.dataset,
        models=args.models,
        protocols=args.protocols,
        conditions=args.conditions,
        seeds=args.seeds,
        subjects=args.subjects,
        device=args.device,
        epochs=args.epochs,
        ica_policy=args.ica_policy,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        save_weights=args.save_weights,
    )
    print(counts)


if __name__ == "__main__":
    main()
