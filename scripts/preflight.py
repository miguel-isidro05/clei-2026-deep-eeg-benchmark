#!/usr/bin/env python3
"""Fail fast on environment, local data, and architecture incompatibilities."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from deepbench.config import (
    MI_SUBJECTS,
    MODEL_NAMES,
    RESULTS_DIR,
    TARGET_SFREQ,
    TRIAL_SAMPLES,
    resolve_mi_data_dir,
)
from deepbench.datasets import load_subject, validate_dataset_dependencies
from deepbench.io import write_json_atomic
from deepbench.models import make_module, parameter_count
from deepbench.reproducibility import get_device
from deepbench.signal_quality import signal_quality_metadata, signal_quality_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--min-cuda-devices", type=int, default=1)
    parser.add_argument("--quality-output-dir", type=Path, default=RESULTS_DIR / "quality")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but torch.cuda.is_available() is False")
    if args.require_cuda and torch.cuda.device_count() < args.min_cuda_devices:
        raise SystemExit(
            f"CUDA device_count={torch.cuda.device_count()} but {args.min_cuda_devices} "
            "visible CUDA devices are required"
        )
    print(f"device={get_device()} torch={torch.__version__}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    validate_dataset_dependencies("Tavakolan2017")
    for package in (
        "braindecode",
        "moabb",
        "mne",
        "skorch",
        "scipy",
        "statsmodels",
        "BCI2kReader",
    ):
        print(f"{package}={importlib.metadata.version(package)}")
    data_dir = resolve_mi_data_dir()
    missing = [subject for subject in MI_SUBJECTS if not (data_dir / f"{subject}.mat").exists()]
    if missing:
        raise SystemExit(f"Missing MI-OpenBCI files in {data_dir}: {missing}")
    quality_tables = []
    quality_metadata = []
    for subject in MI_SUBJECTS:
        recording = load_subject("MI-OpenBCI", subject)
        table = signal_quality_table(recording)
        if not bool((table["finite_fraction"] == 1.0).all()):
            raise SystemExit(f"Non-finite EEG samples detected in MI-OpenBCI/{subject}")
        quality_tables.append(table)
        quality_metadata.append(signal_quality_metadata(recording))
    args.quality_output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.concat(quality_tables, ignore_index=True).to_csv(
        args.quality_output_dir / "signal_quality_channels.csv", index=False
    )
    write_json_atomic(args.quality_output_dir / "signal_quality_metadata.json", quality_metadata)
    write_json_atomic(
        args.quality_output_dir / "signal_quality_status.json",
        {
            "status": "success",
            "datasets": ["MI-OpenBCI"],
            "subjects": len(quality_metadata),
            "nonfinite_detected": False,
            "automatic_exclusion": False,
        },
    )
    sample = torch.from_numpy(np.zeros((2, 15, TRIAL_SAMPLES), dtype=np.float32))
    for model_name in MODEL_NAMES:
        module = make_module(
            model_name,
            n_chans=15,
            n_outputs=2,
            n_times=TRIAL_SAMPLES,
            sfreq=TARGET_SFREQ,
        ).eval()
        with torch.no_grad():
            output = module(sample)
        if tuple(output.shape) != (2, 2):
            raise SystemExit(f"Unexpected {model_name} output: {tuple(output.shape)}")
        parameters = parameter_count(model_name, 15, TRIAL_SAMPLES, TARGET_SFREQ)
        print(f"{model_name}: output={tuple(output.shape)} params={parameters}")
    print("preflight=OK")


if __name__ == "__main__":
    main()
