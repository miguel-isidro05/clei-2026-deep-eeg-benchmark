#!/usr/bin/env python3
"""Fail fast on environment, local data, and architecture incompatibilities."""

from __future__ import annotations

import argparse
import importlib.metadata

import numpy as np
import torch

from deepbench.config import (
    MI_SUBJECTS,
    MODEL_NAMES,
    TARGET_SFREQ,
    TRIAL_SAMPLES,
    resolve_mi_data_dir,
)
from deepbench.datasets import validate_dataset_dependencies
from deepbench.models import make_module, parameter_count
from deepbench.reproducibility import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but torch.cuda.is_available() is False")
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
