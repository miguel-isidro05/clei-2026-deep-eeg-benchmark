"""Deterministic setup and environment manifests."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def configure_determinism() -> None:
    """Request deterministic Torch kernels before model construction."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch before every independent training cell."""
    configure_determinism()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(path: Path, *, config: dict[str, object], root: Path) -> None:
    """Write versions and run configuration before launching expensive jobs."""
    packages = (
        "torch",
        "braindecode",
        "moabb",
        "mne",
        "numpy",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "pandas",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": _git_revision(root),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_policy": "torch_deterministic_warn_only_cudnn_deterministic",
        "requested_device": config.get("device"),
        "auto_detected_device": get_device(),
        "packages": versions,
        "environment": {
            "CLEI_DATA_DIR": os.getenv("CLEI_DATA_DIR"),
            "MNE_DATA": os.getenv("MNE_DATA"),
        },
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
