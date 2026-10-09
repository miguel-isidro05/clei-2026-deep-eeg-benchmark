"""Runtime identity included in every scientific result fingerprint."""

from __future__ import annotations

import importlib.metadata
import platform

import torch

from .identity import canonical_sha256

SCIENTIFIC_PACKAGES = (
    "torch",
    "braindecode",
    "mne",
    "numpy",
    "scipy",
    "scikit-learn",
    "skorch",
)


def runtime_identity(device: str) -> dict[str, object]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in SCIENTIFIC_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        hardware = torch.cuda.get_device_name(target)
    elif target.type == "mps":
        hardware = f"Apple {platform.machine()}"
    else:
        hardware = platform.processor() or platform.machine()
    scientific = {
        "versions": versions,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    return {
        **scientific,
        "environment_sha256": canonical_sha256(scientific),
        "device_type": target.type,
        "hardware": hardware,
    }
