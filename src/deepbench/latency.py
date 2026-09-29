"""Controlled model-only inference latency measurements."""

from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .models import make_module, parameter_count


def latency_code_sha256() -> str:
    """Fingerprint the model-forward latency implementation."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def latency_environment(device: str) -> dict[str, object]:
    target = torch.device(device)
    if target.type == "cuda":
        hardware = torch.cuda.get_device_name(target)
    elif target.type == "mps":
        hardware = f"Apple {platform.machine()}"
    else:
        hardware = platform.processor() or platform.machine()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "hardware": hardware,
        "dtype": "float32",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "timing_clock": "time.perf_counter_ns",
    }


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    elif device.startswith("mps") and hasattr(torch, "mps"):
        torch.mps.synchronize()


def profile_model(
    name: str,
    *,
    n_chans: int,
    n_times: int,
    sfreq: float,
    device: str,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    module = (
        make_module(name, n_chans=n_chans, n_outputs=2, n_times=n_times, sfreq=sfreq)
        .to(device)
        .eval()
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    sample = torch.randn(batch_size, n_chans, n_times, generator=generator).to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            module(sample)
        _synchronize(device)
        elapsed_ms = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            module(sample)
            _synchronize(device)
            elapsed_ms.append((time.perf_counter_ns() - start) / 1e6)
    values = np.asarray(elapsed_ms)
    return {
        "model": name,
        "device": device,
        "batch_size": batch_size,
        "n_chans": n_chans,
        "n_times": n_times,
        "sfreq": sfreq,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "median_batch_ms": float(np.median(values)),
        "p95_batch_ms": float(np.percentile(values, 95)),
        "amortized_median_ms_per_trial": float(np.median(values) / batch_size),
        "individual_latency_ms": float(np.median(values)) if batch_size == 1 else None,
        "parameters": parameter_count(name, n_chans, n_times, sfreq),
        "scope": "model_forward_only_excludes_acquisition_buffering_and_preprocessing",
    }
