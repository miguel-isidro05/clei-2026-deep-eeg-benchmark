"""Deterministic train and stress corruptions for low-cost EEG."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch


@dataclass(frozen=True)
class CorruptionSpec:
    condition_id: str
    family: str
    channels: tuple[int, ...]
    severity: float | int


def build_stress_suite(n_chans: int) -> tuple[CorruptionSpec, ...]:
    if n_chans < 3:
        raise ValueError("Stress suite requires at least three channels")
    items: list[CorruptionSpec] = []
    for missing in (1, 2):
        for channels in combinations(range(n_chans), missing):
            label = "-".join(str(value) for value in channels)
            items.append(CorruptionSpec(f"missing-{missing}-{label}", "missing", channels, missing))
    for channel in range(n_chans):
        items.append(CorruptionSpec(f"flatline-{channel}", "flatline", (channel,), 1))
        for scale in (0.1, 0.5):
            items.append(
                CorruptionSpec(f"attenuation-{scale:g}-{channel}", "attenuation", (channel,), scale)
            )
        for snr in (20, 10, 0):
            items.append(CorruptionSpec(f"gaussian-{snr}-{channel}", "gaussian", (channel,), snr))
        for limit in (1.0, 2.0):
            items.append(
                CorruptionSpec(f"clipping-{limit:g}-{channel}", "clipping", (channel,), limit)
            )
        items.append(CorruptionSpec(f"impulse-{channel}", "impulse", (channel,), 1))
    for duration in (100, 250, 500):
        items.append(
            CorruptionSpec(f"temporal-dropout-{duration}", "temporal_dropout", (), duration)
        )
    return tuple(items)


def apply_corruption(
    x: torch.Tensor, spec: CorruptionSpec, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError("Corruption expects (batch, channels, time)")
    values = x.clone()
    mask = torch.ones((len(x), x.shape[1]), dtype=x.dtype, device=x.device)
    channels = list(spec.channels)
    if spec.family in {"missing", "flatline"}:
        values[:, channels] = 0
        mask[:, channels] = 0
    elif spec.family == "attenuation":
        values[:, channels] *= float(spec.severity)
    elif spec.family == "gaussian":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = values[:, channels]
        signal_rms = selected.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        noise_rms = signal_rms / (10.0 ** (float(spec.severity) / 20.0))
        noise = torch.randn(selected.shape, generator=generator, dtype=selected.dtype)
        values[:, channels] = selected + noise.to(selected.device) * noise_rms
    elif spec.family == "clipping":
        limit = float(spec.severity)
        scale = values[:, channels].std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        values[:, channels] = torch.clamp(values[:, channels], -limit * scale, limit * scale)
    elif spec.family == "impulse":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        time_index = int(torch.randint(0, x.shape[-1], (1,), generator=generator).item())
        scale = values[:, channels].std(dim=-1, unbiased=False).clamp_min(1e-6)
        values[:, channels, time_index] += 10.0 * scale
    elif spec.family == "temporal_dropout":
        samples = max(1, min(x.shape[-1], round(float(spec.severity) * 128.0 / 1000.0)))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        start = int(torch.randint(0, x.shape[-1] - samples + 1, (1,), generator=generator).item())
        values[:, :, start : start + samples] = 0
    else:
        raise ValueError(f"Unsupported corruption family: {spec.family}")
    return values, mask


def training_corruption(
    x: torch.Tensor,
    *,
    family: str,
    max_missing: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    missing = int(torch.randint(1, max_missing + 1, (1,), generator=generator).item())
    indices = tuple(
        int(value) for value in torch.randperm(x.shape[1], generator=generator)[:missing]
    )
    available = ["missing"]
    if family in {"detected_mixed", "broad_mixed"}:
        available += ["flatline", "attenuation"]
    if family == "broad_mixed":
        available += ["gaussian", "clipping", "impulse", "temporal_dropout"]
    selected = available[int(torch.randint(0, len(available), (1,), generator=generator).item())]
    severity: float | int = missing
    if selected == "attenuation":
        severity = (0.1, 0.5)[int(torch.randint(0, 2, (1,), generator=generator).item())]
    elif selected == "gaussian":
        severity = (20, 10, 0)[int(torch.randint(0, 3, (1,), generator=generator).item())]
    elif selected == "clipping":
        severity = (1.0, 2.0)[int(torch.randint(0, 2, (1,), generator=generator).item())]
    elif selected == "temporal_dropout":
        severity = (100, 250, 500)[int(torch.randint(0, 3, (1,), generator=generator).item())]
        indices = ()
    spec = CorruptionSpec(f"train-{selected}", selected, indices, severity)
    corrupted, mask = apply_corruption(x, spec, seed=seed)
    if selected in {"missing", "flatline"}:
        normalized_severity = missing / max_missing
    elif selected == "attenuation":
        normalized_severity = 1.0 - float(severity)
    elif selected == "gaussian":
        normalized_severity = (20.0 - float(severity)) / 20.0
    elif selected == "clipping":
        normalized_severity = 1.0 / float(severity)
    elif selected == "temporal_dropout":
        normalized_severity = float(severity) / 500.0
    else:
        normalized_severity = 1.0
    corruption_level = torch.full(
        (len(x),),
        min(1.0, normalized_severity),
        dtype=x.dtype,
        device=x.device,
    )
    return corrupted, mask, corruption_level
