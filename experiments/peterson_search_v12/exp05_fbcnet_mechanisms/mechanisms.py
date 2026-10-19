"""Mechanism primitives and train-only fitting hooks for Exp05."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from torch import nn
from torch.nn import functional as F

from .catalog import BASELINE_BANDS, MechanismCandidate

EPSILON = 1e-6


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-4), 1.0 - 1e-4)
    return math.log(probability / (1.0 - probability))


class ConstrainedSincFilterBank(nn.Module):
    """Small learnable band-pass bank with ordered, bounded cut-offs."""

    def __init__(
        self,
        bands: tuple[tuple[float, float], ...],
        *,
        sfreq: float,
        kernel_size: int = 63,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Sinc kernel size must be odd")
        if sfreq <= 2.0 * max(high for _, high in bands):
            raise ValueError("Sampling rate must exceed twice the largest band edge")
        self.sfreq = float(sfreq)
        self.kernel_size = int(kernel_size)
        self.n_bands = len(bands)
        self.min_frequency = 4.0
        self.max_frequency = min(self.sfreq / 2.0 - 1.0, 40.0)
        self.min_bandwidth = 1.0
        lows = np.asarray([low for low, _ in bands], dtype=np.float64)
        highs = np.asarray([high for _, high in bands], dtype=np.float64)
        if np.any(np.diff(lows) <= 0.0) or np.any(np.diff(highs) <= 0.0):
            raise ValueError("Sinc band cutoffs must be strictly ordered")
        if np.any(highs - lows <= self.min_bandwidth):
            raise ValueError("Sinc bands must exceed the minimum bandwidth")

        def ordered_bounds(values: np.ndarray, slack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            neighbor_gap = np.full(len(values), np.inf, dtype=np.float64)
            if len(values) > 1:
                gaps = np.diff(values)
                neighbor_gap[:-1] = np.minimum(neighbor_gap[:-1], gaps)
                neighbor_gap[1:] = np.minimum(neighbor_gap[1:], gaps)
            radius = np.minimum(0.25 * neighbor_gap, 0.25 * slack)
            return values - radius, values + radius

        bandwidth_slack = highs - lows - self.min_bandwidth
        low_lower, low_upper = ordered_bounds(lows, bandwidth_slack)
        high_lower, high_upper = ordered_bounds(highs, bandwidth_slack)
        low_lower = np.maximum(low_lower, self.min_frequency)
        high_upper = np.minimum(high_upper, self.max_frequency)
        if np.any(high_lower - low_upper < self.min_bandwidth):
            raise ValueError("Sinc cutoff bounds cannot preserve the minimum bandwidth")
        self.register_buffer("low_lower", torch.tensor(low_lower, dtype=torch.float32))
        self.register_buffer("low_upper", torch.tensor(low_upper, dtype=torch.float32))
        self.register_buffer("high_lower", torch.tensor(high_lower, dtype=torch.float32))
        self.register_buffer("high_upper", torch.tensor(high_upper, dtype=torch.float32))
        self.raw_low = nn.Parameter(torch.zeros(self.n_bands, dtype=torch.float32))
        self.raw_high = nn.Parameter(torch.zeros(self.n_bands, dtype=torch.float32))
        time = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer("time", time / self.sfreq)
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))

    def cutoffs(self) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.low_lower + (self.low_upper - self.low_lower) * torch.sigmoid(self.raw_low)
        high = self.high_lower + (self.high_upper - self.high_lower) * torch.sigmoid(
            self.raw_high
        )
        return low, high

    def kernels(self) -> torch.Tensor:
        low, high = self.cutoffs()
        time = self.time.unsqueeze(0)
        low = low.unsqueeze(1)
        high = high.unsqueeze(1)
        kernel = 2.0 * high * torch.sinc(2.0 * high * time)
        kernel = kernel - 2.0 * low * torch.sinc(2.0 * low * time)
        kernel = kernel * self.window.unsqueeze(0)
        return kernel / kernel.abs().sum(dim=1, keepdim=True).clamp_min(EPSILON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, n_times = x.shape
        filtered = F.conv1d(
            x.reshape(batch * channels, 1, n_times),
            self.kernels().unsqueeze(1),
            padding=self.kernel_size // 2,
        )
        return filtered.view(batch, channels, self.n_bands, n_times).permute(0, 2, 1, 3)


class GlobalBandWeights(nn.Module):
    """Positive band weights normalized to mean one."""

    def __init__(self, n_bands: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_bands))

    def forward(self, filtered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.logits, dim=0) * len(self.logits)
        return filtered * weights[None, :, None, None], weights.expand(len(filtered), -1)


class TrialBandAttention(nn.Module):
    """Content-dependent band reliability initialized to the identity."""

    def __init__(self, n_bands: int, hidden: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_bands, hidden), nn.GELU(), nn.Linear(hidden, n_bands)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, filtered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor = torch.log(filtered.square().mean(dim=(2, 3)).clamp_min(EPSILON))
        weights = torch.softmax(self.network(descriptor), dim=1) * descriptor.shape[1]
        return filtered * weights[:, :, None, None], weights


def segmented_log_variance(x: torch.Tensor, stride: int) -> torch.Tensor:
    """Return flattened log-variance features over ``stride`` temporal segments."""

    remainder = x.shape[-1] % stride
    if remainder:
        x = F.pad(x, (0, stride - remainder))
    reshaped = x.reshape(len(x), x.shape[1], stride, -1)
    return torch.log(reshaped.var(dim=-1, unbiased=False).clamp_min(EPSILON)).flatten(1)


def clipped_log_variance(x: torch.Tensor, *, stride: int, clip_value: float) -> torch.Tensor:
    """Robust log-variance after per-feature RMS-scaled winsorization."""

    scale = x.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(EPSILON)
    clipped = torch.clamp(x / scale, -clip_value, clip_value) * scale
    return segmented_log_variance(clipped, stride)


def _deterministic_lower_median(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Return PyTorch's lower median using only deterministic elementwise comparisons."""

    ordered = list(torch.unbind(values, dim=dim))
    if not ordered:
        raise ValueError("Median requires at least one value")
    for upper in range(len(ordered) - 1, 0, -1):
        for index in range(upper):
            left, right = ordered[index], ordered[index + 1]
            ordered[index] = torch.minimum(left, right)
            ordered[index + 1] = torch.maximum(left, right)
    return ordered[(len(ordered) - 1) // 2]


def median_of_means_log_power(x: torch.Tensor, *, blocks: int) -> torch.Tensor:
    """Robust power feature: median of block-wise mean square values."""

    remainder = x.shape[-1] % blocks
    if remainder:
        x = F.pad(x, (0, blocks - remainder))
    powers = x.reshape(len(x), x.shape[1], blocks, -1).square().mean(dim=-1)
    median = _deterministic_lower_median(powers, dim=2)
    return torch.log(median.clamp_min(EPSILON))


def _covariance(trials: np.ndarray) -> np.ndarray:
    centered = trials - trials.mean(axis=-1, keepdims=True)
    cov = np.einsum("nct,ndt->ncd", centered, centered) / max(1, trials.shape[-1] - 1)
    trace = np.trace(cov, axis1=1, axis2=2)[:, None, None]
    return cov / np.maximum(trace, EPSILON)


def _ordered_csp_filters(class_zero: np.ndarray, class_one: np.ndarray) -> np.ndarray:
    cov_zero = _covariance(class_zero).mean(axis=0)
    cov_one = _covariance(class_one).mean(axis=0)
    regularizer = np.eye(cov_zero.shape[0], dtype=np.float64) * 1e-6
    _, eigenvectors = eigh(cov_zero + regularizer, cov_zero + cov_one + 2.0 * regularizer)
    order: list[int] = []
    for offset in range(len(eigenvectors)):
        index = offset // 2 if offset % 2 == 0 else -(offset // 2 + 1)
        order.append(index)
    filters = eigenvectors[:, order].T
    return filters / np.maximum(np.linalg.norm(filters, axis=1, keepdims=True), EPSILON)


def fit_csp_filters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    sfreq: float,
    bands: tuple[tuple[float, float], ...] = BASELINE_BANDS,
    n_filters: int = 32,
) -> np.ndarray:
    """Fit deterministic per-band CSP filters using training data only."""

    labels = np.asarray(y_train)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("CSP fitting requires both classes")
    fitted: list[np.ndarray] = []
    for low, high in bands:
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        band_data = sosfiltfilt(sos, np.asarray(x_train, dtype=np.float64), axis=-1)
        basis = _ordered_csp_filters(band_data[labels == 0], band_data[labels == 1])
        fitted.append(np.stack([basis[index % len(basis)] for index in range(n_filters)]))
    return np.stack(fitted).astype(np.float32)


def fit_mu_peak(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    sfreq: float,
) -> float:
    """Estimate a discriminative 8--13 Hz peak from training data only."""

    values = np.asarray(x_train, dtype=np.float64)
    labels = np.asarray(y_train)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Mu-peak fitting requires both classes")
    frequencies = np.fft.rfftfreq(values.shape[-1], d=1.0 / sfreq)
    spectrum = np.abs(np.fft.rfft(values, axis=-1)) ** 2
    spectrum = spectrum.mean(axis=1)
    contrast = np.abs(spectrum[labels == 1].mean(axis=0) - spectrum[labels == 0].mean(axis=0))
    selection = (frequencies >= 8.0) & (frequencies <= 13.0)
    return float(frequencies[selection][np.argmax(contrast[selection])])


def quality_descriptors(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    differences = np.diff(values, axis=-1)
    log_rms = np.log(np.sqrt(np.mean(values**2, axis=(1, 2))) + EPSILON)
    log_diff = np.log(np.sqrt(np.mean(differences**2, axis=(1, 2))) + EPSILON)
    peak_ratio = np.max(np.abs(values), axis=(1, 2)) / (
        np.sqrt(np.mean(values**2, axis=(1, 2))) + EPSILON
    )
    return np.stack((log_rms, log_diff, peak_ratio), axis=1)


def fit_quality_weight_profile(x_train: np.ndarray) -> dict[str, object]:
    descriptors = quality_descriptors(x_train)
    median = np.median(descriptors, axis=0)
    mad = np.median(np.abs(descriptors - median), axis=0)
    return {
        "descriptor_median": median.astype(float).tolist(),
        "descriptor_mad": np.maximum(mad, EPSILON).astype(float).tolist(),
        "fit_partition": "train_only",
    }


def prepare_fit_context(
    candidate: MechanismCandidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sfreq: float,
) -> dict[str, object]:
    """Fit candidate-specific state without consulting validation or test data."""

    if candidate.fit_hook is None:
        return {"fit_partition": "none"}
    if candidate.fit_hook == "mu_peak":
        peak = fit_mu_peak(x_train, y_train, sfreq=sfreq)
        first_band = (max(8.0, peak - 2.0), min(14.0, peak + 2.0))
        return {
            "fit_partition": "train_only",
            "mu_peak_hz": peak,
            "bands": (first_band, *BASELINE_BANDS[1:]),
        }
    if candidate.fit_hook == "csp":
        filters = fit_csp_filters(x_train, y_train, sfreq=sfreq)
        return {"fit_partition": "train_only", "csp_filters": filters}
    if candidate.fit_hook == "quality_profile":
        return fit_quality_weight_profile(x_train)
    raise ValueError(f"Unsupported fit hook: {candidate.fit_hook}")


def fourier_surrogate(x: torch.Tensor, *, phase_scale: float, seed: int) -> torch.Tensor:
    """Apply one coherent random phase curve to all channels of each example."""

    spectrum = torch.fft.rfft(x, dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    phase = torch.rand((len(x), 1, spectrum.shape[-1]), generator=generator) * 2.0 - 1.0
    phase = phase.to(device=x.device, dtype=x.real.dtype) * math.pi * phase_scale
    phase[..., 0] = 0.0
    if x.shape[-1] % 2 == 0:
        phase[..., -1] = 0.0
    return torch.fft.irfft(spectrum * torch.exp(1j * phase), n=x.shape[-1], dim=-1)
