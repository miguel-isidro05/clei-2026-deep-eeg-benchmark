from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from deepbench.latency_merge import merge_latency_profiles


def _profile(root: Path, device: str, median_ms: float) -> None:
    root.mkdir()
    table = pd.DataFrame(
        [
            {
                "dataset": "MI-OpenBCI",
                "model": "EEGNet",
                "device": device,
                "hardware": "NVIDIA RTX A6000",
                "batch_size": 1,
                "n_chans": 15,
                "n_times": 256,
                "sfreq": 128.0,
                "median_batch_ms": median_ms,
                "p95_batch_ms": median_ms + 1,
                "amortized_median_ms_per_model_input": median_ms,
                "individual_model_input_ms": median_ms,
                "parameters": 100,
                "measurement_unit": "model_input_epoch_or_window",
                "scope": "model_forward_only",
            }
        ]
    )
    table.to_csv(root / "latency.csv", index=False)
    (root / "latency_environment.json").write_text(
        json.dumps({"hardware": "NVIDIA RTX A6000", "device": device})
    )


def test_merge_latency_keeps_raw_devices_and_aggregates_fairly(tmp_path: Path) -> None:
    gpu0, gpu1, output = tmp_path / "gpu0", tmp_path / "gpu1", tmp_path / "latency"
    _profile(gpu0, "cuda:0", 2.0)
    _profile(gpu1, "cuda:1", 4.0)

    merge_latency_profiles([gpu0, gpu1], output)

    raw = pd.read_csv(output / "latency_per_device.csv")
    merged = pd.read_csv(output / "latency.csv")
    assert set(raw["device"]) == {"cuda:0", "cuda:1"}
    assert merged.loc[0, "median_batch_ms"] == 3.0
    assert merged.loc[0, "gpu_replicates"] == 2
    assert merged.loc[0, "devices"] == "cuda:0,cuda:1"
    manifest = json.loads((output / "latency_manifest.json").read_text())
    assert manifest["aggregation"] == "median_across_complete_device_profiles"
