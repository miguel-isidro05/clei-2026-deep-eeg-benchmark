"""Merge complete latency replications measured on multiple GPUs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .latency import latency_code_sha256

GRID_KEYS = ("dataset", "model", "batch_size", "n_chans", "n_times", "sfreq")
TIMING_COLUMNS = (
    "median_batch_ms",
    "p95_batch_ms",
    "amortized_median_ms_per_model_input",
    "individual_model_input_ms",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_latency_profiles(input_dirs: list[Path], output_dir: Path) -> None:
    """Preserve per-device values and create one median row per benchmark cell."""
    if len(input_dirs) < 2:
        raise ValueError("At least two complete device profiles are required")
    tables = []
    environments = []
    expected_grid: set[tuple[object, ...]] | None = None
    for input_dir in input_dirs:
        table = pd.read_csv(input_dir / "latency.csv")
        grid = set(table.loc[:, list(GRID_KEYS)].itertuples(index=False, name=None))
        if expected_grid is None:
            expected_grid = grid
        elif grid != expected_grid:
            raise ValueError(f"Latency profile grid mismatch in {input_dir}")
        if table["device"].nunique() != 1:
            raise ValueError(f"Expected one device per latency profile in {input_dir}")
        tables.append(table)
        environment_path = input_dir / "latency_environment.json"
        environments.append(json.loads(environment_path.read_text(encoding="utf-8")))
    raw = pd.concat(tables, ignore_index=True)
    if raw["device"].nunique() != len(input_dirs):
        raise ValueError("Latency profiles do not represent distinct devices")

    rows: list[dict[str, object]] = []
    for _, group in raw.groupby(list(GRID_KEYS), observed=True, sort=True):
        row = group.iloc[0].to_dict()
        devices = sorted(group["device"].astype(str).unique())
        hardware = sorted(group["hardware"].astype(str).unique())
        row["device"] = "+".join(devices)
        row["devices"] = ",".join(devices)
        row["hardware"] = hardware[0] if len(hardware) == 1 else "+".join(hardware)
        row["gpu_replicates"] = len(group)
        for column in TIMING_COLUMNS:
            values = group[column].dropna().to_numpy(float)
            row[column] = float(np.median(values)) if len(values) else np.nan
            row[f"{column}_between_gpu_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
        rows.append(row)
    merged = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "latency_per_device.csv"
    merged_path = output_dir / "latency.csv"
    raw.to_csv(raw_path, index=False)
    merged.to_csv(merged_path, index=False)
    json_rows = merged.astype(object).where(pd.notna(merged), None).to_dict("records")
    (output_dir / "latency.json").write_text(
        json.dumps(json_rows, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "latency_environment.json").write_text(
        json.dumps(environments, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "aggregation": "median_across_complete_device_profiles",
        "devices": sorted(raw["device"].astype(str).unique()),
        "latency_code_sha256": latency_code_sha256(),
        "latency_csv_sha256": _sha256(merged_path),
        "latency_per_device_csv_sha256": _sha256(raw_path),
        "input_profiles": [str(path) for path in input_dirs],
    }
    (output_dir / "latency_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
