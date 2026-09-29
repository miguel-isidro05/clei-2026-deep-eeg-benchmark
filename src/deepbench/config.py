"""Single source of truth for datasets, models, seeds, and output paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_NAMES: tuple[str, ...] = (
    "EEGNet",
    "FBCNet",
    "ShallowConvNet",
    "EEGConformer",
    "EEGInceptionMI",
)
PAPER_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
PAPER_EPOCHS = 300
PAPER_PROFILE_VERSION = "clei2026-deep-v5"
SPLIT_SEED = 2026
TARGET_SFREQ = 128.0
TRIAL_SECONDS = 4.0
TRIAL_SAMPLES = int(TARGET_SFREQ * TRIAL_SECONDS)
SOUZA_TRIAL_SECONDS = 3.0
SOUZA_TRIAL_SAMPLES = int(TARGET_SFREQ * SOUZA_TRIAL_SECONDS)
AUGMENT_WINDOW_SAMPLES = int(TARGET_SFREQ * 2.0)
AUGMENT_OVERLAP_STEP = 51

BENCH_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BENCH_ROOT.parent
RESULTS_DIR = Path(os.getenv("DEEP_EEG_RESULTS_DIR", BENCH_ROOT / "results"))

MI_CHANNELS: tuple[str, ...] = (
    "Pz",
    "Cz",
    "T6",
    "T4",
    "F8",
    "P4",
    "C4",
    "F4",
    "Fz",
    "T5",
    "T3",
    "F7",
    "P3",
    "C3",
    "F3",
)
MI_SUBJECTS: tuple[str, ...] = (
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S12",
)
SOUZA_PUBLISHED_SUBJECTS: tuple[str, ...] = ("001", "002", "003", "004", "005", "006")
SOUZA_SUBJECTS: tuple[str, ...] = ("002", "003", "004", "005", "006")
SOUZA_CHANNELS: tuple[str, ...] = (
    "Fp1",
    "Fz",
    "C3",
    "C4",
    "T5",
    "T6",
    "Cz",
    "Pz",
    "F7",
    "F8",
    "F3",
    "F4",
    "T3",
    "T4",
    "P3",
    "P4",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration needed to load one supported dataset."""

    name: str
    task: str
    events: tuple[str, str]
    paper_role: str
    multi_session: bool
    trial_seconds: float = TRIAL_SECONDS


DATASET_SPECS: dict[str, DatasetSpec] = {
    "MI-OpenBCI": DatasetSpec(
        name="MI-OpenBCI",
        task="motor_imagery_vs_rest",
        events=("rest", "motor_imagery"),
        paper_role="low_cost_primary",
        multi_session=False,
    ),
    "Souza2023": DatasetSpec(
        name="Souza2023",
        task="left_hand_vs_right_hand",
        events=("left_hand", "right_hand"),
        paper_role="low_cost_primary_left_right",
        multi_session=True,
        trial_seconds=SOUZA_TRIAL_SECONDS,
    ),
    "Zhou2020": DatasetSpec(
        name="Zhou2020",
        task="right_hand_vs_rest",
        events=("rest", "right_hand"),
        paper_role="research_grade_task_matched",
        multi_session=True,
    ),
    "Tavakolan2017": DatasetSpec(
        name="Tavakolan2017",
        task="right_hand_vs_rest",
        events=("rest", "right_hand"),
        paper_role="research_grade_task_matched_compact",
        multi_session=True,
        trial_seconds=3.0,
    ),
    "BNCI2014_001": DatasetSpec(
        name="BNCI2014_001",
        task="left_hand_vs_right_hand",
        events=("left_hand", "right_hand"),
        paper_role="canonical_cross_session",
        multi_session=True,
    ),
    "AlexMI": DatasetSpec(
        name="AlexMI",
        task="right_hand_vs_rest",
        events=("rest", "right_hand"),
        paper_role="smoke_task_matched",
        multi_session=False,
        trial_seconds=3.0,
    ),
}


def resolve_mi_data_dir() -> Path:
    """Locate the local MI-OpenBCI MAT files without copying the dataset."""
    if value := os.getenv("CLEI_DATA_DIR"):
        return Path(value).expanduser().resolve()
    candidates = (
        BENCH_ROOT / "data" / "mi-openbci",
        WORKSPACE_ROOT / "Code_before" / "DDPM_CLI2026" / "Database-MIOpenBCI-main",
        WORKSPACE_ROOT
        / "ARCHIVE"
        / "training_bundle"
        / "CLEI2026_TRAIN_RTX"
        / "data"
        / "Database-MIOpenBCI-main",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Set CLEI_DATA_DIR to the folder containing S02.mat ... S12.mat")


def resolve_souza_data_dir() -> Path:
    """Locate the local Souza2023 EDF files."""
    if value := os.getenv("SOUZA_DATA_DIR"):
        return Path(value).expanduser().resolve()
    return BENCH_ROOT / "data" / "souza2023"
