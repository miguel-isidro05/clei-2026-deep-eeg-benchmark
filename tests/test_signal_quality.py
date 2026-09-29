from __future__ import annotations

import numpy as np

from deepbench.signal_quality import signal_quality_metadata, signal_quality_table
from deepbench.types import SubjectRecording


def test_signal_quality_reports_flags_without_excluding_data() -> None:
    x = np.ones((4, 2, 32), dtype=np.float32)
    x[:, 1] = np.arange(32, dtype=np.float32)
    recording = SubjectRecording(
        dataset="synthetic",
        subject="1",
        x=x,
        y=np.array([0, 1, 0, 1]),
        sessions=np.array(["a", "a", "b", "b"]),
        sfreq=128.0,
        ch_names=("flat", "variable"),
        task="binary",
        data_sha256="abc",
    )
    table = signal_quality_table(recording)
    assert bool(table.loc[table["channel"] == "flat", "flat_channel_flag"].iloc[0])
    assert "variance_loaded_units" in table.columns
    assert not bool(table["automatic_exclusion"].any())
    metadata = signal_quality_metadata(recording)
    assert metadata["class_counts"] == {"0": 2, "1": 2}
    assert metadata["automatic_exclusion"] is False
