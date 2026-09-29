from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from deepbench.config import MODEL_NAMES
from deepbench.figures import generate_all_figures
from deepbench.runner import cell_path
from deepbench.statistics import statistics_code_sha256


def _load_generate_figures_script():
    path = Path(__file__).parents[1] / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("generate_figures_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_publication_figures_are_generated_from_validated_tables(tmp_path) -> None:
    statistics = tmp_path / "statistics"
    latency = tmp_path / "latency"
    cells = tmp_path / "cells"
    statistics.mkdir()
    latency.mkdir()
    cells.mkdir()
    metric_rows = []
    descriptive_rows = []
    for model_index, model in enumerate(MODEL_NAMES):
        for subject in range(5):
            for seed in range(5):
                for metric in ("accuracy", "kappa"):
                    base = 0.6 if metric == "accuracy" else 0.2
                    for policy in ("none", "kurtosis"):
                        metric_rows.append(
                            {
                                "dataset": "MI-OpenBCI",
                                "task": "binary",
                                "protocol": "within_split",
                                "condition": "full",
                                "ica_policy": policy,
                                "model": model,
                                "subject": str(subject),
                                "seed": seed,
                                "metric": metric,
                                "value": base + model_index * 0.01 + (policy == "kurtosis") * 0.001,
                            }
                        )
        for metric in ("accuracy", "kappa"):
            descriptive_rows.append(
                {
                    "dataset": "MI-OpenBCI",
                    "task": "binary",
                    "protocol": "within_split",
                    "condition": "full",
                    "ica_policy": "none",
                    "metric": metric,
                    "model": model,
                    "mean": 0.65,
                    "sd_between_subjects": 0.05,
                    "ci95_low": 0.60,
                    "ci95_high": 0.70,
                    "mean_within_subject_seed_sd": 0.01,
                }
            )
        destination = cell_path(
            tmp_path, "MI-OpenBCI", "within_session", "full", model, 0, "0", "none"
        )
        destination.write_text(
            json.dumps(
                {
                    "model": model,
                    "subject": "0",
                    "seed": 0,
                    "y_true": [0, 0, 1, 1],
                    "y_pred": [0, 1, 1, 1],
                }
            ),
            encoding="utf-8",
        )
    pd.DataFrame(metric_rows).to_csv(statistics / "all_seed_metrics.csv", index=False)
    pd.DataFrame(descriptive_rows).to_csv(
        statistics / "descriptive_subject_seed_variability.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "dataset": "MI-OpenBCI",
                "metric": "accuracy",
                "model": model,
                "comparison": comparison,
                "mean_difference": 0.01,
                "difference_ci95_low": -0.01,
                "difference_ci95_high": 0.03,
            }
            for model in MODEL_NAMES
            for comparison in ("nonoverlap-center_x2", "overlap-center_x6")
        ]
    ).to_csv(statistics / "augmentation_wilcoxon_holm.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model,
                "batch_size": 1,
                "median_batch_ms": 1.0,
                "p95_batch_ms": 1.5,
            }
            for model in MODEL_NAMES
        ]
    ).to_csv(latency / "latency.csv", index=False)
    generate_all_figures(tmp_path)
    expected_stems = {
        "primary_accuracy",
        "performance_overview_accuracy",
        "performance_overview_kappa",
        "augmentation_effects",
        "optimization_seed_variability",
        "ica_sensitivity",
        "model_inference_latency",
        "primary_confusion_matrices",
    }
    assert {path.stem for path in (tmp_path / "figures").glob("*.png")} == expected_stems
    assert {path.stem for path in (tmp_path / "figures").glob("*.pdf")} == expected_stems


def test_figure_cli_writes_manifest_for_verified_tables(tmp_path, monkeypatch) -> None:
    module = _load_generate_figures_script()
    statistics = tmp_path / "statistics"
    statistics.mkdir()
    names = (
        "all_seed_metrics.csv",
        "descriptive_subject_seed_variability.csv",
        "augmentation_wilcoxon_holm.csv",
    )
    for name in names:
        (statistics / name).write_text("header\n", encoding="utf-8")
    (statistics / "statistics_manifest.json").write_text(
        json.dumps(
            {
                "result_cells_sha256": module.result_cells_sha256(tmp_path),
                "statistics_code_sha256": statistics_code_sha256(),
                "statistics_files": {
                    name: hashlib.sha256((statistics / name).read_bytes()).hexdigest()
                    for name in names
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_generate(root: Path) -> None:
        figures = root / "figures"
        figures.mkdir()
        (figures / "example.pdf").write_bytes(b"pdf")

    monkeypatch.setattr(module, "audit_expected_cells", lambda _root: (None, True, []))
    monkeypatch.setattr(module, "generate_all_figures", fake_generate)
    monkeypatch.setattr(sys, "argv", ["generate_figures.py", "--results-dir", str(tmp_path)])
    module.main()
    manifest = json.loads(
        (tmp_path / "figures" / "figures_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["figures"]["example.pdf"] == hashlib.sha256(b"pdf").hexdigest()


def test_figure_cli_rejects_tampered_statistics(tmp_path, monkeypatch) -> None:
    module = _load_generate_figures_script()
    statistics = tmp_path / "statistics"
    statistics.mkdir()
    names = (
        "all_seed_metrics.csv",
        "descriptive_subject_seed_variability.csv",
        "augmentation_wilcoxon_holm.csv",
    )
    for name in names:
        (statistics / name).write_text("header\n", encoding="utf-8")
    hashes = {name: hashlib.sha256((statistics / name).read_bytes()).hexdigest() for name in names}
    (statistics / "statistics_manifest.json").write_text(
        json.dumps(
            {
                "result_cells_sha256": module.result_cells_sha256(tmp_path),
                "statistics_code_sha256": statistics_code_sha256(),
                "statistics_files": hashes,
            }
        ),
        encoding="utf-8",
    )
    (statistics / names[0]).write_text("tampered\n", encoding="utf-8")
    monkeypatch.setattr(module, "audit_expected_cells", lambda _root: (None, True, []))
    monkeypatch.setattr(sys, "argv", ["generate_figures.py", "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="invalid statistics table"):
        module.main()
