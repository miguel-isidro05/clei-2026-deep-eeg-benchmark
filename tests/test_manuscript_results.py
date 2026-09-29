from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from deepbench.config import MODEL_NAMES
from deepbench.figures import figures_code_sha256
from deepbench.latency import latency_code_sha256
from deepbench.paper_audit import result_cells_sha256
from deepbench.statistics import statistics_code_sha256


def _load_script_module():
    path = Path(__file__).parents[1] / "scripts" / "generate_manuscript_results.py"
    spec = importlib.util.spec_from_file_location("generate_manuscript_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_artifacts(root: Path) -> None:
    statistics = root / "statistics"
    figures = root / "figures"
    latency = root / "latency"
    cells = root / "cells"
    for directory in (statistics, figures, latency, cells):
        directory.mkdir(parents=True)

    descriptive = []
    for dataset, protocol in (
        ("MI-OpenBCI", "within_session"),
        ("Souza2023", "within_session"),
        ("Zhou2020", "cross_session"),
        ("Tavakolan2017", "cross_session"),
    ):
        for model in MODEL_NAMES:
            for metric in ("accuracy", "kappa"):
                descriptive.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "condition": "full",
                        "ica_policy": "none",
                        "model": model,
                        "metric": metric,
                        "mean": 0.6,
                        "sd_between_subjects": 0.1,
                    }
                )
    paired = [
        {
            "dataset": dataset,
            "protocol": "within_session",
            "condition": "full",
            "ica_policy": "none",
            "metric": "accuracy",
            "model_a": model_a,
            "model_b": model_b,
            "n_subjects": 10 if dataset == "MI-OpenBCI" else 5,
            "mean_difference_a_minus_b": 0.01,
            "difference_ci95_low": -0.02,
            "difference_ci95_high": 0.04,
            "rank_biserial_a_minus_b": 0.2,
            "p_raw": 0.1,
            "p_holm": 0.5,
            "reject_holm_0_05": False,
        }
        for dataset in ("MI-OpenBCI", "Souza2023")
        for model_a, model_b in combinations(MODEL_NAMES, 2)
    ]
    augmentation = [
        {
            "dataset": "MI-OpenBCI",
            "metric": "accuracy",
            "model": MODEL_NAMES[0],
            "comparison": "overlap-center_x6",
            "n_subjects": 10,
            "mean_difference": 0.01,
            "difference_ci95_low": -0.02,
            "difference_ci95_high": 0.04,
            "rank_biserial_augmented_minus_center": 0.2,
            "p_raw": 0.1,
            "p_holm": 0.5,
            "reject_holm_0_05": False,
        }
    ]
    tables = {
        "descriptive_subject_seed_variability.csv": pd.DataFrame(descriptive),
        "paired_wilcoxon_holm.csv": pd.DataFrame(paired),
        "augmentation_wilcoxon_holm.csv": pd.DataFrame(augmentation),
    }
    for name, table in tables.items():
        table.to_csv(statistics / name, index=False)
    (statistics / "statistics_manifest.json").write_text(
        json.dumps(
            {
                "result_cells_sha256": result_cells_sha256(root),
                "statistics_code_sha256": statistics_code_sha256(),
                "statistics_files": {name: _sha256(statistics / name) for name in tables},
            }
        ),
        encoding="utf-8",
    )

    latency_path = latency / "latency.csv"
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": model,
                "batch_size": 1,
                "median_batch_ms": 1.0,
                "p95_batch_ms": 1.5,
            }
            for dataset in ("MI-OpenBCI", "Souza2023")
            for model in MODEL_NAMES
        ]
    ).to_csv(latency_path, index=False)
    (latency / "latency_manifest.json").write_text(
        json.dumps(
            {
                "latency_csv_sha256": _sha256(latency_path),
                "latency_code_sha256": latency_code_sha256(),
            }
        ),
        encoding="utf-8",
    )

    figure_names = (
        "performance_overview_accuracy.pdf",
        "performance_overview_kappa.pdf",
        "augmentation_effects.pdf",
        "optimization_seed_variability.pdf",
        "ica_sensitivity.pdf",
        "model_inference_latency.pdf",
        "primary_confusion_matrices.pdf",
        "souza_accuracy.pdf",
        "souza_confusion_matrices.pdf",
    )
    for name in figure_names:
        (figures / name).write_bytes(b"valid-pdf-placeholder")
    (figures / "figures_manifest.json").write_text(
        json.dumps(
            {
                "result_cells_sha256": result_cells_sha256(root),
                "figures_code_sha256": figures_code_sha256(),
                "figures": {name: _sha256(figures / name) for name in figure_names},
            }
        ),
        encoding="utf-8",
    )


def test_manuscript_generation_uses_verified_inferential_artifacts(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    _write_valid_artifacts(tmp_path)
    output = tmp_path / "generated_results.tex"
    monkeypatch.setattr(module, "audit_expected_cells", lambda _root: (None, True, []))
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_manuscript_results.py", "--results-dir", str(tmp_path), "--output", str(output)],
    )
    module.main()
    latex = output.read_text(encoding="utf-8")
    assert "Primary paired model comparisons" in latex
    assert "$r_{rb}$" in latex
    assert "$p_{Holm}$" in latex
    assert "Compute-matched augmentation" in latex
    assert "Souza2023 full-trial performance" in latex


def test_manuscript_generation_rejects_tampered_statistics(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    _write_valid_artifacts(tmp_path)
    target = tmp_path / "statistics" / "paired_wilcoxon_holm.csv"
    target.write_text(target.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    monkeypatch.setattr(module, "audit_expected_cells", lambda _root: (None, True, []))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_manuscript_results.py",
            "--results-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "generated_results.tex"),
        ],
    )
    with pytest.raises(SystemExit, match="invalid statistics table"):
        module.main()


def test_manuscript_generation_rejects_tampered_figure(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    _write_valid_artifacts(tmp_path)
    (tmp_path / "figures" / "ica_sensitivity.pdf").write_bytes(b"tampered")
    monkeypatch.setattr(module, "audit_expected_cells", lambda _root: (None, True, []))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_manuscript_results.py",
            "--results-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "generated_results.tex"),
        ],
    )
    with pytest.raises(SystemExit, match="invalid figure"):
        module.main()
