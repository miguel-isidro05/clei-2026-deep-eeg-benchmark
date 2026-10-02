from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_setup_does_not_download_unrequested_external_datasets() -> None:
    setup = (ROOT / "setup.sh").read_text(encoding="utf-8")

    assert "https://github.com/NiCALab-IMAL/Database-MIOpenBCI.git" in setup
    assert "https://github.com/ProMABLab/Database-MIOpenBCI.git" not in setup
    assert "download_moabb.py" not in setup
    assert "Zhou2020" not in setup
    assert "Tavakolan2017" not in setup


def test_cayetano_run_is_limited_to_peterson_and_souza() -> None:
    runner = (ROOT / "scripts" / "run_cayetano.sh").read_text(encoding="utf-8")

    assert "--datasets MI-OpenBCI Souza2023" in runner
    assert "--phase peterson" in runner
    assert "--phase souza" in runner
    assert "--phase external" not in runner
    assert "Zhou2020" not in runner
    assert "Tavakolan2017" not in runner


def test_validation_run_uses_one_gpu_and_runs_both_datasets_without_loso() -> None:
    runner = (ROOT / "run_validation_no_loso.sh").read_text(encoding="utf-8")

    assert "--min-cuda-devices 1" in runner
    assert "check_low_cost_csp.py --dataset MI-OpenBCI" in runner
    assert "check_low_cost_csp.py --dataset Souza2023" in runner
    assert "--phase no-loso" in runner
    assert "VALIDATION_SCOPE" not in runner
    assert "VALIDATION_SEED" not in runner
    assert "results_validation_no_loso_v7" in runner
    assert "--allow-incomplete" not in runner
    assert "scripts/generate_figures.py" in runner
