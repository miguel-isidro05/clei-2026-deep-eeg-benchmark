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


def test_cayetano_run_is_peterson_only() -> None:
    runner = (ROOT / "scripts" / "run_cayetano.sh").read_text(encoding="utf-8")

    assert "--datasets MI-OpenBCI" in runner
    assert "run_peterson_journal.py" in runner
    assert "--phase souza" not in runner
    assert "Souza2023" not in runner
    assert "run_transfer.py" not in runner


def test_default_setup_does_not_require_souza() -> None:
    setup = (ROOT / "setup.sh").read_text(encoding="utf-8")

    assert "ENABLE_SOUZA" in setup
    assert 'if [[ "${ENABLE_SOUZA:-0}" == "1" ]]' in setup


def test_preflight_dataset_selection_is_not_hardwired_to_peterson() -> None:
    preflight = (ROOT / "scripts" / "preflight.py").read_text(encoding="utf-8")

    assert 'if "MI-OpenBCI" in args.datasets:' in preflight
    assert 'if "Souza2023" in args.datasets:' in preflight


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
