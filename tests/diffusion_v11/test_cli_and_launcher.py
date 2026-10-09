from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cli_plan_only_writes_exact_manifest(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_diffusion_v11.py",
            "--wave",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--num-shards",
            "2",
            "--shard-index",
            "1",
            "--plan-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "planned_total=240" in completed.stdout
    assert "planned_shard=120" in completed.stdout


def test_cli_can_limit_each_shard_for_gpu_smoke(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_diffusion_v11.py",
            "--wave",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--num-shards",
            "2",
            "--shard-index",
            "0",
            "--max-cells-per-shard",
            "1",
            "--plan-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "planned_total=240" in completed.stdout
    assert "planned_shard=1" in completed.stdout


def test_launcher_has_two_shards_and_propagates_failures() -> None:
    launcher = (ROOT / "run_diffusion_cayetano.sh").read_text()
    assert "--num-shards 2 --shard-index 0" in launcher
    assert "--num-shards 2 --shard-index 1" in launcher
    assert "wait \"$pid0\"" in launcher
    assert "wait \"$pid1\"" in launcher
    assert "DIFFUSION_SMOKE" in launcher
    assert "DIFFUSION_MAX_CELLS_PER_SHARD" not in launcher
    assert "DIFFUSION_EPOCHS" not in launcher
    assert "--manifest-name smoke_expected.json" in launcher
    assert "diffusion_v11_smoke=OK" in launcher
    subprocess.run(["bash", "-n", "run_diffusion_cayetano.sh"], cwd=ROOT, check=True)


def test_launcher_requires_exactly_two_devices(tmp_path) -> None:
    environment = os.environ | {
        "DIFFUSION_WAVE": "0",
        "DIFFUSION_RESULTS_DIR": str(tmp_path),
        "DIFFUSION_DEVICES": "cpu",
    }
    completed = subprocess.run(
        ["bash", "run_diffusion_cayetano.sh"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "exactly two devices" in completed.stderr


def test_launcher_rejects_smoke_for_cpu_only_wave_zero(tmp_path) -> None:
    environment = os.environ | {
        "DIFFUSION_WAVE": "0",
        "DIFFUSION_RESULTS_DIR": str(tmp_path),
        "DIFFUSION_DEVICES": "cuda:0 cuda:1",
        "DIFFUSION_SMOKE": "1",
    }
    completed = subprocess.run(
        ["bash", "run_diffusion_cayetano.sh"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "only for wave 1 GPU validation" in completed.stderr


def test_smoke_manifest_cannot_mark_full_run_complete(tmp_path) -> None:
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "smoke_expected.json").write_text("{}")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_diffusion_v11.py",
            "--output-dir",
            str(tmp_path),
            "--manifest-name",
            "smoke_expected.json",
            "--write-complete",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "cannot write run_complete.json" in completed.stderr
