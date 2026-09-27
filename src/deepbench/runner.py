"""Resumable orchestration for independent experiment cells."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import torch

from .config import RESULTS_DIR, SPLIT_SEED
from .datasets import available_subjects, load_subject
from .evaluation import run_loso_cell, run_subject_cell
from .io import read_json, write_json_atomic
from .models import recipe_dict

RUN_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware_identity(device: str) -> str:
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(target)
    if target.type == "mps":
        return f"Apple {platform.machine()}"
    return platform.processor() or platform.machine()


def _run_identity(
    *,
    dataset: str,
    protocol: str,
    condition: str,
    model: str,
    seed: int,
    subject: str,
    ica_policy: str,
    epochs: int,
    device: str,
    save_weights: bool,
) -> tuple[str, dict[str, object]]:
    configuration = {
        "schema_version": RUN_SCHEMA_VERSION,
        "code_sha256": _code_fingerprint(),
        "git_revision": _git_revision(),
        "dataset": dataset,
        "protocol": protocol,
        "condition": condition,
        "model": model,
        "seed": seed,
        "split_seed": SPLIT_SEED,
        "subject": subject,
        "ica_policy": ica_policy,
        "epochs": epochs,
        "device": device,
        "device_type": device.split(":", maxsplit=1)[0],
        "hardware": _hardware_identity(device),
        "deterministic_policy": "torch_deterministic_warn_only_cudnn_deterministic",
        "save_weights": save_weights,
        "recipe": recipe_dict(model, epochs),
    }
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), configuration


def cell_path(
    output_dir: Path,
    dataset: str,
    protocol: str,
    condition: str,
    model: str,
    seed: int,
    subject: str,
    ica_policy: str,
) -> Path:
    safe_subject = str(subject).replace("/", "_").replace(" ", "_")
    return (
        output_dir
        / "cells"
        / (
            f"{dataset}__{protocol}__{condition}__ica-{ica_policy}__{model}__"
            f"seed{seed}__{safe_subject}.json"
        )
    )


def run_job(
    *,
    dataset: str,
    models: list[str],
    protocols: list[str],
    conditions: list[str],
    seeds: list[int],
    subjects: list[str] | None,
    device: str,
    epochs: int,
    ica_policy: str,
    output_dir: Path = RESULTS_DIR,
    overwrite: bool = False,
    save_weights: bool = False,
) -> dict[str, int]:
    selected_subjects = subjects or available_subjects(dataset)
    if "loso" in protocols and len(selected_subjects) < 2:
        raise ValueError("LOSO requires at least two subjects")
    completed = 0
    skipped = 0
    checkpoint_dir = output_dir / "checkpoints" if save_weights else None
    if "loso" in protocols:
        recordings = {subject: load_subject(dataset, subject) for subject in selected_subjects}
    else:
        recordings = {}
    for protocol in protocols:
        for condition in conditions:
            for model in models:
                for seed in seeds:
                    for subject in selected_subjects:
                        destination = cell_path(
                            output_dir,
                            dataset,
                            protocol,
                            condition,
                            model,
                            seed,
                            subject,
                            ica_policy,
                        )
                        fingerprint, run_configuration = _run_identity(
                            dataset=dataset,
                            protocol=protocol,
                            condition=condition,
                            model=model,
                            seed=seed,
                            subject=subject,
                            ica_policy=ica_policy,
                            epochs=epochs,
                            device=device,
                            save_weights=save_weights,
                        )
                        if destination.exists() and not overwrite:
                            previous = read_json(destination)
                            if previous.get("run_fingerprint") != fingerprint:
                                raise RuntimeError(
                                    f"Stale cell {destination.name}: configuration or code "
                                    "changed. "
                                    "Use a new output directory or rerun with --overwrite."
                                )
                            skipped += 1
                            continue
                        if protocol == "loso":
                            result = run_loso_cell(
                                recordings,
                                subject,
                                model,
                                condition=condition,
                                device=device,
                                seed=seed,
                                epochs=epochs,
                                ica_policy=ica_policy,
                                checkpoint_dir=checkpoint_dir,
                            )
                        else:
                            if subject not in recordings:
                                recordings[subject] = load_subject(dataset, subject)
                            recording = recordings[subject]
                            result = run_subject_cell(
                                recording,
                                model,
                                protocol=protocol,
                                condition=condition,
                                device=device,
                                seed=seed,
                                epochs=epochs,
                                ica_policy=ica_policy,
                                checkpoint_dir=checkpoint_dir,
                                fold_cache_dir=output_dir / "fold_cache" / fingerprint,
                            )
                        payload = asdict(result)
                        payload["run_fingerprint"] = fingerprint
                        payload["run_configuration"] = run_configuration
                        write_json_atomic(destination, payload)
                        completed += 1
                        print(f"completed {destination.name}", flush=True)
    return {"completed": completed, "skipped": skipped}
