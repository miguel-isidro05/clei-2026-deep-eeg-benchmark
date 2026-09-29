"""Resumable orchestration for independent experiment cells."""

from __future__ import annotations

import hashlib
import importlib.metadata
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

RUN_SCHEMA_VERSION = 3
SCIENTIFIC_PACKAGES = (
    "torch",
    "braindecode",
    "moabb",
    "mne",
    "numpy",
    "scipy",
    "scikit-learn",
    "skorch",
    "statsmodels",
    "BCI2kReader",
)


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


@lru_cache(maxsize=1)
def _environment_identity() -> tuple[str, dict[str, str | None]]:
    """Fingerprint versions that can change numerical results."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in SCIENTIFIC_PACKAGES:
        try:
            distribution = importlib.metadata.distribution(package)
            versions[package] = distribution.version
            direct_url = distribution.read_text("direct_url.json")
            versions[f"{package}_direct_url"] = (
                json.dumps(json.loads(direct_url), sort_keys=True, separators=(",", ":"))
                if direct_url
                else None
            )
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            versions[f"{package}_direct_url"] = None
    canonical = json.dumps(versions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), versions


def _cohort_data_identity(recordings: dict[str, object]) -> str:
    identities = {
        subject: getattr(recording, "data_sha256", None)
        for subject, recording in sorted(recordings.items())
    }
    if any(value is None for value in identities.values()):
        raise ValueError("Every loaded recording must contain data_sha256")
    canonical = json.dumps(identities, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


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
    data_sha256: str,
    environment_sha256: str,
    environment_versions: dict[str, str | None] | None = None,
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
        "environment_sha256": environment_sha256,
        "environment_versions": environment_versions,
        "data_sha256": data_sha256,
        "save_weights": save_weights,
        "recipe": recipe_dict(model, epochs),
    }
    operational_configuration = {
        key: value for key, value in configuration.items() if key != "git_revision"
    }
    canonical = json.dumps(operational_configuration, sort_keys=True, separators=(",", ":"))
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
    environment_sha256, environment_versions = _environment_identity()
    if "loso" in protocols:
        recordings = {subject: load_subject(dataset, subject) for subject in selected_subjects}
    else:
        recordings = {}
    for protocol in protocols:
        for condition in conditions:
            for model in models:
                for seed in seeds:
                    for subject in selected_subjects:
                        if subject not in recordings:
                            recordings[subject] = load_subject(dataset, subject)
                        recording = recordings[subject]
                        if protocol == "loso":
                            data_sha256 = _cohort_data_identity(recordings)
                        else:
                            if recording.data_sha256 is None:
                                raise ValueError(
                                    f"{dataset}/{subject} does not provide a data fingerprint"
                                )
                            data_sha256 = recording.data_sha256
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
                            data_sha256=data_sha256,
                            environment_sha256=environment_sha256,
                            environment_versions=environment_versions,
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
