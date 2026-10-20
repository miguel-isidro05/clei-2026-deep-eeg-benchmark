"""Subject-sharded Exp06 runner reusing the audited Exp05 nested protocol."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from deepbench.datasets import load_subject
from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.identity import canonical_sha256, git_identity
from experiments.peterson_search_v12.common.io import write_json_atomic
from experiments.peterson_search_v12.common.runner import runtime_environment
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.runner import (
    _confusion_matrix,
    _parameter_report,
    _run_fold,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.splits import build_nested_folds

from .catalog import all_candidates
from .models import build_model


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def exp06_code_sha256(root: Path) -> str:
    paths = list((root / "experiments" / "peterson_search_v12" / "common").rglob("*.py"))
    paths += list(
        (root / "experiments" / "peterson_search_v12" / "exp05_fbcnet_mechanisms").rglob("*.py")
    )
    paths += list(
        (root / "experiments" / "peterson_search_v12" / "exp06_window_fusion").rglob("*.py")
    )
    paths += list((root / "scripts").glob("*peterson_exp06_v12*.py"))
    paths += list((root / "src" / "deepbench").glob("*.py"))
    paths += [root / "run_peterson_exp06_cayetano.sh"]
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cell_path(output_dir: Path, phase: str, cell_id: str) -> Path:
    return output_dir / f"{phase}_cells" / f"{cell_id}.json"


def _fingerprint(
    manifest: Any,
    cell: Any,
    candidate: Any,
    *,
    code_hash: str,
    data_hash: str,
    environment: dict[str, Any],
    git: dict[str, Any],
    split_hash: str,
) -> dict[str, Any]:
    return {
        "config_sha256": manifest.config_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "manifest_sha256": manifest.sha256,
        "parent_decision_sha256": manifest.parent_decision_sha256,
        "screen_decision_sha256": manifest.screen_decision_sha256,
        "cell_config_sha256": cell.config_sha256,
        "candidate_sha256": candidate.sha256,
        "code_sha256": code_hash,
        "data_sha256": data_hash,
        "environment_sha256": canonical_sha256(environment),
        "split_sha256": split_hash,
        "revision": git["revision"],
        "git_dirty": git["dirty"],
        "device": environment["device"],
    }


def _resumable(path: Path, fingerprint: dict[str, Any], cell: Any) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("status") == "completed"
        and payload.get("cell_id") == cell.cell_id
        and payload.get("fingerprint") == fingerprint
    )


def run_phase(
    output_dir: Path,
    config: Any,
    catalog: tuple[Any, ...],
    manifest: Any,
    *,
    phase: str,
    device: str,
    num_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    if phase not in {"screen", "confirm"}:
        raise ValueError("Exp06 phase must be screen or confirm")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid Exp06 shard")
    root = _root()
    git = git_identity(root)
    if config.profile == "full" and git["dirty"] is not False:
        raise RuntimeError("Exp06 scientific runs require a clean Git tree")
    if git["revision"] != manifest.revision:
        raise RuntimeError("Exp06 manifest revision does not match HEAD")
    environment = runtime_environment(device)
    code_hash = exp06_code_sha256(root)
    subjects = tuple(config.subjects)
    owned_subjects = set(subjects[shard_index::num_shards])
    cells = tuple(
        cell for cell in manifest.cells if cell.phase == phase and cell.subject in owned_subjects
    )
    if not cells:
        raise ValueError("Exp06 shard owns no cells")
    lookup = {item.config_id: item for item in all_candidates()}
    started = time.perf_counter()
    completed = skipped = failed = 0
    for cell in cells:
        try:
            cell_started = time.perf_counter()
            recording = load_subject(config.dataset, cell.subject)
            splits = build_nested_folds(
                recording,
                split_seed=int(config.split_seed),
                folds=tuple(cell.folds),
                validation_fraction=float(config.inner_validation_fraction),
            )
            candidate = lookup[cell.config_id]
            split_hash = canonical_sha256([split.sha256 for split in splits])
            fingerprint = _fingerprint(
                manifest,
                cell,
                candidate,
                code_hash=code_hash,
                data_hash=str(recording.data_sha256),
                environment=environment,
                git=git,
                split_hash=split_hash,
            )
            path = _cell_path(output_dir, phase, cell.cell_id)
            if _resumable(path, fingerprint, cell):
                skipped += 1
                print(f"skipped {path.name}", flush=True)
                continue
            labels: list[int] = []
            predictions: list[int] = []
            scores: list[float] = []
            fold_reports: list[dict[str, Any]] = []
            flags: set[str] = set()
            last_model = None
            for split_index, split in enumerate(splits):
                report, last_model = _run_fold(
                    candidate,
                    cell,
                    recording,
                    split,
                    config=config,
                    device=device,
                    split_index=split_index,
                    model_builder=build_model,
                )
                fold_reports.append(report)
                labels.extend(report["y_true"])
                predictions.extend(report["y_pred"])
                scores.extend(report["y_score"])
                flags.update(report["quality_flags"])
            if last_model is None:
                raise RuntimeError("Exp06 cell produced no fitted model")
            y_true = np.asarray(labels, dtype=np.int64)
            y_pred = np.asarray(predictions, dtype=np.int64)
            y_score = np.asarray(scores, dtype=float)
            payload = {
                "schema_version": 1,
                "status": "completed",
                "experiment": "exp06",
                "phase": phase,
                "kind": cell.kind,
                "cell_id": cell.cell_id,
                "config_id": cell.config_id,
                "mechanism": cell.mechanism,
                "subject": cell.subject,
                "seed": int(cell.seed),
                "folds": list(cell.folds),
                "candidate": asdict(candidate),
                "aggregate_metrics": compute_metrics(y_true, y_pred, y_score),
                "y_true": labels,
                "y_pred": predictions,
                "y_score": scores,
                "confusion_matrix": _confusion_matrix(y_true, y_pred),
                "quality_flags": sorted(flags),
                "fold_reports": fold_reports,
                "parameter_report": _parameter_report(last_model),
                "fingerprint": fingerprint,
                "runtime": {
                    "seconds": float(time.perf_counter() - cell_started),
                    "environment": environment,
                },
            }
            write_json_atomic(path, payload)
            completed += 1
            print(f"completed {path.name}", flush=True)
        except Exception:
            failed += 1
            traceback.print_exc()
            raise
    summary = {
        "phase": phase,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "owned_subjects": sorted(owned_subjects),
        "planned": len(cells),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "manifest_sha256": manifest.sha256,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(output_dir / "logs" / f"{phase}-shard-{shard_index}-summary.json", summary)
    return summary


def write_run_failure(
    output_dir: Path, *, phase: str, shard_index: int, error: BaseException
) -> None:
    write_json_atomic(
        output_dir / "logs" / f"{phase}-shard-{shard_index}-failure.json",
        {"phase": phase, "shard_index": shard_index, "error": repr(error), "status": "failed"},
    )
