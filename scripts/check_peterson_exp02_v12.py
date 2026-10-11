#!/usr/bin/env python3
"""Audit exp02 in portable or strict data-recomputing mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deepbench.datasets import load_subject  # noqa: E402
from experiments.peterson_search_v12.common.identity import (  # noqa: E402
    canonical_sha256,
    exp02_code_sha256,
    git_identity,
)
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.common.splits import build_search_splits  # noqa: E402
from experiments.peterson_search_v12.exp02_factorial.audit import audit_results  # noqa: E402
from experiments.peterson_search_v12.exp02_factorial.catalog import (  # noqa: E402
    build_candidate_catalog,
    build_inference_catalog,
)
from experiments.peterson_search_v12.exp02_factorial.config import Exp02Config  # noqa: E402
from experiments.peterson_search_v12.exp02_factorial.manifest import (  # noqa: E402
    ExperimentCell,
    ExperimentManifest,
    build_experiment_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> ExperimentManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentManifest(
        experiment=payload["experiment"],
        revision=payload["revision"],
        config_sha256=payload["config_sha256"],
        catalog_sha256=payload["catalog_sha256"],
        cells=tuple(ExperimentCell(**cell) for cell in payload["cells"]),
        sha256=payload["sha256"],
    )


def _strict_data_audit(output_dir: Path, config: Exp02Config) -> None:
    data_root = Path(os.environ["CLEI_DATA_DIR"])
    combined: list[dict[str, object]] = []
    for subject in config.subjects:
        fragment_path = output_dir / "manifests" / "data" / f"{subject}.json"
        if not fragment_path.is_file():
            raise SystemExit(f"missing data manifest for {subject}")
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        combined.append(fragment)
        raw_path = data_root / f"{subject}.mat"
        if _sha256(raw_path) != fragment["source_sha256"]:
            raise SystemExit(f"raw MAT drift for {subject}")
        recording = load_subject("MI-OpenBCI", subject)
        if recording.data_sha256 != fragment["standardized_data_sha256"]:
            raise SystemExit(f"standardized data drift for {subject}")
        splits = build_search_splits(recording, folds=config.folds, split_seed=config.split_seed)
        expected_split = canonical_sha256([split.sha256 for split in splits])
        subject_cells = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (output_dir / "cells").glob(f"*__{subject}.json")
        ]
        if any(
            cell["fingerprint"]["data_sha256"] != recording.data_sha256
            or cell["fingerprint"]["split_sha256"] != expected_split
            for cell in subject_cells
        ):
            raise SystemExit(f"cell data or split drift for {subject}")
    write_json_atomic(
        output_dir / "manifests" / "data_manifest.json",
        {
            "dataset": "MI-OpenBCI",
            "subjects": combined,
            "sha256": canonical_sha256(combined),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("portable", "strict"), default="strict")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = Exp02Config()
    candidates = build_candidate_catalog(config)
    inference = build_inference_catalog(candidates)
    manifest_name = "smoke-exp02.json" if args.smoke else "experiment-exp02.json"
    manifest = _load_manifest(args.output_dir / "manifests" / manifest_name)
    if not args.smoke:
        rebuilt = build_experiment_manifest(config, candidates, revision=manifest.revision)
        if rebuilt.sha256 != manifest.sha256:
            raise SystemExit("exp02 manifest drift detected")
    report = audit_results(
        args.output_dir,
        manifest,
        candidates,
        inference,
        allow_dirty=args.smoke,
    )
    if args.mode == "strict":
        git = git_identity(ROOT)
        if str(git["revision"] or "unknown") != manifest.revision:
            raise SystemExit("current Git revision differs from the result manifest")
        if report["code_sha256"] != exp02_code_sha256(ROOT):
            raise SystemExit("result code hash differs from the current exp02 implementation")
        _strict_data_audit(args.output_dir, config)
    report.update(
        {
            "status": "smoke_completed" if args.smoke else "completed",
            "mode": args.mode,
            "manifest_sha256": manifest.sha256,
        }
    )
    name = "smoke_complete.json" if args.smoke else "run_complete.json"
    write_json_atomic(args.output_dir / name, report)
    print(
        f"peterson_exp02_audit=OK mode={args.mode} "
        f"expected={report['expected']} found={report['found']}"
    )


if __name__ == "__main__":
    main()
