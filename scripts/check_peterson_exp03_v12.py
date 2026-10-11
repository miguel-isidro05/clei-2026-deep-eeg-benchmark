#!/usr/bin/env python3
"""Audit exp03 recipe or quality results in portable or strict mode."""

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
    exp03_code_sha256,
    git_identity,
)
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.common.splits import build_search_splits  # noqa: E402
from experiments.peterson_search_v12.exp03_quality_fbcnet.audit import (  # noqa: E402
    audit_results,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.config import (  # noqa: E402
    Exp03Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.manifest import (  # noqa: E402
    Exp03Cell,
    Exp03Manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> Exp03Manifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Exp03Manifest(
        experiment=payload["experiment"],
        phase=payload["phase"],
        revision=payload["revision"],
        config_sha256=payload["config_sha256"],
        catalog_sha256=payload["catalog_sha256"],
        parent_decision_sha256=payload["parent_decision_sha256"],
        cells=tuple(
            Exp03Cell(**{**cell, "folds": tuple(cell["folds"])}) for cell in payload["cells"]
        ),
        sha256=payload["sha256"],
    )


def _strict_data_audit(output_dir: Path, config: Exp03Config, phase: str) -> None:
    data_root = Path(os.environ["CLEI_DATA_DIR"])
    fragments: list[dict[str, object]] = []
    folds = config.recipe_folds if phase == "recipe" else config.quality_folds
    for subject in config.subjects:
        fragment = json.loads(
            (output_dir / "manifests" / "data" / f"{subject}.json").read_text(
                encoding="utf-8"
            )
        )
        fragments.append(fragment)
        raw_path = data_root / f"{subject}.mat"
        if _sha256(raw_path) != fragment["source_sha256"]:
            raise SystemExit(f"raw MAT drift for {subject}")
        recording = load_subject("MI-OpenBCI", subject)
        if recording.data_sha256 != fragment["standardized_data_sha256"]:
            raise SystemExit(f"standardized data drift for {subject}")
        splits = build_search_splits(recording, folds=folds, split_seed=config.split_seed)
        split_hash = canonical_sha256([split.sha256 for split in splits])
        cells = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (output_dir / f"{phase}_cells").glob(f"*__{subject}.json")
        ]
        if any(
            cell["fingerprint"]["data_sha256"] != recording.data_sha256
            or cell["fingerprint"]["split_sha256"] != split_hash
            for cell in cells
        ):
            raise SystemExit(f"cell data or split drift for {subject}")
    write_json_atomic(
        output_dir / "manifests" / f"data-manifest-{phase}.json",
        {
            "dataset": "MI-OpenBCI",
            "phase": phase,
            "subjects": fragments,
            "sha256": canonical_sha256(fragments),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("recipe", "quality"), required=True)
    parser.add_argument("--mode", choices=("portable", "strict"), default="strict")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = smoke_config() if args.smoke else Exp03Config()
    manifest = _load_manifest(args.output_dir / "manifests" / f"{args.phase}-manifest.json")
    report = audit_results(args.output_dir, manifest)
    if args.mode == "strict":
        git = git_identity(ROOT)
        if str(git["revision"] or "unknown") != manifest.revision:
            raise SystemExit("current Git revision differs from exp03 result manifest")
        if report["code_sha256"] != exp03_code_sha256(ROOT):
            raise SystemExit("result code hash differs from current exp03 implementation")
        _strict_data_audit(args.output_dir, config, args.phase)
    report.update(
        {
            "status": "smoke_completed" if args.smoke else "completed",
            "mode": args.mode,
            "manifest_sha256": manifest.sha256,
        }
    )
    write_json_atomic(args.output_dir / f"{args.phase}_audit.json", report)
    print(
        f"peterson_exp03_audit=OK phase={args.phase} mode={args.mode} "
        f"expected={report['expected']} found={report['found']}"
    )


if __name__ == "__main__":
    main()

