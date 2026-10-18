#!/usr/bin/env python3
"""Audit Exp04 results in portable or strict mode."""

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
    exp04_code_sha256,
    git_identity,
)
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.common.splits import build_search_splits  # noqa: E402
from experiments.peterson_search_v12.exp04_qcr_ssnet.audit import (  # noqa: E402
    audit_results,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.config import (  # noqa: E402
    Exp04Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.manifest import (  # noqa: E402
    Exp04Cell,
    Exp04Manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> Exp04Manifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    stored = str(body.pop("sha256"))
    if canonical_sha256(body) != stored:
        raise SystemExit("Exp04 manifest signature is invalid")
    return Exp04Manifest(
        experiment=payload["experiment"],
        revision=payload["revision"],
        config_sha256=payload["config_sha256"],
        catalog_sha256=payload["catalog_sha256"],
        source_sha256=payload["source_sha256"],
        cells=tuple(
            Exp04Cell(**{**cell, "folds": tuple(cell["folds"])}) for cell in payload["cells"]
        ),
        sha256=stored,
    )


def _strict_data_audit(output_dir: Path, config: Exp04Config) -> None:
    data_root = Path(os.environ["CLEI_DATA_DIR"])
    fragments: list[dict[str, object]] = []
    for subject in config.subjects:
        fragment = json.loads(
            (output_dir / "manifests" / "data" / f"{subject}.json").read_text(encoding="utf-8")
        )
        fragments.append(fragment)
        if _sha256(data_root / f"{subject}.mat") != fragment["source_sha256"]:
            raise SystemExit(f"raw MAT drift for {subject}")
        recording = load_subject("MI-OpenBCI", subject)
        if recording.data_sha256 != fragment["standardized_data_sha256"]:
            raise SystemExit(f"standardized data drift for {subject}")
        splits = build_search_splits(recording, folds=config.folds, split_seed=config.split_seed)
        split_hash = canonical_sha256([split.sha256 for split in splits])
        cells = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (output_dir / "cells").glob(f"*__{subject}.json")
        ]
        if any(
            cell["fingerprint"]["data_sha256"] != recording.data_sha256
            or cell["fingerprint"]["split_sha256"] != split_hash
            for cell in cells
        ):
            raise SystemExit(f"cell data or split drift for {subject}")
    write_json_atomic(
        output_dir / "manifests" / "data-manifest.json",
        {
            "dataset": "MI-OpenBCI",
            "subjects": fragments,
            "sha256": canonical_sha256(fragments),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("portable", "strict"), default="strict")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = smoke_config() if args.smoke else Exp04Config()
    manifest = _load_manifest(args.output_dir / "manifests" / "discovery-manifest.json")
    catalog = json.loads(
        (args.output_dir / "manifests" / "model-catalog.json").read_text(encoding="utf-8")
    )
    if canonical_sha256(catalog) != manifest.catalog_sha256:
        raise SystemExit("Exp04 catalog differs from the signed manifest")
    for candidate in catalog:
        body = dict(candidate)
        stored = str(body.pop("sha256", ""))
        if canonical_sha256(body) != stored:
            raise SystemExit("Exp04 catalog contains an invalid candidate signature")
    source = json.loads((args.output_dir / "manifests" / "source.json").read_text(encoding="utf-8"))
    source_body = dict(source)
    source_stored = str(source_body.pop("sha256", ""))
    if canonical_sha256(source_body) != source_stored or source_stored != manifest.source_sha256:
        raise SystemExit("Exp04 source differs from the signed manifest")
    report = audit_results(args.output_dir, manifest, allow_dirty=args.smoke)
    if args.mode == "strict":
        git = git_identity(ROOT)
        if str(git["revision"] or "unknown") != manifest.revision:
            raise SystemExit("current Git revision differs from Exp04 result manifest")
        if report["code_sha256"] != exp04_code_sha256(ROOT):
            raise SystemExit("result code hash differs from current Exp04 implementation")
        environment_hashes = {
            canonical_sha256({key: payload[key] for key in ("python", "platform", "packages")})
            for path in (args.output_dir / "manifests").glob("environment-*.json")
            for payload in [json.loads(path.read_text(encoding="utf-8"))]
        }
        if environment_hashes != {report["environment_sha256"]}:
            raise SystemExit("scientific environment manifests differ from result cells")
        _strict_data_audit(args.output_dir, config)
    report.update(
        {
            "status": "smoke_completed" if args.smoke else "completed",
            "mode": args.mode,
            "manifest_sha256": manifest.sha256,
            "catalog_sha256": manifest.catalog_sha256,
            "source_sha256": manifest.source_sha256,
        }
    )
    write_json_atomic(args.output_dir / "audit.json", report)
    print(
        f"peterson_exp04_audit=OK mode={args.mode} "
        f"expected={report['expected']} found={report['found']}"
    )


if __name__ == "__main__":
    main()
