#!/usr/bin/env python3
"""Plan and execute Peterson Exp04 QCR-SSNet discovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.peterson_search_v12.common.identity import (  # noqa: E402
    canonical_sha256,
    exp03_code_sha256,
    git_identity,
)
from experiments.peterson_search_v12.common.io import write_json_atomic  # noqa: E402
from experiments.peterson_search_v12.exp03_quality_fbcnet.analysis import (  # noqa: E402
    analyze_quality_payloads,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.catalog import (  # noqa: E402
    build_quality_catalog as build_exp03_catalog,
)
from experiments.peterson_search_v12.exp03_quality_fbcnet.config import (  # noqa: E402
    Exp03Config,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.catalog import (  # noqa: E402
    build_model_catalog,
    catalog_payload,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.config import (  # noqa: E402
    Exp04Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.manifest import (  # noqa: E402
    build_manifest,
    manifest_payload,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.runner import (  # noqa: E402
    run_discovery,
)
from experiments.peterson_search_v12.exp04_qcr_ssnet.source import (  # noqa: E402
    select_exp03_source,
)


def _signed(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    stored = str(body.pop("sha256", ""))
    if len(stored) != 64 or canonical_sha256(body) != stored:
        raise SystemExit(f"Unsigned Exp03 artifact: {path}")
    return payload


def _load_source(exp03_dir: Path, *, smoke: bool):
    run_complete = json.loads((exp03_dir / "run_complete.json").read_text(encoding="utf-8"))
    if run_complete.get("status") != "completed":
        raise SystemExit("Exp04 requires a completed corrected Exp03 run")
    report = json.loads(
        (exp03_dir / "analysis" / "quality-readout.json").read_text(encoding="utf-8")
    )
    decision = _signed(exp03_dir / "decisions" / "promotion_decision.json")
    audit = json.loads((exp03_dir / "quality_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or audit.get("mode") != "strict":
        raise SystemExit("Exp04 requires a strict completed Exp03 quality audit")
    if run_complete.get("promotion_decision_sha256") != decision["sha256"]:
        raise SystemExit("Exp03 promotion decision does not match run_complete.json")
    if run_complete.get("quality_manifest_sha256") != audit.get("manifest_sha256"):
        raise SystemExit("Exp03 quality manifest does not match run_complete.json")
    if not smoke and audit.get("code_sha256") != exp03_code_sha256(ROOT):
        raise SystemExit("Exp04 requires Exp03 results regenerated with the corrected code")
    for item in report.get("ranking", []):
        if "stress_floor" not in item and isinstance(item.get("mean_stress_accuracy"), dict):
            item["stress_floor"] = min(item["mean_stress_accuracy"].values())
    control = report.get("control", {})
    if "stress_floor" not in control and isinstance(control.get("mean_stress_accuracy"), dict):
        control["stress_floor"] = min(control["mean_stress_accuracy"].values())
    frozen_exp03_catalog = build_exp03_catalog(Exp03Config())
    exp03_candidates = {candidate.config_id: candidate for candidate in frozen_exp03_catalog}
    if not smoke:
        quality_cells = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((exp03_dir / "quality_cells").glob("*.json"))
        ]
        recomputed = analyze_quality_payloads(quality_cells, frozen_exp03_catalog)
        if canonical_sha256(recomputed) != canonical_sha256(report):
            raise SystemExit("Exp03 quality readout does not reproduce from audited cells")
    for item in report.get("ranking", []):
        candidate = exp03_candidates.get(str(item.get("config_id")))
        if candidate is None:
            raise SystemExit("Exp03 readout contains a configuration outside its frozen catalog")
        expected = {
            "gate_id": candidate.gate_id,
            "channel_dropout": candidate.channel_dropout,
            "n_filters_spat": candidate.n_filters_spat,
            "stride_factor": candidate.stride_factor,
        }
        if any(item.get(key) != value for key, value in expected.items()):
            raise SystemExit("Exp03 readout configuration differs from its frozen catalog")
        item["sha256"] = candidate.sha256
    return select_exp03_source(
        report,
        promotion_decision_sha256=str(decision["sha256"]),
        quality_manifest_sha256=str(audit["manifest_sha256"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp03-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = smoke_config() if args.smoke else Exp04Config()
    source = _load_source(args.exp03_dir, smoke=args.smoke)
    catalog = build_model_catalog(config, source)
    revision = str(git_identity(ROOT)["revision"] or "unknown")
    manifest = build_manifest(config, catalog, source, revision=revision)
    write_json_atomic(args.output_dir / "manifests" / "source.json", source.payload())
    write_json_atomic(
        args.output_dir / "manifests" / "model-catalog.json", catalog_payload(catalog)
    )
    write_json_atomic(
        args.output_dir / "manifests" / "discovery-manifest.json",
        manifest_payload(manifest),
    )
    print(f"planned_configs={len(catalog)}")
    print(f"planned_cells={len(manifest.cells)}")
    if args.plan_only:
        return
    summary = run_discovery(
        args.output_dir,
        config,
        catalog,
        source,
        manifest,
        device=args.device,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print(
        f"exp04_shard_complete={summary['completed']}/{summary['planned']} "
        f"failed={summary['failed']}"
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
