#!/usr/bin/env python3
"""Summarize a paper run and its durable console logs in Markdown."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

from deepbench.config import RESULTS_DIR
from deepbench.paper_audit import audit_expected_cells
from deepbench.peterson_audit import audit_peterson_results


def _count(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def _integrity_status(results_dir: Path) -> tuple[str, bool, list[str]]:
    if (results_dir / "manifests" / "peterson-journal-expected.json").exists():
        _audit, issues = audit_peterson_results(results_dir)
        return "peterson_journal_v10", not issues, issues
    _audit, confirmatory, issues = audit_expected_cells(results_dir)
    return "legacy_paper_profile", confirmatory, issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--status", choices=("completed", "failed", "running"), default="completed")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    profile, confirmatory, issues = _integrity_status(args.results_dir)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    lines = [
        "# Registro de ejecución del benchmark CLEI 2026",
        "",
        f"- Fecha UTC: {dt.datetime.now(dt.UTC).isoformat()}",
        f"- Estado del proceso: `{args.status}`",
        f"- Commit: `{revision or 'unavailable'}`",
        f"- Perfil de integridad: `{profile}`",
        f"- Celdas JSON: `{_count(args.results_dir / 'cells', '*.json')}`",
        f"- Figuras PNG: `{_count(args.results_dir / 'figures', '*.png')}`",
        f"- Integridad confirmatoria: `{str(confirmatory).lower()}`",
    ]
    if args.log_file:
        lines.append(f"- Log de consola: `{args.log_file}`")
    lines.extend(["", "## Observaciones de integridad", ""])
    lines.extend([f"- {issue}" for issue in issues] or ["- No se detectaron problemas."])
    lines.extend(
        [
            "",
            "## Artefactos esperados",
            "",
            "- `statistics/descriptive_subject_seed_variability.csv`",
            "- `statistics/paired_wilcoxon_holm.csv`",
            "- `statistics/augmentation_wilcoxon_holm.csv`",
            "- `statistics/sample_accounting.csv`",
            "- `quality/signal_quality_channels.csv`",
            "- `latency/latency.csv`",
            "- `figures/*.png` y `figures/*.pdf`",
            "- `logs/*.log`",
            "",
        ]
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "EXPERIMENT_LOG.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
