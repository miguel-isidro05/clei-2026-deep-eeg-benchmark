"""Canonical identities for reproducible and resumable V11 experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha256(root: Path) -> str:
    """Hash all V11 code/config entry points, independently from generated results."""
    digest = hashlib.sha256()
    paths = list((root / "experiments" / "peterson_diffusion_v11").rglob("*.py"))
    paths += list((root / "experiments" / "peterson_diffusion_v11").rglob("*.json"))
    paths += list((root / "scripts").glob("*diffusion_v11*.py"))
    paths += [root / "run_diffusion_cayetano.sh"]
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_identity(root: Path) -> dict[str, object]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"revision": revision, "dirty": dirty}
