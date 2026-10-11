"""Content identities scoped to the V12 implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.peterson_diffusion_v11.identity import (
    canonical_json,
    canonical_sha256,
    git_identity,
)


def code_sha256(root: Path) -> str:
    paths = list((root / "experiments" / "peterson_search_v12").rglob("*.py"))
    paths += list((root / "experiments" / "peterson_search_v12").rglob("*.json"))
    paths += list((root / "scripts").glob("*peterson_search_v12*.py"))
    paths += [root / "run_peterson_search_cayetano.sh"]
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def exp02_code_sha256(root: Path) -> str:
    """Hash only the executable exp02 surface and its shared V12 dependencies."""
    paths = list((root / "experiments" / "peterson_search_v12" / "common").rglob("*.py"))
    paths += list((root / "experiments" / "peterson_search_v12" / "exp02_factorial").rglob("*.py"))
    paths += list((root / "scripts").glob("*peterson_exp02_v12*.py"))
    paths += list((root / "src" / "deepbench").glob("*.py"))
    paths += [root / "run_peterson_exp02_cayetano.sh"]
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def exp03_code_sha256(root: Path) -> str:
    """Hash only the executable exp03 surface and shared benchmark dependencies."""
    paths = list((root / "experiments" / "peterson_search_v12" / "common").rglob("*.py"))
    paths += list(
        (root / "experiments" / "peterson_search_v12" / "exp03_quality_fbcnet").rglob("*.py")
    )
    paths += list((root / "scripts").glob("*peterson_exp03_v12*.py"))
    paths += list((root / "src" / "deepbench").glob("*.py"))
    paths += [root / "run_peterson_exp03_cayetano.sh"]
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = [
    "canonical_json",
    "canonical_sha256",
    "code_sha256",
    "exp02_code_sha256",
    "exp03_code_sha256",
    "git_identity",
]
