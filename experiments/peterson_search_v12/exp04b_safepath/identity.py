"""Code identity scoped to Exp04b without invalidating frozen Exp04 artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def exp04b_code_sha256(root: Path) -> str:
    paths = list((root / "experiments" / "peterson_search_v12" / "exp04b_safepath").rglob("*.py"))
    paths += list((root / "scripts").glob("*peterson_exp04b_v12*.py"))
    paths += list((root / "src" / "deepbench").glob("*.py"))
    paths += [root / "run_peterson_exp04b_cayetano.sh"]
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
