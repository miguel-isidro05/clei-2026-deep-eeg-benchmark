"""Manifest-verified, clean publication export for Peterson results."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

EXCLUDED_ROOTS = {"fold_cache", "logs"}
MANIFEST_RELATIVE_PATH = Path("manifests/publication_artifacts.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_files(results_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(results_dir.rglob("*")):
        relative = path.relative_to(results_dir)
        if path.is_file() and relative.parts[0] not in EXCLUDED_ROOTS:
            files[relative.as_posix()] = path
    return files


def _verified_manifest_files(results_dir: Path) -> list[Path]:
    manifest_path = results_dir / MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing publication manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("files")
    if not isinstance(records, list):
        raise ValueError("Publication manifest has no file records")

    stable = _stable_files(results_dir)
    manifest_key = MANIFEST_RELATIVE_PATH.as_posix()
    expected = {str(record["path"]) for record in records}
    actual = set(stable) - {manifest_key}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"Publication manifest references missing files: {missing[:5]}")
    if extra:
        raise ValueError(f"Publication directory contains unmanifested files: {extra[:5]}")

    for record in records:
        relative = str(record["path"])
        path = stable[relative]
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Publication artifact size mismatch: {relative}")
        if _sha256(path) != str(record["sha256"]):
            raise ValueError(f"Publication artifact hash mismatch: {relative}")
    return [stable[relative] for relative in sorted(expected)] + [manifest_path]


def create_publication_archive(results_dir: Path, archive_path: Path) -> Path:
    """Create a deterministic archive containing only manifest-verified artifacts."""
    results_dir = results_dir.resolve()
    archive_path = archive_path.resolve()
    if results_dir == archive_path or results_dir in archive_path.parents:
        raise ValueError("Write the publication archive outside the results directory")
    files = _verified_manifest_files(results_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.unlink(missing_ok=True)

    with archive_path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                for path in files:
                    relative = path.relative_to(results_dir)
                    info = archive.gettarinfo(
                        str(path), arcname=f"{results_dir.name}/{relative.as_posix()}"
                    )
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as source:
                        archive.addfile(info, source)

    sidecar = archive_path.with_name(f"{archive_path.name}.sha256")
    sidecar.write_text(f"{_sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    return sidecar
