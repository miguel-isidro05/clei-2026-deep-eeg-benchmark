from __future__ import annotations

import hashlib
import tarfile

import pytest

from deepbench.peterson_export import create_publication_archive
from deepbench.peterson_postprocessing import write_artifact_manifest


def test_publication_archive_is_manifest_verified_and_excludes_runtime_files(tmp_path) -> None:
    results = tmp_path / "results_peterson"
    (results / "cells").mkdir(parents=True)
    (results / "logs").mkdir()
    (results / "fold_cache").mkdir()
    (results / "cells" / "cell.json").write_text('{"accuracy": 0.8}\n', encoding="utf-8")
    (results / "logs" / "failed.log").write_text("historical failure\n", encoding="utf-8")
    (results / "fold_cache" / "fold.json").write_text("{}\n", encoding="utf-8")
    manifest_path = results / "manifests" / "publication_artifacts.json"
    write_artifact_manifest(results, manifest_path)
    archive = tmp_path / "publication.tar.gz"

    sidecar = create_publication_archive(results, archive)

    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert f"{results.name}/cells/cell.json" in names
    assert f"{results.name}/manifests/publication_artifacts.json" in names
    assert not any("/logs/" in name or "/fold_cache/" in name for name in names)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  {archive.name}\n"


def test_publication_archive_rejects_tampered_artifact(tmp_path) -> None:
    results = tmp_path / "results_peterson"
    (results / "cells").mkdir(parents=True)
    cell = results / "cells" / "cell.json"
    cell.write_text('{"accuracy": 0.8}\n', encoding="utf-8")
    manifest_path = results / "manifests" / "publication_artifacts.json"
    write_artifact_manifest(results, manifest_path)
    cell.write_text('{"accuracy": 0.1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        create_publication_archive(results, tmp_path / "publication.tar.gz")


def test_publication_archive_rejects_unmanifested_stable_file(tmp_path) -> None:
    results = tmp_path / "results_peterson"
    (results / "cells").mkdir(parents=True)
    (results / "cells" / "cell.json").write_text('{"accuracy": 0.8}\n', encoding="utf-8")
    manifest_path = results / "manifests" / "publication_artifacts.json"
    write_artifact_manifest(results, manifest_path)
    (results / "unexpected.txt").write_text("not audited\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unmanifested"):
        create_publication_archive(results, tmp_path / "publication.tar.gz")
