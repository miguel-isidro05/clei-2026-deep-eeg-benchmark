from __future__ import annotations

import json

import deepbench.peterson_audit as peterson_audit
from deepbench.models import recipe_dict


def test_peterson_audit_accepts_recipe_after_json_round_trip(tmp_path, monkeypatch) -> None:
    relative = "cells/example.json"
    code_sha256 = "a" * 64
    monkeypatch.setattr(peterson_audit, "expected_cell_names", lambda: [relative])

    manifest = tmp_path / "manifests" / "peterson-journal-expected.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "expected_cells": [relative],
                "scientific_code_sha256": code_sha256,
            }
        ),
        encoding="utf-8",
    )

    recipe_from_json = json.loads(json.dumps(recipe_dict("EEGNet", 300)))
    cell = tmp_path / relative
    cell.parent.mkdir(parents=True)
    cell.write_text(
        json.dumps(
            {
                "dataset": "MI-OpenBCI",
                "model": "EEGNet",
                "seed": 0,
                "run_configuration": {
                    "epochs": 300,
                    "code_sha256": code_sha256,
                    "recipe": recipe_from_json,
                },
                "fold_reports": [
                    {
                        "estimator_family": "deep_learning",
                        "training_history": [
                            {"epoch": epoch} for epoch in range(1, 301)
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    table, issues = peterson_audit.audit_peterson_results(tmp_path)

    assert issues == []
    assert table.to_dict("records") == [
        {"cell": relative, "expected": True, "present": True}
    ]
