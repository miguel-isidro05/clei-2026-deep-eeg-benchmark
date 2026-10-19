from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from deepbench.metrics import compute_metrics
from experiments.peterson_search_v12.common.identity import canonical_sha256
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.analysis import (
    analyze_confirm,
    analyze_screen,
    load_screen_decision,
    select_screen_config_ids,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.audit import validate_cell
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.config import (
    BASELINE_CONFIG_ID,
    Exp05Config,
    smoke_config,
)
from experiments.peterson_search_v12.exp05_fbcnet_mechanisms.manifest import (
    Exp05Manifest,
    build_confirm_manifest,
    build_screen_manifest,
    manifest_from_payload,
    manifest_payload,
    shard_cells_by_subject,
)


@dataclass(frozen=True)
class _Candidate:
    config_id: str
    mechanism: str
    sha256: str


def _catalog() -> tuple[_Candidate, ...]:
    values = []
    for index in range(1, 21):
        body = {"config_id": f"exp05-e{index:02d}", "mechanism": f"mechanism-{index}"}
        values.append(_Candidate(**body, sha256=canonical_sha256(body)))
    return tuple(values)


def test_exp05_scope_budgets_and_holdout_are_frozen() -> None:
    config = Exp05Config()
    assert config.subjects == ("S09", "S03", "S02", "S10", "S12", "S08")
    assert config.holdout_subjects == ("S07", "S04", "S05", "S06")
    assert config.expected_screen_cells == 378
    assert config.expected_confirm_cells(4) == 60
    with pytest.raises(ValueError, match="holdout"):
        replace(config, subjects=("S07",))
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, confirm_seeds=(2, 3))


def test_phase_manifests_are_signed_and_subject_sharded() -> None:
    config = Exp05Config()
    catalog = _catalog()
    screen = build_screen_manifest(config, catalog, revision="abc123")
    assert len(screen.cells) == 378
    assert {cell.config_id for cell in screen.cells} == {
        BASELINE_CONFIG_ID,
        *(candidate.config_id for candidate in catalog),
    }
    left = shard_cells_by_subject(screen.cells, num_shards=2, shard_index=0)
    right = shard_cells_by_subject(screen.cells, num_shards=2, shard_index=1)
    assert {cell.subject for cell in left}.isdisjoint(cell.subject for cell in right)
    assert {cell.cell_id for cell in left}.isdisjoint(cell.cell_id for cell in right)
    assert len(left) + len(right) == 378
    restored = manifest_from_payload(manifest_payload(screen))
    assert restored == screen
    tampered = manifest_payload(screen)
    tampered["revision"] = "tampered"
    with pytest.raises(ValueError, match="signature"):
        manifest_from_payload(tampered)


def test_confirm_manifest_requires_signed_bounded_screen_selection() -> None:
    config = Exp05Config()
    catalog = _catalog()
    selected = tuple(candidate.config_id for candidate in catalog[:4])
    manifest = build_confirm_manifest(
        config,
        catalog,
        promoted_ids=selected,
        revision="abc123",
        screen_decision_sha256="f" * 64,
    )
    assert len(manifest.cells) == 60
    assert {cell.seed for cell in manifest.cells} == {3, 4}
    assert {cell.config_id for cell in manifest.cells} == {BASELINE_CONFIG_ID, *selected}
    with pytest.raises(ValueError, match="budget"):
        build_confirm_manifest(
            config,
            catalog,
            promoted_ids=tuple(candidate.config_id for candidate in catalog[:5]),
            revision="abc123",
            screen_decision_sha256="f" * 64,
        )


def _write_cell(
    root: Path,
    phase: str,
    config_id: str,
    mechanism: str,
    subject: str,
    seed: int,
    accuracy: float,
    kappa: float,
) -> None:
    folder = root / f"{phase}_cells"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": f"{phase}__{config_id}__seed{seed}__{subject}",
        "config_id": config_id,
        "mechanism": mechanism,
        "kind": "baseline" if config_id == BASELINE_CONFIG_ID else "candidate",
        "subject": subject,
        "seed": seed,
        "aggregate_metrics": {"accuracy": accuracy, "kappa": kappa},
        "fold_reports": [],
        "quality_flags": [],
    }
    (folder / f"{payload['cell_id']}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_analysis_promotes_at_most_four_without_confirmatory_pvalues(tmp_path: Path) -> None:
    config = smoke_config()
    catalog = _catalog()
    screen_manifest = build_screen_manifest(config, catalog, revision="abc123")
    (tmp_path / "manifests").mkdir(parents=True)
    (tmp_path / "manifests" / "screen-manifest.json").write_text(
        json.dumps(manifest_payload(screen_manifest)), encoding="utf-8"
    )
    for subject in config.subjects:
        _write_cell(tmp_path, "screen", BASELINE_CONFIG_ID, "frozen_fbcnet", subject, 0, 0.70, 0.40)
        _write_cell(tmp_path, "screen", "exp05-e01", "global-band-weights", subject, 0, 0.72, 0.43)
        _write_cell(tmp_path, "screen", "exp05-e02", "failed", subject, 0, 0.69, 0.38)
    report = analyze_screen(tmp_path, config)
    assert report["ranking"][0]["config_id"] == "exp05-e01"
    decision = json.loads(
        (tmp_path / "decisions" / "screen-promotion.json").read_text(encoding="utf-8")
    )
    assert decision["selected_config_ids"] == ["exp05-e01"]
    assert decision["confirmatory_p_value_used"] is False
    readout_path = tmp_path / "analysis" / "screen-readout.json"
    original_readout = readout_path.read_text(encoding="utf-8")
    readout = json.loads(original_readout)
    readout["ranking"][0]["eligible"] = False
    readout_path.write_text(json.dumps(readout), encoding="utf-8")
    with pytest.raises(ValueError, match="detached"):
        load_screen_decision(tmp_path)
    readout_path.write_text(original_readout, encoding="utf-8")
    confirm_manifest = build_confirm_manifest(
        config,
        catalog,
        promoted_ids=("exp05-e01",),
        revision="abc123",
        screen_decision_sha256=str(decision["sha256"]),
    )
    (tmp_path / "manifests" / "confirm-manifest.json").write_text(
        json.dumps(manifest_payload(confirm_manifest)), encoding="utf-8"
    )
    for subject in config.subjects:
        _write_cell(
            tmp_path, "confirm", BASELINE_CONFIG_ID, "frozen_fbcnet", subject, 1, 0.70, 0.40
        )
        _write_cell(
            tmp_path, "confirm", "exp05-e01", "global-band-weights", subject, 1, 0.72, 0.43
        )
    final = analyze_confirm(tmp_path, config)
    assert final["ranking"][0]["eligible"] is True
    final_decision = json.loads(
        (tmp_path / "decisions" / "final-promotion.json").read_text(encoding="utf-8")
    )
    assert final_decision["selected_config_id"] == "exp05-e01"
    assert final_decision["holdout_opened"] is False


def test_screen_promotion_prefers_distinct_mechanism_families() -> None:
    readout = {
        "ranking": [
            {"config_id": "spectral-1", "family": "spectral", "eligible": True},
            {"config_id": "spectral-2", "family": "spectral", "eligible": True},
            {"config_id": "spatial-1", "family": "spatial", "eligible": True},
            {"config_id": "objective-1", "family": "objective", "eligible": True},
            {"config_id": "temporal-1", "family": "temporal", "eligible": True},
        ]
    }
    assert select_screen_config_ids(readout, max_promoted=4) == [
        "spectral-1",
        "spatial-1",
        "objective-1",
        "temporal-1",
    ]


def test_cell_audit_recomputes_metrics_and_rejects_nested_leakage() -> None:
    config = smoke_config()
    candidate = _catalog()[0]
    manifest = build_screen_manifest(config, _catalog(), revision="abc123")
    expected = next(cell for cell in manifest.cells if cell.config_id == candidate.config_id)
    outer_sha256 = canonical_sha256(
        {
            "session": "session_0",
            "fold": 0,
            "train_indices": [0, 1, 2, 3],
            "test_indices": [4, 5, 6, 7],
        }
    )
    inner_sha256 = canonical_sha256(
        {"inner_train_indices": [0, 1], "inner_validation_indices": [2, 3]}
    )
    split_body = {
        "split_unit": "trial",
        "outer_train_indices": [0, 1, 2, 3],
        "inner_train_indices": [0, 1],
        "inner_validation_indices": [2, 3],
        "outer_test_indices": [4, 5, 6, 7],
        "outer_sha256": outer_sha256,
        "inner_sha256": inner_sha256,
    }
    split = {**split_body, "sha256": canonical_sha256(split_body)}
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_score)
    nested_split_sha256 = canonical_sha256(
        {
            "session": "session_0",
            "fold": 0,
            "inner_train_indices": split["inner_train_indices"],
            "inner_validation_indices": split["inner_validation_indices"],
            "outer_train_indices": split["outer_train_indices"],
            "outer_test_indices": split["outer_test_indices"],
            "outer_sha256": split["outer_sha256"],
            "inner_sha256": split["inner_sha256"],
        }
    )
    environment = {"device": "cpu", "torch": "test"}
    selection_history = [
        {
            "epoch": 1,
            "train_loss": 0.6,
            "validation_loss": 0.5,
            "learning_rate": 0.001,
            "updates": 1,
        }
    ]
    selection_contract_body = {
        "candidate": expected.config_id,
        "seed": 0,
        "max_epochs": 1,
        "patience": 1,
        "scheduler_t_max": 1,
    }
    refit_history = [
        {"epoch": 1, "train_loss": 0.55, "learning_rate": 0.001, "updates": 1}
    ]
    refit_state_sha256 = "e" * 64
    refit_body = {
        "candidate": expected.config_id,
        "seed": 0,
        "epochs": 1,
        "scheduler_t_max": 1,
        "history": refit_history,
        "state_sha256": refit_state_sha256,
        "updates": 1,
    }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "exp05",
        "phase": "screen",
        "cell_id": expected.cell_id,
        "config_id": expected.config_id,
        "mechanism": expected.mechanism,
        "kind": expected.kind,
        "subject": expected.subject,
        "seed": expected.seed,
        "candidate": asdict(candidate),
        "fingerprint": {
            "cell_config_sha256": expected.config_sha256,
            "config_sha256": manifest.config_sha256,
            "catalog_sha256": manifest.catalog_sha256,
            "manifest_sha256": manifest.sha256,
            "candidate_sha256": candidate.sha256,
            "revision": manifest.revision,
            "git_dirty": False,
            "split_sha256": canonical_sha256([nested_split_sha256]),
            "environment_sha256": canonical_sha256(environment),
            "device": "cpu",
            "code_sha256": "c" * 64,
            "data_sha256": "d" * 64,
        },
        "fold_reports": [
            {
                "outer_fold": 0,
                "session": "session_0",
                "split": split,
                "selection_preprocessing": {"fit_partition": "inner_train"},
                "refit_preprocessing": {"fit_partition": "outer_train"},
                "selected_epoch": 1,
                "best_inner_validation_loss": 0.5,
                "selection_history": selection_history,
                "selection_contract": {
                    **selection_contract_body,
                    "sha256": canonical_sha256(selection_contract_body),
                },
                "selection_trajectory_sha256": canonical_sha256(
                    {
                        "candidate": expected.config_id,
                        "seed": 0,
                        "max_epochs": 1,
                        "patience": 1,
                        "history": selection_history,
                        "selected_epoch": 1,
                    }
                ),
                "refit_epochs": 1,
                "refit_history": refit_history,
                "refit_updates": 1,
                "refit_state_sha256": refit_state_sha256,
                "refit_trajectory_sha256": canonical_sha256(refit_body),
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_score": y_score.tolist(),
                "confusion_matrix": [[1, 1], [0, 2]],
                "metrics": metrics,
                "quality_flags": [],
            }
        ],
        "aggregate_metrics": metrics,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_score": y_score.tolist(),
        "confusion_matrix": [[1, 1], [0, 2]],
        "quality_flags": [],
        "parameter_report": {"trainable": 4994},
        "runtime": {"environment": environment},
    }
    validate_cell(payload, expected, manifest, allow_dirty=False)
    tampered_trajectory = json.loads(json.dumps(payload))
    tampered_trajectory["fold_reports"][0]["selection_history"][0][
        "validation_loss"
    ] = 0.4
    with pytest.raises(ValueError, match="inner-validation loss|trajectory signature"):
        validate_cell(tampered_trajectory, expected, manifest, allow_dirty=False)
    tampered_aggregate = json.loads(json.dumps(payload))
    tampered_aggregate["y_pred"] = [1, 1, 1, 1]
    with pytest.raises(ValueError, match="differs from its fold reports"):
        validate_cell(tampered_aggregate, expected, manifest, allow_dirty=False)
    collapsed = json.loads(json.dumps(payload))
    collapsed_pred = [0, 0, 0, 0]
    collapsed_metrics = compute_metrics(y_true, np.asarray(collapsed_pred), y_score)
    collapsed["fold_reports"][0]["y_pred"] = collapsed_pred
    collapsed["fold_reports"][0]["metrics"] = collapsed_metrics
    collapsed["fold_reports"][0]["confusion_matrix"] = [[2, 0], [2, 0]]
    collapsed["aggregate_metrics"] = collapsed_metrics
    collapsed["y_pred"] = collapsed_pred
    collapsed["confusion_matrix"] = [[2, 0], [2, 0]]
    with pytest.raises(ValueError, match="collapse quality flags"):
        validate_cell(collapsed, expected, manifest, allow_dirty=False)
    leaked = json.loads(json.dumps(payload))
    leaked_split = leaked["fold_reports"][0]["split"]
    leaked_split["inner_validation_indices"] = [2, 4]
    leaked_split["inner_sha256"] = canonical_sha256(
        {
            "inner_train_indices": leaked_split["inner_train_indices"],
            "inner_validation_indices": leaked_split["inner_validation_indices"],
        }
    )
    split_body = {key: value for key, value in leaked_split.items() if key != "sha256"}
    leaked_split["sha256"] = canonical_sha256(split_body)
    with pytest.raises(ValueError, match="cover the outer train"):
        validate_cell(leaked, expected, manifest, allow_dirty=False)


def test_manifest_payload_hash_covers_every_cell() -> None:
    manifest = build_screen_manifest(smoke_config(), _catalog(), revision="abc123")
    payload = manifest_payload(manifest)
    body = dict(payload)
    signature = body.pop("sha256")
    assert signature == canonical_sha256(body)
    assert asdict(manifest.cells[0]) == payload["cells"][0]
    assert isinstance(manifest, Exp05Manifest)
