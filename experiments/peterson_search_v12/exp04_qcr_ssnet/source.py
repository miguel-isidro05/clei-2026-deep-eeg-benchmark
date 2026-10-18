"""Signed handoff from the corrected Exp03 clean/robust frontier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from experiments.peterson_search_v12.common.identity import canonical_sha256


@dataclass(frozen=True)
class Exp04Source:
    """Selected clean and robust recipes inherited from corrected Exp03."""

    status: str
    clean_config_id: str
    clean_config_sha256: str
    clean_n_filters_spat: int
    clean_stride_factor: int
    clean_accuracy: float
    clean_stress_floor: float
    robust_config_id: str
    robust_config_sha256: str
    robust_n_filters_spat: int
    robust_stride_factor: int
    robust_channel_dropout: float
    robust_accuracy: float
    robust_stress_floor: float
    control_accuracy: float
    promotion_decision_sha256: str
    quality_manifest_sha256: str
    sha256: str

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _signed_candidate_hash(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("sha256", ""))
    if len(value) == 64:
        return value
    return canonical_sha256(candidate)


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must contain a SHA-256 value")


def select_exp03_source(
    report: dict[str, Any],
    *,
    promotion_decision_sha256: str,
    quality_manifest_sha256: str,
) -> Exp04Source:
    """Select a clean/robust Pareto pair or fail closed."""

    _require_sha256(promotion_decision_sha256, "Promotion decision")
    _require_sha256(quality_manifest_sha256, "Quality manifest")
    control = report.get("control")
    ranking = report.get("ranking")
    if not isinstance(control, dict) or not isinstance(ranking, list) or not ranking:
        raise ValueError("Exp03 quality readout is incomplete")
    valid = [
        item
        for item in ranking
        if isinstance(item, dict) and int(item.get("collapse_count", 1)) == 0
    ]
    clean_pool = [item for item in valid if float(item.get("channel_dropout", -1.0)) == 0.0]
    robust_pool = [item for item in valid if float(item.get("channel_dropout", 0.0)) > 0.0]
    if not clean_pool or not robust_pool:
        raise ValueError("Exp03 did not produce both clean and robust candidates")
    clean = max(
        clean_pool,
        key=lambda item: (
            float(item["mean_accuracy"]),
            float(item["lower_quartile"]),
            float(item["mean_kappa"]),
        ),
    )
    robust = max(
        robust_pool,
        key=lambda item: (
            float(item["stress_floor"]),
            float(item["mean_accuracy"]),
            float(item["lower_quartile"]),
        ),
    )
    control_accuracy = float(control["mean_accuracy"])
    clean_accuracy = float(clean["mean_accuracy"])
    clean_stress = float(clean["stress_floor"])
    robust_accuracy = float(robust["mean_accuracy"])
    robust_stress = float(robust["stress_floor"])
    front_reproduced = bool(
        clean_accuracy >= control_accuracy + 0.01
        and int(clean.get("wins_vs_control", 0)) >= 4
        and robust_stress >= clean_stress + 0.03
        and robust_accuracy >= control_accuracy - 0.10
    )
    if not front_reproduced:
        raise ValueError("Corrected Exp03 did not reproduce the required clean/robust front")
    body = {
        "status": "selected",
        "clean_config_id": str(clean["config_id"]),
        "clean_config_sha256": _signed_candidate_hash(clean),
        "clean_n_filters_spat": int(clean["n_filters_spat"]),
        "clean_stride_factor": int(clean["stride_factor"]),
        "clean_accuracy": clean_accuracy,
        "clean_stress_floor": clean_stress,
        "robust_config_id": str(robust["config_id"]),
        "robust_config_sha256": _signed_candidate_hash(robust),
        "robust_n_filters_spat": int(robust["n_filters_spat"]),
        "robust_stride_factor": int(robust["stride_factor"]),
        "robust_channel_dropout": float(robust["channel_dropout"]),
        "robust_accuracy": robust_accuracy,
        "robust_stress_floor": robust_stress,
        "control_accuracy": control_accuracy,
        "promotion_decision_sha256": promotion_decision_sha256,
        "quality_manifest_sha256": quality_manifest_sha256,
    }
    return Exp04Source(**body, sha256=canonical_sha256(body))
