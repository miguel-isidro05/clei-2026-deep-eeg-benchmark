"""Signed hypothesis-isolating catalogs for Exp04b."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

from experiments.peterson_search_v12.common.identity import canonical_sha256


@dataclass(frozen=True)
class SafePathCandidate:
    """One fully specified SafePath training recipe."""

    config_id: str
    wave: str
    family: str
    role: str
    clean_anchor: str
    robust_dropout: float
    training_mode: str
    router_mode: str
    fusion_mode: str
    objective: str
    corruption_family: str
    hard_clean_bypass: bool
    sha256: str


def _candidate(**payload: object) -> SafePathCandidate:
    return SafePathCandidate(**payload, sha256=canonical_sha256(payload))


_REFERENCE = {
    "wave": "w1",
    "role": "candidate",
    "clean_anchor": "g2",
    "robust_dropout": 0.1,
    "training_mode": "frozen_clean",
    "router_mode": "hard_quality",
    "fusion_mode": "logit_residual",
    "objective": "mean_ce",
    "corruption_family": "missing_zero",
    "hard_clean_bypass": True,
}


def _w1_payloads() -> tuple[dict[str, object], ...]:
    variants: list[dict[str, object]] = []

    def add(config_id: str, family: str, **updates: object) -> None:
        variants.append({**_REFERENCE, "config_id": config_id, "family": family, **updates})

    add(
        "exp04b-w1-c00-fbc-clean",
        "controls",
        role="clean_control",
        clean_anchor="fbc",
        router_mode="none",
    )
    add("exp04b-w1-c01-g2-clean", "controls", role="clean_control", router_mode="none")
    add("exp04b-w1-c02-robust-p10", "controls", role="robust_control", router_mode="always")
    add(
        "exp04b-w1-c03-robust-p20",
        "controls",
        role="robust_control",
        router_mode="always",
        robust_dropout=0.2,
    )

    add(
        "exp04b-w1-g00-joint",
        "gradient",
        training_mode="joint",
        router_mode="learned_joint",
        hard_clean_bypass=False,
    )
    add("exp04b-w1-g01-frozen", "gradient")
    add(
        "exp04b-w1-g02-stopgrad",
        "gradient",
        training_mode="stop_gradient",
        router_mode="learned_detached",
    )
    add(
        "exp04b-w1-g03-two-stage",
        "gradient",
        training_mode="two_stage",
        router_mode="learned_detached",
    )

    add("exp04b-w1-r00-oracle", "router", router_mode="oracle")
    add("exp04b-w1-r01-hard-zero", "router", router_mode="hard_zero")
    add("exp04b-w1-r02-hard-quality", "router")
    add("exp04b-w1-r03-learned", "router", router_mode="learned_detached")
    add("exp04b-w1-r04-hybrid", "router", router_mode="hybrid")

    add("exp04b-w1-f00-logit", "fusion")
    add("exp04b-w1-f01-probability", "fusion", fusion_mode="probability_mixture")
    add("exp04b-w1-f02-affine-logit", "fusion", fusion_mode="affine_logit")
    add("exp04b-w1-f03-quality-film", "fusion", fusion_mode="quality_film")

    add("exp04b-w1-o00-mean", "objective")
    add("exp04b-w1-o01-cvar25", "objective", objective="cvar25")
    add("exp04b-w1-o02-group-dro", "objective", objective="group_dro")
    add("exp04b-w1-o03-softmax-max", "objective", objective="softmax_max")

    add("exp04b-w1-x00-zero", "corruption")
    add("exp04b-w1-x01-detected", "corruption", corruption_family="detected_mixed")
    add("exp04b-w1-x02-broad", "corruption", corruption_family="broad_mixed")
    return tuple(variants)


def build_w1_catalog() -> tuple[SafePathCandidate, ...]:
    catalog = tuple(_candidate(**payload) for payload in _w1_payloads())
    if len(catalog) != 24 or len({item.sha256 for item in catalog}) != 24:
        raise RuntimeError("Exp04b W1 catalog drifted")
    return catalog


def build_w2_catalog() -> tuple[SafePathCandidate, ...]:
    catalog: list[SafePathCandidate] = []
    for index, (router, fusion, objective) in enumerate(
        product(
            ("hard_quality", "learned_detached", "hybrid"),
            ("logit_residual", "probability_mixture"),
            ("cvar25", "group_dro"),
        ),
        start=1,
    ):
        payload = {
            **_REFERENCE,
            "config_id": f"exp04b-w2-i{index:02d}",
            "wave": "w2",
            "family": "interaction",
            "router_mode": router,
            "fusion_mode": fusion,
            "objective": objective,
            "corruption_family": "broad_mixed",
        }
        catalog.append(_candidate(**payload))
    return tuple(catalog)


def build_w3_catalog(selected_w2: tuple[str, str, str]) -> tuple[SafePathCandidate, ...]:
    if len(selected_w2) != 3 or len(set(selected_w2)) != 3:
        raise ValueError("W3 requires exactly three distinct W2 candidates")
    w2 = {candidate.config_id: candidate for candidate in build_w2_catalog()}
    if not set(selected_w2) <= set(w2):
        raise ValueError("W3 selection contains a candidate outside W2")
    promoted: list[SafePathCandidate] = []
    for config_id in selected_w2:
        payload = asdict(w2[config_id])
        payload.pop("sha256")
        payload.update(wave="w3", role="promoted")
        promoted.append(_candidate(**payload))
    controls = (
        _candidate(
            **{
                **_REFERENCE,
                "config_id": "exp04b-w3-fbc-control",
                "wave": "w3",
                "family": "confirmation_control",
                "role": "clean_control",
                "clean_anchor": "fbc",
                "router_mode": "none",
            }
        ),
        _candidate(
            **{
                **_REFERENCE,
                "config_id": "exp04b-w3-g2-control",
                "wave": "w3",
                "family": "confirmation_control",
                "role": "clean_control",
                "router_mode": "none",
            }
        ),
        _candidate(
            **{
                **_REFERENCE,
                "config_id": "exp04b-w3-exp04-negative",
                "wave": "w3",
                "family": "confirmation_control",
                "role": "negative_control",
                "training_mode": "joint",
                "router_mode": "learned_joint",
                "hard_clean_bypass": False,
            }
        ),
    )
    return tuple(promoted) + controls


def catalog_payload(catalog: tuple[SafePathCandidate, ...]) -> list[dict[str, object]]:
    return [asdict(candidate) for candidate in catalog]
