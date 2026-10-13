"""Hybrid-only learning gates for Guardian-DSAC V2.3/V2.3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class GuardianLearningGateConfig:
    interval_steps: int = 4_000

    def __post_init__(self) -> None:
        if self.interval_steps < 1:
            raise ValueError("interval_steps must be positive")


def _number(
    evaluation: Mapping[str, object], *names: str, default: float = np.nan
) -> float:
    for name in names:
        if name in evaluation and evaluation[name] is not None:
            return float(evaluation[name])
    return float(default)


def hybrid_hard_safety_failures(
    evaluation: Mapping[str, object],
) -> list[str]:
    """Evaluate only the protected hard-safety contract (no performance)."""

    failures: list[str] = []
    collision = _number(evaluation, "collision_rate", "collision", default=np.inf)
    road = _number(
        evaluation, "road_departure_rate", "road_departure", default=np.inf
    )
    cost = _number(
        evaluation,
        "total_environment_cost",
        "environment_cost",
        "cost",
        default=np.inf,
    )
    solver = _number(
        evaluation, "solver_finite_rate", "solver_finite", default=0.0
    )
    margin = _number(
        evaluation,
        "min_physical_body_sat_margin",
        "min_physical_margin",
        "physical_margin",
        default=-np.inf,
    )
    error = evaluation.get("error")
    if error:
        failures.append("evaluation_error")
    if not np.isfinite(collision) or collision != 0.0:
        failures.append("collision")
    if not np.isfinite(road) or road != 0.0:
        failures.append("road_departure")
    if not np.isfinite(cost) or cost != 0.0:
        failures.append("environment_cost")
    if not np.isfinite(solver) or solver < 1.0:
        failures.append("solver_finite")
    if not np.isfinite(margin) or margin <= 0.0:
        failures.append("physical_margin")
    return sorted(set(failures))


def apply_hybrid_learning_gate(
    algorithm,
    evaluation: Mapping[str, object],
    *,
    numerical_reasons: Sequence[str] = (),
) -> dict[str, object]:
    """Update the passive snapshot or invoke the scoped rollback path."""

    numerical_reasons = tuple(sorted(set(str(value) for value in numerical_reasons)))
    if numerical_reasons:
        disposition = algorithm.reject_numerical_disaster()
        return {
            "passed": False,
            "kind": "numerical_disaster",
            "failures": list(numerical_reasons),
            **disposition,
        }
    failures = hybrid_hard_safety_failures(evaluation)
    if failures:
        disposition = algorithm.reject_hybrid_safety()
        return {
            "passed": False,
            "kind": "hybrid_hard_safety",
            "failures": failures,
            **disposition,
        }
    disposition = algorithm.promote_hybrid_safe()
    return {
        "passed": True,
        "kind": "hybrid_hard_safety",
        "failures": [],
        **disposition,
    }


def apply_multi_seed_hybrid_learning_gate(
    algorithm,
    evaluations_by_seed: Mapping[int | str, Mapping[str, object]],
    *,
    required_seeds: Sequence[int] = (0, 1),
    numerical_reasons: Sequence[str] = (),
) -> dict[str, object]:
    """Apply one atomic Hybrid gate across every supplied scene seed.

    A missing or malformed seed evaluation fails closed.  Numerical disasters
    use the complete passive snapshot; ordinary hard-safety failures restore
    only the Actor so Q and P can continue learning.
    """

    normalized: dict[int, Mapping[str, object]] = {}
    duplicate_seeds: set[int] = set()
    for raw_seed, evaluation in evaluations_by_seed.items():
        seed = int(raw_seed)
        if seed in normalized:
            duplicate_seeds.add(seed)
        normalized[seed] = evaluation
    required_seeds = tuple(sorted(int(seed) for seed in required_seeds))
    if not required_seeds:
        raise ValueError("multi-seed Hybrid gate requires at least one seed")
    if len(required_seeds) != len(set(required_seeds)):
        raise ValueError("multi-seed Hybrid gate required seeds must be unique")
    if duplicate_seeds:
        raise ValueError(
            f"duplicate normalized Hybrid gate seeds: {sorted(duplicate_seeds)}"
        )

    per_seed_failures: dict[str, list[str]] = {}
    for seed in required_seeds:
        evaluation = normalized.get(seed)
        if evaluation is None or not isinstance(evaluation, Mapping):
            failures = ["missing_evaluation"]
        else:
            failures = hybrid_hard_safety_failures(evaluation)
        per_seed_failures[str(seed)] = failures

    numerical_reasons = tuple(
        sorted(set(str(value) for value in numerical_reasons))
    )
    if numerical_reasons:
        disposition = algorithm.reject_numerical_disaster()
        return {
            "required_seeds": list(required_seeds),
            "per_seed_failures": per_seed_failures,
            "passed": False,
            "kind": "numerical_disaster",
            "failures": list(numerical_reasons),
            **disposition,
        }

    aggregate_failures = sorted(
        {
            f"seed{seed}:{failure}"
            for seed in required_seeds
            for failure in per_seed_failures[str(seed)]
        }
    )
    if aggregate_failures:
        disposition = algorithm.reject_hybrid_safety()
        return {
            "required_seeds": list(required_seeds),
            "per_seed_failures": per_seed_failures,
            "passed": False,
            "kind": "hybrid_hard_safety",
            "failures": aggregate_failures,
            **disposition,
        }

    disposition = algorithm.promote_hybrid_safe()
    return {
        "required_seeds": list(required_seeds),
        "per_seed_failures": per_seed_failures,
        "passed": True,
        "kind": "hybrid_hard_safety",
        "failures": [],
        **disposition,
    }


__all__ = [
    "GuardianLearningGateConfig",
    "apply_hybrid_learning_gate",
    "apply_multi_seed_hybrid_learning_gate",
    "hybrid_hard_safety_failures",
]
