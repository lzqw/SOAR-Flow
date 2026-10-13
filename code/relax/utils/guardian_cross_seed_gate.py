"""Pure, auditable cross-seed promotion rules for Guardian-CalQL V2.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from relax.utils.action_semantics import SPEED_COUPLED_ACTION_SEMANTICS


@dataclass(frozen=True)
class CrossSeedGateConfig:
    online_return_nondegradation_tolerance: float = 0.25
    online_intervention_nondegradation_pp: float = 0.25
    online_return_strict_improvement: float = 0.50
    online_intervention_strict_improvement_pp: float = 0.50
    retention_return_tolerance: float = 0.50
    retention_intervention_tolerance_pp: float = 0.50

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _actual_intervention(mode: Mapping[str, Any]) -> float:
    return float(mode.get("actual_intervention_rate", mode["intervention_rate"]))


def _results(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    results = artifact.get("results")
    if not isinstance(results, Mapping) or not all(
        mode in results for mode in ("hybrid", "ordinary")
    ):
        raise ValueError("evaluation artifact needs Hybrid and Ordinary results")
    return results


def evaluation_binding_failures(
    artifact: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_signature: str,
    expected_manifest_hash: str,
    expected_action_semantics: str = SPEED_COUPLED_ACTION_SEMANTICS,
) -> list[str]:
    failures: list[str] = []
    seed = artifact.get("scene_seed", artifact.get("fixed_scene_seed"))
    if seed != int(expected_seed):
        failures.append("scene_seed")
    if artifact.get("scene_signature") != str(expected_signature):
        failures.append("scene_signature")
    if artifact.get("action_semantics_version") != expected_action_semantics:
        failures.append("action_semantics_version")
    if artifact.get("scene_selection_manifest_hash") != expected_manifest_hash:
        failures.append("scene_selection_manifest_hash")
    return failures


def hard_safety_failures(artifact: Mapping[str, Any], *, label: str) -> list[str]:
    failures: list[str] = []
    try:
        results = _results(artifact)
    except ValueError:
        return [f"{label}:missing_modes"]
    for mode_name in ("hybrid", "ordinary"):
        mode = results[mode_name]
        prefix = f"{label}:{mode_name}"
        if float(mode.get("success_rate", 0.0)) < 1.0:
            failures.append(f"{prefix}:success")
        if float(mode.get("collision_rate", 1.0)) > 0.0:
            failures.append(f"{prefix}:collision")
        if float(mode.get("road_departure_rate", 1.0)) > 0.0:
            failures.append(f"{prefix}:road_departure")
        if float(mode.get("total_environment_cost", np.inf)) > 0.0:
            failures.append(f"{prefix}:environment_cost")
        if float(mode.get("solver_finite_rate", 0.0)) < 1.0:
            failures.append(f"{prefix}:solver_finite")
        if not float(mode.get("min_physical_body_sat_margin", -np.inf)) > 0.0:
            failures.append(f"{prefix}:physical_margin")
    if _actual_intervention(results["ordinary"]) > 0.0:
        failures.append(f"{label}:ordinary:actual_intervention")
    return failures


def cross_seed_candidate_gate(
    *,
    candidate_by_seed: Mapping[int, Mapping[str, Any]],
    champion_by_seed: Mapping[int, Mapping[str, Any]],
    online_scene_seeds: Sequence[int],
    retention_scene_seeds: Sequence[int],
    config: CrossSeedGateConfig = CrossSeedGateConfig(),
) -> dict[str, Any]:
    """Apply adaptation, retention, and hard-safety gates.

    Intervention tolerances are expressed in percentage points on the CLI and
    converted to rates exactly once here.
    """

    online = tuple(int(seed) for seed in online_scene_seeds)
    retention = tuple(int(seed) for seed in retention_scene_seeds)
    if not online or not retention:
        raise ValueError("online and retention seed sets must both be non-empty")
    required = set(online) | set(retention)
    missing = required.difference(candidate_by_seed) | required.difference(
        champion_by_seed
    )
    if missing:
        raise ValueError(
            f"missing candidate/champion evaluations for seeds {sorted(missing)}"
        )

    failures: list[str] = []
    comparisons: dict[str, dict[str, float | bool]] = {}
    strict_improvements: list[bool] = []
    online_i_tol = config.online_intervention_nondegradation_pp / 100.0
    online_i_strict = config.online_intervention_strict_improvement_pp / 100.0
    retention_i_tol = config.retention_intervention_tolerance_pp / 100.0

    for seed in sorted(required):
        failures.extend(
            hard_safety_failures(
                candidate_by_seed[seed], label=f"candidate_seed_{seed}"
            )
        )

    for seed in online:
        candidate = _results(candidate_by_seed[seed])
        champion = _results(champion_by_seed[seed])
        return_delta = float(candidate["ordinary"]["mean_return"]) - float(
            champion["ordinary"]["mean_return"]
        )
        intervention_delta = _actual_intervention(
            candidate["hybrid"]
        ) - _actual_intervention(champion["hybrid"])
        return_ok = return_delta >= -config.online_return_nondegradation_tolerance
        intervention_ok = intervention_delta <= online_i_tol
        strict = bool(
            return_delta >= config.online_return_strict_improvement
            or intervention_delta <= -online_i_strict
        )
        strict_improvements.append(strict)
        if not return_ok:
            failures.append(f"online_seed_{seed}:return_nondegradation")
        if not intervention_ok:
            failures.append(f"online_seed_{seed}:intervention_nondegradation")
        comparisons[str(seed)] = {
            "role_online": True,
            "ordinary_return_delta": return_delta,
            "hybrid_intervention_delta": intervention_delta,
            "return_nondegradation": return_ok,
            "intervention_nondegradation": intervention_ok,
            "strict_improvement": strict,
        }
    if not any(strict_improvements):
        failures.append("online:no_strict_improvement")

    for seed in retention:
        candidate = _results(candidate_by_seed[seed])
        champion = _results(champion_by_seed[seed])
        return_delta = float(candidate["ordinary"]["mean_return"]) - float(
            champion["ordinary"]["mean_return"]
        )
        intervention_delta = _actual_intervention(
            candidate["hybrid"]
        ) - _actual_intervention(champion["hybrid"])
        return_ok = return_delta >= -config.retention_return_tolerance
        intervention_ok = intervention_delta <= retention_i_tol
        if not return_ok:
            failures.append(f"retention_seed_{seed}:return")
        if not intervention_ok:
            failures.append(f"retention_seed_{seed}:intervention")
        row = comparisons.setdefault(str(seed), {})
        row.update(
            {
                "role_retention": True,
                "ordinary_return_delta": return_delta,
                "hybrid_intervention_delta": intervention_delta,
                "return_retained": return_ok,
                "intervention_retained": intervention_ok,
            }
        )

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "online_scene_seeds": list(online),
        "retention_scene_seeds": list(retention),
        "comparisons": comparisons,
    }


__all__ = [
    "CrossSeedGateConfig",
    "cross_seed_candidate_gate",
    "evaluation_binding_failures",
    "hard_safety_failures",
]
