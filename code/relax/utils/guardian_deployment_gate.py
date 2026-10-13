"""Seed0/Seed1 deployment certification for Guardian-DSAC V2.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from relax.utils.guardian_learning_gate import hybrid_hard_safety_failures


@dataclass(frozen=True)
class GuardianDeploymentGateConfig:
    interval_steps: int = 10_000
    return_nondegradation_tolerance: float = 0.5
    intervention_nondegradation_pp: float = 0.5
    return_strict_improvement: float = 0.5
    intervention_strict_improvement_pp: float = 0.5

    def __post_init__(self) -> None:
        if self.interval_steps < 1:
            raise ValueError("interval_steps must be positive")
        values = (
            self.return_nondegradation_tolerance,
            self.intervention_nondegradation_pp,
            self.return_strict_improvement,
            self.intervention_strict_improvement_pp,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("deployment tolerances must be finite/non-negative")


def _number(
    evaluation: Mapping[str, object], *names: str, default: float = np.nan
) -> float:
    for name in names:
        if name in evaluation and evaluation[name] is not None:
            return float(evaluation[name])
    return float(default)


def ordinary_hard_safety_failures(
    evaluation: Mapping[str, object],
) -> list[str]:
    failures = hybrid_hard_safety_failures(evaluation)
    success = _number(evaluation, "success_rate", "success", default=0.0)
    if not np.isfinite(success) or success < 1.0:
        failures.append("success")
    return sorted(set(failures))


def _validate_seed_artifacts(
    artifacts: Mapping[int | str, Mapping[str, Mapping[str, object]]],
    required_seeds: Sequence[int],
) -> tuple[dict[int, Mapping[str, Mapping[str, object]]], list[str]]:
    normalised = {int(seed): value for seed, value in artifacts.items()}
    failures: list[str] = []
    for seed in required_seeds:
        if int(seed) not in normalised:
            failures.append(f"seed_{seed}:missing")
            continue
        modes = normalised[int(seed)]
        for mode in ("hybrid", "ordinary"):
            if mode not in modes:
                failures.append(f"seed_{seed}:{mode}:missing")
                continue
            checker = (
                ordinary_hard_safety_failures
                if mode == "ordinary"
                else hybrid_hard_safety_failures
            )
            failures.extend(
                f"seed_{seed}:{mode}:{reason}"
                for reason in checker(modes[mode])
            )
    return normalised, sorted(set(failures))


def aggregate_deployment_performance(
    artifacts: Mapping[int | str, Mapping[str, Mapping[str, object]]],
    *,
    required_seeds: Sequence[int] = (0, 1),
) -> dict[str, float]:
    normalised = {int(seed): value for seed, value in artifacts.items()}
    ordinary_returns = []
    hybrid_interventions = []
    for seed in required_seeds:
        modes = normalised[int(seed)]
        ordinary_returns.append(
            _number(modes["ordinary"], "mean_return", "return")
        )
        hybrid_interventions.append(
            _number(
                modes["hybrid"],
                "actual_intervention_rate",
                "intervention_rate",
            )
        )
    values = np.asarray(
        [*ordinary_returns, *hybrid_interventions], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("deployment performance metrics must be finite")
    return {
        "average_ordinary_return": float(np.mean(ordinary_returns)),
        "average_hybrid_intervention": float(
            np.mean(hybrid_interventions)
        ),
    }


def deployment_gate(
    *,
    candidate_by_seed: Mapping[
        int | str, Mapping[str, Mapping[str, object]]
    ],
    incumbent_by_seed: Mapping[
        int | str, Mapping[str, Mapping[str, object]]
    ]
    | None = None,
    required_seeds: Sequence[int] = (0, 1),
    config: GuardianDeploymentGateConfig = GuardianDeploymentGateConfig(),
) -> dict[str, object]:
    """Hard safety per seed; performance only on the Seed0/1 averages."""

    required_seeds = tuple(int(seed) for seed in required_seeds)
    candidate, failures = _validate_seed_artifacts(
        candidate_by_seed, required_seeds
    )
    result: dict[str, object] = {
        "passed": False,
        "promote": False,
        "failures": failures,
        "required_seeds": list(required_seeds),
        "first_deployable": incumbent_by_seed is None,
    }
    if failures:
        return result
    candidate_average = aggregate_deployment_performance(
        candidate, required_seeds=required_seeds
    )
    result["candidate_average"] = candidate_average
    if incumbent_by_seed is None:
        result.update(
            passed=True,
            promote=True,
            decision="first_ordinary_safe_checkpoint",
        )
        return result

    incumbent, incumbent_failures = _validate_seed_artifacts(
        incumbent_by_seed, required_seeds
    )
    if incumbent_failures:
        raise ValueError(
            "incumbent deployment artifact is invalid: "
            + ", ".join(incumbent_failures)
        )
    incumbent_average = aggregate_deployment_performance(
        incumbent, required_seeds=required_seeds
    )
    result["incumbent_average"] = incumbent_average
    candidate_return = candidate_average["average_ordinary_return"]
    incumbent_return = incumbent_average["average_ordinary_return"]
    candidate_intervention = candidate_average[
        "average_hybrid_intervention"
    ]
    incumbent_intervention = incumbent_average[
        "average_hybrid_intervention"
    ]
    intervention_tolerance = config.intervention_nondegradation_pp / 100.0
    intervention_strict = config.intervention_strict_improvement_pp / 100.0
    return_non_degraded = bool(
        candidate_return
        >= incumbent_return - config.return_nondegradation_tolerance
    )
    intervention_non_degraded = bool(
        candidate_intervention
        <= incumbent_intervention + intervention_tolerance
    )
    return_improved = bool(
        candidate_return > incumbent_return + config.return_strict_improvement
    )
    intervention_improved = bool(
        candidate_intervention < incumbent_intervention - intervention_strict
    )
    passed = bool(
        return_non_degraded
        and intervention_non_degraded
        and (return_improved or intervention_improved)
    )
    result.update(
        passed=passed,
        promote=passed,
        decision="average_performance_promotion" if passed else "retain_incumbent",
        return_non_degraded=return_non_degraded,
        intervention_non_degraded=intervention_non_degraded,
        return_strictly_improved=return_improved,
        intervention_strictly_improved=intervention_improved,
    )
    if not return_non_degraded:
        result["failures"].append("average_return_degraded")
    if not intervention_non_degraded:
        result["failures"].append("average_intervention_degraded")
    if not (return_improved or intervention_improved):
        result["failures"].append("no_average_strict_improvement")
    result["failures"] = sorted(set(result["failures"]))
    return result


def apply_deployment_gate(
    algorithm,
    *,
    candidate_by_seed,
    incumbent_by_seed=None,
    required_seeds: Sequence[int] = (0, 1),
    config: GuardianDeploymentGateConfig = GuardianDeploymentGateConfig(),
) -> dict[str, object]:
    result = deployment_gate(
        candidate_by_seed=candidate_by_seed,
        incumbent_by_seed=incumbent_by_seed,
        required_seeds=required_seeds,
        config=config,
    )
    if result["promote"]:
        algorithm.promote_deployable(
            {
                "gate": result,
                "evaluations": {
                    str(seed): value
                    for seed, value in candidate_by_seed.items()
                },
            }
        )
    else:
        # Explicitly exercise the no-rollback Ordinary path.
        algorithm.record_ordinary_failure()
    return result


__all__ = [
    "GuardianDeploymentGateConfig",
    "aggregate_deployment_performance",
    "apply_deployment_gate",
    "deployment_gate",
    "ordinary_hard_safety_failures",
]
