"""Hybrid CBF--MPC execution layer for JAX and SB3 Safe SAC.

The learned policy acts in the existing two-dimensional target-point space.
``FastMPCControlWrapper`` remains the only component that touches MetaDrive's
physical steering/throttle action: it solves the fixed, hard-constrained Flat
NMPC problem and applies its certified target projection.  This module adds
the supervisory state machine described by the SafeExplorer-to-SAC design:

* an adjustable *authorization* margin (never a relaxed hard CBF);
* hysteretic RL/REC modes with a multi-step MPC recovery target;
* observations augmented with mode and curriculum context;
* explicit raw/commanded/executed actions and recovery segment metadata.

The critic-facing executed action is the certified high-level target.  The
actual low-level steering/throttle command is also logged in ``info`` and is a
deterministic output of the fixed CBF--MPC layer for the current state/target.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Deque, Iterable

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


HYBRID_INFO_KEY = "hybrid_safety"


class HybridMode(IntEnum):
    RL = 0
    REC = 1


@dataclass(frozen=True)
class HybridSafetyConfig:
    """Supervisor and safety-curriculum parameters.

    ``delta_*`` parameters affect only whether the actor is authorized.  The
    CBF gain, collision clearance, road constraint, and projection reserve are
    configured independently in ``make_safe_sac_metadrive_env`` and remain
    fixed for the complete run.
    """

    delta_initial: float = 0.10
    delta_min: float = 0.02
    delta_max: float = 0.30
    delta_step: float = 0.01
    delta_backoff: float = 0.05
    hysteresis_gap: float = 0.04
    release_hold_steps: int = 3
    max_projection_norm: float = 0.20
    release_projection_norm: float = 0.05
    hard_margin_tolerance: float = 1e-3
    recovery_cost: float = 0.05
    projection_cost: float = 0.10
    margin_observation_scale: float = 0.30

    # Independent, dimensionally consistent hard checks.  These decide
    # whether REC is necessary; ``delta_*`` remains a curriculum/authorization
    # diagnostic and never substitutes for a physical hard constraint.
    hard_cbf_inner_margin: float = 0.0
    obstacle_inner_margin: float = 0.0
    road_inner_margin: float = 0.0
    kinematic_inner_margin: float = 0.0
    raw_projection_tolerance: float = 0.02
    large_projection_patience: int = 2

    # Dimensionally separate Actor-authorization reserves.  These are a
    # curriculum outside the immutable hard constraints above: relax slowly
    # after a clean window and tighten quickly after any safety failure.
    cbf_reserve_initial: float = 0.20
    cbf_reserve_min: float = 0.0
    cbf_reserve_max: float = 1.0
    cbf_reserve_relax_step: float = 0.01
    cbf_reserve_tighten_step: float = 0.05
    obstacle_reserve_initial: float = 0.20
    obstacle_reserve_min: float = 0.0
    obstacle_reserve_max: float = 0.60
    obstacle_reserve_relax_step: float = 0.01
    obstacle_reserve_tighten_step: float = 0.05
    road_reserve_initial: float = 0.05
    road_reserve_min: float = 0.0
    road_reserve_max: float = 0.30
    road_reserve_relax_step: float = 0.005
    road_reserve_tighten_step: float = 0.025
    kinematic_reserve_initial: float = 0.01
    kinematic_reserve_min: float = 0.0
    kinematic_reserve_max: float = 0.05
    kinematic_reserve_relax_step: float = 0.001
    kinematic_reserve_tighten_step: float = 0.005
    actor_authority_trajectory_tolerance: float = 0.08
    recoverability_gate_enabled: bool = True
    recoverability_trigger_hard_margin: float = 1.0
    recoverability_trigger_obstacle_clearance: float = 8.0

    # Recovery is complete only after the obstacle is behind the ego and an
    # independently executable ordinary-MPC trajectory remains stable.
    recovery_pass_distance: float = 1.5
    recovery_independence_tolerance: float = 0.08
    recovery_lane_center_tolerance: float = 0.60

    # Proactive inverse-MPC teacher generation for Safe DAgger.
    teacher_enabled: bool = True
    teacher_execute_ordinary_mpc: bool = True
    teacher_trigger_hard_margin: float = 1.0
    teacher_trigger_obstacle_clearance: float = 8.0
    teacher_min_progress_m: float = 5.0
    teacher_lookahead_fraction: float = 1.0
    teacher_weight: float = 1.0

    curriculum_window: int = 10
    curriculum_cooldown_episodes: int = 5
    curriculum_min_recovery_success: float = 0.95
    curriculum_min_margin_q05: float = 0.01
    curriculum_max_recovery_fraction: float = 0.40
    curriculum_max_projection_q95: float = 0.15
    curriculum_min_independence_fraction: float = 0.80
    curriculum_max_solver_failure_fraction: float = 0.0


@dataclass
class _EpisodeAccumulator:
    steps: int = 0
    recovery_steps: int = 0
    recovery_successes: int = 0
    recovery_failures: int = 0
    hard_violations: int = 0
    margins: list[float] = field(default_factory=list)
    projection_norms: list[float] = field(default_factory=list)
    independent_steps: int = 0
    solver_failures: int = 0
    backoff_applied: bool = False


@dataclass(frozen=True)
class _EpisodeSummary:
    recovery_successes: int
    recovery_failures: int
    hard_violations: int
    recovery_fraction: float
    margin_q05: float
    projection_q95: float
    independence_fraction: float
    solver_failure_fraction: float


class HybridCBFMPCEnv(gym.Wrapper):
    """Hysteretic RL/REC supervisor around ``FastMPCControlWrapper``.

    The wrapped environment must expose ``get_current_mgf_field(action)`` and
    return the existing ``mgf``/``executed_target`` diagnostics from ``step``.
    Keeping this contract duck-typed makes the state machine unit-testable
    without launching MetaDrive or JAX.
    """

    CONTEXT_DIM = 4

    def __init__(
        self,
        env: gym.Env,
        *,
        safety: HybridSafetyConfig | None = None,
        recovery_action: Iterable[float] = (1.0, 0.0),
        max_episode_steps: int = 400,
    ):
        super().__init__(env)
        self.safety = safety or HybridSafetyConfig()
        self._validate_config()
        if not isinstance(env.observation_space, Box) or len(env.observation_space.shape) != 1:
            raise TypeError("HybridCBFMPCEnv requires a flat one-dimensional Box observation")
        if not isinstance(env.action_space, Box) or tuple(env.action_space.shape) != (2,):
            raise TypeError("HybridCBFMPCEnv requires the two-dimensional target-point action space")

        self._recovery_action_value = np.clip(
            np.asarray(tuple(recovery_action), dtype=np.float32).reshape(2),
            env.action_space.low,
            env.action_space.high,
        )
        self.max_episode_steps = max(int(max_episode_steps), 1)
        self.authorization_delta = float(
            np.clip(
                self.safety.delta_initial,
                self.safety.delta_min,
                self.safety.delta_max,
            )
        )
        self._authorization_reserves = {
            "cbf": float(self.safety.cbf_reserve_initial),
            "obstacle": float(self.safety.obstacle_reserve_initial),
            "road": float(self.safety.road_reserve_initial),
            "kinematic": float(self.safety.kinematic_reserve_initial),
        }

        base_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        context_low = np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32)
        context_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = Box(
            low=np.concatenate([base_low, context_low]),
            high=np.concatenate([base_high, context_high]),
            dtype=np.float32,
        )
        self.action_space = env.action_space

        self._mode = HybridMode.RL
        self._last_margin = self.delta_off
        self._last_projection_norm = 0.0
        self._episode_step = 0
        self._episode = _EpisodeAccumulator()
        self._history: Deque[_EpisodeSummary] = deque(maxlen=self.safety.curriculum_window)
        self._episodes_since_curriculum_update = 0

        self._next_segment_id = 0
        self._current_segment_id = -1
        self._segment_steps = 0
        self._release_hold_count = 0
        self._cached_recovery_certificate: dict[str, Any] | None = None
        self._cached_recovery_action: np.ndarray | None = None
        self._cached_recovery_teacher: dict[str, Any] | None = None
        self._large_projection_count = 0
        self._recovery_obstacle_seen = False

    def _validate_config(self) -> None:
        cfg = self.safety
        if not (0.0 <= cfg.delta_min <= cfg.delta_initial <= cfg.delta_max):
            raise ValueError("require delta_min <= delta_initial <= delta_max and non-negative margins")
        if cfg.delta_step <= 0.0 or cfg.delta_backoff <= 0.0:
            raise ValueError("curriculum step sizes must be positive")
        if cfg.hysteresis_gap <= 0.0:
            raise ValueError("hysteresis_gap must be positive")
        if cfg.release_hold_steps < 1:
            raise ValueError("release_hold_steps must be >= 1")
        if cfg.curriculum_window < 1:
            raise ValueError("curriculum_window must be >= 1")
        if cfg.large_projection_patience < 1:
            raise ValueError("large_projection_patience must be >= 1")
        if cfg.raw_projection_tolerance < 0.0:
            raise ValueError("raw_projection_tolerance must be nonnegative")
        if cfg.recovery_pass_distance < 0.0:
            raise ValueError("recovery_pass_distance must be nonnegative")
        if cfg.recovery_independence_tolerance < 0.0:
            raise ValueError("recovery_independence_tolerance must be nonnegative")
        if cfg.recovery_lane_center_tolerance < 0.0:
            raise ValueError("recovery_lane_center_tolerance must be nonnegative")
        if cfg.teacher_weight < 0.0:
            raise ValueError("teacher_weight must be nonnegative")
        if cfg.teacher_min_progress_m < 0.0:
            raise ValueError("teacher_min_progress_m must be nonnegative")
        if not 0.0 < cfg.teacher_lookahead_fraction <= 1.0:
            raise ValueError("teacher_lookahead_fraction must be in (0, 1]")
        for name in ("cbf", "obstacle", "road", "kinematic"):
            minimum = float(getattr(cfg, f"{name}_reserve_min"))
            initial = float(getattr(cfg, f"{name}_reserve_initial"))
            maximum = float(getattr(cfg, f"{name}_reserve_max"))
            relax = float(getattr(cfg, f"{name}_reserve_relax_step"))
            tighten = float(getattr(cfg, f"{name}_reserve_tighten_step"))
            if not 0.0 <= minimum <= initial <= maximum:
                raise ValueError(
                    f"require 0 <= {name}_reserve_min <= initial <= max"
                )
            if relax <= 0.0 or tighten <= 0.0:
                raise ValueError(f"{name} reserve steps must be positive")
            if tighten <= relax:
                raise ValueError(
                    f"{name} reserve must tighten faster than it relaxes"
                )
        if cfg.actor_authority_trajectory_tolerance < 0.0:
            raise ValueError(
                "actor_authority_trajectory_tolerance must be nonnegative"
            )
        if cfg.recoverability_trigger_obstacle_clearance < 0.0:
            raise ValueError(
                "recoverability_trigger_obstacle_clearance must be nonnegative"
            )
        if not 0.0 <= cfg.curriculum_min_independence_fraction <= 1.0:
            raise ValueError(
                "curriculum_min_independence_fraction must be in [0, 1]"
            )
        if not 0.0 <= cfg.curriculum_max_solver_failure_fraction <= 1.0:
            raise ValueError(
                "curriculum_max_solver_failure_fraction must be in [0, 1]"
            )

    @property
    def mode(self) -> HybridMode:
        return self._mode

    @property
    def delta_on(self) -> float:
        return float(self.authorization_delta)

    @property
    def delta_off(self) -> float:
        return float(
            min(
                self.authorization_delta + self.safety.hysteresis_gap,
                self.safety.delta_max + self.safety.hysteresis_gap,
            )
        )

    def set_authorization_delta(self, value: float) -> None:
        self.authorization_delta = float(
            np.clip(value, self.safety.delta_min, self.safety.delta_max)
        )

    @property
    def authorization_reserves(self) -> dict[str, float]:
        return dict(self._authorization_reserves)

    def _adjust_authorization_reserves(
        self, *, tighten: bool
    ) -> tuple[dict[str, float], dict[str, float]]:
        old = self.authorization_reserves
        updated: dict[str, float] = {}
        for name, value in old.items():
            suffix = "tighten_step" if tighten else "relax_step"
            step = float(getattr(self.safety, f"{name}_reserve_{suffix}"))
            bound_name = "max" if tighten else "min"
            bound = float(
                getattr(self.safety, f"{name}_reserve_{bound_name}")
            )
            candidate = value + step if tighten else value - step
            updated[name] = float(
                min(candidate, bound) if tighten else max(candidate, bound)
            )
        self._authorization_reserves = updated
        return old, self.authorization_reserves

    def _normalized_reserve_level(self) -> float:
        fractions = []
        for name, value in self._authorization_reserves.items():
            minimum = float(getattr(self.safety, f"{name}_reserve_min"))
            maximum = float(getattr(self.safety, f"{name}_reserve_max"))
            fractions.append(
                (value - minimum) / max(maximum - minimum, 1e-6)
            )
        return float(np.clip(np.mean(fractions), 0.0, 1.0))

    def recovery_action(self) -> np.ndarray:
        """Return the fixed route-centre target used by the recovery MPC."""

        return self._recovery_action_value.copy()

    def _context(self) -> np.ndarray:
        normalized_delta = self._normalized_reserve_level()
        normalized_margin = np.tanh(
            self._last_margin / max(self.safety.margin_observation_scale, 1e-6)
        )
        normalized_projection = np.clip(
            self._last_projection_norm / max(self.safety.max_projection_norm, 1e-6),
            0.0,
            1.0,
        )
        return np.array(
            [
                float(self._mode == HybridMode.REC),
                normalized_delta,
                normalized_margin,
                normalized_projection,
            ],
            dtype=np.float32,
        )

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32).reshape(-1), self._context()]
        ).astype(np.float32, copy=False)

    def _candidate_margin(self, action: np.ndarray) -> tuple[float, str | None]:
        try:
            field = self.env.get_current_mgf_field(action)  # type: ignore[attr-defined]
            margin = float(field.margin_delta)
            if not np.isfinite(margin):
                return float("-inf"), "non_finite_candidate_margin"
            return margin, None
        except Exception as exc:  # the hard layer still owns the final fallback
            return float("-inf"), f"candidate_solve_error:{type(exc).__name__}"

    def _certify_action(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any] | None, float, str | None]:
        """Return a reusable exact certificate plus the unprojected margin."""

        certify = getattr(self.env, "certify_target_action", None)
        if certify is None:
            margin, error = self._candidate_margin(action)
            return None, margin, error
        try:
            certificate = certify(action)
            raw_field = certificate["raw_solved"]["field"]
            margin = float(raw_field.margin_delta)
            if not np.isfinite(margin):
                return certificate, float("-inf"), "non_finite_candidate_margin"
            return certificate, margin, None
        except Exception as exc:
            return None, float("-inf"), f"candidate_solve_error:{type(exc).__name__}"

    @staticmethod
    def _certificate_executed_action(
        certificate: dict[str, Any] | None,
        fallback: np.ndarray,
    ) -> np.ndarray:
        if certificate is None:
            return np.asarray(fallback, dtype=np.float32).reshape(2).copy()
        return np.asarray(
            certificate["solved"]["action"], dtype=np.float32
        ).reshape(2).copy()

    @staticmethod
    def _certificate_hard_metrics(
        certificate: dict[str, Any] | None,
        *,
        raw: bool,
    ) -> dict[str, Any]:
        """Extract separate physical margins from a target certificate."""

        empty = {
            "available": False,
            "solver_finite": False,
            "hard_cbf_margin": float("nan"),
            "obstacle_clearance": float("nan"),
            "road_clearance": float("nan"),
            "kinematic_slack": float("nan"),
            "has_obstacle": False,
            "has_route": False,
        }
        if certificate is None:
            return empty
        solved = certificate.get("raw_solved" if raw else "solved")
        if not isinstance(solved, dict):
            return empty
        hard = solved.get("hard_certificate")
        if hard is None:
            summary = certificate.get(
                "raw_hard_certificate" if raw else "executed_hard_certificate"
            )
            if not isinstance(summary, dict):
                return empty
            return {
                "available": True,
                "solver_finite": bool(
                    summary.get("finite", True)
                    and solved.get("solver_finite", True)
                ),
                **{
                    name: summary.get(name, default)
                    for name, default in (
                        ("hard_cbf_margin", float("nan")),
                        ("obstacle_clearance", float("nan")),
                        ("road_clearance", float("nan")),
                        ("kinematic_slack", float("nan")),
                        ("has_obstacle", False),
                        ("has_route", False),
                    )
                },
            }
        return {
            "available": True,
            "solver_finite": bool(
                getattr(hard, "finite", False)
                and solved.get("solver_finite", True)
            ),
            "hard_cbf_margin": float(hard.min_hard_cbf_margin),
            "obstacle_clearance": float(hard.min_obstacle_clearance),
            "road_clearance": float(hard.min_road_clearance),
            "kinematic_slack": float(hard.min_kinematic_slack),
            "has_obstacle": bool(hard.has_obstacle),
            "has_route": bool(hard.has_route),
        }

    def _hard_metrics_safe(
        self,
        metrics: dict[str, Any],
        *,
        inner: bool,
        authorization: bool = False,
    ) -> bool:
        if not metrics["available"] or not metrics["solver_finite"]:
            return False
        tolerance = 0.0 if inner else -self.safety.hard_margin_tolerance
        hard_cbf_threshold = (
            self.safety.hard_cbf_inner_margin if inner else tolerance
        )
        obstacle_threshold = (
            self.safety.obstacle_inner_margin if inner else tolerance
        )
        road_threshold = self.safety.road_inner_margin if inner else tolerance
        kinematic_threshold = (
            self.safety.kinematic_inner_margin if inner else tolerance
        )
        if authorization:
            hard_cbf_threshold = max(
                hard_cbf_threshold, self._authorization_reserves["cbf"]
            )
            obstacle_threshold = max(
                obstacle_threshold, self._authorization_reserves["obstacle"]
            )
            road_threshold = max(
                road_threshold, self._authorization_reserves["road"]
            )
            kinematic_threshold = max(
                kinematic_threshold,
                self._authorization_reserves["kinematic"],
            )
        return bool(
            (
                not metrics["has_obstacle"]
                or (
                    np.isfinite(metrics["hard_cbf_margin"])
                    and metrics["hard_cbf_margin"] >= hard_cbf_threshold
                    and np.isfinite(metrics["obstacle_clearance"])
                    and metrics["obstacle_clearance"] >= obstacle_threshold
                )
            )
            and (
                not metrics["has_route"]
                or (
                    np.isfinite(metrics["road_clearance"])
                    and metrics["road_clearance"] >= road_threshold
                )
            )
            and np.isfinite(metrics["kinematic_slack"])
            and metrics["kinematic_slack"] >= kinematic_threshold
        )

    @staticmethod
    def _raw_execution_certificate(
        certificate: dict[str, Any],
    ) -> dict[str, Any]:
        raw_certificate = dict(certificate)
        raw_certificate["solved"] = certificate["raw_solved"]
        raw_certificate["projection_used"] = False
        raw_certificate["projection_failed"] = False
        raw_certificate["executed_hard_certificate"] = certificate.get(
            "raw_hard_certificate"
        )
        return raw_certificate

    def _derive_teacher(
        self,
        source_certificate: dict[str, Any] | None,
    ) -> dict[str, Any]:
        invalid = {
            "valid": False,
            "action": self.recovery_action(),
            "trajectory_rmse": float("nan"),
            "hard_cbf_margin": float("nan"),
            "execution_certificate": None,
            "error": None,
        }
        if not self.safety.teacher_enabled or source_certificate is None:
            return invalid
        derive = getattr(self.env, "derive_inverse_mpc_teacher", None)
        if derive is None:
            return invalid
        try:
            teacher = derive(
                source_certificate["solved"],
                hard_margin=self.safety.hard_cbf_inner_margin,
                spatial_margin=self.safety.obstacle_inner_margin,
                road_margin=self.safety.road_inner_margin,
                kinematic_margin=self.safety.kinematic_inner_margin,
                min_progress_m=self.safety.teacher_min_progress_m,
                lookahead_fraction=self.safety.teacher_lookahead_fraction,
            )
            execution_certificate = teacher.get("execution_certificate")
            if execution_certificate is not None:
                execution_certificate["teacher_trajectory_rmse"] = float(
                    teacher.get("trajectory_rmse", np.nan)
                )
                execution_certificate["teacher_action"] = np.asarray(
                    teacher["action"], dtype=np.float32
                ).copy()
            return teacher
        except Exception as exc:
            invalid["error"] = f"teacher_error:{type(exc).__name__}"
            return invalid

    def _build_recovery_plan(
        self,
        source_certificate: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, np.ndarray, dict[str, Any], str | None]:
        certificate = source_certificate
        error = None
        recovery_planner = getattr(
            self.env, "plan_safe_recovery_trajectory", None
        )
        if recovery_planner is not None:
            try:
                recovery_plan = recovery_planner(
                    hard_margin=self.safety.hard_cbf_inner_margin,
                    spatial_margin=self.safety.obstacle_inner_margin,
                    road_margin=self.safety.road_inner_margin,
                    kinematic_margin=self.safety.kinematic_inner_margin,
                    min_progress_m=self.safety.teacher_min_progress_m,
                    pass_distance=self.safety.recovery_pass_distance,
                )
                if recovery_plan.get("valid", False):
                    certificate = recovery_plan["source_certificate"]
            except Exception as exc:
                error = f"recovery_plan_error:{type(exc).__name__}"
        metrics = self._certificate_hard_metrics(certificate, raw=False)
        source_progress = float("inf")
        if certificate is not None:
            try:
                source_progress = float(
                    np.asarray(
                        certificate["solved"]["result"].predicted_x,
                        dtype=np.float32,
                    )[-1]
                )
            except (KeyError, AttributeError, IndexError, TypeError):
                # Lightweight test doubles do not expose the NMPC result.
                source_progress = float("inf")
        if (
            certificate is None
            or bool(certificate.get("projection_failed", False))
            or (metrics["available"] and not self._hard_metrics_safe(metrics, inner=False))
            or source_progress < self.safety.teacher_min_progress_m
        ):
            certificate, _margin, error = self._certify_action(
                self.recovery_action()
            )
        teacher = self._derive_teacher(certificate)
        if (
            teacher.get("valid", False)
            and self.safety.teacher_execute_ordinary_mpc
            and teacher.get("execution_certificate") is not None
        ):
            certificate = teacher["execution_certificate"]
            action = np.asarray(teacher["action"], dtype=np.float32).reshape(2)
        else:
            action = self._certificate_executed_action(
                certificate, self.recovery_action()
            )
        return certificate, action, teacher, error

    def _begin_recovery(self) -> int:
        if self._mode == HybridMode.REC:
            return self._current_segment_id
        segment_id = self._next_segment_id
        self._next_segment_id += 1
        self._current_segment_id = segment_id
        self._segment_steps = 0
        self._release_hold_count = 0
        self._cached_recovery_certificate = None
        self._cached_recovery_action = None
        self._cached_recovery_teacher = None
        self._large_projection_count = 0
        self._recovery_obstacle_seen = False
        self._mode = HybridMode.REC
        return segment_id

    def _end_recovery(self) -> None:
        self._mode = HybridMode.RL
        self._current_segment_id = -1
        self._segment_steps = 0
        self._release_hold_count = 0
        self._cached_recovery_certificate = None
        self._cached_recovery_action = None
        self._cached_recovery_teacher = None
        self._recovery_obstacle_seen = False

    @staticmethod
    def _physical_violation(info: dict[str, Any]) -> bool:
        return bool(
            info.get("crash_vehicle", False)
            or info.get("crash_object", False)
            or info.get("crash_sidewalk", False)
            or info.get("out_of_road", False)
            or info.get("road_departure", False)
        )

    def _apply_backoff(self, reason: str) -> dict[str, Any] | None:
        old_delta = float(self.authorization_delta)
        self.set_authorization_delta(old_delta + self.safety.delta_backoff)
        old_reserves, new_reserves = self._adjust_authorization_reserves(
            tighten=True
        )
        if (
            self.authorization_delta == old_delta
            and old_reserves == new_reserves
        ):
            return None
        self._episodes_since_curriculum_update = 0
        return {
            "kind": "backoff",
            "reason": reason,
            "old_delta": old_delta,
            "new_delta": float(self.authorization_delta),
            "old_reserves": old_reserves,
            "new_reserves": new_reserves,
        }

    def _episode_summary(self) -> _EpisodeSummary:
        margins = np.asarray(self._episode.margins, dtype=np.float32)
        projections = np.asarray(self._episode.projection_norms, dtype=np.float32)
        return _EpisodeSummary(
            recovery_successes=self._episode.recovery_successes,
            recovery_failures=self._episode.recovery_failures,
            hard_violations=self._episode.hard_violations,
            recovery_fraction=(
                self._episode.recovery_steps / max(self._episode.steps, 1)
            ),
            margin_q05=(float(np.quantile(margins, 0.05)) if margins.size else float("-inf")),
            projection_q95=(
                float(np.quantile(projections, 0.95)) if projections.size else float("inf")
            ),
            independence_fraction=(
                self._episode.independent_steps
                / max(self._episode.steps, 1)
            ),
            solver_failure_fraction=(
                self._episode.solver_failures / max(self._episode.steps, 1)
            ),
        )

    def _finish_episode(self) -> dict[str, Any] | None:
        summary = self._episode_summary()
        self._history.append(summary)
        self._episodes_since_curriculum_update += 1

        if summary.hard_violations > 0 and not self._episode.backoff_applied:
            return self._apply_backoff("hard_violation")
        if summary.recovery_failures > 0 and not self._episode.backoff_applied:
            return self._apply_backoff("recovery_failure")
        if len(self._history) < self.safety.curriculum_window:
            return None
        if self._episodes_since_curriculum_update < self.safety.curriculum_cooldown_episodes:
            return None

        history = tuple(self._history)
        successes = sum(item.recovery_successes for item in history)
        failures = sum(item.recovery_failures for item in history)
        recovery_total = successes + failures
        recovery_success_rate = (
            successes / recovery_total if recovery_total > 0 else 1.0
        )
        hard_violations = sum(item.hard_violations for item in history)
        margin_q05 = float(np.quantile([item.margin_q05 for item in history], 0.05))
        recovery_fraction = float(np.mean([item.recovery_fraction for item in history]))
        projection_q95 = float(np.quantile([item.projection_q95 for item in history], 0.95))
        independence_fraction = float(
            np.mean([item.independence_fraction for item in history])
        )
        solver_failure_fraction = float(
            np.mean([item.solver_failure_fraction for item in history])
        )
        eligible = bool(
            hard_violations == 0
            and recovery_success_rate >= self.safety.curriculum_min_recovery_success
            and margin_q05 >= self.safety.curriculum_min_margin_q05
            and recovery_fraction <= self.safety.curriculum_max_recovery_fraction
            and projection_q95 <= self.safety.curriculum_max_projection_q95
            and independence_fraction
            >= self.safety.curriculum_min_independence_fraction
            and solver_failure_fraction
            <= self.safety.curriculum_max_solver_failure_fraction
        )
        if not eligible:
            return None

        old_delta = float(self.authorization_delta)
        self.set_authorization_delta(old_delta - self.safety.delta_step)
        old_reserves, new_reserves = self._adjust_authorization_reserves(
            tighten=False
        )
        if (
            self.authorization_delta == old_delta
            and old_reserves == new_reserves
        ):
            return None
        self._episodes_since_curriculum_update = 0
        return {
            "kind": "expand",
            "reason": "window_gate_passed",
            "old_delta": old_delta,
            "new_delta": float(self.authorization_delta),
            "old_reserves": old_reserves,
            "new_reserves": new_reserves,
            "recovery_success_rate": recovery_success_rate,
            "hard_violations": hard_violations,
            "margin_q05": margin_q05,
            "recovery_fraction": recovery_fraction,
            "projection_q95": projection_q95,
            "independence_fraction": independence_fraction,
            "solver_failure_fraction": solver_failure_fraction,
        }

    def curriculum_state_dict(self) -> dict[str, Any]:
        """Serializable episode-boundary curriculum state."""

        return {
            "format_version": 1,
            "authorization_delta": float(self.authorization_delta),
            "authorization_reserves": self.authorization_reserves,
            "history": [asdict(item) for item in self._history],
            "episodes_since_curriculum_update": int(
                self._episodes_since_curriculum_update
            ),
        }

    def load_curriculum_state_dict(self, state: dict[str, Any]) -> None:
        """Restore curriculum state without weakening configured hard bounds."""

        self.set_authorization_delta(
            float(state.get("authorization_delta", self.authorization_delta))
        )
        provided = dict(state.get("authorization_reserves", {}))
        restored: dict[str, float] = {}
        for name, current in self._authorization_reserves.items():
            minimum = float(getattr(self.safety, f"{name}_reserve_min"))
            maximum = float(getattr(self.safety, f"{name}_reserve_max"))
            restored[name] = float(
                np.clip(provided.get(name, current), minimum, maximum)
            )
        self._authorization_reserves = restored
        history = []
        for row in state.get("history", []):
            try:
                history.append(_EpisodeSummary(**row))
            except (TypeError, ValueError):
                continue
        self._history.clear()
        self._history.extend(history[-self.safety.curriculum_window :])
        self._episodes_since_curriculum_update = max(
            int(state.get("episodes_since_curriculum_update", 0)), 0
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._mode = HybridMode.RL
        self._current_segment_id = -1
        self._segment_steps = 0
        self._release_hold_count = 0
        self._cached_recovery_certificate = None
        self._cached_recovery_action = None
        self._cached_recovery_teacher = None
        self._large_projection_count = 0
        self._recovery_obstacle_seen = False
        self._last_margin = self.delta_off
        self._last_projection_norm = 0.0
        self._episode_step = 0
        self._episode = _EpisodeAccumulator()
        info = dict(info)
        info[HYBRID_INFO_KEY] = {
            "mode": int(self._mode),
            "delta": float(self.authorization_delta),
            "delta_on": self.delta_on,
            "delta_off": self.delta_off,
            "authorization_reserves": self.authorization_reserves,
        }
        return self._augment_observation(observation), info

    def step(self, action):
        raw_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(2),
            self.action_space.low,
            self.action_space.high,
        )
        delta_used = float(self.authorization_delta)
        state_mode = self._mode
        candidate_margin = float("nan")
        candidate_error: str | None = None
        intervention_reason = "none"
        execution_layer = "raw"
        action_certificate: dict[str, Any] | None = None
        source_certificate: dict[str, Any] | None = None
        independent_authority: dict[str, Any] = {
            "available": False,
            "valid": False,
            "trajectory_rmse": float("nan"),
            "ordinary_solver_finite": False,
        }
        backup_checked = False
        backup_feasible = True
        backup_error: str | None = None
        raw_metrics = self._certificate_hard_metrics(None, raw=True)
        teacher: dict[str, Any] = {
            "valid": False,
            "action": raw_action.copy(),
            "trajectory_rmse": float("nan"),
            "hard_cbf_margin": float("nan"),
            "execution_certificate": None,
            "error": None,
        }

        if state_mode == HybridMode.RL:
            source_certificate, candidate_margin, candidate_error = (
                self._certify_action(raw_action)
            )
            action_certificate = source_certificate
            raw_metrics = self._certificate_hard_metrics(
                source_certificate, raw=True
            )
            if raw_metrics["available"]:
                raw_is_safe = self._hard_metrics_safe(
                    raw_metrics, inner=False
                )
                raw_is_authorized = self._hard_metrics_safe(
                    raw_metrics,
                    inner=False,
                    authorization=True,
                )
                planned_action = self._certificate_executed_action(
                    source_certificate, raw_action
                )
                planned_projection = float(
                    np.linalg.norm(planned_action - raw_action)
                )
                planned_metrics = self._certificate_hard_metrics(
                    source_certificate, raw=False
                )
                planned_is_authorized = self._hard_metrics_safe(
                    planned_metrics,
                    inner=False,
                    authorization=True,
                )
                if not raw_is_safe:
                    if not raw_metrics["solver_finite"]:
                        intervention_reason = "solver_failure"
                    elif (
                        raw_metrics["has_obstacle"]
                        and raw_metrics["hard_cbf_margin"]
                        < -self.safety.hard_margin_tolerance
                    ):
                        intervention_reason = "hard_cbf"
                    elif (
                        raw_metrics["has_obstacle"]
                        and raw_metrics["obstacle_clearance"]
                        < -self.safety.hard_margin_tolerance
                    ):
                        intervention_reason = "obstacle_clearance"
                    elif (
                        raw_metrics["has_route"]
                        and raw_metrics["road_clearance"]
                        < -self.safety.hard_margin_tolerance
                    ):
                        intervention_reason = "road_clearance"
                    else:
                        intervention_reason = "kinematic_slack"
                    self._begin_recovery()
                    execution_layer = "recovery"
                elif not raw_is_authorized:
                    intervention_reason = "authorization_reserve"
                    if (
                        planned_is_authorized
                        and planned_projection
                        <= self.safety.max_projection_norm
                    ):
                        execution_layer = "projection"
                        self._large_projection_count = 0
                    else:
                        self._begin_recovery()
                        execution_layer = "recovery"
                elif planned_projection <= self.safety.raw_projection_tolerance:
                    certify_independent = getattr(
                        self.env, "certify_independent_actor_target", None
                    )
                    if certify_independent is None:
                        # Compatibility for simple controllers and unit-test
                        # doubles. Production Flat NMPC always takes the
                        # stricter ordinary-MPC branch below.
                        action_certificate = self._raw_execution_certificate(
                            source_certificate
                        )
                        self._large_projection_count = 0
                        execution_layer = "raw"
                    else:
                        try:
                            independent_authority = {
                                "available": True,
                                **certify_independent(
                                    raw_action,
                                    reference_solved=source_certificate[
                                        "raw_solved"
                                    ],
                                    hard_margin=self._authorization_reserves[
                                        "cbf"
                                    ],
                                    spatial_margin=self._authorization_reserves[
                                        "obstacle"
                                    ],
                                    road_margin=self._authorization_reserves[
                                        "road"
                                    ],
                                    kinematic_margin=(
                                        self._authorization_reserves[
                                            "kinematic"
                                        ]
                                    ),
                                    trajectory_tolerance=(
                                        self.safety
                                        .actor_authority_trajectory_tolerance
                                    ),
                                ),
                            }
                        except Exception as exc:
                            independent_authority = {
                                "available": True,
                                "valid": False,
                                "trajectory_rmse": float("nan"),
                                "ordinary_solver_finite": False,
                                "error": (
                                    "independent_authority_error:"
                                    f"{type(exc).__name__}"
                                ),
                            }
                        if independent_authority.get("valid", False):
                            action_certificate = independent_authority[
                                "execution_certificate"
                            ]
                            self._large_projection_count = 0
                            execution_layer = "ordinary_raw"
                        else:
                            self._begin_recovery()
                            execution_layer = "recovery"
                            intervention_reason = "ordinary_not_independent"
                else:
                    execution_layer = "projection"
                    intervention_reason = "small_projection"
                    if not planned_is_authorized:
                        self._begin_recovery()
                        execution_layer = "recovery"
                        intervention_reason = "projected_reserve_failure"
                    elif planned_projection > self.safety.max_projection_norm:
                        self._large_projection_count += 1
                        intervention_reason = "large_projection"
                        if (
                            self._large_projection_count
                            >= self.safety.large_projection_patience
                        ):
                            self._begin_recovery()
                            execution_layer = "recovery"
                            intervention_reason = "repeated_large_projection"
                    else:
                        self._large_projection_count = 0

                proactive_teacher = bool(
                    source_certificate is not None
                    and raw_metrics["has_obstacle"]
                    and (
                        raw_metrics["hard_cbf_margin"]
                        <= self.safety.teacher_trigger_hard_margin
                        or raw_metrics["obstacle_clearance"]
                        <= self.safety.teacher_trigger_obstacle_clearance
                    )
                )
                if proactive_teacher:
                    teacher = self._derive_teacher(source_certificate)
                    if (
                        self._mode == HybridMode.RL
                        and execution_layer == "projection"
                        and teacher.get("valid", False)
                        and self.safety.teacher_execute_ordinary_mpc
                        and teacher.get("execution_certificate") is not None
                    ):
                        action_certificate = teacher["execution_certificate"]
                        execution_layer = "ordinary_mpc_projection"
                if (
                    self.safety.recoverability_gate_enabled
                    and self._mode == HybridMode.RL
                    and execution_layer == "ordinary_raw"
                    and raw_metrics["has_obstacle"]
                    and (
                        raw_metrics["hard_cbf_margin"]
                        <= self.safety.recoverability_trigger_hard_margin
                        or raw_metrics["obstacle_clearance"]
                        <= self.safety
                        .recoverability_trigger_obstacle_clearance
                    )
                ):
                    backup_planner = getattr(
                        self.env, "plan_safe_recovery_trajectory", None
                    )
                    if backup_planner is not None:
                        backup_checked = True
                        try:
                            preview = backup_planner(
                                hard_margin=self._authorization_reserves[
                                    "cbf"
                                ],
                                spatial_margin=self._authorization_reserves[
                                    "obstacle"
                                ],
                                road_margin=self._authorization_reserves[
                                    "road"
                                ],
                                kinematic_margin=self._authorization_reserves[
                                    "kinematic"
                                ],
                                min_progress_m=self.safety.teacher_min_progress_m,
                                pass_distance=self.safety.recovery_pass_distance,
                            )
                            backup_feasible = bool(preview.get("valid", False))
                        except Exception as exc:
                            backup_feasible = False
                            backup_error = (
                                "backup_preview_error:"
                                f"{type(exc).__name__}"
                            )
                        if not backup_feasible:
                            self._begin_recovery()
                            execution_layer = "recovery"
                            intervention_reason = "backup_not_feasible"
            else:
                # Compatibility path for test doubles and older controllers.
                if candidate_margin < self.delta_on:
                    self._begin_recovery()
                    execution_layer = "recovery"
                    intervention_reason = "legacy_authorization_margin"
                else:
                    self._large_projection_count = 0

        transition_mode = self._mode
        transition_segment_id = (
            self._current_segment_id if transition_mode == HybridMode.REC else -1
        )
        segment_start = bool(
            transition_mode == HybridMode.REC and self._segment_steps == 0
        )

        if transition_mode == HybridMode.REC:
            execution_layer = "recovery"
            if self._cached_recovery_certificate is not None:
                action_certificate = self._cached_recovery_certificate
                commanded_action = np.asarray(
                    self._cached_recovery_action,
                    dtype=np.float32,
                ).reshape(2)
                teacher = self._cached_recovery_teacher or teacher
                self._cached_recovery_certificate = None
                self._cached_recovery_action = None
                self._cached_recovery_teacher = None
            else:
                (
                    action_certificate,
                    commanded_action,
                    recovery_teacher,
                    recovery_error,
                ) = self._build_recovery_plan(source_certificate)
                teacher = recovery_teacher
                if candidate_error is None and recovery_error is not None:
                    candidate_error = recovery_error
        else:
            commanded_action = raw_action.copy()

        execute_certificate = getattr(self.env, "execute_certified_target", None)
        if action_certificate is not None and execute_certificate is not None:
            observation, task_reward, terminated, truncated, info = (
                execute_certificate(action_certificate)
            )
        else:
            observation, task_reward, terminated, truncated, info = self.env.step(
                commanded_action
            )
        self._episode_step += 1
        info = dict(info)
        if self._episode_step >= self.max_episode_steps and not terminated:
            truncated = True
            info["TimeLimit.truncated"] = True

        mgf_info = dict(info.get("mgf", {}))
        executed_action = np.asarray(
            info.get(
                "executed_target",
                mgf_info.get("executed_target", commanded_action),
            ),
            dtype=np.float32,
        ).reshape(2)
        margin = float(
            info.get("m_delta", mgf_info.get("m_delta", candidate_margin))
        )
        projection_failed = bool(mgf_info.get("projection_failed", False))
        command_projection_norm = float(
            np.linalg.norm(executed_action - commanded_action)
        )
        actor_intervention_norm = float(
            np.linalg.norm(executed_action - raw_action)
        )
        projection_norm = actor_intervention_norm

        hard_summary = mgf_info.get("hard_safety")
        if isinstance(hard_summary, dict):
            executed_metrics = {
                "available": True,
                "solver_finite": bool(
                    hard_summary.get("finite", False)
                    and mgf_info.get("solver_finite", True)
                ),
                "hard_cbf_margin": float(
                    hard_summary.get("hard_cbf_margin", np.nan)
                ),
                "obstacle_clearance": float(
                    hard_summary.get("obstacle_clearance", np.nan)
                ),
                "road_clearance": float(
                    hard_summary.get("road_clearance", np.nan)
                ),
                "kinematic_slack": float(
                    hard_summary.get("kinematic_slack", np.nan)
                ),
                "has_obstacle": bool(hard_summary.get("has_obstacle", False)),
                "has_route": bool(hard_summary.get("has_route", False)),
            }
        else:
            executed_metrics = self._certificate_hard_metrics(
                action_certificate, raw=False
            )
        physical_violation = self._physical_violation(info)
        solver_failure = bool(
            executed_metrics["available"]
            and not executed_metrics["solver_finite"]
        )
        hard_cbf_violation = bool(
            executed_metrics["available"]
            and executed_metrics["has_obstacle"]
            and (
                not np.isfinite(executed_metrics["hard_cbf_margin"])
                or executed_metrics["hard_cbf_margin"]
                < -self.safety.hard_margin_tolerance
            )
        )
        obstacle_violation = bool(
            executed_metrics["available"]
            and executed_metrics["has_obstacle"]
            and (
                not np.isfinite(executed_metrics["obstacle_clearance"])
                or executed_metrics["obstacle_clearance"]
                < -self.safety.hard_margin_tolerance
            )
        )
        road_violation = bool(
            executed_metrics["available"]
            and executed_metrics["has_route"]
            and (
                not np.isfinite(executed_metrics["road_clearance"])
                or executed_metrics["road_clearance"]
                < -self.safety.hard_margin_tolerance
            )
        )
        kinematic_violation = bool(
            executed_metrics["available"]
            and (
                not np.isfinite(executed_metrics["kinematic_slack"])
                or executed_metrics["kinematic_slack"]
                < -self.safety.hard_margin_tolerance
            )
        )
        if executed_metrics["available"]:
            certificate_violation = bool(
                solver_failure
                or hard_cbf_violation
                or obstacle_violation
                or road_violation
                or kinematic_violation
                or projection_failed
            )
        else:
            certificate_violation = bool(
                not np.isfinite(margin)
                or margin < -self.safety.hard_margin_tolerance
                or projection_failed
            )
        hard_violation = bool(physical_violation or certificate_violation)
        done = bool(terminated or truncated)
        task_success = bool(
            info.get("success", False) or info.get("arrive_dest", False)
        )

        obstacle_count = int(mgf_info.get("obstacle_count", 0))
        max_obstacle_longitudinal = float(
            mgf_info.get("max_obstacle_longitudinal", np.nan)
        )
        obstacle_passed = True
        cbf_independent = True
        lane_center_offset = float(
            info.get("mpc_control", {})
            .get("vehicle_state", {})
            .get("lateral_offset", np.nan)
        )
        cbf_trajectory_delta = float(
            mgf_info.get(
                "cbf_trajectory_delta",
                teacher.get("trajectory_rmse", np.nan),
            )
        )
        if independent_authority.get("available", False):
            cbf_independent = bool(
                independent_authority.get("valid", False)
                and not bool(mgf_info.get("obstacle_cbf_enabled", True))
                and np.isfinite(cbf_trajectory_delta)
                and cbf_trajectory_delta
                <= self.safety.actor_authority_trajectory_tolerance
            )
        if transition_mode == HybridMode.REC and executed_metrics["available"]:
            if obstacle_count > 0:
                self._recovery_obstacle_seen = bool(
                    self._recovery_obstacle_seen
                    or max_obstacle_longitudinal
                    >= -self.safety.recovery_pass_distance
                )
            obstacle_passed = bool(
                not self._recovery_obstacle_seen
                or obstacle_count == 0
                or max_obstacle_longitudinal
                <= -self.safety.recovery_pass_distance
            )
            cbf_independent = bool(
                (
                    not executed_metrics["has_obstacle"]
                    or not bool(mgf_info.get("obstacle_cbf_enabled", True))
                )
                and (
                    not np.isfinite(cbf_trajectory_delta)
                    or cbf_trajectory_delta
                    <= self.safety.recovery_independence_tolerance
                )
            )

        segment_end = False
        recovery_success = -1
        release_ready = False
        if transition_mode == HybridMode.REC:
            self._segment_steps += 1
            if executed_metrics["available"]:
                release_ready = bool(
                    not hard_violation
                    and self._hard_metrics_safe(executed_metrics, inner=True)
                    and obstacle_passed
                    and cbf_independent
                    and command_projection_norm
                    <= self.safety.release_projection_norm
                    and np.isfinite(lane_center_offset)
                    and abs(lane_center_offset)
                    <= self.safety.recovery_lane_center_tolerance
                    and not info.get("out_of_road", False)
                    and not info.get("road_departure", False)
                )
            else:
                # Compatibility path for controllers without split margins.
                release_ready = bool(
                    not hard_violation
                    and np.isfinite(margin)
                    and margin >= self.delta_off
                    and command_projection_norm
                    <= self.safety.release_projection_norm
                )
            self._release_hold_count = (
                self._release_hold_count + 1 if release_ready else 0
            )
            if done:
                segment_end = True
                recovery_success = int(
                    task_success and not hard_violation and obstacle_passed
                )
            elif self._release_hold_count >= self.safety.release_hold_steps:
                segment_end = True
                recovery_success = 1

            if segment_end:
                if recovery_success:
                    self._episode.recovery_successes += 1
                else:
                    self._episode.recovery_failures += 1
                self._end_recovery()
        elif not done and hard_violation:
            self._begin_recovery()
            if intervention_reason == "none":
                intervention_reason = "post_execution_hard_violation"

        next_mode = self._mode
        next_recovery_action = self.recovery_action()
        next_recovery_preview_error: str | None = None
        if next_mode == HybridMode.REC and not done:
            (
                self._cached_recovery_certificate,
                self._cached_recovery_action,
                self._cached_recovery_teacher,
                next_recovery_preview_error,
            ) = self._build_recovery_plan()
            next_recovery_action = np.asarray(
                self._cached_recovery_action, dtype=np.float32
            ).reshape(2)
        else:
            self._cached_recovery_certificate = None
            self._cached_recovery_action = None
            self._cached_recovery_teacher = None

        learned_reward = float(task_reward)
        if transition_mode == HybridMode.REC:
            learned_reward -= self.safety.recovery_cost
        learned_reward -= self.safety.projection_cost * projection_norm**2

        observed_margin = (
            executed_metrics["hard_cbf_margin"]
            if executed_metrics["available"]
            and executed_metrics["has_obstacle"]
            and np.isfinite(executed_metrics["hard_cbf_margin"])
            else margin
        )
        self._last_margin = (
            observed_margin
            if np.isfinite(observed_margin)
            else -self.safety.margin_observation_scale
        )
        self._last_projection_norm = projection_norm
        self._episode.steps += 1
        self._episode.recovery_steps += int(transition_mode == HybridMode.REC)
        self._episode.hard_violations += int(hard_violation)
        self._episode.independent_steps += int(cbf_independent)
        self._episode.solver_failures += int(solver_failure)
        if np.isfinite(observed_margin):
            self._episode.margins.append(observed_margin)
        if transition_mode == HybridMode.RL:
            self._episode.projection_norms.append(projection_norm)

        curriculum_update = None
        if hard_violation and not self._episode.backoff_applied:
            curriculum_update = self._apply_backoff("hard_violation")
            self._episode.backoff_applied = curriculum_update is not None
        if done:
            episode_update = self._finish_episode()
            if curriculum_update is None:
                curriculum_update = episode_update

        low_level_action = np.asarray(
            info.get("mpc_control", {}).get(
                "low_level_action", [np.nan, np.nan]
            ),
            dtype=np.float32,
        ).reshape(2)
        actor_controlled = bool(
            transition_mode == HybridMode.RL
            and execution_layer in ("raw", "ordinary_raw")
            and projection_norm <= self.safety.raw_projection_tolerance
            and (
                execution_layer == "raw"  # compatibility-only controller
                or (
                    independent_authority.get("valid", False)
                    and cbf_independent
                    and not hard_violation
                )
            )
        )
        intervention = bool(not actor_controlled)
        teacher_valid = bool(teacher.get("valid", False))
        teacher_action = np.asarray(
            teacher.get("action", executed_action), dtype=np.float32
        ).reshape(2)
        hybrid_info: dict[str, Any] = {
            "mode": int(state_mode),
            "execution_mode": int(transition_mode),
            "execution_layer": execution_layer,
            "actor_controlled": actor_controlled,
            "intervention": intervention,
            "proposal_evaluated": bool(state_mode == HybridMode.RL),
            "next_mode": int(next_mode),
            "delta": delta_used,
            "next_delta": float(self.authorization_delta),
            "delta_on": delta_used,
            "delta_off": float(
                min(
                    delta_used + self.safety.hysteresis_gap,
                    self.safety.delta_max + self.safety.hysteresis_gap,
                )
            ),
            "authorization_reserves": self.authorization_reserves,
            "raw_action": raw_action.copy(),
            "commanded_action": commanded_action.copy(),
            "executed_action": executed_action.copy(),
            "low_level_action": low_level_action.copy(),
            "recovery_action": (
                executed_action.copy()
                if transition_mode == HybridMode.REC
                else self.recovery_action()
            ),
            "next_recovery_action": next_recovery_action.copy(),
            "teacher_action": teacher_action.copy(),
            "teacher_valid": teacher_valid,
            "teacher_weight": (
                self.safety.teacher_weight if teacher_valid else 0.0
            ),
            "teacher_trajectory_rmse": float(
                teacher.get("trajectory_rmse", np.nan)
            ),
            "teacher_hard_cbf_margin": float(
                teacher.get("hard_cbf_margin", np.nan)
            ),
            "teacher_error": teacher.get("error"),
            "candidate_margin": candidate_margin,
            "margin": margin,
            "hard_cbf_margin": float(executed_metrics["hard_cbf_margin"]),
            "obstacle_clearance": float(
                executed_metrics["obstacle_clearance"]
            ),
            "road_clearance": float(executed_metrics["road_clearance"]),
            "kinematic_slack": float(executed_metrics["kinematic_slack"]),
            "solver_failure": solver_failure,
            "hard_cbf_violation": hard_cbf_violation,
            "obstacle_violation": obstacle_violation,
            "road_violation": road_violation,
            "kinematic_violation": kinematic_violation,
            "projection_norm": projection_norm,
            "command_projection_norm": command_projection_norm,
            "actor_intervention_norm": actor_intervention_norm,
            "projection_failed": projection_failed,
            "hard_violation": hard_violation,
            "physical_violation": physical_violation,
            "obstacle_cbf_enabled": bool(
                mgf_info.get("obstacle_cbf_enabled", True)
            ),
            "cbf_trajectory_delta": cbf_trajectory_delta,
            "cbf_independent": cbf_independent,
            "independent_authority_available": bool(
                independent_authority.get("available", False)
            ),
            "independent_authority_valid": bool(
                independent_authority.get("valid", False)
            ),
            "independent_trajectory_rmse": float(
                independent_authority.get("trajectory_rmse", np.nan)
            ),
            "independent_authority_error": independent_authority.get(
                "error"
            ),
            "backup_checked": backup_checked,
            "backup_feasible": backup_feasible,
            "backup_error": backup_error,
            "obstacle_passed": obstacle_passed,
            "lane_center_offset": lane_center_offset,
            "recovery_release_ready": release_ready,
            "intervention_reason": intervention_reason,
            "candidate_error": candidate_error,
            "next_recovery_preview_error": next_recovery_preview_error,
            "segment_id": int(transition_segment_id),
            "segment_start": segment_start,
            "segment_end": segment_end,
            "recovery_success": int(recovery_success),
            "task_reward": float(task_reward),
            "learned_reward": learned_reward,
            "curriculum_update": curriculum_update,
        }
        info[HYBRID_INFO_KEY] = hybrid_info
        info.update(
            {
                "hybrid_mode": int(transition_mode),
                "hybrid_state_mode": int(state_mode),
                "hybrid_next_mode": int(next_mode),
                "hybrid_delta": delta_used,
                "hybrid_margin": margin,
                "hybrid_hard_cbf_margin": float(
                    executed_metrics["hard_cbf_margin"]
                ),
                "hybrid_projection_norm": projection_norm,
                "hybrid_recovery_success": int(recovery_success),
                "hybrid_hard_violation": int(hard_violation),
                "task_reward": float(task_reward),
            }
        )
        return (
            self._augment_observation(observation),
            learned_reward,
            bool(terminated),
            bool(truncated),
            info,
        )


def make_safe100_base_env(
    *,
    use_render: bool = False,
    obstacle_specs: Iterable[tuple[int, float]] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> gym.Env:
    """Build the low-level MetaDrive task used by PPO data collection.

    The returned environment exposes MetaDrive's native steering/throttle
    action.  It deliberately does not construct or warm up Flat NMPC, so the
    bundled MetaDrive PPO expert can drive it directly.  Passing an empty
    ``obstacle_specs`` tuple creates the same three-lane route without parked
    vehicles; ``None`` keeps the online default middle-lane obstacle.
    """

    from relax.env.drive.lane_change import (
        StaticObstacleSpec,
        ThreeLaneStraightEnv,
    )

    parked_obstacles = (
        ((1, 0.48),) if obstacle_specs is None else tuple(obstacle_specs)
    )
    for lane_index, longitudinal_fraction in parked_obstacles:
        if int(lane_index) not in (0, 1, 2):
            raise ValueError("obstacle lane index must be 0, 1, or 2")
        if not 0.0 < float(longitudinal_fraction) < 1.0:
            raise ValueError("obstacle longitudinal fraction must be in (0, 1)")

    class ParkedCarThreeLaneEnv(ThreeLaneStraightEnv):
        """Deterministic route with a configurable set of stopped cars."""

        def _build_static_obstacle_specs(
            self, lane_length: float, spawn_longitude: float
        ):
            del lane_length, spawn_longitude
            return [
                StaticObstacleSpec(
                    lane_index=int(lane_index),
                    spawn_mode="fraction",
                    longitudinal_fraction=float(longitudinal_fraction),
                    length=4.5,
                    width=2.0,
                    vehicle_model="m",
                    circle_count=3,
                )
                for lane_index, longitudinal_fraction in parked_obstacles
            ]

    config: dict[str, Any] = {
        "use_render": bool(use_render),
        "window_size": (1280, 800),
        "show_fps": False,
        "show_logo": False,
        "show_mouse": bool(use_render),
        "camera_height": 6.0,
        "camera_dist": 12.0,
        "camera_pitch": None,
        "camera_smooth": False,
        "use_chase_camera_follow_lane": False,
        "log_level": logging.ERROR,
    }
    if config_overrides:
        config.update(config_overrides)
    return ParkedCarThreeLaneEnv(config)


def make_safe_sac_metadrive_env(
    *,
    use_render: bool = False,
    safety: HybridSafetyConfig | None = None,
    max_episode_steps: int = 400,
    lookahead: float = 10.0,
    lateral_scale: float = 3.7,
    target_velocity: float = 4.8,
    horizon: float = 2.0,
    clearance: float = 0.10,
    cbf_gamma: float = 2.0,
    projection_reserve: float = 0.10,
    projection_kinematic_reserve: float = 0.02,
    sqp_steps: int = 24,
    admm_steps: int = 240,
    obstacle_specs: Iterable[tuple[int, float]] | None = None,
    explicit_obstacle_observation: bool = True,
) -> HybridCBFMPCEnv:
    """Build the three-lane task with high-level Safe-SAC control.

    The physical safety parameters below are deliberately fixed.  Curriculum
    changes only ``HybridSafetyConfig.delta_*`` in the outer supervisor.
    """

    from gymnasium.wrappers import FlattenObservation

    from relax.env.drive.lane_change_mpc import FastMPCControlWrapper

    base_env = make_safe100_base_env(
        use_render=use_render,
        obstacle_specs=obstacle_specs,
    )
    controller = FastMPCControlWrapper(
        FlattenObservation(base_env),
        position_scale=lookahead,
        target_lateral_scale=lateral_scale,
        wheelbase=2.46894,
        max_velocity=8.0,
        max_steering=0.6981317008,
        target_velocity=target_velocity,
        w_tracking=10.0,
        w_velocity=30.0,
        w_risk=0.0,
        w_cbf=1000.0,
        w_road=100.0,
        cbf_solve_weight_scale=1.0,
        road_solve_weight_scale=1.0,
        risk_buffer=1.2,
        risk_temperature=2.0,
        road_buffer=1.5,
        road_temperature=0.5,
        cbf_enable=True,
        d_clearance=clearance,
        cbf_gamma=cbf_gamma,
        max_obstacles=16,
        n_sqp_steps=sqp_steps,
        n_admm_steps=admm_steps,
        lane_penalty_weight=0.0,
        planned_centerline_penalty_weight=0.0,
        road_penalty_weight=0.1,
        planned_road_penalty_weight=0.5,
        n_ctrl_points=5,
        spline_degree=2,
        horizon_time=horizon,
        max_route_samples=16,
        planner_action_mode="target_point",
        mgf_terminal_projection=True,
        mgf_projection_reserve=projection_reserve,
        mgf_projection_kinematic_reserve=projection_kinematic_reserve,
        mgf_margin_softmin_kappa=5000.0,
        mgf_road_tightening=1.10,
        explicit_obstacle_observation=explicit_obstacle_observation,
        verbose=False,
    )
    return HybridCBFMPCEnv(
        controller,
        safety=safety,
        recovery_action=(1.0, 0.0),
        max_episode_steps=max_episode_steps,
    )


__all__ = [
    "HYBRID_INFO_KEY",
    "HybridCBFMPCEnv",
    "HybridMode",
    "HybridSafetyConfig",
    "make_safe100_base_env",
    "make_safe_sac_metadrive_env",
]
