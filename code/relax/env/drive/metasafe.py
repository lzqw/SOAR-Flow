"""Fixed-seed official SafeMetaDrive task with a hard CBF--MPC controller.

Unlike :mod:`relax.env.drive.lane_change`, this module does not replace the
procedural map or inject hand-authored parked vehicles.  It keeps the official
``SafeMetaDriveEnv`` map, accident objects and traffic generation, while
forcing every reset to the same scenario seed.  The small obstacle adapter is
the only bridge required by ``FastMPCControlWrapper``: MetaDrive objects are
converted to conservative moving circles at every controller step.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

from relax.utils.action_semantics import (
    FIXED_SPEED_ACTION_SEMANTICS,
    SPEED_COUPLED_ACTION_SEMANTICS,
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)


def conservative_body_circle_cover(
    center_xy: np.ndarray,
    heading: float,
    length: float,
    width: float,
    *,
    circle_count: int = 2,
    minimum_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return circles whose union covers an oriented rectangular body."""

    center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
    length = max(abs(float(length)), 1e-3)
    width = max(abs(float(width)), 1e-3)
    circle_count = max(int(circle_count), 1)
    cell_length = length / circle_count
    offsets = np.linspace(
        -0.5 * length + 0.5 * cell_length,
        0.5 * length - 0.5 * cell_length,
        circle_count,
        dtype=np.float32,
    )
    direction = np.asarray(
        [np.cos(float(heading)), np.sin(float(heading))],
        dtype=np.float32,
    )
    positions = center_xy[None, :] + offsets[:, None] * direction[None, :]
    radius = max(
        float(np.hypot(0.5 * cell_length, 0.5 * width)),
        max(float(minimum_radius), 0.0),
    )
    radii = np.full((circle_count,), radius, dtype=np.float32)
    return positions.astype(np.float32), radii


def _finite_vector(value, *, size: int, fallback: float = 0.0) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)[:size]
    except Exception:
        vector = np.zeros((0,), dtype=np.float32)
    if vector.size != size or not np.all(np.isfinite(vector)):
        return np.full((size,), fallback, dtype=np.float32)
    return vector


def _positive_dimension(obj: Any, name: str, fallback: float) -> float:
    value = getattr(obj, name.upper(), None)
    if value is None:
        value = getattr(obj, name.lower(), fallback)
    try:
        value = float(value)
    except Exception:
        value = float(fallback)
    return value if np.isfinite(value) and value > 0.0 else float(fallback)


class FixedSeedMetaSafeEnv(SafeMetaDriveEnv):
    """Official ``SafeMetaDriveEnv`` pinned to one procedural scenario."""

    def default_config(self):
        cfg = super().default_config()
        cfg.update(
            {
                "metasafe_fixed_seed": 0,
                "metasafe_body_circle_count": 2,
                "metasafe_small_object_circle_count": 1,
                "metasafe_action_semantics_version": (
                    FIXED_SPEED_ACTION_SEMANTICS
                ),
                "metasafe_lookahead_distance": 10.0,
                "metasafe_max_command_speed": 4.8,
                "metasafe_speed_mapping": "fixed_internal_target_velocity",
                "metasafe_mpc_horizon": 2.0,
            },
            allow_add_new_key=True,
        )
        cfg.update(
            {
                "start_seed": 0,
                "num_scenarios": 1,
                "random_traffic": False,
                "cost_to_reward": False,
                "log_level": logging.ERROR,
            }
        )
        return cfg

    @property
    def fixed_seed(self) -> int:
        return int(self.config.get("metasafe_fixed_seed", 0))

    def reset(self, seed=None, options=None):
        """Ignore caller seed changes so map, traffic and accidents stay fixed."""

        del seed, options
        fixed_seed = self.fixed_seed
        out = super().reset(seed=fixed_seed)
        observation, info = (
            out if isinstance(out, tuple) and len(out) == 2 else (out, {})
        )
        signature, scene = self.scene_signature()
        info = dict(info)
        info.update(
            {
                "metasafe_fixed_seed": fixed_seed,
                "metasafe_scene_signature": signature,
                "metasafe_obstacle_objects": len(scene["objects"]),
                "metasafe_map_blocks": scene["map_blocks"],
                "metasafe_scene": scene,
                "action_semantics_version": str(
                    self.config.get(
                        "metasafe_action_semantics_version",
                        FIXED_SPEED_ACTION_SEMANTICS,
                    )
                ),
                "lookahead_distance": float(
                    self.config.get("metasafe_lookahead_distance", 10.0)
                ),
                "max_command_speed": float(
                    self.config.get("metasafe_max_command_speed", 4.8)
                ),
                "speed_mapping": str(
                    self.config.get(
                        "metasafe_speed_mapping",
                        "fixed_internal_target_velocity",
                    )
                ),
                "mpc_horizon": float(
                    self.config.get("metasafe_mpc_horizon", 2.0)
                ),
                "safe_env_backend": (
                    "metadrive.envs.safe_metadrive_env.SafeMetaDriveEnv"
                ),
            }
        )
        return observation, info

    def _iter_obstacle_objects(self) -> Iterable[Any]:
        """Yield each official accident/traffic object exactly once."""

        try:
            ego = self.vehicle
        except Exception:
            ego = None
        objects: list[Any] = []
        try:
            objects.extend(self.engine.object_manager.spawned_objects.values())
        except Exception:
            pass
        try:
            objects.extend(self.engine.traffic_manager.spawned_objects.values())
        except Exception:
            pass
        try:
            objects.extend(self.engine.traffic_manager.traffic_vehicles)
        except Exception:
            pass

        seen: set[int] = set()
        for obj in objects:
            identity = id(obj)
            if obj is None or identity == id(ego) or identity in seen:
                continue
            seen.add(identity)
            position = _finite_vector(
                getattr(obj, "position", None), size=2, fallback=np.nan
            )
            if np.all(np.isfinite(position)):
                yield obj

    def _object_circle_count(self, obj: Any, length: float, width: float) -> int:
        minimum_radius = float(getattr(obj, "RADIUS", 0.0) or 0.0)
        if minimum_radius > 0.0 or length <= max(1.25 * width, 0.75):
            return max(
                int(self.config.get("metasafe_small_object_circle_count", 1)),
                1,
            )
        return max(int(self.config.get("metasafe_body_circle_count", 2)), 1)

    def get_obstacle_info(self):
        """Return current official traffic and accident geometry for CBF--MPC."""

        position_parts: list[np.ndarray] = []
        radius_parts: list[np.ndarray] = []
        velocity_parts: list[np.ndarray] = []
        for obj in self._iter_obstacle_objects():
            position = _finite_vector(getattr(obj, "position", None), size=2)
            velocity = _finite_vector(getattr(obj, "velocity", np.zeros(2)), size=2)
            heading = float(getattr(obj, "heading_theta", 0.0))
            if not np.isfinite(heading):
                heading = 0.0
            length = _positive_dimension(obj, "length", 0.4)
            width = _positive_dimension(obj, "width", 0.4)
            minimum_radius = float(getattr(obj, "RADIUS", 0.0) or 0.0)
            count = self._object_circle_count(obj, length, width)
            positions, radii = conservative_body_circle_cover(
                position,
                heading,
                length,
                width,
                circle_count=count,
                minimum_radius=minimum_radius,
            )
            position_parts.append(positions)
            radius_parts.append(radii)
            velocity_parts.append(
                np.repeat(velocity[None, :], positions.shape[0], axis=0)
            )

        if not position_parts:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )
        return (
            np.concatenate(position_parts, axis=0).astype(np.float32),
            np.concatenate(radius_parts, axis=0).astype(np.float32),
            np.concatenate(velocity_parts, axis=0).astype(np.float32),
        )

    def scene_signature(self) -> tuple[str, dict[str, Any]]:
        """Hash map and initial object geometry, excluding process-local names."""

        rows = []
        for obj in self._iter_obstacle_objects():
            heading = float(getattr(obj, "heading_theta", 0.0))
            if not np.isfinite(heading):
                heading = 0.0
            rows.append(
                {
                    "type": type(obj).__name__,
                    "position": np.round(
                        _finite_vector(getattr(obj, "position", None), size=2),
                        5,
                    ).tolist(),
                    "heading": round(heading, 5),
                    "length": round(_positive_dimension(obj, "length", 0.4), 5),
                    "width": round(_positive_dimension(obj, "width", 0.4), 5),
                }
            )
        rows.sort(
            key=lambda row: (
                row["type"],
                row["position"][0],
                row["position"][1],
                row["heading"],
            )
        )
        try:
            map_blocks = [
                str(getattr(block, "ID", type(block).__name__))
                for block in self.current_map.blocks
            ]
        except Exception:
            map_blocks = []
        scene = {
            "fixed_seed": self.fixed_seed,
            "map_blocks": map_blocks,
            "objects": rows,
        }
        encoded = json.dumps(scene, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest(), scene


class PostResetFrameWarmup(gym.Wrapper):
    """Advance fixed initialization frames before exposing an observation.

    MetaDrive's observation returned directly by ``reset()`` can contain
    initialization-time sensor/cache state.  The Actor must therefore never
    consume that observation.  This wrapper executes a fixed target-point
    command for the requested number of frames and returns only the resulting
    observation.  The fixed transition is deliberately outside replay and is
    not a policy decision.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        warmup_steps: int = 1,
        warmup_action: tuple[float, float] = (1.0, 0.0),
    ):
        super().__init__(env)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.warmup_action = np.clip(
            np.asarray(warmup_action, dtype=np.float32).reshape(2),
            -1.0,
            1.0,
        )

    def reset(self, **kwargs):
        observation, reset_info = self.env.reset(**kwargs)
        reset_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        observation = reset_observation
        warmup_reward = 0.0
        warmup_cost = 0.0
        projection_used = False
        last_step_info: dict[str, Any] = {}
        for frame in range(self.warmup_steps):
            certificate_fn = getattr(
                self.env,
                "certify_unprotected_target_action",
                self.env.certify_target_action,
            )
            certificate = certificate_fn(self.warmup_action)
            (
                observation,
                reward,
                terminated,
                truncated,
                step_info,
            ) = self.env.execute_certified_target(certificate)
            last_step_info = dict(step_info)
            warmup_reward += float(reward)
            warmup_cost += max(float(step_info.get("cost", 0.0)), 0.0)
            mgf = step_info.get("mgf", {})
            projection_used = bool(
                projection_used
                or mgf.get(
                    "projection_used",
                    float(mgf.get("projection_norm", 0.0)) > 1e-6,
                )
            )
            if terminated or truncated:
                raise RuntimeError(
                    "MetaSafe terminated during post-reset frame warm-up "
                    f"at frame {frame + 1}/{self.warmup_steps}"
                )

        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        info = dict(reset_info)
        info.update(
            {
                "post_reset_warmup_steps": self.warmup_steps,
                "post_reset_warmup_action": self.warmup_action.tolist(),
                "post_reset_warmup_reward": warmup_reward,
                "post_reset_warmup_cost": warmup_cost,
                "post_reset_warmup_projection_used": projection_used,
                "post_reset_observation_linf_delta": float(
                    np.max(np.abs(observation - reset_observation))
                ),
                "post_reset_warmup_collision": bool(
                    last_step_info.get("crash_vehicle", False)
                    or last_step_info.get("crash_object", False)
                ),
                "post_reset_warmup_road_departure": bool(
                    last_step_info.get("out_of_road", False)
                    or last_step_info.get("road_departure", False)
                ),
            }
        )
        return observation, info

    # Gymnasium wrappers do not consistently proxy arbitrary custom methods
    # across releases, so expose the certificate split explicitly.
    def certify_target_action(self, action):
        return self.env.certify_target_action(action)

    def certify_unprotected_target_action(self, action):
        return self.env.certify_unprotected_target_action(action)

    def certify_ordinary_target_action(self, action, **kwargs):
        return self.env.certify_ordinary_target_action(action, **kwargs)

    def certify_independent_actor_target(self, action, **kwargs):
        return self.env.certify_independent_actor_target(action, **kwargs)

    def derive_inverse_mpc_teacher(self, safe_solved, **kwargs):
        return self.env.derive_inverse_mpc_teacher(safe_solved, **kwargs)

    def plan_safe_recovery_trajectory(self, **kwargs):
        return self.env.plan_safe_recovery_trajectory(**kwargs)

    def reset_recovery_bypass_preference(self) -> None:
        self.env.reset_recovery_bypass_preference()

    @property
    def projection_reserves(self) -> dict[str, float]:
        return dict(self.env.projection_reserves)

    def set_projection_reserves(
        self, *, spatial: float, kinematic: float
    ) -> dict[str, float]:
        return self.env.set_projection_reserves(
            spatial=spatial,
            kinematic=kinematic,
        )

    def execute_certified_target(self, certificate):
        return self.env.execute_certified_target(certificate)


def make_fixed_metasafe_env(
    *,
    fixed_seed: int = 0,
    use_render: bool = False,
    action_semantics_version: str = FIXED_SPEED_ACTION_SEMANTICS,
    lookahead_distance: float = 10.0,
    max_command_speed: float = 4.8,
    speed_mapping: str = "fixed_internal_target_velocity",
    mpc_horizon: float = 2.0,
    config_overrides: dict[str, Any] | None = None,
) -> FixedSeedMetaSafeEnv:
    """Build the unmodified official MetaSafe task at one fixed seed."""

    config: dict[str, Any] = {
        "metasafe_fixed_seed": int(fixed_seed),
        "start_seed": int(fixed_seed),
        "num_scenarios": 1,
        "random_traffic": False,
        "use_render": bool(use_render),
        "window_size": (1280, 800),
        "show_fps": False,
        "show_logo": False,
        "show_mouse": bool(use_render),
        "log_level": logging.ERROR,
        "cost_to_reward": False,
        "metasafe_action_semantics_version": str(action_semantics_version),
        "metasafe_lookahead_distance": float(lookahead_distance),
        "metasafe_max_command_speed": float(max_command_speed),
        "metasafe_speed_mapping": str(speed_mapping),
        "metasafe_mpc_horizon": float(mpc_horizon),
    }
    if config_overrides:
        config.update(config_overrides)
    # The fixed-seed invariants take precedence over optional presentation or
    # simulator overrides supplied by a caller.
    config.update(
        {
            "metasafe_fixed_seed": int(fixed_seed),
            "start_seed": int(fixed_seed),
            "num_scenarios": 1,
            "random_traffic": False,
            "cost_to_reward": False,
        }
    )
    return FixedSeedMetaSafeEnv(config)


def make_metasafe_cbf_mpc_env(
    *,
    fixed_seed: int = 0,
    use_render: bool = False,
    lookahead: float = 10.0,
    lateral_scale: float = 3.5,
    target_velocity: float = 4.8,
    action_semantics_version: str = FIXED_SPEED_ACTION_SEMANTICS,
    min_command_speed: float = 1.0,
    max_command_speed: float | None = None,
    horizon: float = 2.0,
    tracking_weight: float = 6.0,
    velocity_weight: float = 30.0,
    road_weight: float = 100.0,
    road_buffer: float = 1.5,
    max_steer_rate: float = 0.1,
    clearance: float = 0.10,
    cbf_gamma: float = 3.0,
    projection_reserve: float = 0.10,
    projection_kinematic_reserve: float = 0.02,
    sqp_steps: int = 24,
    admm_steps: int = 240,
    max_obstacles: int = 32,
    cbf_enable: bool = True,
    terminal_projection: bool = True,
    post_reset_warmup_steps: int = 1,
    config_overrides: dict[str, Any] | None = None,
) -> gym.Env:
    """Build a Flat NMPC controller for fixed MetaSafe.

    ``cbf_enable=True, terminal_projection=True`` is the protected training
    and data-collection path.  Setting both flags to false is the ordinary
    independence test: lower-level NMPC remains, while obstacle CBF and the
    target projection are prevented from modifying the Actor proposal.
    """

    from gymnasium.wrappers import FlattenObservation

    from relax.env.drive.lane_change_mpc import FastMPCControlWrapper

    command_speed = float(
        target_velocity if max_command_speed is None else max_command_speed
    )
    speed_mapping = (
        "state_reachable_speed_linear_v1"
        if action_semantics_version == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
        else "v=max_command_speed*(action_x+1)/2"
        if action_semantics_version == SPEED_COUPLED_ACTION_SEMANTICS
        else "fixed_internal_target_velocity"
    )
    base = make_fixed_metasafe_env(
        fixed_seed=fixed_seed,
        use_render=use_render,
        action_semantics_version=action_semantics_version,
        lookahead_distance=lookahead,
        max_command_speed=command_speed,
        speed_mapping=speed_mapping,
        mpc_horizon=horizon,
        config_overrides=config_overrides,
    )
    controlled = FastMPCControlWrapper(
        FlattenObservation(base),
        position_scale=lookahead,
        target_lateral_scale=lateral_scale,
        wheelbase=2.46894,
        max_velocity=8.0,
        max_steering=0.6981317008,
        max_steer_rate=max_steer_rate,
        target_velocity=target_velocity,
        w_tracking=tracking_weight,
        w_velocity=velocity_weight,
        w_risk=0.0,
        w_cbf=1000.0,
        w_road=road_weight,
        cbf_solve_weight_scale=1.0,
        road_solve_weight_scale=1.0,
        risk_buffer=1.2,
        risk_temperature=2.0,
        road_buffer=road_buffer,
        road_temperature=0.5,
        cbf_enable=bool(cbf_enable),
        d_clearance=clearance,
        cbf_gamma=cbf_gamma,
        max_obstacles=max_obstacles,
        n_sqp_steps=sqp_steps,
        n_admm_steps=admm_steps,
        lane_penalty_weight=0.0,
        planned_centerline_penalty_weight=0.0,
        road_penalty_weight=0.1,
        planned_road_penalty_weight=0.5,
        n_ctrl_points=5,
        spline_degree=2,
        horizon_time=horizon,
        max_route_samples=24,
        planner_action_mode="target_point",
        action_semantics_version=action_semantics_version,
        min_command_speed=min_command_speed,
        max_command_speed=command_speed,
        mgf_terminal_projection=bool(terminal_projection),
        mgf_projection_reserve=projection_reserve,
        mgf_projection_kinematic_reserve=projection_kinematic_reserve,
        mgf_margin_softmin_kappa=5000.0,
        mgf_road_tightening=1.10,
        explicit_obstacle_observation=False,
        verbose=False,
    )
    return PostResetFrameWarmup(
        controlled,
        warmup_steps=post_reset_warmup_steps,
        warmup_action=(1.0, 0.0),
    )


__all__ = [
    "FixedSeedMetaSafeEnv",
    "FIXED_SPEED_ACTION_SEMANTICS",
    "PostResetFrameWarmup",
    "SPEED_COUPLED_ACTION_SEMANTICS",
    "TARGET_SPEED_LATERAL_ACTION_SEMANTICS",
    "conservative_body_circle_cover",
    "make_fixed_metasafe_env",
    "make_metasafe_cbf_mpc_env",
]
