"""Collect MetaDrive's bundled PPO expert for high-level BC initialization.

The bundled expert emits native steering/throttle commands, whereas Safe100's
Actor emits a two-dimensional local target.  We therefore execute PPO in the
unwrapped MetaDrive task and convert its *future realized trajectory* into a
causal training pair ``(observation_t, local_target_t)``.  Those pairs are BC
labels only.  PPO transitions are explicitly marked action-incompatible with
the high-level SAC critic; a separate high-level rollout is required before
offline critic pretraining.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from relax.env.drive.safe_sac import make_safe100_base_env
from relax.env.drive.safe_scenarios import (
    DEFAULT_OFFLINE_SCENARIOS_TEXT,
    ScenarioSpec,
    parse_scenarios,
    scenario_to_dict,
)
from relax.env.drive.observation_features import (
    encode_explicit_obstacle_features,
    inject_explicit_obstacle_features,
    world_obstacles_to_ego,
)


@dataclass(frozen=True)
class PPODatasetConfig:
    scenarios: tuple[ScenarioSpec, ...] = parse_scenarios(
        DEFAULT_OFFLINE_SCENARIOS_TEXT
    )
    episodes_per_scenario: int = 3
    max_episode_steps: int = 400
    lookahead_m: float = 10.0
    lateral_scale_m: float = 3.7
    target_path_distance_m: float = 8.0
    min_forward_progress_m: float = 2.0
    deterministic: bool = True
    require_success: bool = True
    seed: int = 0
    explicit_obstacle_observation: bool = True


def trajectory_to_high_level_targets(
    positions: np.ndarray,
    headings: np.ndarray,
    *,
    lookahead_m: float,
    lateral_scale_m: float,
    target_path_distance_m: float,
    min_forward_progress_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a realized global trajectory to normalized local targets.

    ``positions`` has length ``T + 1`` and ``headings`` has length ``T``.
    The target for step ``t`` is the first future point whose accumulated path
    distance reaches ``target_path_distance_m``.  Tail samples without enough
    forward progress are retained in the dataset but marked invalid.
    """

    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (T + 1, 2)")
    steps = headings.shape[0]
    if positions.shape[0] != steps + 1:
        raise ValueError("positions must contain one more item than headings")
    if lookahead_m <= 0.0 or lateral_scale_m <= 0.0:
        raise ValueError("target scales must be positive")

    segment_length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    targets = np.zeros((steps, 2), dtype=np.float32)
    valid = np.zeros((steps,), dtype=bool)
    future_index = np.full((steps,), -1, dtype=np.int32)

    for step in range(steps):
        distance = 0.0
        selected = steps
        for candidate in range(step + 1, steps + 1):
            distance += float(segment_length[candidate - 1])
            selected = candidate
            if distance >= target_path_distance_m:
                break

        delta = positions[selected] - positions[step]
        heading = float(headings[step])
        cosine, sine = math.cos(heading), math.sin(heading)
        forward = cosine * float(delta[0]) + sine * float(delta[1])
        lateral = -sine * float(delta[0]) + cosine * float(delta[1])
        targets[step] = np.clip(
            np.asarray(
                [
                    2.0 * forward / lookahead_m - 1.0,
                    lateral / lateral_scale_m,
                ],
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )
        valid[step] = bool(
            np.isfinite(forward)
            and np.isfinite(lateral)
            and forward >= min_forward_progress_m
        )
        future_index[step] = selected
    return targets, valid, future_index


def _with_explicit_obstacle_features(
    env, observation: np.ndarray, config: PPODatasetConfig
) -> np.ndarray:
    if not config.explicit_obstacle_observation:
        return np.asarray(observation, dtype=np.float32).reshape(-1).copy()
    world_positions, _radii = env.get_obstacle_info()
    obstacle_ego = world_obstacles_to_ego(
        world_positions,
        ego_position=np.asarray(env.agent.position, dtype=np.float32),
        ego_heading=float(env.agent.heading_theta),
    )
    features = encode_explicit_obstacle_features(
        obstacle_ego,
        target_lateral_scale=config.lateral_scale_m,
        position_scale=config.lookahead_m,
    )
    return inject_explicit_obstacle_features(observation, features)


def _episode_arrays(
    *,
    scenario: ScenarioSpec,
    scenario_index: int,
    episode_index: int,
    config: PPODatasetConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    # Import the NumPy implementation explicitly.  Importing the package-level
    # alias prefers Torch when installed and can initialize CUDA unnecessarily.
    from metadrive.examples.ppo_expert.numpy_expert import expert

    env = make_safe100_base_env(
        use_render=False,
        obstacle_specs=scenario,
    )
    seed = int(config.seed + 1000 * scenario_index + episode_index)
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    low_level_actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    positions: list[np.ndarray] = []
    headings: list[float] = []
    infos: list[dict] = []
    try:
        observation, _ = env.reset(seed=seed)
        observation = _with_explicit_obstacle_features(
            env, observation, config
        )
        positions.append(np.asarray(env.agent.position, dtype=np.float64).copy())
        for _ in range(config.max_episode_steps):
            observations.append(
                np.asarray(observation, dtype=np.float32).reshape(-1).copy()
            )
            headings.append(float(env.agent.heading_theta))
            action = np.asarray(
                expert(env.agent, deterministic=config.deterministic),
                dtype=np.float32,
            ).reshape(2)
            next_observation, reward, terminated, truncated, info = env.step(
                action
            )
            next_observation = _with_explicit_obstacle_features(
                env, next_observation, config
            )
            next_observations.append(
                np.asarray(next_observation, dtype=np.float32)
                .reshape(-1)
                .copy()
            )
            low_level_actions.append(action.copy())
            rewards.append(float(reward))
            dones.append(bool(terminated or truncated))
            positions.append(
                np.asarray(env.agent.position, dtype=np.float64).copy()
            )
            infos.append(dict(info))
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()

    targets, valid, future_index = trajectory_to_high_level_targets(
        np.asarray(positions),
        np.asarray(headings),
        lookahead_m=config.lookahead_m,
        lateral_scale_m=config.lateral_scale_m,
        target_path_distance_m=config.target_path_distance_m,
        min_forward_progress_m=config.min_forward_progress_m,
    )
    last_info = infos[-1] if infos else {}
    success = bool(last_info.get("arrive_dest", False))
    collision = bool(
        last_info.get("crash_vehicle", False)
        or last_info.get("crash_object", False)
    )
    road_departure = bool(
        last_info.get("out_of_road", False)
        or last_info.get("road_departure", False)
    )
    environment_cost = float(
        np.sum([max(float(info.get("cost", 0.0)), 0.0) for info in infos])
    )
    cost_events = int(
        np.sum([float(info.get("cost", 0.0)) > 0.0 for info in infos])
    )
    if config.require_success and not success:
        valid[:] = False
    if collision or road_departure or environment_cost > 0.0:
        valid[:] = False

    length = len(observations)
    arrays = {
        "obs": np.asarray(observations, dtype=np.float32),
        "next_obs": np.asarray(next_observations, dtype=np.float32),
        "teacher_action": targets,
        "expert_low_level_action": np.asarray(
            low_level_actions, dtype=np.float32
        ),
        "reward": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(dones, dtype=bool),
        "bc_valid": valid,
        "future_index": future_index,
        "scenario_index": np.full(length, scenario_index, dtype=np.int32),
        "episode_index": np.full(length, episode_index, dtype=np.int32),
        "step_index": np.arange(length, dtype=np.int32),
        "position": np.asarray(positions[:-1], dtype=np.float32),
        "heading": np.asarray(headings, dtype=np.float32),
        # PPO acts in steering/throttle space.  This explicit flag prevents
        # accidental use of its transitions as high-level SAC transitions.
        "critic_action_compatible": np.zeros(length, dtype=bool),
    }
    summary: dict[str, object] = {
        "scenario": scenario_to_dict(scenario),
        "scenario_index": scenario_index,
        "episode_index": episode_index,
        "seed": seed,
        "steps": length,
        "return": float(np.sum(rewards)),
        "success": success,
        "collision": collision,
        "road_departure": road_departure,
        "environment_cost": environment_cost,
        "cost_events": cost_events,
        "safe_env_backend": (
            "metadrive.envs.safe_metadrive_env.SafeMetaDriveEnv"
        ),
        "valid_bc_samples": int(np.sum(valid)),
    }
    return arrays, summary


def _concatenate(rows: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("no PPO episodes were collected")
    keys = rows[0].keys()
    return {
        key: np.concatenate([row[key] for row in rows], axis=0)
        for key in keys
    }


def collect_ppo_dataset(
    output: Path,
    config: PPODatasetConfig = PPODatasetConfig(),
) -> dict[str, object]:
    """Collect and atomically save a compressed PPO-to-BC dataset."""

    if config.episodes_per_scenario < 1:
        raise ValueError("episodes_per_scenario must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict[str, np.ndarray]] = []
    episodes: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(config.scenarios):
        for episode_index in range(config.episodes_per_scenario):
            arrays, summary = _episode_arrays(
                scenario=scenario,
                scenario_index=scenario_index,
                episode_index=episode_index,
                config=config,
            )
            rows.append(arrays)
            episodes.append(summary)

    dataset = _concatenate(rows)
    np.savez_compressed(output, **dataset)
    valid_samples = int(np.sum(dataset["bc_valid"]))
    metadata: dict[str, object] = {
        "format_version": 3,
        "expert": "metadrive.examples.ppo_expert.numpy_expert.expert",
        "action_semantics": {
            "expert_low_level_action": "MetaDrive steering/throttle",
            "teacher_action": "future-trajectory-derived Safe100 local target",
            "critic_action_compatible": False,
        },
        "config": asdict(config),
        "episodes": episodes,
        "total_samples": int(dataset["obs"].shape[0]),
        "valid_bc_samples": valid_samples,
        "success_rate": float(
            np.mean([bool(row["success"]) for row in episodes])
        ),
        "zero_cost_rate": float(
            np.mean([float(row["environment_cost"]) == 0.0 for row in episodes])
        ),
        "metadrive_base_env": (
            "metadrive.envs.safe_metadrive_env.SafeMetaDriveEnv"
        ),
        "explicit_obstacle_observation": {
            "enabled": bool(config.explicit_obstacle_observation),
            "storage": (
                "last six base-observation slots"
                if config.explicit_obstacle_observation
                else "native MetaDrive observation only"
            ),
            "semantics": (
                "nearest left/center/right distance plus free-lane flags"
                if config.explicit_obstacle_observation
                else "19 state/navigation values plus refreshed 240-ray lidar"
            ),
        },
        "wall_time_seconds": time.time() - started,
        "dataset": str(output),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    if valid_samples == 0:
        raise RuntimeError("PPO collection produced no valid BC samples")
    return metadata
