#!/usr/bin/env python3
"""Fresh-process deterministic evaluator for Guardian-DSAC V2.3/V2.4.x."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.utils.action_semantics import (  # noqa: E402
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)


UNCERTIFIED_PROPOSAL_HANDLING = "safe_termination_without_execution_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-seeds", default="0,1")
    parser.add_argument("--modes", default="hybrid,ordinary")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--policy",
        choices=("learner", "hybrid_safe", "deployable", "teacher"),
        default="learner",
    )
    parser.add_argument(
        "--trajectory-output-dir",
        type=Path,
        default=None,
        help=(
            "Record action-before-step Ordinary trajectories for certified "
            "Safe-Memory import. Hybrid trajectories are never recorded."
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode",
        choices=("hybrid", "ordinary"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-trajectory-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--episode-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.worker and (args.worker_mode is None or args.worker_output is None):
        parser.error("worker mode/output are required")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _run_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    for candidate in (checkpoint.parent, checkpoint.parent.parent):
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("config.json is not adjacent to the checkpoint")


def _load_algorithm(checkpoint: Path, config: dict[str, Any]):
    import jax

    from relax.algorithm.guardian_dsac_v2_3 import (
        GuardianDSACV23,
        GuardianDSACV23Config,
    )
    from relax.network.safe_sac_jax import create_guardian_dsac_net

    hidden_sizes = tuple(int(value) for value in config["hidden_sizes"])
    network, params = create_guardian_dsac_net(
        jax.random.PRNGKey(int(config["seed"])),
        critic_obs_dim=int(config["observation_dim"]),
        actor_obs_dim=int(config["actor_observation_dim"]),
        act_dim=int(config["action_dim"]),
        hidden_sizes=hidden_sizes,
        init_alpha=float(config["init_alpha"]),
        sigma_min=float(config["algorithm_config"]["sigma_min"]),
    )
    algorithm = GuardianDSACV23(
        network,
        params,
        config=GuardianDSACV23Config(**config["algorithm_config"]),
        action_semantics_version=config["action_semantics_version"],
        scene_seeds=config["scene_seeds"],
        scene_signatures=config["scene_signatures"],
        dataset_sha256=config["dataset_sha256"],
    )
    with checkpoint.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("format") == GuardianDSACV23.FORMAT:
        algorithm._load_checkpoint_payload(payload)
        return algorithm

    from relax.algorithm.guardian_dsac_flow_v2_4 import (
        GuardianDSACFlowV24,
        GuardianDSACFlowV24Config,
    )
    from relax.network.flow_policy import create_one_step_flow_policy

    from relax.algorithm.guardian_dsac_flow_v2_4_1 import (
        GuardianDSACFlowV241,
        GuardianDSACFlowV241Config,
    )
    from relax.algorithm.guardian_dsac_flow_v2_4_2 import (
        GuardianDSACFlowV242,
        GuardianDSACFlowV242Config,
    )
    from relax.algorithm.guardian_dsac_flow_v2_4_3 import (
        GuardianDSACFlowV243,
        GuardianDSACFlowV243Config,
    )
    from relax.algorithm.guardian_dsac_flow_paper import (
        GuardianDSACFlowPaper,
        GuardianDSACFlowPaperConfig,
    )

    flow_format = payload.get("format")
    if flow_format not in (
        GuardianDSACFlowV24.FORMAT,
        GuardianDSACFlowV241.FORMAT,
        GuardianDSACFlowV242.FORMAT,
        GuardianDSACFlowV243.FORMAT,
        GuardianDSACFlowPaper.FORMAT,
    ):
        raise ValueError("unsupported Guardian-DSAC checkpoint format")
    algorithm._load_checkpoint_payload(payload["base_v2_3"])
    flow_policy, flow_params = create_one_step_flow_policy(
        jax.random.PRNGKey(int(config["seed"]) + 24_000),
        actor_obs_dim=int(payload["actor_observation_dim"]),
        act_dim=int(payload["action_dim"]),
        hidden_sizes=tuple(int(value) for value in payload["hidden_sizes"]),
        integration_steps=int(payload["integration_steps"]),
    )
    if flow_format == GuardianDSACFlowV24.FORMAT:
        flow = GuardianDSACFlowV24(
            algorithm,
            flow_policy,
            flow_params,
            config=GuardianDSACFlowV24Config(**payload["flow_config"]),
        )
    elif flow_format == GuardianDSACFlowV241.FORMAT:
        flow = GuardianDSACFlowV241(
            algorithm,
            flow_policy,
            flow_params,
            config=GuardianDSACFlowV241Config(**payload["flow_config"]),
        )
    elif flow_format == GuardianDSACFlowV242.FORMAT:
        flow = GuardianDSACFlowV242(
            algorithm,
            flow_policy,
            flow_params,
            config=GuardianDSACFlowV242Config(**payload["flow_config"]),
        )
    elif flow_format == GuardianDSACFlowV243.FORMAT:
        flow = GuardianDSACFlowV243(
            algorithm,
            flow_policy,
            flow_params,
            config=GuardianDSACFlowV243Config(**payload["flow_config"]),
        )
    else:
        flow = GuardianDSACFlowPaper(
            algorithm,
            flow_policy,
            flow_params,
            config=GuardianDSACFlowPaperConfig(**payload["flow_config"]),
        )
    flow._load_checkpoint_payload(payload, load_base=False)
    return flow


def _make_environment(
    config: dict[str, Any],
    *,
    seed: int,
    mode: str,
    config_overrides: dict[str, Any] | None = None,
):
    from relax.env.drive.metasafe import make_metasafe_cbf_mpc_env

    protected = mode == "hybrid"
    env = config["environment"]
    return make_metasafe_cbf_mpc_env(
        fixed_seed=seed,
        use_render=False,
        action_semantics_version=TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        min_command_speed=float(env["min_command_speed"]),
        max_command_speed=float(env["max_command_speed"]),
        lookahead=float(env["lookahead"]),
        lateral_scale=float(env["lateral_scale"]),
        target_velocity=float(env["target_velocity"]),
        horizon=float(env["horizon"]),
        tracking_weight=float(env["mpc_tracking_weight"]),
        velocity_weight=float(env["mpc_velocity_weight"]),
        road_weight=float(env["mpc_road_weight"]),
        road_buffer=float(env["mpc_road_buffer"]),
        max_steer_rate=float(env["mpc_max_steer_rate"]),
        clearance=float(env["clearance"]),
        cbf_gamma=float(env["cbf_gamma"]),
        projection_reserve=float(env["projection_reserve"]),
        projection_kinematic_reserve=float(env["projection_kinematic_reserve"]),
        sqp_steps=int(env["sqp_steps"]),
        admm_steps=int(env["admm_steps"]),
        max_obstacles=int(env["max_obstacles"]),
        cbf_enable=protected,
        terminal_projection=protected,
        post_reset_warmup_steps=int(env["post_reset_warmup_steps"]),
        config_overrides=config_overrides,
    )


def _policy_action(algorithm, policy: str, observation: np.ndarray) -> np.ndarray:
    if policy == "teacher":
        # The non-learning Guardian/MPC teacher requests maximum cruise on the
        # route centreline and delegates the certified trajectory to the same
        # CBF/NMPC Guardian used by every learning method.
        action = np.asarray([1.0, 0.0], dtype=np.float32)
    elif policy == "learner":
        action = algorithm.get_deterministic_action(observation)
    elif policy == "hybrid_safe":
        action = algorithm.get_deterministic_hybrid_safe_action(observation)
    else:
        action = algorithm.get_deterministic_deployable_action(observation)
    return np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)


def _verify_recorded_trajectory_actions(
    algorithm,
    observations: np.ndarray,
    recorded_actions: np.ndarray,
) -> dict[str, Any]:
    """Repeat the exact CPU deployment call and require bitwise identity."""

    observations = np.asarray(observations, dtype=np.float32)
    recorded_actions = np.asarray(recorded_actions, dtype=np.float32)
    if (
        observations.ndim != 2
        or recorded_actions.shape != (len(observations), 2)
        or len(observations) < 1
    ):
        raise RuntimeError("trajectory self-check received invalid arrays")
    repeated_actions = np.stack(
        [
            _policy_action(algorithm, "learner", observation)
            for observation in observations
        ],
        axis=0,
    )
    action_error = np.abs(repeated_actions - recorded_actions)
    exact = bool(np.array_equal(repeated_actions, recorded_actions))
    if not exact:
        raise RuntimeError(
            "trajectory actions failed same-worker bitwise z=0 self-check"
        )
    return {
        "action_generation_mode": "single_observation_live_z0",
        "action_self_check_mode": ("repeat_single_observation_live_z0_same_cpu_worker"),
        "action_self_check_exact": exact,
        "action_self_check_max_abs_error": float(np.max(action_error)),
        "action_self_check_mean_abs_error": float(np.mean(action_error)),
    }


def _minimum(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.min(values)) if values.size else float("nan")


def _optional_mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [
        float(row[name])
        for row in rows
        if row.get(name) is not None and np.isfinite(float(row[name]))
    ]
    return float(np.mean(values)) if values else None


def _worker(args) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    checkpoint = args.checkpoint.expanduser().resolve()
    config = json.loads(
        (_run_dir(checkpoint) / "config.json").read_text(encoding="utf-8")
    )
    algorithm = _load_algorithm(checkpoint, config)
    if args.policy != "teacher":
        algorithm.select_evaluation_checkpoint(args.policy)
    is_flow_actor = hasattr(algorithm, "flow_policy")
    seed = int(args.worker_seed)
    expected_signature = str(config["scene_signatures"][str(seed)])
    env = _make_environment(config, seed=seed, mode=args.worker_mode)
    try:
        observation, reset_info = env.reset(seed=seed)
        signature = str(reset_info.get("metasafe_scene_signature", ""))
        if signature != expected_signature:
            raise RuntimeError("fresh evaluation scene signature mismatch")
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        # Exclude one-time JIT compilation from steady-state deployment
        # latency while still compiling in this fresh worker process.
        _policy_action(algorithm, args.policy, observation)
        task_return = 0.0
        collision = bool(reset_info.get("post_reset_warmup_collision", False))
        road = bool(reset_info.get("post_reset_warmup_road_departure", False))
        cost = float(reset_info.get("post_reset_warmup_cost", 0.0))
        success = False
        solver = []
        margins = []
        interventions = 0
        q_means = []
        q_stds = []
        p_values = []
        speed_fractions = []
        emergency_stops = 0
        inference_latency_ms = []
        guardian_certification_latency_ms = []
        certified_control_latency_ms = []
        trajectory_observations = []
        trajectory_actions = []
        error = None
        steps = 0
        action_proposals = 0
        safe_termination = False
        safe_termination_reason = None
        uncertified_action_count = 0
        for step in range(int(config["environment"]["max_episode_steps"])):
            action_proposals += 1
            action_started = time.perf_counter()
            action = _policy_action(algorithm, args.policy, observation)
            inference_latency_ms.append(
                (time.perf_counter() - action_started) * 1_000.0
            )
            if args.policy != "teacher":
                mean_1, std_1, mean_2, std_2 = algorithm.task_q_distribution(
                    observation[None, :], action[None, :]
                )
                q_means.append(float(min(mean_1[0], mean_2[0])))
                q_stds.append(float(0.5 * (std_1[0] + std_2[0])))
                p_values.append(
                    float(
                        algorithm.proxy_value(observation[None, :], action[None, :])[0]
                    )
                )
            try:
                certified_control_started = time.perf_counter()
                certification_started = time.perf_counter()
                certificate = env.certify_target_action(action)
                guardian_certification_latency_ms.append(
                    (time.perf_counter() - certification_started) * 1_000.0
                )
                if bool(certificate.get("projection_failed", False)):
                    # The Guardian has proved that this proposal has no
                    # certified terminal projection.  This is a protocol-safe
                    # episode outcome, not a worker/infrastructure failure:
                    # never execute it and never manufacture an environment
                    # transition.  Count the refusal as an intervention so
                    # Hybrid dependence cannot be understated.
                    certified_control_latency_ms.append(
                        (time.perf_counter() - certified_control_started) * 1_000.0
                    )
                    hard_certificate = certificate.get(
                        "executed_hard_certificate", {}
                    )
                    solver.append(bool(hard_certificate.get("finite", False)))
                    interventions += 1
                    safe_termination = True
                    safe_termination_reason = "projection_failed"
                    uncertified_action_count = 1
                    success = False
                    break
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.execute_certified_target(certificate)
                certified_control_latency_ms.append(
                    (time.perf_counter() - certified_control_started) * 1_000.0
                )
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
                break
            steps = step + 1
            if args.worker_trajectory_output is not None:
                if args.worker_mode != "ordinary" or args.policy != "learner":
                    raise RuntimeError(
                        "trajectory capture is bound to learner Ordinary evaluation"
                    )
                trajectory_observations.append(observation.copy())
                trajectory_actions.append(action.copy())
            mgf = info.get("mgf", {})
            physical = info.get("mpc_control", {})
            task_return += float(info.get("task_reward", reward))
            cost += max(float(info.get("cost", 0.0)), 0.0)
            collision = bool(
                collision
                or info.get("crash_vehicle", False)
                or info.get("crash_object", False)
            )
            road = bool(
                road
                or info.get("out_of_road", False)
                or info.get("road_departure", False)
            )
            success = bool(
                success or info.get("arrive_dest", False) or info.get("success", False)
            )
            solver.append(bool(mgf.get("solver_finite", False)))
            margins.append(float(physical.get("physical_body_sat_margin", np.nan)))
            interventions += int(bool(mgf.get("actual_control_modified", False)))
            lower = float(mgf.get("reachable_speed_lower", np.nan))
            upper = float(mgf.get("reachable_speed_upper", np.nan))
            speed = float(mgf.get("executed_target_speed", np.nan))
            emergency = bool(mgf.get("emergency_stop", False))
            emergency_stops += int(emergency)
            if (
                all(np.isfinite(value) for value in (lower, upper, speed))
                and not emergency
            ):
                speed_fractions.append(
                    float(np.clip((speed - lower) / (upper - lower + 1e-6), 0.0, 1.0))
                )
            observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
            if terminated or truncated:
                break
        bins = (
            np.minimum((np.asarray(speed_fractions) * 4.0).astype(np.int32), 3)
            if speed_fractions
            else np.zeros((0,), dtype=np.int32)
        )
        trajectory = None
        if args.worker_trajectory_output is not None:
            trajectory_path = args.worker_trajectory_output.expanduser().resolve()
            observations = np.asarray(
                trajectory_observations, dtype=np.float32
            ).reshape(steps, -1)
            actions = np.asarray(trajectory_actions, dtype=np.float32).reshape(
                steps, -1
            )
            import jax

            from relax.algorithm.guardian_dsac_flow_v2_4_2 import (
                _parameter_sha256,
            )

            if not is_flow_actor or jax.default_backend() != "cpu":
                raise RuntimeError(
                    "Safe-Memory trajectory capture requires a CPU Flow worker"
                )
            action_self_check = _verify_recorded_trajectory_actions(
                algorithm,
                observations,
                actions,
            )
            actor_parameter_sha256 = _parameter_sha256(algorithm.state.params.one_step)
            metadata = {
                "format": "guardian_flow_v2_4_3_ordinary_trajectory_v1",
                "seed": seed,
                "mode": args.worker_mode,
                "episode_index": int(args.episode_index),
                "policy": args.policy,
                "scene_signature": signature,
                "checkpoint_sha256": _sha256(checkpoint),
                "actor_parameter_sha256": actor_parameter_sha256,
                "worker_backend": jax.default_backend(),
                "rows": steps,
                "observation_sha256": _array_sha256(observations),
                "action_sha256": _array_sha256(actions),
                **action_self_check,
            }
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trajectory_path,
                observations=observations,
                actions=actions,
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
            )
            trajectory = {
                "format": metadata["format"],
                "path": str(trajectory_path),
                "sha256": _sha256(trajectory_path),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "actor_parameter_sha256": actor_parameter_sha256,
                "worker_backend": metadata["worker_backend"],
                "rows": steps,
                "observation_shape": list(observations.shape),
                "action_shape": list(actions.shape),
                "observation_sha256": metadata["observation_sha256"],
                "action_sha256": metadata["action_sha256"],
                **action_self_check,
            }
        row = {
            "seed": seed,
            "mode": args.worker_mode,
            "episode_index": int(args.episode_index),
            "policy": args.policy,
            "scene_signature": signature,
            "steps": steps,
            "return": task_return,
            "success": success,
            "collision": collision,
            "road_departure": road,
            "environment_cost": cost,
            "action_proposals": action_proposals,
            "executed_environment_steps": steps,
            "intervention_rate": interventions / max(action_proposals, 1),
            "actual_intervention_rate": interventions / max(action_proposals, 1),
            "solver_finite_rate": float(np.mean(solver)) if solver else 0.0,
            # With a first-proposal refusal there is no actor-controlled
            # physical step whose SAT margin could be measured.  Zero is a
            # finite conservative sentinel and, together with the explicit
            # termination fields below, prevents accidental safe-success.
            "min_physical_body_sat_margin": (
                _minimum(margins) if margins else 0.0
            ),
            "q_mean": float(np.mean(q_means)) if q_means else None,
            "q_std": float(np.mean(q_stds)) if q_stds else None,
            "p_mean": float(np.mean(p_values)) if p_values else None,
            "speed_bin_counts": np.bincount(bins, minlength=4).astype(int).tolist(),
            "emergency_stop_rate": emergency_stops / max(steps, 1),
            "inference_latency_ms_mean": float(np.mean(inference_latency_ms)),
            "inference_latency_ms_p95": float(np.quantile(inference_latency_ms, 0.95)),
            "guardian_certification_latency_ms_mean": float(
                np.mean(guardian_certification_latency_ms)
            ),
            "guardian_certification_latency_ms_p95": float(
                np.quantile(guardian_certification_latency_ms, 0.95)
            ),
            "certified_control_latency_ms_mean": float(
                np.mean(certified_control_latency_ms)
            ),
            "certified_control_latency_ms_p95": float(
                np.quantile(certified_control_latency_ms, 0.95)
            ),
            "flow_latent_mode": (
                "zero" if is_flow_actor and args.policy != "teacher" else None
            ),
            "actor_forward_passes": 0 if args.policy == "teacher" else 1,
            "safe_termination": safe_termination,
            "safe_termination_reason": safe_termination_reason,
            "uncertified_action_count": uncertified_action_count,
            "terminated_without_executing_action": safe_termination,
            "error": error,
            "process_isolated": True,
            "worker_pid": os.getpid(),
            "trajectory": trajectory,
        }
    finally:
        env.close()
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps(row, indent=2, allow_nan=True), encoding="utf-8"
    )
    return 0


def _aggregate(mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(rows), 1)
    speed_bins = (
        np.sum(
            [np.asarray(row["speed_bin_counts"], dtype=np.int64) for row in rows],
            axis=0,
        )
        if rows
        else np.zeros((4,), dtype=np.int64)
    )
    return {
        "mode": mode,
        "episodes": len(rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / count,
        "collision_rate": sum(bool(row["collision"]) for row in rows) / count,
        "road_departure_rate": sum(bool(row["road_departure"]) for row in rows) / count,
        "total_environment_cost": sum(float(row["environment_cost"]) for row in rows),
        "mean_return": sum(float(row["return"]) for row in rows) / count,
        "actual_intervention_rate": sum(
            float(row["actual_intervention_rate"]) for row in rows
        )
        / count,
        "intervention_rate": sum(float(row["intervention_rate"]) for row in rows)
        / count,
        "solver_finite_rate": sum(float(row["solver_finite_rate"]) for row in rows)
        / count,
        "min_physical_body_sat_margin": _minimum(
            [row["min_physical_body_sat_margin"] for row in rows]
        ),
        "q_mean": _optional_mean(rows, "q_mean"),
        "q_std": _optional_mean(rows, "q_std"),
        "p_mean": _optional_mean(rows, "p_mean"),
        "speed_bin_counts": speed_bins.astype(int).tolist(),
        "emergency_stop_rate": sum(float(row["emergency_stop_rate"]) for row in rows)
        / count,
        "action_proposals": sum(int(row["action_proposals"]) for row in rows),
        "executed_environment_steps": sum(
            int(row["executed_environment_steps"]) for row in rows
        ),
        "safe_termination_count": sum(
            int(bool(row["safe_termination"])) for row in rows
        ),
        "safe_termination_rate": sum(
            int(bool(row["safe_termination"])) for row in rows
        )
        / count,
        "uncertified_action_count": sum(
            int(row["uncertified_action_count"]) for row in rows
        ),
        "flow_latent_mode": next(
            (
                row.get("flow_latent_mode")
                for row in rows
                if row.get("flow_latent_mode")
            ),
            None,
        ),
        "actor_forward_passes": max(
            (int(row.get("actor_forward_passes", 1)) for row in rows),
            default=1,
        ),
        "inference_latency_ms_mean": sum(
            float(row["inference_latency_ms_mean"]) for row in rows
        )
        / count,
        "inference_latency_ms_p95": max(
            (float(row["inference_latency_ms_p95"]) for row in rows),
            default=float("nan"),
        ),
        "guardian_certification_latency_ms_mean": sum(
            float(row["guardian_certification_latency_ms_mean"]) for row in rows
        )
        / count,
        "guardian_certification_latency_ms_p95": max(
            (float(row["guardian_certification_latency_ms_p95"]) for row in rows),
            default=float("nan"),
        ),
        "certified_control_latency_ms_mean": sum(
            float(row["certified_control_latency_ms_mean"]) for row in rows
        )
        / count,
        "certified_control_latency_ms_p95": max(
            (float(row["certified_control_latency_ms_p95"]) for row in rows),
            default=float("nan"),
        ),
        "error": next((row["error"] for row in rows if row["error"]), None),
        "rows": rows,
    }


def _orchestrate(args) -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    seeds = [int(value) for value in args.scene_seeds.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("scene seeds must be non-empty and unique")
    if any(mode not in ("hybrid", "ordinary") for mode in modes):
        raise ValueError("modes must contain only hybrid/ordinary")
    worker_dir = output.with_suffix("").parent / (output.stem + "_workers")
    per_seed: dict[str, dict[str, Any]] = {}
    commands = []
    jobs = []
    for seed in seeds:
        per_seed[str(seed)] = {}
        for mode in modes:
            for episode in range(args.episodes):
                worker_output = worker_dir / f"seed{seed}_{mode}_{episode:03d}.json"
                worker_trajectory = (
                    args.trajectory_output_dir.expanduser().resolve()
                    / f"seed{seed}_{mode}_{episode:03d}.npz"
                    if args.trajectory_output_dir is not None and mode == "ordinary"
                    else None
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                    "--policy",
                    args.policy,
                    "--worker",
                    "--worker-seed",
                    str(seed),
                    "--worker-mode",
                    mode,
                    "--episode-index",
                    str(episode),
                    "--worker-output",
                    str(worker_output),
                ]
                if worker_trajectory is not None:
                    command.extend(
                        ["--worker-trajectory-output", str(worker_trajectory)]
                    )
                jobs.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "episode": episode,
                        "command": command,
                        "worker_output": worker_output,
                    }
                )

    def run_job(job):
        completed = subprocess.run(
            job["command"],
            cwd=str(REPOSITORY_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {
            **job,
            "returncode": completed.returncode,
            "output": completed.stdout[-4000:],
        }

    max_workers = int(os.environ.get("GUARDIAN_PAPER_EVAL_WORKERS", "1"))
    if max_workers < 1:
        raise ValueError("GUARDIAN_PAPER_EVAL_WORKERS must be positive")
    if max_workers == 1:
        completed_jobs = [run_job(job) for job in jobs]
    else:
        # Every episode remains process-isolated.  This only overlaps
        # independent CPU MetaDrive workers; no second training GPU is used.
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
            completed_jobs = list(pool.map(run_job, jobs))

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {
        (seed, mode): [] for seed in seeds for mode in modes
    }
    for completed in completed_jobs:
        commands.append(
            {
                "command": completed["command"],
                "returncode": completed["returncode"],
                "output": completed["output"],
            }
        )
        if completed["returncode"] != 0:
            raise RuntimeError(
                "isolated evaluation failed for "
                f"seed={completed['seed']}, mode={completed['mode']}:\n"
                + completed["output"]
            )
        grouped[(completed["seed"], completed["mode"])].append(
            json.loads(completed["worker_output"].read_text(encoding="utf-8"))
        )
    for seed in seeds:
        for mode in modes:
            per_seed[str(seed)][mode] = _aggregate(mode, grouped[(seed, mode)])
    payload = {
        "format": "guardian_dsac_isolated_evaluation_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "action_semantics_version": TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        "policy": args.policy,
        "fresh_process_per_episode": True,
        "uncertified_proposal_handling": UNCERTIFIED_PROPOSAL_HANDLING,
        "uncertified_proposals_executed": False,
        "trajectory_capture": {
            "enabled": args.trajectory_output_dir is not None,
            "ordinary_only": True,
            "output_dir": (
                None
                if args.trajectory_output_dir is None
                else str(args.trajectory_output_dir.expanduser().resolve())
            ),
        },
        "per_seed": per_seed,
        "commands": commands,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    return _worker(args) if args.worker else _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
