#!/usr/bin/env python3
"""Evaluate Guardian-CalQL with one fresh Python process per MetaDrive episode.

This runner is intentionally stricter than rebuilding the Gym environment in
one long-lived process.  Every mode/episode starts a new interpreter, JAX
runtime, MPC object, MetaDrive engine, and post-reset wrapper, so a preceding
Hybrid rollout cannot select the numerical branch used by an ordinary run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.train_guardian_calql_metasafe import (  # noqa: E402
    INTERVENTION_ACCOUNTING_DEFINITION,
    INTERVENTION_ACCOUNTING_VERSION,
)
from relax.utils.action_semantics import (  # noqa: E402
    SPEED_COUPLED_ACTION_SEMANTICS,
)


MPC_OVERRIDES = (
    ("mpc_tracking_weight", "--mpc-tracking-weight"),
    ("mpc_velocity_weight", "--mpc-velocity-weight"),
    ("mpc_road_weight", "--mpc-road-weight"),
    ("mpc_road_buffer", "--mpc-road-buffer"),
    ("mpc_max_steer_rate", "--mpc-max-steer-rate"),
    ("lookahead", "--lookahead"),
    ("lateral_scale", "--lateral-scale"),
    ("target_velocity", "--target-velocity"),
    ("horizon", "--horizon"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", default="hybrid,ordinary")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--policy", choices=("champion", "learner"), default="champion"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scene-seeds",
        default=None,
        help="comma-separated seeds; each seed/mode/episode uses a fresh process",
    )
    parser.add_argument("--scene-selection-manifest", type=Path, default=None)
    parser.add_argument("--action-semantics", default=None)
    parser.add_argument("--max-command-speed", type=float, default=None)
    parser.add_argument(
        "--include-step-trace",
        action="store_true",
        help="persist compact per-step Actor/MPC diagnostics in each worker JSON",
    )
    for field, option in MPC_OVERRIDES:
        parser.add_argument(option, dest=field, type=float, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode",
        choices=("hybrid", "ordinary"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--episode-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_command_speed is not None and not (
        math.isfinite(args.max_command_speed) and args.max_command_speed > 0.0
    ):
        parser.error("--max-command-speed must be positive and finite")
    for field, option in MPC_OVERRIDES:
        value = getattr(args, field)
        if value is not None and not (math.isfinite(value) and value > 0.0):
            parser.error(f"{option} must be positive and finite")
    if args.worker and (args.worker_mode is None or args.worker_output is None):
        parser.error("worker mode requires --worker-mode and --worker-output")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    for candidate in (checkpoint.parent, checkpoint.parent.parent):
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("could not find config.json beside checkpoint")


def _minimum(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else float("nan")


def _worker(args) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from scripts.record_guardian_calql_metasafe_video import _load_algorithm
    from scripts.train_guardian_calql_metasafe import (
        _evaluate_policy,
        _policy_hash,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    config = json.loads(
        (_run_dir(checkpoint) / "config.json").read_text(encoding="utf-8")
    )
    algorithm, env_args, _obs_dim = _load_algorithm(checkpoint, config)
    env_args.fixed_scene_seed = int(args.seed)
    if args.action_semantics is not None:
        env_args.action_semantics = str(args.action_semantics)
    if args.max_command_speed is not None:
        env_args.max_command_speed = float(args.max_command_speed)
    env_args.recreate_environment_each_episode = True
    env_args.include_step_trace = bool(args.include_step_trace)
    for field, _option in MPC_OVERRIDES:
        value = getattr(args, field)
        if value is not None:
            setattr(env_args, field, float(value))
    result = _evaluate_policy(
        algorithm,
        env_args,
        mode=args.worker_mode,
        use_champion=args.policy == "champion",
        episodes=1,
    )
    result.update(
        {
            "worker_pid": os.getpid(),
            "episode_index": int(args.episode_index),
            "process_isolated": True,
            "policy_sha256": _policy_hash(
                (
                    algorithm.champion_policy
                    if hasattr(algorithm, "champion_policy")
                    else algorithm.champion_state.params.policy
                )
                if args.policy == "champion"
                else algorithm.get_policy_params()
            ),
        }
    )
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps(result, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return 0


def _aggregate(mode: str, suites: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for suite in suites:
        rows.extend(suite.get("rows", []))
    episodes = len(rows)
    return {
        "mode": mode,
        "episodes": episodes,
        "success_rate": sum(bool(row["success"]) for row in rows) / episodes,
        "collision_rate": sum(bool(row["collision"]) for row in rows) / episodes,
        "road_departure_rate": sum(
            bool(row["road_departure"]) for row in rows
        )
        / episodes,
        "total_environment_cost": sum(
            float(row["environment_cost"]) for row in rows
        ),
        "mean_return": sum(float(row["return"]) for row in rows) / episodes,
        "intervention_rate": sum(
            float(row["intervention_rate"]) for row in rows
        )
        / episodes,
        "actual_intervention_rate": sum(
            float(row.get("actual_intervention_rate", row["intervention_rate"]))
            for row in rows
        )
        / episodes,
        "audit_authority_rate": sum(
            float(row.get("audit_authority_rate", row["intervention_rate"]))
            for row in rows
        )
        / episodes,
        "diagnostic_only_rate": sum(
            float(row.get("diagnostic_only_rate", 0.0)) for row in rows
        )
        / episodes,
        "high_level_projection_rate": sum(
            float(row.get("high_level_projection_rate", 0.0)) for row in rows
        )
        / episodes,
        "actual_low_level_modification_rate": sum(
            float(row.get("actual_low_level_modification_rate", 0.0))
            for row in rows
        )
        / episodes,
        "mean_projection_norm": sum(
            float(row.get("mean_projection_norm", 0.0)) for row in rows
        )
        / episodes,
        "mean_cbf_trajectory_rmse": sum(
            float(row.get("mean_cbf_trajectory_rmse", 0.0)) for row in rows
        )
        / episodes,
        "mean_actor_action_delta_from_champion": sum(
            float(row.get("mean_actor_action_delta_from_champion", 0.0))
            for row in rows
        )
        / episodes,
        "max_actor_action_delta_from_champion": max(
            float(row.get("max_actor_action_delta_from_champion", 0.0))
            for row in rows
        ),
        "solver_finite_rate": sum(
            float(row["solver_finite_rate"]) for row in rows
        )
        / episodes,
        **{
            name: sum(float(row.get(name, float("nan"))) for row in rows)
            / episodes
            for name in (
                "raw_target_speed_mean",
                "raw_target_speed_std",
                "executed_target_speed_mean",
                "executed_target_speed_std",
                "actual_vehicle_speed_mean",
                "actual_vehicle_speed_std",
                "target_speed_p10",
                "target_speed_p50",
                "target_speed_p90",
                "speed_command_delta_mean",
                "speed_only_intervention_rate",
                "braking_command_fraction",
                "episode_duration",
                "progress_per_second",
            )
        },
        "min_shadow_hard_cbf_margin": _minimum(
            row["min_shadow_hard_cbf_margin"] for row in rows
        ),
        "min_ordinary_audit_cbf_margin": _minimum(
            row.get("min_ordinary_audit_cbf_margin", float("nan"))
            for row in rows
        ),
        "min_physical_body_sat_margin": _minimum(
            row["min_physical_body_sat_margin"] for row in rows
        ),
        "steps": [int(row["steps"]) for row in rows],
        "returns": [float(row["return"]) for row in rows],
        "scene_signatures": sorted(
            {str(row.get("scene_signature", "")) for row in rows}
        ),
        "post_reset_warmup_steps": sorted(
            {int(row.get("post_reset_warmup_steps", 0)) for row in rows}
        ),
        "rows": rows,
    }


def _orchestrate_single(args) -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    worker_dir = output.parent / f"{output.stem}_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not modes or any(mode not in ("hybrid", "ordinary") for mode in modes):
        raise ValueError("--modes must contain hybrid and/or ordinary")

    suites: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    script = Path(__file__).resolve()
    for mode in modes:
        for episode_index in range(args.episodes):
            worker_output = worker_dir / f"{mode}_{episode_index:03d}.json"
            worker_log = worker_dir / f"{mode}_{episode_index:03d}.log"
            command = [
                sys.executable,
                str(script),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--policy",
                args.policy,
                "--seed",
                str(args.seed),
                "--worker",
                "--worker-mode",
                mode,
                "--episode-index",
                str(episode_index),
                "--worker-output",
                str(worker_output),
            ]
            if args.include_step_trace:
                command.append("--include-step-trace")
            if args.action_semantics is not None:
                command.extend(["--action-semantics", args.action_semantics])
            if args.max_command_speed is not None:
                command.extend(
                    ["--max-command-speed", str(args.max_command_speed)]
                )
            for field, option in MPC_OVERRIDES:
                value = getattr(args, field)
                if value is not None:
                    command.extend([option, str(value)])
            completed = subprocess.run(
                command,
                cwd=str(script.parents[1]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            worker_log.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0 or not worker_output.is_file():
                raise RuntimeError(
                    f"isolated worker failed for {mode} episode "
                    f"{episode_index}; see {worker_log}"
                )
            suites[mode].append(
                json.loads(worker_output.read_text(encoding="utf-8"))
            )

    aggregates = {
        mode: _aggregate(mode, mode_suites)
        for mode, mode_suites in suites.items()
    }
    failures = []
    for mode, aggregate in aggregates.items():
        if aggregate["success_rate"] < 1.0:
            failures.append(f"{mode}_success")
        if aggregate["collision_rate"] > 0.0:
            failures.append(f"{mode}_collision")
        if aggregate["road_departure_rate"] > 0.0:
            failures.append(f"{mode}_road_departure")
        if aggregate["total_environment_cost"] > 0.0:
            failures.append(f"{mode}_environment_cost")
        if aggregate["solver_finite_rate"] < 1.0:
            failures.append(f"{mode}_solver_nonfinite")
        if not aggregate["min_physical_body_sat_margin"] > 0.0:
            failures.append(f"{mode}_physical_margin")
    if "ordinary" in aggregates and aggregates["ordinary"]["intervention_rate"] > 0.0:
        failures.append("ordinary_intervention")

    policy_hashes = sorted(
        {
            suite["policy_sha256"]
            for mode_suites in suites.values()
            for suite in mode_suites
        }
    )
    signatures = sorted(
        {
            signature
            for aggregate in aggregates.values()
            for signature in aggregate.get("scene_signatures", [])
            if signature
        }
    )
    if len(signatures) != 1:
        failures.append("scene_signature_not_unique")
    run_config = json.loads(
        (_run_dir(checkpoint) / "config.json").read_text(encoding="utf-8")
    )
    saved_args = run_config.get("args", {})
    action_semantics = str(
        args.action_semantics
        or saved_args.get(
            "action_semantics",
            saved_args.get(
                "action_semantics_version", "target_point_fixed_speed_v1"
            ),
        )
    )
    max_command_speed = float(
        args.max_command_speed
        if args.max_command_speed is not None
        else saved_args.get(
            "max_command_speed", saved_args.get("target_velocity", 4.8)
        )
    )
    manifest_hash = ""
    expected_signature = None
    if args.scene_selection_manifest is not None:
        manifest = json.loads(
            args.scene_selection_manifest.expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )
        manifest_hash = str(manifest.get("scene_selection_manifest_hash", ""))
        expected_signature = manifest.get("scene_signatures", {}).get(str(args.seed))
        if expected_signature is None:
            failures.append("scene_seed_absent_from_selection_manifest")
        elif signatures != [str(expected_signature)]:
            failures.append("scene_signature_manifest_mismatch")
        if manifest.get("action_semantics_version") != action_semantics:
            failures.append("action_semantics_manifest_mismatch")
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "policy_sha256": policy_hashes[0] if len(policy_hashes) == 1 else None,
        "policy_hashes": policy_hashes,
        "policy": args.policy,
        "fixed_scene_seed": int(args.seed),
        "scene_seed": int(args.seed),
        "scene_signature": signatures[0] if len(signatures) == 1 else None,
        "action_semantics_version": action_semantics,
        "max_command_speed": max_command_speed,
        "scene_selection_manifest_hash": manifest_hash,
        "episodes_per_mode": int(args.episodes),
        "process_isolation": "fresh_python_process_per_mode_episode",
        "actor_action_transform": "identity_clip",
        "intervention_accounting_version": INTERVENTION_ACCOUNTING_VERSION,
        "intervention_accounting_definition": (
            INTERVENTION_ACCOUNTING_DEFINITION
        ),
        "mpc_overrides": {
            field: getattr(args, field)
            for field, _option in MPC_OVERRIDES
            if getattr(args, field) is not None
        },
        "results": aggregates,
        "gate_passed": not failures,
        "gate_failures": failures,
        "worker_directory": str(worker_dir),
    }
    output.write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    compact = {
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "policy_sha256": summary["policy_sha256"],
        "process_isolation": summary["process_isolation"],
        "results": {
            mode: {
                key: value
                for key, value in aggregate.items()
                if key
                in {
                    "episodes",
                    "success_rate",
                    "collision_rate",
                    "road_departure_rate",
                    "total_environment_cost",
                    "mean_return",
                    "intervention_rate",
                    "actual_intervention_rate",
                    "audit_authority_rate",
                    "diagnostic_only_rate",
                    "high_level_projection_rate",
                    "actual_low_level_modification_rate",
                    "solver_finite_rate",
                    "min_ordinary_audit_cbf_margin",
                    "min_shadow_hard_cbf_margin",
                    "min_physical_body_sat_margin",
                    "steps",
                }
            }
            for mode, aggregate in aggregates.items()
        },
        "gate_passed": summary["gate_passed"],
        "gate_failures": failures,
        "output": str(output),
    }
    print(json.dumps(compact, indent=2, allow_nan=True))
    return 0 if not failures else 2


def _parse_scene_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds:
        raise ValueError("--scene-seeds is empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--scene-seeds contains duplicates")
    return seeds


def _orchestrate(args) -> int:
    if args.scene_seeds is None:
        return _orchestrate_single(args)
    seeds = _parse_scene_seeds(args.scene_seeds)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    per_seed_dir = output.parent / f"{output.stem}_seeds"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    artifacts: dict[str, Any] = {}
    process_failures: list[str] = []
    for seed in seeds:
        seed_output = per_seed_dir / f"seed_{seed}.json"
        seed_log = per_seed_dir / f"seed_{seed}.log"
        command = [
            sys.executable,
            str(script),
            "--checkpoint",
            str(args.checkpoint.expanduser().resolve()),
            "--output",
            str(seed_output),
            "--modes",
            args.modes,
            "--episodes",
            str(args.episodes),
            "--policy",
            args.policy,
            "--seed",
            str(seed),
        ]
        if args.include_step_trace:
            command.append("--include-step-trace")
        if args.scene_selection_manifest is not None:
            command.extend(
                [
                    "--scene-selection-manifest",
                    str(args.scene_selection_manifest.expanduser().resolve()),
                ]
            )
        if args.action_semantics is not None:
            command.extend(["--action-semantics", args.action_semantics])
        if args.max_command_speed is not None:
            command.extend(["--max-command-speed", str(args.max_command_speed)])
        for field, option in MPC_OVERRIDES:
            value = getattr(args, field)
            if value is not None:
                command.extend([option, str(value)])
        completed = subprocess.run(
            command,
            cwd=str(REPOSITORY_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        seed_log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode not in (0, 2) or not seed_output.is_file():
            process_failures.append(f"seed_{seed}:worker_failure")
            continue
        artifacts[str(seed)] = json.loads(seed_output.read_text(encoding="utf-8"))
    gate_failures = list(process_failures)
    for seed, artifact in artifacts.items():
        gate_failures.extend(
            f"seed_{seed}:{failure}" for failure in artifact.get("gate_failures", [])
        )
    manifest_hashes = sorted(
        {
            str(artifact.get("scene_selection_manifest_hash", ""))
            for artifact in artifacts.values()
        }
    )
    summary = {
        "format": "guardian_calql_cross_seed_isolated_evaluation_v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.expanduser().resolve()),
        "policy": args.policy,
        "scene_seeds": seeds,
        "episodes_per_mode": int(args.episodes),
        "process_isolation": "fresh_python_process_per_seed_mode_episode",
        "scene_selection_manifest_hash": (
            manifest_hashes[0] if len(manifest_hashes) == 1 else None
        ),
        "per_seed": artifacts,
        "gate_passed": not gate_failures and len(artifacts) == len(seeds),
        "gate_failures": gate_failures,
        "per_seed_directory": str(per_seed_dir),
    }
    output.write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scene_seeds": seeds,
                "gate_passed": summary["gate_passed"],
                "gate_failures": gate_failures,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0 if summary["gate_passed"] else 2


def main() -> int:
    args = parse_args()
    return _worker(args) if args.worker else _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
