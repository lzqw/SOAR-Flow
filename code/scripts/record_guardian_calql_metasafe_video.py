#!/usr/bin/env python3
"""Record a Guardian-CalQL MetaSafe rollout with explicit authority status."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import jax
import numpy as np

from relax.algorithm.guardian_calql import (
    GuardianCalQLConfig,
    GuardianCalQLSafeSAC,
)
from relax.algorithm.guardian_calql_v2_2_simple import (
    GuardianCalQLV22Config,
    GuardianCalQLV22SimpleSafeSAC,
)
from relax.network.safe_sac_jax import create_guardian_calql_net
from scripts.train_guardian_calql_metasafe import (
    INTERVENTION_ACCOUNTING_DEFINITION,
    INTERVENTION_ACCOUNTING_VERSION,
    _actor_action_for_mpc,
    _audit_low_level_cbf_authority,
    _make_environment,
    _policy_hash,
)
from scripts.video_utils import (
    DARK,
    GREEN,
    ORANGE,
    RED,
    WHITE,
    _alpha_box,
    _draw_vehicle_labels,
    _format_action,
    _format_number,
    _start_encoder,
    _text,
    _write_frame,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("ordinary", "hybrid"), default="ordinary"
    )
    parser.add_argument(
        "--policy", choices=("champion", "learner"), default="champion"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    return parser.parse_args()


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


def _load_algorithm(checkpoint: Path, config: dict):
    saved_args = config["args"]
    hidden_sizes = tuple(
        int(value.strip())
        for value in str(saved_args["hidden_sizes"]).split(",")
        if value.strip()
    )
    obs_dim = int(config["dataset"]["obs_dim"])
    act_dim = int(config["dataset"]["action_dim"])
    network, params = create_guardian_calql_net(
        jax.random.PRNGKey(int(saved_args.get("seed", 0))),
        critic_obs_dim=obs_dim,
        actor_obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        init_alpha=float(saved_args.get("init_alpha", 0.05)),
    )
    if saved_args.get("method") == "v2_2_simple":
        fields = GuardianCalQLV22Config.__dataclass_fields__
        algorithm_config = {
            key: value
            for key, value in config.get("algorithm_config", {}).items()
            if key in fields
        }
        metadata = config.get("checkpoint_metadata", {})
        algorithm = GuardianCalQLV22SimpleSafeSAC(
            network,
            params,
            config=GuardianCalQLV22Config(**algorithm_config),
            action_semantics_version=str(
                metadata.get(
                    "action_semantics_version",
                    saved_args.get(
                        "action_semantics", "target_point_speed_coupled_v1"
                    ),
                )
            ),
            offline_scene_seed=int(
                metadata.get(
                    "offline_scene_seed", saved_args.get("offline_scene_seed", 0)
                )
            ),
            online_scene_seeds=tuple(
                metadata.get(
                    "online_scene_seeds", saved_args.get("online_scene_seeds", [])
                )
            ),
            scene_signatures=metadata.get("scene_signatures", {}),
            scene_selection_manifest_hash=str(
                metadata.get("scene_selection_manifest_hash", "")
            ),
            allow_fixed_speed_ablation=bool(
                metadata.get("fixed_speed_ablation", False)
            ),
        )
    else:
        fields = GuardianCalQLConfig.__dataclass_fields__
        algorithm_config = {
            key: value
            for key, value in config.get("algorithm_config", {}).items()
            if key in fields
        }
        algorithm = GuardianCalQLSafeSAC(
            network,
            params,
            config=GuardianCalQLConfig(**algorithm_config),
        )
    algorithm.load(str(checkpoint))
    return algorithm, SimpleNamespace(**saved_args), obs_dim


def _minimum(values):
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.min(finite)) if finite.size else float("nan")


def _draw_timeline(frame, history, max_steps):
    height, width = frame.shape[:2]
    left, right = 24, width - 24
    top, bottom = height - 48, height - 25
    _alpha_box(frame, (left - 8, top - 27), (right + 8, bottom + 8), alpha=0.78)
    _text(
        frame,
        "Actual intervention timeline (green=Actor, red=modified)",
        (left, top - 8),
        color=WHITE,
        scale=0.46,
    )
    cv2.rectangle(frame, (left, top), (right, bottom), (75, 75, 75), -1)
    span = right - left
    for index, intervened in enumerate(history):
        x0 = left + int(index * span / max_steps)
        x1 = left + max(
            int((index + 1) * span / max_steps),
            int(index * span / max_steps) + 1,
        )
        cv2.rectangle(
            frame,
            (x0, top),
            (min(x1, right), bottom),
            RED if intervened else GREEN,
            -1,
        )


def _annotate(
    rgb,
    base_env,
    *,
    mode,
    step,
    max_steps,
    action,
    network_action,
    info,
    authority,
    history,
    total_cost,
    post_reset_steps,
):
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mgf = info.get("mgf", {})
    hard = mgf.get("raw_hard_safety", mgf.get("hard_safety", {}))
    physical = info.get("mpc_control", {})
    executed = np.asarray(info.get("executed_target", action), np.float32).reshape(2)
    projected = bool(mgf.get("projection_used", False))
    low_level_modified = bool(
        authority.get("actual_low_level_modified", False)
    )
    intervened = bool(authority.get("intervention", False)) if mode == "hybrid" else False
    audited = bool(authority.get("audit_authority", intervened)) if mode == "hybrid" else False
    diagnostic_only = bool(audited and not intervened)
    trajectory_rmse = float(authority.get("trajectory_rmse", 0.0))
    ordinary_audit_cbf_margin = float(
        authority.get("ordinary_hard_certificate", {}).get(
            "hard_cbf_margin", hard.get("hard_cbf_margin", np.nan)
        )
    )
    history.append(intervened)
    status_color = RED if intervened else ORANGE if diagnostic_only else GREEN

    cv2.rectangle(
        frame,
        (5, 5),
        (frame.shape[1] - 6, frame.shape[0] - 6),
        status_color,
        7,
    )
    _alpha_box(frame, (20, 18), (745, 480), color=DARK, alpha=0.88)
    _text(
        frame,
        f"ACTUAL SAFETY INTERVENTION: {'YES' if intervened else 'NO'}",
        (42, 58),
        color=status_color,
        scale=0.92,
        thickness=2,
    )
    if mode == "ordinary":
        authority_line = "TARGET PROJECTION / OBSTACLE CBF: OFF"
        execution = "ACTOR -> ORDINARY MPC"
    else:
        authority_line = "TARGET PROJECTION / OBSTACLE CBF: ON"
        execution = "PROTECTED MPC" if intervened else "ACTOR TARGET"
    _text(
        frame,
        authority_line,
        (42, 87),
        color=GREEN if mode == "ordinary" else ORANGE,
        scale=0.54,
        thickness=2,
    )
    collision = bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
    road_departure = bool(info.get("out_of_road", False) or info.get("road_departure", False))
    rows = [
        f"Step: {step:04d} / {max_steps}    Mode: {mode.upper()}",
        f"Policy begins after reset frame: {post_reset_steps}",
        f"Execution: {execution}",
        f"Raw network target: {_format_action(network_action)}",
        f"Actor target (unscaled): {_format_action(action)}",
        f"Executed target: {_format_action(executed)}",
        f"High-level projection: {'YES' if projected else 'NO'}",
        f"Actual low-level trajectory modified: {'YES' if low_level_modified else 'NO'}",
        f"Audit-only warning: {'YES' if diagnostic_only else 'NO'}",
        f"CBF-on/off trajectory RMSE: {_format_number(trajectory_rmse)} m",
        f"Shadow CBF psi: {_format_number(hard.get('hard_cbf_margin'))}",
        f"Ordinary audit CBF psi: {_format_number(ordinary_audit_cbf_margin)}",
        (
            "Live rectangle SAT margin: "
            f"{_format_number(physical.get('physical_body_sat_margin'))} m"
        ),
        f"collision={int(collision)}  road_departure={int(road_departure)}",
        f"SafeMetaDrive cost: step={_format_number(info.get('cost', 0.0))}  total={_format_number(total_cost)}",
    ]
    for index, row in enumerate(rows):
        color = ORANGE if index in (5, 6, 8, 9) else WHITE
        _text(frame, row, (42, 119 + 25 * index), color=color, scale=0.50)
    _draw_vehicle_labels(frame, base_env)
    _draw_timeline(frame, history, max_steps)
    return frame, {
        "intervened": intervened,
        "projected": projected,
        "audit_authority": audited,
        "diagnostic_only": diagnostic_only,
        "actual_low_level_modified": low_level_modified,
        "trajectory_rmse": trajectory_rmse,
        "shadow_hard_cbf_margin": float(hard.get("hard_cbf_margin", np.nan)),
        "ordinary_audit_cbf_margin": ordinary_audit_cbf_margin,
        "physical_body_sat_margin": float(
            physical.get("physical_body_sat_margin", np.nan)
        ),
    }


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    run_dir = _run_dir(checkpoint)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    algorithm, env_args, obs_dim = _load_algorithm(checkpoint, config)
    max_steps = int(args.max_steps or env_args.max_episode_steps)
    env = _make_environment(env_args, args.mode)
    observation, reset_info = env.reset(seed=args.seed)
    observation = np.asarray(observation, np.float32).reshape(-1)
    if observation.size != obs_dim:
        raise ValueError(f"Actor expected {obs_dim} values, received {observation.size}")
    base_env = env.unwrapped
    if hasattr(base_env, "vehicle"):
        base_env.vehicle._panda_color = (0.10, 0.65, 0.95, 1.0)
    for obstacle in list(getattr(base_env, "_parked_objs", [])):
        obstacle._panda_color = (0.95, 0.20, 0.12, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = _start_encoder(
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        crf=args.crf,
    )
    started = time.perf_counter()
    history = []
    diagnostics = []
    total_cost = float(reset_info.get("post_reset_warmup_cost", 0.0))
    success = collision = road_departure = False
    frames_written = 0
    last_frame = None
    final_info = {}
    try:
        for step in range(max_steps):
            network_action = np.asarray(
                algorithm.get_deterministic_champion_action(observation)
                if args.policy == "champion"
                else algorithm.get_deterministic_action(observation),
                np.float32,
            )
            network_action = np.clip(network_action, -1.0, 1.0)
            action = _actor_action_for_mpc(network_action)
            if args.mode == "hybrid":
                certificate = env.certify_target_action(action)
                authority = _audit_low_level_cbf_authority(
                    env,
                    action,
                    certificate,
                    trajectory_tolerance=float(env_args.cbf_trajectory_tolerance),
                    derive_teacher=False,
                )
            else:
                certificate = env.certify_ordinary_target_action(action)
                authority = {
                    "intervention": False,
                    "actual_control_modified": False,
                    "high_level_projection": False,
                    "high_level_modified": False,
                    "low_level_cbf_authority": False,
                    "actual_low_level_modified": False,
                    "audit_authority": False,
                    "diagnostic_only": False,
                    "trajectory_rmse": 0.0,
                }
            observation, _reward, terminated, truncated, info = (
                env.execute_certified_target(certificate)
            )
            observation = np.asarray(observation, np.float32).reshape(-1)
            final_info = dict(info)
            total_cost += max(float(info.get("cost", 0.0)), 0.0)
            success = bool(success or info.get("arrive_dest", False) or info.get("success", False))
            collision = bool(collision or info.get("crash_vehicle", False) or info.get("crash_object", False))
            road_departure = bool(road_departure or info.get("out_of_road", False) or info.get("road_departure", False))
            rgb = base_env.render(
                mode="top_down",
                window=False,
                screen_size=(args.width, args.height),
                film_size=(1800, 1100),
                scaling=12,
                num_stack=20,
                history_smooth=1,
                semantic_map=False,
                draw_target_vehicle_trajectory=True,
            )
            frame, diagnostic = _annotate(
                rgb,
                base_env,
                mode=args.mode,
                step=step + 1,
                max_steps=max_steps,
                action=action,
                network_action=network_action,
                info=info,
                authority=authority,
                history=history,
                total_cost=total_cost,
                post_reset_steps=int(reset_info.get("post_reset_warmup_steps", 0)),
            )
            diagnostics.append(diagnostic)
            if frames_written == 0:
                for _ in range(args.fps):
                    _write_frame(encoder, frame)
                    frames_written += 1
            _write_frame(encoder, frame)
            frames_written += 1
            last_frame = frame
            if terminated or truncated:
                break

        if last_frame is not None:
            end = last_frame.copy()
            _alpha_box(end, (775, 35), (1245, 176), alpha=0.90)
            passed = success and not collision and not road_departure and total_cost == 0.0
            _text(
                end,
                "NO-CBF SUCCESS" if passed else "ROLLOUT FAILED",
                (800, 78),
                color=GREEN if passed else RED,
                scale=0.82,
                thickness=2,
            )
            _text(
                end,
                f"intervention steps: {sum(history)} / {len(history)}",
                (800, 116),
                color=WHITE,
                scale=0.58,
            )
            _text(
                end,
                f"collision={int(collision)}  cost={_format_number(total_cost)}",
                (800, 150),
                color=WHITE,
                scale=0.56,
            )
            for _ in range(max(int(args.tail_seconds * args.fps), 1)):
                _write_frame(encoder, end)
                frames_written += 1
            cv2.imwrite(str(args.output.with_suffix(".png")), end)
    finally:
        env.close()
        if encoder.stdin is not None:
            encoder.stdin.close()
        stderr = (
            encoder.stderr.read().decode("utf-8", errors="replace")
            if encoder.stderr
            else ""
        )
        return_code = encoder.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr}")

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "policy_sha256": _policy_hash(
            (
                algorithm.champion_policy
                if hasattr(algorithm, "champion_policy")
                else algorithm.champion_state.params.policy
            )
            if args.policy == "champion"
            else algorithm.get_policy_params()
        ),
        "video": str(args.output.resolve()),
        "mode": args.mode,
        "policy": args.policy,
        "actor_action_transform": "identity_clip",
        "intervention_accounting_version": INTERVENTION_ACCOUNTING_VERSION,
        "intervention_accounting_definition": (
            INTERVENTION_ACCOUNTING_DEFINITION
        ),
        "seed": args.seed,
        "actor_observation_dim": obs_dim,
        "post_reset_warmup_steps": int(
            reset_info.get("post_reset_warmup_steps", 0)
        ),
        "steps": len(history),
        "success": success,
        "collision": collision,
        "road_departure": road_departure,
        "environment_cost": total_cost,
        "intervention_steps": int(sum(history)),
        "intervention_rate": sum(history) / max(len(history), 1),
        "actual_intervention_steps": int(sum(history)),
        "actual_intervention_rate": sum(history) / max(len(history), 1),
        "audit_authority_steps": int(
            sum(row["audit_authority"] for row in diagnostics)
        ),
        "diagnostic_only_steps": int(
            sum(row["diagnostic_only"] for row in diagnostics)
        ),
        "target_projection_enabled": args.mode == "hybrid",
        "obstacle_cbf_enabled": args.mode == "hybrid",
        "minimum_shadow_hard_cbf_margin": _minimum(
            [row["shadow_hard_cbf_margin"] for row in diagnostics]
        ),
        "minimum_ordinary_audit_cbf_margin": _minimum(
            [row["ordinary_audit_cbf_margin"] for row in diagnostics]
        ),
        "minimum_physical_body_sat_margin": _minimum(
            [row["physical_body_sat_margin"] for row in diagnostics]
        ),
        "maximum_cbf_trajectory_rmse": max(
            (row["trajectory_rmse"] for row in diagnostics), default=0.0
        ),
        "fps": args.fps,
        "video_frames": frames_written,
        "video_duration_seconds": frames_written / args.fps,
        "recording_wall_seconds": time.perf_counter() - started,
        "final_info_success": bool(
            final_info.get("arrive_dest", False) or final_info.get("success", False)
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    return int(not success or collision or road_departure or total_cost > 0.0)


if __name__ == "__main__":
    raise SystemExit(main())
