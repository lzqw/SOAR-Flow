#!/usr/bin/env python3
"""Train Guardian-DSAC V2.3.1 on fixed MetaSafe Seed0/Seed1."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.utils.action_semantics import (  # noqa: E402
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
    encode_scene_signature,
)
from relax.utils.experience import GuardianDSACExperience  # noqa: E402
from relax.utils.guardian_deployment_gate import (  # noqa: E402
    GuardianDeploymentGateConfig,
    apply_deployment_gate,
)
from relax.utils.guardian_learning_gate import (  # noqa: E402
    apply_multi_seed_hybrid_learning_gate,
)
from scripts.train_guardian_calql_metasafe import (  # noqa: E402
    _audit_low_level_cbf_authority,
    _concatenate_batches,
    _finalize_episode_returns,
    _step_experience,
)


DEFAULT_ONLINE_SEED_CYCLE = (1, 1, 1, 1, 0)
DEFAULT_HYBRID_GATE_SEEDS = (0, 1)
FAST_V231_MAX_COMMAND_SPEED = 10.75


def _parse_seed_sequence(value: str, *, option: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError(f"{option} must contain at least one seed")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError(f"{option} seeds must be non-negative")
    return seeds


@dataclass
class EpisodeSeedCycle:
    """State machine that advances only after a complete online episode."""

    seeds: tuple[int, ...] = DEFAULT_ONLINE_SEED_CYCLE
    completed_episodes: int = 0

    def __post_init__(self) -> None:
        self.seeds = tuple(int(seed) for seed in self.seeds)
        if not self.seeds:
            raise ValueError("online seed cycle must not be empty")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("online seed cycle contains a negative seed")
        if self.completed_episodes < 0:
            raise ValueError("completed_episodes must be non-negative")

    @property
    def current_seed(self) -> int:
        return self.seeds[self.completed_episodes % len(self.seeds)]

    def complete_episode(self) -> int:
        self.completed_episodes += 1
        return self.current_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "runs/metasafe_cbf/fixed_seed0_target_speed_lateral_300ep_v231/"
            "metasafe_cbf_dataset.npz"
        ),
    )
    parser.add_argument("--expected-dataset-sha256", default="")
    parser.add_argument(
        "--expected-speed-sampling-profile",
        choices=("balanced_v23", "fast_v231"),
        default="fast_v231",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/guardian_dsac_v2_3_1")
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--phase", choices=("offline", "online", "full"), default="full")
    parser.add_argument("--offline-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--init-alpha", type=float, default=0.05)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--proxy-discount", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--proxy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha-learning-rate", type=float, default=3e-4)
    parser.add_argument("--distributional-loss", choices=("hdsac", "gaussian_nll"), default="hdsac")
    parser.add_argument("--calql-weight", type=float, default=1.0)
    parser.add_argument("--reference-weight-initial", type=float, default=0.0)
    parser.add_argument("--reference-weight-final", type=float, default=0.0)
    parser.add_argument("--reference-decay-steps", type=int, default=100_000)
    parser.add_argument("--reference-std-weight", type=float, default=0.1)
    parser.add_argument("--proxy-raw-pair-weight", type=float, default=1.0)
    parser.add_argument("--proxy-pair-gap-weight", type=float, default=0.0)
    parser.add_argument("--proxy-actor-weight", type=float, default=0.5)
    parser.add_argument(
        "--proxy-actor-objective",
        choices=("probability", "softplus_logit"),
        default="probability",
        help=(
            "Actor risk transform: original bounded probability or explicit "
            "non-saturating softplus of the calibrated P logit"
        ),
    )
    parser.add_argument("--proximal-weight-initial", type=float, default=1.0)
    parser.add_argument("--proximal-weight-min", type=float, default=0.05)
    parser.add_argument("--proximal-decay-steps", type=int, default=20_000)
    parser.add_argument("--proximal-speed-weight", type=float, default=0.5)
    parser.add_argument("--proximal-lateral-weight", type=float, default=1.0)
    parser.add_argument("--bc-gradient-steps", type=int, default=30_000)
    parser.add_argument("--critic-gradient-steps", type=int, default=50_000)
    parser.add_argument("--proxy-gradient-steps", type=int, default=30_000)
    parser.add_argument("--online-steps", type=int, default=100_000)
    parser.add_argument("--online-buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--offline-ratio-initial", type=float, default=0.50)
    parser.add_argument("--offline-ratio-final", type=float, default=0.20)
    parser.add_argument("--online-exploration-std", type=float, default=0.04)
    parser.add_argument("--stratified-speed-exploration", type=float, default=0.10)
    parser.add_argument(
        "--online-inverse-teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "derive the existing inverse-MPC corrected action for online "
            "interventions so Seed-specific P pairs have a safer action"
        ),
    )
    parser.add_argument(
        "--online-seed-cycle",
        default=",".join(str(seed) for seed in DEFAULT_ONLINE_SEED_CYCLE),
    )
    parser.add_argument(
        "--hybrid-gate-seeds",
        default=",".join(str(seed) for seed in DEFAULT_HYBRID_GATE_SEEDS),
    )
    parser.add_argument("--hybrid-gate-every", type=int, default=4_000)
    parser.add_argument("--deployment-gate-every", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--component-smoke", action="store_true")
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-first-deployable", action="store_true")
    parser.add_argument(
        "--start-from-deployable",
        action="store_true",
        help="continue from the certified deployable state inside a v7 checkpoint",
    )

    parser.add_argument("--min-command-speed", type=float, default=1.0)
    parser.add_argument(
        "--max-command-speed", type=float, default=FAST_V231_MAX_COMMAND_SPEED
    )
    parser.add_argument("--lookahead", type=float, default=10.0)
    parser.add_argument("--lateral-scale", type=float, default=3.5)
    parser.add_argument(
        "--target-velocity", type=float, default=FAST_V231_MAX_COMMAND_SPEED
    )
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--mpc-tracking-weight", type=float, default=6.0)
    parser.add_argument("--mpc-velocity-weight", type=float, default=30.0)
    parser.add_argument("--mpc-road-weight", type=float, default=100.0)
    parser.add_argument("--mpc-road-buffer", type=float, default=1.5)
    parser.add_argument("--mpc-max-steer-rate", type=float, default=0.1)
    parser.add_argument("--clearance", type=float, default=0.10)
    parser.add_argument("--cbf-gamma", type=float, default=3.0)
    parser.add_argument("--projection-reserve", type=float, default=0.10)
    parser.add_argument("--projection-kinematic-reserve", type=float, default=0.02)
    parser.add_argument("--sqp-steps", type=int, default=12)
    parser.add_argument("--admm-steps", type=int, default=120)
    parser.add_argument("--max-obstacles", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--post-reset-warmup-steps", type=int, default=1)
    parser.add_argument("--cbf-trajectory-tolerance", type=float, default=0.08)
    args = parser.parse_args(argv)

    try:
        args.online_seed_cycle = _parse_seed_sequence(
            args.online_seed_cycle, option="--online-seed-cycle"
        )
        args.hybrid_gate_seeds = _parse_seed_sequence(
            args.hybrid_gate_seeds, option="--hybrid-gate-seeds"
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if any(seed not in (0, 1) for seed in args.online_seed_cycle):
        parser.error("V2.3.1 online collection is restricted to Seed0 and Seed1")
    if set(args.hybrid_gate_seeds) != {0, 1} or len(args.hybrid_gate_seeds) != 2:
        parser.error("V2.3.1 Hybrid gate must cover Seed0 and Seed1 exactly once")

    positive_int = (
        "batch_size",
        "online_buffer_size",
        "learning_starts",
        "gradient_steps",
        "policy_delay",
        "hybrid_gate_every",
        "deployment_gate_every",
        "log_every",
        "reference_decay_steps",
        "proximal_decay_steps",
        "max_episode_steps",
        "sqp_steps",
        "admm_steps",
        "max_obstacles",
    )
    if any(getattr(args, name) < 1 for name in positive_int):
        parser.error("positive integer option is non-positive")
    for name in (
        "bc_gradient_steps",
        "critic_gradient_steps",
        "proxy_gradient_steps",
        "online_steps",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.phase == "online" and args.offline_checkpoint is None:
        parser.error("--phase online requires --offline-checkpoint")
    if args.start_from_deployable and args.offline_checkpoint is None:
        parser.error("--start-from-deployable requires --offline-checkpoint")
    if not 0.0 <= args.offline_ratio_final <= args.offline_ratio_initial <= 1.0:
        parser.error("invalid offline replay ratios")
    if not 0.0 <= args.stratified_speed_exploration <= 1.0:
        parser.error("stratified speed exploration must be in [0, 1]")
    if not 0.0 <= args.min_command_speed < args.max_command_speed:
        parser.error("invalid command speed limits")
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (
            args.proxy_actor_weight,
            args.proxy_pair_gap_weight,
            args.proximal_weight_initial,
            args.proximal_weight_min,
            args.proximal_speed_weight,
            args.proximal_lateral_weight,
            args.reference_weight_initial,
            args.reference_weight_final,
        )
    ):
        parser.error("Actor/reference loss weights must be finite and non-negative")
    if (
        not np.isfinite(args.proxy_raw_pair_weight)
        or args.proxy_raw_pair_weight < 1.0
    ):
        parser.error("proxy raw pair weight must be finite and at least one")
    if args.proximal_weight_min > args.proximal_weight_initial:
        parser.error("proximal minimum cannot exceed its initial value")
    if args.proximal_speed_weight + args.proximal_lateral_weight <= 0.0:
        parser.error("at least one proximal axis weight must be positive")
    if args.component_smoke:
        args.bc_gradient_steps = min(args.bc_gradient_steps, 32)
        args.critic_gradient_steps = min(args.critic_gradient_steps, 64)
        args.proxy_gradient_steps = min(args.proxy_gradient_steps, 32)
        args.online_steps = 1_000
        args.learning_starts = min(args.learning_starts, 64)
        args.batch_size = min(args.batch_size, 128)
        args.sqp_steps = min(args.sqp_steps, 4)
        args.admm_steps = min(args.admm_steps, 24)
        args.log_every = min(args.log_every, 50)
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=True, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"cannot JSON serialize {type(value)!r}")


def _grouped_actor_diagnostics(metrics: Mapping[str, float]) -> dict[str, Any]:
    """Add list-valued audit views without changing scalar optimizer metrics."""

    p_keys = tuple(f"p_policy_by_speed_bin_{index}" for index in range(4))
    fraction_keys = tuple(
        f"p_policy_speed_bin_fraction_{index}" for index in range(4)
    )
    grouped: dict[str, Any] = {}
    if all(name in metrics for name in p_keys):
        grouped["p_policy_by_speed_bin"] = [
            float(metrics[name]) for name in p_keys
        ]
    if all(name in metrics for name in fraction_keys):
        grouped["p_policy_speed_bin_fractions"] = [
            float(metrics[name]) for name in fraction_keys
        ]
    return grouped


def _tree_changed(before, after, *, tolerance: float = 0.0) -> bool:
    import jax

    first = jax.tree_util.tree_leaves(before)
    second = jax.tree_util.tree_leaves(after)
    return any(
        np.max(np.abs(np.asarray(left) - np.asarray(right))) > tolerance
        for left, right in zip(first, second)
    )


def _environment_config(args) -> dict[str, Any]:
    return {
        name: getattr(args, name)
        for name in (
            "min_command_speed",
            "max_command_speed",
            "lookahead",
            "lateral_scale",
            "target_velocity",
            "horizon",
            "mpc_tracking_weight",
            "mpc_velocity_weight",
            "mpc_road_weight",
            "mpc_road_buffer",
            "mpc_max_steer_rate",
            "clearance",
            "cbf_gamma",
            "projection_reserve",
            "projection_kinematic_reserve",
            "sqp_steps",
            "admm_steps",
            "max_obstacles",
            "max_episode_steps",
            "post_reset_warmup_steps",
        )
    }


def _make_environment(args, *, seed: int, mode: str):
    from relax.env.drive.metasafe import make_metasafe_cbf_mpc_env

    protected = mode == "hybrid"
    return make_metasafe_cbf_mpc_env(
        fixed_seed=seed,
        use_render=False,
        action_semantics_version=TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        min_command_speed=args.min_command_speed,
        max_command_speed=args.max_command_speed,
        lookahead=args.lookahead,
        lateral_scale=args.lateral_scale,
        target_velocity=args.target_velocity,
        horizon=args.horizon,
        tracking_weight=args.mpc_tracking_weight,
        velocity_weight=args.mpc_velocity_weight,
        road_weight=args.mpc_road_weight,
        road_buffer=args.mpc_road_buffer,
        max_steer_rate=args.mpc_max_steer_rate,
        clearance=args.clearance,
        cbf_gamma=args.cbf_gamma,
        projection_reserve=args.projection_reserve,
        projection_kinematic_reserve=args.projection_kinematic_reserve,
        sqp_steps=args.sqp_steps,
        admm_steps=args.admm_steps,
        max_obstacles=args.max_obstacles,
        cbf_enable=protected,
        terminal_projection=protected,
        post_reset_warmup_steps=args.post_reset_warmup_steps,
    )


def _predeclare_scene_signatures(args, dataset) -> dict[str, str]:
    from relax.env.drive.metasafe import make_fixed_metasafe_env

    signatures = {"0": str(dataset.scene_signature)}
    # Seed2 remains untouched until a later protocol explicitly authorizes a
    # held-out evaluation.
    for seed in (1,):
        env = make_fixed_metasafe_env(
            fixed_seed=seed,
            use_render=False,
            action_semantics_version=TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
            lookahead_distance=args.lookahead,
            max_command_speed=args.max_command_speed,
            speed_mapping="state_reachable_speed_linear_v1",
            mpc_horizon=args.horizon,
        )
        try:
            _, info = env.reset(seed=seed)
            signatures[str(seed)] = str(info["metasafe_scene_signature"])
        finally:
            env.close()
    return signatures


def _step_dsac_experience(*args, **kwargs) -> GuardianDSACExperience:
    base = _step_experience(*args, **kwargs)
    info = args[7] if len(args) > 7 else kwargs["info"]
    mgf = info.get("mgf", {})

    def value(name: str, default):
        return info.get(name, mgf.get(name, default))

    lower = float(value("reachable_speed_lower", np.nan))
    upper = float(value("reachable_speed_upper", np.nan))
    actual = float(value("actual_vehicle_speed", np.nan))
    emergency = bool(value("emergency_stop", False))
    if not all(np.isfinite(number) for number in (lower, upper, actual)):
        raise RuntimeError("online transition lacks finite V2.3 speed bounds")
    if upper < lower:
        raise RuntimeError("online transition has inverted speed bounds")
    return GuardianDSACExperience(
        **base._asdict(),
        reachable_speed_lower=np.float32(lower),
        reachable_speed_upper=np.float32(upper),
        actual_vehicle_speed=np.float32(actual),
        emergency_stop=np.bool_(emergency),
    )


def _algorithm_config(args):
    from relax.algorithm.guardian_dsac_v2_3 import GuardianDSACV23Config

    return GuardianDSACV23Config(
        gamma=args.discount,
        tau=args.tau,
        critic_learning_rate=args.critic_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        proxy_learning_rate=args.proxy_learning_rate,
        alpha_learning_rate=args.alpha_learning_rate,
        distributional_loss=args.distributional_loss,
        calql_weight=args.calql_weight,
        reference_weight_initial=args.reference_weight_initial,
        reference_weight_final=args.reference_weight_final,
        reference_decay_steps=args.reference_decay_steps,
        reference_std_weight=args.reference_std_weight,
        proxy_raw_pair_weight=args.proxy_raw_pair_weight,
        proxy_pair_gap_weight=args.proxy_pair_gap_weight,
        proxy_actor_weight=args.proxy_actor_weight,
        proxy_actor_objective=args.proxy_actor_objective,
        proximal_weight_initial=args.proximal_weight_initial,
        proximal_weight_min=args.proximal_weight_min,
        proximal_decay_steps=args.proximal_decay_steps,
        proximal_speed_weight=args.proximal_speed_weight,
        proximal_lateral_weight=args.proximal_lateral_weight,
    )


def _create_algorithm(args, dataset, scene_signatures, key):
    from relax.algorithm.guardian_dsac_v2_3 import GuardianDSACV23
    from relax.network.safe_sac_jax import create_guardian_dsac_net

    hidden_sizes = tuple(
        int(value.strip()) for value in args.hidden_sizes.split(",") if value.strip()
    )
    network, params = create_guardian_dsac_net(
        key,
        critic_obs_dim=dataset.summary.obs_dim,
        actor_obs_dim=dataset.summary.obs_dim,
        act_dim=dataset.summary.action_dim,
        hidden_sizes=hidden_sizes,
        init_alpha=args.init_alpha,
    )
    algorithm = GuardianDSACV23(
        network,
        params,
        config=_algorithm_config(args),
        action_semantics_version=TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        scene_seeds=(0, 1),
        scene_signatures=scene_signatures,
        dataset_sha256=dataset.sha256,
    )
    return algorithm, hidden_sizes


def _sample_mixed_batch(
    dataset,
    replay,
    *,
    batch_size,
    offline_ratio,
    rng,
    flow: bool = False,
):
    online_count = (
        min(int(round(batch_size * (1.0 - offline_ratio))), batch_size - 1)
        if len(replay) > 0
        else 0
    )
    offline_count = batch_size - online_count
    batches = []
    if offline_count:
        offline_batch = dataset.sample(
            offline_count, split="train", to_jax=False
        )
        if flow:
            from relax.offline.guardian_dsac_multiseed_dataset import (
                guardian_dsac_to_flow_experience,
            )

            offline_batch = guardian_dsac_to_flow_experience(offline_batch)
        batches.append(offline_batch)
    if online_count:
        batches.append(replay.sample(online_count, to_jax=False))
    return _concatenate_batches(batches, rng)


def _offline_validation(algorithm, dataset, *, split: str = "validation") -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for batch in dataset.iter_split(split, batch_size=4096, to_jax=False):
        size = int(np.asarray(batch.reward).shape[0])
        mean_1, std_1, mean_2, std_2 = algorithm.task_q_distribution(
            batch.obs, batch.action
        )
        p = algorithm.proxy_value(batch.obs, batch.raw_action)
        target = np.asarray(batch.guardian_dependence_target, dtype=np.float32)
        valid = np.asarray(batch.guardian_dependence_target_valid, dtype=bool)
        row = {
            "q_mean": float(np.mean(np.minimum(mean_1, mean_2))),
            "q_std_1": float(np.mean(std_1)),
            "q_std_2": float(np.mean(std_2)),
            "q_disagreement": float(np.mean(np.abs(mean_1 - mean_2))),
            "q_calibration_mae": float(
                np.mean(
                    np.abs(
                        np.minimum(mean_1, mean_2)
                        - np.asarray(batch.behavior_return_to_go)
                    )
                )
            ),
            "proxy_mae": float(
                np.mean(np.abs(p[valid] - target[valid])) if np.any(valid) else 0.0
            ),
            "proxy_mean": float(np.mean(p)),
        }
        for name, value in row.items():
            totals[name] = totals.get(name, 0.0) + size * value
        count += size
    return {name: value / max(count, 1) for name, value in totals.items()}


def _run_isolated_evaluation(
    *,
    checkpoint: Path,
    output: Path,
    seeds: tuple[int, ...],
    modes: tuple[str, ...],
    policy: str,
    trajectory_output_dir: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/evaluate_guardian_dsac_isolated.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--scene-seeds",
        ",".join(str(seed) for seed in seeds),
        "--modes",
        ",".join(modes),
        "--episodes",
        "1",
        "--policy",
        policy,
    ]
    if trajectory_output_dir is not None:
        command.extend(
            ["--trajectory-output-dir", str(trajectory_output_dir)]
        )
    completed = subprocess.run(
        command,
        cwd=str(REPOSITORY_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output.with_suffix(output.suffix + ".log").write_text(
        completed.stdout, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated Guardian-DSAC evaluation failed:\n" + completed.stdout[-8000:]
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _offline_train(
    algorithm,
    dataset,
    args,
    *,
    key,
    metrics_path: Path,
) -> tuple[Any, dict[str, Any]]:
    import jax

    initial_policy = algorithm.state.params.policy
    initial_q1 = algorithm.state.params.q1
    initial_proxy = algorithm.state.params.proxy
    latest: dict[str, float] = {}
    for step in range(args.bc_gradient_steps):
        batch = dataset.sample(
            args.batch_size,
            split="train",
            teacher_fraction=0.25,
            to_jax=True,
        )
        key, update_key = jax.random.split(key)
        latest = algorithm.pretrain_bc(update_key, batch)
        if (step + 1) % args.log_every == 0:
            _append_jsonl(metrics_path, {"phase": "offline_bc", "step": step + 1, **latest})
    bc_changed = _tree_changed(initial_policy, algorithm.state.params.policy)

    for step in range(args.critic_gradient_steps):
        batch = dataset.sample(args.batch_size, split="train", to_jax=True)
        key, update_key = jax.random.split(key)
        latest = algorithm.offline_critic_update(update_key, batch)
        if (step + 1) % args.log_every == 0:
            _append_jsonl(metrics_path, {"phase": "offline_calql_dsac", "step": step + 1, **latest})
    q_changed = _tree_changed(initial_q1, algorithm.state.params.q1)

    for step in range(args.proxy_gradient_steps):
        batch = dataset.sample(
            args.batch_size,
            split="train",
            pair_fraction=0.50,
            to_jax=True,
        )
        latest = algorithm.proxy_update(batch)
        if (step + 1) % args.log_every == 0:
            _append_jsonl(metrics_path, {"phase": "offline_proxy", "step": step + 1, **latest})
    proxy_changed = _tree_changed(initial_proxy, algorithm.state.params.proxy)

    validation_indices = dataset.split_transition_indices("validation")
    validation = dataset.experience(validation_indices)
    thresholds = algorithm.mark_offline_reference(jax.device_put(validation))
    metrics = _offline_validation(algorithm, dataset)
    summary = {
        "bc_steps": args.bc_gradient_steps,
        "critic_steps": args.critic_gradient_steps,
        "proxy_steps": args.proxy_gradient_steps,
        "actor_updated": bc_changed,
        "task_critic_updated": q_changed,
        "proxy_updated": proxy_changed,
        "support_thresholds": asdict(thresholds),
        "validation": metrics,
        "speed_coverage": dataset.speed_coverage(),
    }
    return key, summary


def _online_safety_failures(info: Mapping[str, Any]) -> list[str]:
    failures = []
    mgf = info.get("mgf", {})
    physical = info.get("mpc_control", {})
    if not bool(mgf.get("solver_finite", False)):
        failures.append("solver_non_finite")
    if bool(mgf.get("projection_failed", False)):
        failures.append("projection_failed")
    if max(float(info.get("cost", 0.0)), 0.0) > 0.0:
        failures.append("environment_cost")
    if info.get("crash_vehicle", False) or info.get("crash_object", False):
        failures.append("collision")
    if info.get("out_of_road", False) or info.get("road_departure", False):
        failures.append("road_departure")
    margin = float(physical.get("physical_body_sat_margin", np.nan))
    if not np.isfinite(margin) or margin <= 0.0:
        failures.append("physical_margin")
    return sorted(set(failures))


def _store_complete_online_episode(
    replay,
    episode: list[GuardianDSACExperience],
    *,
    scene_seed: int,
    scene_signature: str,
    gamma: float,
    proxy_discount: float,
) -> list[GuardianDSACExperience]:
    """Validate one seed-bound episode, finish its targets, and add it once."""

    if not episode:
        raise ValueError("cannot store an empty online episode")
    if not bool(np.asarray(episode[-1].done)):
        raise ValueError("online replay only accepts complete episodes")
    if any(bool(np.asarray(item.done)) for item in episode[:-1]):
        raise ValueError("online episode contains a premature done transition")
    expected_signature = encode_scene_signature(scene_signature)
    for item in episode:
        recorded_seed = int(np.asarray(item.scene_seed))
        if recorded_seed != int(scene_seed):
            raise RuntimeError(
                f"online replay seed binding mismatch: {recorded_seed} != {scene_seed}"
            )
        recorded_signature = np.asarray(item.scene_signature, dtype=np.uint8)
        if not np.array_equal(recorded_signature, expected_signature):
            raise RuntimeError("online replay scene-signature binding mismatch")
    finalized = _finalize_episode_returns(
        episode,
        gamma=gamma,
        proxy_discount=proxy_discount,
        q_anchor_valid=False,
    )
    for item in finalized:
        replay.add(item)
    return finalized


def _online_train(
    algorithm,
    dataset,
    args,
    *,
    key,
    rng: np.random.Generator,
    output_dir: Path,
    metrics_path: Path,
) -> tuple[Any, dict[str, Any]]:
    import jax
    from relax.buffer.tree import TreeBuffer

    online_seed_cycle = tuple(
        int(seed)
        for seed in getattr(args, "online_seed_cycle", DEFAULT_ONLINE_SEED_CYCLE)
    )
    hybrid_gate_seeds = tuple(
        int(seed)
        for seed in getattr(args, "hybrid_gate_seeds", DEFAULT_HYBRID_GATE_SEEDS)
    )
    known_signatures = {int(seed) for seed in algorithm.scene_signatures}
    requested_seeds = set(online_seed_cycle) | set(hybrid_gate_seeds)
    missing_signatures = sorted(requested_seeds - known_signatures)
    if missing_signatures:
        raise ValueError(
            f"scene signatures missing for online/gate seeds {missing_signatures}"
        )
    seed_cycle = EpisodeSeedCycle(online_seed_cycle)
    current_seed = seed_cycle.current_seed
    statistic_seeds = tuple(sorted(requested_seeds))

    is_flow_actor = hasattr(algorithm, "flow_policy")
    replay_factory = (
        TreeBuffer.from_guardian_flow_dsac_experience
        if is_flow_actor
        else TreeBuffer.from_guardian_dsac_experience
    )
    replay = replay_factory(
        dataset.summary.obs_dim,
        dataset.summary.action_dim,
        args.online_buffer_size,
        seed=args.seed + 31_003,
    )
    critic_state = algorithm.base.state if is_flow_actor else algorithm.state
    initial_q1 = critic_state.params.q1
    initial_proxy = critic_state.params.proxy
    initial_policy = (
        algorithm.state.params.one_step
        if is_flow_actor
        else algorithm.state.params.policy
    )
    method_slug = (
        "guardian_dsac_flow_v2_4"
        if is_flow_actor
        else "guardian_dsac_v2_3_1"
    )
    counters = {
        "online_environment_steps": 0,
        "online_episodes": 0,
        "task_critic_updates": 0,
        "proxy_updates": 0,
        "actor_updates": 0,
        "hybrid_gate_evaluations": 0,
        "hybrid_gate_episode_evaluations": 0,
        "hybrid_gate_passes": 0,
        "hybrid_actor_rollbacks": 0,
        "numerical_full_rollbacks": 0,
        "deployment_gate_evaluations": 0,
        "deployable_promotions": 0,
        "ordinary_failures_without_rollback": 0,
        "uncertified_action_rejections": 0,
        "aborted_incomplete_episodes": 0,
        "online_inverse_teacher_rows": 0,
        "online_proxy_pair_rows": 0,
    }
    initial_path = output_dir / "online_initial.pkl"
    algorithm.save(str(initial_path))
    baseline = _run_isolated_evaluation(
        checkpoint=initial_path,
        output=output_dir / "online_initial_seed01.json",
        seeds=(0, 1),
        modes=("hybrid", "ordinary"),
        policy="learner",
    )
    incumbent_evaluations = (
        baseline["per_seed"]
        if algorithm.has_deployable_checkpoint
        else None
    )

    online_steps_by_seed = {seed: 0 for seed in statistic_seeds}
    online_episodes_by_seed = {seed: 0 for seed in statistic_seeds}
    aborted_episodes_by_seed = {seed: 0 for seed in statistic_seeds}
    speed_bins_by_seed = {
        seed: np.zeros((4,), dtype=np.int64) for seed in statistic_seeds
    }
    intervention_steps_by_seed = {seed: 0 for seed in statistic_seeds}
    inverse_teacher_rows_by_seed = {seed: 0 for seed in statistic_seeds}
    proxy_pair_rows_by_seed = {seed: 0 for seed in statistic_seeds}
    minimum_physical_margin_by_seed = {
        seed: float("inf") for seed in statistic_seeds
    }
    nonfinite_physical_margins_by_seed = {
        seed: 0 for seed in statistic_seeds
    }
    nonpositive_physical_margins_by_seed = {
        seed: 0 for seed in statistic_seeds
    }
    safety_failures_by_seed = {seed: set() for seed in statistic_seeds}
    replay_seed_ledger: deque[int] = deque(maxlen=args.online_buffer_size)

    episode = []
    episode_step = 0
    latest: dict[str, float] = {}
    training_safety_failures: set[str] = set()
    training_min_physical_margin = float("inf")
    training_nonfinite_physical_margins = 0
    training_nonpositive_physical_margins = 0
    speed_bin_counts = np.zeros((4,), dtype=np.int64)
    emergency_stops = 0
    next_hybrid_gate = args.hybrid_gate_every
    next_deployment_gate = args.deployment_gate_every
    deployment_history = []
    hybrid_history = []
    stopped_early = False
    consecutive_uncertified_proposals = 0
    last_hybrid_gate_step: int | None = None
    last_hybrid_evaluation: dict[str, Any] | None = None
    last_hybrid_gate: dict[str, Any] | None = None
    proximal_coefficient_min = float("inf")
    proximal_coefficient_max = -float("inf")
    proximal_coefficient_last = float("nan")
    proximal_step_max = 0
    proximal_step_last = 0

    def fresh_training_environment(current_env, next_seed: int):
        if current_env is not None:
            current_env.close()
        expected = algorithm.scene_signatures[str(next_seed)]
        next_env = _make_environment(args, seed=next_seed, mode="hybrid")
        next_obs, info = next_env.reset(seed=next_seed)
        actual = str(info.get("metasafe_scene_signature", ""))
        if actual != expected:
            next_env.close()
            raise RuntimeError(
                f"Seed{next_seed} training scene signature mismatch"
            )
        return (
            next_env,
            np.asarray(next_obs, dtype=np.float32).reshape(-1),
            expected,
        )

    env, observation, expected_signature = fresh_training_environment(
        None, current_seed
    )

    try:
        while counters["online_environment_steps"] < args.online_steps:
            flow_noise = None
            if is_flow_actor:
                flow_noise = rng.normal(0.0, 1.0, size=2).astype(np.float32)
                action = np.array(
                    algorithm.get_action(observation, flow_noise),
                    dtype=np.float32,
                    copy=True,
                ).reshape(2)
            else:
                action = np.array(
                    algorithm.get_deterministic_action(observation),
                    dtype=np.float32,
                    copy=True,
                ).reshape(2)
                action += rng.normal(
                    0.0, args.online_exploration_std, size=2
                ).astype(np.float32)
                if rng.random() < args.stratified_speed_exploration:
                    speed_bin = int(
                        counters["online_environment_steps"] % 4
                    )
                    fraction = rng.uniform(
                        speed_bin / 4.0, (speed_bin + 1) / 4.0
                    )
                    action[0] = np.float32(2.0 * fraction - 1.0)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            certificate = env.certify_target_action(action)
            if bool(certificate.get("projection_failed", False)):
                consecutive_uncertified_proposals += 1
                counters["uncertified_action_rejections"] += 1
                counters["hybrid_actor_rollbacks"] += 1
                counters["aborted_incomplete_episodes"] += 1
                aborted_episodes_by_seed[current_seed] += 1
                training_safety_failures.add("projection_failed")
                safety_failures_by_seed[current_seed].add("projection_failed")
                disposition = algorithm.reject_hybrid_safety()
                _append_jsonl(
                    metrics_path,
                    {
                        "phase": "uncertified_action_rejection",
                        **counters,
                        "current_seed": current_seed,
                        "consecutive_uncertified_proposals": (
                            consecutive_uncertified_proposals
                        ),
                        **disposition,
                    },
                )
                episode.clear()
                env, observation, expected_signature = (
                    fresh_training_environment(env, current_seed)
                )
                episode_step = 0
                if consecutive_uncertified_proposals >= 20:
                    raise RuntimeError(
                        "20 consecutive Actor proposals had no certified "
                        "target; refusing an unsafe or non-progressing run"
                    )
                continue
            consecutive_uncertified_proposals = 0
            authority = _audit_low_level_cbf_authority(
                env,
                action,
                certificate,
                trajectory_tolerance=args.cbf_trajectory_tolerance,
                derive_teacher=args.online_inverse_teacher,
            )
            next_observation, reward, terminated, truncated, info = (
                env.execute_certified_target(certificate)
            )
            counters["online_environment_steps"] += 1
            online_steps_by_seed[current_seed] += 1
            episode_step += 1
            truncated = bool(
                truncated
                or episode_step >= args.max_episode_steps
                or counters["online_environment_steps"] >= args.online_steps
            )
            transition = _step_dsac_experience(
                observation,
                action,
                reward,
                terminated,
                truncated,
                next_observation,
                certificate,
                info,
                soft_pair_weight=0.25,
                authority_audit=authority,
                scene_seed=current_seed,
                scene_signature=expected_signature,
                action_semantics_version=TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
            )
            if is_flow_actor:
                from relax.offline.guardian_dsac_multiseed_dataset import (
                    guardian_dsac_to_flow_experience,
                )

                transition = guardian_dsac_to_flow_experience(
                    transition,
                    flow_noise=flow_noise,
                    flow_noise_valid=np.bool_(True),
                )
            episode.append(transition)
            step_safety_failures = _online_safety_failures(info)
            training_safety_failures.update(step_safety_failures)
            safety_failures_by_seed[current_seed].update(step_safety_failures)
            physical_margin = float(
                info.get("mpc_control", {}).get(
                    "physical_body_sat_margin", np.nan
                )
            )
            if np.isfinite(physical_margin):
                training_min_physical_margin = min(
                    training_min_physical_margin, physical_margin
                )
                training_nonpositive_physical_margins += int(
                    physical_margin <= 0.0
                )
                minimum_physical_margin_by_seed[current_seed] = min(
                    minimum_physical_margin_by_seed[current_seed],
                    physical_margin,
                )
                nonpositive_physical_margins_by_seed[current_seed] += int(
                    physical_margin <= 0.0
                )
            else:
                training_nonfinite_physical_margins += 1
                nonfinite_physical_margins_by_seed[current_seed] += 1
            intervention_steps_by_seed[current_seed] += int(
                bool(transition.intervention)
            )
            teacher_row = int(bool(transition.teacher_valid))
            pair_row = int(bool(transition.proxy_pair_valid))
            counters["online_inverse_teacher_rows"] += teacher_row
            counters["online_proxy_pair_rows"] += pair_row
            inverse_teacher_rows_by_seed[current_seed] += teacher_row
            proxy_pair_rows_by_seed[current_seed] += pair_row
            emergency_stops += int(bool(transition.emergency_stop))
            if not bool(transition.emergency_stop):
                fraction = float(
                    np.clip(
                        (float(transition.executed_target_speed)
                         - float(transition.reachable_speed_lower))
                        / (
                            float(transition.reachable_speed_upper)
                            - float(transition.reachable_speed_lower)
                            + 1e-6
                        ),
                        0.0,
                        1.0,
                    )
                )
                bin_index = min(int(fraction * 4.0), 3)
                speed_bin_counts[bin_index] += 1
                speed_bins_by_seed[current_seed][bin_index] += 1
            done = bool(terminated or truncated)
            observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
            if done:
                finalized = _store_complete_online_episode(
                    replay,
                    episode,
                    scene_seed=current_seed,
                    scene_signature=expected_signature,
                    gamma=args.discount,
                    proxy_discount=args.proxy_discount,
                )
                replay_seed_ledger.extend(
                    [current_seed] * len(finalized)
                )
                counters["online_episodes"] += 1
                online_episodes_by_seed[current_seed] += 1
                episode.clear()
                seed_cycle.complete_episode()

            progress = counters["online_environment_steps"] / max(args.online_steps, 1)
            offline_ratio = args.offline_ratio_initial + min(progress, 1.0) * (
                args.offline_ratio_final - args.offline_ratio_initial
            )
            # The offline half of the mixed replay is already complete-episode
            # valid, so online updates may start after the environment-step
            # warmup even before the first long route finishes.  Until then
            # the sampler naturally returns an all-offline batch; complete
            # Seed0/Seed1 episodes subsequently enter the shared replay.
            if counters["online_environment_steps"] >= args.learning_starts:
                for gradient_index in range(args.gradient_steps):
                    batch = _sample_mixed_batch(
                        dataset,
                        replay,
                        batch_size=args.batch_size,
                        offline_ratio=offline_ratio,
                        rng=rng,
                        flow=is_flow_actor,
                    )
                    key, update_key = jax.random.split(key)
                    update_actor = bool(
                        (counters["online_environment_steps"] + gradient_index)
                        % args.policy_delay
                        == 0
                    )
                    latest = algorithm.online_joint_update(
                        update_key, batch, update_actor=update_actor
                    )
                    counters["task_critic_updates"] += 1
                    counters["proxy_updates"] += 1
                    counters["actor_updates"] += int(update_actor)
                    if update_actor and "proximal_coefficient" in latest:
                        coefficient = float(latest["proximal_coefficient"])
                        prox_step = int(round(float(latest["proximal_step"])))
                        proximal_coefficient_min = min(
                            proximal_coefficient_min, coefficient
                        )
                        proximal_coefficient_max = max(
                            proximal_coefficient_max, coefficient
                        )
                        proximal_coefficient_last = coefficient
                        proximal_step_max = max(proximal_step_max, prox_step)
                        proximal_step_last = prox_step
                    disaster = algorithm.numerical_disaster_reasons(
                        metrics=latest,
                        observation=np.asarray(batch.obs[:16]),
                        action=np.asarray(batch.action[:16]),
                    )
                    if disaster:
                        disposition = algorithm.reject_numerical_disaster()
                        counters["numerical_full_rollbacks"] += 1
                        _append_jsonl(
                            metrics_path,
                            {
                                "phase": "numerical_disaster",
                                **counters,
                                "reasons": disaster,
                                **disposition,
                            },
                        )
                        break

            step = counters["online_environment_steps"]
            if step % args.log_every == 0:
                _append_jsonl(
                    metrics_path,
                    {
                        "phase": f"online_joint_{method_slug}",
                        **counters,
                        "offline_ratio": offline_ratio,
                        "online_replay_size": len(replay),
                        "speed_bin_counts": speed_bin_counts.tolist(),
                        "online_steps_by_seed": {
                            str(seed): online_steps_by_seed[seed]
                            for seed in statistic_seeds
                        },
                        "online_episodes_by_seed": {
                            str(seed): online_episodes_by_seed[seed]
                            for seed in statistic_seeds
                        },
                        **latest,
                        **_grouped_actor_diagnostics(latest),
                    },
                )

            if step >= next_hybrid_gate:
                while next_hybrid_gate <= step:
                    next_hybrid_gate += args.hybrid_gate_every
                candidate = output_dir / f"hybrid_candidate_{step:08d}.pkl"
                algorithm.save(str(candidate))
                evaluation = _run_isolated_evaluation(
                    checkpoint=candidate,
                    output=output_dir / f"hybrid_candidate_{step:08d}.json",
                    seeds=hybrid_gate_seeds,
                    modes=("hybrid",),
                    policy="learner",
                )
                hybrid_eval = {
                    str(seed): evaluation["per_seed"][str(seed)]["hybrid"]
                    for seed in hybrid_gate_seeds
                }
                gate = apply_multi_seed_hybrid_learning_gate(
                    algorithm,
                    hybrid_eval,
                    required_seeds=hybrid_gate_seeds,
                )
                counters["hybrid_gate_evaluations"] += 1
                counters["hybrid_gate_episode_evaluations"] += len(
                    hybrid_gate_seeds
                )
                counters["hybrid_gate_passes"] += int(gate["passed"])
                counters["hybrid_actor_rollbacks"] += int(
                    gate.get("rollback") == "actor_only"
                )
                counters["numerical_full_rollbacks"] += int(
                    gate.get("rollback") == "hybrid_safe_full_snapshot"
                )
                last_hybrid_gate_step = step
                last_hybrid_evaluation = evaluation
                last_hybrid_gate = gate
                hybrid_history.append(
                    {"step": step, "gate": gate, "evaluation": hybrid_eval}
                )
                _append_jsonl(
                    metrics_path,
                    {
                        "phase": "hybrid_learning_gate",
                        **counters,
                        "gate": gate,
                    },
                )
                algorithm.save(str(output_dir / f"checkpoint_{step:08d}.pkl"))

            if step >= next_deployment_gate:
                while next_deployment_gate <= step:
                    next_deployment_gate += args.deployment_gate_every
                candidate = output_dir / f"deployment_candidate_{step:08d}.pkl"
                algorithm.save(str(candidate))
                evaluation = _run_isolated_evaluation(
                    checkpoint=candidate,
                    output=output_dir / f"deployment_candidate_{step:08d}.json",
                    seeds=(0, 1),
                    modes=("hybrid", "ordinary"),
                    policy="learner",
                )
                gate = apply_deployment_gate(
                    algorithm,
                    candidate_by_seed=evaluation["per_seed"],
                    incumbent_by_seed=incumbent_evaluations,
                    required_seeds=(0, 1),
                    config=GuardianDeploymentGateConfig(
                        interval_steps=args.deployment_gate_every
                    ),
                )
                counters["deployment_gate_evaluations"] += 1
                counters["deployable_promotions"] += int(gate["promote"])
                counters["ordinary_failures_without_rollback"] += int(
                    not gate["promote"]
                )
                if gate["promote"]:
                    incumbent_evaluations = evaluation["per_seed"]
                deployment_history.append({"step": step, "gate": gate})
                _append_jsonl(metrics_path, {"phase": "full_deployment_gate", **counters, "gate": gate})
                algorithm.save(str(output_dir / f"checkpoint_{step:08d}.pkl"))
                if gate["promote"] and args.stop_after_first_deployable:
                    stopped_early = True
                    break

            if done and counters["online_environment_steps"] < args.online_steps:
                current_seed = seed_cycle.current_seed
                env, observation, expected_signature = (
                    fresh_training_environment(env, current_seed)
                )
                episode_step = 0
    finally:
        env.close()

    # Reuse a scheduled endpoint gate (for example the 100k/4k boundary) so
    # the formal 100k run evaluates exactly 25 x 2 Hybrid episodes.  Runs that
    # stop between gate boundaries receive one final two-seed gate here.
    final_pre_path = output_dir / "online_final_learner_pre_gate.pkl"
    algorithm.save(str(final_pre_path))
    final_step = counters["online_environment_steps"]
    final_gate_reused = bool(
        last_hybrid_gate_step == final_step
        and last_hybrid_evaluation is not None
        and last_hybrid_gate is not None
    )
    if final_gate_reused:
        final_hybrid = last_hybrid_evaluation
        final_gate = last_hybrid_gate
        _write_json(output_dir / "online_final_seed01_hybrid.json", final_hybrid)
    else:
        final_hybrid = _run_isolated_evaluation(
            checkpoint=final_pre_path,
            output=output_dir / "online_final_seed01_hybrid.json",
            seeds=hybrid_gate_seeds,
            modes=("hybrid",),
            policy="learner",
        )
        final_hybrid_by_seed = {
            str(seed): final_hybrid["per_seed"][str(seed)]["hybrid"]
            for seed in hybrid_gate_seeds
        }
        final_gate = apply_multi_seed_hybrid_learning_gate(
            algorithm,
            final_hybrid_by_seed,
            required_seeds=hybrid_gate_seeds,
        )
        counters["hybrid_gate_evaluations"] += 1
        counters["hybrid_gate_episode_evaluations"] += len(hybrid_gate_seeds)
        counters["hybrid_gate_passes"] += int(final_gate["passed"])
        counters["hybrid_actor_rollbacks"] += int(
            final_gate.get("rollback") == "actor_only"
        )
        counters["numerical_full_rollbacks"] += int(
            final_gate.get("rollback") == "hybrid_safe_full_snapshot"
        )
        hybrid_history.append(
            {
                "step": final_step,
                "gate": final_gate,
                "evaluation": final_hybrid_by_seed,
                "terminal_gate": True,
            }
        )

    final_path = output_dir / f"final_{method_slug}.pkl"
    algorithm.save(str(final_path))

    total_speed = max(int(np.sum(speed_bin_counts)), 1)
    replay_seed_counts = Counter(replay_seed_ledger)
    replay_total = max(len(replay_seed_ledger), 1)
    total_online_steps = max(sum(online_steps_by_seed.values()), 1)

    def finite_margin(seed: int) -> float:
        margin = minimum_physical_margin_by_seed[seed]
        return margin if np.isfinite(margin) else float("nan")

    online_summary = {
        **counters,
        "stopped_early": stopped_early,
        "online_seed_cycle": list(online_seed_cycle),
        "hybrid_gate_seeds": list(hybrid_gate_seeds),
        "online_replay_size": len(replay),
        "online_steps_by_seed": {
            str(seed): online_steps_by_seed[seed] for seed in statistic_seeds
        },
        "online_transition_fraction_by_seed": {
            str(seed): online_steps_by_seed[seed] / total_online_steps
            for seed in statistic_seeds
        },
        "online_episodes_by_seed": {
            str(seed): online_episodes_by_seed[seed] for seed in statistic_seeds
        },
        "aborted_incomplete_episodes_by_seed": {
            str(seed): aborted_episodes_by_seed[seed]
            for seed in statistic_seeds
        },
        "online_replay_fraction_by_seed": {
            str(seed): replay_seed_counts.get(seed, 0) / replay_total
            for seed in statistic_seeds
        },
        "speed_bins_by_seed": {
            str(seed): speed_bins_by_seed[seed].tolist()
            for seed in statistic_seeds
        },
        "intervention_rate_by_seed": {
            str(seed): intervention_steps_by_seed[seed]
            / max(online_steps_by_seed[seed], 1)
            for seed in statistic_seeds
        },
        "online_inverse_teacher_enabled": bool(args.online_inverse_teacher),
        "online_inverse_teacher_rows_by_seed": {
            str(seed): inverse_teacher_rows_by_seed[seed]
            for seed in statistic_seeds
        },
        "online_proxy_pair_rows_by_seed": {
            str(seed): proxy_pair_rows_by_seed[seed]
            for seed in statistic_seeds
        },
        "minimum_physical_margin_by_seed": {
            str(seed): finite_margin(seed) for seed in statistic_seeds
        },
        "training_hybrid_safety_failures_by_seed": {
            str(seed): sorted(safety_failures_by_seed[seed])
            for seed in statistic_seeds
        },
        "nonpositive_physical_margin_steps_by_seed": {
            str(seed): nonpositive_physical_margins_by_seed[seed]
            for seed in statistic_seeds
        },
        "nonfinite_physical_margin_steps_by_seed": {
            str(seed): nonfinite_physical_margins_by_seed[seed]
            for seed in statistic_seeds
        },
        "training_hybrid_safety_failures": sorted(training_safety_failures),
        "training_min_physical_body_sat_margin": (
            training_min_physical_margin
            if np.isfinite(training_min_physical_margin)
            else float("nan")
        ),
        "training_nonpositive_physical_margin_steps": (
            training_nonpositive_physical_margins
        ),
        "training_nonfinite_physical_margin_steps": (
            training_nonfinite_physical_margins
        ),
        "speed_bin_counts": speed_bin_counts.tolist(),
        "speed_bin_fractions": (speed_bin_counts / total_speed).tolist(),
        "all_speed_bins_visited": bool(np.all(speed_bin_counts > 0)),
        "emergency_stop_count": emergency_stops,
        "proximal_coefficient_min": (
            proximal_coefficient_min
            if np.isfinite(proximal_coefficient_min)
            else float("nan")
        ),
        "proximal_coefficient_max": (
            proximal_coefficient_max
            if np.isfinite(proximal_coefficient_max)
            else float("nan")
        ),
        "proximal_coefficient_last": proximal_coefficient_last,
        "proximal_step_max": proximal_step_max,
        "proximal_step_last_observed": proximal_step_last,
        "final_actor_diagnostics": {
            **{
                name: latest[name]
                for name in (
                    "weighted_sac_loss",
                    "weighted_proxy_loss",
                    "weighted_proximal_loss",
                    "proximal_speed_drift",
                    "proximal_lateral_drift",
                    "proximal_coefficient",
                    "proximal_step",
                    "p_policy_mean",
                )
                if name in latest
            },
            **_grouped_actor_diagnostics(latest),
        },
        "task_critic_updated": _tree_changed(
            initial_q1,
            algorithm.base.state.params.q1
            if is_flow_actor
            else algorithm.state.params.q1,
        ),
        "proxy_updated": _tree_changed(
            initial_proxy,
            algorithm.base.state.params.proxy
            if is_flow_actor
            else algorithm.state.params.proxy,
        ),
        "actor_updated": _tree_changed(
            initial_policy,
            algorithm.state.params.one_step
            if is_flow_actor
            else algorithm.state.params.policy,
        ),
        "hybrid_history": hybrid_history,
        "final_hybrid": final_hybrid,
        "final_hybrid_gate": final_gate,
        "final_hybrid_gate_reused_scheduled_endpoint": final_gate_reused,
        "deployment_history": deployment_history,
        "baseline": baseline,
        "has_deployable_checkpoint": algorithm.has_deployable_checkpoint,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": _sha256(final_path),
    }
    return key, online_summary


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import jax
    import jax.numpy as jnp
    from relax.offline.metasafe_guardian_dataset import (
        GuardianDSACMetaSafeDataset,
    )

    started = time.time()
    backend = jax.default_backend()
    if args.require_gpu and backend != "gpu":
        raise RuntimeError(f"Guardian-DSAC formal training requires GPU, got {backend}")
    dataset_path = args.dataset.expanduser().resolve()
    dataset = GuardianDSACMetaSafeDataset(
        dataset_path,
        gamma=args.discount,
        proxy_discount=args.proxy_discount,
        split_seed=args.seed,
        expected_sha256=args.expected_dataset_sha256 or None,
        expected_scene_seed=0,
        expected_lookahead_distance=args.lookahead,
        expected_max_command_speed=args.max_command_speed,
        expected_speed_mapping="state_reachable_speed_linear_v1",
        expected_mpc_horizon=args.horizon,
        expected_min_command_speed=args.min_command_speed,
        expected_episode_step_limit=args.max_episode_steps,
        expected_speed_sampling_profile=args.expected_speed_sampling_profile,
    )
    scene_signatures = _predeclare_scene_signatures(args, dataset)
    key = jax.random.PRNGKey(args.seed)
    key, network_key = jax.random.split(key)
    algorithm, hidden_sizes = _create_algorithm(
        args, dataset, scene_signatures, network_key
    )

    run_name = args.run_name or (
        ("smoke_" if args.component_smoke else "")
        + f"seed01_mixed_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_root.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"
    config = {
        "format": "guardian_dsac_v2_3_1_run_config_v1",
        "phase": args.phase,
        "seed": args.seed,
        "action_semantics_version": TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        "scene_seeds": [0, 1],
        "scene_signatures": scene_signatures,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset.sha256,
        "observation_dim": dataset.summary.obs_dim,
        "actor_observation_dim": dataset.summary.obs_dim,
        "action_dim": dataset.summary.action_dim,
        "hidden_sizes": list(hidden_sizes),
        "init_alpha": args.init_alpha,
        "algorithm_config": asdict(algorithm.config),
        "expected_episode_step_limit": args.max_episode_steps,
        "expected_speed_sampling_profile": args.expected_speed_sampling_profile,
        "online_seed_cycle": list(args.online_seed_cycle),
        "hybrid_gate_seeds": list(args.hybrid_gate_seeds),
        "online_inverse_teacher": bool(args.online_inverse_teacher),
        "environment": _environment_config(args),
        "budgets": {
            "bc_gradient_steps": args.bc_gradient_steps,
            "critic_gradient_steps": args.critic_gradient_steps,
            "proxy_gradient_steps": args.proxy_gradient_steps,
            "online_steps": args.online_steps,
            "hybrid_gate_every": args.hybrid_gate_every,
            "deployment_gate_every": args.deployment_gate_every,
        },
        "component_smoke": args.component_smoke,
        "start_from_deployable": args.start_from_deployable,
        "offline_checkpoint_source": (
            str(args.offline_checkpoint.expanduser().resolve())
            if args.offline_checkpoint is not None
            else None
        ),
        "backend": backend,
        "devices": [str(device) for device in jax.devices()],
    }
    _write_json(output_dir / "config.json", config)

    offline_summary = None
    offline_checkpoint = output_dir / "offline_guardian_dsac_v2_3_1.pkl"
    if args.offline_checkpoint is not None:
        source = args.offline_checkpoint.expanduser().resolve()
        algorithm.load(str(source))
        if not algorithm.offline_reference_ready:
            raise RuntimeError("loaded offline checkpoint has no frozen reference")
        if args.start_from_deployable:
            if not algorithm.has_deployable_checkpoint:
                raise RuntimeError(
                    "--start-from-deployable checkpoint has no certified state"
                )
            deployable = algorithm.deployable_checkpoint
            algorithm.state = jax.tree_util.tree_map(
                lambda value: jnp.array(value, copy=True), deployable["state"]
            )
            algorithm.hybrid_safe_snapshot = jax.tree_util.tree_map(
                lambda value: jnp.array(value, copy=True), algorithm.state
            )
            algorithm.online_reference_step = int(
                deployable.get(
                    "online_reference_step", algorithm.online_reference_step
                )
            )
            algorithm._safe_online_reference_step = int(
                algorithm.online_reference_step
            )
        offline_summary = {
            "loaded": True,
            "source": str(source),
            "source_sha256": _sha256(source),
            "started_from_deployable": bool(args.start_from_deployable),
            "speed_coverage": dataset.speed_coverage(),
            "validation": _offline_validation(algorithm, dataset),
            "support_thresholds": asdict(algorithm.offline_support_thresholds),
        }
        algorithm.save(str(offline_checkpoint))
    elif args.phase in ("offline", "full"):
        key, offline_summary = _offline_train(
            algorithm,
            dataset,
            args,
            key=key,
            metrics_path=metrics_path,
        )
        algorithm.save(str(offline_checkpoint))
        offline_summary.update(
            checkpoint=str(offline_checkpoint),
            checkpoint_sha256=_sha256(offline_checkpoint),
        )
        _write_json(output_dir / "offline_summary.json", offline_summary)
    else:
        raise RuntimeError("online phase did not load an offline checkpoint")

    online_summary = None
    if args.phase in ("online", "full") and args.online_steps > 0:
        rng = np.random.default_rng(args.seed + 202_608_02)
        key, online_summary = _online_train(
            algorithm,
            dataset,
            args,
            key=key,
            rng=rng,
            output_dir=output_dir,
            metrics_path=metrics_path,
        )
        _write_json(output_dir / "online_summary.json", online_summary)

    final_path = output_dir / "final_guardian_dsac_v2_3_1.pkl"
    if not final_path.exists():
        algorithm.save(str(final_path))
    summary = {
        "format": "guardian_dsac_v2_3_1_training_summary_v1",
        "method": "Guardian-DSAC V2.3.1",
        "action_semantics_version": TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        "distributional_loss": args.distributional_loss,
        "soft_reference_enabled": bool(
            args.reference_weight_initial > 0.0
            or args.reference_weight_final > 0.0
        ),
        "dataset": asdict(dataset.summary),
        "dataset_speed_coverage": dataset.speed_coverage(),
        "offline": offline_summary,
        "online": online_summary,
        "has_deployable_checkpoint": algorithm.has_deployable_checkpoint,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": _sha256(final_path),
        "wall_time_seconds": time.time() - started,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
