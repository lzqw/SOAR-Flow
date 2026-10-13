#!/usr/bin/env python3
"""Run the simplified 10-Seed Guardian-DSAC V2.3.1 main experiment."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.offline.guardian_dsac_multiseed_dataset import (  # noqa: E402
    GuardianDSACMultiSeedDataset,
)
from relax.offline.metasafe_guardian_dataset import (  # noqa: E402
    GuardianDSACMetaSafeDataset,
)
from relax.utils.action_semantics import (  # noqa: E402
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)
from relax.utils.guardian_deployment_gate import (  # noqa: E402
    ordinary_hard_safety_failures,
)
from relax.utils.guardian_learning_gate import (  # noqa: E402
    apply_multi_seed_hybrid_learning_gate,
    hybrid_hard_safety_failures,
)
from scripts.train_guardian_calql_metasafe import (  # noqa: E402
    _audit_low_level_cbf_authority,
)
from scripts.train_guardian_dsac_metasafe import (  # noqa: E402
    EpisodeSeedCycle,
    _algorithm_config,
    _append_jsonl,
    _environment_config,
    _grouped_actor_diagnostics,
    _make_environment,
    _offline_validation,
    _online_safety_failures,
    _run_isolated_evaluation,
    _sample_mixed_batch,
    _sha256,
    _step_dsac_experience,
    _store_complete_online_episode,
    _tree_changed,
    _write_json,
)


OFFLINE_SEEDS = (0, 1, 2, 3, 4)
ALL_SEEDS = tuple(range(10))
ONLINE_EPISODE_SCHEDULE = (
    5,
    6,
    7,
    8,
    0,
    9,
    5,
    6,
    7,
    1,
    8,
    9,
    5,
    6,
    2,
    7,
    8,
    9,
    5,
    3,
    6,
    7,
    8,
    9,
    4,
)
HYBRID_GATE_PAIRS = ((0, 5), (1, 6), (2, 7), (3, 8), (4, 9))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/metasafe_cbf/guardian_dsac_multiseed_v1/manifest.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/guardian_dsac_multiseed")
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--phase", choices=("offline", "online", "full"), default="full"
    )
    parser.add_argument("--offline-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bc-gradient-steps", type=int, default=80_000)
    parser.add_argument("--critic-gradient-steps", type=int, default=150_000)
    parser.add_argument("--proxy-gradient-steps", type=int, default=80_000)
    parser.add_argument("--support-calibration-rows-per-seed", type=int, default=8192)
    parser.add_argument("--online-steps", type=int, default=150_000)
    parser.add_argument("--online-buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--hybrid-gate-every", type=int, default=10_000)
    parser.add_argument("--full-evaluation-every", type=int, default=50_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument(
        "--require-gpu", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--component-smoke", action="store_true")

    parser.add_argument("--init-alpha", type=float, default=0.05)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--proxy-discount", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--proxy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha-learning-rate", type=float, default=3e-4)
    parser.add_argument("--calql-weight", type=float, default=1.0)
    parser.add_argument("--reference-weight-initial", type=float, default=0.0)
    parser.add_argument("--reference-weight-final", type=float, default=0.0)
    parser.add_argument("--reference-decay-steps", type=int, default=100_000)
    parser.add_argument("--reference-std-weight", type=float, default=0.1)
    # Accepted V2.3.1 calibration; the dual-Gaussian and Actor objective stay intact.
    parser.add_argument("--proxy-raw-pair-weight", type=float, default=8.0)
    parser.add_argument("--proxy-pair-gap-weight", type=float, default=8.0)
    parser.add_argument("--proxy-actor-weight", type=float, default=0.15)
    parser.add_argument(
        "--proxy-actor-objective",
        choices=("probability", "softplus_logit"),
        default="softplus_logit",
    )
    parser.add_argument("--proximal-weight-initial", type=float, default=1.0)
    parser.add_argument("--proximal-weight-min", type=float, default=0.05)
    parser.add_argument("--proximal-decay-steps", type=int, default=20_000)
    parser.add_argument("--proximal-speed-weight", type=float, default=0.5)
    parser.add_argument("--proximal-lateral-weight", type=float, default=1.0)
    parser.add_argument("--offline-ratio-initial", type=float, default=0.50)
    parser.add_argument("--offline-ratio-final", type=float, default=0.20)
    parser.add_argument("--online-exploration-std", type=float, default=0.04)
    parser.add_argument("--stratified-speed-exploration", type=float, default=0.10)
    parser.add_argument(
        "--online-inverse-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--min-command-speed", type=float, default=1.0)
    parser.add_argument("--max-command-speed", type=float, default=None)
    parser.add_argument("--lookahead", type=float, default=10.0)
    parser.add_argument("--lateral-scale", type=float, default=3.5)
    parser.add_argument("--target-velocity", type=float, default=None)
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

    args.distributional_loss = "hdsac"
    if args.phase == "online" and args.offline_checkpoint is None:
        parser.error("--phase online requires --offline-checkpoint")
    integer_options = (
        "batch_size",
        "support_calibration_rows_per_seed",
        "online_buffer_size",
        "learning_starts",
        "gradient_steps",
        "policy_delay",
        "hybrid_gate_every",
        "full_evaluation_every",
        "log_every",
        "reference_decay_steps",
        "proximal_decay_steps",
        "sqp_steps",
        "admm_steps",
        "max_obstacles",
    )
    if any(getattr(args, name) < 1 for name in integer_options):
        parser.error("positive integer option is non-positive")
    for name in (
        "bc_gradient_steps",
        "critic_gradient_steps",
        "proxy_gradient_steps",
        "online_steps",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_episode_steps != 1200:
        parser.error("the multi-Seed main experiment keeps horizon=1200")
    if not 0.0 <= args.offline_ratio_final <= args.offline_ratio_initial <= 1.0:
        parser.error("invalid offline replay ratios")
    if not 0.0 <= args.stratified_speed_exploration <= 1.0:
        parser.error("stratified speed exploration must be in [0,1]")
    if args.component_smoke:
        args.bc_gradient_steps = min(args.bc_gradient_steps, 2)
        args.critic_gradient_steps = min(args.critic_gradient_steps, 2)
        args.proxy_gradient_steps = min(args.proxy_gradient_steps, 2)
        args.online_steps = min(args.online_steps, 200)
        args.batch_size = min(args.batch_size, 64)
        args.learning_starts = min(args.learning_starts, 32)
        args.hybrid_gate_every = min(args.hybrid_gate_every, 100)
        args.full_evaluation_every = min(args.full_evaluation_every, 200)
        args.log_every = min(args.log_every, 20)
        args.support_calibration_rows_per_seed = min(
            args.support_calibration_rows_per_seed, 64
        )
    return args


def _load_multiseed_dataset(args):
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "guardian_dsac_multiseed_collection_v1":
        raise ValueError("unsupported multi-Seed collection manifest")
    if tuple(int(seed) for seed in manifest.get("offline_seeds", [])) != OFFLINE_SEEDS:
        raise ValueError("formal manifest must contain offline Seed0--4")
    if int(manifest.get("episodes_per_seed", -1)) != 300:
        raise ValueError("formal manifest must contain 300 episodes per Seed")
    if int(manifest.get("max_episode_steps", -1)) != 1200:
        raise ValueError("formal manifest must retain the 1200-step horizon")
    recorded_speed = float(manifest["max_command_speed"])
    if args.max_command_speed is None:
        args.max_command_speed = recorded_speed
    if args.target_velocity is None:
        args.target_velocity = args.max_command_speed
    if not np.isclose(args.max_command_speed, recorded_speed):
        raise ValueError("training max command speed differs from the manifest")
    if not np.isclose(args.target_velocity, recorded_speed):
        raise ValueError("training target velocity differs from the manifest")

    rows = {int(row["seed"]): row for row in manifest.get("datasets", [])}
    if tuple(sorted(rows)) != OFFLINE_SEEDS:
        raise ValueError("manifest dataset rows do not cover Seed0--4 exactly")
    datasets = []
    for seed in OFFLINE_SEEDS:
        row = rows[seed]
        path = Path(row["dataset"]).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        dataset = GuardianDSACMetaSafeDataset(
            path.resolve(),
            gamma=args.discount,
            proxy_discount=args.proxy_discount,
            split_seed=args.seed,
            expected_sha256=str(row["dataset_sha256"]),
            expected_scene_seed=seed,
            expected_scene_signature=str(row["scene_signature"]),
            expected_lookahead_distance=args.lookahead,
            expected_max_command_speed=args.max_command_speed,
            expected_speed_mapping="state_reachable_speed_linear_v1",
            expected_mpc_horizon=args.horizon,
            expected_min_command_speed=args.min_command_speed,
            expected_episode_step_limit=1200,
            expected_speed_sampling_profile="fast_v231",
        )
        if dataset.summary.episodes != 300:
            raise ValueError(f"Seed{seed} loader did not find 300 episodes")
        if not dataset.speed_coverage()["all_bins_visited"]:
            raise ValueError(f"Seed{seed} does not cover all four speed bins")
        datasets.append(dataset)
    combined = GuardianDSACMultiSeedDataset(
        datasets, seed=args.seed, expected_seeds=OFFLINE_SEEDS
    )
    return manifest_path, manifest, combined


def _predeclare_scene_signatures(args, dataset) -> dict[str, str]:
    from relax.env.drive.metasafe import make_fixed_metasafe_env

    signatures = dict(dataset.scene_signatures)
    for seed in ALL_SEEDS:
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
            signature = str(info["metasafe_scene_signature"])
        finally:
            env.close()
        recorded = signatures.get(str(seed))
        if recorded is not None and recorded != signature:
            raise RuntimeError(f"Seed{seed} scene signature differs from its dataset")
        signatures[str(seed)] = signature
    return signatures


def _create_algorithm(args, dataset, scene_signatures, key):
    from relax.algorithm.guardian_dsac_v2_3 import GuardianDSACV23
    from relax.network.safe_sac_jax import create_guardian_dsac_net

    hidden_sizes = tuple(
        int(value.strip()) for value in args.hidden_sizes.split(",") if value.strip()
    )
    if not hidden_sizes or any(value < 1 for value in hidden_sizes):
        raise ValueError("hidden sizes must be positive integers")
    network, params = create_guardian_dsac_net(
        key,
        critic_obs_dim=dataset.obs_dim,
        actor_obs_dim=dataset.obs_dim,
        act_dim=dataset.action_dim,
        hidden_sizes=hidden_sizes,
        init_alpha=args.init_alpha,
    )
    algorithm = GuardianDSACV23(
        network,
        params,
        config=_algorithm_config(args),
        action_semantics_version=TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        scene_seeds=ALL_SEEDS,
        scene_signatures=scene_signatures,
        dataset_sha256=dataset.sha256,
    )
    return algorithm, hidden_sizes


def _multiseed_validation(algorithm, dataset) -> dict[str, Any]:
    per_seed = {
        str(seed): _offline_validation(algorithm, dataset.by_seed[seed])
        for seed in dataset.scene_seeds
    }
    names = sorted({name for row in per_seed.values() for name in row})
    mean = {
        name: float(np.mean([row[name] for row in per_seed.values()])) for name in names
    }
    return {"mean_across_seeds": mean, "per_seed": per_seed}


def _offline_train(algorithm, dataset, args, *, key, metrics_path: Path):
    import jax

    initial_policy = algorithm.state.params.policy
    initial_q1 = algorithm.state.params.q1
    initial_proxy = algorithm.state.params.proxy
    latest = {}
    stages = (
        ("offline_bc", args.bc_gradient_steps),
        ("offline_calql_dsac", args.critic_gradient_steps),
        ("offline_proxy", args.proxy_gradient_steps),
    )
    for phase, budget in stages:
        for step in range(budget):
            if phase == "offline_bc":
                batch = dataset.sample(
                    args.batch_size, split="train", teacher_fraction=0.25, to_jax=True
                )
                key, update_key = jax.random.split(key)
                latest = algorithm.pretrain_bc(update_key, batch)
            elif phase == "offline_calql_dsac":
                batch = dataset.sample(args.batch_size, split="train", to_jax=True)
                key, update_key = jax.random.split(key)
                latest = algorithm.offline_critic_update(update_key, batch)
            else:
                batch = dataset.sample(
                    args.batch_size, split="train", pair_fraction=0.50, to_jax=True
                )
                latest = algorithm.proxy_update(batch)
            if (step + 1) % args.log_every == 0 or step + 1 == budget:
                row = {"phase": phase, "step": step + 1, **latest}
                _append_jsonl(metrics_path, row)
                print(f"[multiseed] {phase}: {step + 1}/{budget}", flush=True)

    validation = dataset.validation_reference_batch(
        rows_per_seed=args.support_calibration_rows_per_seed, to_jax=True
    )
    thresholds = algorithm.mark_offline_reference(validation)
    return key, {
        "bc_steps": args.bc_gradient_steps,
        "critic_steps": args.critic_gradient_steps,
        "proxy_steps": args.proxy_gradient_steps,
        "actor_updated": _tree_changed(initial_policy, algorithm.state.params.policy),
        "task_critic_updated": _tree_changed(initial_q1, algorithm.state.params.q1),
        "proxy_updated": _tree_changed(initial_proxy, algorithm.state.params.proxy),
        "support_calibration_rows_per_seed": args.support_calibration_rows_per_seed,
        "support_thresholds": asdict(thresholds),
        "validation": _multiseed_validation(algorithm, dataset),
        "speed_coverage": dataset.speed_coverage(),
    }


def _evaluation_summary(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    hybrid_safe_count = 0
    ordinary_safe_count = 0
    ordinary_returns = []
    hybrid_interventions = []
    for seed in ALL_SEEDS:
        modes = evaluation["per_seed"][str(seed)]
        hybrid = modes["hybrid"]
        ordinary = modes["ordinary"]
        hybrid_failures = hybrid_hard_safety_failures(hybrid)
        ordinary_failures = ordinary_hard_safety_failures(ordinary)
        hybrid_safe = not hybrid_failures
        ordinary_safe = not ordinary_failures
        hybrid_safe_count += int(hybrid_safe)
        ordinary_safe_count += int(ordinary_safe)
        if ordinary_safe:
            ordinary_returns.append(float(ordinary["mean_return"]))
        hybrid_interventions.append(float(hybrid["actual_intervention_rate"]))
        rows[str(seed)] = {
            "role": "offline/retention" if seed in OFFLINE_SEEDS else "novel-online",
            "hybrid_safe": hybrid_safe,
            "hybrid_failures": hybrid_failures,
            "hybrid_intervention": float(hybrid["actual_intervention_rate"]),
            "hybrid_return": float(hybrid["mean_return"]),
            "ordinary_safe": ordinary_safe,
            "ordinary_failures": ordinary_failures,
            "ordinary_return": float(ordinary["mean_return"]),
        }
    mean_return = float(np.mean(ordinary_returns)) if ordinary_returns else None
    mean_intervention = float(np.mean(hybrid_interventions))
    return {
        "hybrid_safe_seeds": hybrid_safe_count,
        "ordinary_safe_seeds": ordinary_safe_count,
        "ordinary_safe_rate": ordinary_safe_count / len(ALL_SEEDS),
        "mean_ordinary_return_among_safe_seeds": mean_return,
        "mean_hybrid_intervention": mean_intervention,
        "per_seed": rows,
    }


def _score(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    mean_return = summary["mean_ordinary_return_among_safe_seeds"]
    return (
        float(summary["ordinary_safe_seeds"]),
        -float("inf") if mean_return is None else float(mean_return),
        -float(summary["mean_hybrid_intervention"]),
    )


def _consider_checkpoint(
    algorithm,
    *,
    evaluation: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    step: int,
    label: str,
    output_dir: Path,
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_score = _score(aggregate)
    if best is None or candidate_score > tuple(best["score_tuple"]):
        checkpoint = output_dir / "best_overall.pkl"
        algorithm.save(str(checkpoint))
        best = {
            "step": int(step),
            "label": label,
            "score": {
                "ordinary_safe_seeds": aggregate["ordinary_safe_seeds"],
                "mean_ordinary_return_among_safe_seeds": aggregate[
                    "mean_ordinary_return_among_safe_seeds"
                ],
                "mean_hybrid_intervention": aggregate["mean_hybrid_intervention"],
            },
            "score_tuple": list(candidate_score),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
        }
        _write_json(output_dir / "best_overall.json", best)
    if int(aggregate["ordinary_safe_seeds"]) == len(ALL_SEEDS):
        algorithm.promote_deployable(
            {"step": step, "label": label, "evaluation": evaluation["per_seed"]}
        )
        deployable = output_dir / "all_seed_deployable.pkl"
        algorithm.save(str(deployable))
        _write_json(
            output_dir / "all_seed_deployable.json",
            {
                "step": step,
                "label": label,
                "checkpoint": str(deployable),
                "checkpoint_sha256": _sha256(deployable),
                "metrics": aggregate,
            },
        )
    return best


def _full_evaluation(
    algorithm,
    *,
    step: int,
    label: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = output_dir / f"evaluation_{label}_{step:08d}.pkl"
    output = output_dir / f"evaluation_{label}_{step:08d}.json"
    algorithm.save(str(checkpoint))
    evaluation = _run_isolated_evaluation(
        checkpoint=checkpoint,
        output=output,
        seeds=ALL_SEEDS,
        modes=("hybrid", "ordinary"),
        policy="learner",
    )
    aggregate = _evaluation_summary(evaluation)
    _write_json(output_dir / f"evaluation_{label}_{step:08d}_summary.json", aggregate)
    checkpoint.unlink()
    return evaluation, aggregate


def _online_train(
    algorithm,
    dataset,
    args,
    *,
    key,
    rng: np.random.Generator,
    output_dir: Path,
    metrics_path: Path,
    best: dict[str, Any],
):
    import jax
    from relax.buffer.tree import TreeBuffer

    replay = TreeBuffer.from_guardian_dsac_experience(
        dataset.obs_dim,
        dataset.action_dim,
        args.online_buffer_size,
        seed=args.seed + 31_003,
    )
    seed_cycle = EpisodeSeedCycle(ONLINE_EPISODE_SCHEDULE)
    current_seed = seed_cycle.current_seed
    initial_q1 = algorithm.state.params.q1
    initial_proxy = algorithm.state.params.proxy
    initial_policy = algorithm.state.params.policy
    counters = {
        "online_environment_steps": 0,
        "online_episodes": 0,
        "task_critic_updates": 0,
        "proxy_updates": 0,
        "actor_updates": 0,
        "hybrid_gate_evaluations": 0,
        "hybrid_gate_passes": 0,
        "hybrid_actor_rollbacks": 0,
        "numerical_full_rollbacks": 0,
        "uncertified_action_rejections": 0,
        "aborted_incomplete_episodes": 0,
    }
    steps_by_seed = {seed: 0 for seed in ALL_SEEDS}
    episodes_by_seed = {seed: 0 for seed in ALL_SEEDS}
    interventions_by_seed = {seed: 0 for seed in ALL_SEEDS}
    speed_bins_by_seed = {seed: np.zeros(4, np.int64) for seed in ALL_SEEDS}
    safety_failures_by_seed = {seed: set() for seed in ALL_SEEDS}
    replay_seed_ledger: deque[int] = deque(maxlen=args.online_buffer_size)
    gate_history = []
    full_evaluations = []
    latest = {}
    episode = []
    episode_step = 0
    gate_number = 0
    next_gate = args.hybrid_gate_every
    next_full = args.full_evaluation_every
    consecutive_uncertified = 0

    def fresh_environment(current_env, seed):
        if current_env is not None:
            current_env.close()
        env = _make_environment(args, seed=seed, mode="hybrid")
        observation, info = env.reset(seed=seed)
        signature = str(info.get("metasafe_scene_signature", ""))
        expected = algorithm.scene_signatures[str(seed)]
        if signature != expected:
            env.close()
            raise RuntimeError(f"Seed{seed} training scene signature mismatch")
        return env, np.asarray(observation, np.float32).reshape(-1), signature

    env, observation, scene_signature = fresh_environment(None, current_seed)
    try:
        while counters["online_environment_steps"] < args.online_steps:
            action = np.asarray(
                algorithm.get_deterministic_action(observation), dtype=np.float32
            ).reshape(2)
            action = action + rng.normal(0.0, args.online_exploration_std, 2).astype(
                np.float32
            )
            if rng.random() < args.stratified_speed_exploration:
                bin_index = counters["online_environment_steps"] % 4
                fraction = rng.uniform(bin_index / 4.0, (bin_index + 1) / 4.0)
                action[0] = np.float32(2.0 * fraction - 1.0)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            certificate = env.certify_target_action(action)
            if bool(certificate.get("projection_failed", False)):
                consecutive_uncertified += 1
                counters["uncertified_action_rejections"] += 1
                counters["hybrid_actor_rollbacks"] += 1
                counters["aborted_incomplete_episodes"] += 1
                safety_failures_by_seed[current_seed].add("projection_failed")
                algorithm.reject_hybrid_safety()
                episode.clear()
                episode_step = 0
                env, observation, scene_signature = fresh_environment(env, current_seed)
                if consecutive_uncertified >= 20:
                    raise RuntimeError("20 consecutive uncertified Actor proposals")
                continue
            consecutive_uncertified = 0
            authority = _audit_low_level_cbf_authority(
                env,
                action,
                certificate,
                trajectory_tolerance=args.cbf_trajectory_tolerance,
                derive_teacher=args.online_inverse_teacher,
            )
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.execute_certified_target(certificate)
            counters["online_environment_steps"] += 1
            step = counters["online_environment_steps"]
            steps_by_seed[current_seed] += 1
            episode_step += 1
            truncated = bool(
                truncated
                or episode_step >= args.max_episode_steps
                or step >= args.online_steps
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
                scene_signature=scene_signature,
                action_semantics_version=TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
            )
            episode.append(transition)
            failures = _online_safety_failures(info)
            safety_failures_by_seed[current_seed].update(failures)
            interventions_by_seed[current_seed] += int(bool(transition.intervention))
            if not bool(transition.emergency_stop):
                fraction = float(
                    np.clip(
                        (
                            float(transition.executed_target_speed)
                            - float(transition.reachable_speed_lower)
                        )
                        / (
                            float(transition.reachable_speed_upper)
                            - float(transition.reachable_speed_lower)
                            + 1e-6
                        ),
                        0.0,
                        1.0,
                    )
                )
                speed_bins_by_seed[current_seed][min(int(fraction * 4.0), 3)] += 1
            done = bool(terminated or truncated)
            observation = np.asarray(next_observation, np.float32).reshape(-1)
            if done:
                final = _store_complete_online_episode(
                    replay,
                    episode,
                    scene_seed=current_seed,
                    scene_signature=scene_signature,
                    gamma=args.discount,
                    proxy_discount=args.proxy_discount,
                )
                replay_seed_ledger.extend([current_seed] * len(final))
                counters["online_episodes"] += 1
                episodes_by_seed[current_seed] += 1
                episode.clear()
                seed_cycle.complete_episode()

            progress = step / max(args.online_steps, 1)
            offline_ratio = args.offline_ratio_initial + min(progress, 1.0) * (
                args.offline_ratio_final - args.offline_ratio_initial
            )
            if step >= args.learning_starts:
                for gradient_index in range(args.gradient_steps):
                    batch = _sample_mixed_batch(
                        dataset,
                        replay,
                        batch_size=args.batch_size,
                        offline_ratio=offline_ratio,
                        rng=rng,
                    )
                    key, update_key = jax.random.split(key)
                    update_actor = bool(
                        (step + gradient_index) % args.policy_delay == 0
                    )
                    latest = algorithm.online_joint_update(
                        update_key, batch, update_actor=update_actor
                    )
                    counters["task_critic_updates"] += 1
                    counters["proxy_updates"] += 1
                    counters["actor_updates"] += int(update_actor)
                    disaster = algorithm.numerical_disaster_reasons(
                        metrics=latest,
                        observation=np.asarray(batch.obs[:16]),
                        action=np.asarray(batch.action[:16]),
                    )
                    if disaster:
                        counters["numerical_full_rollbacks"] += 1
                        algorithm.reject_numerical_disaster()
                        _append_jsonl(
                            metrics_path,
                            {
                                "phase": "numerical_disaster",
                                "step": step,
                                "reasons": disaster,
                            },
                        )
                        break

            if step % args.log_every == 0:
                row = {
                    "phase": "online_joint_guardian_dsac_multiseed",
                    **counters,
                    "offline_ratio": offline_ratio,
                    "online_replay_size": len(replay),
                    "online_steps_by_seed": {
                        str(k): v for k, v in steps_by_seed.items()
                    },
                    **latest,
                    **_grouped_actor_diagnostics(latest),
                }
                _append_jsonl(metrics_path, row)
                print(f"[multiseed] online: {step}/{args.online_steps}", flush=True)

            if step >= next_gate:
                while next_gate <= step:
                    next_gate += args.hybrid_gate_every
                pair = HYBRID_GATE_PAIRS[gate_number % len(HYBRID_GATE_PAIRS)]
                gate_number += 1
                checkpoint = output_dir / "hybrid_gate_candidate.pkl"
                algorithm.save(str(checkpoint))
                evaluation = _run_isolated_evaluation(
                    checkpoint=checkpoint,
                    output=output_dir / f"hybrid_gate_{step:08d}.json",
                    seeds=pair,
                    modes=("hybrid",),
                    policy="learner",
                )
                by_seed = {
                    str(seed): evaluation["per_seed"][str(seed)]["hybrid"]
                    for seed in pair
                }
                gate = apply_multi_seed_hybrid_learning_gate(
                    algorithm, by_seed, required_seeds=pair
                )
                counters["hybrid_gate_evaluations"] += 1
                counters["hybrid_gate_passes"] += int(gate["passed"])
                counters["hybrid_actor_rollbacks"] += int(
                    gate.get("rollback") == "actor_only"
                )
                counters["numerical_full_rollbacks"] += int(
                    gate.get("rollback") == "hybrid_safe_full_snapshot"
                )
                if gate["passed"]:
                    latest_safe = output_dir / "latest_hybrid_safe.pkl"
                    algorithm.save(str(latest_safe))
                gate_history.append(
                    {
                        "step": step,
                        "pair": list(pair),
                        "gate": gate,
                        "evaluation": by_seed,
                    }
                )
                _append_jsonl(
                    metrics_path,
                    {
                        "phase": "hybrid_learning_gate",
                        "step": step,
                        "pair": list(pair),
                        "gate": gate,
                    },
                )
                checkpoint.unlink()

            if step >= next_full:
                while next_full <= step:
                    next_full += args.full_evaluation_every
                evaluation, aggregate = _full_evaluation(
                    algorithm,
                    step=step,
                    label="online",
                    output_dir=output_dir,
                )
                best = _consider_checkpoint(
                    algorithm,
                    evaluation=evaluation,
                    aggregate=aggregate,
                    step=step,
                    label="online",
                    output_dir=output_dir,
                    best=best,
                )
                full_evaluations.append(
                    {"step": step, "aggregate": aggregate, "evaluation": evaluation}
                )

            if done and step < args.online_steps:
                current_seed = seed_cycle.current_seed
                env, observation, scene_signature = fresh_environment(env, current_seed)
                episode_step = 0
    finally:
        env.close()

    if not full_evaluations or full_evaluations[-1]["step"] != args.online_steps:
        evaluation, aggregate = _full_evaluation(
            algorithm,
            step=args.online_steps,
            label="online_final",
            output_dir=output_dir,
        )
        best = _consider_checkpoint(
            algorithm,
            evaluation=evaluation,
            aggregate=aggregate,
            step=args.online_steps,
            label="online_final",
            output_dir=output_dir,
            best=best,
        )
        full_evaluations.append(
            {
                "step": args.online_steps,
                "aggregate": aggregate,
                "evaluation": evaluation,
            }
        )

    replay_counts = Counter(replay_seed_ledger)
    replay_total = max(len(replay_seed_ledger), 1)
    summary = {
        **counters,
        "requested_online_steps": args.online_steps,
        "online_episode_schedule": list(ONLINE_EPISODE_SCHEDULE),
        "daily_hybrid_gate_pairs": [list(pair) for pair in HYBRID_GATE_PAIRS],
        "online_steps_by_seed": {str(k): v for k, v in steps_by_seed.items()},
        "online_transition_fraction_by_seed": {
            str(k): v / max(args.online_steps, 1) for k, v in steps_by_seed.items()
        },
        "online_episodes_by_seed": {str(k): v for k, v in episodes_by_seed.items()},
        "online_replay_size": len(replay),
        "online_replay_fraction_by_seed": {
            str(k): replay_counts.get(k, 0) / replay_total for k in ALL_SEEDS
        },
        "intervention_rate_by_seed": {
            str(k): interventions_by_seed[k] / max(steps_by_seed[k], 1)
            for k in ALL_SEEDS
        },
        "speed_bins_by_seed": {
            str(k): speed_bins_by_seed[k].astype(int).tolist() for k in ALL_SEEDS
        },
        "training_hybrid_safety_failures_by_seed": {
            str(k): sorted(safety_failures_by_seed[k]) for k in ALL_SEEDS
        },
        "task_critic_updated": _tree_changed(initial_q1, algorithm.state.params.q1),
        "proxy_updated": _tree_changed(initial_proxy, algorithm.state.params.proxy),
        "actor_updated": _tree_changed(initial_policy, algorithm.state.params.policy),
        "final_actor_diagnostics": {
            **latest,
            **_grouped_actor_diagnostics(latest),
        },
        "hybrid_gate_history": gate_history,
        "full_evaluations": full_evaluations,
        "final_evaluation": full_evaluations[-1]["aggregate"],
        "best_overall": best,
    }
    return key, summary, best


def _format_number(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def _write_result_tables(
    output_dir: Path,
    *,
    offline: Mapping[str, Any],
    online: Mapping[str, Any] | None,
    online_steps: int,
    best: Mapping[str, Any],
) -> None:
    rows = [
        (
            "Offline-only",
            0,
            offline["hybrid_safe_seeds"],
            offline["ordinary_safe_seeds"],
            offline["mean_ordinary_return_among_safe_seeds"],
            offline["mean_hybrid_intervention"],
        )
    ]
    if online is not None:
        rows.append(
            (
                "Guardian-DSAC",
                online_steps,
                online["hybrid_safe_seeds"],
                online["ordinary_safe_seeds"],
                online["mean_ordinary_return_among_safe_seeds"],
                online["mean_hybrid_intervention"],
            )
        )
    lines = [
        "# Guardian-DSAC multi-Seed main result",
        "",
        "| Method | Offline Seeds | Online Steps | Hybrid safe Seeds | Ordinary safe Seeds | Mean Ordinary return | Mean Hybrid intervention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, steps, hybrid, ordinary, result_return, intervention in rows:
        lines.append(
            f"| {method} | 5 | {steps} | {hybrid}/10 | {ordinary}/10 | "
            f"{_format_number(result_return)} | {_format_number(intervention, 5)} |"
        )
    lines.extend(
        [
            "",
            "Ordinary safe requires deterministic fresh-process success with zero collision, road departure, environment cost, solver failure, or non-positive physical margin. Mean Ordinary return is over those safe successful Seeds only.",
            "",
            "## Per-Seed detail",
            "",
            "| Seed | Role | Offline Ordinary | Offline return | Online Ordinary | Online return | Online Hybrid intervention |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in ALL_SEEDS:
        before = offline["per_seed"][str(seed)]
        after = None if online is None else online["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | {before['role']} | {'safe' if before['ordinary_safe'] else 'unsafe'} | "
            f"{_format_number(before['ordinary_return'])} | "
            f"{'—' if after is None else ('safe' if after['ordinary_safe'] else 'unsafe')} | "
            f"{'—' if after is None else _format_number(after['ordinary_return'])} | "
            f"{'—' if after is None else _format_number(after['hybrid_intervention'], 5)} |"
        )
    lines.extend(
        [
            "",
            "## Checkpoint selection",
            "",
            f"`best_overall` was selected at step {best['step']} by the lexicographic score (Ordinary safe Seed count, mean Ordinary return, negative mean Hybrid intervention).",
            "",
        ]
    )
    (output_dir / "main_results.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import jax

    started = time.time()
    backend = jax.default_backend()
    if args.require_gpu and backend != "gpu":
        raise RuntimeError(f"formal multi-Seed training requires GPU, got {backend}")
    manifest_path, manifest, dataset = _load_multiseed_dataset(args)
    scene_signatures = _predeclare_scene_signatures(args, dataset)
    key = jax.random.PRNGKey(args.seed)
    key, network_key = jax.random.split(key)
    algorithm, hidden_sizes = _create_algorithm(
        args, dataset, scene_signatures, network_key
    )
    run_name = args.run_name or (
        ("smoke_" if args.component_smoke else "formal_")
        + time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir = args.output_root.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"
    config = {
        "format": "guardian_dsac_multiseed_run_config_v1",
        "method": "Guardian-DSAC V2.3.1 multi-Seed",
        "phase": args.phase,
        "seed": args.seed,
        "action_semantics_version": TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        "scene_seeds": list(ALL_SEEDS),
        "scene_signatures": scene_signatures,
        "dataset_sha256": dataset.sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "observation_dim": dataset.obs_dim,
        "actor_observation_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "hidden_sizes": list(hidden_sizes),
        "init_alpha": args.init_alpha,
        "algorithm_config": asdict(algorithm.config),
        "environment": _environment_config(args),
        "offline_seeds": list(OFFLINE_SEEDS),
        "online_novel_seeds": [5, 6, 7, 8, 9],
        "online_episode_schedule": list(ONLINE_EPISODE_SCHEDULE),
        "hybrid_gate_pairs": [list(pair) for pair in HYBRID_GATE_PAIRS],
        "full_evaluation_steps": [
            0,
            *range(
                args.full_evaluation_every,
                args.online_steps + 1,
                args.full_evaluation_every,
            ),
            *(
                [args.online_steps]
                if args.online_steps % args.full_evaluation_every
                else []
            ),
        ],
        "budgets": {
            "bc_gradient_steps": args.bc_gradient_steps,
            "critic_gradient_steps": args.critic_gradient_steps,
            "proxy_gradient_steps": args.proxy_gradient_steps,
            "online_steps": args.online_steps,
            "batch_size": args.batch_size,
            "online_buffer_size": args.online_buffer_size,
        },
        "component_smoke": args.component_smoke,
        "backend": backend,
        "devices": [str(device) for device in jax.devices()],
    }
    _write_json(output_dir / "config.json", config)

    offline_checkpoint = output_dir / "offline_multiseed_guardian_dsac.pkl"
    if args.offline_checkpoint is not None:
        source = args.offline_checkpoint.expanduser().resolve()
        algorithm.load(str(source))
        if not algorithm.offline_reference_ready:
            raise RuntimeError("loaded checkpoint has no frozen offline reference")
        offline_summary = {
            "loaded": True,
            "source": str(source),
            "source_sha256": _sha256(source),
            "validation": _multiseed_validation(algorithm, dataset),
            "speed_coverage": dataset.speed_coverage(),
            "support_thresholds": asdict(algorithm.offline_support_thresholds),
        }
        algorithm.save(str(offline_checkpoint))
    else:
        key, offline_summary = _offline_train(
            algorithm, dataset, args, key=key, metrics_path=metrics_path
        )
        algorithm.save(str(offline_checkpoint))
    offline_summary.update(
        checkpoint=str(offline_checkpoint),
        checkpoint_sha256=_sha256(offline_checkpoint),
    )
    _write_json(output_dir / "offline_summary.json", offline_summary)

    # Group A: the shared offline checkpoint on all ten predeclared Seeds.
    offline_evaluation = _run_isolated_evaluation(
        checkpoint=offline_checkpoint,
        output=output_dir / "evaluation_offline_initial_00000000.json",
        seeds=ALL_SEEDS,
        modes=("hybrid", "ordinary"),
        policy="learner",
    )
    offline_aggregate = _evaluation_summary(offline_evaluation)
    _write_json(output_dir / "group_a_offline_only.json", offline_aggregate)
    latest_safe = output_dir / "latest_hybrid_safe.pkl"
    algorithm.save(str(latest_safe))
    best = _consider_checkpoint(
        algorithm,
        evaluation=offline_evaluation,
        aggregate=offline_aggregate,
        step=0,
        label="offline_initial",
        output_dir=output_dir,
        best=None,
    )

    online_summary = None
    if args.phase in ("online", "full") and args.online_steps > 0:
        rng = np.random.default_rng(args.seed + 202_608_05)
        key, online_summary, best = _online_train(
            algorithm,
            dataset,
            args,
            key=key,
            rng=rng,
            output_dir=output_dir,
            metrics_path=metrics_path,
            best=best,
        )
        _write_json(output_dir / "online_summary.json", online_summary)

    online_aggregate = (
        None if online_summary is None else online_summary["final_evaluation"]
    )
    _write_result_tables(
        output_dir,
        offline=offline_aggregate,
        online=online_aggregate,
        online_steps=args.online_steps,
        best=best,
    )
    summary = {
        "format": "guardian_dsac_multiseed_main_summary_v1",
        "method": "Guardian-DSAC V2.3.1",
        "dataset": dataset.summary(),
        "collection_manifest": manifest,
        "group_a_offline_only": offline_aggregate,
        "group_b_guardian_dsac": online_aggregate,
        "offline": offline_summary,
        "online": online_summary,
        "best_overall": best,
        "latest_hybrid_safe": {
            "checkpoint": str(latest_safe),
            "checkpoint_sha256": _sha256(latest_safe),
        },
        "all_seed_deployable": (
            str(output_dir / "all_seed_deployable.pkl")
            if (output_dir / "all_seed_deployable.pkl").is_file()
            else None
        ),
        "wall_time_seconds": time.time() - started,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "group_a": offline_aggregate,
                "group_b": online_aggregate,
                "best_overall": best,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
