#!/usr/bin/env python3
"""Run the formal 10-Seed Guardian-Flow-DSAC V2.4 main experiment."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.offline.guardian_dsac_multiseed_dataset import (  # noqa: E402
    guardian_dsac_to_flow_experience,
)
from relax.utils.action_semantics import (  # noqa: E402
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)
from relax.utils.guardian_learning_gate import (  # noqa: E402
    apply_multi_seed_hybrid_learning_gate,
)
from scripts.train_guardian_calql_metasafe import (  # noqa: E402
    _audit_low_level_cbf_authority,
    _concatenate_batches,
)
from scripts.train_guardian_dsac_metasafe import (  # noqa: E402
    EpisodeSeedCycle,
    _append_jsonl,
    _environment_config,
    _make_environment,
    _offline_validation,
    _online_safety_failures,
    _run_isolated_evaluation,
    _sha256,
    _step_dsac_experience,
    _store_complete_online_episode,
    _tree_changed,
    _write_json,
)
from scripts.train_guardian_dsac_multiseed import (  # noqa: E402
    ALL_SEEDS,
    HYBRID_GATE_PAIRS,
    OFFLINE_SEEDS,
    ONLINE_EPISODE_SCHEDULE,
    _evaluation_summary,
    _load_multiseed_dataset,
    _predeclare_scene_signatures,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-offline-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/guardian_flow_dsac_multiseed")
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--phase", choices=("offline", "online", "full"), default="full")
    parser.add_argument("--flow-offline-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-hidden-sizes", default="256,256")
    parser.add_argument("--flow-integration-steps", type=int, default=8)
    parser.add_argument("--flow-bc-gradient-steps", type=int, default=80_000)
    parser.add_argument("--flow-distill-gradient-steps", type=int, default=40_000)
    parser.add_argument("--flow-offline-joint-steps", type=int, default=50_000)
    parser.add_argument("--flow-learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-actor-tau", type=float, default=0.005)
    parser.add_argument("--q-actor-weight", type=float, default=1.0)
    parser.add_argument("--proxy-actor-weight", type=float, default=None)
    parser.add_argument("--correction-weight", type=float, default=1.0)
    parser.add_argument("--correction-fraction", type=float, default=0.25)
    parser.add_argument("--offline-distill-weight-initial", type=float, default=1.0)
    parser.add_argument("--offline-distill-weight-final", type=float, default=0.2)
    parser.add_argument("--proximal-weight-initial", type=float, default=1.0)
    parser.add_argument("--proximal-weight-min", type=float, default=0.05)
    parser.add_argument("--proximal-decay-steps", type=int, default=20_000)
    parser.add_argument("--proximal-speed-weight", type=float, default=0.5)
    parser.add_argument("--proximal-lateral-weight", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--support-calibration-rows-per-seed", type=int, default=8192)
    parser.add_argument("--online-steps", type=int, default=150_000)
    parser.add_argument("--online-buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--hybrid-gate-every", type=int, default=10_000)
    parser.add_argument("--full-evaluation-every", type=int, default=50_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--offline-ratio-initial", type=float, default=0.50)
    parser.add_argument("--offline-ratio-final", type=float, default=0.20)
    parser.add_argument(
        "--require-gpu", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--component-smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.phase == "online" and args.flow_offline_checkpoint is None:
        parser.error("--phase online requires --flow-offline-checkpoint")
    positive = (
        "flow_integration_steps",
        "proximal_decay_steps",
        "batch_size",
        "support_calibration_rows_per_seed",
        "online_buffer_size",
        "learning_starts",
        "gradient_steps",
        "policy_delay",
        "hybrid_gate_every",
        "full_evaluation_every",
        "log_every",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("positive integer option is non-positive")
    for name in (
        "flow_bc_gradient_steps",
        "flow_distill_gradient_steps",
        "flow_offline_joint_steps",
        "online_steps",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if not 0.0 <= args.offline_ratio_final <= args.offline_ratio_initial <= 1.0:
        parser.error("invalid offline replay ratios")
    if not 0.0 <= args.correction_fraction <= 0.25:
        parser.error("correction fraction must be in [0, 0.25]")
    if args.component_smoke:
        args.flow_bc_gradient_steps = min(args.flow_bc_gradient_steps, 2)
        args.flow_distill_gradient_steps = min(args.flow_distill_gradient_steps, 2)
        args.flow_offline_joint_steps = min(args.flow_offline_joint_steps, 2)
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


def _run_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    for candidate in (checkpoint.parent, checkpoint.parent.parent):
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("config.json is not adjacent to the shared checkpoint")


def _load_source_config(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint = checkpoint.expanduser().resolve()
    config_path = _run_dir(checkpoint) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("method") != "Guardian-DSAC V2.3.1 multi-Seed":
        raise ValueError("shared checkpoint is not the formal V2.3.1 multi-Seed run")
    if tuple(int(value) for value in config.get("offline_seeds", [])) != OFFLINE_SEEDS:
        raise ValueError("shared checkpoint does not bind offline Seed0--4")
    if tuple(int(value) for value in config.get("scene_seeds", [])) != ALL_SEEDS:
        raise ValueError("shared checkpoint does not predeclare Seed0--9")
    if int(config["environment"]["max_episode_steps"]) != 1200:
        raise ValueError("V2.4 comparison requires the same 1200-step horizon")
    return config_path, config


def _runtime_args(args, source_config):
    environment = dict(source_config["environment"])
    values = {
        **environment,
        **vars(args),
        "manifest": (
            args.manifest
            if args.manifest is not None
            else Path(source_config["manifest"])
        ),
        "discount": float(source_config["algorithm_config"]["gamma"]),
        "proxy_discount": 0.97,
        "max_command_speed": float(environment["max_command_speed"]),
        "target_velocity": float(environment["target_velocity"]),
        "online_inverse_teacher": True,
        "cbf_trajectory_tolerance": 0.08,
    }
    return SimpleNamespace(**values)


def _hidden_sizes(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise ValueError("Flow hidden sizes must be positive")
    return result


def _create_algorithm(args, source_config, dataset, flow_key):
    import jax

    from relax.algorithm.guardian_dsac_flow_v2_4 import (
        GuardianDSACFlowV24,
        GuardianDSACFlowV24Config,
    )
    from relax.algorithm.guardian_dsac_v2_3 import (
        GuardianDSACV23,
        GuardianDSACV23Config,
    )
    from relax.network.flow_policy import create_one_step_flow_policy
    from relax.network.safe_sac_jax import create_guardian_dsac_net

    base_hidden_sizes = tuple(int(value) for value in source_config["hidden_sizes"])
    base_network, base_params = create_guardian_dsac_net(
        jax.random.PRNGKey(int(source_config["seed"])),
        critic_obs_dim=int(source_config["observation_dim"]),
        actor_obs_dim=int(source_config["actor_observation_dim"]),
        act_dim=int(source_config["action_dim"]),
        hidden_sizes=base_hidden_sizes,
        init_alpha=float(source_config["init_alpha"]),
        sigma_min=float(source_config["algorithm_config"]["sigma_min"]),
    )
    base = GuardianDSACV23(
        base_network,
        base_params,
        config=GuardianDSACV23Config(**source_config["algorithm_config"]),
        action_semantics_version=source_config["action_semantics_version"],
        scene_seeds=source_config["scene_seeds"],
        scene_signatures=source_config["scene_signatures"],
        dataset_sha256=dataset.sha256,
    )
    base.load(str(args.shared_offline_checkpoint.expanduser().resolve()))
    if not base.offline_reference_ready:
        raise RuntimeError("shared V2.3.1 checkpoint has no frozen offline Q/P reference")
    if base.dataset_sha256 != dataset.sha256:
        raise RuntimeError("shared Q/P checkpoint and five-Seed dataset differ")

    flow_hidden_sizes = _hidden_sizes(args.flow_hidden_sizes)
    flow_policy, flow_params = create_one_step_flow_policy(
        flow_key,
        actor_obs_dim=dataset.obs_dim,
        act_dim=dataset.action_dim,
        hidden_sizes=flow_hidden_sizes,
        integration_steps=args.flow_integration_steps,
    )
    proxy_weight = (
        float(base.config.proxy_actor_weight)
        if args.proxy_actor_weight is None
        else float(args.proxy_actor_weight)
    )
    flow_config = GuardianDSACFlowV24Config(
        learning_rate=args.flow_learning_rate,
        target_actor_tau=args.target_actor_tau,
        q_actor_weight=args.q_actor_weight,
        proxy_actor_weight=proxy_weight,
        offline_distill_weight_initial=args.offline_distill_weight_initial,
        offline_distill_weight_final=args.offline_distill_weight_final,
        offline_distill_decay_steps=max(args.flow_offline_joint_steps, 1),
        correction_weight=args.correction_weight,
        proximal_weight_initial=args.proximal_weight_initial,
        proximal_weight_min=args.proximal_weight_min,
        proximal_decay_steps=args.proximal_decay_steps,
        proximal_speed_weight=args.proximal_speed_weight,
        proximal_lateral_weight=args.proximal_lateral_weight,
    )
    algorithm = GuardianDSACFlowV24(
        base,
        flow_policy,
        flow_params,
        config=flow_config,
    )
    return algorithm, base_hidden_sizes, flow_hidden_sizes


def _multiseed_validation(algorithm, dataset) -> dict[str, Any]:
    per_seed = {
        str(seed): _offline_validation(algorithm, dataset.by_seed[seed])
        for seed in dataset.scene_seeds
    }
    names = sorted({name for row in per_seed.values() for name in row})
    mean = {
        name: float(np.mean([row[name] for row in per_seed.values()]))
        for name in names
    }
    return {"mean_across_seeds": mean, "per_seed": per_seed}


def _flow_distillation_diagnostics(algorithm, dataset, *, rows_per_seed: int):
    import jax
    import jax.numpy as jnp

    batch = dataset.flow_validation_reference_batch(
        rows_per_seed=rows_per_seed, to_jax=True
    )
    count = int(np.asarray(batch.reward).shape[0])
    key_1, key_2 = jax.random.split(jax.random.PRNGKey(24_001))
    noise_1 = jax.random.normal(key_1, (count, dataset.action_dim), jnp.float32)
    noise_2 = jax.random.normal(key_2, (count, dataset.action_dim), jnp.float32)
    teacher = algorithm.flow_policy.bc_action(
        algorithm.state.params.bc_flow, batch.obs, noise_1
    )
    student = algorithm.flow_policy.get_action(
        algorithm.state.params.one_step, batch.obs, noise_1
    )
    diversity = algorithm.flow_policy.latent_action_diversity(
        algorithm.state.params.one_step, batch.obs, noise_1, noise_2
    )
    error = np.asarray(student - teacher)
    return {
        "rows": count,
        "same_latent_mse": float(np.mean(np.sum(np.square(error), axis=-1))),
        "same_latent_mse_speed": float(np.mean(np.square(error[..., 0]))),
        "same_latent_mse_lateral": float(np.mean(np.square(error[..., 1]))),
        "latent_action_diversity": float(np.asarray(diversity)),
    }


def _offline_train(
    algorithm,
    dataset,
    args,
    *,
    key,
    output_dir: Path,
    metrics_path: Path,
):
    import jax

    initial_q1 = algorithm.base.state.params.q1
    initial_proxy = algorithm.base.state.params.proxy
    initial_teacher = algorithm.state.params.bc_flow
    initial_actor = algorithm.state.params.one_step
    latest = {}

    for step in range(args.flow_bc_gradient_steps):
        batch = dataset.sample_flow(args.batch_size, split="train", to_jax=True)
        key, update_key = jax.random.split(key)
        latest = algorithm.pretrain_bc_flow(update_key, batch)
        if (step + 1) % args.log_every == 0 or step + 1 == args.flow_bc_gradient_steps:
            _append_jsonl(
                metrics_path,
                {"phase": "offline_flow_matching", "step": step + 1, **latest},
            )
            print(
                f"[flow-multiseed] flow matching: {step + 1}/{args.flow_bc_gradient_steps}",
                flush=True,
            )

    for step in range(args.flow_distill_gradient_steps):
        batch = dataset.sample_flow(args.batch_size, split="train", to_jax=True)
        key, update_key = jax.random.split(key)
        latest = algorithm.distill_one_step(update_key, batch)
        if (step + 1) % args.log_every == 0 or step + 1 == args.flow_distill_gradient_steps:
            _append_jsonl(
                metrics_path,
                {"phase": "one_step_distillation", "step": step + 1, **latest},
            )
            print(
                f"[flow-multiseed] distillation: {step + 1}/{args.flow_distill_gradient_steps}",
                flush=True,
            )

    pre_joint_diagnostics = _flow_distillation_diagnostics(
        algorithm,
        dataset,
        rows_per_seed=args.support_calibration_rows_per_seed,
    )
    algorithm.freeze_bc_flow()
    frozen_teacher = algorithm.state.params.bc_flow
    distilled_checkpoint = output_dir / "distilled_one_step_flow.pkl"
    algorithm.save(str(distilled_checkpoint))
    distilled_evaluation = _run_isolated_evaluation(
        checkpoint=distilled_checkpoint,
        output=output_dir / "evaluation_post_distillation_00000000.json",
        seeds=ALL_SEEDS,
        modes=("hybrid", "ordinary"),
        policy="learner",
    )
    distilled_aggregate = _evaluation_summary(distilled_evaluation)
    _write_json(
        output_dir / "evaluation_post_distillation_00000000_summary.json",
        distilled_aggregate,
    )

    for step in range(args.flow_offline_joint_steps):
        batch = dataset.sample_flow(
            args.batch_size,
            split="train",
            pair_fraction=0.50,
            to_jax=True,
        )
        key, update_key = jax.random.split(key)
        latest = algorithm.offline_flow_joint_update(update_key, batch)
        if (step + 1) % args.log_every == 0 or step + 1 == args.flow_offline_joint_steps:
            _append_jsonl(
                metrics_path,
                {"phase": "offline_flow_q_guided", "step": step + 1, **latest},
            )
            print(
                f"[flow-multiseed] offline joint: {step + 1}/{args.flow_offline_joint_steps}",
                flush=True,
            )

    validation = dataset.flow_validation_reference_batch(
        rows_per_seed=args.support_calibration_rows_per_seed,
        to_jax=True,
    )
    thresholds = algorithm.refresh_offline_reference(validation)
    post_joint_diagnostics = _flow_distillation_diagnostics(
        algorithm,
        dataset,
        rows_per_seed=args.support_calibration_rows_per_seed,
    )
    if _tree_changed(frozen_teacher, algorithm.state.params.bc_flow):
        raise RuntimeError("offline Flow teacher changed after it was frozen")
    return key, {
        "flow_bc_steps": args.flow_bc_gradient_steps,
        "flow_distill_steps": args.flow_distill_gradient_steps,
        "flow_offline_joint_steps": args.flow_offline_joint_steps,
        "teacher_updated_during_flow_matching": _tree_changed(
            initial_teacher, frozen_teacher
        ),
        "teacher_frozen_after_distillation": algorithm.bc_flow_frozen,
        "teacher_changed_after_freeze": False,
        "actor_updated": _tree_changed(initial_actor, algorithm.state.params.one_step),
        "task_critic_updated_during_joint": _tree_changed(
            initial_q1, algorithm.base.state.params.q1
        ),
        "proxy_updated_during_joint": _tree_changed(
            initial_proxy, algorithm.base.state.params.proxy
        ),
        "support_thresholds": asdict(thresholds),
        "pre_joint_distillation": pre_joint_diagnostics,
        "post_distillation_evaluation": distilled_aggregate,
        "distilled_checkpoint": str(distilled_checkpoint),
        "distilled_checkpoint_sha256": _sha256(distilled_checkpoint),
        "post_joint_distillation": post_joint_diagnostics,
        "validation": _multiseed_validation(algorithm, dataset),
        "speed_coverage": dataset.speed_coverage(),
        "latest_metrics": latest,
    }


def _sample_mixed_flow_batch(
    dataset,
    replay,
    *,
    batch_size: int,
    offline_ratio: float,
    rng: np.random.Generator,
):
    online_count = (
        min(int(round(batch_size * (1.0 - offline_ratio))), batch_size - 1)
        if len(replay) > 0 and batch_size > 1
        else 0
    )
    offline_count = batch_size - online_count
    batches = []
    if offline_count:
        batches.append(dataset.sample_flow(offline_count, split="train", to_jax=False))
    if online_count:
        batches.append(replay.sample(online_count, to_jax=False))
    return _concatenate_batches(batches, rng)


def _sample_actor_flow_batch(
    dataset,
    replay,
    correction_replay,
    *,
    batch_size: int,
    offline_ratio: float,
    correction_fraction: float,
    rng: np.random.Generator,
):
    correction_count = (
        min(int(round(batch_size * correction_fraction)), batch_size - 1)
        if len(correction_replay) > 0 and batch_size > 1
        else 0
    )
    ordinary_count = batch_size - correction_count
    batches = [
        _sample_mixed_flow_batch(
            dataset,
            replay,
            batch_size=ordinary_count,
            offline_ratio=offline_ratio,
            rng=rng,
        )
    ]
    if correction_count:
        batches.append(correction_replay.sample(correction_count, to_jax=False))
    return _concatenate_batches(batches, rng)


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
    evaluation,
    aggregate,
    step: int,
    label: str,
    output_dir: Path,
    best,
):
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


def _full_evaluation(algorithm, *, step: int, label: str, output_dir: Path):
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
    best,
):
    import jax
    from relax.buffer.tree import TreeBuffer

    replay = TreeBuffer.from_guardian_flow_dsac_experience(
        dataset.obs_dim,
        dataset.action_dim,
        args.online_buffer_size,
        seed=args.seed + 31_003,
    )
    correction_replay = TreeBuffer.from_guardian_flow_dsac_experience(
        dataset.obs_dim,
        dataset.action_dim,
        args.online_buffer_size,
        seed=args.seed + 31_004,
    )
    seed_cycle = EpisodeSeedCycle(ONLINE_EPISODE_SCHEDULE)
    current_seed = seed_cycle.current_seed
    initial_q1 = algorithm.base.state.params.q1
    initial_proxy = algorithm.base.state.params.proxy
    initial_actor = algorithm.state.params.one_step
    frozen_teacher = algorithm.state.params.bc_flow
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
        "flow_correction_replay_rows": 0,
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
            flow_noise = rng.normal(0.0, 1.0, dataset.action_dim).astype(np.float32)
            action = np.asarray(
                algorithm.get_action(observation, flow_noise), dtype=np.float32
            ).reshape(dataset.action_dim)
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
                    raise RuntimeError("20 consecutive uncertified Flow proposals")
                continue
            consecutive_uncertified = 0
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
            step = counters["online_environment_steps"]
            steps_by_seed[current_seed] += 1
            episode_step += 1
            truncated = bool(
                truncated
                or episode_step >= args.max_episode_steps
                or step >= args.online_steps
            )
            base_transition = _step_dsac_experience(
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
            transition = guardian_dsac_to_flow_experience(
                base_transition,
                flow_noise=flow_noise,
                flow_noise_valid=np.bool_(True),
            )
            episode.append(transition)
            safety_failures_by_seed[current_seed].update(
                _online_safety_failures(info)
            )
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
                finalized = _store_complete_online_episode(
                    replay,
                    episode,
                    scene_seed=current_seed,
                    scene_signature=scene_signature,
                    gamma=args.discount,
                    proxy_discount=args.proxy_discount,
                )
                for item in finalized:
                    if bool(item.proxy_pair_valid) and bool(item.flow_noise_valid):
                        correction_replay.add(item)
                        counters["flow_correction_replay_rows"] += 1
                replay_seed_ledger.extend([current_seed] * len(finalized))
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
                    batch = _sample_mixed_flow_batch(
                        dataset,
                        replay,
                        batch_size=args.batch_size,
                        offline_ratio=offline_ratio,
                        rng=rng,
                    )
                    update_actor = bool((step + gradient_index) % args.policy_delay == 0)
                    actor_batch = None
                    if update_actor:
                        actor_batch = _sample_actor_flow_batch(
                            dataset,
                            replay,
                            correction_replay,
                            batch_size=args.batch_size,
                            offline_ratio=offline_ratio,
                            correction_fraction=args.correction_fraction,
                            rng=rng,
                        )
                    key, update_key = jax.random.split(key)
                    latest = algorithm.online_joint_update(
                        update_key,
                        batch,
                        actor_data=actor_batch,
                        update_actor=update_actor,
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
                            {"phase": "numerical_disaster", "step": step, "reasons": disaster},
                        )
                        break

            if step % args.log_every == 0:
                _append_jsonl(
                    metrics_path,
                    {
                        "phase": "online_joint_guardian_flow_dsac_multiseed",
                        **counters,
                        "offline_ratio": offline_ratio,
                        "online_replay_size": len(replay),
                        "correction_replay_size": len(correction_replay),
                        "online_steps_by_seed": {
                            str(seed): count for seed, count in steps_by_seed.items()
                        },
                        **latest,
                    },
                )
                print(f"[flow-multiseed] online: {step}/{args.online_steps}", flush=True)

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
                proximal_before = int(np.asarray(algorithm.state.proximal_step))
                gate = apply_multi_seed_hybrid_learning_gate(
                    algorithm, by_seed, required_seeds=pair
                )
                proximal_after = int(np.asarray(algorithm.state.proximal_step))
                if gate["passed"] and proximal_after != proximal_before:
                    raise RuntimeError("passing Flow gate reset proximal decay")
                counters["hybrid_gate_evaluations"] += 1
                counters["hybrid_gate_passes"] += int(gate["passed"])
                counters["hybrid_actor_rollbacks"] += int(
                    gate.get("rollback") == "actor_only"
                )
                counters["numerical_full_rollbacks"] += int(
                    gate.get("rollback") == "hybrid_safe_full_snapshot"
                )
                if gate["passed"]:
                    algorithm.save(str(output_dir / "latest_hybrid_safe.pkl"))
                gate_history.append(
                    {
                        "step": step,
                        "pair": list(pair),
                        "gate": gate,
                        "evaluation": by_seed,
                        "proximal_step_before": proximal_before,
                        "proximal_step_after": proximal_after,
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
            {"step": args.online_steps, "aggregate": aggregate, "evaluation": evaluation}
        )

    if _tree_changed(frozen_teacher, algorithm.state.params.bc_flow):
        raise RuntimeError("online training changed the frozen Flow teacher")
    replay_counts = Counter(replay_seed_ledger)
    replay_total = max(len(replay_seed_ledger), 1)
    summary = {
        **counters,
        "requested_online_steps": args.online_steps,
        "online_episode_schedule": list(ONLINE_EPISODE_SCHEDULE),
        "hybrid_gate_pairs": [list(pair) for pair in HYBRID_GATE_PAIRS],
        "hybrid_gate_every": args.hybrid_gate_every,
        "full_evaluation_every": args.full_evaluation_every,
        "online_steps_by_seed": {str(key): value for key, value in steps_by_seed.items()},
        "online_transition_fraction_by_seed": {
            str(seed): count / max(args.online_steps, 1)
            for seed, count in steps_by_seed.items()
        },
        "online_episodes_by_seed": {
            str(seed): count for seed, count in episodes_by_seed.items()
        },
        "online_replay_size": len(replay),
        "correction_replay_size": len(correction_replay),
        "online_replay_fraction_by_seed": {
            str(seed): replay_counts.get(seed, 0) / replay_total for seed in ALL_SEEDS
        },
        "intervention_rate_by_seed": {
            str(seed): interventions_by_seed[seed] / max(steps_by_seed[seed], 1)
            for seed in ALL_SEEDS
        },
        "speed_bins_by_seed": {
            str(seed): speed_bins_by_seed[seed].astype(int).tolist()
            for seed in ALL_SEEDS
        },
        "training_hybrid_safety_failures_by_seed": {
            str(seed): sorted(safety_failures_by_seed[seed]) for seed in ALL_SEEDS
        },
        "task_critic_updated": _tree_changed(initial_q1, algorithm.base.state.params.q1),
        "proxy_updated": _tree_changed(initial_proxy, algorithm.base.state.params.proxy),
        "actor_updated": _tree_changed(initial_actor, algorithm.state.params.one_step),
        "teacher_frozen_online": True,
        "final_actor_diagnostics": latest,
        "hybrid_gate_history": gate_history,
        "full_evaluations": full_evaluations,
        "final_evaluation": full_evaluations[-1]["aggregate"],
        "best_overall": best,
    }
    return key, summary, best


def _format_number(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def _write_comparison(
    output_dir: Path,
    *,
    gaussian_summary,
    flow_offline,
    flow_online,
    online_steps: int,
    best,
):
    gaussian_offline = gaussian_summary.get("group_a_offline_only")
    gaussian_online = gaussian_summary.get("group_b_guardian_dsac")
    rows = []
    if gaussian_offline is not None:
        rows.append(("Gaussian offline-only", 0, gaussian_offline))
    if gaussian_online is not None:
        rows.append(("Gaussian V2.3.1", 150_000, gaussian_online))
    rows.append(("Flow offline-only", 0, flow_offline))
    if flow_online is not None:
        rows.append(("Guardian-Flow-DSAC V2.4", online_steps, flow_online))
    lines = [
        "# Guardian-Flow-DSAC V2.4 main comparison",
        "",
        "| Method | Online steps | Hybrid safe Seeds | Ordinary safe Seeds | Safe Ordinary mean return | Mean Hybrid intervention |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, steps, row in rows:
        lines.append(
            f"| {method} | {steps} | {row['hybrid_safe_seeds']}/10 | "
            f"{row['ordinary_safe_seeds']}/10 | "
            f"{_format_number(row['mean_ordinary_return_among_safe_seeds'])} | "
            f"{_format_number(row['mean_hybrid_intervention'], 5)} |"
        )
    lines.extend(
        [
            "",
            "Flow and Gaussian rows share the same five-Seed dataset, offline Q/P initialization, Seed0-9 schedule, 1200-step horizon, Guardian, gates and lexicographic checkpoint rule.",
            "",
            f"Flow `best_overall` was selected at step {best['step']}.",
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
        raise RuntimeError(f"formal V2.4 training requires GPU, got {backend}")
    source_config_path, source_config = _load_source_config(
        args.shared_offline_checkpoint
    )
    runtime_args = _runtime_args(args, source_config)
    manifest_path, manifest, dataset = _load_multiseed_dataset(runtime_args)
    scene_signatures = _predeclare_scene_signatures(runtime_args, dataset)
    if scene_signatures != {
        str(key): str(value)
        for key, value in source_config["scene_signatures"].items()
    }:
        raise RuntimeError("V2.4 scenes differ from the Gaussian baseline")
    key = jax.random.PRNGKey(args.seed + 24_000)
    key, flow_key = jax.random.split(key)
    algorithm, base_hidden_sizes, flow_hidden_sizes = _create_algorithm(
        runtime_args, source_config, dataset, flow_key
    )

    source_run_dir = source_config_path.parent
    gaussian_summary_path = source_run_dir / "summary.json"
    if not gaussian_summary_path.is_file():
        raise FileNotFoundError("Gaussian baseline summary.json is missing")
    gaussian_summary = json.loads(gaussian_summary_path.read_text(encoding="utf-8"))
    run_name = args.run_name or (
        ("smoke_" if args.component_smoke else "formal_")
        + time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir = args.output_root.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"
    config = {
        "format": "guardian_flow_dsac_multiseed_run_config_v1",
        "method": "Guardian-Flow-DSAC V2.4 multi-Seed",
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
        "hidden_sizes": list(base_hidden_sizes),
        "flow_hidden_sizes": list(flow_hidden_sizes),
        "flow_integration_steps": args.flow_integration_steps,
        "init_alpha": source_config["init_alpha"],
        "algorithm_config": asdict(algorithm.base.config),
        "flow_config": asdict(algorithm.config),
        "environment": _environment_config(runtime_args),
        "offline_seeds": list(OFFLINE_SEEDS),
        "online_novel_seeds": [5, 6, 7, 8, 9],
        "online_episode_schedule": list(ONLINE_EPISODE_SCHEDULE),
        "hybrid_gate_pairs": [list(pair) for pair in HYBRID_GATE_PAIRS],
        "hybrid_gate_every": args.hybrid_gate_every,
        "full_evaluation_every": args.full_evaluation_every,
        "budgets": {
            "flow_bc_gradient_steps": args.flow_bc_gradient_steps,
            "flow_distill_gradient_steps": args.flow_distill_gradient_steps,
            "flow_offline_joint_steps": args.flow_offline_joint_steps,
            "online_steps": args.online_steps,
            "batch_size": args.batch_size,
            "online_buffer_size": args.online_buffer_size,
            "correction_fraction": args.correction_fraction,
        },
        "shared_offline_checkpoint": str(
            args.shared_offline_checkpoint.expanduser().resolve()
        ),
        "shared_offline_checkpoint_sha256": _sha256(
            args.shared_offline_checkpoint.expanduser().resolve()
        ),
        "gaussian_baseline_summary": str(gaussian_summary_path),
        "gaussian_baseline_summary_sha256": _sha256(gaussian_summary_path),
        "candidate_sampling": False,
        "q_reranking": False,
        "rejection_sampling": False,
        "reverse_ode_likelihood": False,
        "flow_internal_cbf": False,
        "explicit_actor_entropy": False,
        "teacher_online_updates": False,
        "latent_scale": 1.0,
        "component_smoke": args.component_smoke,
        "backend": backend,
        "devices": [str(device) for device in jax.devices()],
    }
    _write_json(output_dir / "config.json", config)

    offline_checkpoint = output_dir / "offline_guardian_flow_dsac.pkl"
    if args.flow_offline_checkpoint is not None:
        source = args.flow_offline_checkpoint.expanduser().resolve()
        algorithm.load(str(source))
        offline_summary = {
            "loaded": True,
            "source": str(source),
            "source_sha256": _sha256(source),
            "teacher_frozen_after_distillation": algorithm.bc_flow_frozen,
            "validation": _multiseed_validation(algorithm, dataset),
            "speed_coverage": dataset.speed_coverage(),
        }
        algorithm.save(str(offline_checkpoint))
    else:
        key, offline_summary = _offline_train(
            algorithm,
            dataset,
            runtime_args,
            key=key,
            output_dir=output_dir,
            metrics_path=metrics_path,
        )
        algorithm.save(str(offline_checkpoint))
    offline_summary.update(
        checkpoint=str(offline_checkpoint),
        checkpoint_sha256=_sha256(offline_checkpoint),
    )
    _write_json(output_dir / "offline_summary.json", offline_summary)

    offline_evaluation = _run_isolated_evaluation(
        checkpoint=offline_checkpoint,
        output=output_dir / "evaluation_offline_initial_00000000.json",
        seeds=ALL_SEEDS,
        modes=("hybrid", "ordinary"),
        policy="learner",
    )
    offline_aggregate = _evaluation_summary(offline_evaluation)
    _write_json(output_dir / "flow_offline_only.json", offline_aggregate)
    if int(offline_aggregate["hybrid_safe_seeds"]) == len(ALL_SEEDS):
        algorithm.promote_hybrid_safe()
        offline_summary["offline_actor_promoted_hybrid_safe"] = True
    else:
        offline_summary["offline_actor_promoted_hybrid_safe"] = False
        offline_summary["safe_snapshot"] = "post-distillation pre-joint Actor"
    algorithm.save(str(offline_checkpoint))
    offline_summary.update(
        checkpoint=str(offline_checkpoint),
        checkpoint_sha256=_sha256(offline_checkpoint),
    )
    latest_safe = output_dir / "latest_hybrid_safe.pkl"
    algorithm.save(str(latest_safe))
    _write_json(output_dir / "offline_summary.json", offline_summary)
    best = _consider_checkpoint(
        algorithm,
        evaluation=offline_evaluation,
        aggregate=offline_aggregate,
        step=0,
        label="flow_offline_initial",
        output_dir=output_dir,
        best=None,
    )

    online_summary = None
    if args.phase in ("online", "full") and args.online_steps > 0:
        rng = np.random.default_rng(args.seed + 24_202_608)
        key, online_summary, best = _online_train(
            algorithm,
            dataset,
            runtime_args,
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
    _write_comparison(
        output_dir,
        gaussian_summary=gaussian_summary,
        flow_offline=offline_aggregate,
        flow_online=online_aggregate,
        online_steps=args.online_steps,
        best=best,
    )
    final_checkpoint = output_dir / "final_guardian_flow_dsac.pkl"
    algorithm.save(str(final_checkpoint))
    summary = {
        "format": "guardian_flow_dsac_multiseed_main_summary_v1",
        "method": "Guardian-Flow-DSAC V2.4",
        "dataset": dataset.summary(),
        "collection_manifest": manifest,
        "gaussian_baseline": gaussian_summary,
        "flow_offline_only": offline_aggregate,
        "flow_online": online_aggregate,
        "offline": offline_summary,
        "online": online_summary,
        "best_overall": best,
        "latest_hybrid_safe": {
            "checkpoint": str(latest_safe),
            "checkpoint_sha256": _sha256(latest_safe),
        },
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": _sha256(final_checkpoint),
        "wall_time_seconds": time.time() - started,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "flow_offline": offline_aggregate,
                "flow_online": online_aggregate,
                "best_overall": best,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
