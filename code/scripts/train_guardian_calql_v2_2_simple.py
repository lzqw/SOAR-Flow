#!/usr/bin/env python3
"""Strict cross-seed trainer for Guardian-CalQL V2.2-Simple.

The public entry point is ``train_guardian_calql_metasafe.py --method
v2_2_simple``.  Keeping this implementation separate makes the historical
V2.1 training path bit-for-bit runnable while avoiding compatibility branches
inside the V2.2 losses and state machine.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
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

from relax.utils.action_semantics import (
    FIXED_SPEED_ACTION_SEMANTICS,
    SPEED_COUPLED_ACTION_SEMANTICS,
)
from relax.utils.guardian_cross_seed_gate import (
    CrossSeedGateConfig,
    cross_seed_candidate_gate,
    evaluation_binding_failures,
    hard_safety_failures,
)
from scripts.scan_metasafe_seeds import _canonical_hash
from scripts.train_guardian_calql_metasafe import (
    _actor_action_for_mpc,
    _append_jsonl,
    _audit_low_level_cbf_authority,
    _concatenate_batches,
    _file_sha256,
    _finalize_episode_returns,
    _make_environment,
    _policy_hash,
    _step_experience,
    _write_json,
)


def _parse_seeds(
    value: str, *, name: str, allow_empty: bool = False
) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not seeds and not allow_empty:
        raise ValueError(f"{name} must contain at least one seed")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{name} contains duplicate seeds")
    return seeds


def _proxy_delta_for_trust(
    observed_delta: float,
    *,
    proxy_actor_weight: float,
    non_degradation_margin: float,
) -> tuple[float, bool]:
    """Remove the indirect P-to-Actor path in the no-P ablation.

    The P network remains available for diagnostics and numerical audits in
    Experiment C, but a disabled relative-P Actor term must also prevent P from
    tightening or relaxing the Champion trust coefficient.
    """

    enabled = bool(float(proxy_actor_weight) > 0.0)
    return (
        float(observed_delta) if enabled else float(non_degradation_margin),
        enabled,
    )


def _load_manifest(
    args,
) -> tuple[dict[str, Any], str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    path = args.scene_selection_manifest.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = str(manifest.get("scene_selection_manifest_hash", ""))
    unhashed = dict(manifest)
    unhashed.pop("scene_selection_manifest_hash", None)
    actual_hash = _canonical_hash(unhashed)
    if not claimed_hash or claimed_hash != actual_hash:
        raise ValueError("scene selection manifest hash is missing or invalid")
    if not bool(manifest.get("created_before_online_training", False)):
        raise ValueError("scene selection manifest was not predeclared")
    if bool(manifest.get("selection_uses_online_candidate_results", True)):
        raise ValueError("scene selection manifest used post-training evidence")
    if manifest.get("action_semantics_version") != args.action_semantics:
        raise ValueError("manifest action semantics mismatch")
    if not np.isclose(
        float(manifest.get("max_command_speed", np.nan)),
        float(args.max_command_speed),
        atol=1e-8,
    ):
        raise ValueError("manifest max command speed mismatch")

    declared_online = tuple(int(seed) for seed in manifest["online_scene_seeds"])
    declared_retention = tuple(int(seed) for seed in manifest["retention_scene_seeds"])
    declared_heldout = tuple(int(seed) for seed in manifest["heldout_scene_seeds"])
    requested_online = _parse_seeds(
        args.online_scene_seeds, name="online scene seeds", allow_empty=True
    )
    requested_retention = _parse_seeds(
        args.retention_scene_seeds, name="retention scene seeds"
    )
    requested_heldout = _parse_seeds(
        args.heldout_scene_seeds, name="held-out scene seeds", allow_empty=True
    )
    online = requested_online or declared_online
    retention = requested_retention or declared_retention
    heldout = requested_heldout or declared_heldout
    if online != declared_online:
        raise ValueError("online scene seeds differ from predeclared manifest")
    if retention != declared_retention:
        raise ValueError("retention scene seeds differ from predeclared manifest")
    if heldout != declared_heldout:
        raise ValueError("held-out scene seeds differ from predeclared manifest")
    if int(args.offline_scene_seed) != int(manifest["offline_scene_seed"]):
        raise ValueError("offline scene seed differs from predeclared manifest")
    all_roles = set(online) | set(retention) | set(heldout)
    if set(online) & set(heldout) or set(retention) & set(heldout):
        raise ValueError("held-out seeds may not participate in training or retention")
    signatures = manifest.get("scene_signatures", {})
    missing = [seed for seed in all_roles if str(seed) not in signatures]
    if missing:
        raise ValueError(f"manifest lacks scene signatures for {missing}")
    return manifest, claimed_hash, online, retention, heldout


def _offline_bootstrap_binding(args):
    dataset_path = args.dataset.expanduser().resolve()
    with np.load(dataset_path, allow_pickle=False) as archive:
        if "scene_seed" not in archive or "scene_signature" not in archive:
            raise ValueError("V2.2 bootstrap dataset lacks scene binding metadata")
        seeds = np.unique(np.asarray(archive["scene_seed"], dtype=np.int32))
        signatures = np.unique(np.asarray(archive["scene_signature"]).astype(str))
    if seeds.tolist() != [int(args.offline_scene_seed)] or signatures.size != 1:
        raise ValueError("offline bootstrap dataset is not bound to one expected scene")
    manifest = {
        "format": "guardian_v2_2_offline_bootstrap_binding_v1",
        "created_before_online_training": True,
        "selection_uses_online_candidate_results": False,
        "offline_scene_seed": int(args.offline_scene_seed),
        "online_scene_seeds": [],
        "retention_scene_seeds": [int(args.offline_scene_seed)],
        "heldout_scene_seeds": [],
        "action_semantics_version": args.action_semantics,
        "max_command_speed": float(args.max_command_speed),
        "scene_signatures": {str(int(args.offline_scene_seed)): str(signatures[0])},
    }
    manifest_hash = _canonical_hash(manifest)
    manifest["scene_selection_manifest_hash"] = manifest_hash
    return (
        manifest,
        manifest_hash,
        (),
        (int(args.offline_scene_seed),),
        (),
    )


def _offline_metrics(
    algorithm, dataset, split: str, batch_size: int
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for batch in dataset.iter_split(split, batch_size=batch_size, to_jax=False):
        size = int(np.asarray(batch.reward).shape[0])
        q1, q2 = algorithm.task_q(batch.obs, batch.action)
        q_lcb = 0.5 * (q1 + q2) - algorithm.config.q_lcb_beta * 0.5 * np.abs(q1 - q2)
        p_raw = algorithm.proxy_value(batch.obs, batch.raw_action)
        p_corr = algorithm.proxy_value(batch.obs, batch.proxy_corrected_action)
        raw_valid = np.asarray(batch.guardian_dependence_target_valid, dtype=bool)
        corr_valid = np.asarray(batch.guardian_dependence_corrected_valid, dtype=bool)
        target = np.asarray(batch.guardian_dependence_target, dtype=np.float32)
        actor_action = algorithm.get_deterministic_action(batch.obs)
        champion_action = algorithm.get_deterministic_champion_action(batch.obs)
        p_policy = algorithm.proxy_value(batch.obs, actor_action)
        p_champion = algorithm.proxy_value(batch.obs, champion_action)
        q1_policy, q2_policy = algorithm.task_q(batch.obs, actor_action)
        q1_champion, q2_champion = algorithm.task_q(batch.obs, champion_action)
        q_policy = 0.5 * (q1_policy + q2_policy) - (
            algorithm.config.q_lcb_beta * 0.5 * np.abs(q1_policy - q2_policy)
        )
        q_champion = 0.5 * (q1_champion + q2_champion) - (
            algorithm.config.q_lcb_beta * 0.5 * np.abs(q1_champion - q2_champion)
        )
        teacher_valid = np.asarray(batch.teacher_valid, dtype=bool)
        bc_error = np.sum(
            np.square(actor_action - np.asarray(batch.teacher_action)), axis=-1
        )
        row = {
            "q_calibration_mae": float(
                np.mean(
                    np.abs(
                        q_lcb
                        - np.asarray(batch.behavior_return_to_go, dtype=np.float32)
                    )
                )
            ),
            "q_disagreement": float(np.mean(np.abs(q1 - q2))),
            "proxy_mae": float(
                np.mean(np.abs(p_raw[raw_valid] - target[raw_valid]))
                if np.any(raw_valid)
                else 0.0
            ),
            "proxy_brier": float(
                np.mean(np.square(p_raw[raw_valid] - target[raw_valid]))
                if np.any(raw_valid)
                else 0.0
            ),
            "proxy_pair_accuracy": float(
                np.mean(p_raw[corr_valid] > p_corr[corr_valid])
                if np.any(corr_valid)
                else 0.0
            ),
            "p_policy_minus_champion": float(np.mean(p_policy - p_champion)),
            "p_improvement_fraction": float(np.mean(p_policy <= p_champion)),
            "q_policy_minus_champion": float(np.mean(q_policy - q_champion)),
            "bc_mse": float(
                np.mean(bc_error[teacher_valid]) if np.any(teacher_valid) else 0.0
            ),
            "actor_support_fraction": float(np.mean(batch.actor_controlled)),
        }
        for name, value in row.items():
            totals[name] = totals.get(name, 0.0) + value * size
        count += size
    return {name: value / max(count, 1) for name, value in totals.items()}


def _sample_mixed_batch(
    dataset,
    online_replay,
    *,
    batch_size: int,
    offline_ratio: float,
    rng: np.random.Generator,
):
    online_count = (
        min(int(round(batch_size * (1.0 - offline_ratio))), batch_size - 1)
        if len(online_replay) > 0
        else 0
    )
    offline_count = batch_size - online_count
    batches = []
    if offline_count:
        batches.append(dataset.sample(offline_count, split="train", to_jax=False))
    if online_count:
        batches.append(online_replay.sample(online_count, to_jax=False))
    return _concatenate_batches(batches, rng)


def _run_isolated_evaluation(
    *,
    checkpoint: Path,
    output: Path,
    scene_seeds: tuple[int, ...],
    policy: str,
    episodes: int,
    manifest_path: Path,
    args,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/evaluate_guardian_calql_isolated.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--scene-seeds",
        ",".join(str(seed) for seed in scene_seeds),
        "--episodes",
        str(episodes),
        "--policy",
        policy,
        "--scene-selection-manifest",
        str(manifest_path),
        "--action-semantics",
        args.action_semantics,
        "--max-command-speed",
        str(args.max_command_speed),
    ]
    completed = subprocess.run(
        command,
        cwd=str(REPOSITORY_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output.with_suffix(".log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode not in (0, 2) or not output.is_file():
        raise RuntimeError(
            f"isolated evaluation failed; see {output.with_suffix('.log')}"
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected = {str(seed) for seed in scene_seeds}
    if set(payload.get("per_seed", {})) != expected:
        raise RuntimeError("isolated evaluation omitted a requested scene seed")
    return payload


def _binding_failures(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    signatures: Mapping[str, str],
    manifest_hash: str,
    action_semantics: str,
) -> list[str]:
    failures: list[str] = []
    for seed_text, artifact in artifacts.items():
        failures.extend(
            f"seed_{seed_text}:{name}"
            for name in evaluation_binding_failures(
                artifact,
                expected_seed=int(seed_text),
                expected_signature=signatures[seed_text],
                expected_manifest_hash=manifest_hash,
                expected_action_semantics=action_semantics,
            )
        )
    return failures


def _initial_champion_failures(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    online_seeds: tuple[int, ...],
    retention_seeds: tuple[int, ...],
) -> list[str]:
    """Require Hybrid safety without assuming initial Ordinary competence.

    Seed0 bootstrap selection and new-scene selection both intentionally use
    only the Guardian-protected Hybrid hard-safety gate.  Ordinary performance
    is audited and frozen as the adaptation baseline, but it must not prevent
    online training from starting.  Candidate promotion remains stricter and
    requires both modes to be safe on online and retention seeds.
    """

    failures: list[str] = []
    seed_roles = [("retention", seed) for seed in retention_seeds]
    seed_roles += [("online", seed) for seed in online_seeds]
    for role, seed in seed_roles:
        artifact = artifacts.get(str(seed), {})
        hybrid = artifact.get("results", {}).get("hybrid")
        label = f"{role}_seed_{seed}:hybrid"
        if not isinstance(hybrid, Mapping):
            failures.append(f"{label}:missing")
            continue
        if float(hybrid.get("success_rate", 0.0)) < 1.0:
            failures.append(f"{label}:success")
        if float(hybrid.get("collision_rate", 1.0)) > 0.0:
            failures.append(f"{label}:collision")
        if float(hybrid.get("road_departure_rate", 1.0)) > 0.0:
            failures.append(f"{label}:road_departure")
        if float(hybrid.get("total_environment_cost", np.inf)) > 0.0:
            failures.append(f"{label}:environment_cost")
        if float(hybrid.get("solver_finite_rate", 0.0)) < 1.0:
            failures.append(f"{label}:solver_finite")
        if not float(hybrid.get("min_physical_body_sat_margin", -np.inf)) > 0.0:
            failures.append(f"{label}:physical_margin")
    return sorted(set(failures))


def _online_step_safety_failures(info: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if bool(info.get("crash_vehicle", False) or info.get("crash_object", False)):
        failures.append("collision")
    if bool(info.get("out_of_road", False) or info.get("road_departure", False)):
        failures.append("road_departure")
    if max(float(info.get("cost", 0.0)), 0.0) > 0.0:
        failures.append("environment_cost")
    if not bool(info.get("mgf", {}).get("solver_finite", False)):
        failures.append("solver_nonfinite")
    physical_margin = float(
        info.get("mpc_control", {}).get("physical_body_sat_margin", -np.inf)
    )
    if not np.isfinite(physical_margin) or physical_margin <= 0.0:
        failures.append("physical_margin")
    return failures


def _collect_online(
    algorithm,
    dataset,
    args,
    *,
    online_seed: int,
    expected_signature: str,
    manifest_path: Path,
    manifest_hash: str,
    signatures: Mapping[str, str],
    output_dir: Path,
    metrics_path: Path,
    champion_evaluations: dict[int, Mapping[str, Any]],
    key,
    rng: np.random.Generator,
):
    import jax
    from relax.buffer.tree import TreeBuffer

    replay = TreeBuffer.from_guardian_calql_experience(
        dataset.summary.obs_dim,
        dataset.summary.action_dim,
        args.online_buffer_size,
        seed=args.seed + 310,
    )
    gate_config = CrossSeedGateConfig(
        online_return_nondegradation_tolerance=(
            args.online_return_nondegradation_tolerance
        ),
        online_intervention_nondegradation_pp=(
            args.online_intervention_nondegradation_pp
        ),
        online_return_strict_improvement=(args.online_return_strict_improvement),
        online_intervention_strict_improvement_pp=(
            args.online_intervention_strict_improvement_pp
        ),
        retention_return_tolerance=args.retention_return_tolerance,
        retention_intervention_tolerance_pp=(args.retention_intervention_tolerance_pp),
    )
    counters = {
        "online_environment_steps": 0,
        "online_episodes": 0,
        "task_critic_updates": 0,
        "proxy_updates": 0,
        "actor_updates": 0,
        "candidate_evaluations": 0,
        "champion_promotions": 0,
        "performance_rejections": 0,
        "safety_rejections": 0,
        "numerical_disaster_rollbacks": 0,
        "rtg_p_finalized_transitions": 0,
    }
    if args.online_steps <= 0:
        return key, counters, champion_evaluations

    original_fixed_seed = args.fixed_scene_seed
    args.fixed_scene_seed = int(online_seed)
    env = _make_environment(args, "hybrid")
    episode: list[Any] = []
    observation, reset_info = env.reset(seed=online_seed)
    if reset_info.get("metasafe_scene_signature") != expected_signature:
        env.close()
        raise RuntimeError("online environment scene signature mismatch")
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    episode_step = 0
    episode_safety_failures: set[str] = set()
    episode_safety_rejected = False
    next_gate = max(int(args.candidate_every), 1)
    latest: dict[str, float] = {}

    def reset_environment(current_env):
        current_env.close()
        next_env = _make_environment(args, "hybrid")
        next_observation, info = next_env.reset(seed=online_seed)
        if info.get("metasafe_scene_signature") != expected_signature:
            next_env.close()
            raise RuntimeError("online reset changed the fixed scene")
        return next_env, np.asarray(next_observation, np.float32).reshape(-1)

    try:
        while counters["online_environment_steps"] < args.online_steps or episode:
            raw_action = algorithm.get_deterministic_action(observation)
            raw_action = np.clip(
                np.asarray(raw_action, dtype=np.float32)
                + rng.normal(0.0, args.online_exploration_std, size=2),
                -1.0,
                1.0,
            ).astype(np.float32)
            certificate = env.certify_target_action(raw_action)
            authority = _audit_low_level_cbf_authority(
                env,
                raw_action,
                certificate,
                trajectory_tolerance=args.cbf_trajectory_tolerance,
                derive_teacher=False,
            )
            next_observation, reward, terminated, truncated, info = (
                env.execute_certified_target(certificate)
            )
            episode_step += 1
            truncated = bool(truncated or episode_step >= args.max_episode_steps)
            transition = _step_experience(
                observation,
                raw_action,
                reward,
                terminated,
                truncated,
                next_observation,
                certificate,
                info,
                soft_pair_weight=args.soft_pair_weight,
                authority_audit=authority,
                scene_seed=online_seed,
                scene_signature=expected_signature,
                action_semantics_version=args.action_semantics,
            )
            episode.append(transition)
            episode_safety_failures.update(_online_step_safety_failures(info))
            counters["online_environment_steps"] += 1
            done = bool(terminated or truncated)
            observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
            if done:
                finalized = _finalize_episode_returns(
                    episode,
                    gamma=args.discount,
                    proxy_discount=args.proxy_discount,
                    q_anchor_valid=False,
                )
                for item in finalized:
                    replay.add(item)
                counters["rtg_p_finalized_transitions"] += len(finalized)
                counters["online_episodes"] += 1
                episode.clear()

            progress = min(
                counters["online_environment_steps"] / max(args.online_steps, 1), 1.0
            )
            offline_ratio = args.offline_ratio_initial + progress * (
                args.offline_ratio_final - args.offline_ratio_initial
            )
            numerical_disaster_this_step = False
            if len(replay) >= args.learning_starts:
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
                        counters["online_environment_steps"] >= args.warmup_min_steps
                        and (counters["online_environment_steps"] + gradient_index)
                        % args.policy_delay
                        == 0
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
                        rollback = algorithm.reject_numerical_disaster()
                        numerical_disaster_this_step = True
                        counters["numerical_disaster_rollbacks"] += 1
                        _append_jsonl(
                            metrics_path,
                            {
                                "phase": "numerical_disaster",
                                **counters,
                                "reasons": disaster,
                                **rollback,
                            },
                        )
                        break

            if episode_safety_failures and not episode_safety_rejected:
                if numerical_disaster_this_step:
                    # The accepted full snapshot already restored the Actor.
                    episode_safety_rejected = True
                else:
                    rollback = algorithm.reject_safety()
                    episode_safety_rejected = True
                    counters["safety_rejections"] += 1
                    _append_jsonl(
                        metrics_path,
                        {
                            "phase": "online_step_safety_reject",
                            **counters,
                            "reasons": sorted(episode_safety_failures),
                            **rollback,
                        },
                    )

            if done:
                env, observation = reset_environment(env)
                episode_step = 0
                episode_safety_failures.clear()
                episode_safety_rejected = False

            if counters["online_environment_steps"] % args.log_every == 0:
                _append_jsonl(
                    metrics_path,
                    {
                        "phase": "online_joint_v2_2_simple",
                        **counters,
                        "offline_ratio": offline_ratio,
                        "online_replay_size": len(replay),
                        **latest,
                    },
                )

            online_step = counters["online_environment_steps"]
            if args.candidate_every > 0 and done and online_step >= next_gate:
                while next_gate <= online_step:
                    next_gate += args.candidate_every
                candidate_path = output_dir / f"candidate_{online_step:08d}.pkl"
                algorithm.save(str(candidate_path))
                seeds = tuple(
                    sorted(
                        set(
                            (
                                online_seed,
                                *[
                                    int(s)
                                    for s in args.retention_scene_seeds.split(",")
                                    if s.strip()
                                ],
                            )
                        )
                    )
                )
                candidate_payload = _run_isolated_evaluation(
                    checkpoint=candidate_path,
                    output=output_dir / f"candidate_{online_step:08d}_cross_seed.json",
                    scene_seeds=seeds,
                    policy="learner",
                    episodes=args.candidate_eval_episodes,
                    manifest_path=manifest_path,
                    args=args,
                )
                candidate_evaluations = {
                    int(seed): artifact
                    for seed, artifact in candidate_payload["per_seed"].items()
                }
                bindings = _binding_failures(
                    candidate_payload["per_seed"],
                    signatures=signatures,
                    manifest_hash=manifest_hash,
                    action_semantics=args.action_semantics,
                )
                gate = cross_seed_candidate_gate(
                    candidate_by_seed=candidate_evaluations,
                    champion_by_seed=champion_evaluations,
                    online_scene_seeds=(online_seed,),
                    retention_scene_seeds=tuple(
                        int(seed)
                        for seed in args.retention_scene_seeds.split(",")
                        if seed.strip()
                    ),
                    config=gate_config,
                )
                gate["failures"].extend(bindings)
                gate["failures"] = sorted(set(gate["failures"]))
                gate["passed"] = bool(gate["passed"] and not bindings)
                validation = _offline_metrics(
                    algorithm,
                    dataset,
                    "validation",
                    args.evaluation_batch_size,
                )
                observed_p_delta = validation["p_policy_minus_champion"]
                trust_p_delta, trust_p_enabled = _proxy_delta_for_trust(
                    observed_p_delta,
                    proxy_actor_weight=args.proxy_actor_weight,
                    non_degradation_margin=algorithm.config.p_non_degradation_margin,
                )
                trust = algorithm.adapt_trust_from_validation(
                    q_policy_minus_champion=validation["q_policy_minus_champion"],
                    p_policy_minus_champion=trust_p_delta,
                )
                trust["p_constraint_enabled"] = trust_p_enabled
                trust["observed_p_policy_minus_champion"] = observed_p_delta
                counters["candidate_evaluations"] += 1
                if gate["passed"]:
                    algorithm.promote_champion()
                    champion_evaluations = candidate_evaluations
                    counters["champion_promotions"] += 1
                    disposition = {"disposition": "promote"}
                elif any(
                    failure.startswith("candidate_seed_")
                    and any(
                        token in failure
                        for token in (
                            ":collision",
                            ":road_departure",
                            ":environment_cost",
                            ":solver_finite",
                            ":physical_margin",
                            ":success",
                            ":actual_intervention",
                        )
                    )
                    for failure in gate["failures"]
                ):
                    disposition = algorithm.reject_safety()
                    counters["safety_rejections"] += 1
                else:
                    disposition = algorithm.reject_performance()
                    counters["performance_rejections"] += 1
                gate_record = {
                    "phase": "cross_seed_candidate_gate",
                    **counters,
                    "candidate_checkpoint": str(candidate_path),
                    "candidate_checkpoint_sha256": _file_sha256(candidate_path),
                    "scene_selection_manifest_hash": manifest_hash,
                    "gate": gate,
                    "offline_validation": validation,
                    "trust_validation": trust,
                    **disposition,
                }
                _append_jsonl(metrics_path, gate_record)
                _write_json(
                    output_dir / f"candidate_{online_step:08d}_gate.json",
                    gate_record,
                )
                algorithm.save(str(output_dir / f"checkpoint_{online_step:08d}.pkl"))
                if gate["passed"] and args.stop_after_first_promotion:
                    break
    finally:
        env.close()
        args.fixed_scene_seed = original_fixed_seed
    counters["online_replay_size"] = len(replay)
    return key, counters, champion_evaluations


def main_from_args(args) -> int:
    if args.action_semantics != SPEED_COUPLED_ACTION_SEMANTICS and not (
        args.allow_fixed_speed_ablation
        and args.action_semantics == FIXED_SPEED_ACTION_SEMANTICS
    ):
        raise ValueError("V2.2-Simple requires speed-coupled action semantics")
    if args.bootstrap_offline_only:
        manifest, manifest_hash, online_seeds, retention_seeds, heldout_seeds = (
            _offline_bootstrap_binding(args)
        )
    else:
        manifest, manifest_hash, online_seeds, retention_seeds, heldout_seeds = (
            _load_manifest(args)
        )
    if not args.bootstrap_offline_only and len(online_seeds) != 1:
        raise ValueError(
            "the current formal V2.2 runner requires exactly one online seed"
        )
    args.online_scene_seeds = ",".join(str(seed) for seed in online_seeds)
    args.retention_scene_seeds = ",".join(str(seed) for seed in retention_seeds)
    args.heldout_scene_seeds = ",".join(str(seed) for seed in heldout_seeds)
    signatures = {
        str(seed): str(value) for seed, value in manifest["scene_signatures"].items()
    }

    import jax
    from relax.algorithm.guardian_calql_v2_2_simple import (
        GuardianCalQLV22Config,
        GuardianCalQLV22SimpleSafeSAC,
    )
    from relax.network.safe_sac_jax import create_guardian_calql_net
    from relax.offline.metasafe_guardian_dataset import GuardianMetaSafeDataset

    started = time.time()
    backend = jax.default_backend()
    devices = [str(device) for device in jax.devices()]
    if args.require_gpu and backend != "gpu":
        raise RuntimeError(
            f"formal training requires GPU, got backend={backend}, devices={devices}"
        )
    dataset = GuardianMetaSafeDataset(
        args.dataset,
        gamma=args.discount,
        proxy_discount=args.proxy_discount,
        split_seed=args.seed,
        soft_pair_weight=args.soft_pair_weight,
        soft_teacher_weight=args.soft_teacher_weight,
        exclude_legacy_reset_step=args.post_reset_warmup_steps > 0,
        expected_sha256=args.expected_dataset_sha256 or None,
        expected_action_semantics_version=args.action_semantics,
        expected_scene_seed=args.offline_scene_seed,
        expected_scene_signature=signatures[str(args.offline_scene_seed)],
        expected_scene_selection_manifest_hash=(
            None if args.bootstrap_offline_only else manifest_hash
        ),
        expected_lookahead_distance=args.lookahead,
        expected_max_command_speed=args.max_command_speed,
        expected_speed_mapping=(
            "fixed_internal_target_velocity"
            if args.action_semantics == FIXED_SPEED_ACTION_SEMANTICS
            else "v=max_command_speed*(action_x+1)/2"
        ),
        expected_mpc_horizon=args.horizon,
    )
    hidden_sizes = tuple(
        int(item.strip()) for item in args.hidden_sizes.split(",") if item.strip()
    )
    key = jax.random.PRNGKey(args.seed)
    key, network_key = jax.random.split(key)
    network, params = create_guardian_calql_net(
        network_key,
        critic_obs_dim=dataset.summary.obs_dim,
        actor_obs_dim=dataset.summary.obs_dim,
        act_dim=dataset.summary.action_dim,
        hidden_sizes=hidden_sizes,
        init_alpha=args.init_alpha,
    )
    config = GuardianCalQLV22Config(
        gamma=args.discount,
        tau=args.tau,
        critic_learning_rate=args.critic_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        proxy_learning_rate=args.proxy_learning_rate,
        alpha_learning_rate=args.alpha_learning_rate,
        q_lcb_beta=args.q_lcb_beta,
        calql_weight=args.calql_weight,
        q_anchor_weight_initial=args.q_anchor_weight_initial,
        q_anchor_weight_final=args.q_anchor_weight_final,
        q_anchor_decay_steps=args.q_anchor_decay_steps,
        proxy_corrected_weight=args.proxy_corrected_weight,
        proxy_actor_weight=args.proxy_actor_weight,
        proxy_relative_margin=args.proxy_relative_margin,
        actor_sac_weight=args.actor_sac_weight,
        trust_weight=args.trust_weight,
        trust_action_weight=args.champion_action_weight,
        trust_relax_multiplier=args.trust_relax_multiplier,
        trust_tighten_multiplier=args.trust_tighten_multiplier,
        trust_min=args.trust_min,
        trust_max=args.trust_max,
        q_disaster_threshold=max(args.critic_rollback_factor, 1.0) * 1e3,
    )
    checkpoint_metadata = {
        "action_semantics_version": args.action_semantics,
        "offline_scene_seed": int(args.offline_scene_seed),
        "online_scene_seeds": list(online_seeds),
        "scene_signatures": {
            str(seed): signatures[str(seed)]
            for seed in set(
                (
                    args.offline_scene_seed,
                    *online_seeds,
                    *retention_seeds,
                    *heldout_seeds,
                )
            )
        },
        "scene_selection_manifest_hash": manifest_hash,
        "fixed_speed_ablation": bool(args.allow_fixed_speed_ablation),
    }
    algorithm = GuardianCalQLV22SimpleSafeSAC(
        network,
        params,
        config=config,
        allow_fixed_speed_ablation=bool(args.allow_fixed_speed_ablation),
        **{
            name: value
            for name, value in checkpoint_metadata.items()
            if name != "fixed_speed_ablation"
        },
    )
    run_name = args.run_name or (
        f"v2_2_simple_seed0_to_seed{online_seeds[0]}_trainseed{args.seed}"
    )
    output_dir = (args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "config.json"
    _write_json(
        config_path,
        {
            "args": vars(args),
            "algorithm_config": asdict(config),
            "checkpoint_metadata": checkpoint_metadata,
            "dataset": {**asdict(dataset.summary), **dataset.action_metadata},
            "split_episode_ids": dataset.split_episode_ids(),
            "scene_selection_manifest": (
                str(args.scene_selection_manifest.expanduser().resolve())
                if args.scene_selection_manifest is not None
                else None
            ),
            "scene_selection_manifest_hash": manifest_hash,
            "jax_backend": backend,
            "jax_devices": devices,
            "jax_version": jax.__version__,
        },
    )
    rng = np.random.default_rng(args.seed + 77)
    resumed = False
    if args.online_initial_checkpoint is not None:
        checkpoint = args.online_initial_checkpoint.expanduser().resolve()
        algorithm.load(str(checkpoint))
        resumed = True
    if args.online_initial_policy_checkpoint is not None:
        from scripts.train_guardian_calql_metasafe import (
            _load_compatible_actor_policy,
        )

        policy_path = args.online_initial_policy_checkpoint.expanduser().resolve()
        imported_policy, source_kind = _load_compatible_actor_policy(
            policy_path,
            algorithm.get_policy_params(),
            allow_width_expansion=bool(args.allow_fixed_speed_ablation),
        )
        algorithm.state = algorithm.state._replace(
            params=algorithm.state.params._replace(policy=imported_policy),
            opt_state=algorithm.state.opt_state._replace(
                policy=algorithm.actor_optim.init(imported_policy)
            ),
        )
        _write_json(
            output_dir / "imported_initial_policy.json",
            {
                "checkpoint": str(policy_path),
                "checkpoint_sha256": _file_sha256(policy_path),
                "source_kind": source_kind,
                "function_preserving_width_expansion": bool(
                    source_kind.endswith("+zero_padded_hidden_width_expansion")
                ),
                "policy_sha256": _policy_hash(imported_policy),
            },
        )

    # Offline Actor initialization is teacher BC only.
    for step in range(1, args.bc_gradient_steps + 1):
        batch = dataset.sample(
            args.batch_size,
            split="train",
            teacher_fraction=min(args.bc_intervention_ratio, 0.49),
            to_jax=True,
        )
        key, update_key = jax.random.split(key)
        latest = algorithm.pretrain_bc(update_key, batch)
        if step % args.log_every == 0 or step == args.bc_gradient_steps:
            _append_jsonl(metrics_path, {"phase": "offline_bc", "step": step, **latest})
        if args.bootstrap_offline_only and (
            step % args.validation_every == 0 or step == args.bc_gradient_steps
        ):
            # Seed selection must use a named, immutable initial Champion.  Save
            # periodic BC states so the earliest Seed-0 closed-loop-safe policy
            # can be chosen without looking at any candidate online seed.
            algorithm.mark_policy_as_initial()
            checkpoint_path = output_dir / f"bc_champion_{step:08d}.pkl"
            algorithm.save(str(checkpoint_path))
            _append_jsonl(
                metrics_path,
                {
                    "phase": "offline_bc_champion_checkpoint",
                    "step": step,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": _file_sha256(checkpoint_path),
                },
            )
    if args.bc_gradient_steps > 0 or not resumed:
        algorithm.mark_policy_as_initial()
    algorithm.save(str(output_dir / "stage1_bc.pkl"))

    for step in range(1, args.critic_gradient_steps + 1):
        batch = dataset.sample(args.batch_size, split="train", to_jax=True)
        key, update_key = jax.random.split(key)
        latest = algorithm.offline_critic_update(update_key, batch, mode="calql")
        if step % args.log_every == 0 or step == args.critic_gradient_steps:
            _append_jsonl(
                metrics_path, {"phase": "offline_calql", "step": step, **latest}
            )
    algorithm.save(str(output_dir / "stage2_offline_calql.pkl"))

    for step in range(1, args.proxy_gradient_steps + 1):
        batch = dataset.sample(
            args.batch_size,
            split="train",
            pair_fraction=min(args.intervention_ratio, 0.49),
            to_jax=True,
        )
        key, update_key = jax.random.split(key)
        latest = algorithm.offline_proxy_update(update_key, batch)
        if step % args.log_every == 0 or step == args.proxy_gradient_steps:
            _append_jsonl(
                metrics_path,
                {"phase": "offline_proxy_episode_target", "step": step, **latest},
            )
    if args.critic_gradient_steps > 0 or not algorithm.offline_q_reference_ready:
        algorithm.mark_offline_q_reference()
    if (
        args.bc_gradient_steps > 0
        or args.critic_gradient_steps > 0
        or args.proxy_gradient_steps > 0
        or not resumed
    ):
        algorithm.promote_champion()
    stage3 = output_dir / "stage3_offline_champion.pkl"
    algorithm.save(str(stage3))
    offline_validation = _offline_metrics(
        algorithm, dataset, "validation", args.evaluation_batch_size
    )
    _write_json(output_dir / "offline_validation.json", offline_validation)

    if args.bootstrap_offline_only:
        _write_json(output_dir / "bootstrap_scene_binding.json", manifest)
        final_checkpoint = output_dir / "final_champion.pkl"
        algorithm.save(str(final_checkpoint))
        summary = {
            "method": "v2_2_simple_offline_bootstrap",
            "format": algorithm.FORMAT,
            "run_directory": str(output_dir),
            "elapsed_seconds": time.time() - started,
            "dataset_sha256": dataset.sha256,
            "offline_scene_seed": args.offline_scene_seed,
            "action_semantics_version": args.action_semantics,
            "offline_validation": offline_validation,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": _file_sha256(final_checkpoint),
            "final_champion_policy_sha256": _policy_hash(algorithm.champion_policy),
            "online_training_performed": False,
            "seed_selection_performed": False,
        }
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0

    gate_seeds = tuple(sorted(set((*online_seeds, *retention_seeds))))
    initial_eval_payload = _run_isolated_evaluation(
        checkpoint=stage3,
        output=output_dir / "initial_champion_cross_seed.json",
        scene_seeds=gate_seeds,
        policy="champion",
        episodes=args.candidate_eval_episodes,
        manifest_path=args.scene_selection_manifest.expanduser().resolve(),
        args=args,
    )
    binding_failures = _binding_failures(
        initial_eval_payload["per_seed"],
        signatures=signatures,
        manifest_hash=manifest_hash,
        action_semantics=args.action_semantics,
    )
    initial_safety_failures = _initial_champion_failures(
        initial_eval_payload["per_seed"],
        online_seeds=tuple(online_seeds),
        retention_seeds=tuple(retention_seeds),
    )
    if binding_failures or initial_safety_failures:
        raise RuntimeError(
            "initial Champion failed required protected safety/binding gate: "
            + ", ".join(binding_failures + initial_safety_failures)
        )
    champion_evaluations = {
        int(seed): artifact
        for seed, artifact in initial_eval_payload["per_seed"].items()
    }

    key, online_summary, champion_evaluations = _collect_online(
        algorithm,
        dataset,
        args,
        online_seed=online_seeds[0],
        expected_signature=signatures[str(online_seeds[0])],
        manifest_path=args.scene_selection_manifest.expanduser().resolve(),
        manifest_hash=manifest_hash,
        signatures=signatures,
        output_dir=output_dir,
        metrics_path=metrics_path,
        champion_evaluations=champion_evaluations,
        key=key,
        rng=rng,
    )
    final_checkpoint = output_dir / "final_champion.pkl"
    algorithm.save(str(final_checkpoint))
    final_gate = _run_isolated_evaluation(
        checkpoint=final_checkpoint,
        output=output_dir / "final_champion_cross_seed.json",
        scene_seeds=gate_seeds,
        policy="champion",
        episodes=args.candidate_eval_episodes,
        manifest_path=args.scene_selection_manifest.expanduser().resolve(),
        args=args,
    )
    heldout = _run_isolated_evaluation(
        checkpoint=final_checkpoint,
        output=output_dir / "final_champion_heldout.json",
        scene_seeds=heldout_seeds,
        policy="champion",
        episodes=5,
        manifest_path=args.scene_selection_manifest.expanduser().resolve(),
        args=args,
    )
    final_test = _offline_metrics(
        algorithm, dataset, "test", args.evaluation_batch_size
    )
    summary = {
        "method": "v2_2_simple",
        "format": algorithm.FORMAT,
        "run_directory": str(output_dir),
        "elapsed_seconds": time.time() - started,
        "dataset_sha256": dataset.sha256,
        "scene_selection_manifest_hash": manifest_hash,
        "offline_scene_seed": args.offline_scene_seed,
        "online_scene_seeds": list(online_seeds),
        "retention_scene_seeds": list(retention_seeds),
        "heldout_scene_seeds": list(heldout_seeds),
        "action_semantics_version": args.action_semantics,
        "max_command_speed": args.max_command_speed,
        "offline_validation": offline_validation,
        "online": online_summary,
        "final_cross_seed_gate": final_gate,
        "heldout_generalization": heldout,
        "final_offline_test": final_test,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": _file_sha256(final_checkpoint),
        "final_champion_policy_sha256": _policy_hash(algorithm.champion_policy),
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {"run_directory": str(output_dir), "summary": summary},
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    from scripts.train_guardian_calql_metasafe import parse_args

    raise SystemExit(main_from_args(parse_args()))
