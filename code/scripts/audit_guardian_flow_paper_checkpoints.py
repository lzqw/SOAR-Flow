#!/usr/bin/env python3
"""Continuously audit immutable Guardian-Flow-DSAC paper checkpoints.

This auditor is deliberately independent of JAX, PyTorch, and the training
process.  It validates every completed evaluation/checkpoint pair that is
already present, records stable SHA256 evidence, and separately checks the
latest atomic resume snapshot.  It never loads a model or mutates a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import sys
import time
from typing import Any


PAPER_SCENES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)
PAPER_SCENE_KEYS = {str(seed) for seed in PAPER_SCENES}
OFFLINE_SCENES = (0, 1, 2, 3, 4)
NOVEL_SCENES = (5, 6, 7, 8, 10)
ALGORITHM_SEEDS = (0, 1, 2)
OFFLINE_EPISODES_PER_SCENE = 300
OFFLINE_EPISODES_TOTAL = 1_500
OFFLINE_MANIFEST_FORMAT = "guardian_dsac_multiseed_collection_v1"
PAPER_MODES = ("ordinary", "hybrid")
METHOD_SEEDS = {
    "flow_bc": (0, 1, 2),
    "gaussian": (0, 1, 2),
    "full_flow": (0, 1, 2),
    "flow_q_only": (0,),
}
ABLATION_METHODS = {"flow_no_p", "flow_no_correction"}
ONLINE_METHODS = {"gaussian", "full_flow", "flow_q_only", *ABLATION_METHODS}
FLOW_METHODS = {"flow_bc", "full_flow", "flow_q_only", *ABLATION_METHODS}
ALLOWED_NUMERICAL_ROLLBACK_REASONS = {
    "non_finite_parameter_or_optimizer",
    "non_finite_loss_or_metric",
    "non_finite_q_distribution",
    "non_positive_q_std",
}
SUMMARY_PATTERN = re.compile(r"evaluation_(\d{8})_summary\.json$")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
TRANSITION_TARGET_WEIGHTS = {
    str(seed): 0.04 if seed in OFFLINE_SCENES else 0.16
    for seed in PAPER_SCENES
}
UNIFORM_CORRECTION_WEIGHTS = {str(seed): 0.1 for seed in PAPER_SCENES}
UNIFORM_ZERO_LATENT = {str(seed): 0.5 for seed in PAPER_SCENES}
MAX_FINAL_TRANSITION_FRACTION_DEVIATION = 0.003
FORMAL_TRAINING_REVISION = "99383a489fe687509fecc6b06d457ebbafecf7ef"
PAPER_TRAINER_ENTRYPOINT = "scripts/train_guardian_flow_dsac_paper.py"
PAPER_TRAINER_ENTRYPOINT_SHA256 = (
    "19e8c8a0b5e46568e06847844dc8203febf4af3aed5476e289c2fbd5ee663746"
)
PAPER_ISOLATED_EVALUATOR_ENTRYPOINT = (
    "scripts/evaluate_guardian_dsac_isolated.py"
)
PAPER_ISOLATED_EVALUATOR_SHA256 = (
    "1accbf8c5254dd58d890d26a3bba8b9f47013f312768851b7f3aaa9e1beca40e"
)
PAPER_EVALUATION_WRAPPER_ENTRYPOINT = (
    "scripts/evaluate_guardian_flow_dsac_paper.py"
)
PAPER_EVALUATION_WRAPPER_SHA256 = (
    "07557702ac04f46b3bc3cbb796cfad3d76d551f065c890be714fdabc68ba1f21"
)
DETERMINISTIC_EVALUATION_PROTOCOL = {
    "learner_action_selection": {
        "flow": "one_step_flow_actor_exact_zero_latent",
        "gaussian": "squashed_gaussian_distribution_mean",
    },
    "flow_latent_mode": "zero",
    "actor_forward_passes_per_action": 1,
    "fresh_process_per_episode": True,
    "episodes_per_scene_mode_checkpoint": 1,
    "scene_count": 10,
    "modes": ["ordinary", "hybrid"],
    "evaluation_changes_training": False,
    "isolated_evaluator_entrypoint": PAPER_ISOLATED_EVALUATOR_ENTRYPOINT,
    "isolated_evaluator_sha256": PAPER_ISOLATED_EVALUATOR_SHA256,
    "paper_wrapper_entrypoint": PAPER_EVALUATION_WRAPPER_ENTRYPOINT,
    "paper_wrapper_sha256": PAPER_EVALUATION_WRAPPER_SHA256,
}
GUARDIAN_DEPENDENCE_PROTOCOL = {
    "target": "G_t^P=c_t+(1-c_t)*gamma_P*G_{t+1}^P",
    "discount_gamma_p": 0.97,
    "construction": "reverse_complete_episode",
    "replay_commit_boundary": "terminal_or_truncated_episode_only",
    "incomplete_episode_policy": "discard_without_replay",
    "trainer_entrypoint": PAPER_TRAINER_ENTRYPOINT,
    "trainer_entrypoint_sha256": PAPER_TRAINER_ENTRYPOINT_SHA256,
    "training_source_revision": FORMAL_TRAINING_REVISION,
}
EXPECTED_GUARDIAN_ENVIRONMENT = {
    "min_command_speed": 1.0,
    "max_command_speed": 10.75,
    "lookahead": 10.0,
    "lateral_scale": 3.5,
    "target_velocity": 10.75,
    "horizon": 2.0,
    "mpc_tracking_weight": 6.0,
    "mpc_velocity_weight": 30.0,
    "mpc_road_weight": 100.0,
    "mpc_road_buffer": 1.5,
    "mpc_max_steer_rate": 0.1,
    "clearance": 0.1,
    "cbf_gamma": 3.0,
    "projection_reserve": 0.1,
    "projection_kinematic_reserve": 0.02,
    "sqp_steps": 12,
    "admm_steps": 120,
    "max_obstacles": 32,
    "max_episode_steps": 1200,
    "post_reset_warmup_steps": 1,
}
EVALUATION_UNCERTIFIED_PROPOSAL_HANDLING = (
    "safe_termination_without_execution_v1"
)
INVALIDATION_MARKER = "INVALIDATED_FOR_PAPER.json"
RUN_STATE_INDEPENDENCE_FORMAT = (
    "guardian_flow_paper_run_state_independence_v1"
)
RUN_STATE_SERIALIZATION_CONTRACT = {
    "algorithm_checkpoint": (
        "complete_model_and_optimizer_train_state"
    ),
    "trainer_state_fields": [
        "jax_key",
        "numpy_rng_state",
        "scheduler",
        "latent_counts",
        "counters",
        "statistics",
        "evaluations",
        "online_replay_state",
        "correction_replay_state",
    ],
    "resume_index_format": "guardian_flow_paper_resume_v2_atomic",
    "replay_states_embedded_atomically": True,
    "evidence_basis": (
        "clean_frozen_training_revision_and_trainer_entrypoint_sha256"
    ),
    "auditor_unpickles_training_state": False,
}


class AuditFailure(RuntimeError):
    """Raised when immutable paper evidence violates the protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"invalid_json:{path}:{type(exc).__name__}") from exc


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _finite_tree(value: Any, location: str = "root") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"non_finite:{location}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{location}[{index}]")


def _same_float(observed: Any, expected: float, *, tolerance: float = 1e-9) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, float(expected), rel_tol=tolerance, abs_tol=tolerance
    )


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _audit_training_counters(metric: dict[str, Any], *, location: str) -> dict[str, int]:
    """Validate protocol counters without rejecting certified safe termination.

    The paper protocol explicitly permits an uncertified proposal to end/reset
    the current episode, provided no transition is fabricated and no
    performance rollback is performed.  Those events must therefore be
    counted and exposed, not treated as an audit failure.
    """
    names = (
        "performance_rollbacks",
        "uncertified_action_count",
        "safe_termination_resets",
        "aborted_incomplete_episodes",
    )
    counters = {}
    for name in names:
        try:
            counters[name] = int(metric.get(name, -1))
        except (TypeError, ValueError) as exc:
            raise AuditFailure(f"counter:{name}:{location}") from exc
    _require(
        counters["performance_rollbacks"] == 0,
        f"counter:performance_rollbacks:{location}",
    )
    for name in (
        "uncertified_action_count",
        "safe_termination_resets",
        "aborted_incomplete_episodes",
    ):
        _require(counters[name] >= 0, f"counter:{name}:{location}")
    _require(
        counters["safe_termination_resets"]
        == counters["uncertified_action_count"],
        f"counter:uncertified_reset_binding:{location}",
    )
    return counters


def _argument(command: list[str], name: str) -> str | None:
    try:
        index = command.index(name)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


class StableDigests:
    """Hash stable files once and reuse evidence when size/mtime are unchanged."""

    def __init__(self, previous: dict[str, Any] | None = None):
        self.previous = previous or {}
        self.current: dict[str, Any] = {}

    def sha256(self, path: Path) -> str:
        path = path.resolve()
        _require(path.is_file(), f"missing_file:{path}")
        key = str(path)
        for _ in range(3):
            before = path.stat()
            cached = self.previous.get(key, {})
            if (
                cached.get("size") == before.st_size
                and cached.get("mtime_ns") == before.st_mtime_ns
                and SHA256_PATTERN.fullmatch(str(cached.get("sha256", "")))
            ):
                digest = str(cached["sha256"])
            else:
                hasher = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
            after = path.stat()
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                self.current[key] = {
                    "size": after.st_size,
                    "mtime_ns": after.st_mtime_ns,
                    "sha256": digest,
                }
                return digest
        raise AuditFailure(f"file_changed_while_hashing:{path}")


def _owned_file_identity(
    path: Path,
    run_dir: Path,
    digests: StableDigests,
    *,
    role: str,
) -> dict[str, Any]:
    """Bind one state-bearing regular file to exactly one run directory."""

    _require(path.is_file(), f"state_artifact_missing:{role}:{path}")
    _require(not path.is_symlink(), f"state_artifact_symlink:{role}:{path}")
    resolved = path.resolve()
    owner = run_dir.resolve()
    _require(
        resolved.parent == owner,
        f"state_artifact_outside_run:{role}:{path}",
    )
    digest = digests.sha256(resolved)
    observed = resolved.stat()
    _require(
        stat.S_ISREG(observed.st_mode),
        f"state_artifact_not_regular:{role}:{path}",
    )
    _require(
        observed.st_nlink == 1,
        f"state_artifact_hard_linked:{role}:{path}",
    )
    cached = digests.current[str(resolved)]
    _require(
        (
            cached["size"],
            cached["mtime_ns"],
        )
        == (
            observed.st_size,
            observed.st_mtime_ns,
        ),
        f"state_artifact_changed_after_hash:{role}:{path}",
    )
    return {
        "role": role,
        "path": str(resolved),
        "sha256": digest,
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "hard_link_count": observed.st_nlink,
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "regular_file": True,
        "symlink": False,
        "owned_by_run": True,
    }


def _audit_offline_dataset_manifest(
    config: dict[str, Any], digests: StableDigests
) -> dict[str, Any]:
    """Bind every formal run to the predeclared 5 x 300 offline dataset."""

    manifest_path = Path(str(config.get("manifest", ""))).expanduser().resolve()
    _require(manifest_path.is_file(), f"offline_manifest_missing:{manifest_path}")
    manifest_sha256 = digests.sha256(manifest_path)
    _require(
        manifest_sha256 == config.get("manifest_sha256"),
        f"offline_manifest_sha256:{manifest_path}",
    )
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("format") == OFFLINE_MANIFEST_FORMAT,
        f"offline_manifest_format:{manifest_path}",
    )
    _require(
        tuple(manifest.get("offline_seeds", ())) == OFFLINE_SCENES,
        f"offline_manifest_seeds:{manifest_path}",
    )
    _require(
        int(manifest.get("episodes_per_seed", -1))
        == OFFLINE_EPISODES_PER_SCENE,
        f"offline_manifest_episodes_per_seed:{manifest_path}",
    )
    _require(
        int(manifest.get("total_episodes", -1)) == OFFLINE_EPISODES_TOTAL,
        f"offline_manifest_total_episodes:{manifest_path}",
    )
    _require(
        int(manifest.get("max_episode_steps", -1)) == 1_200,
        f"offline_manifest_horizon:{manifest_path}",
    )
    _require(
        _same_float(manifest.get("max_command_speed"), 10.75),
        f"offline_manifest_max_speed:{manifest_path}",
    )
    _require(
        manifest.get("action_semantics_version") == "target_speed_lateral_v1",
        f"offline_manifest_action_semantics:{manifest_path}",
    )
    datasets = manifest.get("datasets")
    _require(
        isinstance(datasets, list) and len(datasets) == len(OFFLINE_SCENES),
        f"offline_manifest_dataset_count:{manifest_path}",
    )
    rows_by_seed = {
        int(row.get("seed", -1)): row for row in datasets if isinstance(row, dict)
    }
    _require(
        set(rows_by_seed) == set(OFFLINE_SCENES),
        f"offline_manifest_dataset_seeds:{manifest_path}",
    )
    _require(
        all(
            int(rows_by_seed[seed].get("accepted_episodes", -1))
            == OFFLINE_EPISODES_PER_SCENE
            and SHA256_PATTERN.fullmatch(
                str(rows_by_seed[seed].get("dataset_sha256", ""))
            )
            for seed in OFFLINE_SCENES
        ),
        f"offline_manifest_dataset_rows:{manifest_path}",
    )
    _require(
        sum(int(rows_by_seed[seed]["accepted_episodes"]) for seed in OFFLINE_SCENES)
        == OFFLINE_EPISODES_TOTAL,
        f"offline_manifest_episode_sum:{manifest_path}",
    )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "format": OFFLINE_MANIFEST_FORMAT,
        "offline_seeds": list(OFFLINE_SCENES),
        "episodes_per_seed": OFFLINE_EPISODES_PER_SCENE,
        "total_episodes": OFFLINE_EPISODES_TOTAL,
        "max_episode_steps": 1_200,
        "max_command_speed": 10.75,
        "action_semantics_version": "target_speed_lateral_v1",
        "datasets": [
            {
                "seed": seed,
                "accepted_episodes": int(rows_by_seed[seed]["accepted_episodes"]),
                "dataset_sha256": str(rows_by_seed[seed]["dataset_sha256"]),
            }
            for seed in OFFLINE_SCENES
        ],
    }


def _audit_config(
    run_dir: Path,
    method: str,
    seed: int,
    expected_revision: str,
    *,
    expected_trainer_sha256: str = PAPER_TRAINER_ENTRYPOINT_SHA256,
) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    _require(config.get("component_smoke") is False, f"component_smoke:{run_dir}")
    _require(config.get("method_slug") == method, f"method_slug:{run_dir}")
    _require(int(config.get("seed", -1)) == seed, f"algorithm_seed:{run_dir}")
    _require(
        tuple(config.get("algorithm_random_seeds", ())) == ALGORITHM_SEEDS,
        f"algorithm_seed_contract:{run_dir}",
    )
    _require(tuple(config.get("scene_seeds", ())) == PAPER_SCENES, f"scenes:{run_dir}")
    _require(
        tuple(config.get("offline_seeds", ())) == OFFLINE_SCENES,
        f"offline_scenes:{run_dir}",
    )
    _require(
        tuple(config.get("online_novel_seeds", ())) == NOVEL_SCENES,
        f"novel_scenes:{run_dir}",
    )
    _require(9 not in config.get("scene_seeds", ()), f"seed9_present:{run_dir}")
    _require(
        config.get("action_semantics_version") == "target_speed_lateral_v1",
        f"action_semantics:{run_dir}",
    )
    signatures = config.get("scene_signatures")
    _require(isinstance(signatures, dict), f"scene_signatures:{run_dir}")
    _require(set(signatures) == PAPER_SCENE_KEYS, f"scene_signature_keys:{run_dir}")
    _require(
        all(SHA256_PATTERN.fullmatch(str(value)) for value in signatures.values()),
        f"scene_signature_format:{run_dir}",
    )

    source = config.get("source_code", {})
    _require(source.get("clean") is True, f"dirty_source:{run_dir}")
    _require(
        source.get("git_revision") == expected_revision,
        f"source_revision:{run_dir}",
    )
    _require(not source.get("missing_source_files"), f"missing_source:{run_dir}")
    source_hashes = source.get("source_sha256", {})
    _require(bool(source_hashes), f"source_hashes_missing:{run_dir}")
    _require(
        all(SHA256_PATTERN.fullmatch(str(value)) for value in source_hashes.values()),
        f"source_hash_format:{run_dir}",
    )
    _require(
        source_hashes.get(PAPER_TRAINER_ENTRYPOINT)
        == expected_trainer_sha256,
        f"guardian_dependence_trainer_source:{run_dir}",
    )
    _require(
        source_hashes.get(PAPER_ISOLATED_EVALUATOR_ENTRYPOINT)
        == PAPER_ISOLATED_EVALUATOR_SHA256
        and source_hashes.get(PAPER_EVALUATION_WRAPPER_ENTRYPOINT)
        == PAPER_EVALUATION_WRAPPER_SHA256,
        f"deterministic_evaluator_source:{run_dir}",
    )

    gpu = config.get("nvidia_smi", {})
    _require(int(gpu.get("returncode", -1)) == 0, f"nvidia_smi:{run_dir}")
    _require(
        any(
            "NVIDIA GeForce RTX 4080 SUPER" in str(row)
            for row in gpu.get("rows", ())
        ),
        f"gpu_identity:{run_dir}",
    )
    environment = config.get("environment", {})
    _require(
        isinstance(environment, dict)
        and set(environment) == set(EXPECTED_GUARDIAN_ENVIRONMENT)
        and all(
            _same_float(environment.get(key), value)
            for key, value in EXPECTED_GUARDIAN_ENVIRONMENT.items()
        ),
        f"guardian_environment_protocol:{run_dir}",
    )
    _require(
        int(config.get("observation_dim", -1)) == 259
        and int(config.get("actor_observation_dim", -1)) == 259
        and int(config.get("action_dim", -1)) == 2,
        f"network_dimensions:{run_dir}",
    )
    _require(
        tuple(config.get("hidden_sizes", ())) == (256, 256),
        f"shared_hidden_sizes:{run_dir}",
    )
    _require(
        _same_float(config.get("init_alpha"), 0.05),
        f"initial_entropy_temperature:{run_dir}",
    )
    _require(config.get("backend") == "gpu", f"training_backend:{run_dir}")

    budgets = config.get("budgets", {})
    expected = {
        "batch_size": 512,
        "online_buffer_size": 500_000,
        "gradient_steps": 1,
        "policy_delay": 2,
        "offline_q_guided_actor_steps": 0,
        "evaluation_every": 50_000,
        "checkpoint_every": 50_000,
        "online_steps": (
            300_000
            if method in ABLATION_METHODS
            else 1_000_000
            if method in ONLINE_METHODS
            else 0
        ),
        "flow_matching_steps": 80_000 if method in FLOW_METHODS else 0,
        "same_latent_distillation_steps": 40_000 if method in FLOW_METHODS else 0,
    }
    _require(
        all(int(budgets.get(key, -1)) == value for key, value in expected.items()),
        f"formal_budgets:{run_dir}",
    )
    _require(
        _same_float(budgets.get("offline_replay_ratio"), 0.20),
        f"offline_replay_ratio:{run_dir}",
    )
    _require(
        _same_float(budgets.get("correction_fraction"), 0.25),
        f"correction_fraction:{run_dir}",
    )
    transition_weights = config.get("transition_target_weights")
    _require(
        isinstance(transition_weights, dict)
        and set(transition_weights) == PAPER_SCENE_KEYS
        and all(
            _same_float(transition_weights[key], value)
            for key, value in TRANSITION_TARGET_WEIGHTS.items()
        ),
        f"transition_target_weights:{run_dir}",
    )

    flow = method in FLOW_METHODS
    expected_objective = (
        ["task_q", "teacher_anchor"]
        if method == "flow_q_only"
        else ["task_q", "correction", "teacher_anchor"]
        if method == "flow_no_p"
        else ["task_q", "guardian_p", "teacher_anchor"]
        if method == "flow_no_correction"
        else ["task_q", "guardian_p", "correction", "teacher_anchor"]
        if flow
        else ["gaussian_dsac", "guardian_p"]
    )
    _require(
        config.get("actor_objective_terms") == expected_objective,
        f"actor_objective_terms:{run_dir}",
    )
    correction_weights = config.get("correction_seed_sampling_weights")
    zero_latent = config.get("zero_latent_probability_by_seed")
    flow_config = config.get("flow_config")
    if flow:
        _require(isinstance(flow_config, dict), f"flow_config:{run_dir}")
        _require(
            tuple(config.get("flow_hidden_sizes", ())) == (256, 256),
            f"flow_hidden_sizes:{run_dir}",
        )
        _require(
            isinstance(correction_weights, dict)
            and set(correction_weights) == PAPER_SCENE_KEYS
            and all(
                _same_float(correction_weights[key], value)
                for key, value in UNIFORM_CORRECTION_WEIGHTS.items()
            ),
            f"uniform_correction_sampling:{run_dir}",
        )
        _require(
            isinstance(zero_latent, dict)
            and set(zero_latent) == PAPER_SCENE_KEYS
            and all(
                _same_float(zero_latent[key], value)
                for key, value in UNIFORM_ZERO_LATENT.items()
            ),
            f"uniform_zero_latent:{run_dir}",
        )
        expected_proxy_weight = (
            0.0 if method in ("flow_q_only", "flow_no_p") else 0.15
        )
        expected_correction_weight = (
            0.0 if method in ("flow_q_only", "flow_no_correction") else 2.0
        )
        expected_flow_scalars = {
            "q_actor_weight": 1.0,
            "proxy_actor_weight": expected_proxy_weight,
            "correction_weight": expected_correction_weight,
            "correction_speed_weight": 2.0,
            "correction_lateral_weight": 1.0,
            "zero_latent_correction_weight": 1.0,
            "anchor_weight": 0.05,
            "proximal_weight_initial": 0.0,
            "proximal_weight_min": 0.0,
        }
        _require(
            all(
                _same_float(flow_config.get(key), value)
                for key, value in expected_flow_scalars.items()
            ),
            f"flow_objective_scalars:{run_dir}",
        )
        _require(
            tuple(flow_config.get("anchor_seeds", ())) == OFFLINE_SCENES,
            f"flow_anchor_scenes:{run_dir}",
        )
        _require(
            int(config.get("flow_integration_steps", -1)) == 8,
            f"flow_integration_steps:{run_dir}",
        )
    else:
        _require(flow_config is None, f"gaussian_flow_config:{run_dir}")
        _require(
            correction_weights is None
            and zero_latent is None
            and config.get("flow_hidden_sizes") is None
            and config.get("flow_integration_steps") is None,
            f"gaussian_latent_or_correction:{run_dir}",
        )

    algorithm_config = config.get("algorithm_config")
    _require(isinstance(algorithm_config, dict), f"algorithm_config:{run_dir}")
    _require(
        algorithm_config.get("distributional_loss") == "hdsac",
        f"distributional_task_q:{run_dir}",
    )
    _require(
        _same_float(algorithm_config.get("reference_weight_initial"), 0.0)
        and _same_float(algorithm_config.get("reference_weight_final"), 0.0)
        and _same_float(algorithm_config.get("proximal_weight_initial"), 0.0)
        and _same_float(algorithm_config.get("proximal_weight_min"), 0.0),
        f"disabled_reference_or_proximal:{run_dir}",
    )
    for flag in (
        "moving_safe_proximal",
        "safe_trajectory_memory",
        "seed_specific_logic",
        "performance_rollback",
        "conditional_launch",
        "evaluation_changes_training",
    ):
        _require(config.get(flag) is False, f"forbidden_flag:{flag}:{run_dir}")
    _require(
        config.get("numerical_disaster_rollback_only") is True,
        f"numerical_rollback_only:{run_dir}",
    )
    # The frozen main revision predates these two declarative config keys,
    # while the independently frozen ablation revision emits them explicitly.
    # Absence is therefore accepted only under the already SHA-bound legacy
    # schema; an explicit contradictory value is always rejected.
    _require(
        config.get("independent_single_model_run", True) is True,
        (
            f"ablation_independent_model:{run_dir}"
            if method in ABLATION_METHODS
            else f"single_model_run:{run_dir}"
        ),
    )
    _require(
        config.get("composite_or_ensemble", False) is False,
        (
            f"ablation_composite_or_ensemble:{run_dir}"
            if method in ABLATION_METHODS
            else f"composite_or_ensemble:{run_dir}"
        ),
    )
    if method in ABLATION_METHODS:
        _require(seed == 1, f"ablation_algorithm_seed:{run_dir}")
        _require(
            config.get("independent_single_model_run") is True,
            f"ablation_independent_model:{run_dir}",
        )
        _require(
            config.get("composite_or_ensemble") is False,
            f"ablation_composite_or_ensemble:{run_dir}",
        )
        initialization = config.get("offline_initialization_binding")
        _require(
            isinstance(initialization, dict)
            and initialization.get("revision") == FORMAL_TRAINING_REVISION
            and initialization.get("trainer_sha256")
            == PAPER_TRAINER_ENTRYPOINT_SHA256
            and initialization.get("method_slug") == "flow_bc"
            and int(initialization.get("algorithm_seed", -1)) == 1,
            f"ablation_offline_initialization:{run_dir}",
        )
    return config


def _audit_commands(
    commands: Any,
    checkpoint: Path,
    raw_path: Path,
    expected_pairs: set[tuple[int, str]],
    digests: StableDigests,
    *,
    expected_policy: str = "learner",
) -> dict[tuple[int, str], Path]:
    _require(isinstance(commands, list), f"commands_schema:{raw_path}")
    _require(len(commands) == len(expected_pairs), f"command_count:{raw_path}")
    observed = set()
    outputs = set()
    workers = {}
    for record in commands:
        command = record.get("command")
        _require(isinstance(command, list), f"command_schema:{raw_path}")
        _require(
            len(command) >= 2
            and Path(str(command[0])).is_file()
            and Path(str(command[0])).name.startswith("python"),
            f"command_python_entrypoint:{raw_path}",
        )
        evaluator = Path(str(command[1]))
        _require(
            evaluator.is_absolute()
            and evaluator.name == Path(PAPER_ISOLATED_EVALUATOR_ENTRYPOINT).name
            and digests.sha256(evaluator) == PAPER_ISOLATED_EVALUATOR_SHA256,
            f"command_evaluator_entrypoint:{raw_path}",
        )
        _require(int(record.get("returncode", -1)) == 0, f"worker_returncode:{raw_path}")
        _require(_argument(command, "--checkpoint") == str(checkpoint), f"command_checkpoint:{raw_path}")
        _require(_argument(command, "--output") == str(raw_path), f"command_output:{raw_path}")
        _require(
            _argument(command, "--policy") == expected_policy,
            f"command_policy:{raw_path}",
        )
        _require(
            "--trajectory-output-dir" not in command
            and "--worker-trajectory-output" not in command,
            f"command_trajectory_capture:{raw_path}",
        )
        _require("--worker" in command, f"worker_flag:{raw_path}")
        seed = int(_argument(command, "--worker-seed") or -1)
        mode = str(_argument(command, "--worker-mode"))
        episode = int(_argument(command, "--episode-index") or -1)
        output = _argument(command, "--worker-output")
        _require(episode == 0, f"command_episode:{raw_path}")
        _require(output is not None and output not in outputs, f"worker_output:{raw_path}")
        outputs.add(output)
        observed.add((seed, mode))
        expected_output = (
            raw_path.parent
            / f"{raw_path.stem}_workers"
            / f"seed{seed}_{mode}_000.json"
        ).resolve()
        worker_path = Path(str(output)).resolve()
        _require(worker_path == expected_output, f"worker_output_path:{raw_path}")
        workers[(seed, mode)] = worker_path
    _require(observed == expected_pairs, f"command_pairs:{raw_path}")
    return workers


def _audit_evaluation(
    run_dir: Path,
    method: str,
    config: dict[str, Any],
    summary_path: Path,
    digests: StableDigests,
) -> dict[str, Any]:
    match = SUMMARY_PATTERN.fullmatch(summary_path.name)
    _require(match is not None, f"summary_name:{summary_path}")
    step = int(match.group(1))
    maximum_step = int(config.get("budgets", {}).get("online_steps", 0))
    _require(
        step == 0 or (step <= maximum_step and step % 50_000 == 0),
        f"step:{summary_path}",
    )
    _require(method != "flow_bc" or step == 0, f"flow_bc_step:{summary_path}")

    tag = f"{step:08d}"
    checkpoint = (run_dir / f"checkpoint_{tag}.pkl").resolve()
    raw_path = (run_dir / f"evaluation_{tag}.json").resolve()
    summary = _read_json(summary_path)
    raw = _read_json(raw_path)
    _finite_tree(summary, str(summary_path))
    _finite_tree(raw, str(raw_path))

    checkpoint_sha = digests.sha256(checkpoint)
    raw_sha = digests.sha256(raw_path)
    _require(int(summary.get("step", -1)) == step, f"summary_step:{summary_path}")
    _require(Path(str(summary.get("checkpoint", ""))).resolve() == checkpoint, f"summary_checkpoint:{summary_path}")
    _require(Path(str(summary.get("evaluation", ""))).resolve() == raw_path, f"summary_evaluation:{summary_path}")
    _require(summary.get("checkpoint_sha256") == checkpoint_sha, f"summary_checkpoint_sha:{summary_path}")
    _require(summary.get("evaluation_sha256") == raw_sha, f"summary_evaluation_sha:{summary_path}")
    _require(int(summary.get("scene_count", -1)) == 10, f"scene_count:{summary_path}")
    _require(summary.get("all_scene_denominator") is True, f"denominator:{summary_path}")
    _require(set(summary.get("per_seed", {})) == PAPER_SCENE_KEYS, f"summary_seeds:{summary_path}")

    _require(raw.get("format") == "guardian_dsac_isolated_evaluation_v1", f"raw_format:{raw_path}")
    _require(raw.get("fresh_process_per_episode") is True, f"fresh_process:{raw_path}")
    _require(
        raw.get("uncertified_proposal_handling")
        == EVALUATION_UNCERTIFIED_PROPOSAL_HANDLING,
        f"uncertified_handling:{raw_path}",
    )
    _require(
        raw.get("uncertified_proposals_executed") is False,
        f"uncertified_execution:{raw_path}",
    )
    _require(raw.get("policy") == "learner", f"policy:{raw_path}")
    trajectory_capture = raw.get("trajectory_capture")
    _require(
        isinstance(trajectory_capture, dict)
        and trajectory_capture.get("enabled") is False
        and trajectory_capture.get("ordinary_only") is True
        and trajectory_capture.get("output_dir") is None,
        f"trajectory_capture_disabled:{raw_path}",
    )
    _require(Path(str(raw.get("checkpoint", ""))).resolve() == checkpoint, f"raw_checkpoint:{raw_path}")
    _require(raw.get("checkpoint_sha256") == checkpoint_sha, f"raw_checkpoint_sha:{raw_path}")
    per_seed = raw.get("per_seed")
    _require(isinstance(per_seed, dict) and set(per_seed) == PAPER_SCENE_KEYS, f"raw_seeds:{raw_path}")

    expected_pairs = {(seed, mode) for seed in PAPER_SCENES for mode in PAPER_MODES}
    worker_paths = _audit_commands(
        raw.get("commands"), checkpoint, raw_path, expected_pairs, digests
    )
    worker_pids = set()
    worker_evidence = []
    pairs = set()
    per_seed_expected: dict[str, dict[str, float]] = {}
    numeric_fields = (
        "success_rate",
        "collision_rate",
        "road_departure_rate",
        "total_environment_cost",
        "mean_return",
        "actual_intervention_rate",
        "intervention_rate",
        "solver_finite_rate",
        "min_physical_body_sat_margin",
        "q_mean",
        "q_std",
        "p_mean",
        "inference_latency_ms_mean",
        "inference_latency_ms_p95",
        "guardian_certification_latency_ms_mean",
        "guardian_certification_latency_ms_p95",
        "certified_control_latency_ms_mean",
        "certified_control_latency_ms_p95",
        "safe_termination_rate",
    )
    expected_latent = "zero" if method in FLOW_METHODS else None
    signatures = config["scene_signatures"]
    for seed in PAPER_SCENES:
        modes = per_seed[str(seed)]
        _require(isinstance(modes, dict) and set(modes) == set(PAPER_MODES), f"modes:Seed{seed}:{raw_path}")
        for mode in PAPER_MODES:
            aggregate = modes[mode]
            _require(aggregate.get("mode") == mode, f"aggregate_mode:Seed{seed}:{mode}:{raw_path}")
            _require(int(aggregate.get("episodes", -1)) == 1, f"episodes:Seed{seed}:{mode}:{raw_path}")
            _require(aggregate.get("error") is None, f"aggregate_error:Seed{seed}:{mode}:{raw_path}")
            _require(aggregate.get("flow_latent_mode") == expected_latent, f"latent:Seed{seed}:{mode}:{raw_path}")
            _require(int(aggregate.get("actor_forward_passes", -1)) == 1, f"actor_passes:Seed{seed}:{mode}:{raw_path}")
            _require(
                all(_is_finite_number(aggregate.get(field)) for field in numeric_fields),
                f"numeric:Seed{seed}:{mode}:{raw_path}",
            )
            rows = aggregate.get("rows")
            _require(isinstance(rows, list) and len(rows) == 1, f"rows:Seed{seed}:{mode}:{raw_path}")
            row = rows[0]
            _require(int(row.get("seed", -1)) == seed, f"row_seed:Seed{seed}:{mode}:{raw_path}")
            _require(row.get("mode") == mode, f"row_mode:Seed{seed}:{mode}:{raw_path}")
            _require(int(row.get("episode_index", -1)) == 0, f"row_episode:Seed{seed}:{mode}:{raw_path}")
            _require(row.get("policy") == "learner", f"row_policy:Seed{seed}:{mode}:{raw_path}")
            _require(row.get("error") is None, f"row_error:Seed{seed}:{mode}:{raw_path}")
            _require(row.get("process_isolated") is True, f"row_isolation:Seed{seed}:{mode}:{raw_path}")
            _require(
                row.get("trajectory") is None,
                f"row_trajectory_disabled:Seed{seed}:{mode}:{raw_path}",
            )
            _require(row.get("scene_signature") == signatures[str(seed)], f"row_signature:Seed{seed}:{mode}:{raw_path}")
            _require(row.get("flow_latent_mode") == expected_latent, f"row_latent:Seed{seed}:{mode}:{raw_path}")
            _require(int(row.get("actor_forward_passes", -1)) == 1, f"row_actor_passes:Seed{seed}:{mode}:{raw_path}")
            safe_termination = row.get("safe_termination")
            _require(isinstance(safe_termination, bool), f"safe_termination_schema:Seed{seed}:{mode}:{raw_path}")
            safe_count = int(safe_termination)
            steps = int(row.get("steps", -1))
            proposals = int(row.get("action_proposals", -1))
            _require(0 <= steps <= 1200, f"row_steps:Seed{seed}:{mode}:{raw_path}")
            _require(
                int(row.get("executed_environment_steps", -1)) == steps
                and proposals == steps + safe_count,
                f"step_proposal_binding:Seed{seed}:{mode}:{raw_path}",
            )
            _require(
                int(row.get("uncertified_action_count", -1)) == safe_count
                and row.get("terminated_without_executing_action") is safe_termination
                and row.get("safe_termination_reason")
                == ("projection_failed" if safe_termination else None),
                f"safe_termination_binding:Seed{seed}:{mode}:{raw_path}",
            )
            _require(
                int(aggregate.get("safe_termination_count", -1)) == safe_count
                and int(aggregate.get("uncertified_action_count", -1)) == safe_count
                and int(aggregate.get("executed_environment_steps", -1)) == steps
                and int(aggregate.get("action_proposals", -1)) == proposals
                and _same_float(aggregate.get("safe_termination_rate"), safe_count),
                f"aggregate_safe_termination_binding:Seed{seed}:{mode}:{raw_path}",
            )
            if safe_termination:
                _require(
                    _same_float(aggregate.get("success_rate"), 0.0)
                    and float(aggregate.get("actual_intervention_rate", 0.0))
                    + 1e-12
                    >= 1.0 / max(proposals, 1),
                    f"safe_termination_outcome:Seed{seed}:{mode}:{raw_path}",
                )
            else:
                _require(steps >= 1, f"row_steps:Seed{seed}:{mode}:{raw_path}")
                _require(_same_float(row.get("solver_finite_rate"), 1.0), f"solver_finite:Seed{seed}:{mode}:{raw_path}")
            pid = int(row.get("worker_pid", -1))
            _require(pid > 0 and pid not in worker_pids, f"worker_pid:Seed{seed}:{mode}:{raw_path}")
            worker_pids.add(pid)
            pairs.add((seed, mode))
            worker_path = worker_paths[(seed, mode)]
            worker_row = _read_json(worker_path)
            _finite_tree(worker_row, str(worker_path))
            _require(worker_row == row, f"worker_row_binding:Seed{seed}:{mode}:{raw_path}")
            worker_evidence.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "path": str(worker_path),
                    "sha256": digests.sha256(worker_path),
                }
            )

        ordinary = modes["ordinary"]
        hybrid = modes["hybrid"]
        success = float(ordinary["success_rate"])
        collision = float(ordinary["collision_rate"])
        road = float(ordinary["road_departure_rate"])
        cost = float(ordinary["total_environment_cost"])
        margin = float(ordinary["min_physical_body_sat_margin"])
        safe_success = float(
            success == 1.0
            and collision == 0.0
            and road == 0.0
            and cost == 0.0
            and margin > 0.0
        )
        expected = {
            "ordinary_return": float(ordinary["mean_return"]),
            "ordinary_cost": cost,
            "ordinary_success": success,
            "ordinary_safe_success": safe_success,
            "ordinary_collision": collision,
            "ordinary_road_departure": road,
            "ordinary_min_physical_margin": margin,
            "ordinary_safe_termination": float(
                ordinary["safe_termination_rate"]
            ),
            "hybrid_intervention": float(hybrid["actual_intervention_rate"]),
            "hybrid_return": float(hybrid["mean_return"]),
            "hybrid_safe_termination": float(hybrid["safe_termination_rate"]),
        }
        per_seed_expected[str(seed)] = expected
        observed = summary["per_seed"][str(seed)]
        _require(
            all(_same_float(observed.get(key), value) for key, value in expected.items()),
            f"per_seed_summary:Seed{seed}:{summary_path}",
        )
    _require(pairs == expected_pairs, f"worker_pairs:{raw_path}")
    _require(len(worker_pids) == 20, f"worker_pid_count:{raw_path}")

    expected_global = {
        "ordinary_mean_return": statistics.fmean(row["ordinary_return"] for row in per_seed_expected.values()),
        "ordinary_success_rate": statistics.fmean(row["ordinary_success"] for row in per_seed_expected.values()),
        "ordinary_safe_success_rate": statistics.fmean(row["ordinary_safe_success"] for row in per_seed_expected.values()),
        "ordinary_mean_cost": statistics.fmean(row["ordinary_cost"] for row in per_seed_expected.values()),
        "ordinary_collision_rate": statistics.fmean(row["ordinary_collision"] for row in per_seed_expected.values()),
        "hybrid_intervention_rate": statistics.fmean(row["hybrid_intervention"] for row in per_seed_expected.values()),
        "ordinary_safe_termination_rate": statistics.fmean(row["ordinary_safe_termination"] for row in per_seed_expected.values()),
        "hybrid_safe_termination_rate": statistics.fmean(row["hybrid_safe_termination"] for row in per_seed_expected.values()),
        "ordinary_worst_physical_margin": min(row["ordinary_min_physical_margin"] for row in per_seed_expected.values()),
    }
    _require(
        all(_same_float(summary.get(key), value) for key, value in expected_global.items()),
        f"global_summary:{summary_path}",
    )

    curve_path = run_dir / "learning_curves.jsonl"
    _require(curve_path.is_file(), f"learning_curves_missing:{run_dir}")
    curves = []
    for line in curve_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                curves.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AuditFailure(f"learning_curves_json:{curve_path}") from exc
    matching = [row for row in curves if int(row.get("step", -1)) == step]
    _require(len(matching) == 1 and matching[0] == summary, f"learning_curve_binding:{summary_path}")

    training_counters = None
    if method in ONLINE_METHODS and step > 0:
        metrics_path = run_dir / "metrics.jsonl"
        metric_rows = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("online_environment_steps", -1)) == step:
                metric_rows.append(row)
        _require(bool(metric_rows), f"metric_step:{summary_path}")
        metric = metric_rows[-1]
        _finite_tree(metric, f"metric:{run_dir}:{step}")
        updates = max(step - 511, 0)
        _require(int(metric.get("task_critic_updates", -1)) == updates, f"critic_updates:{summary_path}")
        _require(int(metric.get("proxy_updates", -1)) == updates, f"proxy_updates:{summary_path}")
        _require(int(metric.get("actor_updates", -1)) == (updates + 1) // 2, f"actor_updates:{summary_path}")
        training_counters = _audit_training_counters(
            metric, location=str(summary_path)
        )
        numerical_rollbacks = int(metric.get("numerical_rollbacks", -1))
        _require(numerical_rollbacks >= 0, f"counter:numerical_rollbacks:{summary_path}")
        rollback_rows = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                candidate.get("phase") == "numerical_rollback"
                and int(candidate.get("step", step + 1)) <= step
            ):
                rollback_rows.append(candidate)
        _require(
            len(rollback_rows) == numerical_rollbacks,
            f"numerical_rollback_count:{summary_path}",
        )
        for rollback in rollback_rows:
            reasons = rollback.get("reasons")
            _require(
                isinstance(reasons, list)
                and bool(reasons)
                and set(reasons) <= ALLOWED_NUMERICAL_ROLLBACK_REASONS,
                f"numerical_rollback_reason:{summary_path}",
            )
            restored = Path(str(rollback.get("checkpoint", "")))
            _require(restored.is_file(), f"numerical_rollback_checkpoint:{summary_path}")
        steps_by_seed = metric.get("steps_by_seed", {})
        _require(set(steps_by_seed) == PAPER_SCENE_KEYS, f"metric_scene_keys:{summary_path}")
        _require(sum(int(value) for value in steps_by_seed.values()) == step, f"metric_scene_total:{summary_path}")

    return {
        "method": method,
        "algorithm_seed": int(config["seed"]),
        "step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "evaluation": str(raw_path),
        "evaluation_sha256": raw_sha,
        "summary": str(summary_path.resolve()),
        "summary_sha256": digests.sha256(summary_path),
        "worker_pairs": 20,
        "unique_worker_processes": 20,
        "worker_evidence": sorted(
            worker_evidence, key=lambda item: (item["seed"], item["mode"])
        ),
        "all_scene_denominator": True,
        "evaluation_safe_terminations": {
            "ordinary_count": sum(
                int(row["ordinary_safe_termination"])
                for row in per_seed_expected.values()
            ),
            "hybrid_count": sum(
                int(row["hybrid_safe_termination"])
                for row in per_seed_expected.values()
            ),
            "ordinary_rate": float(
                summary["ordinary_safe_termination_rate"]
            ),
            "hybrid_rate": float(summary["hybrid_safe_termination_rate"]),
            "per_seed": {
                seed: {
                    "ordinary": bool(row["ordinary_safe_termination"]),
                    "hybrid": bool(row["hybrid_safe_termination"]),
                }
                for seed, row in per_seed_expected.items()
            },
            "uncertified_proposals_executed": False,
        },
        "training_counters": training_counters,
        "audit_passed": True,
    }


def _audit_resume(
    run_dir: Path,
    digests: StableDigests,
    *,
    settle_seconds: float,
) -> dict[str, Any] | None:
    index_path = run_dir / "resume_index.json"
    if not index_path.is_file():
        return None
    if time.time() - index_path.stat().st_mtime < settle_seconds:
        return None
    transient_failure = None
    for _ in range(5):
        before = index_path.read_bytes()
        try:
            index = json.loads(before)
        except json.JSONDecodeError as exc:
            raise AuditFailure(f"resume_index_json:{index_path}") from exc
        step = int(index.get("step", -1))
        permanent = run_dir / f"checkpoint_{step:08d}.pkl"
        latest = run_dir / "checkpoint_latest.pkl"
        trainer = run_dir / "trainer_state_latest.pkl"
        if any(
            not path.is_file()
            or time.time() - path.stat().st_mtime < settle_seconds
            for path in (permanent, latest, trainer)
        ):
            return None
        state_artifacts = {
            "resume_index": _owned_file_identity(
                index_path, run_dir, digests, role="resume_index"
            ),
            "algorithm_checkpoint": _owned_file_identity(
                permanent,
                run_dir,
                digests,
                role="algorithm_checkpoint",
            ),
            "checkpoint_latest": _owned_file_identity(
                latest, run_dir, digests, role="checkpoint_latest"
            ),
            "trainer_state": _owned_file_identity(
                trainer, run_dir, digests, role="trainer_state"
            ),
        }
        permanent_sha = state_artifacts["algorithm_checkpoint"]["sha256"]
        latest_sha = state_artifacts["checkpoint_latest"]["sha256"]
        trainer_sha = state_artifacts["trainer_state"]["sha256"]
        after = index_path.read_bytes()
        if before != after:
            time.sleep(0.1)
            continue
        checks = (
            (index.get("format") == "guardian_flow_paper_resume_v2_atomic", "resume_format"),
            (index.get("replay_states_embedded_atomically") is True, "resume_replay"),
            (index.get("algorithm_checkpoint") == permanent.name, "resume_checkpoint_name"),
            (index.get("algorithm_checkpoint_backup") == latest.name, "resume_backup_name"),
            (index.get("trainer_state") == trainer.name, "resume_trainer_name"),
            (permanent_sha == latest_sha, "resume_checkpoint_copy"),
            (index.get("algorithm_checkpoint_sha256") == permanent_sha, "resume_checkpoint_sha"),
            (index.get("algorithm_checkpoint_backup_sha256") == latest_sha, "resume_backup_sha"),
            (index.get("trainer_state_sha256") == trainer_sha, "resume_trainer_sha"),
        )
        failed = [name for passed, name in checks if not passed]
        if failed:
            if any(
                time.time() - path.stat().st_mtime < max(settle_seconds, 60.0)
                for path in (index_path, latest, trainer)
            ):
                return None
            transient_failure = f"{','.join(failed)}:{run_dir}"
            time.sleep(0.25)
            continue
        return {
            "step": step,
            "format": index["format"],
            "checkpoint_sha256": permanent_sha,
            "trainer_state_sha256": trainer_sha,
            "replay_states_embedded_atomically": True,
            "run_directory": str(run_dir.resolve()),
            "state_artifacts": state_artifacts,
            "audit_passed": True,
        }
    raise AuditFailure(transient_failure or f"resume_index_changed_repeatedly:{run_dir}")


def _audit_final_sampling_protocol(
    online: dict[str, Any],
    *,
    method: str,
    expected_step: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Verify the persisted final 80/20 schedule and Flow latent stratification."""

    def nonnegative_counts(value: Any, *, label: str) -> dict[str, int]:
        _require(isinstance(value, dict), f"{label}:{run_dir}")
        _require(set(value) == PAPER_SCENE_KEYS, f"{label}_scene_keys:{run_dir}")
        counts: dict[str, int] = {}
        for scene in PAPER_SCENE_KEYS:
            observed = value[scene]
            _require(
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed >= 0,
                f"{label}_count:Seed{scene}:{run_dir}",
            )
            counts[scene] = observed
        return counts

    steps = nonnegative_counts(
        online.get("steps_by_seed"), label="final_transition_steps"
    )
    _require(
        sum(steps.values()) == expected_step,
        f"final_transition_step_total:{run_dir}",
    )
    fractions = online.get("transition_fraction_by_seed")
    _require(
        isinstance(fractions, dict) and set(fractions) == PAPER_SCENE_KEYS,
        f"final_transition_fraction_scene_keys:{run_dir}",
    )
    deviations: dict[str, float] = {}
    for scene in PAPER_SCENE_KEYS:
        value = fractions[scene]
        _require(
            _is_finite_number(value),
            f"final_transition_fraction_finite:Seed{scene}:{run_dir}",
        )
        observed = float(value)
        _require(
            _same_float(observed, steps[scene] / expected_step),
            f"final_transition_fraction_binding:Seed{scene}:{run_dir}",
        )
        deviation = abs(observed - TRANSITION_TARGET_WEIGHTS[scene])
        _require(
            deviation <= MAX_FINAL_TRANSITION_FRACTION_DEVIATION,
            f"final_transition_schedule_tolerance:Seed{scene}:{run_dir}",
        )
        deviations[scene] = deviation

    episodes = nonnegative_counts(
        online.get("episodes_by_seed"), label="final_completed_episodes"
    )
    online_episodes = online.get("online_episodes")
    _require(
        isinstance(online_episodes, int)
        and not isinstance(online_episodes, bool)
        and online_episodes >= 0,
        f"final_completed_episode_count:{run_dir}",
    )
    _require(
        sum(episodes.values()) == online_episodes,
        f"final_completed_episode_total:{run_dir}",
    )

    latent_evidence = None
    if method in FLOW_METHODS:
        latent_steps = online.get("latent_steps_by_seed")
        latent_counts = online.get("latent_episode_counts_by_seed")
        _require(
            isinstance(latent_steps, dict)
            and set(latent_steps) == PAPER_SCENE_KEYS,
            f"final_latent_step_scene_keys:{run_dir}",
        )
        _require(
            isinstance(latent_counts, dict)
            and set(latent_counts) == PAPER_SCENE_KEYS,
            f"final_latent_episode_scene_keys:{run_dir}",
        )
        episode_counts: dict[str, dict[str, int]] = {}
        step_counts: dict[str, dict[str, int]] = {}
        for scene in PAPER_SCENE_KEYS:
            scene_steps = latent_steps[scene]
            scene_episodes = latent_counts[scene]
            _require(
                isinstance(scene_steps, dict)
                and set(scene_steps) == {"zero", "random"},
                f"final_latent_step_modes:Seed{scene}:{run_dir}",
            )
            _require(
                isinstance(scene_episodes, dict)
                and set(scene_episodes) == {"zero", "random"},
                f"final_latent_episode_modes:Seed{scene}:{run_dir}",
            )
            for mode in ("zero", "random"):
                _require(
                    isinstance(scene_steps[mode], int)
                    and not isinstance(scene_steps[mode], bool)
                    and scene_steps[mode] >= 0,
                    f"final_latent_step_count:Seed{scene}:{mode}:{run_dir}",
                )
                _require(
                    isinstance(scene_episodes[mode], int)
                    and not isinstance(scene_episodes[mode], bool)
                    and scene_episodes[mode] >= 1,
                    f"final_latent_episode_count:Seed{scene}:{mode}:{run_dir}",
                )
            _require(
                sum(scene_steps.values()) == steps[scene],
                f"final_latent_step_binding:Seed{scene}:{run_dir}",
            )
            _require(
                abs(scene_episodes["zero"] - scene_episodes["random"]) <= 1,
                f"final_latent_episode_balance:Seed{scene}:{run_dir}",
            )
            step_counts[scene] = dict(scene_steps)
            episode_counts[scene] = dict(scene_episodes)
        _require(
            sum(sum(row.values()) for row in episode_counts.values())
            >= online_episodes,
            f"final_latent_episode_attempt_total:{run_dir}",
        )
        latent_evidence = {
            "policy": "per_scene_episode_stratified_50_percent_exact_zero",
            "maximum_episode_count_imbalance": max(
                abs(row["zero"] - row["random"])
                for row in episode_counts.values()
            ),
            "steps_by_scene_and_mode": step_counts,
            "episode_attempts_by_scene_and_mode": episode_counts,
        }
    else:
        _require(
            online.get("latent_steps_by_seed") is None
            and online.get("latent_episode_counts_by_seed") is None,
            f"final_gaussian_latent_statistics:{run_dir}",
        )

    return {
        "transition_target_weights": dict(TRANSITION_TARGET_WEIGHTS),
        "transition_steps_by_seed": steps,
        "transition_fraction_by_seed": {
            scene: float(fractions[scene]) for scene in PAPER_SCENE_KEYS
        },
        "maximum_transition_fraction_deviation": max(deviations.values()),
        "maximum_allowed_transition_fraction_deviation": (
            MAX_FINAL_TRANSITION_FRACTION_DEVIATION
        ),
        "completed_episodes_by_seed": episodes,
        "latent_protocol": latent_evidence,
        "audit_passed": True,
    }


def _audit_final_binding(
    run_dir: Path,
    method: str,
    seed: int,
    entries: list[dict[str, Any]],
    digests: StableDigests,
    *,
    settle_seconds: float = 0.0,
) -> dict[str, Any] | None:
    """Bind a completed run's published final model to its final evaluation."""

    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    if time.time() - summary_path.stat().st_mtime < settle_seconds:
        return None
    summary = _read_json(summary_path)
    _require(
        summary.get("format") == "guardian_flow_dsac_paper_run_summary_v1",
        f"final_summary_format:{run_dir}",
    )
    _require(summary.get("method_slug") == method, f"final_method:{run_dir}")
    _require(
        int(summary.get("algorithm_seed", -1)) == seed,
        f"final_algorithm_seed:{run_dir}",
    )
    expected_step = (
        0
        if method == "flow_bc"
        else 300_000
        if method in ABLATION_METHODS
        else 1_000_000
    )
    final_evaluation = summary.get("final_evaluation")
    _require(isinstance(final_evaluation, dict), f"final_evaluation:{run_dir}")
    _require(
        int(final_evaluation.get("step", -1)) == expected_step,
        f"final_evaluation_step:{run_dir}",
    )
    sampling_protocol = None
    if method == "flow_bc":
        _require(summary.get("online") is None, f"flow_bc_online_summary:{run_dir}")
    else:
        online = summary.get("online")
        _require(isinstance(online, dict), f"final_online_summary:{run_dir}")
        _require(
            int(online.get("actual_online_steps", -1)) == expected_step,
            f"final_online_steps:{run_dir}",
        )
        _require(
            online.get("completed_full_budget") is True
            and summary.get("completed_full_budget") is True,
            f"final_budget_flag:{run_dir}",
        )
        evaluations = online.get("evaluations")
        _require(
            isinstance(evaluations, list)
            and len(evaluations) == expected_step // 50_000 + 1
            and evaluations[-1] == final_evaluation
            and online.get("final_evaluation") == final_evaluation,
            f"final_online_evaluation_binding:{run_dir}",
        )
        sampling_protocol = _audit_final_sampling_protocol(
            online,
            method=method,
            expected_step=expected_step,
            run_dir=run_dir,
        )

    matching = [
        entry
        for entry in entries
        if entry["method"] == method
        and int(entry["algorithm_seed"]) == seed
        and int(entry["step"]) == expected_step
    ]
    _require(len(matching) == 1, f"final_audit_entry:{run_dir}")
    entry = matching[0]
    permanent = run_dir / f"checkpoint_{expected_step:08d}.pkl"
    final_checkpoint = run_dir / "final.pkl"
    evaluation_summary = run_dir / f"evaluation_{expected_step:08d}_summary.json"
    if any(
        not path.is_file()
        or time.time() - path.stat().st_mtime < settle_seconds
        for path in (permanent, final_checkpoint, evaluation_summary)
    ):
        return None
    permanent_sha = digests.sha256(permanent)
    final_sha = digests.sha256(final_checkpoint)
    summary_sha = digests.sha256(summary_path)
    _require(permanent_sha == final_sha, f"final_checkpoint_copy:{run_dir}")
    _require(
        Path(str(summary.get("final_checkpoint", ""))).resolve()
        == final_checkpoint.resolve(),
        f"final_checkpoint_path:{run_dir}",
    )
    _require(
        summary.get("final_checkpoint_sha256") == final_sha,
        f"final_checkpoint_sha:{run_dir}",
    )
    _require(
        Path(str(final_evaluation.get("checkpoint", ""))).resolve()
        == permanent.resolve()
        and final_evaluation.get("checkpoint_sha256") == permanent_sha,
        f"final_evaluation_checkpoint:{run_dir}",
    )
    _require(
        Path(str(final_evaluation.get("evaluation", ""))).resolve()
        == Path(entry["evaluation"]).resolve()
        and final_evaluation.get("evaluation_sha256") == entry["evaluation_sha256"],
        f"final_evaluation_raw:{run_dir}",
    )
    _require(
        _read_json(evaluation_summary) == final_evaluation
        and Path(entry["summary"]).resolve() == evaluation_summary.resolve(),
        f"final_evaluation_summary:{run_dir}",
    )
    _require(
        entry["checkpoint_sha256"] == permanent_sha,
        f"final_checkpoint_audit_sha:{run_dir}",
    )
    state_artifacts = {
        "permanent_checkpoint": _owned_file_identity(
            permanent,
            run_dir,
            digests,
            role="permanent_checkpoint",
        ),
        "final_checkpoint": _owned_file_identity(
            final_checkpoint,
            run_dir,
            digests,
            role="final_checkpoint",
        ),
    }
    return {
        "method": method,
        "algorithm_seed": seed,
        "step": expected_step,
        "run_summary": str(summary_path.resolve()),
        "run_summary_sha256": summary_sha,
        "permanent_checkpoint": str(permanent.resolve()),
        "final_checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_sha256": final_sha,
        "evaluation": entry["evaluation"],
        "evaluation_sha256": entry["evaluation_sha256"],
        "evaluation_summary": str(evaluation_summary.resolve()),
        "evaluation_summary_sha256": entry["summary_sha256"],
        "sampling_protocol": sampling_protocol,
        "run_directory": str(run_dir.resolve()),
        "state_artifacts": state_artifacts,
        "one_million_final": expected_step == 1_000_000,
        "audit_passed": True,
    }


def _audit_run_state_independence(
    run_root: Path,
    *,
    method_seeds: dict[str, tuple[int, ...]],
    discovered_runs: list[str],
    resumes: list[dict[str, Any]],
    final_bindings: list[dict[str, Any]],
    expected_source_revision: str,
    expected_trainer_sha256: str,
    require_complete: bool,
    scope: str,
) -> dict[str, Any]:
    """Prove that post-initialization state files are owned by one run only.

    Equal bytes are deliberately allowed because multiple runs share the
    frozen offline initialization and can therefore match at step zero.  What
    is forbidden is sharing a path, symlink, or hard-linked inode after that
    initialization boundary.
    """

    run_root = run_root.resolve()
    expected_pairs = [
        (method, seed)
        for method, seeds in method_seeds.items()
        for seed in seeds
    ]
    expected_runs = {
        f"{method}/train_seed_{seed}" for method, seed in expected_pairs
    }
    observed_runs = set(discovered_runs)
    _require(
        len(discovered_runs) == len(observed_runs)
        and observed_runs <= expected_runs,
        f"state_independence_discovered_runs:{scope}",
    )

    def keyed(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, int], dict]:
        result: dict[tuple[str, int], dict] = {}
        for row in rows:
            key = (str(row.get("method")), int(row.get("algorithm_seed", -1)))
            _require(
                key in expected_pairs and key not in result,
                f"state_independence_{label}_key:{scope}:{key}",
            )
            result[key] = row
        return result

    resume_by_run = keyed(resumes, "resume")
    final_by_run = keyed(final_bindings, "final")
    rows = []
    pending_runs = []
    path_owners: dict[str, str] = {}
    inode_owners: dict[tuple[int, int], str] = {}

    for method, seed in expected_pairs:
        relative = f"{method}/train_seed_{seed}"
        if relative not in observed_runs:
            pending_runs.append(relative)
            continue
        run_dir = run_root / method / f"train_seed_{seed}"
        _require(
            run_dir.is_dir() and not run_dir.is_symlink(),
            f"state_independence_run_directory:{scope}:{relative}",
        )
        resolved_run = run_dir.resolve()
        _require(
            resolved_run == run_root / method / f"train_seed_{seed}",
            f"state_independence_run_path:{scope}:{relative}",
        )
        config = _read_json(run_dir / "config.json")
        trainer_sha = (
            config.get("source_code", {})
            .get("source_sha256", {})
            .get(PAPER_TRAINER_ENTRYPOINT)
        )
        _require(
            trainer_sha == expected_trainer_sha256,
            f"state_independence_trainer_source:{scope}:{relative}",
        )
        _require(
            config.get("composite_or_ensemble", False) is False
            and config.get("independent_single_model_run", True) is True,
            f"state_independence_single_model_config:{scope}:{relative}",
        )

        resume = resume_by_run.get((method, seed))
        final = final_by_run.get((method, seed))
        artifacts: dict[str, dict[str, Any]] = {}
        if resume is not None:
            _require(
                resume.get("run_directory") == str(resolved_run)
                and resume.get("replay_states_embedded_atomically") is True,
                f"state_independence_resume_binding:{scope}:{relative}",
            )
            artifacts.update(resume.get("state_artifacts") or {})
        if final is not None:
            _require(
                final.get("run_directory") == str(resolved_run),
                f"state_independence_final_binding:{scope}:{relative}",
            )
            final_artifacts = final.get("state_artifacts") or {}
            if resume is not None:
                _require(
                    int(resume.get("step", -1)) == int(final.get("step", -2))
                    and final_artifacts.get("permanent_checkpoint", {}).get("path")
                    == artifacts.get("algorithm_checkpoint", {}).get("path")
                    and final_artifacts.get("permanent_checkpoint", {}).get("sha256")
                    == artifacts.get("algorithm_checkpoint", {}).get("sha256"),
                    f"state_independence_final_resume_binding:{scope}:{relative}",
                )
                artifacts["final_checkpoint"] = final_artifacts.get(
                    "final_checkpoint"
                )
            else:
                artifacts.update(final_artifacts)

        online = method != "flow_bc"
        required_observed_roles = (
            {
                "resume_index",
                "algorithm_checkpoint",
                "checkpoint_latest",
                "trainer_state",
            }
            if online and resume is not None
            else {"permanent_checkpoint", "final_checkpoint"}
            if not online and final is not None
            else set()
        )
        _require(
            required_observed_roles <= set(artifacts),
            f"state_independence_artifact_roles:{scope}:{relative}",
        )
        if (online and resume is None) or final is None:
            pending_runs.append(relative)

        for role, artifact in artifacts.items():
            _require(
                isinstance(artifact, dict)
                and artifact.get("role") == role
                and artifact.get("regular_file") is True
                and artifact.get("symlink") is False
                and artifact.get("owned_by_run") is True
                and int(artifact.get("hard_link_count", -1)) == 1
                and Path(str(artifact.get("path", ""))).parent == resolved_run
                and SHA256_PATTERN.fullmatch(str(artifact.get("sha256", ""))),
                f"state_independence_artifact:{scope}:{relative}:{role}",
            )
            owner = relative
            path = str(artifact["path"])
            identity = (int(artifact["device"]), int(artifact["inode"]))
            _require(
                path_owners.get(path, owner) == owner,
                f"state_independence_shared_path:{scope}:{path}",
            )
            _require(
                inode_owners.get(identity, owner) == owner,
                f"state_independence_shared_inode:{scope}:{identity}",
            )
            path_owners[path] = owner
            inode_owners[identity] = owner

        if artifacts:
            rows.append(
                {
                    "method": method,
                    "algorithm_seed": seed,
                    "run_directory": str(resolved_run),
                    "online_run": online,
                    "atomic_resume_step": (
                        int(resume["step"]) if resume is not None else None
                    ),
                    "final_binding_present": final is not None,
                    "trainer_entrypoint_sha256": trainer_sha,
                    "artifacts": artifacts,
                    "independent_single_model": True,
                    "composite_or_ensemble": False,
                    "audit_passed": True,
                }
            )

    pending_runs = sorted(set(pending_runs))
    matrix_complete = not pending_runs and observed_runs == expected_runs
    if require_complete:
        _require(
            matrix_complete,
            f"state_independence_complete_matrix:{scope}",
        )
    return {
        "format": RUN_STATE_INDEPENDENCE_FORMAT,
        "scope": scope,
        "expected_source_revision": expected_source_revision,
        "expected_trainer_entrypoint_sha256": expected_trainer_sha256,
        "serialization_contract": RUN_STATE_SERIALIZATION_CONTRACT,
        "expected_runs": sorted(expected_runs),
        "audited_runs": sorted(
            f"{row['method']}/train_seed_{row['algorithm_seed']}" for row in rows
        ),
        "pending_runs": pending_runs,
        "rows": rows,
        "state_artifact_count": sum(len(row["artifacts"]) for row in rows),
        "matrix_complete": matrix_complete,
        "unique_run_directories": True,
        "unique_state_artifact_paths_across_runs": True,
        "unique_state_artifact_inodes_across_runs": True,
        "single_link_regular_state_files": True,
        "shared_offline_initialization_only": True,
        "post_initialization_model_optimizer_replay_rng_independent": True,
        "shared_state_artifact_count": 0,
        "composite_or_ensemble": False,
        "audit_passed": True,
    }


def _audit_teacher(root: Path, digests: StableDigests) -> dict[str, Any] | None:
    path = root / "guardian_mpc" / "teacher_evaluation.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    _finite_tree(payload, str(path))
    _require(payload.get("format") == "guardian_flow_dsac_paper_evaluation_v1", "teacher_format")
    _require(payload.get("policy") == "teacher", "teacher_policy")
    _require(int(payload.get("episodes_per_scene", -1)) == 1, "teacher_episodes")
    _require(tuple(payload.get("scene_seeds", ())) == PAPER_SCENES, "teacher_scenes")
    checkpoint = Path(str(payload.get("checkpoint", "")))
    raw = Path(str(payload.get("raw_evaluation", "")))
    _require(digests.sha256(checkpoint) == payload.get("checkpoint_sha256"), "teacher_checkpoint_sha")
    _require(digests.sha256(raw) == payload.get("raw_evaluation_sha256"), "teacher_raw_sha")
    raw_payload = _read_json(raw)
    _finite_tree(raw_payload, str(raw))
    _require(
        raw_payload.get("format") == "guardian_dsac_isolated_evaluation_v1",
        "teacher_raw_format",
    )
    _require(raw_payload.get("policy") == "teacher", "teacher_raw_policy")
    trajectory_capture = raw_payload.get("trajectory_capture")
    _require(
        isinstance(trajectory_capture, dict)
        and trajectory_capture.get("enabled") is False
        and trajectory_capture.get("ordinary_only") is True
        and trajectory_capture.get("output_dir") is None,
        "teacher_trajectory_capture_disabled",
    )
    _require(
        raw_payload.get("fresh_process_per_episode") is True,
        "teacher_fresh_process",
    )
    _require(
        raw_payload.get("uncertified_proposal_handling")
        == EVALUATION_UNCERTIFIED_PROPOSAL_HANDLING
        and raw_payload.get("uncertified_proposals_executed") is False,
        "teacher_uncertified_handling",
    )
    _require(
        Path(str(raw_payload.get("checkpoint", ""))).resolve()
        == checkpoint.resolve()
        and raw_payload.get("checkpoint_sha256") == payload["checkpoint_sha256"],
        "teacher_raw_checkpoint",
    )
    per_seed = raw_payload.get("per_seed")
    _require(
        isinstance(per_seed, dict) and set(per_seed) == PAPER_SCENE_KEYS,
        "teacher_raw_scenes",
    )
    expected_pairs = {(seed, "hybrid") for seed in PAPER_SCENES}
    worker_paths = _audit_commands(
        raw_payload.get("commands"),
        checkpoint.resolve(),
        raw.resolve(),
        expected_pairs,
        digests,
        expected_policy="teacher",
    )
    config = _read_json(root / "flow_bc" / "train_seed_0" / "config.json")
    signatures = config.get("scene_signatures", {})
    _require(set(signatures) == PAPER_SCENE_KEYS, "teacher_scene_signatures")

    summary = payload.get("summary", {})
    _require(int(summary.get("scene_count", -1)) == 10, "teacher_scene_count")
    _require(summary.get("all_scene_denominator") is True, "teacher_denominator")
    _require(set(summary.get("per_seed", {})) == PAPER_SCENE_KEYS, "teacher_summary_scenes")
    worker_pids = set()
    worker_evidence = []
    expected_summary_rows = {}
    numeric_fields = (
        "success_rate",
        "collision_rate",
        "road_departure_rate",
        "total_environment_cost",
        "mean_return",
        "actual_intervention_rate",
        "intervention_rate",
        "solver_finite_rate",
        "min_physical_body_sat_margin",
        "inference_latency_ms_mean",
        "inference_latency_ms_p95",
        "guardian_certification_latency_ms_mean",
        "guardian_certification_latency_ms_p95",
        "certified_control_latency_ms_mean",
        "certified_control_latency_ms_p95",
        "safe_termination_rate",
    )
    for seed in PAPER_SCENES:
        modes = per_seed[str(seed)]
        _require(
            isinstance(modes, dict) and set(modes) == {"hybrid"},
            f"teacher_modes:Seed{seed}",
        )
        aggregate = modes["hybrid"]
        _require(aggregate.get("mode") == "hybrid", f"teacher_mode:Seed{seed}")
        _require(int(aggregate.get("episodes", -1)) == 1, f"teacher_episodes:Seed{seed}")
        _require(aggregate.get("error") is None, f"teacher_error:Seed{seed}")
        _require(aggregate.get("flow_latent_mode") is None, f"teacher_latent:Seed{seed}")
        _require(
            int(aggregate.get("actor_forward_passes", -1)) == 0,
            f"teacher_actor_passes:Seed{seed}",
        )
        _require(
            all(_is_finite_number(aggregate.get(name)) for name in numeric_fields),
            f"teacher_numeric:Seed{seed}",
        )
        rows = aggregate.get("rows")
        _require(isinstance(rows, list) and len(rows) == 1, f"teacher_rows:Seed{seed}")
        row = rows[0]
        _require(
            int(row.get("seed", -1)) == seed
            and row.get("mode") == "hybrid"
            and int(row.get("episode_index", -1)) == 0
            and row.get("policy") == "teacher",
            f"teacher_row_binding:Seed{seed}",
        )
        _require(row.get("error") is None, f"teacher_row_error:Seed{seed}")
        _require(row.get("process_isolated") is True, f"teacher_isolation:Seed{seed}")
        _require(
            row.get("trajectory") is None,
            f"teacher_worker_trajectory_disabled:Seed{seed}",
        )
        _require(
            row.get("scene_signature") == signatures[str(seed)],
            f"teacher_signature:Seed{seed}",
        )
        _require(
            row.get("flow_latent_mode") is None
            and int(row.get("actor_forward_passes", -1)) == 0,
            f"teacher_actor_identity:Seed{seed}",
        )
        safe = row.get("safe_termination")
        _require(isinstance(safe, bool), f"teacher_safe_schema:Seed{seed}")
        safe_count = int(safe)
        steps = int(row.get("steps", -1))
        proposals = int(row.get("action_proposals", -1))
        _require(
            0 <= steps <= 1200
            and int(row.get("executed_environment_steps", -1)) == steps
            and proposals == steps + safe_count,
            f"teacher_step_proposal_binding:Seed{seed}",
        )
        _require(
            int(row.get("uncertified_action_count", -1)) == safe_count
            and row.get("terminated_without_executing_action") is safe
            and row.get("safe_termination_reason")
            == ("projection_failed" if safe else None)
            and int(aggregate.get("safe_termination_count", -1)) == safe_count
            and int(aggregate.get("uncertified_action_count", -1)) == safe_count
            and int(aggregate.get("executed_environment_steps", -1)) == steps
            and int(aggregate.get("action_proposals", -1)) == proposals
            and _same_float(aggregate.get("safe_termination_rate"), safe_count),
            f"teacher_safe_binding:Seed{seed}",
        )
        if safe:
            _require(
                _same_float(aggregate.get("success_rate"), 0.0),
                f"teacher_safe_outcome:Seed{seed}",
            )
        else:
            _require(steps >= 1, f"teacher_steps:Seed{seed}")
            _require(
                _same_float(row.get("solver_finite_rate"), 1.0),
                f"teacher_solver_finite:Seed{seed}",
            )
        direct_bindings = {
            "success_rate": float(bool(row["success"])),
            "collision_rate": float(bool(row["collision"])),
            "road_departure_rate": float(bool(row["road_departure"])),
            "total_environment_cost": float(row["environment_cost"]),
            "mean_return": float(row["return"]),
            "actual_intervention_rate": float(row["actual_intervention_rate"]),
            "intervention_rate": float(row["intervention_rate"]),
            "solver_finite_rate": float(row["solver_finite_rate"]),
            "min_physical_body_sat_margin": float(
                row["min_physical_body_sat_margin"]
            ),
            "inference_latency_ms_mean": float(row["inference_latency_ms_mean"]),
            "inference_latency_ms_p95": float(row["inference_latency_ms_p95"]),
            "guardian_certification_latency_ms_mean": float(
                row["guardian_certification_latency_ms_mean"]
            ),
            "guardian_certification_latency_ms_p95": float(
                row["guardian_certification_latency_ms_p95"]
            ),
            "certified_control_latency_ms_mean": float(
                row["certified_control_latency_ms_mean"]
            ),
            "certified_control_latency_ms_p95": float(
                row["certified_control_latency_ms_p95"]
            ),
            "safe_termination_rate": float(safe_count),
        }
        _require(
            all(
                _same_float(aggregate.get(name), value)
                for name, value in direct_bindings.items()
            ),
            f"teacher_aggregate_row_binding:Seed{seed}",
        )
        pid = int(row.get("worker_pid", -1))
        _require(pid > 0 and pid not in worker_pids, f"teacher_worker_pid:Seed{seed}")
        worker_pids.add(pid)
        worker_path = worker_paths[(seed, "hybrid")]
        worker_row = _read_json(worker_path)
        _require(worker_row == row, f"teacher_worker_row_binding:Seed{seed}")
        worker_evidence.append(
            {
                "seed": seed,
                "mode": "hybrid",
                "path": str(worker_path),
                "sha256": digests.sha256(worker_path),
            }
        )
        cost = float(aggregate["total_environment_cost"])
        success = float(aggregate["success_rate"])
        collision = float(aggregate["collision_rate"])
        road = float(aggregate["road_departure_rate"])
        margin = float(aggregate["min_physical_body_sat_margin"])
        safe_success = float(
            success == 1.0
            and collision == 0.0
            and road == 0.0
            and cost == 0.0
            and margin > 0.0
            and not safe
        )
        expected_summary_rows[str(seed)] = {
            "hybrid_return": float(aggregate["mean_return"]),
            "hybrid_cost": cost,
            "hybrid_success": success,
            "hybrid_safe_success": safe_success,
            "hybrid_collision": collision,
            "hybrid_road_departure": road,
            "hybrid_min_physical_margin": margin,
            "hybrid_safe_termination": float(safe_count),
            "policy_inference_latency_ms_mean": float(
                aggregate["inference_latency_ms_mean"]
            ),
            "guardian_certification_latency_ms_mean": float(
                aggregate["guardian_certification_latency_ms_mean"]
            ),
            "certified_control_latency_ms_mean": float(
                aggregate["certified_control_latency_ms_mean"]
            ),
            "environment_steps": steps,
        }
        observed_summary = summary["per_seed"][str(seed)]
        _require(
            all(
                _same_float(observed_summary.get(name), value)
                for name, value in expected_summary_rows[str(seed)].items()
            ),
            f"teacher_summary_row:Seed{seed}",
        )

    expected_global = {
        "teacher_mean_return": statistics.fmean(
            row["hybrid_return"] for row in expected_summary_rows.values()
        ),
        "teacher_mean_cost": statistics.fmean(
            row["hybrid_cost"] for row in expected_summary_rows.values()
        ),
        "teacher_success_rate": statistics.fmean(
            row["hybrid_success"] for row in expected_summary_rows.values()
        ),
        "teacher_safe_success_rate": statistics.fmean(
            row["hybrid_safe_success"] for row in expected_summary_rows.values()
        ),
        "teacher_collision_rate": statistics.fmean(
            row["hybrid_collision"] for row in expected_summary_rows.values()
        ),
        "teacher_safe_termination_rate": statistics.fmean(
            row["hybrid_safe_termination"] for row in expected_summary_rows.values()
        ),
        "teacher_worst_physical_margin": min(
            row["hybrid_min_physical_margin"]
            for row in expected_summary_rows.values()
        ),
        "teacher_policy_inference_latency_ms_mean": statistics.fmean(
            row["policy_inference_latency_ms_mean"]
            for row in expected_summary_rows.values()
        ),
        "teacher_guardian_certification_latency_ms_mean": statistics.fmean(
            row["guardian_certification_latency_ms_mean"]
            for row in expected_summary_rows.values()
        ),
        "teacher_certified_control_latency_ms_mean": statistics.fmean(
            row["certified_control_latency_ms_mean"]
            for row in expected_summary_rows.values()
        ),
    }
    _require(
        all(_same_float(summary.get(name), value) for name, value in expected_global.items()),
        "teacher_global_summary",
    )
    _require(
        int(summary.get("teacher_environment_steps_total", -1))
        == sum(row["environment_steps"] for row in expected_summary_rows.values())
        and summary.get("ordinary_metrics") is None
        and summary.get("hybrid_intervention_rate") is None,
        "teacher_summary_contract",
    )
    return {
        "path": str(path.resolve()),
        "sha256": digests.sha256(path),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "raw_evaluation_sha256": payload["raw_evaluation_sha256"],
        "worker_pairs": 10,
        "unique_worker_processes": len(worker_pids),
        "worker_evidence": sorted(worker_evidence, key=lambda row: row["seed"]),
        "safe_termination_count": sum(
            int(row["hybrid_safe_termination"])
            for row in expected_summary_rows.values()
        ),
        "uncertified_proposals_executed": False,
        "audit_passed": True,
    }


def audit_results(
    root: Path,
    *,
    output: Path,
    expected_revision: str,
    require_complete_matrix: bool = False,
    settle_seconds: float = 3.0,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    _require(root.is_dir(), f"results_root:{root}")
    invalidation = root / INVALIDATION_MARKER
    _require(not invalidation.exists(), f"results_invalidated:{invalidation}")
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected_revision) is not None,
        "expected_revision_format",
    )
    _require(
        expected_revision == FORMAL_TRAINING_REVISION,
        f"unsupported_training_revision:{expected_revision}",
    )
    previous = _read_json(output) if output.is_file() else {}
    digests = StableDigests(previous.get("file_evidence"))
    entries = []
    resumes = []
    final_bindings = []
    failures = []
    discovered_runs = []
    offline_dataset = None
    guardian_environment = None
    now = time.time()

    for method, seeds in METHOD_SEEDS.items():
        for seed in seeds:
            run_dir = root / method / f"train_seed_{seed}"
            if not run_dir.is_dir():
                continue
            discovered_runs.append(f"{method}/train_seed_{seed}")
            try:
                config = _audit_config(run_dir, method, seed, expected_revision)
                observed_offline_dataset = _audit_offline_dataset_manifest(
                    config, digests
                )
                if offline_dataset is None:
                    offline_dataset = observed_offline_dataset
                else:
                    _require(
                        observed_offline_dataset == offline_dataset,
                        f"offline_manifest_binding:{run_dir}",
                    )
                observed_environment = config["environment"]
                if guardian_environment is None:
                    guardian_environment = observed_environment
                else:
                    _require(
                        observed_environment == guardian_environment,
                        f"guardian_environment_binding:{run_dir}",
                    )
                summaries = sorted(run_dir.glob("evaluation_*_summary.json"))
                stable_summaries = [
                    path
                    for path in summaries
                    if now - path.stat().st_mtime >= settle_seconds
                ]
                observed_steps = [
                    int(SUMMARY_PATTERN.fullmatch(path.name).group(1))
                    for path in stable_summaries
                    if SUMMARY_PATTERN.fullmatch(path.name)
                ]
                if observed_steps:
                    expected_steps = (
                        [0]
                        if method == "flow_bc"
                        else list(range(0, max(observed_steps) + 1, 50_000))
                    )
                    _require(observed_steps == expected_steps, f"evaluation_sequence:{run_dir}")
                for summary_path in stable_summaries:
                    entries.append(
                        _audit_evaluation(
                            run_dir, method, config, summary_path, digests
                        )
                    )
                resume = _audit_resume(
                    run_dir, digests, settle_seconds=settle_seconds
                )
                if resume is not None:
                    resumes.append(
                        {
                            "method": method,
                            "algorithm_seed": seed,
                            **resume,
                        }
                    )
                final_binding = _audit_final_binding(
                    run_dir,
                    method,
                    seed,
                    entries,
                    digests,
                    settle_seconds=settle_seconds,
                )
                if final_binding is not None:
                    final_bindings.append(final_binding)
            except (AuditFailure, OSError, ValueError, TypeError) as exc:
                failures.append(f"{method}/train_seed_{seed}:{exc}")

    teacher = None
    try:
        teacher = _audit_teacher(root, digests)
    except (AuditFailure, OSError, ValueError, TypeError) as exc:
        failures.append(f"guardian_mpc:{exc}")

    run_state_independence = None
    try:
        run_state_independence = _audit_run_state_independence(
            root,
            method_seeds=METHOD_SEEDS,
            discovered_runs=discovered_runs,
            resumes=resumes,
            final_bindings=final_bindings,
            expected_source_revision=expected_revision,
            expected_trainer_sha256=PAPER_TRAINER_ENTRYPOINT_SHA256,
            require_complete=require_complete_matrix,
            scope="main",
        )
    except (AuditFailure, OSError, ValueError, TypeError) as exc:
        failures.append(f"run_state_independence:{exc}")

    expected_entry_count = 3 + 21 * (3 + 3 + 1)
    suite_summary = root / "suite_summary.json"
    suite_complete = False
    if suite_summary.is_file():
        try:
            suite_complete = _read_json(suite_summary).get("completed") is True
        except AuditFailure as exc:
            failures.append(f"suite_summary:{exc}")
    if require_complete_matrix:
        if not suite_complete:
            failures.append("complete_matrix:suite_not_complete")
        if teacher is None:
            failures.append("complete_matrix:teacher_missing")
        if len(entries) != expected_entry_count:
            failures.append(
                f"complete_matrix:checkpoint_count:{len(entries)}:{expected_entry_count}"
            )
        expected_runs = {
            f"{method}/train_seed_{seed}"
            for method, seeds in METHOD_SEEDS.items()
            for seed in seeds
        }
        if set(discovered_runs) != expected_runs:
            failures.append("complete_matrix:run_set")
        if {
            f"{row['method']}/train_seed_{row['algorithm_seed']}"
            for row in final_bindings
        } != expected_runs:
            failures.append("complete_matrix:final_binding_set")

    payload = {
        "format": "guardian_flow_paper_checkpoint_audit_v1",
        "generated_at_unix": time.time(),
        "results_root": str(root),
        "expected_source_revision": expected_revision,
        "exact_training_gpu": "NVIDIA GeForce RTX 4080 SUPER",
        "paper_scenes": list(PAPER_SCENES),
        "offline_dataset": offline_dataset,
        "guardian_environment": guardian_environment,
        "guardian_dependence_protocol": GUARDIAN_DEPENDENCE_PROTOCOL,
        "deterministic_evaluation_protocol": DETERMINISTIC_EVALUATION_PROTOCOL,
        # Every accepted entry above is bound to one method/algorithm-seed run,
        # one immutable checkpoint, and that method's frozen objective.  The
        # auditor aggregates evidence only; it never constructs a policy from
        # multiple runs or checkpoints.
        "composite_or_ensemble": False,
        "run_state_independence": run_state_independence,
        "discovered_runs": discovered_runs,
        "checkpoint_evaluations": sorted(
            entries,
            key=lambda row: (
                tuple(METHOD_SEEDS).index(row["method"]),
                row["algorithm_seed"],
                row["step"],
            ),
        ),
        "atomic_resume_snapshots": sorted(
            resumes,
            key=lambda row: (
                tuple(METHOD_SEEDS).index(row["method"]),
                row["algorithm_seed"],
            ),
        ),
        "final_checkpoint_bindings": sorted(
            final_bindings,
            key=lambda row: (
                tuple(METHOD_SEEDS).index(row["method"]),
                row["algorithm_seed"],
            ),
        ),
        "teacher": teacher,
        "suite_complete": suite_complete,
        "require_complete_matrix": bool(require_complete_matrix),
        "expected_complete_checkpoint_evaluations": expected_entry_count,
        "audit_passed": not failures,
        "failures": failures,
        "file_evidence": digests.current,
    }
    if previous:
        previous_semantics = dict(previous)
        current_semantics = dict(payload)
        previous_generated_at = previous_semantics.pop("generated_at_unix", None)
        current_semantics.pop("generated_at_unix", None)
        if previous_semantics == current_semantics and _is_finite_number(
            previous_generated_at
        ):
            payload["generated_at_unix"] = float(previous_generated_at)
    _atomic_json(output, payload)
    if failures:
        raise AuditFailure(" | ".join(failures))
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--require-complete-matrix", action="store_true")
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output or args.results_root / "checkpoint_audit.json"
    try:
        payload = audit_results(
            args.results_root,
            output=output,
            expected_revision=args.expected_source_revision,
            require_complete_matrix=args.require_complete_matrix,
            settle_seconds=max(args.settle_seconds, 0.0),
        )
    except AuditFailure as exc:
        print(f"CHECKPOINT_AUDIT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "audit_passed": True,
                "checkpoint_evaluations": len(payload["checkpoint_evaluations"]),
                "atomic_resume_snapshots": len(payload["atomic_resume_snapshots"]),
                "suite_complete": payload["suite_complete"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
