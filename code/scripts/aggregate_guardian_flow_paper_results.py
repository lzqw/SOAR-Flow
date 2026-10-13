#!/usr/bin/env python3
"""Audit and aggregate the unified Guardian-Flow-DSAC paper experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.utils.flow_paper import (  # noqa: E402
    ALGORITHM_SEEDS,
    CORRECTION_SEED_WEIGHTS,
    NOVEL_SEEDS,
    OFFLINE_SEEDS,
    PAPER_SCENE_SEEDS,
    TRANSITION_TARGET_WEIGHTS,
)
from scripts.train_guardian_dsac_metasafe import _sha256, _write_json  # noqa: E402
from scripts.audit_guardian_flow_paper_checkpoints import (  # noqa: E402
    RUN_STATE_INDEPENDENCE_FORMAT,
    RUN_STATE_SERIALIZATION_CONTRACT,
)
from scripts.train_guardian_flow_dsac_paper import (  # noqa: E402
    METHOD_NAMES,
    _validate_paper_evaluation,
)


METRICS = (
    "ordinary_mean_return",
    "ordinary_success_rate",
    "ordinary_safe_success_rate",
    "ordinary_mean_cost",
    "ordinary_collision_rate",
    "hybrid_intervention_rate",
)
AUXILIARY_CURVE_METRICS = (
    "ordinary_safe_termination_rate",
    "hybrid_safe_termination_rate",
)
DISPLAY = {
    "ordinary_mean_return": "Ordinary Return ↑",
    "ordinary_success_rate": "Success ↑",
    "ordinary_safe_success_rate": "Safe Success ↑",
    "ordinary_mean_cost": "Cost ↓",
    "ordinary_collision_rate": "Collision ↓",
    "hybrid_intervention_rate": "Hybrid Intervention ↓",
}
METHOD_SCOPE = {
    "independent_rows": [
        "guardian_mpc",
        "flow_bc",
        "gaussian",
        "flow_q_only",
        "full_flow",
    ],
    "independent_ablation_rows": ["flow_no_p", "flow_no_correction"],
    "shared_offline_initialization_only": True,
    "post_initialization_model_optimizer_replay_independent": True,
    "evaluation_checkpoint_owned_by_method_run": True,
    "aggregation_combines_metrics_not_policies": True,
    "parameter_fusion": False,
    "checkpoint_splicing": False,
    "cross_method_voting": False,
    "cross_method_weighting": False,
    "additional_composite_or_ensemble_method": False,
    "ordinary_hybrid_are_evaluation_modes": True,
}
ABLATION_METHODS = ("flow_no_p", "flow_no_correction")
FLOW_ONLINE_METHODS = {
    "full_flow",
    "flow_q_only",
    "flow_no_p",
    "flow_no_correction",
}
MAX_FINAL_TRANSITION_FRACTION_DEVIATION = 0.003
ABLATION_TRAINING_REVISION = "389f26e78814854bc475497131d7fc9e3a5ab86d"
ABLATION_TRAINER_SHA256 = (
    "faf726830a698b79e047b318499c3cb05cf7337de53eaa370c9a2155d8091f6b"
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
GUARDIAN_DEPENDENCE_PROTOCOL = {
    "target": "G_t^P=c_t+(1-c_t)*gamma_P*G_{t+1}^P",
    "discount_gamma_p": 0.97,
    "construction": "reverse_complete_episode",
    "replay_commit_boundary": "terminal_or_truncated_episode_only",
    "incomplete_episode_policy": "discard_without_replay",
    "trainer_entrypoint": "scripts/train_guardian_flow_dsac_paper.py",
    "trainer_entrypoint_sha256": (
        "19e8c8a0b5e46568e06847844dc8203febf4af3aed5476e289c2fbd5ee663746"
    ),
    "training_source_revision": "99383a489fe687509fecc6b06d457ebbafecf7ef",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-ablations", action="store_true")
    return parser.parse_args(argv)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _run_paths(root: Path, method: str):
    return [root / method / f"train_seed_{seed}" for seed in ALGORITHM_SEEDS]


def _expected_steps(method: str, component_smoke: bool) -> int:
    if method == "flow_bc":
        return 0
    if method in ABLATION_METHODS:
        return 200 if component_smoke else 300_000
    return 200 if component_smoke else 1_000_000


def _weights_match(observed, expected) -> bool:
    if not isinstance(observed, dict) or set(observed) != {
        str(seed) for seed in expected
    }:
        return False
    return all(
        np.isclose(float(observed[str(seed)]), float(value))
        for seed, value in expected.items()
    )


def _audit_evaluation(evaluation, *, expected_step: int) -> list[str]:
    failures = []
    if int(evaluation.get("step", -1)) != int(expected_step):
        failures.append(f"evaluation_step_{expected_step}")
    path = Path(str(evaluation.get("evaluation", "")))
    if not path.is_file() or _sha256(path) != evaluation.get("evaluation_sha256"):
        failures.append(f"evaluation_sha256_{expected_step}")
        return failures
    try:
        raw = _read(path)
        _validate_paper_evaluation(
            raw,
            required_modes=("ordinary", "hybrid"),
            require_value_network_metrics=True,
        )
    except (
        OSError,
        json.JSONDecodeError,
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        failures.append(f"evaluation_integrity_{expected_step}")
    if not all(
        np.isfinite(float(evaluation.get(metric, np.nan))) for metric in METRICS
    ):
        failures.append(f"evaluation_finite_metrics_{expected_step}")
    checkpoint = Path(str(evaluation.get("checkpoint", "")))
    if not checkpoint.is_file() or _sha256(checkpoint) != evaluation.get(
        "checkpoint_sha256"
    ):
        failures.append(f"evaluation_checkpoint_sha256_{expected_step}")
    return failures


def _audit_run(path: Path, method: str, seed: int) -> dict[str, Any]:
    summary_path = path / "summary.json"
    config_path = path / "config.json"
    if not summary_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"missing run summary/config: {path}")
    summary = _read(summary_path)
    config = _read(config_path)
    failures = []
    if config.get("method_slug") != method or int(config.get("seed", -1)) != seed:
        failures.append("method_or_algorithm_seed")
    if tuple(config.get("scene_seeds", ())) != PAPER_SCENE_SEEDS:
        failures.append("paper_scene_set")
    if tuple(config.get("offline_seeds", ())) != OFFLINE_SEEDS:
        failures.append("offline_seed_roles")
    if tuple(config.get("online_novel_seeds", ())) != NOVEL_SEEDS:
        failures.append("novel_seed_roles")
    if 9 in config.get("scene_seeds", ()):
        failures.append("seed9_present")
    for flag in (
        "moving_safe_proximal",
        "safe_trajectory_memory",
        "seed_specific_logic",
        "performance_rollback",
        "conditional_launch",
        "evaluation_changes_training",
    ):
        if config.get(flag) is not False:
            failures.append(flag)
    if config.get("numerical_disaster_rollback_only") is not True:
        failures.append("numerical_rollback_binding")
    gpu_rows = config.get("nvidia_smi", {}).get("rows", ())
    if not any("NVIDIA GeForce RTX 4080 SUPER" in row for row in gpu_rows):
        failures.append("rtx4080_super_evidence")
    if int(config["environment"]["max_episode_steps"]) != 1200:
        failures.append("horizon")
    if not np.isclose(float(config["environment"]["max_command_speed"]), 10.75):
        failures.append("maximum_speed")
    expected_network = {
        "observation_dim": 259,
        "actor_observation_dim": 259,
        "action_dim": 2,
    }
    if any(
        int(config.get(name, -1)) != value
        for name, value in expected_network.items()
    ):
        failures.append("network_dimensions")
    if tuple(config.get("hidden_sizes", ())) != (256, 256):
        failures.append("shared_hidden_sizes")
    if not np.isclose(float(config.get("init_alpha", np.nan)), 0.05):
        failures.append("initial_entropy_temperature")
    if config.get("backend") != "gpu":
        failures.append("training_backend")
    environment = config.get("environment")
    if (
        not isinstance(environment, dict)
        or set(environment) != set(EXPECTED_GUARDIAN_ENVIRONMENT)
        or any(
            not np.isclose(float(environment.get(key, np.nan)), value)
            for key, value in EXPECTED_GUARDIAN_ENVIRONMENT.items()
        )
    ):
        failures.append("guardian_environment_protocol")
    if not _weights_match(
        config.get("transition_target_weights"), TRANSITION_TARGET_WEIGHTS
    ):
        failures.append("transition_target_weights")
    flow = method != "gaussian"
    expected_correction = CORRECTION_SEED_WEIGHTS if flow else None
    observed_correction = config.get("correction_seed_sampling_weights")
    if flow:
        if tuple(config.get("flow_hidden_sizes", ())) != (256, 256):
            failures.append("flow_hidden_sizes")
        if int(config.get("flow_integration_steps", -1)) != 8:
            failures.append("flow_teacher_integration_steps")
        if not _weights_match(observed_correction, expected_correction):
            failures.append("correction_seed_sampling_weights")
        zero_latent = config.get("zero_latent_probability_by_seed")
        if not isinstance(zero_latent, dict) or any(
            not np.isclose(float(zero_latent.get(str(seed), np.nan)), 0.5)
            for seed in PAPER_SCENE_SEEDS
        ):
            failures.append("zero_latent_protocol")
        flow_config = config.get("flow_config") or {}
        expected_proxy_weight = (
            0.0 if method in ("flow_q_only", "flow_no_p") else 0.15
        )
        expected_correction_weight = (
            0.0 if method in ("flow_q_only", "flow_no_correction") else 2.0
        )
        expected_scalars = {
            "q_actor_weight": 1.0,
            "proxy_actor_weight": expected_proxy_weight,
            "correction_weight": expected_correction_weight,
            "correction_speed_weight": 2.0,
            "correction_lateral_weight": 1.0,
            "zero_latent_correction_weight": 1.0,
            "anchor_weight": 0.05,
        }
        if any(
            not np.isclose(float(flow_config.get(name, np.nan)), value)
            for name, value in expected_scalars.items()
        ):
            failures.append("flow_objective_weights")
    elif (
        observed_correction is not None
        or config.get("zero_latent_probability_by_seed") is not None
        or config.get("flow_hidden_sizes") is not None
        or config.get("flow_integration_steps") is not None
    ):
        failures.append("gaussian_latent_not_applicable")
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
    if config.get("actor_objective_terms") != expected_objective:
        failures.append("actor_objective_terms")
    budgets = config.get("budgets", {})
    component_smoke = bool(config.get("component_smoke"))
    if not component_smoke:
        formal_budget = {
            "flow_matching_steps": 80_000 if flow else 0,
            "same_latent_distillation_steps": 40_000 if flow else 0,
            "offline_q_guided_actor_steps": 0,
            "online_steps": (
                0
                if method == "flow_bc"
                else 300_000
                if method in ABLATION_METHODS
                else 1_000_000
            ),
            "batch_size": 512,
            "online_buffer_size": 500_000,
            "gradient_steps": 1,
            "policy_delay": 2,
            "evaluation_every": 50_000,
            "checkpoint_every": 50_000,
        }
        if any(
            int(budgets.get(name, -1)) != value for name, value in formal_budget.items()
        ):
            failures.append("formal_budget")
        if not np.isclose(float(budgets.get("offline_replay_ratio", np.nan)), 0.20):
            failures.append("offline_replay_ratio")
        if flow and not np.isclose(
            float(budgets.get("correction_fraction", np.nan)), 0.25
        ):
            failures.append("correction_fraction")
    source_code = config.get("source_code", {})
    if not bool(config.get("component_smoke")):
        if source_code.get("clean") is not True:
            failures.append("dirty_formal_source")
        if not re.fullmatch(r"[0-9a-f]{40}", str(source_code.get("git_revision", ""))):
            failures.append("source_revision")
        source_hashes = source_code.get("source_sha256", {})
        if not source_hashes or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in source_hashes.values()
        ):
            failures.append("source_sha256")
        if (
            source_hashes.get(PAPER_ISOLATED_EVALUATOR_ENTRYPOINT)
            != PAPER_ISOLATED_EVALUATOR_SHA256
            or source_hashes.get(PAPER_EVALUATION_WRAPPER_ENTRYPOINT)
            != PAPER_EVALUATION_WRAPPER_SHA256
        ):
            failures.append("deterministic_evaluator_source")
        if source_code.get("missing_source_files"):
            failures.append("missing_source_file")
        if method in ABLATION_METHODS:
            if source_code.get("git_revision") != ABLATION_TRAINING_REVISION:
                failures.append("ablation_source_revision")
            if source_code.get("source_sha256", {}).get(
                "scripts/train_guardian_flow_dsac_paper.py"
            ) != ABLATION_TRAINER_SHA256:
                failures.append("ablation_trainer_sha256")
            if seed != 1:
                failures.append("ablation_algorithm_seed")
            if config.get("independent_single_model_run") is not True:
                failures.append("ablation_independent_model")
            if config.get("composite_or_ensemble") is not False:
                failures.append("ablation_composite_or_ensemble")
            initialization = config.get("offline_initialization_binding") or {}
            if (
                initialization.get("revision")
                != GUARDIAN_DEPENDENCE_PROTOCOL["training_source_revision"]
                or initialization.get("trainer_sha256")
                != GUARDIAN_DEPENDENCE_PROTOCOL["trainer_entrypoint_sha256"]
                or initialization.get("method_slug") != "flow_bc"
                or int(initialization.get("algorithm_seed", -1)) != 1
            ):
                failures.append("ablation_offline_initialization_binding")
    if method != "flow_bc":
        online = summary.get("online") or {}
        expected = _expected_steps(method, component_smoke)
        if int(online.get("actual_online_steps", -1)) != expected:
            failures.append("complete_online_budget")
        if online.get("completed_full_budget") is not True:
            failures.append("full_budget_flag")
        if int(online.get("performance_rollbacks", -1)) != 0:
            failures.append("performance_rollback_count")
        if online.get("evaluation_changes_training") is not False:
            failures.append("evaluation_mutated_training")
        for flag in ("task_q_updated", "p_updated", "actor_updated"):
            if online.get(flag) is not True:
                failures.append(flag)
        if flow and online.get("frozen_teacher_unchanged") is not True:
            failures.append("frozen_teacher_changed")
        if not component_smoke:
            observed_schedule = online.get("transition_fraction_by_seed", {})
            if set(observed_schedule) != {
                str(scene_seed) for scene_seed in PAPER_SCENE_SEEDS
            } or any(
                not np.isfinite(float(observed_schedule[str(scene_seed)]))
                or abs(
                    float(observed_schedule[str(scene_seed)])
                    - TRANSITION_TARGET_WEIGHTS[scene_seed]
                )
                > 0.003
                for scene_seed in PAPER_SCENE_SEEDS
            ):
                failures.append("transition_schedule_tolerance")
            if (
                sum(int(value) for value in online.get("steps_by_seed", {}).values())
                != expected
            ):
                failures.append("transition_schedule_total")
            if flow:
                latent_counts = online.get("latent_episode_counts_by_seed") or {}
                if any(
                    int(latent_counts.get(str(scene_seed), {}).get("zero", 0)) < 1
                    or int(latent_counts.get(str(scene_seed), {}).get("random", 0)) < 1
                    or abs(
                        int(latent_counts[str(scene_seed)]["zero"])
                        - int(latent_counts[str(scene_seed)]["random"])
                    )
                    > 1
                    for scene_seed in PAPER_SCENE_SEEDS
                ):
                    failures.append("latent_episode_balance")
            elif (
                online.get("latent_steps_by_seed") is not None
                or online.get("latent_episode_counts_by_seed") is not None
            ):
                failures.append("gaussian_latent_statistics")
        observed_steps = [int(row["step"]) for row in online.get("evaluations", ())]
        interval = int(config["budgets"]["evaluation_every"])
        expected_curve = list(range(0, expected + 1, interval))
        if observed_steps != expected_curve:
            failures.append("evaluation_curve_steps")
        for evaluation, step in zip(online.get("evaluations", ()), expected_curve):
            failures.extend(_audit_evaluation(evaluation, expected_step=step))
    else:
        failures.extend(
            _audit_evaluation(summary.get("final_evaluation", {}), expected_step=0)
        )
    final = summary.get("final_evaluation") or {}
    expected_final_step = _expected_steps(method, component_smoke)
    if int(final.get("step", -1)) != expected_final_step:
        failures.append("final_evaluation_step")
    permanent = path / f"checkpoint_{expected_final_step:08d}.pkl"
    if (
        not permanent.is_file()
        or Path(str(final.get("checkpoint", ""))).resolve() != permanent.resolve()
        or _sha256(permanent) != final.get("checkpoint_sha256")
    ):
        failures.append("final_evaluation_checkpoint_binding")
    if method != "flow_bc":
        online = summary.get("online") or {}
        if (
            online.get("final_evaluation") != final
            or not online.get("evaluations")
            or online["evaluations"][-1] != final
        ):
            failures.append("final_online_evaluation_binding")
    if int(final.get("scene_count", -1)) != len(PAPER_SCENE_SEEDS):
        failures.append("final_scene_denominator")
    if final.get("all_scene_denominator") is not True:
        failures.append("all_scene_return_cost")
    if set(final.get("per_seed", {})) != {str(seed) for seed in PAPER_SCENE_SEEDS}:
        failures.append("final_per_seed_rows")
    checkpoint = Path(summary.get("final_checkpoint", ""))
    if not checkpoint.is_file() or _sha256(checkpoint) != summary.get(
        "final_checkpoint_sha256"
    ):
        failures.append("final_checkpoint_sha256")
    elif (
        checkpoint.resolve() != (path / "final.pkl").resolve()
        or _sha256(checkpoint) != final.get("checkpoint_sha256")
    ):
        failures.append("final_checkpoint_evaluation_binding")
    if failures:
        raise RuntimeError(
            f"paper run audit failed for {path}: {sorted(set(failures))}"
        )
    return {
        "path": str(path),
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "summary": summary,
        "config": config,
        "final": final,
        "audit_passed": True,
    }


def _statistics(runs, metric):
    values = np.asarray([float(run["final"][metric]) for run in runs], np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "values": values.tolist(),
        "n": int(values.size),
    }


def _format(stat, digits=3):
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _teacher_row(root: Path):
    path = root / "guardian_mpc" / "teacher_evaluation.json"
    if not path.is_file():
        raise FileNotFoundError("Guardian/MPC teacher evaluation is missing")
    payload = _read(path)
    if payload.get("format") != "guardian_flow_dsac_paper_evaluation_v1":
        raise RuntimeError("teacher evaluation format mismatch")
    if (
        payload.get("policy") != "teacher"
        or int(payload.get("episodes_per_scene", -1)) != 1
    ):
        raise RuntimeError("teacher evaluation protocol mismatch")
    if tuple(payload.get("scene_seeds", ())) != PAPER_SCENE_SEEDS:
        raise RuntimeError("teacher evaluation scene set mismatch")
    checkpoint = Path(str(payload.get("checkpoint", "")))
    if not checkpoint.is_file() or _sha256(checkpoint) != payload.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("teacher checkpoint SHA256 mismatch")
    raw_path = Path(str(payload.get("raw_evaluation", "")))
    if not raw_path.is_file() or _sha256(raw_path) != payload.get(
        "raw_evaluation_sha256"
    ):
        raise RuntimeError("teacher raw evaluation SHA256 mismatch")
    raw = _read(raw_path)
    _validate_paper_evaluation(
        raw,
        required_modes=("hybrid",),
        require_value_network_metrics=False,
    )
    summary = payload["summary"]
    if (
        int(summary.get("scene_count", -1)) != len(PAPER_SCENE_SEEDS)
        or summary.get("all_scene_denominator") is not True
    ):
        raise RuntimeError("teacher evaluation denominator mismatch")
    return path, payload, summary


def _audit_matrix(methods, teacher_payload) -> dict[str, Any]:
    runs = [run for method_runs in methods.values() for run in method_runs]
    if not runs:
        raise RuntimeError("paper matrix has no learning runs")

    def unique_json(selector):
        return {
            json.dumps(selector(run), sort_keys=True, separators=(",", ":"))
            for run in runs
        }

    bindings = {
        "dataset_sha256": {run["config"]["dataset_sha256"] for run in runs},
        "manifest_sha256": {run["config"]["manifest_sha256"] for run in runs},
        "shared_qp_sha256": {
            run["summary"]["offline"]["shared_qp_checkpoint_sha256"] for run in runs
        },
        "source_revision": {
            run["config"].get("source_code", {}).get("git_revision") for run in runs
        },
        "scene_signatures": unique_json(lambda run: run["config"]["scene_signatures"]),
        "environment": unique_json(lambda run: run["config"]["environment"]),
        "source_sha256": unique_json(
            lambda run: run["config"].get("source_code", {}).get("source_sha256")
        ),
    }
    divergent = sorted(name for name, values in bindings.items() if len(values) != 1)
    if divergent:
        raise RuntimeError("paper matrix binding mismatch: " + ", ".join(divergent))

    flow_bc = {
        int(run["summary"]["algorithm_seed"]): run for run in methods.get("flow_bc", ())
    }
    for method in ("full_flow", "flow_q_only"):
        for run in methods.get(method, ()):
            seed = int(run["summary"]["algorithm_seed"])
            if seed not in flow_bc or run["summary"]["offline"].get(
                "source_sha256"
            ) != flow_bc[seed]["summary"].get("final_checkpoint_sha256"):
                raise RuntimeError(
                    f"{method} algorithm Seed{seed} is not bound to its Flow-BC run"
                )
    if 0 in flow_bc and teacher_payload.get("checkpoint_sha256") != flow_bc[0][
        "summary"
    ].get("final_checkpoint_sha256"):
        raise RuntimeError(
            "Guardian/MPC teacher is not bound to Flow-BC Seed0 metadata"
        )
    result = {name: next(iter(values)) for name, values in bindings.items()}
    for name in ("scene_signatures", "environment", "source_sha256"):
        result[name] = json.loads(result[name])
    return result


def _validated_run_state_evidence(
    audit: dict[str, Any],
    *,
    expected_runs: set[str],
    expected_scope: str,
    expected_revision: str,
    expected_trainer_sha256: str,
) -> dict[str, Any]:
    evidence = audit.get("run_state_independence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("format") != RUN_STATE_INDEPENDENCE_FORMAT
        or evidence.get("scope") != expected_scope
        or evidence.get("expected_source_revision") != expected_revision
        or evidence.get("expected_trainer_entrypoint_sha256")
        != expected_trainer_sha256
        or evidence.get("serialization_contract")
        != RUN_STATE_SERIALIZATION_CONTRACT
        or set(evidence.get("expected_runs", ())) != expected_runs
        or set(evidence.get("audited_runs", ())) != expected_runs
        or evidence.get("pending_runs") != []
        or evidence.get("matrix_complete") is not True
        or evidence.get("unique_run_directories") is not True
        or evidence.get("unique_state_artifact_paths_across_runs") is not True
        or evidence.get("unique_state_artifact_inodes_across_runs") is not True
        or evidence.get("single_link_regular_state_files") is not True
        or evidence.get("post_initialization_model_optimizer_replay_rng_independent")
        is not True
        or int(evidence.get("shared_state_artifact_count", -1)) != 0
        or evidence.get("composite_or_ensemble") is not False
        or evidence.get("audit_passed") is not True
    ):
        raise RuntimeError(
            f"{expected_scope} run-state independence audit is incomplete"
        )
    rows = evidence.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected_runs)
        or {
            f"{row.get('method')}/train_seed_{row.get('algorithm_seed')}"
            for row in rows
        }
        != expected_runs
        or any(
            row.get("independent_single_model") is not True
            or row.get("composite_or_ensemble") is not False
            or row.get("audit_passed") is not True
            for row in rows
        )
    ):
        raise RuntimeError(f"{expected_scope} run-state evidence rows mismatch")
    return evidence


def _experiment_setup(methods, matrix_bindings) -> dict[str, Any]:
    runs = [run for method_runs in methods.values() for run in method_runs]
    if not runs:
        raise RuntimeError("paper matrix has no experiment setup")
    common = {
        (
            int(run["config"]["observation_dim"]),
            int(run["config"]["actor_observation_dim"]),
            int(run["config"]["action_dim"]),
            tuple(run["config"]["hidden_sizes"]),
            float(run["config"]["init_alpha"]),
            run["config"]["backend"],
        )
        for run in runs
    }
    if common != {(259, 259, 2, (256, 256), 0.05, "gpu")}:
        raise RuntimeError("paper matrix network setup mismatch")
    if matrix_bindings.get("environment") != EXPECTED_GUARDIAN_ENVIRONMENT:
        raise RuntimeError("paper matrix Guardian environment protocol mismatch")
    source_hashes = matrix_bindings.get("source_sha256")
    if (
        matrix_bindings.get("source_revision")
        != GUARDIAN_DEPENDENCE_PROTOCOL["training_source_revision"]
        or not isinstance(source_hashes, dict)
        or source_hashes.get(GUARDIAN_DEPENDENCE_PROTOCOL["trainer_entrypoint"])
        != GUARDIAN_DEPENDENCE_PROTOCOL["trainer_entrypoint_sha256"]
    ):
        raise RuntimeError("paper matrix Guardian-dependence source mismatch")
    if {
        run["config"].get("action_semantics_version") for run in runs
    } != {"target_speed_lateral_v1"}:
        raise RuntimeError("paper matrix action semantics mismatch")
    flow = [
        run
        for method, method_runs in methods.items()
        if method != "gaussian"
        for run in method_runs
    ]
    if {
        (
            tuple(run["config"]["flow_hidden_sizes"]),
            int(run["config"]["flow_integration_steps"]),
        )
        for run in flow
    } != {((256, 256), 8)}:
        raise RuntimeError("paper matrix Flow network setup mismatch")
    gaussian = methods.get("gaussian", ())
    if any(
        run["config"].get("flow_hidden_sizes") is not None
        or run["config"].get("flow_integration_steps") is not None
        for run in gaussian
    ):
        raise RuntimeError("Gaussian run contains Flow network configuration")
    return {
        "observation_dim": 259,
        "actor_observation_dim": 259,
        "action_dim": 2,
        "common_actor_q_p_hidden_sizes": [256, 256],
        "gaussian_actor_hidden_sizes": [256, 256],
        "flow_teacher_hidden_sizes": [256, 256],
        "one_step_flow_actor_hidden_sizes": [256, 256],
        "flow_teacher_integration_steps": 8,
        "one_step_flow_actor_forward_passes": 1,
        "initial_entropy_temperature": 0.05,
        "training_backend": "gpu",
        "task_q_distributional_loss": "hdsac",
        "environment_backend": (
            "metadrive.envs.safe_metadrive_env.SafeMetaDriveEnv"
        ),
        "environment_factory": (
            "relax.env.drive.metasafe.make_metasafe_cbf_mpc_env"
        ),
        "action_semantics_version": "target_speed_lateral_v1",
        "guardian_environment": matrix_bindings["environment"],
        "guardian_dependence_protocol": GUARDIAN_DEPENDENCE_PROTOCOL,
        "deterministic_evaluation_protocol": DETERMINISTIC_EVALUATION_PROTOCOL,
    }


def _audit_ablation_matrix(
    root: Path,
    methods: dict[str, list[dict[str, Any]]],
    ablations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    flow_bc_seed1 = [
        run
        for run in methods.get("flow_bc", ())
        if int(run["summary"]["algorithm_seed"]) == 1
    ]
    if len(flow_bc_seed1) != 1:
        raise RuntimeError("ablation matrix requires Flow-BC Seed1")
    source = flow_bc_seed1[0]
    source_checkpoint = Path(source["summary"]["final_checkpoint"]).resolve()
    source_sha = source["summary"]["final_checkpoint_sha256"]
    source_config = source["config"]
    run_dirs = set()
    final_checkpoints = set()
    bindings = []
    for method in ABLATION_METHODS:
        runs = ablations.get(method, ())
        if len(runs) != 1:
            raise RuntimeError(f"ablation matrix requires exactly one {method} run")
        run = runs[0]
        run_dir = Path(run["path"]).resolve()
        final_checkpoint = Path(run["summary"]["final_checkpoint"]).resolve()
        offline = run["summary"].get("offline") or {}
        config = run["config"]
        common_bindings = (
            config.get("dataset_sha256") == source_config.get("dataset_sha256")
            and config.get("manifest_sha256")
            == source_config.get("manifest_sha256")
            and config.get("scene_signatures")
            == source_config.get("scene_signatures")
            and config.get("environment") == source_config.get("environment")
            and config.get("algorithm_config")
            == source_config.get("algorithm_config")
            and config.get("transition_target_weights")
            == source_config.get("transition_target_weights")
            and config.get("correction_seed_sampling_weights")
            == source_config.get("correction_seed_sampling_weights")
            and config.get("zero_latent_probability_by_seed")
            == source_config.get("zero_latent_probability_by_seed")
        )
        if not common_bindings:
            raise RuntimeError(f"{method} diverges from shared Seed1 protocol")
        if (
            offline.get("loaded") is not True
            or Path(str(offline.get("source", ""))).resolve() != source_checkpoint
            or offline.get("source_sha256") != source_sha
            or offline.get("teacher_frozen") is not True
        ):
            raise RuntimeError(f"{method} does not use frozen Flow-BC Seed1")
        if source_checkpoint in (run_dir, final_checkpoint):
            raise RuntimeError(f"{method} overwrites its shared initialization")
        if run_dir in run_dirs or final_checkpoint in final_checkpoints:
            raise RuntimeError("ablation runs share a run directory or final checkpoint")
        if run_dir not in final_checkpoint.parents:
            raise RuntimeError(f"{method} final checkpoint is outside its run directory")
        run_dirs.add(run_dir)
        final_checkpoints.add(final_checkpoint)
        bindings.append(
            {
                "method": method,
                "algorithm_seed": 1,
                "run_directory": str(run_dir),
                "final_checkpoint": str(final_checkpoint),
                "final_checkpoint_sha256": run["summary"][
                    "final_checkpoint_sha256"
                ],
                "shared_flow_bc_checkpoint": str(source_checkpoint),
                "shared_flow_bc_checkpoint_sha256": source_sha,
                "independent_model": True,
                "independent_optimizer": True,
                "independent_online_replay": True,
                "composite_or_ensemble": False,
            }
        )
    return {
        "shared_flow_bc_seed": 1,
        "shared_flow_bc_checkpoint": str(source_checkpoint),
        "shared_flow_bc_checkpoint_sha256": source_sha,
        "runs": bindings,
        "independent_models": True,
        "shared_optimizer": False,
        "shared_online_replay": False,
        "composite_or_ensemble": False,
    }


def _sampling_audit(
    methods: dict[str, list[dict[str, Any]]],
    ablations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Expose the verified final scheduler and latent counts run by run."""

    rows = []
    groups = [
        *methods.items(),
        *ablations.items(),
    ]
    for method, runs in groups:
        if method == "flow_bc":
            continue
        for run in runs:
            summary = run["summary"]
            config = run["config"]
            online = summary.get("online") or {}
            component_smoke = bool(config.get("component_smoke"))
            expected_step = _expected_steps(method, component_smoke)
            if component_smoke:
                continue
            scene_keys = {str(seed) for seed in PAPER_SCENE_SEEDS}
            steps = online.get("steps_by_seed")
            fractions = online.get("transition_fraction_by_seed")
            if (
                not isinstance(steps, dict)
                or set(steps) != scene_keys
                or not isinstance(fractions, dict)
                or set(fractions) != scene_keys
            ):
                raise RuntimeError(f"sampling audit has incomplete scenes: {run['path']}")
            normalized_steps = {scene: int(steps[scene]) for scene in scene_keys}
            normalized_fractions = {
                scene: float(fractions[scene]) for scene in scene_keys
            }
            if sum(normalized_steps.values()) != expected_step or any(
                not np.isfinite(normalized_fractions[scene])
                or not np.isclose(
                    normalized_fractions[scene],
                    normalized_steps[scene] / max(expected_step, 1),
                    rtol=1e-12,
                    atol=1e-12,
                )
                for scene in scene_keys
            ):
                raise RuntimeError(f"sampling audit step/fraction mismatch: {run['path']}")
            deviations = {
                scene: abs(
                    normalized_fractions[scene]
                    - TRANSITION_TARGET_WEIGHTS[int(scene)]
                )
                for scene in scene_keys
            }
            maximum_deviation = max(deviations.values())
            if maximum_deviation > MAX_FINAL_TRANSITION_FRACTION_DEVIATION:
                raise RuntimeError(f"sampling audit schedule drift: {run['path']}")

            latent_counts = None
            maximum_latent_imbalance = None
            if method in FLOW_ONLINE_METHODS:
                observed = online.get("latent_episode_counts_by_seed")
                latent_steps = online.get("latent_steps_by_seed")
                if (
                    not isinstance(observed, dict)
                    or set(observed) != scene_keys
                    or not isinstance(latent_steps, dict)
                    or set(latent_steps) != scene_keys
                ):
                    raise RuntimeError(
                        f"sampling audit Flow latent scenes: {run['path']}"
                    )
                latent_counts = {}
                imbalances = []
                for scene in scene_keys:
                    attempts = observed[scene]
                    scene_steps = latent_steps[scene]
                    if (
                        not isinstance(attempts, dict)
                        or set(attempts) != {"zero", "random"}
                        or not isinstance(scene_steps, dict)
                        or set(scene_steps) != {"zero", "random"}
                    ):
                        raise RuntimeError(
                            f"sampling audit Flow latent modes: {run['path']}:Seed{scene}"
                        )
                    attempts = {
                        mode: int(attempts[mode]) for mode in ("zero", "random")
                    }
                    scene_steps = {
                        mode: int(scene_steps[mode])
                        for mode in ("zero", "random")
                    }
                    imbalance = abs(attempts["zero"] - attempts["random"])
                    if (
                        any(value < 1 for value in attempts.values())
                        or any(value < 0 for value in scene_steps.values())
                        or sum(scene_steps.values()) != normalized_steps[scene]
                        or imbalance > 1
                    ):
                        raise RuntimeError(
                            f"sampling audit Flow latent binding: {run['path']}:Seed{scene}"
                        )
                    latent_counts[scene] = {
                        "episode_attempts": attempts,
                        "transition_steps": scene_steps,
                        "episode_count_imbalance": imbalance,
                    }
                    imbalances.append(imbalance)
                maximum_latent_imbalance = max(imbalances)
            elif (
                online.get("latent_episode_counts_by_seed") is not None
                or online.get("latent_steps_by_seed") is not None
            ):
                raise RuntimeError(
                    f"sampling audit Gaussian has Flow statistics: {run['path']}"
                )

            rows.append(
                {
                    "method": method,
                    "algorithm_seed": int(summary["algorithm_seed"]),
                    "online_steps": expected_step,
                    "component_smoke": component_smoke,
                    "transition_steps_by_seed": normalized_steps,
                    "transition_fraction_by_seed": normalized_fractions,
                    "transition_fraction_deviation_by_seed": deviations,
                    "maximum_transition_fraction_deviation": maximum_deviation,
                    "latent_episode_and_step_evidence_by_seed": latent_counts,
                    "maximum_flow_latent_episode_count_imbalance": (
                        maximum_latent_imbalance
                    ),
                    "independent_single_model": True,
                    "composite_or_ensemble": False,
                    "audit_passed": True,
                }
            )

    formal_rows = rows
    flow_rows = [
        row
        for row in formal_rows
        if row["maximum_flow_latent_episode_count_imbalance"] is not None
    ]
    return {
        "format": "guardian_flow_paper_final_sampling_audit_v1",
        "transition_target_weights": {
            str(seed): value for seed, value in TRANSITION_TARGET_WEIGHTS.items()
        },
        "maximum_allowed_transition_fraction_deviation": (
            MAX_FINAL_TRANSITION_FRACTION_DEVIATION
        ),
        "maximum_observed_transition_fraction_deviation": (
            max(
                row["maximum_transition_fraction_deviation"]
                for row in formal_rows
            )
            if formal_rows
            else None
        ),
        "maximum_flow_latent_episode_count_imbalance": (
            max(
                row["maximum_flow_latent_episode_count_imbalance"]
                for row in flow_rows
            )
            if flow_rows
            else None
        ),
        "formal_online_runs": len(formal_rows),
        "rows": rows,
        "independent_single_models": True,
        "composite_or_ensemble": False,
        "audit_passed": all(row["audit_passed"] for row in rows),
    }


def _main_table(root, methods, teacher):
    teacher_values = {
        "ordinary_mean_return": teacher["teacher_mean_return"],
        "ordinary_success_rate": teacher["teacher_success_rate"],
        "ordinary_safe_success_rate": teacher["teacher_safe_success_rate"],
        "ordinary_mean_cost": teacher["teacher_mean_cost"],
        "ordinary_collision_rate": teacher["teacher_collision_rate"],
        "hybrid_intervention_rate": None,
    }
    rows = [
        {
            "Method": "Guardian/MPC",
            "Actor": "MPC",
            "Online Steps": 0,
            "Algorithm Runs": 1,
            **{
                DISPLAY[name]: ("N/A" if value is None else f"{float(value):.3f}")
                for name, value in teacher_values.items()
            },
        }
    ]
    actor = {
        "flow_bc": "Flow",
        "gaussian": "Gaussian",
        "flow_q_only": "Flow",
        "full_flow": "Flow",
    }
    statistics = {}
    for method in ("flow_bc", "gaussian", "flow_q_only", "full_flow"):
        if method not in methods:
            continue
        method_stats = {
            metric: _statistics(methods[method], metric) for metric in METRICS
        }
        statistics[method] = method_stats
        rows.append(
            {
                "Method": METHOD_NAMES[method],
                "Actor": actor[method],
                "Online Steps": int(
                    methods[method][0]["config"]["budgets"]["online_steps"]
                ),
                "Algorithm Runs": len(methods[method]),
                **{DISPLAY[name]: _format(method_stats[name]) for name in METRICS},
            }
        )
    fields = [
        "Method",
        "Actor",
        "Online Steps",
        "Algorithm Runs",
        *[DISPLAY[name] for name in METRICS],
    ]
    _write_csv(root / "main_table.csv", fields, rows)
    markdown = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    markdown.extend(
        "| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows
    )
    (root / "main_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return rows, statistics


def _per_seed_table(root, methods):
    rows = []
    for seed in PAPER_SCENE_SEEDS:
        row = {"Seed": seed, "Role": "Offline" if seed in OFFLINE_SEEDS else "Novel"}
        for method, prefix in (
            ("flow_bc", "Flow-BC"),
            ("gaussian", "Gaussian"),
            ("flow_q_only", "Flow-Q-only"),
            ("full_flow", "Full Flow"),
        ):
            if method not in methods:
                continue
            entries = [run["final"]["per_seed"][str(seed)] for run in methods[method]]
            row[f"{prefix} Return"] = float(
                np.mean([entry["ordinary_return"] for entry in entries])
            )
            row[f"{prefix} Cost"] = float(
                np.mean([entry["ordinary_cost"] for entry in entries])
            )
            row[f"{prefix} Success"] = float(
                np.mean([entry["ordinary_success"] for entry in entries])
            )
            row[f"{prefix} Safe Success"] = float(
                np.mean([entry["ordinary_safe_success"] for entry in entries])
            )
            row[f"{prefix} Ordinary Safe Termination"] = float(
                np.mean(
                    [entry["ordinary_safe_termination"] for entry in entries]
                )
            )
            row[f"{prefix} Hybrid Safe Termination"] = float(
                np.mean(
                    [entry["hybrid_safe_termination"] for entry in entries]
                )
            )
        rows.append(row)
    fields = ["Seed", "Role"] + sorted(
        {key for row in rows for key in row} - {"Seed", "Role"}
    )
    _write_csv(root / "per_seed_table.csv", fields, rows)
    return rows


def _learning_curves(root, methods):
    rows = []
    for method, runs in methods.items():
        for run in runs:
            seed = int(run["summary"]["algorithm_seed"])
            evaluations = (
                [run["final"]]
                if method == "flow_bc"
                else run["summary"]["online"]["evaluations"]
            )
            for evaluation in evaluations:
                rows.append(
                    {
                        "row_type": "run",
                        "method": method,
                        "algorithm_seed": seed,
                        "step": int(evaluation["step"]),
                        **{
                            metric: float(evaluation[metric])
                            for metric in METRICS + AUXILIARY_CURVE_METRICS
                        },
                    }
                )
        steps = sorted(
            {
                row["step"]
                for row in rows
                if row["method"] == method and row["row_type"] == "run"
            }
        )
        for step in steps:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and row["row_type"] == "run"
                and row["step"] == step
            ]
            aggregate = {
                "row_type": "aggregate",
                "method": method,
                "algorithm_seed": "mean_std",
                "step": step,
            }
            for metric in METRICS + AUXILIARY_CURVE_METRICS:
                values = np.asarray([row[metric] for row in selected], np.float64)
                aggregate[metric] = float(np.mean(values))
                aggregate[metric + "_std"] = float(np.std(values, ddof=0))
            rows.append(aggregate)
    fields = ["row_type", "method", "algorithm_seed", "step"]
    for metric in METRICS + AUXILIARY_CURVE_METRICS:
        fields.extend((metric, metric + "_std"))
    normalized = [{field: row.get(field, "") for field in fields} for row in rows]
    _write_csv(root / "learning_curves.csv", fields, normalized)
    return normalized


def _evaluation_at(run: dict[str, Any], step: int) -> dict[str, Any]:
    evaluations = run["summary"].get("online", {}).get("evaluations", ())
    matching = [row for row in evaluations if int(row.get("step", -1)) == step]
    if len(matching) != 1:
        raise RuntimeError(
            f"run {run['path']} has {len(matching)} evaluations at step {step}"
        )
    return matching[0]


def _ablation_outputs(
    root: Path,
    methods: dict[str, list[dict[str, Any]]],
    ablations: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_seed1 = [
        run
        for run in methods.get("full_flow", ())
        if int(run["summary"]["algorithm_seed"]) == 1
    ]
    if len(full_seed1) != 1:
        raise RuntimeError("ablation comparison requires Full Flow algorithm Seed1")
    comparison = [
        {
            "method": "full_flow",
            "display_name": "Guardian-Flow-DSAC (Full)",
            "actor_objective": "Q + 0.15 P + 2.0 correction + 0.05 anchor",
            "algorithm_seed": 1,
            "online_steps": 300_000,
            "evaluation": _evaluation_at(full_seed1[0], 300_000),
            "run": full_seed1[0],
        }
    ]
    objectives = {
        "flow_no_p": "Q + 2.0 correction + 0.05 anchor",
        "flow_no_correction": "Q + 0.15 P + 0.05 anchor",
    }
    for method in ABLATION_METHODS:
        runs = ablations.get(method, ())
        if len(runs) != 1:
            raise RuntimeError(f"ablation comparison requires one {method} run")
        run = runs[0]
        comparison.append(
            {
                "method": method,
                "display_name": METHOD_NAMES[method],
                "actor_objective": objectives[method],
                "algorithm_seed": 1,
                "online_steps": 300_000,
                "evaluation": run["final"],
                "run": run,
            }
        )

    fields = [
        "Method",
        "Method Slug",
        "Actor Objective",
        "Algorithm Seed",
        "Online Steps",
        *[DISPLAY[name] for name in METRICS],
    ]
    table_rows = []
    for item in comparison:
        evaluation = item["evaluation"]
        table_rows.append(
            {
                "Method": item["display_name"],
                "Method Slug": item["method"],
                "Actor Objective": item["actor_objective"],
                "Algorithm Seed": item["algorithm_seed"],
                "Online Steps": item["online_steps"],
                **{
                    DISPLAY[name]: f"{float(evaluation[name]):.6f}"
                    for name in METRICS
                },
            }
        )
    _write_csv(root / "ablation_table.csv", fields, table_rows)
    markdown = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
        *(
            "| " + " | ".join(str(row[field]) for field in fields) + " |"
            for row in table_rows
        ),
    ]
    (root / "ablation_table.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    curve_rows = []
    for item in comparison:
        run = item["run"]
        for evaluation in run["summary"]["online"]["evaluations"]:
            step = int(evaluation["step"])
            if step > 300_000:
                continue
            curve_rows.append(
                {
                    "row_type": "run",
                    "method": item["method"],
                    "algorithm_seed": 1,
                    "step": step,
                    **{
                        metric: float(evaluation[metric])
                        for metric in METRICS + AUXILIARY_CURVE_METRICS
                    },
                }
            )
    curve_fields = ["row_type", "method", "algorithm_seed", "step"]
    for metric in METRICS + AUXILIARY_CURVE_METRICS:
        curve_fields.extend((metric, metric + "_std"))
    normalized_curves = [
        {field: row.get(field, "") for field in curve_fields} for row in curve_rows
    ]
    _write_csv(root / "ablation_curves.csv", curve_fields, normalized_curves)
    return table_rows, normalized_curves


def _report(
    root,
    methods,
    main_rows,
    audits,
    teacher_path,
    teacher,
    partial,
    experiment_setup,
    ablation_rows=None,
    ablation_audit=None,
    sampling_audit=None,
    run_state_independence=None,
):
    q_runs = len(methods.get("flow_q_only", ()))
    revisions = sorted(
        {
            run["config"].get("source_code", {}).get("git_revision", "unknown")
            for runs in methods.values()
            for run in runs
        }
    )
    sampling_rows = (sampling_audit or {}).get("rows", ())
    if sampling_rows:
        observed_maximum = sampling_audit[
            "maximum_observed_transition_fraction_deviation"
        ]
        observed_latent_imbalance = sampling_audit[
            "maximum_flow_latent_episode_count_imbalance"
        ]
        sampling_report_lines = [
            "- 最终 sampling 审计逐 run 明细：",
            "",
            "| 方法 | 算法 Seed | Online Steps | 最大 transition 比例偏差 | Flow latent episode 最大计数差 | 单模型 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            *[
                "| {method} | {algorithm_seed} | {online_steps} | {deviation:.6f} | {latent} | 是 |".format(
                    method=row["method"],
                    algorithm_seed=row["algorithm_seed"],
                    online_steps=row["online_steps"],
                    deviation=float(
                        row["maximum_transition_fraction_deviation"]
                    ),
                    latent=(
                        "N/A"
                        if row[
                            "maximum_flow_latent_episode_count_imbalance"
                        ]
                        is None
                        else row[
                            "maximum_flow_latent_episode_count_imbalance"
                        ]
                    ),
                )
                for row in sampling_rows
            ],
            "",
            "- 全部正式在线 run 的实际最大 transition 比例偏差为 "
            f"`{float(observed_maximum):.6f}`（允许上限 `0.003`）；"
            "全部 Flow run 的 zero/random episode 实际最大累计计数差为 "
            + (
                "`N/A`（当前没有完成 Flow run）。"
                if observed_latent_imbalance is None
                else f"`{int(observed_latent_imbalance)}`（允许上限 `1`）。"
            ),
        ]
    else:
        sampling_report_lines = [
            "- 当前没有完成的正式在线 run sampling 证据；partial/smoke 不作为论文结论。"
        ]
    if run_state_independence:
        main_state = run_state_independence["main"]
        ablation_state = run_state_independence["ablations"]
        state_report_lines = [
            "- run-state 独立性终审：主矩阵 "
            f"`{len(main_state['rows'])}/{len(main_state['expected_runs'])}` 个 run、"
            f"`{main_state['state_artifact_count']}` 个状态文件；两个消融 "
            f"`{len(ablation_state['rows'])}/{len(ablation_state['expected_runs'])}` 个 run、"
            f"`{ablation_state['state_artifact_count']}` 个状态文件。所有文件均位于所属 "
            "run 目录，均为非符号链接、单硬链接的普通文件，且不同 run 间解析路径和 "
            "device/inode 均不复用。",
            "- 冻结 trainer 序列化契约将模型与优化器写入各 run 的算法 checkpoint，"
            "将在线 replay、correction replay、JAX key 与 NumPy RNG 写入各 run 的 "
            "atomic trainer state；审计不反序列化训练状态，而以冻结 Git revision、"
            "trainer SHA256、resume index SHA256 和磁盘文件身份共同绑定。",
        ]
    else:
        state_report_lines = [
            "- run-state 独立性终审尚未完成；partial/smoke 报告不作为论文结论。"
        ]
    lines = [
        "# Guardian-Flow-DSAC 论文主实验报告",
        "",
        "## 5.1 实验设置（Experimental Setup）",
        "",
        "- 正式场景：Seeds `0,1,2,3,4,5,6,7,8,10`；Seed9 未进入新 benchmark。",
        "- 离线/retention：Seeds `0–4`；novel-online：Seeds `5,6,7,8,10`。",
        "- 离线数据：Seeds `0–4` 每个 `300` 个已接受完整 episode，总计 `1500` 个 episode。",
        "- 算法随机 Seed：`0,1,2`；曲线先对十场景 macro-average，再跨训练 Seed 统计。",
        "- episode horizon：`1200`；最大/目标速度：`10.75 m/s`。",
        "- 正式训练硬件：`NVIDIA GeForce RTX 4080 SUPER`。",
        f"- 训练程序 Git revision：`{', '.join(revisions)}`。",
        (
            f"- 两个独立消融的冻结训练 Git revision：`{ablation_audit.get('expected_ablation_source_revision')}`；训练入口 SHA256：`{ablation_audit.get('expected_ablation_trainer_sha256')}`。"
            if ablation_audit
            else "- 两个独立消融的冻结训练版本将在严格消融审计完成后写入正式报告。"
        ),
        "- 网络结构：Actor/Q/P 均采用两层 "
        f"`{experiment_setup['common_actor_q_p_hidden_sizes']}` MLP；"
        f"输入观测 `{experiment_setup['observation_dim']}` 维、动作 "
        f"`{experiment_setup['action_dim']}` 维。Gaussian Actor 输出高斯策略参数；"
        "Flow teacher 与 one-step Flow Actor 分别为 "
        f"`{experiment_setup['flow_teacher_hidden_sizes']}` / "
        f"`{experiment_setup['one_step_flow_actor_hidden_sizes']}`，冻结 teacher 使用 "
        f"`{experiment_setup['flow_teacher_integration_steps']}` 步积分，在线/部署 "
        f"Actor 每个动作只做 `{experiment_setup['one_step_flow_actor_forward_passes']}` 次前向。",
        "- 环境后端：官方 `metadrive.envs.safe_metadrive_env.SafeMetaDriveEnv`，通过 `relax.env.drive.metasafe.make_metasafe_cbf_mpc_env` 固定场景并接入 Guardian/CBF/NMPC；二维动作语义为 `target_speed_lateral_v1`。",
        "- Guardian/CBF/NMPC：所有方法共享同一完整环境与控制器配置。NMPC horizon `2.0 s`，tracking/velocity/road 权重为 `6/30/100`，road buffer `1.5`，最大转向变化 `0.1`；CBF gamma `3.0`，SQP/ADMM 步数 `12/120`，障碍上限 `32`。机器可读的完整参数写入 `final_results.json` 的 `experiment_setup.guardian_environment`。",
        "- Flow 离线初始化：`80k` Flow matching + `40k` same-latent distillation；offline Q-guided Actor 更新固定为 `0`。",
        "- Full Flow Actor 目标：`Q + 0.15 P + 2.0 correction + 0.05 frozen-teacher anchor`；correction 的速度/横向权重为 `2/1`，zero-latent 权重为 `1`。Flow-Q-only 仅保留 `Q + anchor`。",
        "- P 使用完整 episode 反向构造：`G_t^P = c_t + (1-c_t) * 0.97 * G_{t+1}^P`。只有 terminal/truncated 的完整 episode 才一次性写入 replay；未完整 episode 丢弃，不生成或写入 P target。该语义由冻结训练提交与入口脚本 SHA256 共同绑定，并写入 `final_results.json`。",
        "- 在线 latent：每个场景按 episode 分层平衡 `50% z=0 + 50% z~N(0,I)`，两类 episode 累计计数差不超过 `1`；novel/retention transition 目标分别为每场景 `0.16/0.04`。",
        (
            "- 在线预算：每个正式在线 run 为 `1,000,000` environment steps；batch `512`、buffer `500000`、每步 `1` 次梯度更新、policy delay `2`、offline replay `0.20`、correction subbatch `0.25`，每 `50k` 固定评估与 checkpoint。"
            if not partial
            else "- 当前为 component smoke；主表中的 Online Steps 显示实际 smoke 预算，不作为论文性能结果。"
        ),
        "- 评估不改变训练；仅 NaN/Inf、非有限 loss/参数/优化器或损坏 checkpoint 允许恢复。",
        "- 确定性评估：Flow learner 使用 one-step Actor 的 exact `z=0` 分支；Gaussian learner 使用 squash 后的策略分布均值。每个 checkpoint 对每个场景/模式恰好运行一个 fresh-process episode，Actor 每个动作一次前向。两层 evaluator 的冻结 SHA256 写入 `final_results.json` 的 `experiment_setup.deterministic_evaluation_protocol`。",
        "- 训练期每个动作必须经 Guardian 认证；未认证 proposal 只触发安全 reset/终止并计数，不伪造 Task-Q transition，也不触发 performance rollback。",
        "- 评估期采用同一安全语义：未认证 proposal 不执行、不计为环境 transition；该 episode 记为失败并保留在十场景固定分母中，Ordinary/Hybrid 安全终止率单独写入曲线 CSV 与逐场景表。",
        "- 方法边界：Guardian/MPC、Flow-BC、Gaussian、Flow-Q-only 与 Full Flow 均作为独立方法行评估/统计。协议要求的共享离线初始化完成后，各方法的模型、优化器和 replay 独立；聚合只合并指标，不合并策略。不构造额外复合或集成方法，不做跨方法投票、加权或 checkpoint 拼接。Ordinary/Hybrid 仅是同一 checkpoint 的两种执行评估模式，不是复合模型。",
        "",
        "## 5.2 主结果（Main Results）",
        "",
        (root / "main_table.md").read_text(encoding="utf-8").strip(),
        "",
        "Guardian/MPC 行的 Return、Success、Safe Success、Cost 和 Collision 来自其唯一适用的 certified Guardian execution；它没有 learner Ordinary 模式，Hybrid intervention 记为 N/A。",
        "",
        "Return 和 cost 均以全部十个场景为分母；std 为算法随机 Seed 的总体标准差（ddof=0）。",
        "",
        "## 5.3 学习效率（Learning Efficiency）",
        "",
        "- `learning_curves.csv` 同时保存逐算法 Seed 行与跨算法 Seed 的 mean/std 行；每个点已经先对固定十场景做 macro-average。",
        "- `learning_curves.pdf` 与 `learning_curves.png` 包含 Ordinary return、success、safe-success、cost、collision 以及 Hybrid intervention 六个面板；cost 面板显示 `log(1+C)`，原始 cost 保留在 CSV。",
        "- 所有在线曲线使用 `0,50k,...,1M` 的共同横轴；评估只生成观测证据，不触发门禁、性能回滚或训练状态选择。",
        "",
        "## 5.4 Actor 表示（Actor Representation）",
        "",
        "Gaussian Guardian-DSAC 与 Guardian-Flow-DSAC 是直接 Actor 表示对比：二者共享数据、Q/P 初始化、Guardian、在线预算和评估协议，但分别使用 Gaussian Actor 与 one-step Flow Actor。主表与六条曲线中的两行均来自各自独立 checkpoint；该比较不使用投票、权重融合或策略拼接。数值差异是描述性结果，统计离散度仅来自算法随机 Seed `0,1,2`。",
        "",
        "## 5.5 最小目标消融（Ablation）",
        "",
        (
            "两个核心消融均为算法 Seed1、300k 在线步数，并与 Full Flow Seed1 的 300k 检查点进行同 Seed、同预算、同冻结 Flow-BC 初始化的描述性比较。`flow_no_p` 只移除 Actor 的 `0.15 P` 项并保留 `2.0 correction`；`flow_no_correction` 只移除 `2.0 correction` 项并保留 `0.15 P`。三行都是独立单模型、独立优化器与独立在线 replay，不做融合、投票、加权或 checkpoint 拼接。结果见 `ablation_table.md` 与 `ablation_curves.pdf`。"
            if ablation_rows
            else "正式核心消融尚未聚合；partial/smoke 报告不用于论文结论。"
        ),
        "Flow-Q-only 仍作为 `Q + frozen-teacher anchor` 的联合补充消融保留。主表中 Flow-Q-only 的 `±0` 只是单 run 的定义结果，不代表不确定性为零；它同时移除了 P 与 correction，不能替代上面的两个单因素消融，不能把差异分别归因给其中一项。",
        *(("", (root / "ablation_table.md").read_text(encoding="utf-8").strip()) if ablation_rows else ()),
        "",
        "## 完整性审计",
        "",
        f"- 已审计学习 run：`{len(audits)}`；所有已列 run 均通过场景、预算、硬件、checkpoint SHA256 与无性能回滚检查。",
        *state_report_lines,
        *sampling_report_lines,
        f"- Guardian/MPC teacher 原始结果：`{teacher_path}`。",
        f"- Flow-Q-only 完成算法随机 Seed 数：`{q_runs}`。",
        (
            f"- 独立消融审计：`{len(ablation_audit.get('checkpoint_evaluations', ()))}/14` 个检查点评测，完成状态 `{ablation_audit.get('suite_complete')}`，复合/集成标记 `{ablation_audit.get('composite_or_ensemble')}`。"
            if ablation_audit
            else "- 独立消融审计：当前未纳入 partial/smoke 聚合。"
        ),
        "- Guardian/MPC 平均 Guardian 认证耗时："
        f"`{float(teacher['teacher_guardian_certification_latency_ms_mean']):.3f} ms/step`；"
        "完整 certified-control 耗时："
        f"`{float(teacher['teacher_certified_control_latency_ms_mean']):.3f} ms/step`。",
        f"- 当前聚合是否为允许的 partial/smoke：`{partial}`。",
        "- 论文主值始终使用 1M final；中间 checkpoint 不替代 final。",
        "",
        "## 产物",
        "",
        "- `main_table.csv` / `main_table.md`：论文主表。",
        "- `per_seed_table.csv`：逐场景分析。",
        "- `learning_curves.csv` / `learning_curves.pdf`：训练曲线数据与图。",
        "- `ablation_table.csv` / `ablation_table.md`：两个独立单因素消融与 Full Flow Seed1@300k 对照。",
        "- `ablation_curves.csv` / `ablation_curves.pdf`：独立消融 0–300k 曲线数据与图。",
        "- `final_results.json`：机器可读汇总和审计证据。",
        "- `SHA256SUMS`：由 suite 在绘图完成后生成的论文产物校验清单。",
    ]
    (root / "EXPERIMENT_REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.results_root.expanduser().resolve()
    teacher_path, teacher_payload, teacher = _teacher_row(root)
    methods = {}
    audits = []
    required = {
        "flow_bc": ALGORITHM_SEEDS,
        "gaussian": ALGORITHM_SEEDS,
        "full_flow": ALGORITHM_SEEDS,
        "flow_q_only": (0,),
    }
    for method, seeds in required.items():
        runs = []
        for seed in seeds:
            path = root / method / f"train_seed_{seed}"
            if not path.is_dir():
                if args.allow_partial:
                    continue
                raise FileNotFoundError(f"required paper run is missing: {path}")
            run = _audit_run(path, method, seed)
            runs.append(run)
            audits.append(
                {
                    key: value
                    for key, value in run.items()
                    if key not in ("summary", "config", "final")
                }
            )
        if runs:
            methods[method] = runs
    if not args.allow_partial and any(method not in methods for method in required):
        raise RuntimeError("paper aggregation is missing a required method")
    matrix_bindings = _audit_matrix(methods, teacher_payload)
    experiment_setup = _experiment_setup(methods, matrix_bindings)
    main_rows, statistics = _main_table(root, methods, teacher)
    per_seed = _per_seed_table(root, methods)
    curves = _learning_curves(root, methods)
    ablations = {}
    ablation_audit = None
    ablation_audit_binding = None
    ablation_rows = []
    ablation_curves = []
    ablation_matrix = None
    run_state_independence = None
    if args.require_ablations:
        main_audit_path = root / "checkpoint_audit.json"
        if not main_audit_path.is_file():
            raise FileNotFoundError("strict main checkpoint audit is missing")
        main_audit = _read(main_audit_path)
        if (
            main_audit.get("format")
            != "guardian_flow_paper_checkpoint_audit_v1"
            or main_audit.get("audit_passed") is not True
            or main_audit.get("suite_complete") is not True
            or main_audit.get("require_complete_matrix") is not True
            or main_audit.get("composite_or_ensemble") is not False
            or main_audit.get("expected_source_revision")
            != GUARDIAN_DEPENDENCE_PROTOCOL["training_source_revision"]
            or len(main_audit.get("checkpoint_evaluations", ())) != 150
        ):
            raise RuntimeError("strict main checkpoint audit is incomplete")
        main_expected_runs = {
            f"{method}/train_seed_{seed}"
            for method, seeds in required.items()
            for seed in seeds
        }
        main_state = _validated_run_state_evidence(
            main_audit,
            expected_runs=main_expected_runs,
            expected_scope="main",
            expected_revision=GUARDIAN_DEPENDENCE_PROTOCOL[
                "training_source_revision"
            ],
            expected_trainer_sha256=GUARDIAN_DEPENDENCE_PROTOCOL[
                "trainer_entrypoint_sha256"
            ],
        )
        audit_path = root / "ablation_checkpoint_audit.json"
        if not audit_path.is_file():
            raise FileNotFoundError("independent ablation checkpoint audit is missing")
        ablation_audit = _read(audit_path)
        if (
            ablation_audit.get("format")
            != "guardian_flow_paper_ablation_checkpoint_audit_v1"
            or ablation_audit.get("audit_passed") is not True
            or ablation_audit.get("suite_complete") is not True
            or ablation_audit.get("require_complete") is not True
            or ablation_audit.get("composite_or_ensemble") is not False
            or ablation_audit.get("expected_ablation_source_revision")
            != ABLATION_TRAINING_REVISION
            or ablation_audit.get("expected_ablation_trainer_sha256")
            != ABLATION_TRAINER_SHA256
            or len(ablation_audit.get("checkpoint_evaluations", ())) != 14
        ):
            raise RuntimeError("independent ablation checkpoint audit is incomplete")
        ablation_expected_runs = {
            f"{method}/train_seed_{seed}"
            for method, seeds in {
                "flow_no_p": (1,),
                "flow_no_correction": (1,),
            }.items()
            for seed in seeds
        }
        ablation_state = _validated_run_state_evidence(
            ablation_audit,
            expected_runs=ablation_expected_runs,
            expected_scope="ablations",
            expected_revision=ABLATION_TRAINING_REVISION,
            expected_trainer_sha256=ABLATION_TRAINER_SHA256,
        )
        run_state_independence = {
            "format": "guardian_flow_paper_final_run_state_independence_v1",
            "main": main_state,
            "ablations": ablation_state,
            "shared_offline_initialization_only": True,
            "post_initialization_model_optimizer_replay_rng_independent": True,
            "shared_state_artifact_count": 0,
            "composite_or_ensemble": False,
            "audit_passed": True,
        }
        ablation_audit_binding = {
            "path": str(audit_path.resolve()),
            "sha256": _sha256(audit_path),
            "format": ablation_audit["format"],
            "checkpoint_evaluations": 14,
            "suite_complete": True,
            "audit_passed": True,
            "composite_or_ensemble": False,
        }
        for method in ABLATION_METHODS:
            run = _audit_run(
                root / "ablations" / method / "train_seed_1", method, 1
            )
            ablations[method] = [run]
        ablation_matrix = _audit_ablation_matrix(root, methods, ablations)
        ablation_rows, ablation_curves = _ablation_outputs(
            root, methods, ablations
        )
    sampling_audit = _sampling_audit(methods, ablations)
    payload = {
        "format": "guardian_flow_dsac_paper_final_results_v1",
        "paper_scene_seeds": list(PAPER_SCENE_SEEDS),
        "offline_seeds": list(OFFLINE_SEEDS),
        "novel_seeds": list(NOVEL_SEEDS),
        "algorithm_seeds": list(ALGORITHM_SEEDS),
        "teacher": teacher_payload,
        "methods": {
            method: {
                "runs": [
                    {
                        "algorithm_seed": run["summary"]["algorithm_seed"],
                        "path": run["path"],
                        "final": run["final"],
                    }
                    for run in runs
                ],
                "statistics": statistics.get(method),
            }
            for method, runs in methods.items()
        },
        "main_table": main_rows,
        "per_seed_table": per_seed,
        "learning_curve_rows": len(curves),
        "audits": audits,
        "matrix_bindings": matrix_bindings,
        "experiment_setup": experiment_setup,
        "sampling_audit": sampling_audit,
        "method_scope": METHOD_SCOPE,
        "run_state_independence": run_state_independence,
        "ablations": {
            method: {
                "runs": [
                    {
                        "algorithm_seed": 1,
                        "path": run["path"],
                        "final": run["final"],
                        "actor_objective_terms": run["config"][
                            "actor_objective_terms"
                        ],
                        "flow_config": run["config"]["flow_config"],
                    }
                    for run in runs
                ]
            }
            for method, runs in ablations.items()
        },
        "ablation_table": ablation_rows,
        "ablation_curve_rows": len(ablation_curves),
        "ablation_matrix_bindings": ablation_matrix,
        "ablation_checkpoint_audit": ablation_audit_binding,
        "allow_partial": bool(args.allow_partial),
        "require_ablations": bool(args.require_ablations),
    }
    _write_json(root / "final_results.json", payload)
    _report(
        root,
        methods,
        main_rows,
        audits,
        teacher_path,
        teacher,
        args.allow_partial,
        experiment_setup,
        ablation_rows,
        ablation_audit,
        sampling_audit,
        run_state_independence,
    )
    print(
        json.dumps({"results_root": str(root), "audited_runs": len(audits)}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
