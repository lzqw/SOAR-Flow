#!/usr/bin/env python3
"""Train one method/algorithm-Seed in the unified paper protocol.

Formal runs are bound to one NVIDIA GeForce RTX 4080 SUPER, the fixed paper
scene set, 1200-step episodes, 10.75 m/s, and evaluation/checkpoint intervals
of 50k.  Evaluation is observational: it never changes the training state.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.offline.guardian_dsac_multiseed_dataset import (  # noqa: E402
    guardian_dsac_to_flow_long_run_experience,
)
from relax.utils.action_semantics import (  # noqa: E402
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)
from relax.utils.flow_paper import (  # noqa: E402
    ALGORITHM_SEEDS,
    CORRECTION_SEED_WEIGHTS,
    NOVEL_SEEDS,
    OFFLINE_SEEDS,
    PAPER_SCENE_SEEDS,
    TRANSITION_TARGET_WEIGHTS,
    PaperTransitionDeficitScheduler,
    UniformEpisodeLatentPolicy,
)
from scripts.train_guardian_calql_metasafe import (  # noqa: E402
    _audit_low_level_cbf_authority,
    _concatenate_batches,
)
from scripts.train_guardian_dsac_metasafe import (  # noqa: E402
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
    _load_multiseed_dataset,
)


ABLATION_METHODS = ("flow_no_p", "flow_no_correction")
SAME_SEED_ABLATION_METHODS = ("full_flow", *ABLATION_METHODS)
MAIN_EXPERIMENT_PROFILE = "main"
SAME_SEED_ABLATION_PROFILE = "same_seed_ablation"
SAME_SEED_ABLATION_1M_PROFILE = "same_seed_ablation_1m"
PROXY_ANNEALING_1M_PROFILE = "proxy_annealing_1m"
EXPERIMENT_PROFILES = (
    MAIN_EXPERIMENT_PROFILE,
    SAME_SEED_ABLATION_PROFILE,
    SAME_SEED_ABLATION_1M_PROFILE,
    PROXY_ANNEALING_1M_PROFILE,
)
SAME_SEED_ABLATION_EXTENSION_FROM_STEP = 300_000
SAME_SEED_ABLATION_EXTENSION_TARGET_STEP = 1_000_000
SAME_SEED_ABLATION_EXTENSION_FORMAT = "guardian_flow_same_seed_ablation_1m_extension_v1"
PROXY_ANNEALING_EXTENSION_FORMAT = "guardian_flow_proxy_annealing_1m_extension_v1"
FIXED_PROXY_WEIGHT_SCHEDULE = "fixed"
PROXY_WEIGHT_SCHEDULES = (
    FIXED_PROXY_WEIGHT_SCHEDULE,
    "p_hard_off_300k",
    "p_decay_300k_500k",
    "p_decay_300k_700k",
    "p_decay_floor_003",
)
METHODS = (
    "flow_bc",
    "gaussian",
    "flow_q_only",
    "full_flow",
    *ABLATION_METHODS,
)
FROZEN_MAIN_TRAINING_REVISION = "99383a489fe687509fecc6b06d457ebbafecf7ef"
FROZEN_MAIN_TRAINER_SHA256 = (
    "19e8c8a0b5e46568e06847844dc8203febf4af3aed5476e289c2fbd5ee663746"
)
EVALUATION_UNCERTIFIED_PROPOSAL_HANDLING = "safe_termination_without_execution_v1"
METHOD_NAMES = {
    "flow_bc": "Flow-BC Offline-only",
    "gaussian": "Gaussian Guardian-DSAC",
    "flow_q_only": "Flow-Q-only",
    "full_flow": "Guardian-Flow-DSAC",
    "flow_no_p": "Guardian-Flow-DSAC without P actor term",
    "flow_no_correction": "Guardian-Flow-DSAC without correction actor term",
}
PAPER_SOURCE_FILES = (
    "relax/algorithm/guardian_dsac_flow_paper.py",
    "relax/buffer/seed_indexed.py",
    "relax/utils/flow_paper.py",
    "scripts/aggregate_guardian_flow_paper_results.py",
    "scripts/dispatch_guardian_flow_paper_ablations.sh",
    "scripts/evaluate_guardian_dsac_isolated.py",
    "scripts/evaluate_guardian_flow_dsac_paper.py",
    "scripts/launch_guardian_flow_proxy_annealing_seed1.sh",
    "scripts/plot_guardian_flow_paper_curves.py",
    "scripts/prepare_guardian_flow_proxy_annealing_1m.py",
    "scripts/prepare_guardian_flow_same_seed_ablation_1m.py",
    "scripts/run_guardian_flow_dsac_paper_suite.py",
    "scripts/supervise_guardian_flow_paper_run.sh",
    "scripts/train_guardian_flow_dsac_paper.py",
)


def _sampling_protocol_payload(*, flow: bool) -> dict[str, Any]:
    return {
        "transition_target_weights": {
            str(seed): value for seed, value in TRANSITION_TARGET_WEIGHTS.items()
        },
        "correction_seed_sampling_weights": (
            {str(seed): value for seed, value in CORRECTION_SEED_WEIGHTS.items()}
            if flow
            else None
        ),
        "zero_latent_probability_by_seed": (
            {str(seed): 0.5 for seed in PAPER_SCENE_SEEDS} if flow else None
        ),
    }


def _actor_objective_terms(method: str) -> list[str]:
    """Return the exact single-model actor objective used by one method."""

    if method == "gaussian":
        return ["gaussian_dsac", "guardian_p"]
    terms = ["task_q"]
    if method not in ("flow_q_only", "flow_no_p"):
        terms.append("guardian_p")
    if method not in ("flow_q_only", "flow_no_correction"):
        terms.append("correction")
    terms.append("teacher_anchor")
    return terms


def _uses_same_seed_ablation_budget(args: argparse.Namespace) -> bool:
    """Identify methods bound to the controlled Seed1 comparison protocol."""

    return args.method in ABLATION_METHODS or args.experiment_profile in (
        SAME_SEED_ABLATION_PROFILE,
        SAME_SEED_ABLATION_1M_PROFILE,
        PROXY_ANNEALING_1M_PROFILE,
    )


def _formal_online_steps(args: argparse.Namespace) -> int:
    if args.experiment_profile in (
        SAME_SEED_ABLATION_1M_PROFILE,
        PROXY_ANNEALING_1M_PROFILE,
    ):
        return SAME_SEED_ABLATION_EXTENSION_TARGET_STEP
    if (
        args.method in ABLATION_METHODS
        or args.experiment_profile == SAME_SEED_ABLATION_PROFILE
    ):
        return SAME_SEED_ABLATION_EXTENSION_FROM_STEP
    return 1_000_000


def _proxy_weight_at_step(
    schedule: str,
    step: int,
    initial_weight: float = 0.15,
) -> float:
    """Return the Actor-only proxy coefficient at one global online step."""

    if schedule not in PROXY_WEIGHT_SCHEDULES:
        raise ValueError(f"unknown proxy-weight schedule: {schedule}")
    step = int(step)
    initial_weight = float(initial_weight)
    if step < 0 or not np.isfinite(initial_weight) or initial_weight < 0.0:
        raise ValueError("proxy schedule step/initial weight is invalid")
    if schedule == FIXED_PROXY_WEIGHT_SCHEDULE:
        return initial_weight
    if schedule == "p_hard_off_300k":
        return initial_weight if step < 300_000 else 0.0
    end_step = 700_000 if schedule == "p_decay_300k_700k" else 500_000
    floor = 0.03 if schedule == "p_decay_floor_003" else 0.0
    if step <= 300_000:
        return initial_weight
    if step >= end_step:
        return floor
    fraction = (step - 300_000) / float(end_step - 300_000)
    return float(initial_weight + fraction * (floor - initial_weight))


def _proxy_schedule_payload(schedule: str, initial_weight: float) -> dict[str, Any]:
    endpoints = {
        "fixed": (None, None, float(initial_weight)),
        "p_hard_off_300k": (300_000, 300_000, 0.0),
        "p_decay_300k_500k": (300_000, 500_000, 0.0),
        "p_decay_300k_700k": (300_000, 700_000, 0.0),
        "p_decay_floor_003": (300_000, 500_000, 0.03),
    }
    start, end, floor = endpoints[schedule]
    return {
        "name": schedule,
        "actor_loss_only": True,
        "initial_weight": float(initial_weight),
        "anneal_start_step": start,
        "anneal_end_step": end,
        "final_weight": floor,
        "p_network_training_unchanged": True,
    }


def _content_addressed_resume(args: argparse.Namespace) -> bool:
    return bool(args.extend_ablation_from_300k or args.extend_proxy_annealing_from_300k)


def _source_identity() -> dict[str, Any]:
    """Bind every run to an auditable repository revision and source bytes."""

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=str(REPOSITORY_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    revision = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    source_paths = {name: REPOSITORY_ROOT / name for name in PAPER_SOURCE_FILES}
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "git_status_returncode": int(status.returncode),
        "git_status_porcelain": status.stdout.splitlines(),
        "clean": status.returncode == 0 and not status.stdout.strip(),
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items() if path.is_file()
        },
        "missing_source_files": missing,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--shared-offline-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--flow-offline-checkpoint",
        type=Path,
        default=None,
        help="Frozen 80k+40k paper Flow checkpoint for online Flow methods.",
    )
    parser.add_argument(
        "--phase", choices=("offline", "online", "full"), default="full"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/guardian_flow_paper")
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-run", type=Path, default=None)
    parser.add_argument(
        "--extend-ablation-from-300k",
        action="store_true",
        help=(
            "Continue a separately copied, complete 300k Seed1 ablation atomic "
            "state to the fixed 1M budget."
        ),
    )
    parser.add_argument(
        "--extend-proxy-annealing-from-300k",
        action="store_true",
        help=(
            "Continue a separately copied Full Flow 300k atomic state with an "
            "Actor-only proxy-weight schedule to the fixed 1M budget."
        ),
    )
    parser.add_argument(
        "--extension-parent-run",
        type=Path,
        default=None,
        help="Immutable 300k parent run copied into --resume-run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--experiment-profile",
        choices=EXPERIMENT_PROFILES,
        default=MAIN_EXPERIMENT_PROFILE,
        help=(
            "Budget/reporting profile. same_seed_ablation preserves the 300k "
            "comparison; same_seed_ablation_1m is its explicit atomic extension."
        ),
    )

    parser.add_argument("--flow-hidden-sizes", default="256,256")
    parser.add_argument("--flow-integration-steps", type=int, default=8)
    parser.add_argument("--flow-bc-gradient-steps", type=int, default=80_000)
    parser.add_argument("--flow-distill-gradient-steps", type=int, default=40_000)
    parser.add_argument("--flow-learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-actor-tau", type=float, default=0.005)
    parser.add_argument("--q-actor-weight", type=float, default=1.0)
    parser.add_argument("--proxy-actor-weight", type=float, default=0.15)
    parser.add_argument(
        "--proxy-weight-schedule",
        choices=PROXY_WEIGHT_SCHEDULES,
        default=FIXED_PROXY_WEIGHT_SCHEDULE,
    )
    parser.add_argument("--correction-weight", type=float, default=2.0)
    parser.add_argument("--correction-fraction", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--correction-speed-weight", type=float, default=2.0)
    parser.add_argument("--correction-lateral-weight", type=float, default=1.0)
    parser.add_argument("--zero-latent-correction-weight", type=float, default=1.0)
    parser.add_argument("--actor-learning-rate-initial", type=float, default=1e-4)
    parser.add_argument("--actor-learning-rate-final", type=float, default=1e-5)
    parser.add_argument("--actor-learning-rate-decay-steps", type=int, default=500_000)
    parser.add_argument("--critic-learning-rate-initial", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate-final", type=float, default=3e-5)
    parser.add_argument(
        "--critic-learning-rate-decay-steps", type=int, default=1_000_000
    )
    parser.add_argument("--proxy-learning-rate-initial", type=float, default=3e-4)
    parser.add_argument("--proxy-learning-rate-final", type=float, default=3e-5)
    parser.add_argument(
        "--proxy-learning-rate-decay-steps", type=int, default=1_000_000
    )

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--online-steps", type=int, default=1_000_000)
    parser.add_argument("--online-buffer-size", type=int, default=500_000)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--offline-replay-ratio", type=float, default=0.20)
    parser.add_argument("--evaluation-every", type=int, default=50_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--support-calibration-rows-per-seed", type=int, default=8192)
    parser.add_argument(
        "--require-gpu", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-rtx4080-super",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--component-smoke", action="store_true")

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

    if args.seed not in ALGORITHM_SEEDS:
        parser.error("--seed must be one of the predeclared algorithm Seeds 0,1,2")
    if (
        args.experiment_profile == SAME_SEED_ABLATION_PROFILE
        and args.method != "full_flow"
    ):
        parser.error(
            "--experiment-profile same_seed_ablation is the full_flow reference"
        )
    if args.experiment_profile == SAME_SEED_ABLATION_PROFILE and args.phase != "online":
        parser.error("same_seed_ablation is an online-only comparison profile")
    if args.experiment_profile == SAME_SEED_ABLATION_1M_PROFILE:
        if args.extend_proxy_annealing_from_300k:
            parser.error("proxy and ablation extension modes are mutually exclusive")
        if args.method not in SAME_SEED_ABLATION_METHODS:
            parser.error(
                "same_seed_ablation_1m only supports Full Flow and its two "
                "single-factor ablations"
            )
        if args.seed != 1:
            parser.error("same_seed_ablation_1m fixes algorithm Seed to 1")
        if args.phase != "online":
            parser.error("same_seed_ablation_1m is an online-only profile")
        if not args.extend_ablation_from_300k:
            parser.error("same_seed_ablation_1m requires --extend-ablation-from-300k")
        if args.resume_run is None or args.extension_parent_run is None:
            parser.error(
                "the 300k extension requires --resume-run and --extension-parent-run"
            )
        if (
            args.resume_run.expanduser().resolve()
            == args.extension_parent_run.expanduser().resolve()
        ):
            parser.error("the 1M extension must not overwrite its 300k parent run")
    elif args.experiment_profile == PROXY_ANNEALING_1M_PROFILE:
        if args.method != "full_flow":
            parser.error("proxy_annealing_1m only supports Full Flow")
        if args.seed not in (0, 1):
            parser.error("proxy_annealing_1m supports validation Seeds 0 and 1")
        if args.phase != "online":
            parser.error("proxy_annealing_1m is an online-only profile")
        if not args.extend_proxy_annealing_from_300k:
            parser.error(
                "proxy_annealing_1m requires --extend-proxy-annealing-from-300k"
            )
        if args.extend_ablation_from_300k:
            parser.error("proxy and ablation extension modes are mutually exclusive")
        if args.resume_run is None or args.extension_parent_run is None:
            parser.error(
                "the proxy extension requires --resume-run and --extension-parent-run"
            )
        if (
            args.resume_run.expanduser().resolve()
            == args.extension_parent_run.expanduser().resolve()
        ):
            parser.error("the proxy extension must not overwrite its 300k parent")
        if args.proxy_weight_schedule == FIXED_PROXY_WEIGHT_SCHEDULE:
            parser.error("proxy_annealing_1m requires a non-fixed proxy schedule")
        if not np.isclose(args.proxy_actor_weight, 0.15):
            parser.error("proxy annealing must start from lambda_P=0.15")
        if not any(np.isclose(args.correction_weight, value) for value in (1, 2, 3)):
            parser.error("proxy annealing correction weight must be 1.0, 2.0, or 3.0")
        if not np.isclose(args.anchor_weight, 0.05):
            parser.error("proxy annealing fixes the anchor weight at 0.05")
    elif (
        args.extend_ablation_from_300k
        or args.extend_proxy_annealing_from_300k
        or args.extension_parent_run is not None
    ):
        parser.error("extension arguments require an explicit 1M extension profile")
    if (
        args.experiment_profile != PROXY_ANNEALING_1M_PROFILE
        and args.proxy_weight_schedule != FIXED_PROXY_WEIGHT_SCHEDULE
    ):
        parser.error("non-fixed proxy schedules require proxy_annealing_1m")
    if args.method == "flow_bc" and args.phase == "online":
        parser.error("Flow-BC has no online phase")
    if args.method == "gaussian" and args.phase == "offline":
        parser.error("Gaussian reuses the shared offline checkpoint")
    if (
        args.method in ("flow_q_only", "full_flow", *ABLATION_METHODS)
        and args.phase == "online"
        and args.flow_offline_checkpoint is None
    ):
        parser.error("online Flow training requires --flow-offline-checkpoint")
    if args.method in ABLATION_METHODS and args.flow_offline_checkpoint is None:
        parser.error(
            "minimal no-P/no-correction ablations require the shared frozen "
            "Flow-BC checkpoint"
        )
    for name in (
        "flow_integration_steps",
        "batch_size",
        "online_buffer_size",
        "learning_starts",
        "gradient_steps",
        "policy_delay",
        "evaluation_every",
        "checkpoint_every",
        "log_every",
        "support_calibration_rows_per_seed",
        "actor_learning_rate_decay_steps",
        "critic_learning_rate_decay_steps",
        "proxy_learning_rate_decay_steps",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "flow_bc_gradient_steps",
        "flow_distill_gradient_steps",
        "online_steps",
    ):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_episode_steps != 1200:
        parser.error("paper protocol fixes the episode horizon at 1200")
    if not np.isclose(args.offline_replay_ratio, 0.20):
        parser.error("paper protocol fixes offline replay at 0.20")
    if not np.isclose(args.correction_fraction, 0.25):
        parser.error("paper protocol fixes the correction subbatch at 0.25")
    if args.checkpoint_every != args.evaluation_every:
        parser.error("paper checkpoints and evaluations must share one interval")
    if args.method in ABLATION_METHODS and args.seed != 1:
        parser.error("minimal no-P/no-correction ablations fix algorithm Seed to 1")
    if _content_addressed_resume(args) and args.component_smoke:
        parser.error("the atomic 300k-to-1M extension cannot be a component smoke")
    if not args.component_smoke:
        expected_online_steps = _formal_online_steps(args)
        formal = {
            "batch_size": 512,
            "online_steps": expected_online_steps,
            "online_buffer_size": 500_000,
            "gradient_steps": 1,
            "policy_delay": 2,
            "evaluation_every": 50_000,
            "checkpoint_every": 50_000,
            "flow_bc_gradient_steps": 80_000,
            "flow_distill_gradient_steps": 40_000,
        }
        mismatches = {
            name: (getattr(args, name), expected)
            for name, expected in formal.items()
            if getattr(args, name) != expected
        }
        if mismatches:
            parser.error(f"formal paper budget mismatch: {mismatches}")
    else:
        args.flow_bc_gradient_steps = min(args.flow_bc_gradient_steps, 2)
        args.flow_distill_gradient_steps = min(args.flow_distill_gradient_steps, 2)
        args.online_steps = min(args.online_steps, 200)
        args.batch_size = min(args.batch_size, 64)
        args.online_buffer_size = min(args.online_buffer_size, 5_000)
        args.learning_starts = min(args.learning_starts, 32)
        args.evaluation_every = min(args.evaluation_every, 200)
        args.checkpoint_every = min(args.checkpoint_every, 200)
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
    path = _run_dir(checkpoint) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("method") != "Guardian-DSAC V2.3.1 multi-Seed":
        raise ValueError("shared checkpoint is not the five-Seed V2.3.1 source")
    if tuple(int(seed) for seed in config.get("offline_seeds", ())) != OFFLINE_SEEDS:
        raise ValueError("shared checkpoint offline Seeds are not Seed0--4")
    if int(config["environment"]["max_episode_steps"]) != 1200:
        raise ValueError("shared checkpoint does not use the 1200-step horizon")
    return path, config


def _runtime_args(args, source_config):
    environment = dict(source_config["environment"])
    return SimpleNamespace(
        **{
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
        }
    )


def _hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("hidden sizes must be positive integers")
    return sizes


def _predeclare_paper_scene_signatures(args, dataset) -> dict[str, str]:
    from relax.env.drive.metasafe import make_fixed_metasafe_env

    signatures = dict(dataset.scene_signatures)
    for seed in PAPER_SCENE_SEEDS:
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
            raise RuntimeError(f"Seed{seed} differs from its offline dataset")
        signatures[str(seed)] = signature
    return {str(seed): signatures[str(seed)] for seed in PAPER_SCENE_SEEDS}


def _create_rebound_base(args, source_config, dataset, scene_signatures):
    import jax
    import jax.numpy as jnp
    import optax

    from relax.algorithm.guardian_dsac_v2_3 import (
        GuardianDSACV23,
        GuardianDSACV23Config,
    )
    from relax.network.safe_sac_jax import create_guardian_dsac_net

    hidden_sizes = tuple(int(value) for value in source_config["hidden_sizes"])
    network, params = create_guardian_dsac_net(
        jax.random.PRNGKey(int(source_config["seed"])),
        critic_obs_dim=int(source_config["observation_dim"]),
        actor_obs_dim=int(source_config["actor_observation_dim"]),
        act_dim=int(source_config["action_dim"]),
        hidden_sizes=hidden_sizes,
        init_alpha=float(source_config["init_alpha"]),
        sigma_min=float(source_config["algorithm_config"]["sigma_min"]),
    )
    base = GuardianDSACV23(
        network,
        params,
        config=GuardianDSACV23Config(**source_config["algorithm_config"]),
        action_semantics_version=source_config["action_semantics_version"],
        scene_seeds=source_config["scene_seeds"],
        scene_signatures=source_config["scene_signatures"],
        dataset_sha256=dataset.sha256,
    )
    base.load(str(args.shared_offline_checkpoint.expanduser().resolve()))
    if not base.offline_reference_ready or base.dataset_sha256 != dataset.sha256:
        raise RuntimeError("shared Q/P checkpoint is not the bound five-Seed state")

    # State tensors are scene-independent.  Rebind metadata only after the
    # strict historical checkpoint load, replacing Seed9 with predeclared 10.
    base.scene_seeds = PAPER_SCENE_SEEDS
    base.scene_signatures = dict(scene_signatures)
    base.config = replace(
        base.config,
        proximal_weight_initial=0.0,
        proximal_weight_min=0.0,
    )
    clipped = lambda rate: optax.chain(
        optax.clip_by_global_norm(base.config.gradient_clip_norm),
        optax.adam(rate),
    )
    base.critic_optim = clipped(base.config.critic_learning_rate)
    base.actor_optim = clipped(base.config.actor_learning_rate)
    base.proxy_optim = clipped(base.config.proxy_learning_rate)
    base.log_alpha_optim = optax.adam(base.config.alpha_learning_rate)
    current = base.state.params
    base.state = base.state._replace(
        opt_state=base.state.opt_state._replace(
            q1=base.critic_optim.init(current.q1),
            q2=base.critic_optim.init(current.q2),
            proxy=base.proxy_optim.init(current.proxy),
            policy=base.actor_optim.init(current.policy),
            log_alpha=base.log_alpha_optim.init(current.log_alpha),
        ),
        proximal_step=jnp.asarray(0, dtype=jnp.int32),
    )
    base.hybrid_safe_snapshot = jax.tree_util.tree_map(
        lambda value: jnp.array(value, copy=True), base.state
    )
    base.online_reference_step = 0
    base._safe_online_reference_step = 0
    base._consecutive_large_critic_losses = 0
    base._build_updates()
    return base, hidden_sizes


def _flow_config(args, *, method: str):
    from relax.algorithm.guardian_dsac_flow_paper import (
        GuardianDSACFlowPaperConfig,
    )

    return GuardianDSACFlowPaperConfig(
        learning_rate=args.flow_learning_rate,
        target_actor_tau=args.target_actor_tau,
        q_actor_weight=args.q_actor_weight,
        proxy_actor_weight=(
            0.0 if method in ("flow_q_only", "flow_no_p") else args.proxy_actor_weight
        ),
        offline_distill_weight_initial=1.0,
        offline_distill_weight_final=1.0,
        offline_distill_decay_steps=1,
        correction_weight=(
            0.0
            if method in ("flow_q_only", "flow_no_correction")
            else (
                2.0
                if args.experiment_profile == PROXY_ANNEALING_1M_PROFILE
                else args.correction_weight
            )
        ),
        correction_speed_weight=args.correction_speed_weight,
        correction_lateral_weight=args.correction_lateral_weight,
        zero_latent_correction_weight=args.zero_latent_correction_weight,
        anchor_weight=args.anchor_weight,
        anchor_seeds=OFFLINE_SEEDS,
        actor_learning_rate_initial=args.actor_learning_rate_initial,
        actor_learning_rate_final=args.actor_learning_rate_final,
        actor_learning_rate_decay_steps=args.actor_learning_rate_decay_steps,
        critic_learning_rate_initial=args.critic_learning_rate_initial,
        critic_learning_rate_final=args.critic_learning_rate_final,
        critic_learning_rate_decay_steps=args.critic_learning_rate_decay_steps,
        proxy_learning_rate_initial=args.proxy_learning_rate_initial,
        proxy_learning_rate_final=args.proxy_learning_rate_final,
        proxy_learning_rate_decay_steps=args.proxy_learning_rate_decay_steps,
    )


def _create_flow(args, base, dataset, key, *, config):
    from relax.algorithm.guardian_dsac_flow_paper import GuardianDSACFlowPaper
    from relax.network.flow_policy import create_one_step_flow_policy

    policy, params = create_one_step_flow_policy(
        key,
        actor_obs_dim=dataset.obs_dim,
        act_dim=dataset.action_dim,
        hidden_sizes=_hidden_sizes(args.flow_hidden_sizes),
        integration_steps=args.flow_integration_steps,
    )
    return GuardianDSACFlowPaper(base, policy, params, config=config)


def _multiseed_validation(algorithm, dataset) -> dict[str, Any]:
    target = algorithm.base if hasattr(algorithm, "base") else algorithm
    per_seed = {
        str(seed): _offline_validation(target, dataset.by_seed[seed])
        for seed in OFFLINE_SEEDS
    }
    names = sorted({name for row in per_seed.values() for name in row})
    return {
        "mean_across_seeds": {
            name: float(np.mean([row[name] for row in per_seed.values()]))
            for name in names
        },
        "per_seed": per_seed,
    }


def _flow_diagnostics(algorithm, dataset, *, rows_per_seed: int):
    import jax
    import jax.numpy as jnp

    batch = dataset.flow_validation_reference_batch(
        rows_per_seed=rows_per_seed, to_jax=True
    )
    count = int(np.asarray(batch.reward).shape[0])
    key_1, key_2 = jax.random.split(jax.random.PRNGKey(31_415))
    noise_1 = jax.random.normal(key_1, (count, dataset.action_dim), jnp.float32)
    noise_2 = jax.random.normal(key_2, (count, dataset.action_dim), jnp.float32)
    teacher = algorithm.flow_policy.bc_action(
        algorithm.state.params.bc_flow, batch.obs, noise_1
    )
    student = algorithm.flow_policy.get_action(
        algorithm.state.params.one_step, batch.obs, noise_1
    )
    error = np.asarray(student - teacher)
    return {
        "rows": count,
        "same_latent_mse": float(np.mean(np.sum(np.square(error), axis=-1))),
        "same_latent_mse_speed": float(np.mean(np.square(error[..., 0]))),
        "same_latent_mse_lateral": float(np.mean(np.square(error[..., 1]))),
        "latent_action_diversity": float(
            np.asarray(
                algorithm.flow_policy.latent_action_diversity(
                    algorithm.state.params.one_step,
                    batch.obs,
                    noise_1,
                    noise_2,
                )
            )
        ),
    }


def _offline_flow_train(algorithm, dataset, args, *, key, output_dir, metrics_path):
    import jax

    q1_before = algorithm.base.state.params.q1
    proxy_before = algorithm.base.state.params.proxy
    teacher_before = algorithm.state.params.bc_flow
    actor_before = algorithm.state.params.one_step
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
                f"[paper] flow matching {step + 1}/{args.flow_bc_gradient_steps}",
                flush=True,
            )
    teacher_trained = algorithm.state.params.bc_flow
    for step in range(args.flow_distill_gradient_steps):
        batch = dataset.sample_flow(args.batch_size, split="train", to_jax=True)
        key, update_key = jax.random.split(key)
        latest = algorithm.distill_one_step(update_key, batch)
        if (
            step + 1
        ) % args.log_every == 0 or step + 1 == args.flow_distill_gradient_steps:
            _append_jsonl(
                metrics_path,
                {"phase": "same_latent_distillation", "step": step + 1, **latest},
            )
            print(
                f"[paper] distillation {step + 1}/{args.flow_distill_gradient_steps}",
                flush=True,
            )
    diagnostics = _flow_diagnostics(
        algorithm,
        dataset,
        rows_per_seed=args.support_calibration_rows_per_seed,
    )
    algorithm.freeze_bc_flow()
    if _tree_changed(q1_before, algorithm.base.state.params.q1):
        raise RuntimeError("offline Flow training changed the shared Task Q")
    if _tree_changed(proxy_before, algorithm.base.state.params.proxy):
        raise RuntimeError("offline Flow training changed the shared P network")
    if not _tree_changed(teacher_before, teacher_trained):
        raise RuntimeError("80k Flow matching did not update the teacher")
    return key, {
        "flow_matching_steps": int(args.flow_bc_gradient_steps),
        "same_latent_distillation_steps": int(args.flow_distill_gradient_steps),
        "offline_q_guided_actor_steps": 0,
        "teacher_updated": True,
        "teacher_frozen": bool(algorithm.bc_flow_frozen),
        "actor_updated": _tree_changed(actor_before, algorithm.state.params.one_step),
        "shared_task_q_unchanged": True,
        "shared_p_unchanged": True,
        "distillation": diagnostics,
        "validation": _multiseed_validation(algorithm, dataset),
        "latest_metrics": latest,
    }


def _sample_mixed_batch(dataset, replay, *, batch_size, flow, rng):
    online_count = (
        min(int(round(batch_size * 0.80)), batch_size - 1)
        if len(replay) > 0 and batch_size > 1
        else 0
    )
    offline_count = batch_size - online_count
    batches = []
    if offline_count:
        batches.append(
            dataset.sample_flow_long_run(offline_count, split="train", to_jax=False)
            if flow
            else dataset.sample(offline_count, split="train", to_jax=False)
        )
    if online_count:
        batches.append(
            replay.sample_global_seed_mix(
                online_count,
                balanced_fraction=0.50,
                seed_weights=TRANSITION_TARGET_WEIGHTS,
                to_jax=False,
            )
        )
    return _concatenate_batches(batches, rng)


def _sample_flow_actor_batch(dataset, replay, correction_replay, *, batch_size, rng):
    correction_count = (
        min(int(round(batch_size * 0.25)), batch_size - 1)
        if len(correction_replay) > 0 and batch_size > 1
        else 0
    )
    offline_count = min(int(round(batch_size * 0.20)), batch_size - correction_count)
    online_count = batch_size - offline_count - correction_count
    if len(replay) == 0:
        offline_count += online_count
        online_count = 0
    batches = []
    if offline_count:
        batches.append(
            dataset.sample_flow_long_run(offline_count, split="train", to_jax=False)
        )
    if online_count:
        batches.append(
            replay.sample_global_seed_mix(
                online_count,
                balanced_fraction=0.50,
                seed_weights=TRANSITION_TARGET_WEIGHTS,
                to_jax=False,
            )
        )
    if correction_count:
        batches.append(
            correction_replay.sample_seed_weighted(
                correction_count,
                weights=CORRECTION_SEED_WEIGHTS,
                to_jax=False,
            )
        )
    return _concatenate_batches(batches, rng)


def _validate_paper_evaluation(
    evaluation: Mapping[str, Any],
    *,
    required_modes: tuple[str, ...],
    require_value_network_metrics: bool,
) -> None:
    """Reject incomplete or numerically invalid paper evaluation episodes."""

    per_seed = evaluation.get("per_seed")
    expected_seeds = {str(seed) for seed in PAPER_SCENE_SEEDS}
    if (
        evaluation.get("uncertified_proposal_handling")
        != EVALUATION_UNCERTIFIED_PROPOSAL_HANDLING
        or evaluation.get("uncertified_proposals_executed") is not False
    ):
        raise RuntimeError(
            "paper evaluation does not bind safe termination for uncertified proposals"
        )
    if not isinstance(per_seed, Mapping) or set(per_seed) != expected_seeds:
        raise RuntimeError("paper evaluation does not contain the exact ten scenes")
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
    )
    if require_value_network_metrics:
        numeric_fields += ("q_mean", "q_std", "p_mean")
    failures = []
    for seed in PAPER_SCENE_SEEDS:
        modes = per_seed[str(seed)]
        if not isinstance(modes, Mapping) or set(modes) != set(required_modes):
            failures.append(f"Seed{seed}:mode_set")
            continue
        for mode in required_modes:
            result = modes[mode]
            rows = result.get("rows", ())
            if int(result.get("episodes", -1)) != 1 or len(rows) != 1:
                failures.append(f"Seed{seed}:{mode}:episode_count")
                continue
            row = rows[0]
            if result.get("error") is not None or row.get("error") is not None:
                failures.append(f"Seed{seed}:{mode}:worker_error")
            if row.get("process_isolated") is not True:
                failures.append(f"Seed{seed}:{mode}:process_isolation")
            if int(row.get("seed", -1)) != seed or row.get("mode") != mode:
                failures.append(f"Seed{seed}:{mode}:row_binding")
            try:
                safe_termination = row.get("safe_termination")
                if not isinstance(safe_termination, bool):
                    raise ValueError("safe_termination must be boolean")
                uncertified = int(row["uncertified_action_count"])
                steps = int(row["steps"])
                executed_steps = int(row["executed_environment_steps"])
                action_proposals = int(row["action_proposals"])
                aggregate_safe_count = int(result["safe_termination_count"])
                aggregate_uncertified = int(result["uncertified_action_count"])
                aggregate_executed_steps = int(result["executed_environment_steps"])
                aggregate_action_proposals = int(result["action_proposals"])
            except (KeyError, TypeError, ValueError):
                failures.append(f"Seed{seed}:{mode}:safe_termination_schema")
            else:
                expected_count = int(safe_termination)
                safe_contract = (
                    uncertified == expected_count
                    and row.get("terminated_without_executing_action")
                    is safe_termination
                    and row.get("safe_termination_reason")
                    == ("projection_failed" if safe_termination else None)
                    and 0 <= steps <= 1200
                    and executed_steps == steps
                    and action_proposals == steps + expected_count
                    and aggregate_safe_count == expected_count
                    and aggregate_uncertified == expected_count
                    and aggregate_executed_steps == steps
                    and aggregate_action_proposals == action_proposals
                    and np.isclose(
                        float(result.get("safe_termination_rate", np.nan)),
                        float(expected_count),
                    )
                    and (
                        not safe_termination
                        or (
                            np.isclose(float(result.get("success_rate", np.nan)), 0.0)
                            and float(result.get("actual_intervention_rate", np.nan))
                            + 1e-12
                            >= 1.0 / max(action_proposals, 1)
                        )
                    )
                )
                if not safe_contract:
                    failures.append(f"Seed{seed}:{mode}:safe_termination_binding")
            try:
                values = np.asarray(
                    [float(result[name]) for name in numeric_fields], np.float64
                )
            except (KeyError, TypeError, ValueError):
                failures.append(f"Seed{seed}:{mode}:metric_schema")
            else:
                if not np.all(np.isfinite(values)):
                    failures.append(f"Seed{seed}:{mode}:non_finite_metric")
    if failures:
        raise RuntimeError(
            "paper evaluation integrity failure: " + ", ".join(sorted(failures))
        )


def _paper_evaluation_summary(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    _validate_paper_evaluation(
        evaluation,
        required_modes=("ordinary", "hybrid"),
        require_value_network_metrics=True,
    )
    rows = {}
    returns = []
    costs = []
    successes = []
    safe_successes = []
    collisions = []
    interventions = []
    margins = []
    ordinary_safe_terminations = []
    hybrid_safe_terminations = []
    for seed in PAPER_SCENE_SEEDS:
        modes = evaluation["per_seed"][str(seed)]
        ordinary = modes["ordinary"]
        hybrid = modes["hybrid"]
        success = float(ordinary["success_rate"])
        collision = float(ordinary["collision_rate"])
        road = float(ordinary["road_departure_rate"])
        cost = float(ordinary["total_environment_cost"]) / max(
            int(ordinary["episodes"]), 1
        )
        margin = float(ordinary["min_physical_body_sat_margin"])
        ordinary_safe_termination = float(ordinary["safe_termination_rate"])
        hybrid_safe_termination = float(hybrid["safe_termination_rate"])
        safe_success = float(
            success == 1.0
            and collision == 0.0
            and road == 0.0
            and cost == 0.0
            and margin > 0.0
            and ordinary_safe_termination == 0.0
            and ordinary.get("error") is None
        )
        result = {
            "ordinary_return": float(ordinary["mean_return"]),
            "ordinary_cost": cost,
            "ordinary_success": success,
            "ordinary_safe_success": safe_success,
            "ordinary_collision": collision,
            "ordinary_road_departure": road,
            "ordinary_min_physical_margin": margin,
            "ordinary_safe_termination": ordinary_safe_termination,
            "hybrid_intervention": float(hybrid["actual_intervention_rate"]),
            "hybrid_return": float(hybrid["mean_return"]),
            "hybrid_safe_termination": hybrid_safe_termination,
        }
        rows[str(seed)] = result
        returns.append(result["ordinary_return"])
        costs.append(cost)
        successes.append(success)
        safe_successes.append(safe_success)
        collisions.append(collision)
        interventions.append(result["hybrid_intervention"])
        margins.append(margin)
        ordinary_safe_terminations.append(ordinary_safe_termination)
        hybrid_safe_terminations.append(hybrid_safe_termination)
    return {
        "scene_count": len(PAPER_SCENE_SEEDS),
        "ordinary_mean_return": float(np.mean(returns)),
        "ordinary_success_rate": float(np.mean(successes)),
        "ordinary_safe_success_rate": float(np.mean(safe_successes)),
        "ordinary_mean_cost": float(np.mean(costs)),
        "ordinary_collision_rate": float(np.mean(collisions)),
        "hybrid_intervention_rate": float(np.mean(interventions)),
        "ordinary_safe_termination_rate": float(np.mean(ordinary_safe_terminations)),
        "hybrid_safe_termination_rate": float(np.mean(hybrid_safe_terminations)),
        "ordinary_worst_physical_margin": float(np.min(margins)),
        "per_seed": rows,
        "all_scene_denominator": True,
    }


@contextmanager
def _formal_evaluation_lock():
    """Optionally serialize isolated evaluator batches across independent runs."""

    lock_path = os.environ.get("GUARDIAN_PAPER_EVALUATION_LOCK")
    if not lock_path:
        yield
        return
    path = Path(lock_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _evaluate(
    algorithm,
    *,
    step: int,
    output_dir: Path,
    checkpoint_already_saved: bool = False,
):
    checkpoint = output_dir / f"checkpoint_{step:08d}.pkl"
    if checkpoint_already_saved:
        if not checkpoint.is_file():
            raise FileNotFoundError("pre-saved evaluation checkpoint is missing")
    else:
        algorithm.save(str(checkpoint))
    output = output_dir / f"evaluation_{step:08d}.json"
    with _formal_evaluation_lock():
        evaluation = _run_isolated_evaluation(
            checkpoint=checkpoint,
            output=output,
            seeds=PAPER_SCENE_SEEDS,
            modes=("ordinary", "hybrid"),
            policy="learner",
        )
    summary = _paper_evaluation_summary(evaluation)
    summary.update(
        step=int(step),
        checkpoint=str(checkpoint),
        checkpoint_sha256=_sha256(checkpoint),
        evaluation=str(output),
        evaluation_sha256=_sha256(output),
    )
    _write_json(output_dir / f"evaluation_{step:08d}_summary.json", summary)
    _append_jsonl(output_dir / "learning_curves.jsonl", summary)
    return summary


def _strict_numerical_reasons(algorithm, metrics, batch) -> list[str]:
    import jax

    target = algorithm.base if hasattr(algorithm, "base") else algorithm
    states = (
        (target.state, algorithm.state)
        if hasattr(algorithm, "base")
        else (target.state,)
    )
    reasons = []
    if any(
        not np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(states)
    ):
        reasons.append("non_finite_parameter_or_optimizer")
    numeric = []
    for value in metrics.values():
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if any(not np.isfinite(value) for value in numeric):
        reasons.append("non_finite_loss_or_metric")
    observation = np.asarray(batch.obs[:16])
    action = np.asarray(batch.action[:16])
    mean_1, std_1, mean_2, std_2 = target.task_q_distribution(observation, action)
    if not all(np.all(np.isfinite(value)) for value in (mean_1, std_1, mean_2, std_2)):
        reasons.append("non_finite_q_distribution")
    if np.any(std_1 <= 0.0) or np.any(std_2 <= 0.0):
        reasons.append("non_positive_q_std")
    proxy = target.proxy_value(observation, action)
    if not np.all(np.isfinite(proxy)):
        reasons.append("non_finite_p")
    return sorted(set(reasons))


def _atomic_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream)
    os.replace(temporary, path)


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    os.replace(temporary, path)


def _load_numerical_checkpoint(algorithm, output_dir: Path, *, step: int):
    """Load the latest finite checkpoint, falling back from a corrupt alias."""

    candidates = (
        output_dir / "checkpoint_latest.pkl",
        output_dir / f"checkpoint_{int(step):08d}.pkl",
    )
    failures = []
    for candidate in candidates:
        if not candidate.is_file():
            failures.append(f"missing:{candidate}")
            continue
        try:
            algorithm.load(str(candidate))
            if candidate != candidates[0]:
                repaired_temporary = candidates[0].with_suffix(
                    candidates[0].suffix + ".repair.tmp"
                )
                shutil.copyfile(candidate, repaired_temporary)
                os.replace(repaired_temporary, candidates[0])
            return candidate
        except (
            OSError,
            EOFError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            pickle.UnpicklingError,
        ) as exc:
            failures.append(f"{candidate}:{type(exc).__name__}:{exc}")
    raise RuntimeError(
        "no intact numerical checkpoint is available: " + " | ".join(failures)
    )


def _save_resume(
    algorithm,
    replay,
    correction_replay,
    *,
    output_dir,
    key,
    rng,
    scheduler,
    latent_policy,
    counters,
    statistics,
    evaluations,
    episode_rows,
    content_addressed=False,
):
    step = int(counters["online_environment_steps"])
    index_path = output_dir / "resume_index.json"
    prior_index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.is_file()
        else None
    )
    permanent_checkpoint = output_dir / f"checkpoint_{step:08d}.pkl"
    if not permanent_checkpoint.is_file():
        raise FileNotFoundError(
            "an evaluated permanent checkpoint must precede a resume snapshot"
        )
    checkpoint_sha256 = _sha256(permanent_checkpoint)
    latest = output_dir / "checkpoint_latest.pkl"
    latest_temporary = latest.with_suffix(latest.suffix + ".tmp")
    shutil.copyfile(permanent_checkpoint, latest_temporary)
    os.replace(latest_temporary, latest)
    resume_format = (
        "guardian_flow_paper_resume_v3_atomic"
        if content_addressed
        else "guardian_flow_paper_resume_v2_atomic"
    )
    control = {
        "format": resume_format,
        "step": step,
        "jax_key": np.asarray(key),
        "numpy_rng_state": rng.bit_generator.state,
        "scheduler": {
            "steps": dict(scheduler.steps),
            "episodes": dict(scheduler.episodes),
            "current_seed": int(scheduler.current_seed),
        },
        "latent_counts": latent_policy.counts if latent_policy is not None else None,
        "counters": counters,
        "statistics": statistics,
        "evaluations": evaluations,
        "discard_incomplete_episode_rows_on_resume": int(episode_rows),
        "algorithm_checkpoint": permanent_checkpoint.name,
        "algorithm_checkpoint_sha256": checkpoint_sha256,
        "algorithm_checkpoint_backup": latest.name,
        "algorithm_checkpoint_backup_sha256": _sha256(latest),
        "online_replay_state": replay.state_dict(),
        "correction_replay_state": (
            correction_replay.state_dict() if correction_replay is not None else None
        ),
    }
    if content_addressed:
        pending_trainer = output_dir / f"trainer_state_{step:08d}.pending.pkl"
        _atomic_pickle(pending_trainer, control)
        trainer_sha256 = _sha256(pending_trainer)
        trainer_state = output_dir / f"trainer_state_{step:08d}_{trainer_sha256}.pkl"
        if trainer_state.is_file():
            if _sha256(trainer_state) != trainer_sha256:
                raise RuntimeError(
                    "content-addressed trainer state has conflicting bytes"
                )
            pending_trainer.unlink()
        else:
            os.replace(pending_trainer, trainer_state)
    else:
        trainer_state = output_dir / "trainer_state_latest.pkl"
        _atomic_pickle(trainer_state, control)
        trainer_sha256 = _sha256(trainer_state)
    _atomic_json(
        index_path,
        {
            "format": control["format"],
            "step": control["step"],
            "trainer_state": trainer_state.name,
            "trainer_state_sha256": trainer_sha256,
            "algorithm_checkpoint": permanent_checkpoint.name,
            "algorithm_checkpoint_sha256": control["algorithm_checkpoint_sha256"],
            "algorithm_checkpoint_backup": latest.name,
            "algorithm_checkpoint_backup_sha256": control[
                "algorithm_checkpoint_backup_sha256"
            ],
            "replay_states_embedded_atomically": True,
        },
    )
    if content_addressed and prior_index is not None:
        prior_name = str(prior_index.get("trainer_state", ""))
        prior_path = output_dir / prior_name
        safe_prior = Path(prior_name).name == prior_name and (
            prior_name == "trainer_state_latest.pkl"
            or (prior_name.startswith("trainer_state_") and prior_name.endswith(".pkl"))
        )
        if safe_prior and prior_path != trainer_state and prior_path.is_file():
            prior_path.unlink()


def _load_atomic_resume_algorithm(algorithm, output_dir: Path, control):
    expected = str(control["algorithm_checkpoint_sha256"])
    if str(control.get("algorithm_checkpoint_backup_sha256", "")) != expected:
        raise RuntimeError("atomic resume checkpoint backup SHA256 binding mismatch")
    checkpoint_name = str(control["algorithm_checkpoint"])
    backup_name = str(control["algorithm_checkpoint_backup"])
    if (
        Path(checkpoint_name).name != checkpoint_name
        or Path(backup_name).name != backup_name
    ):
        raise RuntimeError("atomic resume checkpoint path escapes the run directory")
    candidates = (output_dir / checkpoint_name, output_dir / backup_name)
    failures = []
    for candidate in candidates:
        if not candidate.is_file():
            failures.append(f"missing:{candidate}")
            continue
        observed = _sha256(candidate)
        if observed != expected:
            failures.append(f"sha256:{candidate}:{observed}")
            continue
        try:
            algorithm.load(str(candidate))
            if candidate != candidates[0]:
                repaired_temporary = candidates[0].with_suffix(
                    candidates[0].suffix + ".repair.tmp"
                )
                shutil.copyfile(candidate, repaired_temporary)
                os.replace(repaired_temporary, candidates[0])
            return candidate
        except (
            OSError,
            EOFError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            pickle.UnpicklingError,
        ) as exc:
            failures.append(f"{candidate}:{type(exc).__name__}:{exc}")
    raise RuntimeError(
        "no intact algorithm state is available for the atomic resume: "
        + " | ".join(failures)
    )


def _load_atomic_trainer_control(output_dir: Path) -> dict[str, Any]:
    index_path = output_dir / "resume_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    trainer_name = str(index.get("trainer_state", ""))
    trainer_path = output_dir / trainer_name
    if (
        index.get("format")
        not in (
            "guardian_flow_paper_resume_v2_atomic",
            "guardian_flow_paper_resume_v3_atomic",
        )
        or index.get("replay_states_embedded_atomically") is not True
        or Path(trainer_name).name != trainer_name
        or not (
            trainer_name == "trainer_state_latest.pkl"
            or (
                trainer_name.startswith("trainer_state_")
                and trainer_name.endswith(".pkl")
            )
        )
        or not trainer_path.is_file()
    ):
        raise RuntimeError("atomic trainer resume index is incomplete")
    observed_sha = _sha256(trainer_path)
    if observed_sha != str(index.get("trainer_state_sha256", "")):
        raise RuntimeError("atomic trainer-state SHA256 binding mismatch")
    with trainer_path.open("rb") as stream:
        control = pickle.load(stream)
    if (
        control.get("format") != index["format"]
        or int(control.get("step", -1)) != int(index.get("step", -1))
        or control.get("algorithm_checkpoint") != index.get("algorithm_checkpoint")
        or control.get("algorithm_checkpoint_sha256")
        != index.get("algorithm_checkpoint_sha256")
        or control.get("algorithm_checkpoint_backup")
        != index.get("algorithm_checkpoint_backup")
        or control.get("algorithm_checkpoint_backup_sha256")
        != index.get("algorithm_checkpoint_backup_sha256")
    ):
        raise RuntimeError("atomic trainer state does not match its resume index")
    return control


def _online_train(algorithm, dataset, args, *, key, output_dir, resume=False):
    import jax

    from relax.buffer.seed_indexed import SeedIndexedReplay
    from relax.buffer.tree import TreeBuffer

    flow = hasattr(algorithm, "flow_policy")
    factory = (
        TreeBuffer.from_guardian_flow_dsac_long_run_experience
        if flow
        else TreeBuffer.from_guardian_dsac_experience
    )
    replay = SeedIndexedReplay(
        factory(
            dataset.obs_dim,
            dataset.action_dim,
            args.online_buffer_size,
            seed=args.seed + 31_003,
        ),
        seeds=PAPER_SCENE_SEEDS,
        seed=args.seed + 41_003,
    )
    correction_replay = (
        SeedIndexedReplay(
            TreeBuffer.from_guardian_flow_dsac_long_run_experience(
                dataset.obs_dim,
                dataset.action_dim,
                args.online_buffer_size,
                seed=args.seed + 31_004,
            ),
            seeds=PAPER_SCENE_SEEDS,
            seed=args.seed + 41_004,
        )
        if flow
        else None
    )
    scheduler = PaperTransitionDeficitScheduler()
    latent_policy = UniformEpisodeLatentPolicy() if flow else None
    rng = np.random.default_rng(args.seed + 24_202_608)
    counters = {
        "online_environment_steps": 0,
        "online_episodes": 0,
        "task_critic_updates": 0,
        "proxy_updates": 0,
        "actor_updates": 0,
        "numerical_rollbacks": 0,
        "performance_rollbacks": 0,
        "uncertified_action_count": 0,
        "safe_termination_resets": 0,
        "aborted_incomplete_episodes": 0,
        "correction_replay_rows": 0,
        "evaluations": 0,
        "checkpoints": 0,
    }
    statistics = {
        "episodes_by_seed": {seed: 0 for seed in PAPER_SCENE_SEEDS},
        "interventions_by_seed": {seed: 0 for seed in PAPER_SCENE_SEEDS},
        "latent_steps_by_seed": {
            seed: {"zero": 0, "random": 0} for seed in PAPER_SCENE_SEEDS
        },
        "safety_failures_by_seed": {seed: set() for seed in PAPER_SCENE_SEEDS},
    }
    evaluations = []
    # These are the immutable offline baselines, captured before a resume
    # checkpoint is loaded so final update evidence remains meaningful across
    # process restarts.
    initial_q1 = (algorithm.base if flow else algorithm).state.params.q1
    initial_proxy = (algorithm.base if flow else algorithm).state.params.proxy
    initial_actor = (
        algorithm.state.params.one_step if flow else algorithm.state.params.policy
    )
    frozen_teacher = algorithm.state.params.bc_flow if flow else None
    if resume:
        control = _load_atomic_trainer_control(output_dir)
        restored_checkpoint = _load_atomic_resume_algorithm(
            algorithm, output_dir, control
        )
        replay.load_state_dict(control["online_replay_state"])
        if correction_replay is not None:
            if control["correction_replay_state"] is None:
                raise ValueError("Flow resume state has no correction replay")
            correction_replay.load_state_dict(control["correction_replay_state"])
        elif control["correction_replay_state"] is not None:
            raise ValueError("Gaussian resume state unexpectedly has correction replay")
        key = jax.device_put(np.asarray(control["jax_key"], dtype=np.uint32))
        rng.bit_generator.state = control["numpy_rng_state"]
        scheduler.steps = {
            int(k): int(v) for k, v in control["scheduler"]["steps"].items()
        }
        scheduler.episodes = {
            int(k): int(v) for k, v in control["scheduler"]["episodes"].items()
        }
        scheduler.current_seed = int(control["scheduler"]["current_seed"])
        if flow:
            if control["latent_counts"] is None:
                raise ValueError("Flow resume state has no latent protocol counters")
            latent_policy.counts = {
                int(seed): {name: int(value) for name, value in row.items()}
                for seed, row in control["latent_counts"].items()
            }
        counters = {name: int(value) for name, value in control["counters"].items()}
        raw_statistics = control["statistics"]
        statistics = {
            "episodes_by_seed": {
                int(k): int(v) for k, v in raw_statistics["episodes_by_seed"].items()
            },
            "interventions_by_seed": {
                int(k): int(v)
                for k, v in raw_statistics["interventions_by_seed"].items()
            },
            "latent_steps_by_seed": {
                int(k): {name: int(value) for name, value in row.items()}
                for k, row in raw_statistics["latent_steps_by_seed"].items()
            },
            "safety_failures_by_seed": {
                int(k): set(value)
                for k, value in raw_statistics["safety_failures_by_seed"].items()
            },
        }
        evaluations = list(control["evaluations"])
        if _sha256(restored_checkpoint) != control["algorithm_checkpoint_sha256"]:
            raise RuntimeError("resume checkpoint SHA256 binding mismatch")
        if int(control["discard_incomplete_episode_rows_on_resume"]) > 0:
            counters["aborted_incomplete_episodes"] += 1
            scheduler.abort_episode()

    def fresh_environment(current_env, seed):
        if current_env is not None:
            current_env.close()
        env = _make_environment(args, seed=seed, mode="hybrid")
        observation, info = env.reset(seed=seed)
        signature = str(info.get("metasafe_scene_signature", ""))
        if signature != algorithm.scene_signatures[str(seed)]:
            env.close()
            raise RuntimeError(f"Seed{seed} paper scene signature mismatch")
        return env, np.asarray(observation, np.float32).reshape(-1), signature

    current_seed = scheduler.current_seed
    env = None
    observation = None
    scene_signature = None
    latent_mode = None
    if counters["online_environment_steps"] < args.online_steps:
        env, observation, scene_signature = fresh_environment(None, current_seed)
        latent_mode = latent_policy.sample(current_seed, rng) if flow else None
    episode = []
    episode_step = 0
    consecutive_uncertified = 0
    next_eval = (
        (counters["online_environment_steps"] // args.evaluation_every) + 1
    ) * args.evaluation_every
    if not evaluations and counters["online_environment_steps"] == 0:
        if flow:
            algorithm.promote_numerical_checkpoint()
        initial_checkpoint = output_dir / "checkpoint_00000000.pkl"
        algorithm.save(str(initial_checkpoint))
        # Establish a complete step-zero resume generation before the first
        # isolated evaluation.  An interruption during evaluation can then
        # restart safely without manual directory surgery.
        _save_resume(
            algorithm,
            replay,
            correction_replay,
            output_dir=output_dir,
            key=key,
            rng=rng,
            scheduler=scheduler,
            latent_policy=latent_policy,
            counters=counters,
            statistics=statistics,
            evaluations=evaluations,
            episode_rows=0,
            content_addressed=_content_addressed_resume(args),
        )
        evaluations.append(
            _evaluate(
                algorithm,
                step=0,
                output_dir=output_dir,
                checkpoint_already_saved=True,
            )
        )
        counters["evaluations"] += 1
        counters["checkpoints"] += 1
        _save_resume(
            algorithm,
            replay,
            correction_replay,
            output_dir=output_dir,
            key=key,
            rng=rng,
            scheduler=scheduler,
            latent_policy=latent_policy,
            counters=counters,
            statistics=statistics,
            evaluations=evaluations,
            episode_rows=0,
            content_addressed=_content_addressed_resume(args),
        )
    latest = {}
    try:
        while counters["online_environment_steps"] < args.online_steps:
            if flow:
                zero_latent = latent_mode == "zero"
                flow_noise = (
                    np.zeros(dataset.action_dim, np.float32)
                    if zero_latent
                    else rng.normal(0.0, 1.0, dataset.action_dim).astype(np.float32)
                )
                action = algorithm.get_action(observation, flow_noise)
            else:
                zero_latent = False
                flow_noise = None
                key, action_key = jax.random.split(key)
                action = algorithm.get_action(action_key, observation)
            action = np.clip(
                np.asarray(action, np.float32).reshape(dataset.action_dim), -1.0, 1.0
            )
            certificate = env.certify_target_action(action)
            if bool(certificate.get("projection_failed", False)):
                consecutive_uncertified += 1
                counters["uncertified_action_count"] += 1
                counters["safe_termination_resets"] += 1
                counters["aborted_incomplete_episodes"] += int(bool(episode))
                statistics["safety_failures_by_seed"][current_seed].add(
                    "projection_failed"
                )
                episode.clear()
                episode_step = 0
                current_seed = scheduler.abort_episode()
                env, observation, scene_signature = fresh_environment(env, current_seed)
                latent_mode = latent_policy.sample(current_seed, rng) if flow else None
                if consecutive_uncertified >= 20:
                    raise RuntimeError("20 consecutive uncertified paper proposals")
                continue
            consecutive_uncertified = 0
            authority = _audit_low_level_cbf_authority(
                env,
                action,
                certificate,
                trajectory_tolerance=args.cbf_trajectory_tolerance,
                derive_teacher=True,
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
            scheduler.record_step(current_seed)
            if flow:
                statistics["latent_steps_by_seed"][current_seed][latent_mode] += 1
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
            transition = (
                guardian_dsac_to_flow_long_run_experience(
                    base_transition,
                    flow_noise=flow_noise,
                    flow_noise_valid=np.bool_(True),
                    flow_zero_latent=np.bool_(zero_latent),
                )
                if flow
                else base_transition
            )
            episode.append(transition)
            statistics["safety_failures_by_seed"][current_seed].update(
                _online_safety_failures(info)
            )
            statistics["interventions_by_seed"][current_seed] += int(
                bool(transition.intervention)
            )
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
                if flow:
                    for item in finalized:
                        if (
                            bool(item.actual_control_modified)
                            and (
                                bool(item.intervention_hard)
                                or bool(item.intervention_soft)
                            )
                            and bool(item.flow_noise_valid)
                        ):
                            correction_replay.add(item)
                            counters["correction_replay_rows"] += 1
                counters["online_episodes"] += 1
                statistics["episodes_by_seed"][current_seed] += 1
                episode.clear()
                current_seed = scheduler.complete_episode()

            if step >= args.learning_starts:
                for gradient_index in range(args.gradient_steps):
                    batch = _sample_mixed_batch(
                        dataset,
                        replay,
                        batch_size=args.batch_size,
                        flow=flow,
                        rng=rng,
                    )
                    update_actor = bool(
                        (step + gradient_index) % args.policy_delay == 0
                    )
                    key, update_key = jax.random.split(key)
                    if flow:
                        actor_batch = (
                            _sample_flow_actor_batch(
                                dataset,
                                replay,
                                correction_replay,
                                batch_size=args.batch_size,
                                rng=rng,
                            )
                            if update_actor
                            else None
                        )
                        latest = algorithm.online_joint_update(
                            update_key,
                            batch,
                            actor_data=actor_batch,
                            update_actor=update_actor,
                            proxy_actor_weight=_proxy_weight_at_step(
                                args.proxy_weight_schedule,
                                step,
                                algorithm.config.proxy_actor_weight,
                            ),
                            correction_weight=(
                                args.correction_weight
                                if args.experiment_profile == PROXY_ANNEALING_1M_PROFILE
                                else algorithm.config.correction_weight
                            ),
                        )
                    else:
                        latest = algorithm.online_joint_update(
                            update_key, batch, update_actor=update_actor
                        )
                    counters["task_critic_updates"] += 1
                    counters["proxy_updates"] += 1
                    counters["actor_updates"] += int(update_actor)
                    reasons = _strict_numerical_reasons(algorithm, latest, batch)
                    if reasons:
                        counters["numerical_rollbacks"] += 1
                        restored_checkpoint = _load_numerical_checkpoint(
                            algorithm,
                            output_dir,
                            step=int(evaluations[-1]["step"]),
                        )
                        _append_jsonl(
                            output_dir / "metrics.jsonl",
                            {
                                "phase": "numerical_rollback",
                                "step": step,
                                "reasons": reasons,
                                "checkpoint": str(restored_checkpoint),
                            },
                        )
                        break

            if step % args.log_every == 0:
                _append_jsonl(
                    output_dir / "metrics.jsonl",
                    {
                        "phase": "paper_online",
                        "step": step,
                        **counters,
                        "online_replay_size": len(replay),
                        "correction_replay_size": len(correction_replay) if flow else 0,
                        "steps_by_seed": {
                            str(seed): count for seed, count in scheduler.steps.items()
                        },
                        **latest,
                    },
                )
                print(
                    f"[paper:{args.method}:seed{args.seed}] {step}/{args.online_steps}",
                    flush=True,
                )

            if step >= next_eval:
                while next_eval <= step:
                    next_eval += args.evaluation_every
                if flow:
                    algorithm.promote_numerical_checkpoint()
                evaluations.append(
                    _evaluate(algorithm, step=step, output_dir=output_dir)
                )
                counters["evaluations"] += 1
                counters["checkpoints"] += 1
                _save_resume(
                    algorithm,
                    replay,
                    correction_replay,
                    output_dir=output_dir,
                    key=key,
                    rng=rng,
                    scheduler=scheduler,
                    latent_policy=latent_policy,
                    counters=counters,
                    statistics=statistics,
                    evaluations=evaluations,
                    episode_rows=len(episode),
                    content_addressed=_content_addressed_resume(args),
                )

            if done and step < args.online_steps:
                env, observation, scene_signature = fresh_environment(env, current_seed)
                latent_mode = latent_policy.sample(current_seed, rng) if flow else None
                episode_step = 0
    finally:
        if env is not None:
            env.close()

    final_step = counters["online_environment_steps"]
    if not evaluations or evaluations[-1]["step"] != final_step:
        if flow:
            algorithm.promote_numerical_checkpoint()
        evaluations.append(_evaluate(algorithm, step=final_step, output_dir=output_dir))
        counters["evaluations"] += 1
        counters["checkpoints"] += 1
    _save_resume(
        algorithm,
        replay,
        correction_replay,
        output_dir=output_dir,
        key=key,
        rng=rng,
        scheduler=scheduler,
        latent_policy=latent_policy,
        counters=counters,
        statistics=statistics,
        evaluations=evaluations,
        episode_rows=len(episode),
        content_addressed=_content_addressed_resume(args),
    )
    if final_step != args.online_steps:
        raise RuntimeError("paper run did not complete the requested online budget")
    if flow and _tree_changed(frozen_teacher, algorithm.state.params.bc_flow):
        raise RuntimeError("online paper training changed the frozen Flow teacher")
    replay_counts = replay.seed_counts()
    total_replay = max(sum(replay_counts.values()), 1)
    return key, {
        **counters,
        "requested_online_steps": int(args.online_steps),
        "actual_online_steps": int(final_step),
        "completed_full_budget": final_step == args.online_steps,
        "performance_rollbacks": 0,
        "evaluation_changes_training": False,
        **_sampling_protocol_payload(flow=flow),
        "steps_by_seed": {str(k): v for k, v in scheduler.steps.items()},
        "transition_fraction_by_seed": {
            str(k): v / max(final_step, 1) for k, v in scheduler.steps.items()
        },
        "episodes_by_seed": {
            str(k): v for k, v in statistics["episodes_by_seed"].items()
        },
        "latent_steps_by_seed": (
            {str(k): v for k, v in statistics["latent_steps_by_seed"].items()}
            if flow
            else None
        ),
        "latent_episode_counts_by_seed": (
            {str(k): v for k, v in latent_policy.counts.items()} if flow else None
        ),
        "online_replay_size": len(replay),
        "online_replay_fraction_by_seed": {
            str(k): replay_counts[k] / total_replay for k in PAPER_SCENE_SEEDS
        },
        "correction_replay_size": len(correction_replay) if flow else 0,
        "training_safety_failures_by_seed": {
            str(k): sorted(v) for k, v in statistics["safety_failures_by_seed"].items()
        },
        "task_q_updated": _tree_changed(
            initial_q1, (algorithm.base if flow else algorithm).state.params.q1
        ),
        "p_updated": _tree_changed(
            initial_proxy, (algorithm.base if flow else algorithm).state.params.proxy
        ),
        "actor_updated": _tree_changed(
            initial_actor,
            algorithm.state.params.one_step if flow else algorithm.state.params.policy,
        ),
        "frozen_teacher_unchanged": bool(flow),
        "evaluations": evaluations,
        "final_evaluation": evaluations[-1],
    }


_EXTENSION_IMMUTABLE_CONFIG_FIELDS = (
    "format",
    "method",
    "method_slug",
    "seed",
    "algorithm_random_seeds",
    "action_semantics_version",
    "scene_seeds",
    "offline_seeds",
    "online_novel_seeds",
    "scene_signatures",
    "dataset_sha256",
    "manifest_sha256",
    "observation_dim",
    "actor_observation_dim",
    "action_dim",
    "hidden_sizes",
    "flow_hidden_sizes",
    "flow_integration_steps",
    "init_alpha",
    "algorithm_config",
    "flow_config",
    "environment",
    "transition_target_weights",
    "correction_seed_sampling_weights",
    "zero_latent_probability_by_seed",
    "actor_objective_terms",
    "independent_single_model_run",
    "composite_or_ensemble",
    "moving_safe_proximal",
    "safe_trajectory_memory",
    "seed_specific_logic",
    "performance_rollback",
    "conditional_launch",
    "evaluation_changes_training",
    "numerical_disaster_rollback_only",
    "component_smoke",
    "source_config_sha256",
)
_EXTENSION_IMMUTABLE_BUDGET_FIELDS = (
    "flow_matching_steps",
    "same_latent_distillation_steps",
    "offline_q_guided_actor_steps",
    "batch_size",
    "online_buffer_size",
    "gradient_steps",
    "policy_delay",
    "offline_replay_ratio",
    "correction_fraction",
    "evaluation_every",
    "checkpoint_every",
)
_EXTENSION_RUNTIME_SOURCE_FILES = (
    "relax/algorithm/guardian_dsac_flow_paper.py",
    "relax/buffer/seed_indexed.py",
    "relax/utils/flow_paper.py",
    "scripts/evaluate_guardian_dsac_isolated.py",
)


def _json_normalize(value: Any) -> Any:
    """Match the representation written to JSON before comparing configs."""

    return json.loads(json.dumps(value, sort_keys=True))


def _extension_config_failures(
    parent_config: Mapping[str, Any], desired_config: Mapping[str, Any]
) -> list[str]:
    parent_config = _json_normalize(parent_config)
    desired_config = _json_normalize(desired_config)
    failures = [
        field
        for field in _EXTENSION_IMMUTABLE_CONFIG_FIELDS
        if parent_config.get(field) != desired_config.get(field)
    ]
    parent_budgets = parent_config.get("budgets", {})
    desired_budgets = desired_config.get("budgets", {})
    failures.extend(
        f"budgets.{field}"
        for field in _EXTENSION_IMMUTABLE_BUDGET_FIELDS
        if parent_budgets.get(field) != desired_budgets.get(field)
    )
    parent_sources = parent_config.get("source_code", {}).get("source_sha256", {})
    desired_sources = desired_config.get("source_code", {}).get("source_sha256", {})
    failures.extend(
        f"source_code.source_sha256.{name}"
        for name in _EXTENSION_RUNTIME_SOURCE_FILES
        if parent_sources.get(name) != desired_sources.get(name)
    )
    return sorted(set(failures))


def _proxy_extension_config_failures(
    parent_config: Mapping[str, Any], desired_config: Mapping[str, Any]
) -> list[str]:
    """Allow only the declared source/schedule transition after atomic 300k."""

    return [
        failure
        for failure in _extension_config_failures(parent_config, desired_config)
        if not failure.startswith("source_code.source_sha256.")
    ]


def _validate_same_seed_extension(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    previous_config: Mapping[str, Any],
    desired_config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    parent = args.extension_parent_run.expanduser().resolve()
    if output_dir == parent:
        raise ValueError("the 1M extension cannot overwrite its 300k parent")
    preparation_path = output_dir / "extension_preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    parent_config_path = parent / "config.json"
    copied_parent_config_path = output_dir / "parent_config_300k.json"
    parent_summary_path = parent / "summary.json"
    parent_resume_path = parent / "resume_index.json"
    parent_trainer_path = parent / "trainer_state_latest.pkl"
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    copied_parent_config = json.loads(
        copied_parent_config_path.read_text(encoding="utf-8")
    )
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    parent_resume = json.loads(parent_resume_path.read_text(encoding="utf-8"))
    child_resume_path = output_dir / "resume_index.json"
    child_resume = json.loads(child_resume_path.read_text(encoding="utf-8"))
    failures = []
    expected_preparation = {
        "format": SAME_SEED_ABLATION_EXTENSION_FORMAT,
        "method_slug": args.method,
        "algorithm_seed": 1,
        "extended_from_step": SAME_SEED_ABLATION_EXTENSION_FROM_STEP,
        "target_online_steps": SAME_SEED_ABLATION_EXTENSION_TARGET_STEP,
        "parent_run": str(parent),
        "destination_run": str(output_dir),
        "normalized_parent_profile": SAME_SEED_ABLATION_PROFILE,
        "child_profile": SAME_SEED_ABLATION_1M_PROFILE,
        "copied_atomic_resume_state": True,
        "replay_optimizer_rng_scheduler_and_latent_state_preserved": True,
    }
    for field, expected in expected_preparation.items():
        if preparation.get(field) != expected:
            failures.append(f"extension_preparation.{field}")
    if _sha256(parent_config_path) != preparation.get("parent_config_sha256"):
        failures.append("parent_config_sha256")
    if _sha256(copied_parent_config_path) != preparation.get("parent_config_sha256"):
        failures.append("copied_parent_config_sha256")
    if parent_config != copied_parent_config:
        failures.append("copied_parent_config_content")
    if _sha256(parent_summary_path) != preparation.get("parent_summary_sha256"):
        failures.append("parent_summary_sha256")
    if _sha256(parent_resume_path) != preparation.get("parent_resume_index_sha256"):
        failures.append("parent_resume_index_sha256")
    if _sha256(parent_trainer_path) != preparation.get("parent_trainer_state_sha256"):
        failures.append("parent_trainer_state_sha256")
    parent_checkpoint = parent / str(parent_resume.get("algorithm_checkpoint", ""))
    if int(
        parent_resume.get("step", -1)
    ) != SAME_SEED_ABLATION_EXTENSION_FROM_STEP or _sha256(
        parent_checkpoint
    ) != preparation.get("parent_checkpoint_sha256"):
        failures.append("parent_checkpoint_sha256")
    parent_online = parent_summary.get("online") or {}
    parent_actual_steps = int(
        parent_online.get(
            "actual_online_steps",
            parent_online.get("online_environment_steps", -1),
        )
    )
    if (
        int(parent_config.get("budgets", {}).get("online_steps", -1))
        != SAME_SEED_ABLATION_EXTENSION_FROM_STEP
        or parent_actual_steps != SAME_SEED_ABLATION_EXTENSION_FROM_STEP
        or parent_summary.get("completed_full_budget") is not True
    ):
        failures.append("parent_300k_completion")
    if parent_config.get("method_slug") != args.method or int(
        parent_config.get("seed", -1)
    ) != int(args.seed):
        failures.append("parent_method_or_seed")
    observed_parent_profile = parent_config.get("experiment_profile")
    if args.method == "full_flow":
        if observed_parent_profile != SAME_SEED_ABLATION_PROFILE:
            failures.append("parent_experiment_profile")
    elif observed_parent_profile not in (None, SAME_SEED_ABLATION_PROFILE):
        failures.append("parent_experiment_profile")
    failures.extend(_extension_config_failures(parent_config, desired_config))
    parent_offline = json.loads(
        (parent / "offline_summary.json").read_text(encoding="utf-8")
    )
    child_offline = json.loads(
        (output_dir / "offline_summary.json").read_text(encoding="utf-8")
    )
    desired_initialization = desired_config.get("offline_initialization_binding") or {}
    if not (
        parent_offline.get("source_sha256")
        == child_offline.get("source_sha256")
        == desired_initialization.get("checkpoint_sha256")
        == preparation.get("flow_bc_seed1_initialization_sha256")
    ):
        failures.append("flow_bc_seed1_initialization_sha256")
    child_step = int(child_resume.get("step", -1))
    if (
        child_step < SAME_SEED_ABLATION_EXTENSION_FROM_STEP
        or child_step > SAME_SEED_ABLATION_EXTENSION_TARGET_STEP
        or child_step % 50_000 != 0
    ):
        failures.append("child_atomic_resume_step")
    initial_extension = (
        int(previous_config.get("budgets", {}).get("online_steps", -1))
        == SAME_SEED_ABLATION_EXTENSION_FROM_STEP
    )
    if initial_extension:
        if _sha256(output_dir / "config.json") != preparation.get(
            "parent_config_sha256"
        ):
            failures.append("initial_copied_config_sha256")
        if _sha256(child_resume_path) != preparation.get("parent_resume_index_sha256"):
            failures.append("initial_copied_resume_index_sha256")
        if child_resume.get("trainer_state_sha256") != preparation.get(
            "parent_trainer_state_sha256"
        ):
            failures.append("initial_copied_trainer_state_sha256")
        if child_resume.get("algorithm_checkpoint_sha256") != preparation.get(
            "parent_checkpoint_sha256"
        ):
            failures.append("initial_copied_checkpoint_sha256")
    else:
        if (
            previous_config.get("experiment_profile") != SAME_SEED_ABLATION_1M_PROFILE
            or int(previous_config.get("budgets", {}).get("online_steps", -1))
            != SAME_SEED_ABLATION_EXTENSION_TARGET_STEP
            or previous_config.get("parent_run") != str(parent)
            or previous_config.get("parent_checkpoint_sha256")
            != preparation.get("parent_checkpoint_sha256")
            or previous_config.get("source_code", {}).get("git_revision")
            != source_identity.get("git_revision")
        ):
            failures.append("continued_extension_config")
        failures.extend(_extension_config_failures(previous_config, desired_config))
    if failures:
        raise ValueError(
            "300k-to-1M extension protocol mismatch: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "extended_from_step": SAME_SEED_ABLATION_EXTENSION_FROM_STEP,
        "target_online_steps": SAME_SEED_ABLATION_EXTENSION_TARGET_STEP,
        "parent_run": str(parent),
        "parent_checkpoint_sha256": preparation["parent_checkpoint_sha256"],
        "parent_trainer_state_sha256": preparation["parent_trainer_state_sha256"],
        "extension_preparation": str(preparation_path),
        "extension_preparation_sha256": _sha256(preparation_path),
        "parent_experiment_profile_observed": observed_parent_profile,
        "normalized_parent_profile": SAME_SEED_ABLATION_PROFILE,
        "atomic_resume_step_at_extension_start": (
            SAME_SEED_ABLATION_EXTENSION_FROM_STEP
        ),
    }


def _validate_proxy_annealing_extension(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    previous_config: Mapping[str, Any],
    desired_config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a proxy schedule to one independently copied 300k atomic state."""

    parent = args.extension_parent_run.expanduser().resolve()
    if output_dir == parent:
        raise ValueError("the proxy extension cannot overwrite its 300k parent")
    preparation_path = output_dir / "extension_preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    parent_config_path = parent / "config.json"
    parent_summary_path = parent / "summary.json"
    parent_resume_path = parent / "resume_index.json"
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    parent_resume = json.loads(parent_resume_path.read_text(encoding="utf-8"))
    copied_parent_path = output_dir / "parent_config_300k.json"
    copied_parent = json.loads(copied_parent_path.read_text(encoding="utf-8"))
    child_resume_path = output_dir / "resume_index.json"
    child_resume = json.loads(child_resume_path.read_text(encoding="utf-8"))
    parent_trainer_name = str(parent_resume.get("trainer_state", ""))
    parent_trainer_path = parent / parent_trainer_name
    parent_checkpoint = parent / str(parent_resume.get("algorithm_checkpoint", ""))
    desired_schedule = desired_config.get("proxy_weight_schedule") or {}
    failures = []
    expected_preparation = {
        "format": PROXY_ANNEALING_EXTENSION_FORMAT,
        "method_slug": "full_flow",
        "algorithm_seed": int(args.seed),
        "proxy_weight_schedule": args.proxy_weight_schedule,
        "effective_actor_correction_weight": float(args.correction_weight),
        "extended_from_step": SAME_SEED_ABLATION_EXTENSION_FROM_STEP,
        "target_online_steps": SAME_SEED_ABLATION_EXTENSION_TARGET_STEP,
        "parent_run": str(parent),
        "destination_run": str(output_dir),
        "child_profile": PROXY_ANNEALING_1M_PROFILE,
        "copied_atomic_resume_state": True,
        "replay_optimizer_rng_scheduler_and_latent_state_preserved": True,
    }
    for field, expected in expected_preparation.items():
        if preparation.get(field) != expected:
            failures.append(f"extension_preparation.{field}")
    if _sha256(parent_config_path) != preparation.get("parent_config_sha256"):
        failures.append("parent_config_sha256")
    if (
        _sha256(copied_parent_path) != preparation.get("parent_config_sha256")
        or copied_parent != parent_config
    ):
        failures.append("copied_parent_config")
    if _sha256(parent_summary_path) != preparation.get("parent_summary_sha256"):
        failures.append("parent_summary_sha256")
    if _sha256(parent_resume_path) != preparation.get("parent_resume_index_sha256"):
        failures.append("parent_resume_index_sha256")
    if (
        Path(parent_trainer_name).name != parent_trainer_name
        or not parent_trainer_path.is_file()
        or _sha256(parent_trainer_path)
        != preparation.get("parent_trainer_state_sha256")
    ):
        failures.append("parent_trainer_state_sha256")
    if (
        int(parent_resume.get("step", -1)) != SAME_SEED_ABLATION_EXTENSION_FROM_STEP
        or not parent_checkpoint.is_file()
        or _sha256(parent_checkpoint) != preparation.get("parent_checkpoint_sha256")
    ):
        failures.append("parent_checkpoint_sha256")
    parent_online = parent_summary.get("online") or {}
    parent_actual_steps = int(
        parent_online.get(
            "actual_online_steps",
            parent_online.get("online_environment_steps", -1),
        )
    )
    if not (
        parent_config.get("method_slug") == "full_flow"
        and int(parent_config.get("seed", -1)) == int(args.seed)
        and int(parent_config.get("budgets", {}).get("online_steps", -1))
        == SAME_SEED_ABLATION_EXTENSION_FROM_STEP
        and parent_actual_steps == SAME_SEED_ABLATION_EXTENSION_FROM_STEP
        and parent_summary.get("completed_full_budget") is True
    ):
        failures.append("parent_300k_full_flow_completion")
    parent_flow = parent_config.get("flow_config") or {}
    if not (
        np.isclose(float(parent_flow.get("proxy_actor_weight", -1.0)), 0.15)
        and np.isclose(float(parent_flow.get("correction_weight", -1.0)), 2.0)
        and np.isclose(float(parent_flow.get("anchor_weight", -1.0)), 0.05)
    ):
        failures.append("parent_actor_weights")
    failures.extend(_proxy_extension_config_failures(parent_config, desired_config))
    expected_schedule = _proxy_schedule_payload(
        args.proxy_weight_schedule,
        float(parent_flow.get("proxy_actor_weight", 0.15)),
    )
    if desired_schedule != expected_schedule:
        failures.append("proxy_weight_schedule")
    if desired_config.get("effective_actor_correction_weight") != float(
        args.correction_weight
    ):
        failures.append("effective_actor_correction_weight")
    parent_offline_path = parent / "offline_summary.json"
    child_offline_path = output_dir / "offline_summary.json"
    parent_offline = json.loads(parent_offline_path.read_text(encoding="utf-8"))
    child_offline = json.loads(child_offline_path.read_text(encoding="utf-8"))
    if not (
        parent_offline == child_offline
        and parent_offline.get("source_sha256")
        == preparation.get("flow_bc_initialization_sha256")
        and parent_offline.get("shared_qp_checkpoint_sha256")
        == preparation.get("shared_qp_checkpoint_sha256")
    ):
        failures.append("offline_initialization_binding")
    child_step = int(child_resume.get("step", -1))
    if not (
        SAME_SEED_ABLATION_EXTENSION_FROM_STEP
        <= child_step
        <= SAME_SEED_ABLATION_EXTENSION_TARGET_STEP
        and child_step % 50_000 == 0
    ):
        failures.append("child_atomic_resume_step")
    initial_extension = (
        int(previous_config.get("budgets", {}).get("online_steps", -1))
        == SAME_SEED_ABLATION_EXTENSION_FROM_STEP
    )
    if initial_extension:
        child_trainer = output_dir / str(child_resume.get("trainer_state", ""))
        if _sha256(output_dir / "config.json") != preparation.get(
            "parent_config_sha256"
        ):
            failures.append("initial_copied_config_sha256")
        if _sha256(child_resume_path) != preparation.get("parent_resume_index_sha256"):
            failures.append("initial_copied_resume_index_sha256")
        if not child_trainer.is_file() or _sha256(child_trainer) != preparation.get(
            "parent_trainer_state_sha256"
        ):
            failures.append("initial_copied_trainer_state_sha256")
        if child_resume.get("algorithm_checkpoint_sha256") != preparation.get(
            "parent_checkpoint_sha256"
        ):
            failures.append("initial_copied_checkpoint_sha256")
    else:
        if not (
            previous_config.get("experiment_profile") == PROXY_ANNEALING_1M_PROFILE
            and int(previous_config.get("budgets", {}).get("online_steps", -1))
            == SAME_SEED_ABLATION_EXTENSION_TARGET_STEP
            and previous_config.get("parent_run") == str(parent)
            and previous_config.get("proxy_weight_schedule") == expected_schedule
            and previous_config.get("effective_actor_correction_weight")
            == float(args.correction_weight)
            and previous_config.get("source_code", {}).get("git_revision")
            == source_identity.get("git_revision")
        ):
            failures.append("continued_proxy_extension_config")
        failures.extend(_extension_config_failures(previous_config, desired_config))
    if failures:
        raise ValueError(
            "proxy 300k-to-1M extension protocol mismatch: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "extended_from_step": SAME_SEED_ABLATION_EXTENSION_FROM_STEP,
        "target_online_steps": SAME_SEED_ABLATION_EXTENSION_TARGET_STEP,
        "parent_run": str(parent),
        "parent_checkpoint_sha256": preparation["parent_checkpoint_sha256"],
        "parent_trainer_state_sha256": preparation["parent_trainer_state_sha256"],
        "extension_preparation": str(preparation_path),
        "extension_preparation_sha256": _sha256(preparation_path),
        "parent_experiment_profile_observed": parent_config.get("experiment_profile"),
        "atomic_resume_step_at_extension_start": (
            SAME_SEED_ABLATION_EXTENSION_FROM_STEP
        ),
    }


def _gpu_identity() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "rows": [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    source_identity = _source_identity()
    if not args.component_smoke and (
        not source_identity["clean"]
        or source_identity["git_revision"] is None
        or source_identity["missing_source_files"]
    ):
        raise RuntimeError(
            "formal paper runs require a clean, complete Git source revision: "
            f"{source_identity}"
        )
    formal_output_base = (args.resume_run or args.output_root).expanduser().resolve()
    if not args.component_smoke and (
        formal_output_base == REPOSITORY_ROOT
        or REPOSITORY_ROOT in formal_output_base.parents
    ):
        raise RuntimeError(
            "formal results must be outside the Git checkout so source provenance "
            "remains clean"
        )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("GUARDIAN_PAPER_EVAL_WORKERS", "4")
    import jax

    started = time.time()
    backend = jax.default_backend()
    device_kinds = [
        str(getattr(device, "device_kind", "unknown")) for device in jax.devices()
    ]
    gpu_identity = _gpu_identity()
    if args.require_gpu and backend != "gpu":
        raise RuntimeError(f"paper training requires GPU, got {backend}")
    if args.require_rtx4080_super and not any(
        "NVIDIA GeForce RTX 4080 SUPER" in row for row in gpu_identity["rows"]
    ):
        raise RuntimeError(f"formal hardware is not RTX 4080 SUPER: {gpu_identity}")

    source_config_path, source_config = _load_source_config(
        args.shared_offline_checkpoint
    )
    runtime_args = _runtime_args(args, source_config)
    manifest_path, _manifest, dataset = _load_multiseed_dataset(runtime_args)
    if not np.isclose(runtime_args.max_command_speed, 10.75):
        raise RuntimeError("paper protocol fixes calibrated maximum speed at 10.75 m/s")
    scene_signatures = _predeclare_paper_scene_signatures(runtime_args, dataset)
    key = jax.random.PRNGKey(args.seed + 24_000)
    key, flow_key = jax.random.split(key)
    base, base_hidden_sizes = _create_rebound_base(
        runtime_args, source_config, dataset, scene_signatures
    )

    flow = args.method != "gaussian"
    algorithm = base
    flow_hidden_sizes = None
    offline_summary = {
        "loaded_shared_qp": True,
        "shared_qp_checkpoint": str(
            args.shared_offline_checkpoint.expanduser().resolve()
        ),
        "shared_qp_checkpoint_sha256": _sha256(
            args.shared_offline_checkpoint.expanduser().resolve()
        ),
    }
    if flow:
        config = _flow_config(runtime_args, method=args.method)
        algorithm = _create_flow(runtime_args, base, dataset, flow_key, config=config)
        flow_hidden_sizes = list(_hidden_sizes(args.flow_hidden_sizes))

    if args.resume_run is not None:
        output_dir = args.resume_run.expanduser().resolve()
        if not (output_dir / "config.json").is_file():
            raise FileNotFoundError("resume run has no config.json")
        previous_config = json.loads(
            (output_dir / "config.json").read_text(encoding="utf-8")
        )
        if (
            previous_config["method_slug"] != args.method
            or int(previous_config["seed"]) != args.seed
        ):
            raise ValueError("resume run method/algorithm Seed mismatch")
        resume = True
    else:
        run_name = args.run_name or f"{args.method}/train_seed_{args.seed}"
        output_dir = args.output_root.expanduser().resolve() / run_name
        output_dir.mkdir(parents=True, exist_ok=False)
        resume = False

    if resume and (output_dir / "offline_summary.json").is_file():
        offline_summary = json.loads(
            (output_dir / "offline_summary.json").read_text(encoding="utf-8")
        )

    config_payload = {
        "format": "guardian_flow_dsac_paper_config_v1",
        "method": METHOD_NAMES[args.method],
        "method_slug": args.method,
        "experiment_profile": args.experiment_profile,
        "seed": int(args.seed),
        "algorithm_random_seeds": list(ALGORITHM_SEEDS),
        "action_semantics_version": TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        "scene_seeds": list(PAPER_SCENE_SEEDS),
        "offline_seeds": list(OFFLINE_SEEDS),
        "online_novel_seeds": list(NOVEL_SEEDS),
        "scene_signatures": scene_signatures,
        "dataset_sha256": dataset.sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "observation_dim": dataset.obs_dim,
        "actor_observation_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "hidden_sizes": list(base_hidden_sizes),
        "flow_hidden_sizes": flow_hidden_sizes,
        "flow_integration_steps": args.flow_integration_steps if flow else None,
        "init_alpha": source_config["init_alpha"],
        "algorithm_config": asdict(base.config),
        "flow_config": asdict(algorithm.config) if flow else None,
        "proxy_weight_schedule": (
            _proxy_schedule_payload(
                args.proxy_weight_schedule,
                algorithm.config.proxy_actor_weight,
            )
            if flow
            else None
        ),
        "effective_actor_correction_weight": (
            float(args.correction_weight)
            if args.experiment_profile == PROXY_ANNEALING_1M_PROFILE
            else (float(algorithm.config.correction_weight) if flow else None)
        ),
        "environment": _environment_config(runtime_args),
        "budgets": {
            "flow_matching_steps": args.flow_bc_gradient_steps if flow else 0,
            "same_latent_distillation_steps": args.flow_distill_gradient_steps
            if flow
            else 0,
            "offline_q_guided_actor_steps": 0,
            "online_steps": 0 if args.method == "flow_bc" else args.online_steps,
            "batch_size": args.batch_size,
            "online_buffer_size": args.online_buffer_size,
            "gradient_steps": args.gradient_steps,
            "policy_delay": args.policy_delay,
            "offline_replay_ratio": args.offline_replay_ratio,
            "correction_fraction": args.correction_fraction,
            "evaluation_every": args.evaluation_every,
            "checkpoint_every": args.checkpoint_every,
        },
        **_sampling_protocol_payload(flow=flow),
        "actor_objective_terms": _actor_objective_terms(args.method),
        "independent_single_model_run": True,
        "composite_or_ensemble": False,
        "evaluation_lock": os.environ.get("GUARDIAN_PAPER_EVALUATION_LOCK"),
        "evaluation_workers": int(os.environ.get("GUARDIAN_PAPER_EVAL_WORKERS", "4")),
        "offline_initialization_binding": (
            {
                "revision": FROZEN_MAIN_TRAINING_REVISION,
                "trainer_sha256": FROZEN_MAIN_TRAINER_SHA256,
                "method_slug": "flow_bc",
                "algorithm_seed": int(args.seed),
                "checkpoint": str(args.flow_offline_checkpoint.expanduser().resolve()),
                "checkpoint_sha256": _sha256(
                    args.flow_offline_checkpoint.expanduser().resolve()
                ),
            }
            if _uses_same_seed_ablation_budget(args)
            else None
        ),
        "moving_safe_proximal": False,
        "safe_trajectory_memory": False,
        "seed_specific_logic": False,
        "performance_rollback": False,
        "conditional_launch": False,
        "evaluation_changes_training": False,
        "numerical_disaster_rollback_only": True,
        "component_smoke": bool(args.component_smoke),
        "backend": backend,
        "devices": [str(device) for device in jax.devices()],
        "device_kinds": device_kinds,
        "nvidia_smi": gpu_identity,
        "source_config": str(source_config_path),
        "source_config_sha256": _sha256(source_config_path),
        "source_code": source_identity,
    }
    if resume and not args.component_smoke:
        if args.extend_ablation_from_300k:
            extension_metadata = _validate_same_seed_extension(
                args=args,
                output_dir=output_dir,
                previous_config=previous_config,
                desired_config=config_payload,
                source_identity=source_identity,
            )
            config_payload.update(extension_metadata)
            _atomic_json(output_dir / "config.json", config_payload)
        elif args.extend_proxy_annealing_from_300k:
            extension_metadata = _validate_proxy_annealing_extension(
                args=args,
                output_dir=output_dir,
                previous_config=previous_config,
                desired_config=config_payload,
                source_identity=source_identity,
            )
            config_payload.update(extension_metadata)
            _atomic_json(output_dir / "config.json", config_payload)
        else:
            resume_failures = []
            if previous_config.get("source_code", {}).get(
                "git_revision"
            ) != source_identity.get("git_revision"):
                resume_failures.append("source_revision")
            if int(previous_config.get("budgets", {}).get("online_steps", -1)) != int(
                args.online_steps
            ):
                resume_failures.append("online_steps")
            if previous_config.get("actor_objective_terms") != _actor_objective_terms(
                args.method
            ):
                resume_failures.append("actor_objective_terms")
            if (
                previous_config.get("experiment_profile", MAIN_EXPERIMENT_PROFILE)
                != args.experiment_profile
            ):
                resume_failures.append("experiment_profile")
            if previous_config.get("composite_or_ensemble") is not False:
                resume_failures.append("composite_or_ensemble")
            if resume_failures:
                raise ValueError(
                    "resume run protocol mismatch: "
                    + ", ".join(sorted(resume_failures))
                )
    elif not resume:
        _write_json(output_dir / "config.json", config_payload)

    offline_checkpoint = output_dir / "offline_flow_bc.pkl"
    if flow and not resume:
        if args.flow_offline_checkpoint is not None:
            source = args.flow_offline_checkpoint.expanduser().resolve()
            source_run = _run_dir(source)
            source_run_config = json.loads(
                (source_run / "config.json").read_text(encoding="utf-8")
            )
            source_summary_path = source_run / "summary.json"
            if not source_summary_path.is_file():
                raise FileNotFoundError("Flow offline source has no completed summary")
            source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
            expected_offline_steps = 2 if args.component_smoke else 80_000
            expected_distill_steps = 2 if args.component_smoke else 40_000
            source_offline = source_summary.get("offline") or {}
            source_failures = []
            if (
                source_run_config.get("method_slug") != "flow_bc"
                or int(source_run_config.get("seed", -1)) != args.seed
            ):
                source_failures.append("method_or_algorithm_seed")
            if bool(source_run_config.get("component_smoke")) != bool(
                args.component_smoke
            ):
                source_failures.append("smoke_formal_mismatch")
            if source_run_config.get("dataset_sha256") != dataset.sha256:
                source_failures.append("dataset_sha256")
            if (
                int(source_offline.get("flow_matching_steps", -1))
                != expected_offline_steps
            ):
                source_failures.append("flow_matching_steps")
            if (
                int(source_offline.get("same_latent_distillation_steps", -1))
                != expected_distill_steps
            ):
                source_failures.append("same_latent_distillation_steps")
            if int(source_offline.get("offline_q_guided_actor_steps", -1)) != 0:
                source_failures.append("offline_q_guided_actor_steps")
            for flag in (
                "teacher_updated",
                "teacher_frozen",
                "actor_updated",
                "shared_task_q_unchanged",
                "shared_p_unchanged",
            ):
                if source_offline.get(flag) is not True:
                    source_failures.append(flag)
            if Path(
                str(source_summary.get("final_checkpoint", ""))
            ).resolve() != source or source_summary.get(
                "final_checkpoint_sha256"
            ) != _sha256(source):
                source_failures.append("final_checkpoint_sha256")
            if not args.component_smoke:
                expected_source_revision = (
                    FROZEN_MAIN_TRAINING_REVISION
                    if _uses_same_seed_ablation_budget(args)
                    else source_identity["git_revision"]
                )
                source_code = source_run_config.get("source_code", {})
                if source_code.get("git_revision") != expected_source_revision:
                    source_failures.append("source_revision")
                if (
                    _uses_same_seed_ablation_budget(args)
                    and source_code.get("source_sha256", {}).get(
                        "scripts/train_guardian_flow_dsac_paper.py"
                    )
                    != FROZEN_MAIN_TRAINER_SHA256
                ):
                    source_failures.append("source_trainer_sha256")
            if source_failures:
                raise RuntimeError(
                    "Flow offline source protocol mismatch: "
                    + ", ".join(sorted(set(source_failures)))
                )
            # Load with the source objective binding, then switch only online
            # loss coefficients for Flow-Q-only.
            with source.open("rb") as stream:
                payload = pickle.load(stream)
            source_config_flow = _flow_config(runtime_args, method="full_flow")
            if dict(payload.get("flow_config", {})) != asdict(source_config_flow):
                raise ValueError(
                    "Flow offline checkpoint is not the shared paper BC state"
                )
            source_algorithm = _create_flow(
                runtime_args, base, dataset, flow_key, config=source_config_flow
            )
            source_algorithm.load(str(source))
            algorithm = source_algorithm
            desired_flow_config = _flow_config(runtime_args, method=args.method)
            if algorithm.config != desired_flow_config:
                algorithm.config = desired_flow_config
                algorithm._build_long_run_actor_update()
            offline_summary.update(
                loaded=True,
                source=str(source),
                source_sha256=_sha256(source),
                teacher_frozen=bool(algorithm.bc_flow_frozen),
                validation=_multiseed_validation(algorithm, dataset),
            )
        else:
            key, trained = _offline_flow_train(
                algorithm,
                dataset,
                runtime_args,
                key=key,
                output_dir=output_dir,
                metrics_path=output_dir / "metrics.jsonl",
            )
            offline_summary.update(trained, loaded=False)
        algorithm.save(str(offline_checkpoint))
        offline_summary.update(
            checkpoint=str(offline_checkpoint),
            checkpoint_sha256=_sha256(offline_checkpoint),
        )
        _write_json(output_dir / "offline_summary.json", offline_summary)

    if flow and resume:
        if not offline_checkpoint.is_file():
            raise FileNotFoundError(
                "Flow resume run has no offline baseline checkpoint"
            )
        expected_offline_sha = str(offline_summary.get("checkpoint_sha256", ""))
        if _sha256(offline_checkpoint) != expected_offline_sha:
            raise RuntimeError("Flow resume offline baseline SHA256 mismatch")
        algorithm.load(str(offline_checkpoint))

    if args.method == "flow_bc":
        evaluation = _evaluate(algorithm, step=0, output_dir=output_dir)
        final_checkpoint = output_dir / "final.pkl"
        algorithm.save(str(final_checkpoint))
        summary = {
            "format": "guardian_flow_dsac_paper_run_summary_v1",
            "method": METHOD_NAMES[args.method],
            "method_slug": args.method,
            "algorithm_seed": args.seed,
            "offline": offline_summary,
            "online": None,
            "final_evaluation": evaluation,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": _sha256(final_checkpoint),
            "wall_time_seconds": time.time() - started,
            "output_dir": str(output_dir),
        }
        _write_json(output_dir / "summary.json", summary)
        return 0

    if flow and not resume and not algorithm.online_schedule_prepared:
        offline_summary["online_preparation"] = algorithm.prepare_online_training()
        algorithm.save(str(offline_checkpoint))
        offline_summary.update(checkpoint_sha256=_sha256(offline_checkpoint))
        _write_json(output_dir / "offline_summary.json", offline_summary)
    elif flow and resume and not algorithm.online_schedule_prepared:
        raise RuntimeError("Flow resume offline baseline is not online-prepared")

    key, online_summary = _online_train(
        algorithm,
        dataset,
        runtime_args,
        key=key,
        output_dir=output_dir,
        resume=resume,
    )
    _write_json(output_dir / "online_summary.json", online_summary)
    final_checkpoint = output_dir / "final.pkl"
    algorithm.save(str(final_checkpoint))
    summary = {
        "format": "guardian_flow_dsac_paper_run_summary_v1",
        "method": METHOD_NAMES[args.method],
        "method_slug": args.method,
        "algorithm_seed": args.seed,
        "offline": offline_summary,
        "online": online_summary,
        "final_evaluation": online_summary["final_evaluation"],
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": _sha256(final_checkpoint),
        "completed_full_budget": bool(online_summary["completed_full_budget"]),
        "wall_time_seconds": time.time() - started,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
