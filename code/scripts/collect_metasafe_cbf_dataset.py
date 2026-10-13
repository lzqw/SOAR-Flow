#!/usr/bin/env python3
"""Collect successful fixed-seed MetaSafe episodes with parallel hard CBF--MPC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# Running this file as ``python scripts/...`` otherwise places only the
# ``scripts`` directory at sys.path[0].  Spawned collector processes must
# resolve the in-tree ``relax`` package, including newly added environment
# modules, rather than an older site-packages installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.utils.experience import (  # noqa: E402
    INTERVENTION_DIAGNOSTIC,
    INTERVENTION_HARD,
    INTERVENTION_NONE,
    INTERVENTION_SOFT,
    certificate_has_physical_violation,
)
from relax.offline.metasafe_guardian_dataset import (  # noqa: E402
    discounted_guardian_dependence_targets,
)
from relax.utils.action_semantics import (  # noqa: E402
    FIXED_SPEED_ACTION_SEMANTICS,
    SPEED_COUPLED_ACTION_SEMANTICS,
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)


BALANCED_SPEED_SAMPLING_PROFILE = "balanced_v23"
FAST_SPEED_SAMPLING_PROFILE = "fast_v231"
V23_MAX_COMMAND_SPEED = 4.8
# Recalibrated for the fixed 1,200-step V2.3.1 contract after the 4.8 m/s
# ceiling was shown to be physically unable to complete the Seed0 route.
FAST_V231_MAX_COMMAND_SPEED = 10.75
SPEED_SAMPLING_PROFILES: dict[str, dict[str, tuple[float, ...]]] = {
    BALANCED_SPEED_SAMPLING_PROFILE: {
        "open": (0.05, 0.10, 0.35, 0.50),
        "near": (0.25, 0.50, 0.25, 0.00),
        "critical": (0.70, 0.30, 0.00, 0.00),
    },
    FAST_SPEED_SAMPLING_PROFILE: {
        "open": (0.00, 0.03, 0.25, 0.72),
        "near": (0.10, 0.25, 0.40, 0.25),
        "critical": (0.55, 0.45, 0.00, 0.00),
    },
}
FAST_V231_LATERAL_NOISE_SCALES = (1.0, 1.0, 0.75, 0.50)
FAST_V231_WITHIN_BIN_DISTRIBUTIONS = {
    "open": "beta_3_1_low_discrepancy_golden_ratio_v1",
    "near": "beta_3_1_low_discrepancy_golden_ratio_v1",
    "critical": "uniform",
}


class FastV231SpeedBinScheduler:
    """Bound categorical and within-bin drift for the fast profile."""

    def __init__(self) -> None:
        self._credits: dict[str, np.ndarray] = {}
        self._within_bin_phases: dict[str, float] = {}

    def within_bin_fraction(
        self,
        speed_context: str,
        sampled_uniform: float,
    ) -> float:
        """Return a high-biased low-discrepancy fraction for open/near road."""

        sampled_uniform = float(sampled_uniform)
        if not 0.0 <= sampled_uniform <= 1.0:
            raise ValueError("within-bin uniform sample must be in [0, 1]")
        if speed_context not in ("open", "near"):
            return sampled_uniform
        if speed_context in self._within_bin_phases:
            phase = (
                self._within_bin_phases[speed_context]
                + 0.6180339887498949
            ) % 1.0
        else:
            # The random initial phase retains per-episode diversity, while
            # irrational rotation prevents seed-dependent runs of low draws.
            phase = sampled_uniform
        self._within_bin_phases[speed_context] = phase
        return float(np.cbrt(phase))

    def choose(
        self,
        speed_context: str,
        probabilities: tuple[float, ...],
        sampled_bin: int,
        within_bin: float,
    ) -> int:
        probabilities_array = np.asarray(probabilities, dtype=np.float64)
        if probabilities_array.shape != (4,):
            raise ValueError("speed-bin scheduler requires four probabilities")
        if speed_context not in ("open", "near"):
            return int(sampled_bin)
        if speed_context not in self._credits:
            phase = (int(sampled_bin) + float(within_bin)) / 4.0
            initial = 0.20 * np.mod(
                phase + 0.6180339887498949 * np.arange(4), 1.0
            )
            initial[probabilities_array <= 0.0] = -1e9
            self._credits[speed_context] = initial
        credit = self._credits[speed_context]
        positive = probabilities_array > 0.0
        credit[positive] += probabilities_array[positive]
        chosen_bin = int(np.argmax(credit))
        credit[chosen_bin] -= 1.0
        return chosen_bin


@dataclass(frozen=True)
class CollectionConfig:
    output_dir: str
    behavior_policy: str = "protected_nominal"
    episodes: int = 300
    num_envs: int = 4
    scene_seed: int = 0
    collector_seed: int = 20260717
    max_episode_steps: int = 1200
    minimum_attempt_success_rate: float = 0.99
    max_attempt_multiplier: float = 1.25
    require_zero_cost: bool = True
    nominal_longitudinal: float = 1.0
    nominal_lateral: float = 0.0
    longitudinal_exploration_std: float = 0.02
    lateral_exploration_std: float = 0.04
    exploration_hold_steps: int = 10
    speed_sampling_profile: str = BALANCED_SPEED_SAMPLING_PROFILE
    minimum_high_speed_bin_fraction: float = 0.0
    maximum_high_speed_bin_fraction: float = 0.70
    minimum_mid_high_bin_fraction: float = 0.0
    minimum_low_speed_bins_fraction: float = 0.0
    minimum_longitudinal_action: float = 0.75
    maximum_lateral_action: float = 0.20
    target_velocity: float = V23_MAX_COMMAND_SPEED
    action_semantics_version: str = FIXED_SPEED_ACTION_SEMANTICS
    min_command_speed: float = 1.0
    max_command_speed: float = V23_MAX_COMMAND_SPEED
    lookahead: float = 10.0
    lateral_scale: float = 3.5
    horizon: float = 2.0
    mpc_tracking_weight: float = 6.0
    mpc_velocity_weight: float = 30.0
    mpc_road_weight: float = 100.0
    mpc_road_buffer: float = 1.5
    mpc_max_steer_rate: float = 0.1
    clearance: float = 0.10
    cbf_gamma: float = 3.0
    projection_reserve: float = 0.10
    projection_kinematic_reserve: float = 0.02
    sqp_steps: int = 24
    admm_steps: int = 240
    max_obstacles: int = 32
    return_discount: float = 0.99
    proxy_discount: float = 0.97
    soft_pair_weight: float = 0.25
    post_reset_warmup_steps: int = 1
    state_aligned_authority_audit: bool = True
    authority_query_hard_margin: float = 50.0
    authority_trajectory_tolerance: float = 0.08
    inverse_mpc_teacher: bool = True
    proactive_dagger_teacher: bool = True
    ordinary_recovery_use_action_history: bool = True
    ordinary_recovery_bypass_sign: float | None = None
    ordinary_recovery_prefer_inverse: bool = False
    dagger_pre_steps: int = 40
    dagger_weight_decay: float = 0.98
    recreate_environment_each_attempt: bool = True
    combine_shards: bool = True
    overwrite: bool = False


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/metasafe_cbf/fixed_seed0_300ep"),
    )
    parser.add_argument(
        "--behavior-policy",
        choices=("protected_nominal", "ordinary_recovery"),
        default="protected_nominal",
        help=(
            "execute the protected nominal Guardian, or collect a coherent "
            "ordinary-MPC recovery teacher trajectory with no CBF authority"
        ),
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--scene-seed", type=int, default=0)
    parser.add_argument("--collector-seed", type=int, default=20260717)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--minimum-attempt-success-rate", type=float, default=0.99)
    parser.add_argument("--max-attempt-multiplier", type=float, default=1.25)
    parser.add_argument(
        "--require-zero-cost",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--nominal-longitudinal", type=float, default=1.0)
    parser.add_argument("--nominal-lateral", type=float, default=0.0)
    parser.add_argument("--longitudinal-exploration-std", type=float, default=0.02)
    parser.add_argument("--lateral-exploration-std", type=float, default=0.04)
    parser.add_argument("--exploration-hold-steps", type=int, default=None)
    parser.add_argument(
        "--speed-sampling-profile",
        choices=tuple(SPEED_SAMPLING_PROFILES),
        default=BALANCED_SPEED_SAMPLING_PROFILE,
    )
    parser.add_argument("--minimum-high-speed-bin-fraction", type=float, default=None)
    parser.add_argument("--maximum-high-speed-bin-fraction", type=float, default=None)
    parser.add_argument("--minimum-mid-high-bin-fraction", type=float, default=None)
    parser.add_argument("--minimum-low-speed-bins-fraction", type=float, default=None)
    parser.add_argument("--minimum-longitudinal-action", type=float, default=0.75)
    parser.add_argument("--maximum-lateral-action", type=float, default=0.20)
    parser.add_argument("--target-velocity", type=float, default=None)
    parser.add_argument(
        "--action-semantics",
        dest="action_semantics_version",
        choices=(
            FIXED_SPEED_ACTION_SEMANTICS,
            SPEED_COUPLED_ACTION_SEMANTICS,
            TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        ),
        default=FIXED_SPEED_ACTION_SEMANTICS,
    )
    parser.add_argument("--min-command-speed", type=float, default=1.0)
    parser.add_argument("--max-command-speed", type=float, default=None)
    parser.add_argument(
        "--allow-recalibrated-fast-v231-speed",
        action="store_true",
        help=(
            "permit an explicitly re-calibrated fast_v231 speed ceiling; "
            "the target velocity must equal that ceiling"
        ),
    )
    parser.add_argument("--lookahead", type=float, default=10.0)
    parser.add_argument("--lateral-scale", type=float, default=3.5)
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
    parser.add_argument("--sqp-steps", type=int, default=24)
    parser.add_argument("--admm-steps", type=int, default=240)
    parser.add_argument("--max-obstacles", type=int, default=32)
    parser.add_argument("--return-discount", type=float, default=0.99)
    parser.add_argument("--proxy-discount", type=float, default=0.97)
    parser.add_argument("--soft-pair-weight", type=float, default=0.25)
    parser.add_argument("--post-reset-warmup-steps", type=int, default=1)
    parser.add_argument(
        "--state-aligned-authority-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "compare protected and ordinary MPC at safety-relevant states so "
            "hidden low-level CBF authority is not mislabeled Actor-controlled"
        ),
    )
    parser.add_argument("--authority-query-hard-margin", type=float, default=50.0)
    parser.add_argument("--authority-trajectory-tolerance", type=float, default=0.08)
    parser.add_argument(
        "--inverse-mpc-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "derive an ordinary-MPC high-level action that reproduces each "
            "audited protected trajectory"
        ),
    )
    parser.add_argument(
        "--proactive-dagger-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "query a same-state recovery target before takeover and activate "
            "it only when an intervention follows inside the local window"
        ),
    )
    parser.add_argument(
        "--ordinary-recovery-use-action-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "allow the ordinary recovery teacher to use its previous target; "
            "disable for a single-valued same-state DAgger teacher"
        ),
    )
    parser.add_argument(
        "--ordinary-recovery-bypass-sign",
        type=float,
        default=None,
        help="optionally fix the recovery bypass side to -1 or +1",
    )
    parser.add_argument(
        "--ordinary-recovery-prefer-inverse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "prefer the continuous inverse-MPC action reproducing the safe "
            "recovery path before trying its discrete target-bank action"
        ),
    )
    parser.add_argument("--dagger-pre-steps", type=int, default=40)
    parser.add_argument("--dagger-weight-decay", type=float, default=0.98)
    parser.add_argument(
        "--recreate-environment-each-attempt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "close and rebuild MetaDrive between episodes so simulator-local "
            "state cannot select a different MPC branch after reset"
        ),
    )
    parser.add_argument(
        "--combine-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    fast_profile = args.speed_sampling_profile == FAST_SPEED_SAMPLING_PROFILE
    profile_max_command_speed = (
        FAST_V231_MAX_COMMAND_SPEED if fast_profile else V23_MAX_COMMAND_SPEED
    )
    if args.max_command_speed is None:
        args.max_command_speed = profile_max_command_speed
    if args.target_velocity is None:
        args.target_velocity = profile_max_command_speed
    if args.exploration_hold_steps is None:
        args.exploration_hold_steps = 15 if fast_profile else 10
    profile_gate_defaults = {
        "minimum_high_speed_bin_fraction": 0.45 if fast_profile else 0.0,
        "maximum_high_speed_bin_fraction": 0.65 if fast_profile else 0.70,
        "minimum_mid_high_bin_fraction": 0.18 if fast_profile else 0.0,
        "minimum_low_speed_bins_fraction": 0.15 if fast_profile else 0.0,
    }
    for name, default in profile_gate_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.ordinary_recovery_bypass_sign is not None and not (
        -1.0 <= args.ordinary_recovery_bypass_sign <= 1.0
        and args.ordinary_recovery_bypass_sign != 0.0
    ):
        parser.error("--ordinary-recovery-bypass-sign must be -1..1 and nonzero")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be positive")
    if not 0.0 < args.minimum_attempt_success_rate <= 1.0:
        parser.error("--minimum-attempt-success-rate must be in (0, 1]")
    if args.max_attempt_multiplier < 1.0:
        parser.error("--max-attempt-multiplier must be >= 1")
    if args.exploration_hold_steps < 1:
        parser.error("--exploration-hold-steps must be positive")
    if fast_profile:
        if args.action_semantics_version != TARGET_SPEED_LATERAL_ACTION_SEMANTICS:
            parser.error("fast_v231 requires --action-semantics target_speed_lateral_v1")
        if args.max_episode_steps != 1200:
            parser.error("fast_v231 requires --max-episode-steps 1200")
        if args.exploration_hold_steps != 15:
            parser.error("fast_v231 requires --exploration-hold-steps 15")
        if (
            not args.allow_recalibrated_fast_v231_speed
            and not np.isclose(
                args.max_command_speed, FAST_V231_MAX_COMMAND_SPEED, atol=1e-8
            )
        ):
            parser.error(
                "fast_v231 requires calibrated --max-command-speed "
                f"{FAST_V231_MAX_COMMAND_SPEED}"
            )
        if (
            not args.allow_recalibrated_fast_v231_speed
            and not np.isclose(
                args.target_velocity, FAST_V231_MAX_COMMAND_SPEED, atol=1e-8
            )
        ):
            parser.error(
                "fast_v231 requires calibrated --target-velocity "
                f"{FAST_V231_MAX_COMMAND_SPEED}"
            )
        if args.allow_recalibrated_fast_v231_speed and not np.isclose(
            args.target_velocity, args.max_command_speed, atol=1e-8
        ):
            parser.error(
                "recalibrated fast_v231 requires --target-velocity to equal "
                "--max-command-speed"
            )
    speed_gate_values = (
        args.minimum_high_speed_bin_fraction,
        args.maximum_high_speed_bin_fraction,
        args.minimum_mid_high_bin_fraction,
        args.minimum_low_speed_bins_fraction,
    )
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in speed_gate_values):
        parser.error("speed-bin fraction gates must be finite and in [0, 1]")
    if args.minimum_high_speed_bin_fraction > args.maximum_high_speed_bin_fraction:
        parser.error("minimum high-speed fraction exceeds maximum")
    if args.longitudinal_exploration_std < 0.0:
        parser.error("--longitudinal-exploration-std must be nonnegative")
    if args.lateral_exploration_std < 0.0:
        parser.error("--lateral-exploration-std must be nonnegative")
    if not -1.0 <= args.minimum_longitudinal_action <= 1.0:
        parser.error("--minimum-longitudinal-action must be in [-1, 1]")
    if not 0.0 <= args.maximum_lateral_action <= 1.0:
        parser.error("--maximum-lateral-action must be in [0, 1]")
    if args.sqp_steps < 1 or args.admm_steps < 1:
        parser.error("MPC solver step counts must be positive")
    for name in (
        "mpc_tracking_weight",
        "mpc_velocity_weight",
        "mpc_road_weight",
        "mpc_road_buffer",
        "mpc_max_steer_rate",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.max_obstacles < 1:
        parser.error("--max-obstacles must be positive")
    if not 0.0 <= args.return_discount <= 1.0:
        parser.error("--return-discount must be in [0, 1]")
    if not 0.0 < args.proxy_discount < 1.0:
        parser.error("--proxy-discount must be in (0, 1)")
    if not np.isfinite(args.max_command_speed) or args.max_command_speed <= 0.0:
        parser.error("--max-command-speed must be positive and finite")
    if not 0.0 <= args.soft_pair_weight <= 1.0:
        parser.error("--soft-pair-weight must be in [0, 1]")
    if args.post_reset_warmup_steps < 0:
        parser.error("--post-reset-warmup-steps must be nonnegative")
    if not np.isfinite(args.authority_query_hard_margin):
        parser.error("--authority-query-hard-margin must be finite")
    if args.authority_trajectory_tolerance < 0.0:
        parser.error("--authority-trajectory-tolerance must be nonnegative")
    if args.dagger_pre_steps < 0:
        parser.error("--dagger-pre-steps must be nonnegative")
    if not 0.0 < args.dagger_weight_decay <= 1.0:
        parser.error("--dagger-weight-decay must be in (0, 1]")
    return args


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def partition_episode_targets(total: int, workers: int) -> list[int]:
    workers = min(max(int(workers), 1), max(int(total), 1))
    quotient, remainder = divmod(int(total), workers)
    return [quotient + int(index < remainder) for index in range(workers)]


def _configure_worker_runtime() -> None:
    # Each process owns one MetaDrive engine and one CPU MPC instance.  Keeping
    # the numerical libraries single-threaded prevents N workers from each
    # creating a full 28-thread pool on the collection machine.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def sampled_nominal_action(
    config: CollectionConfig,
    rng: np.random.Generator,
    previous: np.ndarray,
    *,
    refresh: bool,
    speed_context: str = "open",
    speed_bin_scheduler: FastV231SpeedBinScheduler | None = None,
) -> np.ndarray:
    """Generate mild target exploration without changing the MetaSafe scene."""

    previous = np.asarray(previous, dtype=np.float32).reshape(2)
    if not refresh:
        return previous.copy()
    speed_bin: int | None = None
    if config.action_semantics_version == TARGET_SPEED_LATERAL_ACTION_SEMANTICS:
        # Use planned obstacle clearance as context: open road emphasizes
        # mid/high-speed progress, while near-obstacle states deliberately
        # contribute low/mid-speed actions.  The fast profile's open-road
        # top-bin rate is capped at 72%, while aggregate gates require 45-65%.
        if config.speed_sampling_profile not in SPEED_SAMPLING_PROFILES:
            raise ValueError(
                f"unknown speed sampling profile: {config.speed_sampling_profile}"
            )
        probabilities = SPEED_SAMPLING_PROFILES[config.speed_sampling_profile]
        if speed_context not in probabilities:
            raise ValueError(f"unknown speed sampling context: {speed_context}")
        speed_bin = int(rng.choice(4, p=probabilities[speed_context]))
        within_bin = float(rng.uniform(0.0, 1.0))
        # At the recalibrated cap, lower within-bin draws leave a long tail of
        # otherwise-safe episodes just short of the 1,200-step deadline.  Bias
        # only open/near draws toward the upper part of the already-selected
        # bin.  Production collection additionally uses an episode-local
        # low-discrepancy rotation so unlucky RNG streaks cannot dominate one
        # worker; critical states retain the original uniform draw.
        if (
            config.speed_sampling_profile == FAST_SPEED_SAMPLING_PROFILE
            and speed_context in ("open", "near")
        ):
            if speed_bin_scheduler is None:
                within_bin = float(np.cbrt(within_bin))
            else:
                within_bin = speed_bin_scheduler.within_bin_fraction(
                    speed_context,
                    within_bin,
                )
        if (
            speed_bin_scheduler is not None
            and config.speed_sampling_profile == FAST_SPEED_SAMPLING_PROFILE
        ):
            speed_bin = speed_bin_scheduler.choose(
                speed_context,
                probabilities[speed_context],
                speed_bin,
                within_bin,
            )
        fraction = (speed_bin + within_bin) / 4.0
        longitudinal = 2.0 * fraction - 1.0
    else:
        longitudinal = float(config.nominal_longitudinal) + float(
            rng.normal(0.0, config.longitudinal_exploration_std)
        )
    if (
        config.action_semantics_version == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
        and config.speed_sampling_profile == FAST_SPEED_SAMPLING_PROFILE
    ):
        assert speed_bin is not None
        lateral_scale = FAST_V231_LATERAL_NOISE_SCALES[speed_bin]
    else:
        # Preserve the exact V2.3 contextual lateral exploration for the
        # balanced_v23 reproduction profile.
        lateral_scale = (
            0.20
            if speed_context == "critical"
            else 0.40
            if speed_context == "near"
            else 0.75
        )
    lateral = float(config.nominal_lateral) + float(
        rng.normal(0.0, config.lateral_exploration_std * lateral_scale)
    )
    return np.asarray(
        [
            np.clip(
                longitudinal,
                -1.0
                if config.action_semantics_version
                == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
                else config.minimum_longitudinal_action,
                1.0,
            ),
            np.clip(
                lateral,
                -config.maximum_lateral_action,
                config.maximum_lateral_action,
            ),
        ],
        dtype=np.float32,
    )


def target_speed_sampling_context(certificate: dict[str, Any]) -> str:
    """Classify the raw planned obstacle clearance for speed stratification."""

    summary = certificate.get("raw_hard_certificate", {})
    clearance = float(summary.get("obstacle_clearance", np.nan))
    # Begin braking before the two-second MPC horizon becomes obstacle-bound.
    # This lead distance absorbs receding-horizon/model mismatch instead of
    # asking the terminal projection to recover from a late high-speed state.
    if not np.isfinite(clearance) or clearance < 3.0:
        return "critical"
    if clearance < 8.0:
        return "near"
    return "open"


def _finite_min(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.min(finite)) if finite.size else float("nan")


def _finite_max(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else float("nan")


def _certificate_values(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(
        float(summary.get(name, np.nan))
        for name in (
            "hard_cbf_margin",
            "obstacle_clearance",
            "road_clearance",
            "kinematic_slack",
        )
    )


def _state_aligned_authority(
    env,
    raw_action: np.ndarray,
    certificate: dict[str, Any],
    config: CollectionConfig,
) -> dict[str, Any]:
    """Audit CBF authority and derive an ordinary-MPC teacher.

    Far-field rows keep the ordinary guardian label without an extra solve.
    Once the raw certificate approaches the configurable boundary (or a
    high-level projection already occurred), the same-state protected and
    ordinary trajectories are compared.  A rejected proposal may then get an
    inverse-MPC action that reproduces the protected trajectory using ordinary
    MPC.  A trajectory difference with a feasible positive ordinary CBF
    certificate is diagnostic-only because inactive constraints/local SQP
    branches can also differ.  The inverse action is counterfactual: it
    supervises BC/DAgger and the proxy pair, but never replaces task-TD action.
    """

    raw_summary = dict(certificate.get("raw_hard_certificate", {}))
    high_level_projection = bool(certificate.get("projection_used", False))
    solved_action = certificate.get("solved", {}).get("action")
    high_level_modified = bool(
        high_level_projection
        and (
            solved_action is None
            or np.linalg.norm(
                np.asarray(solved_action, dtype=np.float32).reshape(2)
                - np.asarray(raw_action, dtype=np.float32).reshape(2)
            )
            > 1e-6
        )
    )
    raw_hard_margin = float(raw_summary.get("hard_cbf_margin", np.inf))
    query = bool(
        config.state_aligned_authority_audit
        and (
            high_level_projection
            or not np.isfinite(raw_hard_margin)
            or raw_hard_margin <= config.authority_query_hard_margin
        )
    )
    if not query and hasattr(env, "reset_recovery_bypass_preference"):
        # A far-field frame separates obstacle-local recovery decisions.  This
        # state change is internal to the teacher planner and never enters the
        # Actor observation or the environment transition.
        env.reset_recovery_bypass_preference()
    ordinary_summary: dict[str, Any] = {}
    ordinary_solver_finite = True
    ordinary_cbf_violation = False
    ordinary_hard_violation = False
    trajectory_rmse = 0.0
    trajectory_changed = False
    audit_only_trajectory_difference = False
    actual_low_level_modified = False
    audit_failed = False
    if query:
        try:
            independent = env.certify_independent_actor_target(
                raw_action,
                reference_solved=certificate.get(
                    "raw_solved", certificate["solved"]
                ),
                trajectory_tolerance=config.authority_trajectory_tolerance,
            )
            ordinary_summary = {
                name: independent.get(name, np.nan)
                for name in (
                    "hard_cbf_margin",
                    "obstacle_clearance",
                    "road_clearance",
                    "kinematic_slack",
                )
            }
            ordinary_solver_finite = bool(
                independent.get("ordinary_solved", {}).get(
                    "solver_finite", False
                )
            )
            values = _certificate_values(ordinary_summary)
            ordinary_cbf_violation = bool(
                ordinary_solver_finite
                and np.isfinite(values[0])
                and values[0] < 0.0
            )
            ordinary_hard_violation = bool(
                ordinary_solver_finite
                and certificate_has_physical_violation(ordinary_summary)
            )
            trajectory_rmse = float(
                independent.get("trajectory_rmse", float("inf"))
            )
            trajectory_changed = bool(
                ordinary_solver_finite
                and np.isfinite(trajectory_rmse)
                and trajectory_rmse > config.authority_trajectory_tolerance
            )
            actual_low_level_modified = bool(
                trajectory_changed and ordinary_cbf_violation
            )
            audit_only_trajectory_difference = bool(
                trajectory_changed and not actual_low_level_modified
            )
        except Exception:
            ordinary_solver_finite = False
            trajectory_rmse = float("inf")
            trajectory_changed = False
            audit_only_trajectory_difference = False
            actual_low_level_modified = False
            audit_failed = True

    audit_authority = bool(
        high_level_projection
        or not ordinary_solver_finite
        or ordinary_cbf_violation
        or ordinary_hard_violation
        or trajectory_changed
    )
    intervention = bool(high_level_modified or actual_low_level_modified)
    diagnostic_only = bool(audit_authority and not intervention)
    teacher_action = None
    teacher_valid = False
    teacher_trajectory_rmse = float("nan")
    teacher_source = 0
    inverse_failed = False
    if intervention and config.inverse_mpc_teacher:
        try:
            inverse = env.derive_inverse_mpc_teacher(certificate["solved"])
            candidate = np.asarray(inverse.get("action"), dtype=np.float32).reshape(2)
            teacher_valid = bool(
                inverse.get("valid", False) and np.all(np.isfinite(candidate))
            )
            if teacher_valid:
                teacher_action = candidate
                teacher_trajectory_rmse = float(
                    inverse.get("trajectory_rmse", float("nan"))
                )
                teacher_source = 2
            else:
                inverse_failed = True
        except Exception:
            inverse_failed = True

    # A genuine high-level projection already supplies a corrected target even
    # when inverse trajectory fitting fails.  Hidden low-level authority has no
    # such target and remains teacher-invalid rather than receiving a fake one.
    if intervention and not teacher_valid and high_level_projection:
        projected = np.asarray(
            certificate["solved"]["action"], dtype=np.float32
        ).reshape(2)
        if np.all(np.isfinite(projected)):
            teacher_action = projected
            teacher_valid = True
            teacher_source = 1

    proactive_failed = False
    if (
        not intervention
        and query
        and config.proactive_dagger_teacher
        and hasattr(env, "plan_safe_recovery_trajectory")
    ):
        try:
            recovery = env.plan_safe_recovery_trajectory()
            if bool(recovery.get("obstacle_ahead", False)):
                if not bool(recovery.get("valid", False)):
                    proactive_failed = True
                else:
                    safe_solved = recovery["safe_solved"]
                    recovery_action = np.asarray(
                        recovery["action"], dtype=np.float32
                    ).reshape(2)
                    independent = env.certify_independent_actor_target(
                        recovery_action,
                        reference_solved=safe_solved,
                        trajectory_tolerance=(
                            config.authority_trajectory_tolerance
                        ),
                    )
                    candidate = recovery_action
                    candidate_valid = bool(
                        independent.get("valid", False)
                        and np.all(np.isfinite(candidate))
                    )
                    candidate_rmse = float(
                        independent.get("trajectory_rmse", float("nan"))
                    )
                    if not candidate_valid and config.inverse_mpc_teacher:
                        inverse = env.derive_inverse_mpc_teacher(safe_solved)
                        candidate = np.asarray(
                            inverse.get("action"), dtype=np.float32
                        ).reshape(2)
                        candidate_valid = bool(
                            inverse.get("valid", False)
                            and np.all(np.isfinite(candidate))
                        )
                        candidate_rmse = float(
                            inverse.get("trajectory_rmse", float("nan"))
                        )
                    if candidate_valid:
                        teacher_action = candidate
                        teacher_valid = True
                        teacher_source = 3
                        teacher_trajectory_rmse = candidate_rmse
                    else:
                        proactive_failed = True
        except Exception:
            proactive_failed = True

    return {
        "query": query,
        "audit_failed": audit_failed,
        "inverse_failed": inverse_failed,
        "proactive_failed": proactive_failed,
        "intervention": intervention,
        "actual_control_modified": intervention,
        "high_level_projection": high_level_projection,
        "high_level_modified": high_level_modified,
        "low_level_authority": actual_low_level_modified,
        "actual_low_level_modified": actual_low_level_modified,
        "audit_only_trajectory_difference": audit_only_trajectory_difference,
        "audit_authority": audit_authority,
        "diagnostic_only": diagnostic_only,
        "ordinary_solver_finite": ordinary_solver_finite,
        "ordinary_cbf_violation": ordinary_cbf_violation,
        "ordinary_hard_violation": ordinary_hard_violation,
        "ordinary_hard_certificate": ordinary_summary or raw_summary,
        "trajectory_rmse": trajectory_rmse,
        "teacher_action": teacher_action,
        "teacher_valid": teacher_valid,
        "teacher_source": teacher_source,
        "teacher_trajectory_rmse": teacher_trajectory_rmse,
    }


def _ordinary_recovery_behavior(
    env,
    nominal_action: np.ndarray,
    nominal_certificate: dict[str, Any],
    config: CollectionConfig,
) -> dict[str, Any]:
    """Choose and certify one coherent no-CBF Guardian behavior action."""

    raw_summary = dict(nominal_certificate.get("raw_hard_certificate", {}))
    raw_margin = float(raw_summary.get("hard_cbf_margin", np.inf))
    has_obstacle = bool(raw_summary.get("has_obstacle", True))
    query = bool(
        has_obstacle
        and (not np.isfinite(raw_margin) or raw_margin <= config.authority_query_hard_margin)
    )
    used = False
    failed = False
    selected_action = np.asarray(nominal_action, dtype=np.float32).reshape(2)
    execution_certificate = None
    trajectory_rmse = 0.0

    if query:
        try:
            recovery_kwargs = {}
            if not config.ordinary_recovery_use_action_history:
                recovery_kwargs["use_action_history"] = False
            if config.ordinary_recovery_bypass_sign is not None:
                recovery_kwargs["preferred_bypass_sign"] = float(
                    config.ordinary_recovery_bypass_sign
                )
            recovery = env.plan_safe_recovery_trajectory(**recovery_kwargs)
            if bool(recovery.get("valid", False)) and bool(
                recovery.get("obstacle_ahead", False)
            ):
                recovery_action = np.asarray(
                    recovery["action"], dtype=np.float32
                ).reshape(2)
                if (
                    config.ordinary_recovery_prefer_inverse
                    and config.inverse_mpc_teacher
                ):
                    inverse = env.derive_inverse_mpc_teacher(
                        recovery["safe_solved"]
                    )
                    if bool(inverse.get("valid", False)):
                        selected_action = np.asarray(
                            inverse["action"], dtype=np.float32
                        ).reshape(2)
                        execution_certificate = inverse.get(
                            "execution_certificate"
                        )
                        trajectory_rmse = float(
                            inverse.get("trajectory_rmse", 0.0)
                        )
                if execution_certificate is None:
                    independent = env.certify_independent_actor_target(
                        recovery_action,
                        reference_solved=recovery["safe_solved"],
                        trajectory_tolerance=(
                            config.authority_trajectory_tolerance
                        ),
                    )
                    if bool(independent.get("valid", False)):
                        selected_action = recovery_action
                        execution_certificate = independent.get(
                            "execution_certificate"
                        )
                        trajectory_rmse = float(
                            independent.get("trajectory_rmse", 0.0)
                        )
                if (
                    execution_certificate is None
                    and config.inverse_mpc_teacher
                    and not config.ordinary_recovery_prefer_inverse
                ):
                    inverse = env.derive_inverse_mpc_teacher(
                        recovery["safe_solved"]
                    )
                    if bool(inverse.get("valid", False)):
                        selected_action = np.asarray(
                            inverse["action"], dtype=np.float32
                        ).reshape(2)
                        execution_certificate = inverse.get(
                            "execution_certificate"
                        )
                        trajectory_rmse = float(
                            inverse.get("trajectory_rmse", 0.0)
                        )
                used = execution_certificate is not None
                failed = not used
            else:
                failed = bool(recovery.get("obstacle_ahead", False))
        except Exception:
            failed = True
    elif hasattr(env, "reset_recovery_bypass_preference"):
        env.reset_recovery_bypass_preference()

    if execution_certificate is None:
        execution_certificate = env.certify_ordinary_target_action(
            nominal_action
        )
        selected_action = np.asarray(
            execution_certificate["solved"]["action"], dtype=np.float32
        ).reshape(2)

    return {
        "action": selected_action,
        "certificate": execution_certificate,
        "query": query,
        "used": used,
        "failed": failed,
        "trajectory_rmse": trajectory_rmse,
    }


def _activate_pre_intervention_teachers(
    rows: dict[str, list[Any]],
    *,
    boundary_index: int,
    horizon: int,
    decay: float,
    boundary_confidence: float,
) -> list[int]:
    """Activate only state-aligned latent labels preceding a real takeover."""

    if horizon <= 0 or boundary_index <= 0 or boundary_confidence <= 0.0:
        return []
    activated = []
    start = max(0, int(boundary_index) - int(horizon))
    for index in range(start, int(boundary_index)):
        if (
            bool(rows["teacher_valid"][index])
            and int(rows["teacher_source"][index]) == 3
            and not bool(rows["intervention"][index])
        ):
            distance = int(boundary_index) - index
            weight = float(boundary_confidence) * float(decay) ** distance
            previous = float(rows["teacher_weight"][index])
            rows["teacher_weight"][index] = np.float32(max(previous, weight))
            if previous <= 0.0 and weight > 0.0:
                activated.append(index)
    return activated


def _concat_rows(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        return {}
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("episode array schemas do not match")
    return {key: np.concatenate([row[key] for row in rows], axis=0) for key in keys}


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_episode(
    env,
    config: CollectionConfig,
    *,
    worker_id: int,
    attempt_index: int,
    accepted_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    policy_seed = int(
        config.collector_seed + 1_000_003 * worker_id + 10_007 * attempt_index
    )
    rng = np.random.default_rng(policy_seed)
    observation, reset_info = env.reset(seed=config.scene_seed)
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    initial_observation_sha256 = hashlib.sha256(observation.tobytes()).hexdigest()
    scene_signature = str(reset_info.get("metasafe_scene_signature", ""))
    action_semantics_version = str(
        reset_info.get(
            "action_semantics_version",
            FIXED_SPEED_ACTION_SEMANTICS,
        )
    )
    if action_semantics_version != config.action_semantics_version:
        raise RuntimeError(
            "environment action semantics mismatch: "
            f"{action_semantics_version} != "
            f"{config.action_semantics_version}"
        )
    nominal_action = np.asarray(
        [config.nominal_longitudinal, config.nominal_lateral],
        dtype=np.float32,
    )
    speed_bin_scheduler = (
        FastV231SpeedBinScheduler()
        if config.speed_sampling_profile == FAST_SPEED_SAMPLING_PROFILE
        and config.action_semantics_version
        == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
        else None
    )

    rows: dict[str, list[Any]] = {
        "obs": [],
        "next_obs": [],
        "raw_action": [],
        "action": [],
        "proxy_corrected_action": [],
        "teacher_action": [],
        "teacher_valid": [],
        "teacher_weight": [],
        "teacher_source": [],
        "teacher_trajectory_rmse": [],
        "low_level_action": [],
        "reward": [],
        "task_reward": [],
        "cost": [],
        "done": [],
        "terminated": [],
        "truncated": [],
        "episode_uid": [],
        "worker_id": [],
        "attempt_index": [],
        "step_index": [],
        "scene_seed": [],
        "scene_signature": [],
        "action_semantics_version": [],
        "episode_step_limit": [],
        "speed_sampling_profile": [],
        "lookahead_distance": [],
        "min_command_speed": [],
        "max_command_speed": [],
        "speed_mapping": [],
        "mpc_horizon": [],
        "raw_target_speed": [],
        "executed_target_speed": [],
        "speed_command_delta": [],
        "speed_only_intervention": [],
        "reachable_speed_lower": [],
        "reachable_speed_upper": [],
        "actual_vehicle_speed": [],
        "emergency_stop": [],
        "policy_seed": [],
        "projection_used": [],
        "projection_norm": [],
        "proxy_pair_norm": [],
        "projection_failed": [],
        "solver_finite": [],
        "actor_controlled": [],
        "intervention": [],
        "intervention_hard": [],
        "intervention_soft": [],
        "intervention_numerical": [],
        "intervention_kind": [],
        "actual_control_modified": [],
        "actual_low_level_modified": [],
        "audit_only_trajectory_difference": [],
        "audit_authority": [],
        "filter_version": [],
        "spatial_reserve": [],
        "kinematic_reserve": [],
        "proposal_evaluated": [],
        "proxy_pair_valid": [],
        "proxy_pair_weight": [],
        "backup_feasible": [],
        "recovery_success": [],
        "execution_layer": [],
        "authority_audit_queried": [],
        "authority_audit_failed": [],
        "inverse_teacher_failed": [],
        "low_level_cbf_authority": [],
        "ordinary_solver_finite": [],
        "ordinary_cbf_violation": [],
        "ordinary_hard_violation": [],
        "cbf_trajectory_rmse": [],
        "hard_cbf_margin": [],
        "raw_hard_cbf_margin": [],
        "obstacle_clearance": [],
        "raw_obstacle_clearance": [],
        "road_clearance": [],
        "raw_road_clearance": [],
        "kinematic_slack": [],
        "raw_kinematic_slack": [],
        "physical_body_sat_margin": [],
        "route_completion": [],
        "critic_action_compatible": [],
        "bc_valid": [],
        "post_reset_warmup_steps": [],
        "ordinary_recovery_teacher": [],
    }

    task_return = 0.0
    environment_cost = 0.0
    cost_events = 0
    collision = False
    road_departure = False
    success = False
    solver_failures = 0
    projection_failures = 0
    projection_steps = 0
    intervention_steps = 0
    authority_queries = 0
    authority_audit_failures = 0
    low_level_authority_steps = 0
    inverse_teacher_failures = 0
    inverse_teacher_steps = 0
    proactive_teacher_failures = 0
    proactive_teacher_steps = 0
    proactive_teacher_activated: set[int] = set()
    ordinary_recovery_steps = 0
    ordinary_recovery_failures = 0
    controller_error: str | None = None
    terminated = truncated = False
    hard_margins: list[float] = []
    obstacle_clearances: list[float] = []
    road_clearances: list[float] = []
    kinematic_slacks: list[float] = []
    physical_margins: list[float] = []
    projection_norms: list[float] = []
    speed_context = "open"

    episode_uid = int(worker_id * 1_000_000 + accepted_index)
    for step in range(config.max_episode_steps):
        nominal_action = sampled_nominal_action(
            config,
            rng,
            nominal_action,
            refresh=(step % config.exploration_hold_steps == 0),
            speed_context=speed_context,
            speed_bin_scheduler=speed_bin_scheduler,
        )
        try:
            nominal_certificate = env.certify_target_action(nominal_action)
            if (
                config.action_semantics_version
                == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
            ):
                planned_context = target_speed_sampling_context(
                    nominal_certificate
                )
                if planned_context != speed_context:
                    nominal_action = sampled_nominal_action(
                        config,
                        rng,
                        nominal_action,
                        refresh=True,
                        speed_context=planned_context,
                        speed_bin_scheduler=speed_bin_scheduler,
                    )
                    nominal_certificate = env.certify_target_action(
                        nominal_action
                    )
                speed_context = target_speed_sampling_context(
                    nominal_certificate
                )
            recorded_raw_action = nominal_action.copy()
            ordinary_recovery_used = False
            if config.behavior_policy == "ordinary_recovery":
                behavior = _ordinary_recovery_behavior(
                    env, nominal_action, nominal_certificate, config
                )
                certificate = behavior["certificate"]
                recorded_raw_action = np.asarray(
                    behavior["action"], dtype=np.float32
                ).reshape(2)
                ordinary_recovery_used = bool(behavior["used"])
                executed_summary = dict(
                    certificate.get("raw_hard_certificate", {})
                )
                solved_finite = bool(
                    certificate.get("solved", {}).get("solver_finite", False)
                )
                authority = {
                    "query": bool(behavior["query"]),
                    "audit_failed": False,
                    "inverse_failed": False,
                    "proactive_failed": False,
                    "intervention": False,
                    "actual_control_modified": False,
                    "high_level_projection": False,
                    "high_level_modified": False,
                    "low_level_authority": False,
                    "actual_low_level_modified": False,
                    "audit_only_trajectory_difference": False,
                    "audit_authority": False,
                    "diagnostic_only": False,
                    "ordinary_solver_finite": solved_finite,
                    "ordinary_cbf_violation": bool(
                        solved_finite
                        and np.isfinite(
                            float(executed_summary.get("hard_cbf_margin", np.nan))
                        )
                        and float(executed_summary.get("hard_cbf_margin", np.nan))
                        < 0.0
                    ),
                    "ordinary_hard_violation": bool(
                        solved_finite
                        and certificate_has_physical_violation(executed_summary)
                    ),
                    "ordinary_hard_certificate": executed_summary,
                    "trajectory_rmse": 0.0,
                    "teacher_action": recorded_raw_action.copy(),
                    "teacher_valid": True,
                    "teacher_source": 4 if ordinary_recovery_used else 0,
                    "teacher_trajectory_rmse": float(
                        behavior["trajectory_rmse"]
                    ),
                }
                ordinary_recovery_failures += int(behavior["failed"])
            else:
                certificate = nominal_certificate
                authority = _state_aligned_authority(
                    env, nominal_action, certificate, config
                )
            if (
                config.action_semantics_version
                == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
                and bool(certificate.get("projection_failed", False))
            ):
                # An empty certified target set is an episode rejection, not
                # permission to execute the least-unsafe NMPC candidate.  The
                # next attempt starts from a clean fixed-scene process.
                projection_failures += 1
                controller_error = (
                    "UncertifiedTarget:terminal_projection_failed"
                )
                break
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.execute_certified_target(certificate)
        except Exception as exc:
            controller_error = f"{type(exc).__name__}:{exc}"
            break

        next_observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        mgf = info.get("mgf", {})
        hard = mgf.get("hard_safety", {})
        raw_hard = mgf.get(
            "raw_hard_safety", certificate.get("raw_hard_certificate", {})
        )
        physical = info.get("mpc_control", {})
        executed_action = np.asarray(
            info.get("executed_target", nominal_action), dtype=np.float32
        ).reshape(2)
        low_level_action = np.asarray(
            physical.get("low_level_action", [np.nan, np.nan]),
            dtype=np.float32,
        ).reshape(2)
        step_cost = max(float(info.get("cost", 0.0)), 0.0)
        projection_norm = float(mgf.get("projection_norm", 0.0))
        projection_used = bool(mgf.get("projection_used", projection_norm > 1e-6))
        projection_failed = bool(mgf.get("projection_failed", False))
        solver_finite = bool(mgf.get("solver_finite", False))
        hard_margin = float(hard.get("hard_cbf_margin", np.nan))
        obstacle_clearance = float(hard.get("obstacle_clearance", np.nan))
        road_clearance = float(hard.get("road_clearance", np.nan))
        kinematic_slack = float(hard.get("kinematic_slack", np.nan))
        raw_hard_margin = float(raw_hard.get("hard_cbf_margin", np.nan))
        raw_obstacle_clearance = float(raw_hard.get("obstacle_clearance", np.nan))
        raw_road_clearance = float(raw_hard.get("road_clearance", np.nan))
        raw_kinematic_slack = float(raw_hard.get("kinematic_slack", np.nan))
        authority_query = bool(authority["query"])
        actual_high_level_modified = bool(
            np.linalg.norm(executed_action - recorded_raw_action) > 1e-6
        )
        actual_low_level_modified = bool(
            authority.get("actual_low_level_modified", False)
        )
        low_level_authority = actual_low_level_modified
        intervention = bool(
            actual_high_level_modified or actual_low_level_modified
        )
        audit_authority = bool(
            authority.get("audit_authority", intervention) or intervention
        )
        ordinary_solver_finite = bool(authority["ordinary_solver_finite"])
        intervention_numerical = bool(
            projection_failed
            or not solver_finite
        )
        physical_margin = float(
            physical.get("physical_body_sat_margin", np.nan)
        )
        environment_hard_failure = bool(
            step_cost > 0.0
            or info.get("crash_vehicle", False)
            or info.get("crash_object", False)
            or info.get("out_of_road", False)
            or info.get("road_departure", False)
            or (np.isfinite(physical_margin) and physical_margin <= 0.0)
        )
        intervention_hard = bool(
            intervention
            and not intervention_numerical
            and (
                authority["ordinary_hard_violation"]
                or certificate_has_physical_violation(
                    authority["ordinary_hard_certificate"]
                )
                or environment_hard_failure
            )
        )
        intervention_soft = bool(
            intervention and not intervention_hard and not intervention_numerical
        )
        diagnostic_only = bool(
            (audit_authority and not intervention) or intervention_numerical
        )
        intervention_kind = (
            INTERVENTION_HARD
            if intervention_hard
            else INTERVENTION_SOFT
            if intervention_soft
            else INTERVENTION_DIAGNOSTIC
            if diagnostic_only
            else INTERVENTION_NONE
        )
        proxy_pair_weight = (
            1.0
            if intervention_hard
            else config.soft_pair_weight
            if intervention_soft
            else 0.0
        )
        if int(authority["teacher_source"]) == 4:
            teacher_action = np.asarray(
                authority["teacher_action"], dtype=np.float32
            ).reshape(2)
            teacher_valid = True
            teacher_weight = 1.0
            teacher_source = 4
        elif intervention and bool(authority["teacher_valid"]):
            teacher_action = np.asarray(
                authority["teacher_action"], dtype=np.float32
            ).reshape(2)
            teacher_valid = True
            teacher_weight = proxy_pair_weight
            teacher_source = int(authority["teacher_source"])
        elif bool(authority["teacher_valid"]) and int(
            authority["teacher_source"]
        ) == 3:
            teacher_action = np.asarray(
                authority["teacher_action"], dtype=np.float32
            ).reshape(2)
            teacher_valid = True
            # The label is a same-state counterfactual query.  It becomes BC
            # supervision only if an actual intervention follows shortly.
            teacher_weight = 0.0
            teacher_source = 3
        elif not intervention:
            teacher_action = executed_action.copy()
            teacher_valid = True
            teacher_weight = 1.0
            teacher_source = 0
        else:
            teacher_action = executed_action.copy()
            teacher_valid = False
            teacher_weight = 0.0
            teacher_source = -1
        proxy_corrected_action = (
            teacher_action.copy()
            if intervention and teacher_valid
            else executed_action.copy()
        )
        proxy_pair_norm = float(
            np.linalg.norm(proxy_corrected_action - recorded_raw_action)
        )
        proxy_pair_valid = bool(
            intervention
            and not intervention_numerical
            and teacher_valid
            and proxy_pair_norm > 1e-6
        )
        proposal_evaluated = bool(
            solver_finite and (not authority_query or ordinary_solver_finite)
        )
        done = bool(terminated or truncated)

        rows["obs"].append(observation.copy())
        rows["next_obs"].append(next_observation.copy())
        rows["raw_action"].append(recorded_raw_action.copy())
        rows["action"].append(executed_action.copy())
        rows["proxy_corrected_action"].append(proxy_corrected_action)
        rows["teacher_action"].append(teacher_action)
        rows["teacher_valid"].append(np.bool_(teacher_valid))
        rows["teacher_weight"].append(np.float32(teacher_weight))
        rows["teacher_source"].append(np.int8(teacher_source))
        rows["teacher_trajectory_rmse"].append(
            np.float32(authority["teacher_trajectory_rmse"])
        )
        rows["low_level_action"].append(low_level_action)
        rows["reward"].append(np.float32(reward))
        rows["task_reward"].append(np.float32(info.get("task_reward", reward)))
        rows["cost"].append(np.float32(step_cost))
        rows["done"].append(np.bool_(done))
        rows["terminated"].append(np.bool_(terminated))
        rows["truncated"].append(np.bool_(truncated))
        rows["episode_uid"].append(np.int64(episode_uid))
        rows["worker_id"].append(np.int16(worker_id))
        rows["attempt_index"].append(np.int32(attempt_index))
        rows["step_index"].append(np.int32(step))
        rows["scene_seed"].append(np.int32(config.scene_seed))
        rows["scene_signature"].append(scene_signature)
        rows["action_semantics_version"].append(
            config.action_semantics_version
        )
        rows["episode_step_limit"].append(np.int32(config.max_episode_steps))
        rows["speed_sampling_profile"].append(config.speed_sampling_profile)
        rows["lookahead_distance"].append(np.float32(config.lookahead))
        rows["min_command_speed"].append(
            np.float32(config.min_command_speed)
        )
        rows["max_command_speed"].append(
            np.float32(config.max_command_speed)
        )
        rows["speed_mapping"].append(
            str(info.get("speed_mapping", ""))
        )
        rows["mpc_horizon"].append(np.float32(config.horizon))
        rows["raw_target_speed"].append(
            np.float32(info.get("raw_target_speed", 0.0))
        )
        rows["executed_target_speed"].append(
            np.float32(info.get("executed_target_speed", 0.0))
        )
        rows["speed_command_delta"].append(
            np.float32(info.get("speed_command_delta", 0.0))
        )
        rows["speed_only_intervention"].append(
            np.bool_(info.get("speed_only_intervention", False))
        )
        rows["reachable_speed_lower"].append(
            np.float32(
                info.get(
                    "reachable_speed_lower",
                    mgf.get("reachable_speed_lower", np.nan),
                )
            )
        )
        rows["reachable_speed_upper"].append(
            np.float32(
                info.get(
                    "reachable_speed_upper",
                    mgf.get("reachable_speed_upper", np.nan),
                )
            )
        )
        rows["actual_vehicle_speed"].append(
            np.float32(
                info.get(
                    "actual_vehicle_speed",
                    mgf.get("actual_vehicle_speed", np.nan),
                )
            )
        )
        rows["emergency_stop"].append(
            np.bool_(
                info.get("emergency_stop", mgf.get("emergency_stop", False))
            )
        )
        rows["policy_seed"].append(np.int64(policy_seed))
        rows["projection_used"].append(np.bool_(projection_used))
        rows["projection_norm"].append(np.float32(projection_norm))
        rows["proxy_pair_norm"].append(np.float32(proxy_pair_norm))
        rows["projection_failed"].append(np.bool_(projection_failed))
        rows["solver_finite"].append(np.bool_(solver_finite))
        rows["actor_controlled"].append(
            np.bool_(not intervention and not intervention_numerical)
        )
        rows["intervention"].append(np.bool_(intervention))
        rows["intervention_hard"].append(np.bool_(intervention_hard))
        rows["intervention_soft"].append(np.bool_(intervention_soft))
        rows["intervention_numerical"].append(np.bool_(intervention_numerical))
        rows["intervention_kind"].append(np.int8(intervention_kind))
        rows["actual_control_modified"].append(np.bool_(intervention))
        rows["actual_low_level_modified"].append(
            np.bool_(actual_low_level_modified)
        )
        rows["audit_only_trajectory_difference"].append(
            np.bool_(
                authority.get("audit_only_trajectory_difference", False)
            )
        )
        rows["audit_authority"].append(np.bool_(audit_authority))
        rows["filter_version"].append(np.int32(0))
        rows["spatial_reserve"].append(np.float32(config.projection_reserve))
        rows["kinematic_reserve"].append(
            np.float32(config.projection_kinematic_reserve)
        )
        rows["proposal_evaluated"].append(np.bool_(proposal_evaluated))
        rows["proxy_pair_valid"].append(np.bool_(proxy_pair_valid))
        rows["proxy_pair_weight"].append(np.float32(proxy_pair_weight))
        rows["backup_feasible"].append(
            np.bool_(solver_finite and not projection_failed)
        )
        rows["recovery_success"].append(
            np.bool_(solver_finite and not projection_failed)
        )
        rows["execution_layer"].append(
            np.int8(
                (2 if actual_low_level_modified else 0)
                + (1 if actual_high_level_modified else 0)
            )
        )
        rows["authority_audit_queried"].append(np.bool_(authority_query))
        rows["authority_audit_failed"].append(
            np.bool_(authority["audit_failed"])
        )
        rows["inverse_teacher_failed"].append(
            np.bool_(authority["inverse_failed"])
        )
        rows["low_level_cbf_authority"].append(
            np.bool_(low_level_authority)
        )
        rows["ordinary_solver_finite"].append(
            np.bool_(ordinary_solver_finite)
        )
        rows["ordinary_cbf_violation"].append(
            np.bool_(authority.get("ordinary_cbf_violation", False))
        )
        rows["ordinary_hard_violation"].append(
            np.bool_(authority["ordinary_hard_violation"])
        )
        rows["cbf_trajectory_rmse"].append(
            np.float32(authority["trajectory_rmse"])
        )
        rows["hard_cbf_margin"].append(np.float32(hard_margin))
        rows["raw_hard_cbf_margin"].append(np.float32(raw_hard_margin))
        rows["obstacle_clearance"].append(np.float32(obstacle_clearance))
        rows["raw_obstacle_clearance"].append(np.float32(raw_obstacle_clearance))
        rows["road_clearance"].append(np.float32(road_clearance))
        rows["raw_road_clearance"].append(np.float32(raw_road_clearance))
        rows["kinematic_slack"].append(np.float32(kinematic_slack))
        rows["raw_kinematic_slack"].append(np.float32(raw_kinematic_slack))
        rows["physical_body_sat_margin"].append(np.float32(physical_margin))
        rows["route_completion"].append(
            np.float32(physical.get("route_completion", np.nan))
        )
        rows["critic_action_compatible"].append(np.bool_(True))
        rows["bc_valid"].append(np.bool_(teacher_valid))
        rows["post_reset_warmup_steps"].append(
            np.int16(config.post_reset_warmup_steps)
        )
        rows["ordinary_recovery_teacher"].append(
            np.bool_(ordinary_recovery_used)
        )

        if intervention:
            proactive_teacher_activated.update(
                _activate_pre_intervention_teachers(
                    rows,
                    boundary_index=len(rows["obs"]) - 1,
                    horizon=config.dagger_pre_steps,
                    decay=config.dagger_weight_decay,
                    boundary_confidence=proxy_pair_weight,
                )
            )

        task_return += float(info.get("task_reward", reward))
        environment_cost += step_cost
        cost_events += int(step_cost > 0.0)
        collision = bool(
            collision
            or info.get("crash_vehicle", False)
            or info.get("crash_object", False)
        )
        road_departure = bool(
            road_departure
            or info.get("out_of_road", False)
            or info.get("road_departure", False)
        )
        success = bool(
            success or info.get("arrive_dest", False) or info.get("success", False)
        )
        solver_failures += int(not solver_finite)
        projection_failures += int(projection_failed)
        projection_steps += int(projection_used)
        intervention_steps += int(intervention)
        authority_queries += int(authority_query)
        authority_audit_failures += int(authority["audit_failed"])
        low_level_authority_steps += int(low_level_authority)
        inverse_teacher_failures += int(authority["inverse_failed"])
        inverse_teacher_steps += int(teacher_source == 2)
        proactive_teacher_failures += int(authority["proactive_failed"])
        proactive_teacher_steps += int(teacher_source == 3)
        ordinary_recovery_steps += int(teacher_source == 4)
        hard_margins.append(hard_margin)
        obstacle_clearances.append(obstacle_clearance)
        road_clearances.append(road_clearance)
        kinematic_slacks.append(kinematic_slack)
        physical_margins.append(physical_margin)
        projection_norms.append(projection_norm)
        observation = next_observation
        if done:
            break

    steps = len(rows["obs"])
    collector_timeout = bool(
        steps >= config.max_episode_steps and not (terminated or truncated)
    )
    rejection_reasons = []
    if not success:
        rejection_reasons.append("no_success")
    if collision:
        rejection_reasons.append("collision")
    if road_departure:
        rejection_reasons.append("road_departure")
    if collector_timeout:
        rejection_reasons.append("collector_timeout")
    if controller_error is not None:
        rejection_reasons.append("controller_error")
    if solver_failures:
        rejection_reasons.append("solver_failure")
    if projection_failures:
        rejection_reasons.append("projection_failure")
    if config.require_zero_cost and environment_cost > 0.0:
        rejection_reasons.append("positive_environment_cost")
    accepted = not rejection_reasons
    episode_speed_bin_counts = np.zeros((4,), dtype=np.int64)
    episode_speed_bin_fractions = np.zeros((4,), dtype=np.float64)
    if (
        steps
        and config.action_semantics_version
        == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
    ):
        lower = np.asarray(rows["reachable_speed_lower"], dtype=np.float64)
        upper = np.asarray(rows["reachable_speed_upper"], dtype=np.float64)
        speed = np.asarray(rows["executed_target_speed"], dtype=np.float64)
        emergency = np.asarray(rows["emergency_stop"], dtype=bool)
        fraction = np.clip(
            (speed - lower) / (upper - lower + 1e-6), 0.0, 1.0
        )
        bins = np.minimum((fraction * 4.0).astype(np.int32), 3)
        episode_speed_bin_counts = np.bincount(
            bins[~emergency], minlength=4
        )
        episode_speed_bin_fractions = episode_speed_bin_counts / max(
            int(np.sum(episode_speed_bin_counts)), 1
        )
    summary = {
        "worker_id": worker_id,
        "attempt_index": attempt_index,
        "accepted_index": accepted_index if accepted else None,
        "episode_uid": episode_uid if accepted else None,
        "policy_seed": policy_seed,
        "scene_seed": config.scene_seed,
        "scene_signature": scene_signature,
        "episode_step_limit": config.max_episode_steps,
        "speed_sampling_profile": config.speed_sampling_profile,
        "initial_observation_sha256": initial_observation_sha256,
        "post_reset_warmup_steps": int(
            reset_info.get("post_reset_warmup_steps", 0)
        ),
        "post_reset_observation_linf_delta": float(
            reset_info.get("post_reset_observation_linf_delta", 0.0)
        ),
        "map_blocks": reset_info.get("metasafe_map_blocks", []),
        "scene_descriptor": reset_info.get("metasafe_scene", {}),
        "obstacle_objects": int(reset_info.get("metasafe_obstacle_objects", 0)),
        "steps": steps,
        "task_return": task_return,
        "raw_target_speed_mean": (
            float(np.mean(rows["raw_target_speed"])) if steps else float("nan")
        ),
        "executed_target_speed_mean": (
            float(np.mean(rows["executed_target_speed"]))
            if steps
            else float("nan")
        ),
        "actual_vehicle_speed_mean": (
            float(np.mean(rows["actual_vehicle_speed"]))
            if steps
            else float("nan")
        ),
        "speed_bin_counts": episode_speed_bin_counts.tolist(),
        "speed_bin_fractions": episode_speed_bin_fractions.tolist(),
        "success": success,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "collision": collision,
        "road_departure": road_departure,
        "environment_cost": environment_cost,
        "cost_events": cost_events,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "collector_timeout": collector_timeout,
        "controller_error": controller_error,
        "solver_failures": solver_failures,
        "projection_failures": projection_failures,
        "projection_rate": projection_steps / max(steps, 1),
        "authority_query_rate": authority_queries / max(steps, 1),
        "authority_audit_failures": authority_audit_failures,
        "low_level_authority_rate": low_level_authority_steps / max(steps, 1),
        "complete_intervention_rate": (
            intervention_steps / max(steps, 1)
        ),
        "inverse_teacher_rate": inverse_teacher_steps / max(steps, 1),
        "inverse_teacher_failures": inverse_teacher_failures,
        "proactive_teacher_rate": proactive_teacher_steps / max(steps, 1),
        "proactive_teacher_rows": proactive_teacher_steps,
        "proactive_teacher_activated_rows": len(proactive_teacher_activated),
        "proactive_teacher_failures": proactive_teacher_failures,
        "ordinary_recovery_teacher_rows": ordinary_recovery_steps,
        "ordinary_recovery_teacher_failures": ordinary_recovery_failures,
        "max_projection_norm": _finite_max(projection_norms),
        "min_hard_cbf_margin": _finite_min(hard_margins),
        "min_obstacle_clearance": _finite_min(obstacle_clearances),
        "min_road_clearance": _finite_min(road_clearances),
        "min_kinematic_slack": _finite_min(kinematic_slacks),
        "min_physical_body_sat_margin": _finite_min(physical_margins),
    }
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    behavior_return_to_go = np.zeros((steps,), dtype=np.float32)
    running_return = 0.0
    for index in range(steps - 1, -1, -1):
        running_return = float(arrays["reward"][index]) + (
            config.return_discount * running_return
        )
        behavior_return_to_go[index] = np.float32(running_return)
    arrays["behavior_return_to_go"] = behavior_return_to_go
    arrays["behavior_return_valid"] = np.ones((steps,), dtype=bool)
    arrays["q_anchor_valid"] = np.ones((steps,), dtype=bool)
    (
        arrays["guardian_dependence_target"],
        arrays["guardian_dependence_corrected_target"],
        arrays["guardian_dependence_target_valid"],
        arrays["guardian_dependence_corrected_valid"],
    ) = discounted_guardian_dependence_targets(
        arrays["intervention_hard"],
        arrays["intervention_soft"],
        arrays["intervention_numerical"],
        arrays["proxy_pair_valid"],
        arrays["done"],
        np.zeros((steps,), dtype=np.int32),
        gamma=config.proxy_discount,
    )
    return summary, arrays


def _worker_main(payload: dict[str, Any]) -> dict[str, Any]:
    _configure_worker_runtime()
    config = CollectionConfig(**payload["config"])
    worker_id = int(payload["worker_id"])
    target_episodes = int(payload["target_episodes"])
    output_dir = Path(config.output_dir)
    shard_path = output_dir / "shards" / f"worker_{worker_id:02d}.npz"
    worker_summary_path = output_dir / "workers" / f"worker_{worker_id:02d}.json"
    max_attempts = max(
        target_episodes,
        int(math.ceil(target_episodes * config.max_attempt_multiplier)),
    )

    from relax.env.drive.metasafe import make_metasafe_cbf_mpc_env

    started = time.time()
    def make_environment():
        return make_metasafe_cbf_mpc_env(
            fixed_seed=config.scene_seed,
            use_render=False,
            lookahead=config.lookahead,
            lateral_scale=config.lateral_scale,
            target_velocity=config.target_velocity,
            action_semantics_version=config.action_semantics_version,
            min_command_speed=config.min_command_speed,
            max_command_speed=config.max_command_speed,
            horizon=config.horizon,
            tracking_weight=config.mpc_tracking_weight,
            velocity_weight=config.mpc_velocity_weight,
            road_weight=config.mpc_road_weight,
            road_buffer=config.mpc_road_buffer,
            max_steer_rate=config.mpc_max_steer_rate,
            clearance=config.clearance,
            cbf_gamma=config.cbf_gamma,
            projection_reserve=config.projection_reserve,
            projection_kinematic_reserve=config.projection_kinematic_reserve,
            sqp_steps=config.sqp_steps,
            admm_steps=config.admm_steps,
            max_obstacles=config.max_obstacles,
            post_reset_warmup_steps=config.post_reset_warmup_steps,
        )

    env = None
    accepted_arrays: list[dict[str, np.ndarray]] = []
    episode_summaries: list[dict[str, Any]] = []
    accepted = 0
    try:
        for attempt_index in range(max_attempts):
            if env is None:
                env = make_environment()
            try:
                summary, arrays = _run_episode(
                    env,
                    config,
                    worker_id=worker_id,
                    attempt_index=attempt_index,
                    accepted_index=accepted,
                )
            finally:
                if config.recreate_environment_each_attempt:
                    env.close()
                    env = None
            episode_summaries.append(summary)
            if summary["accepted"]:
                accepted_arrays.append(arrays)
                accepted += 1
            print(
                f"[metasafe worker {worker_id}] accepted={accepted}/"
                f"{target_episodes}, attempts={attempt_index + 1}, "
                f"last_success={summary['success']}, "
                f"last_cost={summary['environment_cost']:.3f}, "
                "rejection="
                + (
                    ",".join(summary["rejection_reasons"])
                    if summary["rejection_reasons"]
                    else "none"
                ),
                flush=True,
            )
            if accepted >= target_episodes:
                break
    finally:
        if env is not None:
            env.close()

    arrays = _concat_rows(accepted_arrays)
    if arrays:
        _atomic_save_npz(shard_path, arrays)
    scene_descriptor = (
        episode_summaries[0].pop("scene_descriptor", {}) if episode_summaries else {}
    )
    for row in episode_summaries[1:]:
        row.pop("scene_descriptor", None)
    worker_summary = {
        "worker_id": worker_id,
        "pid": os.getpid(),
        "target_episodes": target_episodes,
        "accepted_episodes": accepted,
        "attempts": len(episode_summaries),
        "attempt_success_rate": accepted / max(len(episode_summaries), 1),
        "transitions": int(arrays.get("obs", np.zeros((0,))).shape[0]),
        "wall_time_seconds": time.time() - started,
        "scene_descriptor": scene_descriptor,
        "shard": str(shard_path) if arrays else None,
        "shard_sha256": _sha256(shard_path) if arrays else None,
        "episodes": episode_summaries,
    }
    worker_summary_path.parent.mkdir(parents=True, exist_ok=True)
    worker_summary_path.write_text(
        json.dumps(
            worker_summary,
            indent=2,
            allow_nan=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return {
        key: worker_summary[key]
        for key in (
            "worker_id",
            "pid",
            "target_episodes",
            "accepted_episodes",
            "attempts",
            "attempt_success_rate",
            "transitions",
            "wall_time_seconds",
            "shard",
            "shard_sha256",
        )
    }


def _combine_shards(
    shard_paths: list[Path], output_path: Path
) -> tuple[dict[str, np.ndarray], dict[int, int]]:
    parts = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as data:
            parts.append({key: np.asarray(data[key]) for key in data.files})
    combined = _concat_rows(parts)
    unique_uids = sorted(int(value) for value in np.unique(combined["episode_uid"]))
    uid_to_episode = {
        uid: episode_index for episode_index, uid in enumerate(unique_uids)
    }
    combined["episode_id"] = np.asarray(
        [uid_to_episode[int(value)] for value in combined["episode_uid"]],
        dtype=np.int32,
    )
    _atomic_save_npz(output_path, combined)
    return combined, uid_to_episode


def _prepare_output(config: CollectionConfig) -> Path:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    (output_dir / "workers").mkdir(parents=True, exist_ok=True)
    return output_dir


def collect(config: CollectionConfig) -> dict[str, Any]:
    _configure_worker_runtime()
    output_dir = _prepare_output(config)
    started = time.time()
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(asdict(config), indent=2, allow_nan=True),
        encoding="utf-8",
    )

    targets = partition_episode_targets(config.episodes, config.num_envs)
    worker_payloads = [
        {
            "config": asdict(config),
            "worker_id": worker_id,
            "target_episodes": target,
        }
        for worker_id, target in enumerate(targets)
    ]
    worker_results = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(worker_payloads), mp_context=context
    ) as executor:
        futures = {
            executor.submit(_worker_main, payload): payload["worker_id"]
            for payload in worker_payloads
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            result = future.result()
            worker_results.append(result)
            print(
                f"[metasafe] worker {worker_id} finished: "
                f"{result['accepted_episodes']}/"
                f"{result['target_episodes']} accepted",
                flush=True,
            )
    worker_results.sort(key=lambda row: row["worker_id"])

    full_worker_summaries = [
        json.loads(
            (output_dir / "workers" / f"worker_{worker_id:02d}.json").read_text(
                encoding="utf-8"
            )
        )
        for worker_id in range(len(targets))
    ]
    attempts = [
        episode for worker in full_worker_summaries for episode in worker["episodes"]
    ]
    accepted_rows = [row for row in attempts if row["accepted"]]
    accepted_total = len(accepted_rows)
    attempt_total = len(attempts)
    attempt_success_rate = accepted_total / max(attempt_total, 1)
    scene_signatures = sorted({str(row["scene_signature"]) for row in attempts})
    initial_observation_hashes = sorted(
        {str(row["initial_observation_sha256"]) for row in attempts}
    )
    fixed_scene_verified = bool(
        len(scene_signatures) == 1
        and scene_signatures[0]
        and len(initial_observation_hashes) == 1
        and {int(row["scene_seed"]) for row in attempts} == {config.scene_seed}
    )

    shard_paths = [Path(row["shard"]) for row in worker_results if row["shard"]]
    dataset_path = output_dir / "metasafe_cbf_dataset.npz"
    uid_to_episode: dict[int, int] = {}
    combined: dict[str, np.ndarray] = {}
    if config.combine_shards and shard_paths:
        combined, uid_to_episode = _combine_shards(shard_paths, dataset_path)
    for row in accepted_rows:
        if row["episode_uid"] is not None and uid_to_episode:
            row["episode_id"] = uid_to_episode[int(row["episode_uid"])]
    accepted_rows.sort(
        key=lambda row: (
            row.get("episode_id", row["episode_uid"]),
            row["worker_id"],
        )
    )
    episodes_path = output_dir / "episodes.jsonl"
    episodes_path.write_text(
        "".join(
            json.dumps(row, allow_nan=True, default=_json_default) + "\n"
            for row in accepted_rows
        ),
        encoding="utf-8",
    )

    transition_count = int(
        combined.get(
            "obs",
            np.zeros((sum(row["transitions"] for row in worker_results),)),
        ).shape[0]
    )
    action_std = (
        np.std(combined["action"], axis=0).tolist()
        if combined
        else [float("nan"), float("nan")]
    )
    raw_action_std = (
        np.std(combined["raw_action"], axis=0).tolist()
        if combined
        else [float("nan"), float("nan")]
    )
    teacher_action_std = (
        np.std(combined["teacher_action"][combined["teacher_valid"]], axis=0).tolist()
        if combined and np.any(combined["teacher_valid"])
        else [float("nan"), float("nan")]
    )
    gate_failures = []
    if accepted_total != config.episodes:
        gate_failures.append(f"accepted {accepted_total}, expected {config.episodes}")
    if attempt_success_rate < config.minimum_attempt_success_rate:
        gate_failures.append(
            "attempt success rate "
            f"{attempt_success_rate:.6f} below "
            f"{config.minimum_attempt_success_rate:.6f}"
        )
    if not fixed_scene_verified:
        gate_failures.append("fixed-scene signature/initial observation mismatch")
    if any(float(row["environment_cost"]) > 0.0 for row in accepted_rows):
        gate_failures.append("primary dataset contains positive-cost episode")
    if any(bool(row["collision"]) for row in accepted_rows):
        gate_failures.append("primary dataset contains a collision episode")
    if any(bool(row["road_departure"]) for row in accepted_rows):
        gate_failures.append("primary dataset contains a road-departure episode")
    if any(int(row["steps"]) > config.max_episode_steps for row in accepted_rows):
        gate_failures.append("accepted episode exceeds the configured step limit")
    if any(
        not np.isfinite(float(row["min_physical_body_sat_margin"]))
        or float(row["min_physical_body_sat_margin"]) <= 0.0
        for row in accepted_rows
    ):
        gate_failures.append("accepted episode lacks a positive physical margin")
    speed_bin_counts = np.zeros((4,), dtype=np.int64)
    speed_bin_fractions = np.zeros((4,), dtype=np.float64)
    if (
        combined
        and config.action_semantics_version
        == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
    ):
        lower = np.asarray(combined["reachable_speed_lower"], dtype=np.float64)
        upper = np.asarray(combined["reachable_speed_upper"], dtype=np.float64)
        speed = np.asarray(combined["executed_target_speed"], dtype=np.float64)
        emergency = np.asarray(combined["emergency_stop"], dtype=bool)
        fraction = np.clip((speed - lower) / (upper - lower + 1e-6), 0.0, 1.0)
        speed_bins = np.minimum((fraction * 4.0).astype(np.int32), 3)
        speed_bin_counts = np.bincount(speed_bins[~emergency], minlength=4)
        speed_bin_fractions = speed_bin_counts / max(
            int(np.sum(speed_bin_counts)), 1
        )
        if np.any(speed_bin_counts == 0):
            gate_failures.append("one or more reachable-speed bins are empty")
        if speed_bin_fractions[3] < config.minimum_high_speed_bin_fraction:
            gate_failures.append(
                "highest reachable-speed bin is below configured minimum"
            )
        if speed_bin_fractions[3] > config.maximum_high_speed_bin_fraction:
            gate_failures.append(
                "highest reachable-speed bin exceeds configured maximum"
            )
        if speed_bin_fractions[2] < config.minimum_mid_high_bin_fraction:
            gate_failures.append(
                "mid-high reachable-speed bin is below configured minimum"
            )
        if (
            speed_bin_fractions[0] + speed_bin_fractions[1]
            < config.minimum_low_speed_bins_fraction
        ):
            gate_failures.append(
                "combined low-speed bins are below configured minimum"
            )

    episode_steps = np.asarray(
        [int(row["steps"]) for row in accepted_rows], dtype=np.int32
    )

    summary = {
        "format_version": 4,
        "environment": ("metadrive.envs.safe_metadrive_env.SafeMetaDriveEnv"),
        "environment_adapter": ("relax.env.drive.metasafe.FixedSeedMetaSafeEnv"),
        "collector": (
            "coherent ordinary-recovery Guardian"
            if config.behavior_policy == "ordinary_recovery"
            else "state-aligned guardian + hard CBF Flat NMPC"
        ),
        "behavior_policy": config.behavior_policy,
        "action_semantics_version": config.action_semantics_version,
        "episode_step_limit": config.max_episode_steps,
        "speed_sampling_profile": config.speed_sampling_profile,
        "speed_sampling_probabilities": {
            context: list(probabilities)
            for context, probabilities in SPEED_SAMPLING_PROFILES[
                config.speed_sampling_profile
            ].items()
        },
        "speed_sampling_within_bin_distributions": (
            FAST_V231_WITHIN_BIN_DISTRIBUTIONS
            if config.speed_sampling_profile == FAST_SPEED_SAMPLING_PROFILE
            else {
                "open": "uniform",
                "near": "uniform",
                "critical": "uniform",
            }
        ),
        "speed_sampling_categorical_scheduler": (
            "weighted_fair_credit_v1"
            if config.speed_sampling_profile == FAST_SPEED_SAMPLING_PROFILE
            else "iid_categorical"
        ),
        "exploration_hold_steps": config.exploration_hold_steps,
        "speed_bin_fraction_gate": {
            "minimum_high_speed_bin_fraction": (
                config.minimum_high_speed_bin_fraction
            ),
            "maximum_high_speed_bin_fraction": (
                config.maximum_high_speed_bin_fraction
            ),
            "minimum_mid_high_bin_fraction": (
                config.minimum_mid_high_bin_fraction
            ),
            "minimum_low_speed_bins_fraction": (
                config.minimum_low_speed_bins_fraction
            ),
        },
        "min_command_speed": config.min_command_speed,
        "max_command_speed": config.max_command_speed,
        "speed_mapping": (
            "state_reachable_speed_linear_v1"
            if config.action_semantics_version
            == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
            else "v=max_command_speed*(action_x+1)/2"
            if config.action_semantics_version == SPEED_COUPLED_ACTION_SEMANTICS
            else "fixed_internal_target_velocity"
        ),
        "reachable_speed_bin_counts": speed_bin_counts.astype(int).tolist(),
        "reachable_speed_bin_fractions": speed_bin_fractions.astype(float).tolist(),
        # Stable V2.3.1 artifact names; retain the explicit reachable-speed
        # aliases above for V2.3 analysis compatibility.
        "speed_bin_counts": speed_bin_counts.astype(int).tolist(),
        "speed_bin_fractions": speed_bin_fractions.astype(float).tolist(),
        "highest_speed_bin_within_70_percent": bool(
            speed_bin_fractions[3] <= 0.70
        ),
        "highest_speed_bin_within_configured_maximum": bool(
            speed_bin_fractions[3] <= config.maximum_high_speed_bin_fraction
        ),
        "post_reset_warmup_steps": config.post_reset_warmup_steps,
        "fixed_scene_seed": config.scene_seed,
        "fixed_scene_verified": fixed_scene_verified,
        "scene_signatures": scene_signatures,
        "initial_observation_hashes": initial_observation_hashes,
        "map_blocks": accepted_rows[0]["map_blocks"] if accepted_rows else [],
        "scene_descriptor": full_worker_summaries[0].get("scene_descriptor", {}),
        "obstacle_objects": (
            accepted_rows[0]["obstacle_objects"] if accepted_rows else 0
        ),
        "requested_episodes": config.episodes,
        "accepted_episodes": accepted_total,
        "attempts": attempt_total,
        "attempt_success_rate": attempt_success_rate,
        "accepted_episode_success_rate": (
            float(np.mean([row["success"] for row in accepted_rows]))
            if accepted_rows
            else 0.0
        ),
        "accepted_collision_rate": (
            float(np.mean([row["collision"] for row in accepted_rows]))
            if accepted_rows
            else 0.0
        ),
        "accepted_road_departure_rate": (
            float(np.mean([row["road_departure"] for row in accepted_rows]))
            if accepted_rows
            else 0.0
        ),
        "accepted_total_environment_cost": float(
            np.sum([row["environment_cost"] for row in accepted_rows])
        ),
        "transitions": transition_count,
        "observation_dim": (int(combined["obs"].shape[1]) if combined else None),
        "raw_action_std": raw_action_std,
        "executed_action_std": action_std,
        "teacher_action_std": teacher_action_std,
        "raw_target_speed_mean": (
            float(np.mean(combined["raw_target_speed"])) if combined else float("nan")
        ),
        "raw_target_speed_std": (
            float(np.std(combined["raw_target_speed"])) if combined else float("nan")
        ),
        "executed_target_speed_mean": (
            float(np.mean(combined["executed_target_speed"]))
            if combined
            else float("nan")
        ),
        "executed_target_speed_std": (
            float(np.std(combined["executed_target_speed"]))
            if combined
            else float("nan")
        ),
        "target_speed_p10": (
            float(np.percentile(combined["executed_target_speed"], 10))
            if combined
            else float("nan")
        ),
        "target_speed_p50": (
            float(np.percentile(combined["executed_target_speed"], 50))
            if combined
            else float("nan")
        ),
        "target_speed_p90": (
            float(np.percentile(combined["executed_target_speed"], 90))
            if combined
            else float("nan")
        ),
        "speed_command_delta_mean": (
            float(np.mean(combined["speed_command_delta"]))
            if combined
            else float("nan")
        ),
        "speed_only_intervention_rate": (
            float(np.mean(combined["speed_only_intervention"]))
            if combined
            else 0.0
        ),
        "actual_vehicle_speed_mean": (
            float(np.mean(combined["actual_vehicle_speed"]))
            if combined
            else float("nan")
        ),
        "emergency_stop_rate": (
            float(np.mean(combined["emergency_stop"])) if combined else 0.0
        ),
        "state_aligned_authority_audit": config.state_aligned_authority_audit,
        "recreate_environment_each_attempt": (
            config.recreate_environment_each_attempt
        ),
        "authority_query_rate": (
            float(np.mean(combined["authority_audit_queried"]))
            if combined
            else 0.0
        ),
        "authority_audit_failures": (
            int(np.count_nonzero(combined["authority_audit_failed"]))
            if combined
            else 0
        ),
        "high_level_projection_rate": (
            float(np.mean(combined["projection_used"])) if combined else 0.0
        ),
        "low_level_authority_rate": (
            float(np.mean(combined["low_level_cbf_authority"]))
            if combined
            else 0.0
        ),
        "complete_intervention_rate": (
            float(np.mean(combined["intervention"])) if combined else 0.0
        ),
        "actual_intervention_rate": (
            float(np.mean(combined["actual_control_modified"]))
            if combined
            else 0.0
        ),
        "audit_authority_rate": (
            float(np.mean(combined["audit_authority"])) if combined else 0.0
        ),
        "diagnostic_only_rate": (
            float(
                np.mean(
                    combined["audit_authority"]
                    & ~combined["actual_control_modified"]
                )
            )
            if combined
            else 0.0
        ),
        "inverse_teacher_rows": (
            int(np.count_nonzero(combined["teacher_source"] == 2))
            if combined
            else 0
        ),
        "inverse_teacher_failures": (
            int(np.count_nonzero(combined["inverse_teacher_failed"]))
            if combined
            else 0
        ),
        "proactive_teacher_rows": (
            int(np.count_nonzero(combined["teacher_source"] == 3))
            if combined
            else 0
        ),
        "proactive_teacher_activated_rows": (
            int(
                np.count_nonzero(
                    (combined["teacher_source"] == 3)
                    & (combined["teacher_weight"] > 0.0)
                )
            )
            if combined
            else 0
        ),
        "ordinary_recovery_teacher_rows": (
            int(np.count_nonzero(combined["ordinary_recovery_teacher"]))
            if combined
            else 0
        ),
        "proxy_pair_rows": (
            int(np.count_nonzero(combined["proxy_pair_valid"]))
            if combined
            else 0
        ),
        "mean_episode_steps": (
            float(np.mean(episode_steps))
            if episode_steps.size
            else 0.0
        ),
        "episode_steps_mean": (
            float(np.mean(episode_steps)) if episode_steps.size else 0.0
        ),
        "episode_steps_p50": (
            float(np.percentile(episode_steps, 50)) if episode_steps.size else 0.0
        ),
        "episode_steps_p90": (
            float(np.percentile(episode_steps, 90)) if episode_steps.size else 0.0
        ),
        "episode_steps_max": (
            int(np.max(episode_steps)) if episode_steps.size else 0
        ),
        "all_episodes_within_step_limit": bool(
            episode_steps.size
            and np.all(episode_steps <= config.max_episode_steps)
        ),
        "mean_projection_rate": (
            float(np.mean([row["projection_rate"] for row in accepted_rows]))
            if accepted_rows
            else 0.0
        ),
        "minimum_hard_cbf_margin": _finite_min(
            [row["min_hard_cbf_margin"] for row in accepted_rows]
        ),
        "minimum_physical_body_sat_margin": _finite_min(
            [row["min_physical_body_sat_margin"] for row in accepted_rows]
        ),
        "num_parallel_envs": len(targets),
        "worker_results": worker_results,
        "dataset": str(dataset_path) if dataset_path.exists() else None,
        "dataset_sha256": (_sha256(dataset_path) if dataset_path.exists() else None),
        "shards": [str(path) for path in shard_paths],
        "episodes_file": str(episodes_path),
        "wall_time_seconds": time.time() - started,
        "gate_passed": not gate_failures,
        "gate_failures": gate_failures,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    if gate_failures:
        raise RuntimeError(
            "MetaSafe CBF collection failed acceptance gate: "
            + "; ".join(gate_failures)
        )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = CollectionConfig(
        output_dir=str(args.output_dir),
        behavior_policy=args.behavior_policy,
        episodes=args.episodes,
        num_envs=args.num_envs,
        scene_seed=args.scene_seed,
        collector_seed=args.collector_seed,
        max_episode_steps=args.max_episode_steps,
        minimum_attempt_success_rate=args.minimum_attempt_success_rate,
        max_attempt_multiplier=args.max_attempt_multiplier,
        require_zero_cost=args.require_zero_cost,
        nominal_longitudinal=args.nominal_longitudinal,
        nominal_lateral=args.nominal_lateral,
        longitudinal_exploration_std=args.longitudinal_exploration_std,
        lateral_exploration_std=args.lateral_exploration_std,
        exploration_hold_steps=args.exploration_hold_steps,
        speed_sampling_profile=args.speed_sampling_profile,
        minimum_high_speed_bin_fraction=(
            args.minimum_high_speed_bin_fraction
        ),
        maximum_high_speed_bin_fraction=(
            args.maximum_high_speed_bin_fraction
        ),
        minimum_mid_high_bin_fraction=(
            args.minimum_mid_high_bin_fraction
        ),
        minimum_low_speed_bins_fraction=(
            args.minimum_low_speed_bins_fraction
        ),
        minimum_longitudinal_action=args.minimum_longitudinal_action,
        maximum_lateral_action=args.maximum_lateral_action,
        target_velocity=args.target_velocity,
        action_semantics_version=args.action_semantics_version,
        min_command_speed=args.min_command_speed,
        max_command_speed=args.max_command_speed,
        lookahead=args.lookahead,
        lateral_scale=args.lateral_scale,
        horizon=args.horizon,
        mpc_tracking_weight=args.mpc_tracking_weight,
        mpc_velocity_weight=args.mpc_velocity_weight,
        mpc_road_weight=args.mpc_road_weight,
        mpc_road_buffer=args.mpc_road_buffer,
        mpc_max_steer_rate=args.mpc_max_steer_rate,
        clearance=args.clearance,
        cbf_gamma=args.cbf_gamma,
        projection_reserve=args.projection_reserve,
        projection_kinematic_reserve=args.projection_kinematic_reserve,
        sqp_steps=args.sqp_steps,
        admm_steps=args.admm_steps,
        max_obstacles=args.max_obstacles,
        return_discount=args.return_discount,
        proxy_discount=args.proxy_discount,
        soft_pair_weight=args.soft_pair_weight,
        post_reset_warmup_steps=args.post_reset_warmup_steps,
        state_aligned_authority_audit=args.state_aligned_authority_audit,
        authority_query_hard_margin=args.authority_query_hard_margin,
        authority_trajectory_tolerance=args.authority_trajectory_tolerance,
        inverse_mpc_teacher=args.inverse_mpc_teacher,
        proactive_dagger_teacher=args.proactive_dagger_teacher,
        ordinary_recovery_use_action_history=(
            args.ordinary_recovery_use_action_history
        ),
        ordinary_recovery_bypass_sign=args.ordinary_recovery_bypass_sign,
        ordinary_recovery_prefer_inverse=(
            args.ordinary_recovery_prefer_inverse
        ),
        dagger_pre_steps=args.dagger_pre_steps,
        dagger_weight_decay=args.dagger_weight_decay,
        recreate_environment_each_attempt=(
            args.recreate_environment_each_attempt
        ),
        combine_shards=args.combine_shards,
        overwrite=args.overwrite,
    )
    result = collect(config)
    print(
        json.dumps(
            result,
            indent=2,
            allow_nan=True,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
