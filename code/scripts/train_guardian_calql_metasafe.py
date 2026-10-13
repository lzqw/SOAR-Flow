#!/usr/bin/env python3
"""Train Guardian-Calibrated Offline-to-Online Safe SAC on fixed MetaSafe.

The script consumes the high-level 300-episode CBF--MPC dataset directly.  It
does not reinterpret old steering/throttle PPO transitions as target actions.
Every split is made by complete episode, and every formal run records the JAX
backend so a CPU fallback cannot silently satisfy a requested GPU experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DATASET_SHA256 = "54aa277badbc04ba2fa6131c828aea31c4a43ffd68c4d76ac522146cff6b977a"
INTERVENTION_ACCOUNTING_VERSION = "active_cbf_v2"
INTERVENTION_ACCOUNTING_DEFINITION = (
    "high-level target modification, or CBF-on/off trajectory modification "
    "with a negative ordinary CBF-rate certificate; feasible solver-branch "
    "differences are audit-only"
)


@dataclass(frozen=True)
class MethodSpec:
    critic_mode: str | None
    train_proxy: bool
    offline_actor: bool
    online: bool
    critic_warmup: bool
    use_proxy_actor: bool
    use_trust_region: bool
    use_dagger: bool
    use_authorization_curriculum: bool = False
    offline_ratio_initial_override: float | None = None
    offline_ratio_final_override: float | None = None
    dagger_bc_only: bool = False
    use_q_anchor: bool = False
    dependency_proxy: bool = False
    use_awac: bool = False


METHODS = {
    "bc": MethodSpec(None, False, False, False, False, False, False, False),
    "bc_dagger": MethodSpec(
        None,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        dagger_bc_only=True,
    ),
    "cql": MethodSpec("cql", False, True, False, False, False, True, False),
    "calql": MethodSpec("calql", False, True, False, False, False, True, False),
    "calql_proxy": MethodSpec("calql", True, True, False, False, True, True, True),
    "full": MethodSpec("calql", True, True, True, True, True, True, True, True),
    "full_no_proxy": MethodSpec(
        "calql", False, True, True, True, False, True, True, True
    ),
    "full_no_warmup": MethodSpec(
        "calql", True, True, True, False, True, True, True, True
    ),
    "full_no_curriculum": MethodSpec(
        "calql", True, True, True, True, True, True, True, False
    ),
    "full_fixed_50_offline": MethodSpec(
        "calql",
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        0.50,
        0.50,
    ),
    "full_no_offline_replay": MethodSpec(
        "calql",
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        0.0,
        0.0,
    ),
    # Guardian-CalQL Policy Improvement V2 fixed-scene experiments.
    # Historical V2 names remain runnable for comparison.  V2.1 evaluates the
    # raw learner and relies on learning-rate/KL/action anchors rather than
    # interpolating network parameters.
    "v2_q_anchor": MethodSpec(
        critic_mode="calql",
        train_proxy=True,
        offline_actor=True,
        online=True,
        critic_warmup=True,
        use_proxy_actor=True,
        use_trust_region=True,
        use_dagger=True,
        use_q_anchor=True,
    ),
    "v2_trust": MethodSpec(
        critic_mode="calql",
        train_proxy=True,
        offline_actor=True,
        online=True,
        critic_warmup=True,
        use_proxy_actor=True,
        use_trust_region=True,
        use_dagger=True,
        use_q_anchor=True,
        dependency_proxy=True,
    ),
    "v2_awac": MethodSpec(
        critic_mode="calql",
        train_proxy=True,
        offline_actor=True,
        online=True,
        critic_warmup=True,
        use_proxy_actor=True,
        use_trust_region=False,
        use_dagger=True,
        use_q_anchor=True,
        dependency_proxy=True,
        use_awac=True,
    ),
    "v2_trust_awac": MethodSpec(
        critic_mode="calql",
        train_proxy=True,
        offline_actor=True,
        online=True,
        critic_warmup=True,
        use_proxy_actor=True,
        use_trust_region=True,
        use_dagger=True,
        use_q_anchor=True,
        dependency_proxy=True,
        use_awac=True,
    ),
    "v2_1": MethodSpec(
        critic_mode="calql",
        train_proxy=True,
        offline_actor=True,
        online=True,
        critic_warmup=True,
        use_proxy_actor=True,
        use_trust_region=True,
        use_dagger=True,
        use_authorization_curriculum=True,
        use_q_anchor=True,
        dependency_proxy=True,
        use_awac=True,
    ),
    "v2_2_simple": MethodSpec(
        critic_mode="calql",
        train_proxy=True,
        offline_actor=False,
        online=True,
        critic_warmup=True,
        use_proxy_actor=True,
        use_trust_region=True,
        use_dagger=False,
        use_authorization_curriculum=False,
        use_q_anchor=True,
        dependency_proxy=True,
        use_awac=False,
    ),
}


def parse_args(argv: Iterable[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "runs/metasafe_cbf/fixed_seed0_cbf_mpc_300ep_v1/" "metasafe_cbf_dataset.npz"
        ),
    )
    parser.add_argument("--expected-dataset-sha256", default=DATASET_SHA256)
    parser.add_argument("--output-root", type=Path, default=Path("runs/guardian_calql"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--online-initial-checkpoint",
        type=Path,
        default=None,
        help=(
            "load a complete Guardian-CalQL v3 checkpoint before any offline "
            "stage; set offline budgets to zero for an online-only continuation"
        ),
    )
    parser.add_argument(
        "--online-restart-from-loaded-champion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after loading a complete checkpoint, preserve its current Q/P and "
            "offline-Q reference but explicitly restart Learner, Champion, "
            "LastSafe, Actor optimizer, and Actor duals from its accepted "
            "Champion policy"
        ),
    )
    parser.add_argument(
        "--online-reset-loaded-proxy-dual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after loading a complete checkpoint, reset only its P-to-Actor "
            "Lagrange multiplier to --proxy-actor-weight; preserve Learner, "
            "Champion, LastSafe, Actor optimizer, Q/P, and offline-Q reference"
        ),
    )
    parser.add_argument(
        "--online-reset-loaded-actor-optimizers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after loading a complete checkpoint, reset only Learner, "
            "Champion, and LastSafe Actor optimizer moments; preserve all "
            "policy parameters, Q/P, duals, and offline-Q reference"
        ),
    )
    parser.add_argument(
        "--online-initial-policy-checkpoint",
        type=Path,
        default=None,
        help=(
            "install only the Actor before offline Q/P calibration; accepts "
            "compatible Guardian-CalQL or legacy JaxSafeSAC checkpoints"
        ),
    )
    parser.add_argument("--method", choices=tuple(METHODS), default="v2_1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-scene-seed", type=int, default=0)
    parser.add_argument("--action-semantics", default="target_point_fixed_speed_v1")
    parser.add_argument("--max-command-speed", type=float, default=4.8)
    parser.add_argument("--offline-scene-seed", type=int, default=0)
    parser.add_argument("--online-scene-seeds", default="")
    parser.add_argument("--retention-scene-seeds", default="0")
    parser.add_argument("--heldout-scene-seeds", default="")
    parser.add_argument("--scene-selection-manifest", type=Path, default=None)
    parser.add_argument(
        "--bootstrap-offline-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "V2.2 only: build the Seed-0 initial Champion before the "
            "predeclared seed scan; performs no online training or selection"
        ),
    )
    parser.add_argument(
        "--allow-fixed-speed-ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="V2.2 experiment B only; retains simplified losses with legacy fixed speed",
    )
    parser.add_argument(
        "--require-gpu", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--jax-memory-fraction", type=float, default=0.35)
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--proxy-discount", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--proxy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha-learning-rate", type=float, default=3e-4)
    parser.add_argument("--init-alpha", type=float, default=0.05)
    parser.add_argument("--calql-weight", type=float, default=1.0)
    parser.add_argument(
        "--online-critic-mode",
        choices=("td", "calql"),
        default="td",
        help=(
            "online Task-Q objective: TD is the non-conservative default; "
            "calql is an explicit conservative ablation"
        ),
    )
    parser.add_argument("--q-mc-weight", type=float, default=1.0)
    parser.add_argument("--q-mc-huber-delta", type=float, default=1.0)
    parser.add_argument("--q-lcb-beta", type=float, default=1.0)
    parser.add_argument("--q-anchor-weight-initial", type=float, default=1.0)
    parser.add_argument("--q-anchor-weight-final", type=float, default=0.05)
    parser.add_argument("--q-anchor-decay-steps", type=int, default=10_000)
    parser.add_argument("--soft-pair-weight", type=float, default=0.25)
    parser.add_argument("--soft-teacher-weight", type=float, default=0.25)
    parser.add_argument(
        "--bc-intervention-ratio",
        type=float,
        default=0.25,
        help=(
            "fraction of each BC batch reserved for nontrivial valid teacher "
            "rows (intervention corrections or activated pre-intervention labels)"
        ),
    )
    parser.add_argument("--bc-selection-intervention-weight", type=float, default=0.25)
    parser.add_argument(
        "--bc-canonical-maneuver",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--closed-loop-bc-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "for formal from-scratch runs, select the earliest BC validation "
            "checkpoint that passes the real Hybrid/Ordinary safety gate"
        ),
    )
    parser.add_argument("--proxy-pair-margin", type=float, default=0.5)
    parser.add_argument("--proxy-td-loss-weight", type=float, default=1.0)
    parser.add_argument("--proxy-td-target-min", type=float, default=0.0)
    parser.add_argument("--proxy-td-target-max", type=float, default=1.0)
    parser.add_argument("--proxy-regression-loss-weight", type=float, default=1.0)
    parser.add_argument("--proxy-pair-loss-weight", type=float, default=2.0)
    parser.add_argument("--proxy-l2-weight", type=float, default=1e-3)
    parser.add_argument("--proxy-min-value", type=float, default=0.0)
    parser.add_argument("--proxy-dependency-safe-threshold", type=float, default=0.25)
    parser.add_argument("--proxy-relative-margin", type=float, default=0.0)
    parser.add_argument("--proxy-corrected-weight", type=float, default=1.0)
    parser.add_argument("--proxy-actor-weight", type=float, default=0.1)
    parser.add_argument("--proxy-dual-learning-rate", type=float, default=1e-3)
    parser.add_argument("--proxy-target", type=float, default=0.02)
    parser.add_argument(
        "--proxy-actor-boundary-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "optionally restrict the Champion-relative P hinge to audited "
            "intervention/pair states"
        ),
    )
    parser.add_argument("--actor-sac-weight", type=float, default=1.0)
    parser.add_argument("--awac-beta", type=float, default=0.25)
    parser.add_argument("--awac-temperature", type=float, default=1.0)
    parser.add_argument("--awac-max-weight", type=float, default=20.0)
    parser.add_argument("--teacher-bc-weight", type=float, default=5.0)
    parser.add_argument("--champion-action-weight", type=float, default=1.0)
    parser.add_argument("--champion-boundary-action-weight", type=float, default=0.0)
    parser.add_argument("--champion-kl-weight", type=float, default=0.5)
    parser.add_argument("--champion-kl-min", type=float, default=0.0)
    parser.add_argument("--champion-kl-target", type=float, default=0.0075)
    parser.add_argument("--champion-kl-dual-learning-rate", type=float, default=0.05)
    parser.add_argument("--champion-kl-failure-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--champion-kl-validation-relax-multiplier",
        type=float,
        default=0.95,
    )
    parser.add_argument("--champion-kl-max", type=float, default=5.0)
    parser.add_argument("--trust-weight", type=float, default=10.0)
    parser.add_argument("--trust-relax-multiplier", type=float, default=0.95)
    parser.add_argument("--trust-tighten-multiplier", type=float, default=1.25)
    parser.add_argument("--trust-min", type=float, default=1.0)
    parser.add_argument("--trust-max", type=float, default=200.0)
    parser.add_argument(
        "--initial-policy-kl-weight",
        type=float,
        default=5.0,
        help="BC-reference KL used only by the limited offline Actor stage",
    )

    parser.add_argument("--bc-gradient-steps", type=int, default=30_000)
    parser.add_argument("--critic-gradient-steps", type=int, default=50_000)
    parser.add_argument("--proxy-gradient-steps", type=int, default=30_000)
    parser.add_argument("--offline-actor-gradient-steps", type=int, default=2_000)
    parser.add_argument("--validation-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--bc-min-gradient-steps", type=int, default=3_000)
    parser.add_argument("--critic-min-gradient-steps", type=int, default=10_000)
    parser.add_argument("--proxy-min-gradient-steps", type=int, default=5_000)

    parser.add_argument("--warmup-min-steps", type=int, default=5_000)
    parser.add_argument("--warmup-max-steps", type=int, default=15_000)
    parser.add_argument("--warmup-window", type=int, default=500)
    parser.add_argument("--warmup-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--online-steps", type=int, default=50_000)
    parser.add_argument("--online-buffer-size", type=int, default=200_000)
    parser.add_argument("--intervention-buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--offline-ratio-initial", type=float, default=0.50)
    parser.add_argument("--offline-ratio-final", type=float, default=0.25)
    parser.add_argument("--intervention-ratio", type=float, default=0.25)
    parser.add_argument(
        "--ordinary-champion-anchor-ratio",
        type=float,
        default=0.0,
        help=(
            "optional fraction of every online update batch reserved for a "
            "verified no-CBF Champion trajectory; this anchors the exact "
            "ordinary state-action manifold without scaling Actor outputs"
        ),
    )
    parser.add_argument(
        "--ordinary-champion-anchor-episodes",
        type=int,
        default=1,
        help="number of verified ordinary policy episodes cached as anchors",
    )
    parser.add_argument(
        "--ordinary-anchor-policy",
        choices=("champion", "learner"),
        default="champion",
        help=(
            "policy used to collect the verified ordinary anchor cache; "
            "'learner' requires an unambiguous full-checkpoint continuation"
        ),
    )
    parser.add_argument("--online-exploration-std", type=float, default=0.04)
    parser.add_argument("--dagger-pre-steps", type=int, default=25)
    parser.add_argument("--dagger-weight-decay", type=float, default=0.95)
    parser.add_argument(
        "--counterfactual-dagger-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "query the fixed [1,0] Guardian target at each visited state so "
            "pre-intervention labels stay aligned with their own ego frame"
        ),
    )
    parser.add_argument(
        "--counterfactual-dagger-query-hard-margin",
        type=float,
        default=50.0,
        help=(
            "only run the extra Guardian/ordinary distillation solve once "
            "the existing Actor-action shadow CBF margin is at or below "
            "this value; far-field labels remain the fixed [1,0] target"
        ),
    )
    parser.add_argument(
        "--cbf-trajectory-tolerance",
        type=float,
        default=0.08,
        help=(
            "minimum same-state CBF-on/off MPC trajectory RMSE used for the "
            "authority audit; it counts as actual low-level CBF authority "
            "only when the ordinary CBF-rate certificate is also negative"
        ),
    )

    parser.add_argument("--candidate-every", type=int, default=3_000)
    parser.add_argument("--candidate-eval-episodes", type=int, default=3)
    parser.add_argument(
        "--minimum-candidate-actor-action-delta",
        type=float,
        default=1e-5,
        help=(
            "minimum mean same-observation L2 action change from the current "
            "Champion required for an online candidate to be promotable"
        ),
    )
    parser.add_argument("--return-noninferiority", type=float, default=0.0)
    parser.add_argument("--intervention-regression", type=float, default=0.0)
    parser.add_argument(
        "--require-strict-intervention-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stop-after-first-promotion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="finish a formal run as soon as the first strictly better Champion passes",
    )
    parser.add_argument("--post-filter-proxy-warmup-steps", type=int, default=1_000)
    parser.add_argument("--ordinary-success-regression", type=float, default=0.0)
    parser.add_argument("--minimum-success-rate", type=float, default=1.0)
    parser.add_argument(
        "--online-return-nondegradation-tolerance", type=float, default=0.25
    )
    parser.add_argument(
        "--online-intervention-nondegradation-pp", type=float, default=0.25
    )
    parser.add_argument("--online-return-strict-improvement", type=float, default=0.50)
    parser.add_argument(
        "--online-intervention-strict-improvement-pp", type=float, default=0.50
    )
    parser.add_argument("--retention-return-tolerance", type=float, default=0.50)
    parser.add_argument(
        "--retention-intervention-tolerance-pp", type=float, default=0.50
    )
    parser.add_argument("--critic-rollback-factor", type=float, default=2.0)
    parser.add_argument(
        "--candidate-score-intervention-penalty", type=float, default=10.0
    )
    parser.add_argument("--candidate-score-safety-penalty", type=float, default=1_000.0)

    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--post-reset-warmup-steps", type=int, default=1)
    parser.add_argument(
        "--recreate-environment-each-episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "close and rebuild MetaDrive between episodes so simulator-local "
            "state cannot leak through reset into policy training/evaluation"
        ),
    )
    parser.add_argument("--lookahead", type=float, default=10.0)
    parser.add_argument("--lateral-scale", type=float, default=3.5)
    parser.add_argument("--target-velocity", type=float, default=4.8)
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
    parser.add_argument(
        "--authorization-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "slowly relax only the extra projection reserves after clean "
            "episodes and tighten them quickly after a rejected hard "
            "proposal, numerical failure, or physical safety event"
        ),
    )
    parser.add_argument("--projection-reserve-min", type=float, default=1e-4)
    parser.add_argument("--projection-reserve-max", type=float, default=0.30)
    parser.add_argument("--projection-reserve-relax-step", type=float, default=0.01)
    parser.add_argument("--projection-reserve-backoff-step", type=float, default=0.05)
    parser.add_argument("--projection-kinematic-reserve-min", type=float, default=1e-4)
    parser.add_argument("--projection-kinematic-reserve-max", type=float, default=0.08)
    parser.add_argument(
        "--projection-kinematic-reserve-relax-step", type=float, default=0.002
    )
    parser.add_argument(
        "--projection-kinematic-reserve-backoff-step", type=float, default=0.01
    )
    parser.add_argument("--curriculum-safe-window", type=int, default=3)
    parser.add_argument("--sqp-steps", type=int, default=12)
    parser.add_argument("--admm-steps", type=int, default=120)
    parser.add_argument("--max-obstacles", type=int, default=32)
    args = parser.parse_args(raw_argv)
    if args.method == "v2_2_simple":
        deleted_options = {
            "--q-mc-huber-delta",
            "--awac-temperature",
            "--awac-max-weight",
            "--proxy-pair-margin",
            "--proxy-td-target-min",
            "--proxy-td-target-max",
            "--proxy-regression-loss-weight",
            "--proxy-l2-weight",
            "--proxy-min-value",
            "--proxy-dependency-safe-threshold",
            "--proxy-target",
            "--proxy-actor-boundary-only",
            "--champion-boundary-action-weight",
            "--champion-kl-weight",
            "--champion-kl-min",
            "--champion-kl-target",
            "--champion-kl-dual-learning-rate",
            "--champion-kl-failure-multiplier",
            "--champion-kl-validation-relax-multiplier",
            "--champion-kl-max",
        }
        for option in sorted(deleted_options):
            if any(
                token == option or token.startswith(option + "=") for token in raw_argv
            ):
                parser.error(f"{option} is removed for --method v2_2_simple")
        removed = {
            "--q-mc-weight": "q_mc_weight",
            "--awac-beta": "awac_beta",
            "--proxy-td-loss-weight": "proxy_td_loss_weight",
            "--proxy-pair-loss-weight": "proxy_pair_loss_weight",
            "--teacher-bc-weight": "teacher_bc_weight",
            "--proxy-dual-learning-rate": "proxy_dual_learning_rate",
        }
        for option, attribute in removed.items():
            explicit = any(
                token == option or token.startswith(option + "=") for token in raw_argv
            )
            if explicit and float(getattr(args, attribute)) != 0.0:
                parser.error(f"{option} must be zero for --method v2_2_simple")
            setattr(args, attribute, 0.0)
        if (
            any(
                token == "--online-critic-mode"
                or token.startswith("--online-critic-mode=")
                for token in raw_argv
            )
            and args.online_critic_mode != "td"
        ):
            parser.error("V2.2 online critic mode must be td")
        args.online_critic_mode = "td"
        if args.offline_actor_gradient_steps != 0 and any(
            token == "--offline-actor-gradient-steps"
            or token.startswith("--offline-actor-gradient-steps=")
            for token in raw_argv
        ):
            parser.error(
                "--offline-actor-gradient-steps must be zero for V2.2; "
                "offline Actor initialization is BC only"
            )
        args.offline_actor_gradient_steps = 0
        if not any(
            token == "--action-semantics" or token.startswith("--action-semantics=")
            for token in raw_argv
        ):
            args.action_semantics = "target_point_speed_coupled_v1"
        if args.action_semantics != "target_point_speed_coupled_v1" and not (
            args.allow_fixed_speed_ablation
            and args.action_semantics == "target_point_fixed_speed_v1"
        ):
            parser.error("V2.2 requires target_point_speed_coupled_v1")
        if not any(
            token == "--offline-ratio-final"
            or token.startswith("--offline-ratio-final=")
            for token in raw_argv
        ):
            args.offline_ratio_final = 0.20
        if not any(
            token == "--expected-dataset-sha256"
            or token.startswith("--expected-dataset-sha256=")
            for token in raw_argv
        ):
            args.expected_dataset_sha256 = ""
        if args.scene_selection_manifest is None and not args.bootstrap_offline_only:
            parser.error("V2.2 requires --scene-selection-manifest")
        if args.bootstrap_offline_only:
            args.online_steps = 0
            args.candidate_every = 0
    if (
        args.online_restart_from_loaded_champion
        and args.online_initial_checkpoint is None
    ):
        parser.error(
            "--online-restart-from-loaded-champion requires "
            "--online-initial-checkpoint"
        )
    if (
        args.online_restart_from_loaded_champion
        and args.online_initial_policy_checkpoint is not None
    ):
        parser.error(
            "--online-restart-from-loaded-champion cannot be combined with "
            "--online-initial-policy-checkpoint"
        )
    if args.online_reset_loaded_proxy_dual and args.online_initial_checkpoint is None:
        parser.error(
            "--online-reset-loaded-proxy-dual requires " "--online-initial-checkpoint"
        )
    if args.online_reset_loaded_proxy_dual and args.online_restart_from_loaded_champion:
        parser.error(
            "--online-reset-loaded-proxy-dual cannot be combined with "
            "--online-restart-from-loaded-champion"
        )
    if (
        args.online_reset_loaded_proxy_dual
        and args.online_initial_policy_checkpoint is not None
    ):
        parser.error(
            "--online-reset-loaded-proxy-dual cannot be combined with "
            "--online-initial-policy-checkpoint"
        )
    if (
        args.online_reset_loaded_actor_optimizers
        and args.online_initial_checkpoint is None
    ):
        parser.error(
            "--online-reset-loaded-actor-optimizers requires "
            "--online-initial-checkpoint"
        )
    if (
        args.online_reset_loaded_actor_optimizers
        and args.online_restart_from_loaded_champion
    ):
        parser.error(
            "--online-reset-loaded-actor-optimizers cannot be combined with "
            "--online-restart-from-loaded-champion"
        )
    if (
        args.online_reset_loaded_actor_optimizers
        and args.online_initial_policy_checkpoint is not None
    ):
        parser.error(
            "--online-reset-loaded-actor-optimizers cannot be combined with "
            "--online-initial-policy-checkpoint"
        )

    integer_positive = (
        "batch_size",
        "validation_every",
        "log_every",
        "evaluation_batch_size",
        "online_buffer_size",
        "intervention_buffer_size",
        "policy_delay",
        "q_anchor_decay_steps",
        "max_episode_steps",
        "curriculum_safe_window",
        "sqp_steps",
        "admm_steps",
        "max_obstacles",
    )
    for name in integer_positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "bc_gradient_steps",
        "critic_gradient_steps",
        "proxy_gradient_steps",
        "offline_actor_gradient_steps",
        "warmup_min_steps",
        "warmup_max_steps",
        "online_steps",
        "candidate_every",
        "ordinary_champion_anchor_episodes",
        "early_stop_patience",
        "bc_min_gradient_steps",
        "critic_min_gradient_steps",
        "proxy_min_gradient_steps",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
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
    if args.post_reset_warmup_steps < 0:
        parser.error("--post-reset-warmup-steps must be nonnegative")
    if not np.isfinite(args.max_command_speed) or args.max_command_speed <= 0.0:
        parser.error("--max-command-speed must be positive and finite")
    if args.proxy_corrected_weight < 0.0:
        parser.error("--proxy-corrected-weight must be nonnegative")
    if args.method == "v2_2_simple":
        if not 0.0 < args.trust_relax_multiplier < 1.0:
            parser.error("--trust-relax-multiplier must be in (0, 1)")
        if args.trust_tighten_multiplier <= 1.0:
            parser.error("--trust-tighten-multiplier must be > 1")
        if not 0.0 <= args.trust_min <= args.trust_weight <= args.trust_max:
            parser.error("require 0 <= trust-min <= trust-weight <= trust-max")
        for name in (
            "online_return_nondegradation_tolerance",
            "online_intervention_nondegradation_pp",
            "online_return_strict_improvement",
            "online_intervention_strict_improvement_pp",
            "retention_return_tolerance",
            "retention_intervention_tolerance_pp",
        ):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
                parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.warmup_min_steps > args.warmup_max_steps:
        parser.error("warmup min steps cannot exceed max steps")
    if not 0.0 <= args.offline_ratio_final <= args.offline_ratio_initial <= 1.0:
        parser.error("require 0 <= final offline ratio <= initial ratio <= 1")
    if not 0.0 <= args.intervention_ratio < 1.0:
        parser.error("--intervention-ratio must be in [0, 1)")
    if not 0.0 <= args.ordinary_champion_anchor_ratio < 1.0:
        parser.error("--ordinary-champion-anchor-ratio must be in [0, 1)")
    if (
        args.ordinary_champion_anchor_ratio > 0.0
        and args.ordinary_champion_anchor_episodes < 1
    ):
        parser.error(
            "positive ordinary Champion anchor ratio requires at least one episode"
        )
    if args.ordinary_anchor_policy == "learner":
        if args.online_initial_checkpoint is None:
            parser.error(
                "--ordinary-anchor-policy learner requires "
                "--online-initial-checkpoint"
            )
        if args.online_restart_from_loaded_champion:
            parser.error(
                "--ordinary-anchor-policy learner cannot be combined with "
                "--online-restart-from-loaded-champion"
            )
        if args.online_initial_policy_checkpoint is not None:
            parser.error(
                "--ordinary-anchor-policy learner cannot be combined with "
                "--online-initial-policy-checkpoint"
            )
    if not 0.0 <= args.bc_intervention_ratio < 1.0:
        parser.error("--bc-intervention-ratio must be in [0, 1)")
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        parser.error("--minimum-success-rate must be in [0, 1]")
    if args.soft_teacher_weight < 0.0:
        parser.error("--soft-teacher-weight must be nonnegative")
    if args.bc_selection_intervention_weight < 0.0:
        parser.error("--bc-selection-intervention-weight must be nonnegative")
    for name in (
        "proxy_actor_weight",
        "actor_sac_weight",
        "teacher_bc_weight",
        "champion_action_weight",
        "champion_boundary_action_weight",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0.0 <= args.champion_kl_min <= args.champion_kl_weight:
        parser.error("require 0 <= champion KL min <= initial weight")
    if args.champion_kl_target < 0.0:
        parser.error("--champion-kl-target must be nonnegative")
    if args.champion_kl_dual_learning_rate < 0.0:
        parser.error("--champion-kl-dual-learning-rate must be nonnegative")
    if args.champion_kl_failure_multiplier < 1.0:
        parser.error("--champion-kl-failure-multiplier must be >= 1")
    if not 0.0 <= args.champion_kl_validation_relax_multiplier <= 1.0:
        parser.error("--champion-kl-validation-relax-multiplier must be in [0, 1]")
    if args.champion_kl_max < args.champion_kl_weight:
        parser.error("--champion-kl-max must be >= initial weight")
    if not 0.0 < args.dagger_weight_decay <= 1.0:
        parser.error("--dagger-weight-decay must be in (0, 1]")
    if args.cbf_trajectory_tolerance < 0.0:
        parser.error("--cbf-trajectory-tolerance must be nonnegative")
    if args.minimum_candidate_actor_action_delta < 0.0:
        parser.error("--minimum-candidate-actor-action-delta must be nonnegative")
    if not (0.0 <= args.q_anchor_weight_final <= args.q_anchor_weight_initial):
        parser.error("require 0 <= Q anchor final weight <= Q anchor initial weight")
    if args.proxy_regression_loss_weight < 0.0:
        parser.error("--proxy-regression-loss-weight must be nonnegative")
    if args.proxy_td_loss_weight < 0.0:
        parser.error("--proxy-td-loss-weight must be nonnegative")
    if args.proxy_td_target_max < args.proxy_td_target_min:
        parser.error("--proxy-td-target-max must be >= --proxy-td-target-min")
    if args.proxy_relative_margin < 0.0:
        parser.error("--proxy-relative-margin must be nonnegative")
    if args.proxy_dependency_safe_threshold < 0.0:
        parser.error("--proxy-dependency-safe-threshold must be nonnegative")
    if args.q_mc_weight < 0.0:
        parser.error("--q-mc-weight must be nonnegative")
    if args.q_mc_huber_delta <= 0.0:
        parser.error("--q-mc-huber-delta must be positive")
    if args.awac_beta < 0.0:
        parser.error("--awac-beta must be nonnegative")
    if args.awac_temperature <= 0.0:
        parser.error("--awac-temperature must be positive")
    if args.awac_max_weight < 1.0:
        parser.error("--awac-max-weight must be >= 1")
    for name in (
        "candidate_score_intervention_penalty",
        "candidate_score_safety_penalty",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.post_filter_proxy_warmup_steps < 0:
        parser.error("--post-filter-proxy-warmup-steps must be nonnegative")
    if args.counterfactual_dagger_query_hard_margin < 0.0:
        parser.error("--counterfactual-dagger-query-hard-margin must be nonnegative")
    for prefix, initial in (
        ("projection_reserve", args.projection_reserve),
        (
            "projection_kinematic_reserve",
            args.projection_kinematic_reserve,
        ),
    ):
        minimum = float(getattr(args, f"{prefix}_min"))
        maximum = float(getattr(args, f"{prefix}_max"))
        relax_step = float(getattr(args, f"{prefix}_relax_step"))
        backoff_step = float(getattr(args, f"{prefix}_backoff_step"))
        if not 0.0 <= minimum <= float(initial) <= maximum:
            parser.error(
                f"require 0 <= {prefix.replace('_', '-')} min <= initial <= max"
            )
        if relax_step <= 0.0 or backoff_step <= relax_step:
            parser.error(
                f"{prefix.replace('_', '-')} backoff must be larger than relax step"
            )
    if args.smoke:
        if (
            args.online_initial_checkpoint is None
            and args.online_initial_policy_checkpoint is None
        ):
            args.hidden_sizes = "32,32"
        args.batch_size = min(args.batch_size, 32)
        args.bc_gradient_steps = min(args.bc_gradient_steps, 8)
        args.critic_gradient_steps = min(args.critic_gradient_steps, 8)
        args.proxy_gradient_steps = min(args.proxy_gradient_steps, 8)
        args.offline_actor_gradient_steps = min(args.offline_actor_gradient_steps, 4)
        args.warmup_min_steps = 0
        args.warmup_max_steps = min(args.warmup_max_steps, 8)
        args.online_steps = min(args.online_steps, 8)
        args.learning_starts = min(args.learning_starts, 4)
        args.dagger_pre_steps = min(args.dagger_pre_steps, 2)
        args.candidate_every = 0
        args.max_episode_steps = min(args.max_episode_steps, 24)
        args.sqp_steps = min(args.sqp_steps, 2)
        args.admm_steps = min(args.admm_steps, 8)
        args.validation_every = min(args.validation_every, 4)
        args.log_every = min(args.log_every, 2)
        # Evaluation is read-only and benefits from large batches even in a
        # smoke run over the full 333k-transition archive.  Shrinking this to
        # 64 made a code-path check slower than the actual tiny updates.
        args.evaluation_batch_size = min(args.evaluation_batch_size, 4096)
        args.bc_min_gradient_steps = 0
        args.critic_min_gradient_steps = 0
        args.proxy_min_gradient_steps = 0
    return args


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _actor_action_for_mpc(action) -> np.ndarray:
    """Preserve the Actor action semantics at the MPC boundary.

    Closed-loop deployment must not hide a post-training lateral gain.  MPC
    feasibility and tracking trade-offs belong in the MPC objective itself;
    this boundary only enforces the declared normalized action bounds.
    """

    return np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0).astype(
        np.float32
    )


def _json_sanitize(value):
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_sanitize(payload),
            indent=2,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _json_sanitize(payload),
                default=_json_default,
                allow_nan=False,
            )
            + "\n"
        )


def _finite(value: float, fallback: float = 0.0) -> float:
    value = float(value)
    return value if np.isfinite(value) else float(fallback)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    positive = int(np.count_nonzero(labels))
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = float(np.sum(ranks[labels]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    finite = np.isfinite(first) & np.isfinite(second)
    first, second = first[finite], second[finite]
    if first.size < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _ranking_accuracy(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    max_pairs: int = 100_000,
) -> float:
    """Deterministic pairwise ranking accuracy for Q_R versus Monte-Carlo RTG."""

    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[finite], target[finite]
    if prediction.size < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    count = min(int(max_pairs), max(prediction.size * 2, 1))
    first = rng.integers(0, prediction.size, size=count)
    second = rng.integers(0, prediction.size, size=count)
    informative = target[first] != target[second]
    if not np.any(informative):
        return float("nan")
    predicted_delta = prediction[first[informative]] - prediction[second[informative]]
    target_delta = target[first[informative]] - target[second[informative]]
    correct = np.sign(predicted_delta) == np.sign(target_delta)
    ties = predicted_delta == 0.0
    return float(np.mean(correct.astype(np.float64) + 0.5 * ties))


def _policy_hash(policy_params) -> str:
    import jax

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(jax.device_get(policy_params)):
        array = np.asarray(leaf)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actor_width_expansion_compatible(candidate_shapes, reference_shapes) -> bool:
    """Accept widening of only the two hidden layers in the policy MLP."""

    if len(candidate_shapes) != 6 or len(reference_shapes) != 6:
        return False
    source = [tuple(shape) for shape in candidate_shapes]
    target = [tuple(shape) for shape in reference_shapes]
    if tuple(map(len, source)) != (1, 2, 1, 2, 1, 2):
        return False
    if tuple(map(len, target)) != (1, 2, 1, 2, 1, 2):
        return False
    source_h1, source_h2 = source[0][0], source[2][0]
    target_h1, target_h2 = target[0][0], target[2][0]
    source_output, target_output = source[4][0], target[4][0]
    return bool(
        target_h1 >= source_h1
        and target_h2 >= source_h2
        and source[1] == (target[1][0], source_h1)
        and target[1] == (target[1][0], target_h1)
        and source[3] == (source_h1, source_h2)
        and target[3] == (target_h1, target_h2)
        and source[4] == target[4]
        and source[5] == (source_h2, source_output)
        and target[5] == (target_h2, target_output)
        and source_output == target_output
    )


def _load_compatible_actor_policy(
    path: Path, reference_policy, *, allow_width_expansion: bool = False
):
    """Load an Actor tree, optionally widening only hidden layers by zero-pad."""

    import jax
    import jax.numpy as jnp

    with path.open("rb") as stream:
        payload = pickle.load(stream)
    candidate = None
    source_kind = None
    if isinstance(payload, dict) and "champion_policy" in payload:
        candidate = payload["champion_policy"]
        source_kind = "guardian_calql_v6_champion_policy"
    if candidate is None and isinstance(payload, dict) and "learner_state" in payload:
        state = payload["learner_state"]
        params = getattr(state, "params", None)
        candidate = getattr(params, "policy", None)
        if candidate is not None:
            source_kind = "guardian_calql_v6_learner_state"
    if isinstance(payload, dict) and "state" in payload:
        state = payload["state"]
        params = getattr(state, "params", None)
        candidate = getattr(params, "policy", None)
        if candidate is not None:
            source_kind = type(state).__name__
    if candidate is None and isinstance(payload, dict):
        candidate = payload.get("champion_policy_params")
        if candidate is not None:
            source_kind = "champion_policy_params"
    if candidate is None and isinstance(payload, dict):
        state = payload.get("champion_state")
        params = getattr(state, "params", None)
        candidate = getattr(params, "policy", None)
        if candidate is not None:
            source_kind = "champion_state"
    if candidate is None and isinstance(payload, dict):
        if any(str(key).startswith("policy_net/") for key in payload):
            candidate = payload
            source_kind = "raw_policy_tree"
    if candidate is None:
        raise ValueError(f"checkpoint {path} does not contain an Actor policy")

    reference_leaves, reference_structure = jax.tree_util.tree_flatten(reference_policy)
    candidate_leaves, candidate_structure = jax.tree_util.tree_flatten(candidate)
    if candidate_structure != reference_structure:
        raise ValueError(
            "imported Actor parameter structure does not match current Actor"
        )
    candidate_shapes = [
        tuple(np.asarray(jax.device_get(value)).shape) for value in candidate_leaves
    ]
    reference_shapes = [tuple(value.shape) for value in reference_leaves]
    shapes_match = candidate_shapes == reference_shapes
    width_expansion = bool(
        not shapes_match
        and allow_width_expansion
        and _actor_width_expansion_compatible(candidate_shapes, reference_shapes)
    )
    if not shapes_match and not width_expansion:
        mismatch = next(
            index
            for index, (source_shape, target_shape) in enumerate(
                zip(candidate_shapes, reference_shapes, strict=True)
            )
            if source_shape != target_shape
        )
        raise ValueError(
            f"imported Actor leaf {mismatch} has shape "
            f"{candidate_shapes[mismatch]}, expected {reference_shapes[mismatch]}"
        )
    converted = []
    for index, (reference, value) in enumerate(
        zip(reference_leaves, candidate_leaves, strict=True)
    ):
        array = np.asarray(jax.device_get(value))
        if not np.all(np.isfinite(array)):
            raise ValueError(f"imported Actor leaf {index} is non-finite")
        if width_expansion and tuple(array.shape) != tuple(reference.shape):
            expanded = np.zeros(tuple(reference.shape), dtype=array.dtype)
            expanded[tuple(slice(0, size) for size in array.shape)] = array
            array = expanded
        converted.append(jnp.asarray(array, dtype=reference.dtype))
    if width_expansion:
        source_kind = f"{source_kind}+zero_padded_hidden_width_expansion"
    return jax.tree_util.tree_unflatten(reference_structure, converted), source_kind


def _copy_tree(tree):
    import jax

    return jax.tree_util.tree_map(
        lambda value: np.array(value, copy=True), jax.device_get(tree)
    )


def _evaluate_offline(
    algorithm,
    dataset,
    split: str,
    *,
    batch_size: int,
    use_champion: bool = False,
) -> dict:
    q_values = []
    q_policy_values = []
    q_champion_values = []
    q_guardian_values = []
    q_disagreements = []
    returns = []
    return_valid_masks = []
    executed_actions = []
    actor_actions = []
    teachers = []
    teacher_weights = []
    intervention_masks = []
    teacher_focus_masks = []
    canonical_masks = []
    proxy_policy = []
    proxy_champion = []
    proxy_raw = []
    proxy_corrected = []
    pair_valid = []
    hard_intervention_masks = []
    soft_intervention_masks = []
    proposal_evaluated_masks = []
    future_labels = []
    from relax.offline.metasafe_guardian_dataset import future_intervention_label

    all_future = future_intervention_label(
        dataset.fields["intervention"], dataset.episode_id, horizon=20
    )
    # ``iter_split`` yields only policy-decision rows.  Use the identically
    # filtered index sequence so reset-time rows cannot shift future-label or
    # mirrored-manoeuvre metrics by one frame per episode.
    split_indices = dataset.split_transition_indices(split, decisions_only=True)
    offset = 0
    for batch in dataset.iter_split(split, batch_size=batch_size, to_jax=False):
        obs = np.asarray(batch.obs)
        action = np.asarray(batch.action)
        raw_action = np.asarray(batch.raw_action)
        actor_action = (
            algorithm.get_deterministic_champion_action(obs)
            if use_champion
            else algorithm.get_deterministic_action(obs)
        )
        champion_action = algorithm.get_deterministic_champion_action(obs)
        q1, q2 = algorithm.task_q(obs, action)
        q_values.append(np.minimum(q1, q2))
        q_disagreements.append(np.abs(q1 - q2))
        policy_q1, policy_q2 = algorithm.task_q(obs, actor_action)
        q_policy_values.append(
            0.5 * (policy_q1 + policy_q2)
            - algorithm.config.q_lcb_beta * 0.5 * np.abs(policy_q1 - policy_q2)
        )
        champion_q1, champion_q2 = algorithm.task_q(obs, champion_action)
        q_champion_values.append(
            0.5 * (champion_q1 + champion_q2)
            - algorithm.config.q_lcb_beta * 0.5 * np.abs(champion_q1 - champion_q2)
        )
        guardian_q1, guardian_q2 = algorithm.task_q(
            obs, np.asarray(batch.teacher_action)
        )
        q_guardian_values.append(np.minimum(guardian_q1, guardian_q2))
        returns.append(np.asarray(batch.behavior_return_to_go))
        return_valid_masks.append(np.asarray(batch.behavior_return_valid, dtype=bool))
        executed_actions.append(action)
        actor_actions.append(actor_action)
        teachers.append(np.asarray(batch.teacher_action))
        teacher_weights.append(
            np.asarray(batch.teacher_valid, np.float32)
            * np.asarray(batch.teacher_weight, np.float32)
        )
        intervention_masks.append(np.asarray(batch.intervention, dtype=bool))
        proxy_policy.append(algorithm.proxy_value(obs, actor_action))
        proxy_champion.append(algorithm.proxy_value(obs, champion_action))
        proxy_raw.append(algorithm.proxy_value(obs, raw_action))
        proxy_corrected.append(
            algorithm.proxy_value(obs, np.asarray(batch.proxy_corrected_action))
        )
        pair_valid.append(np.asarray(batch.proxy_pair_valid, dtype=bool))
        hard_intervention_masks.append(np.asarray(batch.intervention_hard, dtype=bool))
        soft_intervention_masks.append(np.asarray(batch.intervention_soft, dtype=bool))
        proposal_evaluated_masks.append(
            np.asarray(batch.proposal_evaluated, dtype=bool)
        )
        count = obs.shape[0]
        current_indices = split_indices[offset : offset + count]
        future_labels.append(all_future[current_indices])
        canonical_masks.append(dataset.canonical_transition_mask[current_indices])
        teacher_focus_masks.append(dataset.teacher_focus_mask[current_indices])
        offset += count
    q_values = np.concatenate(q_values)
    q_policy_values = np.concatenate(q_policy_values)
    q_champion_values = np.concatenate(q_champion_values)
    q_guardian_values = np.concatenate(q_guardian_values)
    q_disagreements = np.concatenate(q_disagreements)
    returns = np.concatenate(returns)
    return_valid_masks = np.concatenate(return_valid_masks)
    executed_actions = np.concatenate(executed_actions)
    actor_actions = np.concatenate(actor_actions)
    teachers = np.concatenate(teachers)
    teacher_weights = np.concatenate(teacher_weights)
    intervention_masks = np.concatenate(intervention_masks)
    teacher_focus_masks = np.concatenate(teacher_focus_masks)
    canonical_masks = np.concatenate(canonical_masks)
    proxy_policy = np.concatenate(proxy_policy)
    proxy_champion = np.concatenate(proxy_champion)
    proxy_raw = np.concatenate(proxy_raw)
    proxy_corrected = np.concatenate(proxy_corrected)
    pair_valid = np.concatenate(pair_valid)
    hard_intervention_masks = np.concatenate(hard_intervention_masks)
    soft_intervention_masks = np.concatenate(soft_intervention_masks)
    proposal_evaluated_masks = np.concatenate(proposal_evaluated_masks)
    future_labels = np.concatenate(future_labels)
    teacher_error = np.sum(np.square(actor_actions - teachers), axis=-1)
    bc_mse = float(
        np.sum(teacher_error * teacher_weights)
        / max(float(np.sum(teacher_weights)), 1.0)
    )
    positive_teacher = teacher_weights > 0.0
    pass_mask = ~intervention_masks & positive_teacher
    teacher_focus_mask = positive_teacher & teacher_focus_masks
    bc_mse_pass = (
        float(np.mean(teacher_error[pass_mask])) if np.any(pass_mask) else float("nan")
    )
    bc_mse_intervention = (
        float(np.mean(teacher_error[intervention_masks]))
        if np.any(intervention_masks)
        else float("nan")
    )
    bc_mse_teacher_focus = (
        float(
            np.sum(
                teacher_error[teacher_focus_mask] * teacher_weights[teacher_focus_mask]
            )
            / max(float(np.sum(teacher_weights[teacher_focus_mask])), 1.0)
        )
        if np.any(teacher_focus_mask)
        else float("nan")
    )
    canonical_pass = canonical_masks & pass_mask
    canonical_intervention = canonical_masks & intervention_masks
    canonical_teacher_focus = canonical_masks & teacher_focus_mask
    bc_mse_canonical_actor_controlled = (
        float(np.mean(teacher_error[canonical_pass]))
        if np.any(canonical_pass)
        else float("nan")
    )
    bc_mse_canonical_intervention = (
        float(np.mean(teacher_error[canonical_intervention]))
        if np.any(canonical_intervention)
        else float("nan")
    )
    bc_mse_canonical_teacher_focus = (
        float(
            np.sum(
                teacher_error[canonical_teacher_focus]
                * teacher_weights[canonical_teacher_focus]
            )
            / max(
                float(np.sum(teacher_weights[canonical_teacher_focus])),
                1.0,
            )
        )
        if np.any(canonical_teacher_focus)
        else float("nan")
    )
    dependency_proxy = algorithm.config.proxy_objective == "dependency_regression"
    pair_correct = (
        proxy_raw > proxy_corrected if dependency_proxy else proxy_corrected > proxy_raw
    )
    pair_accuracy = (
        float(np.mean(pair_correct[pair_valid])) if np.any(pair_valid) else float("nan")
    )
    dependency_labels = hard_intervention_masks.astype(
        np.float32
    ) + 0.5 * soft_intervention_masks.astype(np.float32)
    regression_mask = proposal_evaluated_masks
    proxy_regression_mae = (
        float(
            np.mean(
                np.abs(proxy_raw[regression_mask] - dependency_labels[regression_mask])
            )
        )
        if np.any(regression_mask)
        else float("nan")
    )
    # Both proxy objectives expose a common "larger means more dependence"
    # diagnostic score even though the legacy signed P has the opposite sign.
    proxy_policy_risk = proxy_policy if dependency_proxy else -proxy_policy
    proxy_raw_risk = proxy_raw if dependency_proxy else -proxy_raw
    intervention_probability = 1.0 / (
        1.0 + np.exp(np.clip(-proxy_policy_risk, -30.0, 30.0))
    )
    guardian_valid = teacher_weights > 0.0
    q_policy_guardian_delta = (
        float(
            np.sum(
                (q_policy_values - q_guardian_values)[guardian_valid]
                * teacher_weights[guardian_valid]
            )
            / max(float(np.sum(teacher_weights[guardian_valid])), 1.0)
        )
        if np.any(guardian_valid)
        else float("nan")
    )
    calibration_mask = return_valid_masks & np.isfinite(returns) & np.isfinite(q_values)
    calibration_q = q_values[calibration_mask]
    calibration_returns = returns[calibration_mask]
    policy_advantage = q_policy_values[calibration_mask] - calibration_returns
    q_policy_minus_champion = q_policy_values - q_champion_values
    p_policy_minus_champion = proxy_policy - proxy_champion
    p_improved = (
        p_policy_minus_champion <= 0.0
        if dependency_proxy
        else p_policy_minus_champion >= 0.0
    )
    return {
        "split": split,
        "use_champion": use_champion,
        "samples": int(q_values.size),
        "bc_mse": bc_mse,
        "bc_mse_actor_controlled": bc_mse_pass,
        "bc_mse_intervention": bc_mse_intervention,
        "bc_mse_teacher_focus": bc_mse_teacher_focus,
        "bc_mse_canonical_actor_controlled": bc_mse_canonical_actor_controlled,
        "bc_mse_canonical_intervention": bc_mse_canonical_intervention,
        "bc_mse_canonical_teacher_focus": bc_mse_canonical_teacher_focus,
        "behavior_return_valid_fraction": float(np.mean(return_valid_masks)),
        "q_calibration_mae": (
            float(np.mean(np.abs(calibration_q - calibration_returns)))
            if calibration_q.size
            else float("nan")
        ),
        "q_calibration_bias": (
            float(np.mean(calibration_q - calibration_returns))
            if calibration_q.size
            else float("nan")
        ),
        "q_mc_correlation": _correlation(calibration_q, calibration_returns),
        "q_return_correlation": _correlation(calibration_q, calibration_returns),
        "advantage_ranking_accuracy": _ranking_accuracy(
            calibration_q, calibration_returns
        ),
        "policy_advantage_mean": (
            float(np.mean(policy_advantage)) if policy_advantage.size else float("nan")
        ),
        "policy_advantage_positive_fraction": (
            float(np.mean(policy_advantage > 0.0))
            if policy_advantage.size
            else float("nan")
        ),
        "q_policy_minus_guardian": q_policy_guardian_delta,
        "q_policy_minus_champion": float(np.mean(q_policy_minus_champion)),
        "q_disagreement": float(np.mean(q_disagreements)),
        "proxy_regression_mae": proxy_regression_mae,
        "proxy_pair_accuracy": pair_accuracy,
        "proxy_future_intervention_auc_h20": _binary_auc(
            future_labels, proxy_policy_risk
        ),
        "proxy_future_intervention_brier_h20": float(
            np.mean(np.square(intervention_probability - future_labels.astype(float)))
        ),
        "future_intervention_rate_h20": float(np.mean(future_labels)),
        "proxy_hard_intervention_auc": _binary_auc(
            hard_intervention_masks, proxy_raw_risk
        ),
        "proxy_policy_mean": float(np.mean(proxy_policy)),
        "proxy_champion_mean": float(np.mean(proxy_champion)),
        "p_policy_minus_champion": float(np.mean(p_policy_minus_champion)),
        "p_improvement_fraction": float(np.mean(p_improved)),
        "proxy_policy_safe_fraction": float(
            np.mean(
                proxy_policy <= algorithm.config.proxy_dependency_safe_threshold
                if dependency_proxy
                else proxy_policy >= algorithm.config.proxy_min_value
            )
        ),
        "actual_intervention_rate": float(np.mean(intervention_masks)),
    }


def _make_environment(
    args,
    mode: str,
    *,
    projection_reserve: float | None = None,
    projection_kinematic_reserve: float | None = None,
):
    from relax.env.drive.metasafe import make_metasafe_cbf_mpc_env

    if mode not in ("hybrid", "ordinary"):
        raise ValueError(f"unknown evaluation mode: {mode}")
    protected = mode == "hybrid"
    return make_metasafe_cbf_mpc_env(
        fixed_seed=args.fixed_scene_seed,
        use_render=False,
        action_semantics_version=getattr(
            args,
            "action_semantics",
            getattr(args, "action_semantics_version", "target_point_fixed_speed_v1"),
        ),
        max_command_speed=float(
            getattr(args, "max_command_speed", args.target_velocity)
        ),
        lookahead=args.lookahead,
        lateral_scale=args.lateral_scale,
        target_velocity=args.target_velocity,
        horizon=args.horizon,
        tracking_weight=getattr(args, "mpc_tracking_weight", 6.0),
        velocity_weight=getattr(args, "mpc_velocity_weight", 30.0),
        road_weight=getattr(args, "mpc_road_weight", 100.0),
        road_buffer=getattr(args, "mpc_road_buffer", 1.5),
        max_steer_rate=getattr(args, "mpc_max_steer_rate", 0.1),
        clearance=args.clearance,
        cbf_gamma=args.cbf_gamma,
        projection_reserve=(
            args.projection_reserve
            if projection_reserve is None
            else projection_reserve
        ),
        projection_kinematic_reserve=(
            args.projection_kinematic_reserve
            if projection_kinematic_reserve is None
            else projection_kinematic_reserve
        ),
        sqp_steps=args.sqp_steps,
        admm_steps=args.admm_steps,
        max_obstacles=args.max_obstacles,
        cbf_enable=protected,
        terminal_projection=protected,
        post_reset_warmup_steps=args.post_reset_warmup_steps,
    )


def _minimum_finite(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.min(finite)) if finite.size else float("nan")


def _evaluate_policy(
    algorithm,
    args,
    *,
    mode: str,
    use_champion: bool,
    episodes: int,
    projection_reserve: float | None = None,
    projection_kinematic_reserve: float | None = None,
) -> dict[str, Any]:
    env = None
    rows = []
    try:
        for episode in range(episodes):
            if env is None:
                env = _make_environment(
                    args,
                    mode,
                    projection_reserve=projection_reserve,
                    projection_kinematic_reserve=(projection_kinematic_reserve),
                )
            observation, reset_info = env.reset(seed=args.fixed_scene_seed)
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            task_return = 0.0
            total_cost = float(reset_info.get("post_reset_warmup_cost", 0.0))
            collision = bool(reset_info.get("post_reset_warmup_collision", False))
            road_departure = bool(
                reset_info.get("post_reset_warmup_road_departure", False)
            )
            success = False
            interventions = 0
            audit_authorities = 0
            diagnostic_only_steps = 0
            high_level_projections = 0
            actual_low_level_modifications = 0
            solver_finite = 0
            projection_norms = []
            trajectory_deltas = []
            actor_action_deltas = []
            hard_margins = []
            ordinary_audit_cbf_margins = []
            physical_margins = []
            raw_target_speeds = []
            executed_target_speeds = []
            actual_vehicle_speeds = []
            speed_command_deltas = []
            speed_only_interventions = 0
            braking_commands = 0
            route_completion = 0.0
            try:
                env_config = env.unwrapped.config
                control_interval = float(env_config.get("decision_repeat", 0)) * float(
                    env_config.get("physics_world_step_size", 0.0)
                )
            except Exception:
                control_interval = 0.0
            if not np.isfinite(control_interval) or control_interval <= 0.0:
                control_interval = 0.1
            cbf_physical_confusion = {
                "reject_and_physical_failure": 0,
                "reject_and_physical_safe": 0,
                "accept_and_physical_failure": 0,
                "accept_and_physical_safe": 0,
            }
            safety_events = []
            step_trace = []
            previous_hard_alert = False
            previous_physical_alert = False
            previous_cost = total_cost > 0.0
            error = None
            for step in range(args.max_episode_steps):
                network_action = (
                    algorithm.get_deterministic_champion_action(observation)
                    if use_champion
                    else algorithm.get_deterministic_action(observation)
                )
                network_action = np.clip(
                    np.asarray(network_action, np.float32), -1.0, 1.0
                )
                if use_champion:
                    champion_network_action = network_action
                else:
                    champion_network_action = np.clip(
                        np.asarray(
                            algorithm.get_deterministic_champion_action(observation),
                            np.float32,
                        ),
                        -1.0,
                        1.0,
                    )
                actor_action_delta = float(
                    np.linalg.norm(network_action - champion_network_action)
                )
                actor_action_deltas.append(actor_action_delta)
                action = _actor_action_for_mpc(network_action)
                try:
                    certificate = env.certify_target_action(action)
                    if mode == "hybrid":
                        authority = _audit_low_level_cbf_authority(
                            env,
                            action,
                            certificate,
                            trajectory_tolerance=args.cbf_trajectory_tolerance,
                            derive_teacher=False,
                        )
                    else:
                        authority = {
                            "intervention": bool(
                                certificate.get("projection_used", False)
                            ),
                            "actual_control_modified": bool(
                                certificate.get("projection_used", False)
                            ),
                            "high_level_projection": bool(
                                certificate.get("projection_used", False)
                            ),
                            "high_level_modified": bool(
                                certificate.get("projection_used", False)
                            ),
                            "low_level_cbf_authority": False,
                            "actual_low_level_modified": False,
                            "audit_authority": bool(
                                certificate.get("projection_used", False)
                            ),
                            "diagnostic_only": False,
                            "trajectory_rmse": 0.0,
                        }
                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.execute_certified_target(certificate)
                except Exception as exc:
                    error = f"{type(exc).__name__}:{exc}"
                    break
                mgf = info.get("mgf", {})
                hard = mgf.get("raw_hard_safety", mgf.get("hard_safety", {}))
                physical = info.get("mpc_control", {})
                raw_target_speed = float(
                    info.get(
                        "raw_target_speed",
                        mgf.get("raw_target_speed", np.nan),
                    )
                )
                executed_target_speed = float(
                    info.get(
                        "executed_target_speed",
                        mgf.get("executed_target_speed", np.nan),
                    )
                )
                actual_vehicle_speed = float(
                    physical.get("vehicle_state", {}).get("velocity", np.nan)
                )
                speed_command_delta = float(
                    info.get(
                        "speed_command_delta",
                        mgf.get(
                            "speed_command_delta",
                            executed_target_speed - raw_target_speed,
                        ),
                    )
                )
                if np.isfinite(raw_target_speed):
                    raw_target_speeds.append(raw_target_speed)
                if np.isfinite(executed_target_speed):
                    executed_target_speeds.append(executed_target_speed)
                if np.isfinite(actual_vehicle_speed):
                    actual_vehicle_speeds.append(actual_vehicle_speed)
                if np.isfinite(speed_command_delta):
                    speed_command_deltas.append(speed_command_delta)
                speed_only_interventions += int(
                    bool(
                        info.get(
                            "speed_only_intervention",
                            mgf.get("speed_only_intervention", False),
                        )
                    )
                )
                braking_commands += int(
                    np.isfinite(executed_target_speed)
                    and np.isfinite(actual_vehicle_speed)
                    and executed_target_speed < actual_vehicle_speed - 1e-6
                )
                completion = float(physical.get("route_completion", np.nan))
                if np.isfinite(completion):
                    route_completion = max(route_completion, completion)
                projection_norm = float(mgf.get("projection_norm", 0.0))
                projected = bool(mgf.get("projection_used", projection_norm > 1e-6))
                low_level_modified = bool(
                    authority.get("actual_low_level_modified", False)
                )
                intervened = bool(authority["intervention"])
                audited_authority = bool(authority.get("audit_authority", intervened))
                diagnostic_only = bool(
                    authority.get(
                        "diagnostic_only", audited_authority and not intervened
                    )
                )
                trajectory_delta = float(authority["trajectory_rmse"])
                interventions += int(intervened)
                audit_authorities += int(audited_authority)
                diagnostic_only_steps += int(diagnostic_only)
                high_level_projections += int(projected)
                actual_low_level_modifications += int(low_level_modified)
                solver_finite += int(bool(mgf.get("solver_finite", False)))
                projection_norms.append(projection_norm)
                trajectory_deltas.append(trajectory_delta)
                hard_margins.append(float(hard.get("hard_cbf_margin", np.nan)))
                hard_margin = float(hard.get("hard_cbf_margin", np.nan))
                ordinary_audit_cbf_margin = float(
                    authority.get("ordinary_hard_certificate", {}).get(
                        "hard_cbf_margin", hard_margin
                    )
                )
                ordinary_audit_cbf_margins.append(ordinary_audit_cbf_margin)
                physical_margin = float(
                    physical.get("physical_body_sat_margin", np.nan)
                )
                physical_margins.append(physical_margin)
                step_cost = max(float(info.get("cost", 0.0)), 0.0)
                total_cost += step_cost
                task_return += float(info.get("task_reward", reward))
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
                    success
                    or info.get("arrive_dest", False)
                    or info.get("success", False)
                )
                hard_alert = bool(np.isfinite(hard_margin) and hard_margin < 0.0)
                physical_alert = bool(
                    np.isfinite(physical_margin) and physical_margin < 0.0
                )
                cost_alert = step_cost > 0.0
                step_collision = bool(
                    info.get("crash_vehicle", False) or info.get("crash_object", False)
                )
                step_road_departure = bool(
                    info.get("out_of_road", False) or info.get("road_departure", False)
                )
                physical_failure = bool(
                    physical_alert
                    or cost_alert
                    or step_collision
                    or step_road_departure
                )
                if bool(getattr(args, "include_step_trace", False)):
                    mpc = info.get("mpc_control", {})
                    vehicle_state = mpc.get("vehicle_state", {})
                    low_level = np.asarray(
                        mpc.get("low_level_action", [np.nan, np.nan]),
                        dtype=np.float32,
                    ).reshape(-1)
                    reference_y = np.asarray(
                        mpc.get("mpc_ref_traj_y", []), dtype=np.float32
                    ).reshape(-1)
                    optimized_y = np.asarray(
                        mpc.get("mpc_traj_y", []), dtype=np.float32
                    ).reshape(-1)
                    step_trace.append(
                        {
                            "step": step,
                            "network_action": network_action.tolist(),
                            "champion_network_action": (
                                champion_network_action.tolist()
                            ),
                            "actor_action_delta_from_champion": (actor_action_delta),
                            "actor_action": action.tolist(),
                            "steering": float(
                                low_level[0] if low_level.size > 0 else np.nan
                            ),
                            "throttle": float(
                                low_level[1] if low_level.size > 1 else np.nan
                            ),
                            "lateral_offset": float(
                                vehicle_state.get("lateral_offset", np.nan)
                            ),
                            "heading_error": float(
                                vehicle_state.get("heading_error", np.nan)
                            ),
                            "velocity": float(vehicle_state.get("velocity", np.nan)),
                            "world_x": float(mpc.get("world_x", np.nan)),
                            "world_y": float(mpc.get("world_y", np.nan)),
                            "route_completion": float(
                                mpc.get("route_completion", np.nan)
                            ),
                            "current_road_clearance": float(
                                mpc.get("current_road_clearance", np.nan)
                            ),
                            "mpc_min_road_clearance": float(
                                mpc.get("mpc_min_road_clearance", np.nan)
                            ),
                            "mpc_cost_tracking": float(
                                mpc.get("mpc_cost_tracking", np.nan)
                            ),
                            "mpc_cost_road": float(mpc.get("mpc_cost_road", np.nan)),
                            "reference_y_min": _minimum_finite(reference_y.tolist()),
                            "reference_y_max": float(
                                np.max(reference_y) if reference_y.size else np.nan
                            ),
                            "optimized_y_min": _minimum_finite(optimized_y.tolist()),
                            "optimized_y_max": float(
                                np.max(optimized_y) if optimized_y.size else np.nan
                            ),
                            "shadow_hard_cbf_margin": hard_margin,
                            "ordinary_audit_cbf_margin": (ordinary_audit_cbf_margin),
                            "physical_body_sat_margin": physical_margin,
                            "cost": step_cost,
                            "collision": step_collision,
                            "road_departure": step_road_departure,
                        }
                    )
                if np.isfinite(hard_margin):
                    decision = "reject" if hard_alert else "accept"
                    outcome = (
                        "physical_failure" if physical_failure else "physical_safe"
                    )
                    cbf_physical_confusion[f"{decision}_and_{outcome}"] += 1
                if len(safety_events) < 256 and (
                    audited_authority
                    or (cost_alert and not previous_cost)
                    or (hard_alert and not previous_hard_alert)
                    or (physical_alert and not previous_physical_alert)
                ):
                    safety_events.append(
                        {
                            "step": step,
                            "network_action": network_action.tolist(),
                            "raw_action": action.tolist(),
                            "executed_action": np.asarray(
                                info.get("executed_target", action),
                                dtype=np.float32,
                            ).tolist(),
                            "projected": projected,
                            "intervened": intervened,
                            "audit_authority": audited_authority,
                            "diagnostic_only": diagnostic_only,
                            "actual_low_level_modified": low_level_modified,
                            "projection_norm": projection_norm,
                            "cbf_trajectory_rmse": trajectory_delta,
                            "cost": step_cost,
                            "shadow_hard_cbf_margin": hard_margin,
                            "ordinary_audit_cbf_margin": (ordinary_audit_cbf_margin),
                            "physical_body_sat_margin": physical_margin,
                            "route_completion": float(
                                physical.get("route_completion", np.nan)
                            ),
                        }
                    )
                previous_hard_alert = hard_alert
                previous_physical_alert = physical_alert
                previous_cost = cost_alert
                observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
                if terminated or truncated:
                    break
            executed_steps = step + 1 if error is None else step
            rows.append(
                {
                    "episode": episode,
                    "scene_signature": reset_info.get("metasafe_scene_signature", ""),
                    "post_reset_warmup_steps": int(
                        reset_info.get("post_reset_warmup_steps", 0)
                    ),
                    "post_reset_observation_linf_delta": float(
                        reset_info.get("post_reset_observation_linf_delta", 0.0)
                    ),
                    "steps": int(executed_steps),
                    "return": task_return,
                    "success": success,
                    "collision": collision,
                    "road_departure": road_departure,
                    "environment_cost": total_cost,
                    "intervention_rate": interventions / max(executed_steps, 1),
                    "actual_intervention_rate": interventions / max(executed_steps, 1),
                    "audit_authority_rate": audit_authorities / max(executed_steps, 1),
                    "diagnostic_only_rate": diagnostic_only_steps
                    / max(executed_steps, 1),
                    "high_level_projection_rate": high_level_projections
                    / max(executed_steps, 1),
                    "actual_low_level_modification_rate": actual_low_level_modifications
                    / max(executed_steps, 1),
                    "mean_projection_norm": float(
                        np.mean(projection_norms) if projection_norms else 0.0
                    ),
                    "mean_cbf_trajectory_rmse": float(
                        np.mean(trajectory_deltas) if trajectory_deltas else 0.0
                    ),
                    "mean_actor_action_delta_from_champion": float(
                        np.mean(actor_action_deltas) if actor_action_deltas else 0.0
                    ),
                    "max_actor_action_delta_from_champion": float(
                        np.max(actor_action_deltas) if actor_action_deltas else 0.0
                    ),
                    "solver_finite_rate": solver_finite / max(executed_steps, 1),
                    "raw_target_speed_mean": float(
                        np.mean(raw_target_speeds) if raw_target_speeds else np.nan
                    ),
                    "raw_target_speed_std": float(
                        np.std(raw_target_speeds) if raw_target_speeds else np.nan
                    ),
                    "executed_target_speed_mean": float(
                        np.mean(executed_target_speeds)
                        if executed_target_speeds
                        else np.nan
                    ),
                    "executed_target_speed_std": float(
                        np.std(executed_target_speeds)
                        if executed_target_speeds
                        else np.nan
                    ),
                    "actual_vehicle_speed_mean": float(
                        np.mean(actual_vehicle_speeds)
                        if actual_vehicle_speeds
                        else np.nan
                    ),
                    "actual_vehicle_speed_std": float(
                        np.std(actual_vehicle_speeds)
                        if actual_vehicle_speeds
                        else np.nan
                    ),
                    "target_speed_p10": float(
                        np.percentile(executed_target_speeds, 10)
                        if executed_target_speeds
                        else np.nan
                    ),
                    "target_speed_p50": float(
                        np.percentile(executed_target_speeds, 50)
                        if executed_target_speeds
                        else np.nan
                    ),
                    "target_speed_p90": float(
                        np.percentile(executed_target_speeds, 90)
                        if executed_target_speeds
                        else np.nan
                    ),
                    "speed_command_delta_mean": float(
                        np.mean(speed_command_deltas)
                        if speed_command_deltas
                        else np.nan
                    ),
                    "speed_only_intervention_rate": speed_only_interventions
                    / max(executed_steps, 1),
                    "braking_command_fraction": braking_commands
                    / max(executed_steps, 1),
                    "episode_duration": float(executed_steps * control_interval),
                    "progress_per_second": float(
                        route_completion / max(executed_steps * control_interval, 1e-9)
                    ),
                    "min_shadow_hard_cbf_margin": _minimum_finite(hard_margins),
                    "min_ordinary_audit_cbf_margin": _minimum_finite(
                        ordinary_audit_cbf_margins
                    ),
                    "min_physical_body_sat_margin": _minimum_finite(physical_margins),
                    "cbf_physical_confusion": cbf_physical_confusion,
                    "successful_trajectory_shadow_rejections": int(
                        cbf_physical_confusion["reject_and_physical_safe"]
                        if success
                        and not collision
                        and not road_departure
                        and total_cost <= 0.0
                        else 0
                    ),
                    "error": error,
                    "safety_events": safety_events,
                    "step_trace": step_trace,
                }
            )
            if args.recreate_environment_each_episode:
                env.close()
                env = None
    finally:
        if env is not None:
            env.close()
    aggregate_confusion = {
        key: int(sum(row["cbf_physical_confusion"][key] for row in rows))
        for key in (
            "reject_and_physical_failure",
            "reject_and_physical_safe",
            "accept_and_physical_failure",
            "accept_and_physical_safe",
        )
    }
    classified_steps = max(sum(aggregate_confusion.values()), 1)
    return {
        "mode": mode,
        "use_champion": use_champion,
        "actor_action_transform": "identity_clip",
        "intervention_accounting_version": INTERVENTION_ACCOUNTING_VERSION,
        "intervention_accounting_definition": (INTERVENTION_ACCOUNTING_DEFINITION),
        "episodes": episodes,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
        "road_departure_rate": float(np.mean([row["road_departure"] for row in rows])),
        "total_environment_cost": float(
            np.sum([row["environment_cost"] for row in rows])
        ),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "returns": [float(row["return"]) for row in rows],
        "intervention_rate": float(np.mean([row["intervention_rate"] for row in rows])),
        "actual_intervention_rate": float(
            np.mean([row["actual_intervention_rate"] for row in rows])
        ),
        "audit_authority_rate": float(
            np.mean([row["audit_authority_rate"] for row in rows])
        ),
        "diagnostic_only_rate": float(
            np.mean([row["diagnostic_only_rate"] for row in rows])
        ),
        "high_level_projection_rate": float(
            np.mean([row["high_level_projection_rate"] for row in rows])
        ),
        "actual_low_level_modification_rate": float(
            np.mean([row["actual_low_level_modification_rate"] for row in rows])
        ),
        "mean_projection_norm": float(
            np.mean([row["mean_projection_norm"] for row in rows])
        ),
        "mean_cbf_trajectory_rmse": float(
            np.mean([row["mean_cbf_trajectory_rmse"] for row in rows])
        ),
        "mean_actor_action_delta_from_champion": float(
            np.mean([row["mean_actor_action_delta_from_champion"] for row in rows])
        ),
        "max_actor_action_delta_from_champion": float(
            np.max([row["max_actor_action_delta_from_champion"] for row in rows])
        ),
        "solver_finite_rate": float(
            np.mean([row["solver_finite_rate"] for row in rows])
        ),
        "raw_target_speed_mean": float(
            np.nanmean([row["raw_target_speed_mean"] for row in rows])
        ),
        "raw_target_speed_std": float(
            np.nanmean([row["raw_target_speed_std"] for row in rows])
        ),
        "executed_target_speed_mean": float(
            np.nanmean([row["executed_target_speed_mean"] for row in rows])
        ),
        "executed_target_speed_std": float(
            np.nanmean([row["executed_target_speed_std"] for row in rows])
        ),
        "actual_vehicle_speed_mean": float(
            np.nanmean([row["actual_vehicle_speed_mean"] for row in rows])
        ),
        "actual_vehicle_speed_std": float(
            np.nanmean([row["actual_vehicle_speed_std"] for row in rows])
        ),
        "target_speed_p10": float(
            np.nanmean([row["target_speed_p10"] for row in rows])
        ),
        "target_speed_p50": float(
            np.nanmean([row["target_speed_p50"] for row in rows])
        ),
        "target_speed_p90": float(
            np.nanmean([row["target_speed_p90"] for row in rows])
        ),
        "speed_command_delta_mean": float(
            np.nanmean([row["speed_command_delta_mean"] for row in rows])
        ),
        "speed_only_intervention_rate": float(
            np.mean([row["speed_only_intervention_rate"] for row in rows])
        ),
        "braking_command_fraction": float(
            np.mean([row["braking_command_fraction"] for row in rows])
        ),
        "episode_duration": float(np.mean([row["episode_duration"] for row in rows])),
        "progress_per_second": float(
            np.mean([row["progress_per_second"] for row in rows])
        ),
        "min_shadow_hard_cbf_margin": _minimum_finite(
            [row["min_shadow_hard_cbf_margin"] for row in rows]
        ),
        "min_ordinary_audit_cbf_margin": _minimum_finite(
            [row["min_ordinary_audit_cbf_margin"] for row in rows]
        ),
        "min_physical_body_sat_margin": _minimum_finite(
            [row["min_physical_body_sat_margin"] for row in rows]
        ),
        "cbf_physical_confusion": aggregate_confusion,
        "shadow_reject_rate": float(
            (
                aggregate_confusion["reject_and_physical_failure"]
                + aggregate_confusion["reject_and_physical_safe"]
            )
            / classified_steps
        ),
        "successful_trajectory_shadow_rejections": int(
            sum(row["successful_trajectory_shadow_rejections"] for row in rows)
        ),
        "errors": [row["error"] for row in rows if row["error"]],
        "rows": rows,
    }


def _evaluate_suite(
    algorithm,
    args,
    *,
    use_champion: bool,
    projection_reserve: float | None = None,
    projection_kinematic_reserve: float | None = None,
) -> dict[str, Any]:
    return {
        mode: _evaluate_policy(
            algorithm,
            args,
            mode=mode,
            use_champion=use_champion,
            episodes=args.candidate_eval_episodes,
            projection_reserve=projection_reserve,
            projection_kinematic_reserve=projection_kinematic_reserve,
        )
        for mode in ("hybrid", "ordinary")
    }


def _return_lcb(candidate: list[float], champion: list[float]) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    champion = np.asarray(champion, dtype=np.float64)
    count = min(candidate.size, champion.size)
    if count < 1:
        return float("-inf")
    difference = candidate[:count] - champion[:count]
    if count == 1:
        return float(difference[0])
    standard_error = float(np.std(difference, ddof=1) / math.sqrt(count))
    return float(np.mean(difference) - 1.645 * standard_error)


def _closed_loop_safety_failures(
    suite: dict[str, Any],
    args,
) -> list[str]:
    """Return hard deployment failures shared by BC and online gates."""

    failures = []
    for mode in ("hybrid", "ordinary"):
        result = suite[mode]
        if result["success_rate"] < args.minimum_success_rate:
            failures.append(f"{mode}_success_below_minimum")
        if result["collision_rate"] > 0.0:
            failures.append(f"{mode}_collision")
        if result["road_departure_rate"] > 0.0:
            failures.append(f"{mode}_road_departure")
        if result["total_environment_cost"] > 0.0:
            failures.append(f"{mode}_environment_cost")
        if result["solver_finite_rate"] < 1.0 or result["errors"]:
            failures.append(f"{mode}_solver")
        physical = result["min_physical_body_sat_margin"]
        if not np.isfinite(physical) or physical <= 0.0:
            failures.append(f"{mode}_physical_sat")
    ordinary = suite["ordinary"]
    ordinary_actual = float(
        ordinary.get(
            "actual_intervention_rate",
            ordinary.get("intervention_rate", 0.0),
        )
    )
    if ordinary_actual > 0.0:
        failures.append("ordinary_actual_intervention")
    if float(ordinary.get("high_level_projection_rate", 0.0)) > 0.0:
        failures.append("ordinary_projection")
    return failures


def _candidate_gate(
    candidate: dict[str, Any],
    champion: dict[str, Any],
    offline_metrics: dict[str, Any],
    champion_offline_metrics: dict[str, Any],
    args,
    *,
    require_performance_improvement: bool = True,
) -> tuple[bool, list[str], bool]:
    failures = _closed_loop_safety_failures(candidate, args)
    actual_rate = lambda result: float(
        result.get("actual_intervention_rate", result.get("intervention_rate", 0.0))
    )
    if candidate["ordinary"]["success_rate"] < (
        champion["ordinary"]["success_rate"] - args.ordinary_success_regression
    ):
        failures.append("ordinary_success_regression")
    if require_performance_improvement:
        candidate_actual = actual_rate(candidate["hybrid"])
        champion_actual = actual_rate(champion["hybrid"])
        if args.require_strict_intervention_improvement:
            if candidate_actual >= champion_actual - 1e-12:
                failures.append("actual_intervention_not_strictly_lower")
        elif candidate_actual > champion_actual + args.intervention_regression:
            failures.append("intervention_regression")
        lcb = _return_lcb(
            candidate["ordinary"]["returns"], champion["ordinary"]["returns"]
        )
        if lcb < -args.return_noninferiority:
            failures.append("ordinary_return_lcb")
    candidate_q_error = float(offline_metrics["q_calibration_mae"])
    champion_q_error = max(float(champion_offline_metrics["q_calibration_mae"]), 1e-6)
    critic_failure = bool(
        not np.isfinite(candidate_q_error)
        or candidate_q_error > args.critic_rollback_factor * champion_q_error
    )
    if critic_failure:
        failures.append("critic_calibration")
    return not failures, failures, critic_failure


_SAFETY_REJECTION_SUFFIXES = (
    "_collision",
    "_road_departure",
    "_environment_cost",
    "_solver",
    "_physical_sat",
)


def _candidate_rejection_class(
    failures: Iterable[str],
    *,
    critic_failure: bool,
) -> str:
    """Classify a failed gate for the three-level Actor recovery policy."""

    failures = tuple(str(failure) for failure in failures)
    if critic_failure or "critic_calibration" in failures:
        return "critic"
    if any(failure.endswith(_SAFETY_REJECTION_SUFFIXES) for failure in failures):
        return "safety"
    return "performance"


def _candidate_score(suite: dict[str, Any], args) -> dict[str, float]:
    """Score a fixed-scene policy while keeping safety visible as a component."""

    ordinary = suite["ordinary"]
    hybrid = suite["hybrid"]
    task_return = float(
        ordinary.get(
            "mean_return",
            np.mean(np.asarray(ordinary.get("returns", [float("-inf")]))),
        )
    )
    intervention = float(
        hybrid.get(
            "actual_intervention_rate",
            hybrid.get("intervention_rate", 0.0),
        )
    )
    safety_violation = 0.0
    for result in (hybrid, ordinary):
        safety_violation += (
            float(result.get("collision_rate", 0.0))
            + float(result.get("road_departure_rate", 0.0))
            + max(float(result.get("total_environment_cost", 0.0)), 0.0)
        )
    score = (
        task_return
        - args.candidate_score_intervention_penalty * intervention
        - args.candidate_score_safety_penalty * safety_violation
    )
    return {
        "score": float(score),
        "task_return": task_return,
        "actual_intervention_rate": intervention,
        "safety_violation": float(safety_violation),
    }


def _online_learning_gate_failures(
    candidate: dict[str, Any], counters: dict[str, int], args
) -> list[str]:
    """Prove that a promoted online policy came from all three learners."""

    failures = []
    for counter, failure in (
        ("task_critic_updates", "no_online_task_critic_update"),
        ("proxy_updates", "no_online_proxy_update"),
        ("actor_updates", "no_online_actor_update"),
    ):
        if int(counters.get(counter, 0)) <= 0:
            failures.append(failure)
    action_delta = max(
        float(candidate[mode].get("mean_actor_action_delta_from_champion", 0.0))
        for mode in ("hybrid", "ordinary")
    )
    if (
        not np.isfinite(action_delta)
        or action_delta < args.minimum_candidate_actor_action_delta
    ):
        failures.append("candidate_actor_unchanged")
    return failures


def _hard_certificate_values(
    summary: dict[str, Any]
) -> tuple[float, float, float, float]:
    return (
        float(summary.get("hard_cbf_margin", np.nan)),
        float(summary.get("obstacle_clearance", np.nan)),
        float(summary.get("road_clearance", np.nan)),
        float(summary.get("kinematic_slack", np.nan)),
    )


def _certificate_trajectory_difference(first_certificate, second_certificate) -> dict:
    """Compare two same-state MPC certificates without advancing the world."""

    first = first_certificate["solved"]["result"]
    second = second_certificate["solved"]["result"]
    first_xy = np.stack([first.predicted_x, first.predicted_y], axis=1).astype(
        np.float32
    )
    second_xy = np.stack([second.predicted_x, second.predicted_y], axis=1).astype(
        np.float32
    )
    count = min(first_xy.shape[0], second_xy.shape[0])
    if count < 1:
        return {
            "trajectory_rmse": float("inf"),
            "trajectory_max_error": float("inf"),
            "control_delta": float("inf"),
        }
    error = np.linalg.norm(first_xy[:count] - second_xy[:count], axis=1)
    first_control = np.asarray([first.steering, first.throttle], dtype=np.float32)
    second_control = np.asarray([second.steering, second.throttle], dtype=np.float32)
    return {
        "trajectory_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "trajectory_max_error": float(np.max(error)),
        "control_delta": float(np.linalg.norm(first_control - second_control)),
    }


def _audit_low_level_cbf_authority(
    env,
    raw_action,
    actor_certificate,
    *,
    trajectory_tolerance: float,
    derive_teacher: bool,
) -> dict[str, Any]:
    """Separate active-CBF control modification from shadow diagnostics.

    The protected and ordinary solves use the same state, target and previous
    warm start.  A finite CBF-on/off trajectory difference beyond tolerance is
    evidence that the two solver branches differ, but counts as active low-level
    CBF authority only when the ordinary CBF-rate certificate is also negative.
    A positive-certificate branch delta, negative shadow certificate without a
    control change, or failed counterfactual solve remains diagnostic-only and
    cannot create a Proxy pair or Guardian teacher label.
    """

    high_level_projection = bool(actor_certificate.get("projection_used", False))
    protected_raw_solved = actor_certificate.get(
        "raw_solved", actor_certificate.get("solved")
    )
    solved_action = actor_certificate.get("solved", {}).get("action")
    if solved_action is None:
        high_level_modified = high_level_projection
    else:
        high_level_modified = bool(
            high_level_projection
            and np.linalg.norm(
                np.asarray(solved_action, dtype=np.float32).reshape(2)
                - np.asarray(raw_action, dtype=np.float32).reshape(2)
            )
            > 1e-6
        )
    try:
        # Use the protected solution as the ordinary solver's reference warm
        # start.  Comparing two independently initialized local NMPC solves
        # can otherwise report a large trajectory delta even when obstacle
        # CBF constraints are inactive (including immediately after reset).
        # Older test doubles retain the explicit ordinary-certificate path.
        if hasattr(env, "certify_independent_actor_target"):
            independent = env.certify_independent_actor_target(
                np.asarray(raw_action, dtype=np.float32),
                reference_solved=protected_raw_solved,
                trajectory_tolerance=trajectory_tolerance,
            )
            summary = {
                name: independent.get(name, np.nan)
                for name in (
                    "hard_cbf_margin",
                    "obstacle_clearance",
                    "road_clearance",
                    "kinematic_slack",
                )
            }
            summary["has_obstacle"] = bool(independent.get("has_obstacle", True))
            summary["finite"] = bool(
                independent.get("ordinary_solved", {}).get("solver_finite", False)
            )
            difference = {
                name: float(independent.get(name, float("inf")))
                for name in (
                    "trajectory_rmse",
                    "trajectory_max_error",
                    "control_delta",
                )
            }
            ordinary = independent.get("execution_certificate")
            ordinary_solved = independent.get("ordinary_solved", {})
            ordinary_finite_flag = bool(ordinary_solved.get("solver_finite", False))
        else:
            ordinary = env.certify_ordinary_target_action(
                np.asarray(raw_action, dtype=np.float32),
                warm_start_mode="reference",
            )
            difference = _certificate_trajectory_difference(
                {"solved": protected_raw_solved}, ordinary
            )
            summary = dict(ordinary.get("raw_hard_certificate", {}))
            ordinary_finite_flag = bool(
                ordinary.get("solved", {}).get("solver_finite", False)
            )
        values = _hard_certificate_values(summary)
        ordinary_finite = bool(
            ordinary_finite_flag
            and summary.get("finite", True)
            and all(np.isfinite(value) for value in values)
            and np.isfinite(difference["trajectory_rmse"])
        )
        # A negative CBF rate condition is useful audit evidence, but it is
        # not by itself a physical collision/road/kinematic violation.  Keep
        # the two meanings separate so a conservative CBF-only rejection is
        # learned as a low-weight Soft pair instead of a full-weight Hard one.
        cbf_violation = bool(ordinary_finite and values[0] < 0.0)
        hard_violation = bool(
            ordinary_finite and any(value < 0.0 for value in values[1:])
        )
        certificate_violation = bool(cbf_violation or hard_violation)
        trajectory_changed = bool(
            ordinary_finite
            and np.isfinite(difference["trajectory_rmse"])
            and difference["trajectory_rmse"] > trajectory_tolerance
        )
        # A CBF-on/off trajectory delta alone is a counterfactual solver audit.
        # Count it as actual obstacle-CBF authority only when the independently
        # solved ordinary trajectory also violates the CBF-rate condition.
        # Otherwise inactive constraints or local SQP branches can produce a
        # large delta even in far-field, physically safe frames.
        actual_low_level_modified = bool(trajectory_changed and cbf_violation)
        audit_only_trajectory_difference = bool(
            trajectory_changed and not actual_low_level_modified
        )
        error = None
    except Exception as exc:
        ordinary = None
        summary = {}
        difference = {
            "trajectory_rmse": float("inf"),
            "trajectory_max_error": float("inf"),
            "control_delta": float("inf"),
        }
        ordinary_finite = False
        cbf_violation = False
        hard_violation = False
        certificate_violation = False
        trajectory_changed = False
        audit_only_trajectory_difference = False
        actual_low_level_modified = False
        error = f"{type(exc).__name__}:{exc}"

    audit_authority = bool(
        high_level_projection
        or not ordinary_finite
        or certificate_violation
        or trajectory_changed
    )
    intervention = bool(high_level_modified or actual_low_level_modified)
    diagnostic_only = bool(audit_authority and not intervention)
    teacher_action = None
    teacher_valid = False
    teacher_error = None
    teacher_trajectory_rmse = float("nan")
    if derive_teacher and intervention:
        try:
            inverse = env.derive_inverse_mpc_teacher(actor_certificate["solved"])
            candidate = np.asarray(inverse.get("action"), dtype=np.float32).reshape(2)
            teacher_valid = bool(
                inverse.get("valid", False) and np.all(np.isfinite(candidate))
            )
            if teacher_valid:
                teacher_action = candidate
                teacher_trajectory_rmse = float(
                    inverse.get("trajectory_rmse", float("nan"))
                )
            else:
                teacher_error = "invalid_inverse_mpc_teacher"
        except Exception as exc:
            teacher_error = f"{type(exc).__name__}:{exc}"

    return {
        "intervention": intervention,
        "actual_control_modified": intervention,
        "high_level_projection": high_level_projection,
        "high_level_modified": high_level_modified,
        "low_level_cbf_authority": actual_low_level_modified,
        "actual_low_level_modified": actual_low_level_modified,
        "audit_only_trajectory_difference": audit_only_trajectory_difference,
        "audit_authority": audit_authority,
        "diagnostic_only": diagnostic_only,
        "ordinary_solver_finite": ordinary_finite,
        "ordinary_cbf_violation": cbf_violation,
        "ordinary_hard_violation": hard_violation,
        "ordinary_hard_certificate": summary,
        "ordinary_certificate": ordinary,
        "teacher_action": teacher_action,
        "teacher_valid": teacher_valid,
        "teacher_error": teacher_error,
        "teacher_trajectory_rmse": teacher_trajectory_rmse,
        "audit_error": error,
        **difference,
    }


def _step_experience(
    observation,
    raw_action,
    reward,
    terminated,
    truncated,
    next_observation,
    certificate,
    info,
    *,
    soft_pair_weight: float,
    authority_audit: dict[str, Any] | None = None,
    counterfactual_teacher_action=None,
    counterfactual_teacher_valid: bool = False,
    activate_counterfactual_teacher: bool = False,
    filter_version: int = 0,
    spatial_reserve: float = 0.0,
    kinematic_reserve: float = 0.0,
    scene_seed: int | None = None,
    scene_signature: str | None = None,
    action_semantics_version: str | None = None,
):
    from relax.utils.experience import (
        INTERVENTION_DIAGNOSTIC,
        INTERVENTION_HARD,
        INTERVENTION_NONE,
        INTERVENTION_SOFT,
        GuardianCalQLExperience,
        certificate_has_physical_violation,
    )
    from relax.utils.action_semantics import (
        FIXED_SPEED_ACTION_SEMANTICS,
        action_semantics_code,
        encode_scene_signature,
    )

    mgf = info.get("mgf", {})
    semantics = str(
        action_semantics_version
        or info.get(
            "action_semantics_version",
            mgf.get(
                "action_semantics_version",
                FIXED_SPEED_ACTION_SEMANTICS,
            ),
        )
    )
    resolved_scene_seed = int(
        info.get("metasafe_fixed_seed", 0) if scene_seed is None else scene_seed
    )
    resolved_scene_signature = str(
        info.get("metasafe_scene_signature", "")
        if scene_signature is None
        else scene_signature
    )
    executed_hard = mgf.get("hard_safety", {})
    raw_hard = mgf.get("raw_hard_safety", certificate.get("raw_hard_certificate", {}))
    actual_executed_action = np.asarray(
        info.get("executed_target", raw_action), dtype=np.float32
    ).reshape(2)
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(2)
    actual_projection_norm = float(np.linalg.norm(actual_executed_action - raw_action))
    audited = authority_audit is not None
    authority_audit = authority_audit or {}
    actual_low_level_modified = bool(
        authority_audit.get("actual_low_level_modified", False)
    )
    actual_high_level_modified = bool(
        actual_projection_norm > 1e-6
        or (
            not audited
            and mgf.get("projection_used", False)
            and actual_projection_norm > 0.0
        )
    )
    intervention = bool(actual_high_level_modified or actual_low_level_modified)
    audit_authority = bool(
        authority_audit.get("audit_authority", intervention) or intervention
    )
    authority_teacher_valid = bool(authority_audit.get("teacher_valid", False))
    if audited and intervention and authority_teacher_valid:
        proxy_corrected_action = np.asarray(
            authority_audit["teacher_action"], dtype=np.float32
        ).reshape(2)
    else:
        proxy_corrected_action = actual_executed_action.copy()
    pair_norm = float(np.linalg.norm(proxy_corrected_action - raw_action))
    raw_hard = authority_audit.get("ordinary_hard_certificate", raw_hard)
    solver_finite = bool(mgf.get("solver_finite", False))
    proposal_evaluated = bool(
        solver_finite and authority_audit.get("ordinary_solver_finite", True)
    )
    projection_failed = bool(mgf.get("projection_failed", False))
    numerical = bool(projection_failed or not solver_finite)
    environment_hard_failure = bool(
        max(float(info.get("cost", 0.0)), 0.0) > 0.0
        or info.get("crash_vehicle", False)
        or info.get("crash_object", False)
        or info.get("out_of_road", False)
        or info.get("road_departure", False)
    )
    physical_body_margin = float(
        info.get("mpc_control", {}).get("physical_body_sat_margin", np.nan)
    )
    environment_hard_failure = bool(
        environment_hard_failure
        or (np.isfinite(physical_body_margin) and physical_body_margin <= 0.0)
    )
    hard = bool(
        intervention
        and not numerical
        and (certificate_has_physical_violation(raw_hard) or environment_hard_failure)
    )
    soft = bool(intervention and not hard and not numerical)
    diagnostic = bool((audit_authority and not intervention) or numerical)
    pair_valid = bool(
        intervention
        and not numerical
        and pair_norm > 1e-6
        and (not audited or authority_teacher_valid)
    )
    pair_weight = 1.0 if hard else soft_pair_weight if soft else 0.0
    intervention_kind = (
        INTERVENTION_HARD
        if hard
        else (
            INTERVENTION_SOFT
            if soft
            else INTERVENTION_DIAGNOSTIC if diagnostic else INTERVENTION_NONE
        )
    )
    hard_margin, obstacle, road, kinematic = _hard_certificate_values(executed_hard)
    normalized = math.tanh(_finite(hard_margin) / 100.0)
    if activate_counterfactual_teacher and counterfactual_teacher_valid:
        # Pure DAgger learns the coherent ordinary expert even if a
        # conservative CBF-on/off audit also reports an intervention.  The
        # audit-derived inverse action remains the proxy-corrected pair, but
        # it must not replace the behaviour-cloning target.
        teacher_action = np.asarray(
            counterfactual_teacher_action, dtype=np.float32
        ).reshape(2)
        teacher_valid = bool(np.all(np.isfinite(teacher_action)))
        teacher_weight = 1.0 if teacher_valid else 0.0
    elif intervention and not numerical and (not audited or authority_teacher_valid):
        teacher_action = proxy_corrected_action.copy()
        teacher_valid = True
        teacher_weight = pair_weight
    elif counterfactual_teacher_valid:
        teacher_action = np.asarray(
            counterfactual_teacher_action, dtype=np.float32
        ).reshape(2)
        teacher_valid = bool(np.all(np.isfinite(teacher_action)))
        # Full Guardian-CalQL keeps proactive labels latent until a future
        # intervention validates the short pre-intervention window.  Pure BC
        # DAgger instead follows the standard aggregation rule and learns the
        # expert label at every state actually visited by the current Actor.
        teacher_weight = 1.0 if activate_counterfactual_teacher else 0.0
    else:
        teacher_action = actual_executed_action.copy()
        teacher_valid = False
        teacher_weight = 0.0
    return GuardianCalQLExperience(
        obs=np.asarray(observation, dtype=np.float32).reshape(-1),
        # Task TD must stay on the action that actually caused this filtered
        # transition.  The inverse-MPC label is counterfactual and belongs only
        # to the proxy pair and DAgger supervision below.
        action=actual_executed_action,
        reward=np.float32(reward),
        done=np.bool_(bool(terminated or truncated)),
        next_obs=np.asarray(next_observation, dtype=np.float32).reshape(-1),
        raw_action=raw_action,
        proxy_corrected_action=proxy_corrected_action,
        teacher_action=teacher_action,
        teacher_valid=np.bool_(teacher_valid),
        teacher_weight=np.float32(teacher_weight),
        actor_controlled=np.bool_(not intervention and not numerical),
        intervention=np.bool_(intervention),
        intervention_hard=np.bool_(hard),
        intervention_soft=np.bool_(soft),
        intervention_numerical=np.bool_(numerical),
        intervention_kind=np.int8(intervention_kind),
        actual_control_modified=np.bool_(intervention),
        actual_low_level_modified=np.bool_(actual_low_level_modified),
        audit_authority=np.bool_(audit_authority),
        proposal_evaluated=np.bool_(proposal_evaluated),
        proxy_pair_valid=np.bool_(pair_valid),
        proxy_pair_weight=np.float32(pair_weight),
        proxy_cost=np.float32(0.0),
        guardian_dependence_target=np.float32(0.0),
        guardian_dependence_corrected_target=np.float32(0.0),
        guardian_dependence_target_valid=np.bool_(False),
        guardian_dependence_corrected_valid=np.bool_(False),
        behavior_return_to_go=np.float32(0.0),
        behavior_return_valid=np.bool_(False),
        q_anchor_valid=np.bool_(False),
        raw_target_speed=np.float32(
            info.get("raw_target_speed", mgf.get("raw_target_speed", 0.0))
        ),
        executed_target_speed=np.float32(
            info.get(
                "executed_target_speed",
                mgf.get("executed_target_speed", 0.0),
            )
        ),
        speed_command_delta=np.float32(
            info.get(
                "speed_command_delta",
                mgf.get("speed_command_delta", 0.0),
            )
        ),
        speed_only_intervention=np.bool_(
            info.get(
                "speed_only_intervention",
                mgf.get("speed_only_intervention", False),
            )
        ),
        action_semantics_version=action_semantics_code(semantics),
        scene_seed=np.int32(resolved_scene_seed),
        scene_signature=encode_scene_signature(resolved_scene_signature),
        filter_version=np.int32(filter_version),
        spatial_reserve=np.float32(spatial_reserve),
        kinematic_reserve=np.float32(kinematic_reserve),
        execution_layer=np.int8(
            (2 if actual_low_level_modified else 0)
            + (1 if actual_high_level_modified else 0)
        ),
        projection_norm=np.float32(actual_projection_norm),
        hard_cbf_margin=np.float32(_finite(hard_margin)),
        cbf_margin_normalized=np.float32(normalized),
        obstacle_clearance=np.float32(_finite(obstacle)),
        road_clearance=np.float32(_finite(road)),
        kinematic_slack=np.float32(_finite(kinematic)),
        solver_finite=np.bool_(solver_finite),
        backup_feasible=np.bool_(solver_finite and not projection_failed),
        recovery_success=np.bool_(solver_finite and not projection_failed),
    )


def _finalize_episode_returns(
    experiences,
    *,
    gamma: float,
    proxy_discount: float | None = None,
    q_anchor_valid: bool = False,
) -> list:
    """Attach exact discounted RTG to one completed episode.

    Online rows stay outside replay until this function succeeds.  The
    explicit validity bit makes a legitimate zero-return trajectory
    distinguishable from an unfinished trajectory without a numeric
    sentinel.
    """

    experiences = list(experiences)
    if not experiences:
        return []
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("episode RTG discount must be in [0, 1]")
    if not bool(experiences[-1].done):
        raise ValueError("cannot finalize RTG for an incomplete episode")
    if any(bool(experience.done) for experience in experiences[:-1]):
        raise ValueError("episode staging contains an early terminal transition")
    if proxy_discount is not None and not 0.0 < float(proxy_discount) < 1.0:
        raise ValueError("episode Proxy discount must be in (0, 1)")

    running = 0.0
    dependence_next = 0.0
    finalized = []
    for experience in reversed(experiences):
        running = float(experience.reward) + float(gamma) * running
        hard = bool(experience.intervention_hard)
        soft = bool(experience.intervention_soft)
        numerical = bool(experience.intervention_numerical)
        proposal_evaluated = bool(experience.proposal_evaluated)
        dependence_cost = 1.0 if hard else 0.5 if soft else 0.0
        if proxy_discount is None:
            dependence_raw = float(experience.guardian_dependence_target)
            dependence_corrected = float(
                experience.guardian_dependence_corrected_target
            )
            dependence_valid = bool(experience.guardian_dependence_target_valid)
            dependence_corrected_valid = bool(
                experience.guardian_dependence_corrected_valid
            )
        else:
            dependence_corrected = (
                0.0 if numerical else float(proxy_discount) * dependence_next
            )
            dependence_raw = (
                0.0
                if numerical
                else dependence_cost + (1.0 - dependence_cost) * dependence_corrected
            )
            dependence_raw = float(np.clip(dependence_raw, 0.0, 1.0))
            dependence_corrected = float(np.clip(dependence_corrected, 0.0, 1.0))
            dependence_valid = bool(not numerical and proposal_evaluated)
            dependence_corrected_valid = bool(
                dependence_valid
                and np.linalg.norm(
                    np.asarray(experience.raw_action, dtype=np.float32)
                    - np.asarray(experience.proxy_corrected_action, dtype=np.float32)
                )
                > 1e-6
            )
        finalized.append(
            experience._replace(
                behavior_return_to_go=np.float32(running),
                behavior_return_valid=np.bool_(True),
                q_anchor_valid=np.bool_(q_anchor_valid),
                guardian_dependence_target=np.float32(dependence_raw),
                guardian_dependence_corrected_target=np.float32(dependence_corrected),
                guardian_dependence_target_valid=np.bool_(dependence_valid),
                guardian_dependence_corrected_valid=np.bool_(
                    dependence_corrected_valid
                ),
            )
        )
        if proxy_discount is not None:
            dependence_next = dependence_raw
    finalized.reverse()
    return finalized


def _collect_ordinary_champion_anchor(
    algorithm,
    args,
    replay,
) -> dict[str, Any]:
    """Cache verified no-CBF policy transitions for functional anchoring.

    The default executes the accepted Champion.  An explicit full-checkpoint
    continuation may instead freeze its loaded LastSafe Learner trajectory,
    which preserves a high-return candidate away from the sparse boundary rows
    being corrected.  Either mode uses raw, unscaled actions in ordinary MPC,
    rejects the entire cache on any failure, and reserves an explicit batch
    slice for those exact state-action pairs.
    """

    requested = int(args.ordinary_champion_anchor_episodes)
    anchor_policy = str(args.ordinary_anchor_policy)
    policy_params = (
        algorithm.champion_state.params.policy
        if anchor_policy == "champion"
        else algorithm.get_policy_params()
    )
    policy_sha256 = _policy_hash(policy_params)
    if args.ordinary_champion_anchor_ratio <= 0.0 or requested <= 0:
        return {
            "enabled": False,
            "episodes": 0,
            "transitions": 0,
            "policy": anchor_policy,
            "policy_sha256": policy_sha256,
            "actor_action_transform": "identity_clip",
        }

    episode_rows = []
    for episode in range(requested):
        env = _make_environment(args, "ordinary")
        experiences = []
        task_return = 0.0
        total_cost = 0.0
        collision = False
        road_departure = False
        success = False
        solver_finite = True
        try:
            observation, reset_info = env.reset(seed=args.fixed_scene_seed)
            observation = np.asarray(observation, np.float32).reshape(-1)
            total_cost += float(reset_info.get("post_reset_warmup_cost", 0.0))
            collision = bool(reset_info.get("post_reset_warmup_collision", False))
            road_departure = bool(
                reset_info.get("post_reset_warmup_road_departure", False)
            )
            for _step in range(args.max_episode_steps):
                action = _actor_action_for_mpc(
                    algorithm.get_deterministic_champion_action(observation)
                    if anchor_policy == "champion"
                    else algorithm.get_deterministic_action(observation)
                )
                certificate = env.certify_ordinary_target_action(action)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.execute_certified_target(certificate)
                mgf = info.get("mgf", {})
                finite = bool(mgf.get("solver_finite", False))
                solver_finite = bool(solver_finite and finite)
                ordinary_summary = dict(
                    mgf.get(
                        "raw_hard_safety",
                        mgf.get(
                            "hard_safety",
                            certificate.get("raw_hard_certificate", {}),
                        ),
                    )
                )
                experience = _step_experience(
                    observation,
                    action,
                    reward,
                    terminated,
                    truncated,
                    next_observation,
                    certificate,
                    info,
                    soft_pair_weight=args.soft_pair_weight,
                    authority_audit={
                        "intervention": False,
                        "actual_control_modified": False,
                        "actual_low_level_modified": False,
                        "audit_authority": False,
                        "diagnostic_only": False,
                        "ordinary_solver_finite": finite,
                        "ordinary_hard_certificate": ordinary_summary,
                    },
                    filter_version=0,
                    spatial_reserve=args.projection_reserve,
                    kinematic_reserve=args.projection_kinematic_reserve,
                )._replace(
                    teacher_action=np.asarray(action, dtype=np.float32),
                    teacher_valid=np.bool_(True),
                    teacher_weight=np.float32(1.0),
                    actor_controlled=np.bool_(True),
                )
                if bool(experience.intervention):
                    raise AssertionError(
                        f"ordinary {anchor_policy} anchor unexpectedly intervened"
                    )
                experiences.append(experience)
                step_cost = max(float(info.get("cost", 0.0)), 0.0)
                total_cost += step_cost
                task_return += float(info.get("task_reward", reward))
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
                    success
                    or info.get("arrive_dest", False)
                    or info.get("success", False)
                )
                observation = np.asarray(next_observation, np.float32).reshape(-1)
                if terminated or truncated:
                    break
        finally:
            env.close()

        clean = bool(
            success
            and not collision
            and not road_departure
            and total_cost <= 0.0
            and solver_finite
            and experiences
        )
        if not clean:
            raise RuntimeError(
                f"ordinary {anchor_policy} anchor failed validation: "
                f"episode={episode}, success={success}, collision={collision}, "
                f"road_departure={road_departure}, cost={total_cost}, "
                f"solver_finite={solver_finite}"
            )
        calibrated = _finalize_episode_returns(
            experiences,
            gamma=args.discount,
            q_anchor_valid=True,
        )
        for experience in calibrated:
            replay.add(experience)
        episode_rows.append(
            {
                "episode": episode,
                "steps": len(experiences),
                "return": task_return,
                "success": success,
                "collision": collision,
                "road_departure": road_departure,
                "environment_cost": total_cost,
                "solver_finite": solver_finite,
            }
        )

    return {
        "enabled": True,
        "episodes": requested,
        "transitions": len(replay),
        "batch_ratio": float(args.ordinary_champion_anchor_ratio),
        "policy": anchor_policy,
        "policy_sha256": policy_sha256,
        "actor_action_transform": "identity_clip",
        "rows": episode_rows,
    }


class _PendingDAgger:
    def __init__(self, horizon: int, decay: float):
        self.horizon = max(int(horizon), 0)
        self.decay = float(decay)
        self.items = deque()

    def push(self, experience) -> list:
        if bool(experience.intervention) and bool(experience.teacher_valid):
            pending = list(self.items)
            updated = deque()
            teacher_confidence = max(float(experience.teacher_weight), 0.0)
            for index, item in enumerate(pending):
                distance = len(pending) - index
                weight = teacher_confidence * self.decay**distance
                if bool(item.teacher_valid):
                    item = item._replace(
                        # Each pending transition already contains a Guardian
                        # query made in that transition's own ego frame.
                        teacher_weight=np.float32(
                            max(float(item.teacher_weight), weight)
                        ),
                    )
                updated.append(item)
            self.items = updated
        self.items.append(experience)
        flushed = []
        while len(self.items) > self.horizon:
            flushed.append(self.items.popleft())
        return flushed

    def flush(self) -> list:
        flushed = list(self.items)
        self.items.clear()
        return flushed


def _counterfactual_dagger_teacher(
    env,
    authority_audit,
    *,
    enabled: bool,
    trajectory_tolerance: float = 0.08,
    query_hard_margin: float = 50.0,
    allow_intervention: bool = False,
    prefer_independent_recovery_action: bool = False,
    query_anchor_certificate: bool = False,
):
    """Distill the same-state Guardian path into an ordinary-MPC target.

    A fixed straight target by itself is not a useful pre-takeover label: it
    can be locally feasible while still postponing the avoidance manoeuvre.
    Instead, solve the Guardian's straight-target response at this historical
    state, audit its CBF-on/off trajectory, and invert the protected path when
    CBF authority is present.  Nothing here advances the environment or adds
    fields to the Actor observation.
    """

    if not enabled or (
        bool(authority_audit.get("intervention", False)) and not allow_intervention
    ):
        return None, False, None
    try:
        anchor = np.asarray([1.0, 0.0], dtype=np.float32)
        anchor_certificate = None
        if query_anchor_certificate:
            # A DAgger expert must not change its recovery trigger when the
            # student changes its proposal.  Certify the fixed nominal anchor
            # in the current state and use that certificate solely to query
            # the ordinary expert; this does not advance the environment.
            anchor_certificate = env.certify_target_action(anchor)
            existing_summary = anchor_certificate.get("raw_hard_certificate", {})
        else:
            existing_summary = authority_audit.get("ordinary_hard_certificate", {})
        existing_margin = float(existing_summary.get("hard_cbf_margin", np.nan))
        has_obstacle = bool(existing_summary.get("has_obstacle", True))
        # Pending labels are activated only over a short pre-intervention
        # window.  Far from every obstacle the fixed straight label is exact
        # for this fixed straight-road task, so two additional MPC solves add
        # no information.  Begin full Guardian path distillation before the
        # barrier enters the risky range.
        if not has_obstacle or (
            np.isfinite(existing_margin) and existing_margin > query_hard_margin
        ):
            if hasattr(env, "reset_recovery_bypass_preference"):
                env.reset_recovery_bypass_preference()
            return anchor, True, None
        # Before the straight-target Guardian itself needs to intervene, ask
        # the deterministic recovery planner for a proactive bypass in this
        # exact ego frame.  Invert that protected path back to a high-level
        # action which ordinary MPC can execute.  Pending DAgger keeps this
        # label latent unless a real intervention follows shortly afterwards.
        if hasattr(env, "plan_safe_recovery_trajectory"):
            recovery_kwargs = {}
            if prefer_independent_recovery_action:
                # Pure DAgger needs a single-valued expert target for a given
                # native observation.  Fix the canonical bypass side and do
                # not let unobserved teacher action history alter the label.
                recovery_kwargs = {
                    "preferred_bypass_sign": -1.0,
                    "use_action_history": False,
                }
            recovery = env.plan_safe_recovery_trajectory(**recovery_kwargs)
            if bool(recovery.get("valid", False)) and bool(
                recovery.get("obstacle_ahead", False)
            ):
                if prefer_independent_recovery_action and hasattr(
                    env, "certify_independent_actor_target"
                ):
                    recovery_action = np.asarray(
                        recovery.get("action"), dtype=np.float32
                    ).reshape(2)
                    independent = env.certify_independent_actor_target(
                        recovery_action,
                        reference_solved=recovery["safe_solved"],
                        trajectory_tolerance=trajectory_tolerance,
                    )
                    if bool(independent.get("valid", False)) and np.all(
                        np.isfinite(recovery_action)
                    ):
                        return recovery_action, True, None
                proactive = env.derive_inverse_mpc_teacher(recovery["safe_solved"])
                proactive_action = np.asarray(
                    proactive.get("action"), dtype=np.float32
                ).reshape(2)
                if bool(proactive.get("valid", False)) and np.all(
                    np.isfinite(proactive_action)
                ):
                    return proactive_action, True, None

        guardian = (
            anchor_certificate
            if anchor_certificate is not None
            else env.certify_target_action(anchor)
        )
        distilled = _audit_low_level_cbf_authority(
            env,
            anchor,
            guardian,
            trajectory_tolerance=trajectory_tolerance,
            derive_teacher=True,
        )
        if distilled["intervention"]:
            action = distilled.get("teacher_action")
            valid = bool(distilled.get("teacher_valid", False))
            if not valid or action is None:
                return (
                    None,
                    False,
                    distilled.get("teacher_error")
                    or distilled.get("audit_error")
                    or "invalid_inverse_counterfactual_teacher",
                )
        else:
            ordinary = distilled.get("ordinary_certificate") or {}
            action = ordinary.get("raw_action", anchor)
            valid = bool(distilled.get("ordinary_solver_finite", False))
        action = np.asarray(action, dtype=np.float32).reshape(2)
        valid = bool(valid and np.all(np.isfinite(action)))
        if not valid:
            return None, False, "invalid_counterfactual_guardian_teacher"
        return action, True, None
    except Exception as exc:
        return None, False, f"{type(exc).__name__}:{exc}"


def _concatenate_batches(batches: list, rng: np.random.Generator):
    import jax

    batches = [batch for batch in batches if batch is not None]
    if not batches:
        raise ValueError("cannot concatenate an empty batch list")
    experience_type = type(batches[0])
    combined = experience_type(
        *[
            np.concatenate(
                [np.asarray(getattr(batch, field)) for batch in batches], axis=0
            )
            for field in experience_type._fields
        ]
    )
    order = rng.permutation(np.asarray(combined.reward).shape[0])
    combined = experience_type(
        *[np.asarray(getattr(combined, field))[order] for field in combined._fields]
    )
    return jax.device_put(combined)


def _mask_stale_soft_supervision(batch, current_filter_version: int):
    """Invalidate only version-sensitive Soft labels after a reserve change.

    Executed transitions remain valid task-Q data.  Hard boundary pairs remain
    valid because hard CBF geometry is immutable.  Soft Proxy/teacher labels
    from an older reserve are conservatively relabelled diagnostic-only until
    the new filter version collects a state-aligned correction.
    """

    stale = np.asarray(batch.intervention_soft, dtype=bool) & (
        np.asarray(batch.filter_version, dtype=np.int32)
        != np.int32(current_filter_version)
    )
    if not np.any(stale):
        return batch
    return batch._replace(
        proxy_pair_valid=np.asarray(batch.proxy_pair_valid, dtype=bool) & ~stale,
        proxy_pair_weight=np.where(
            stale, 0.0, np.asarray(batch.proxy_pair_weight, dtype=np.float32)
        ).astype(np.float32),
        teacher_weight=np.where(
            stale, 0.0, np.asarray(batch.teacher_weight, dtype=np.float32)
        ).astype(np.float32),
    )


def _sample_mixed_batch(
    dataset,
    online_replay,
    intervention_replay,
    ordinary_anchor_replay=None,
    *,
    batch_size: int,
    offline_ratio: float,
    intervention_ratio: float,
    ordinary_anchor_ratio: float = 0.0,
    canonical_only: bool,
    teacher_focus: bool = False,
    current_filter_version: int = 0,
    rng: np.random.Generator,
):
    anchor_count = (
        min(int(round(batch_size * ordinary_anchor_ratio)), batch_size - 1)
        if ordinary_anchor_replay is not None and len(ordinary_anchor_replay) > 0
        else 0
    )
    non_anchor_count = batch_size - anchor_count
    intervention_count = (
        min(
            int(round(non_anchor_count * intervention_ratio)),
            max(non_anchor_count - 1, 0),
        )
        if len(intervention_replay) > 0
        else 0
    )
    remaining = non_anchor_count - intervention_count
    online_count = (
        min(int(round(remaining * (1.0 - offline_ratio))), remaining)
        if len(online_replay) > 0
        else 0
    )
    offline_count = remaining - online_count
    batches = []
    if offline_count:
        batches.append(
            dataset.sample(
                offline_count,
                split="train",
                pair_fraction=(0.0 if teacher_focus else min(intervention_ratio, 0.5)),
                teacher_fraction=(
                    min(intervention_ratio, 0.5) if teacher_focus else 0.0
                ),
                canonical_only=canonical_only,
                to_jax=False,
            )
        )
    if online_count:
        batches.append(online_replay.sample(online_count, to_jax=False))
    if intervention_count:
        batches.append(intervention_replay.sample(intervention_count, to_jax=False))
    if anchor_count:
        batches.append(ordinary_anchor_replay.sample(anchor_count, to_jax=False))
    batches = [
        _mask_stale_soft_supervision(batch, current_filter_version) for batch in batches
    ]
    return _concatenate_batches(batches, rng)


class _CalibrationMonitor:
    def __init__(self, window: int, relative_tolerance: float):
        self.window = max(int(window), 4)
        self.relative_tolerance = float(relative_tolerance)
        self.series = {
            name: deque(maxlen=2 * self.window)
            for name in (
                "task_bellman_loss",
                "task_q_mean",
                "task_q_disagreement",
                "proxy_td_loss",
                "proxy_pair_loss",
                "proxy_pair_accuracy",
            )
        }

    def add(self, metrics: dict[str, float]) -> None:
        for name, values in self.series.items():
            values.append(float(metrics.get(name, np.nan)))

    def stable(self) -> bool:
        if any(len(values) < 2 * self.window for values in self.series.values()):
            return False
        for values in self.series.values():
            array = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(array)):
                return False
            old = float(np.mean(array[: self.window]))
            new = float(np.mean(array[self.window :]))
            scale = max(abs(old), abs(new), 1e-4)
            relative = abs(new - old) / scale
            if relative > self.relative_tolerance:
                return False
        return True


class _AuthorizationReserveCurriculum:
    """Episode-level curriculum over soft projection reserves only.

    The Actor never observes this state.  Hard CBF geometry, ``cbf_gamma``,
    collision clearance, road bounds, actuator limits, and solver feasibility
    tolerances remain immutable.  A clean window relaxes slowly; any rejected
    hard proposal, numerical failure, physical violation, or official cost
    tightens immediately with a larger step.
    """

    def __init__(
        self,
        args,
        *,
        enabled: bool,
        defer_to_gate: bool = False,
        start_locked: bool = False,
    ):
        self.enabled = bool(enabled)
        self.defer_to_gate = bool(defer_to_gate)
        self.locked = bool(start_locked)
        self.filter_version = 0
        self.safe_window = max(int(args.curriculum_safe_window), 1)
        self.spatial = float(args.projection_reserve)
        self.kinematic = float(args.projection_kinematic_reserve)
        self.spatial_min = float(args.projection_reserve_min)
        self.spatial_max = float(args.projection_reserve_max)
        self.spatial_relax = float(args.projection_reserve_relax_step)
        self.spatial_backoff = float(args.projection_reserve_backoff_step)
        self.kinematic_min = float(args.projection_kinematic_reserve_min)
        self.kinematic_max = float(args.projection_kinematic_reserve_max)
        self.kinematic_relax = float(args.projection_kinematic_reserve_relax_step)
        self.kinematic_backoff = float(args.projection_kinematic_reserve_backoff_step)
        self.clean_streak = 0
        self.relaxations = 0
        self.backoffs = 0
        self.history: list[dict[str, Any]] = []
        self._episode_steps = 0
        self._episode_reasons: set[str] = set()
        self._window_episodes: list[dict[str, Any]] = []

    def values(self) -> dict[str, float]:
        return {
            "projection_reserve": float(self.spatial),
            "projection_kinematic_reserve": float(self.kinematic),
        }

    def apply(self, env) -> dict[str, float]:
        return env.set_projection_reserves(
            spatial=self.spatial,
            kinematic=self.kinematic,
        )

    def unlock(self) -> None:
        """Allow reserve updates only after an Actor Champion was promoted."""

        if self.enabled:
            self.locked = False

    def _apply_update(self, update: str) -> bool:
        before = (self.spatial, self.kinematic)
        if update == "tighten":
            self.clean_streak = 0
            self.spatial = min(self.spatial_max, self.spatial + self.spatial_backoff)
            self.kinematic = min(
                self.kinematic_max, self.kinematic + self.kinematic_backoff
            )
            self.backoffs += 1
        elif update == "relax":
            self.spatial = max(self.spatial_min, self.spatial - self.spatial_relax)
            self.kinematic = max(
                self.kinematic_min, self.kinematic - self.kinematic_relax
            )
            self.clean_streak = 0
            self.relaxations += 1
        changed = before != (self.spatial, self.kinematic)
        if changed:
            self.filter_version += 1
        return changed

    def observe(self, experience, info: dict[str, Any]) -> None:
        self._episode_steps += 1
        if bool(experience.intervention_hard):
            self._episode_reasons.add("hard_proposal_rejected")
        if bool(experience.intervention_numerical) or not bool(
            experience.solver_finite
        ):
            self._episode_reasons.add("solver_or_numerical_failure")
        if max(float(info.get("cost", 0.0)), 0.0) > 0.0:
            self._episode_reasons.add("environment_cost")
        if bool(info.get("crash_vehicle", False) or info.get("crash_object", False)):
            self._episode_reasons.add("collision")
        if bool(info.get("out_of_road", False) or info.get("road_departure", False)):
            self._episode_reasons.add("road_departure")
        physical_margin = float(
            info.get("mpc_control", {}).get("physical_body_sat_margin", np.nan)
        )
        if np.isfinite(physical_margin) and physical_margin <= 0.0:
            self._episode_reasons.add("physical_sat_violation")

    def finish_episode(self, env, *, phase: str, episode: int) -> dict[str, Any]:
        reasons = sorted(self._episode_reasons)
        clean = not reasons and self._episode_steps > 0
        before = self.values()
        update = "hold"
        if self.defer_to_gate:
            self._window_episodes.append(
                {"clean": bool(clean), "reasons": reasons, "episode": int(episode)}
            )
            update = "locked" if self.locked else "deferred_to_candidate_gate"
        elif self.enabled and not self.locked and not clean:
            update = "tighten"
            self._apply_update(update)
        elif self.enabled and not self.locked and clean:
            self.clean_streak += 1
            if self.clean_streak >= self.safe_window:
                update = "relax"
                self._apply_update(update)
        applied = self.apply(env)
        event = {
            "phase": phase,
            "episode": int(episode),
            "steps": int(self._episode_steps),
            "clean": bool(clean),
            "reasons": reasons,
            "update": update if self.enabled else "disabled",
            "before": before,
            "after": self.values(),
            "applied": applied,
            "clean_streak": int(self.clean_streak),
            "filter_version": int(self.filter_version),
        }
        self.history.append(event)
        self.discard_partial_episode()
        return event

    def finish_candidate_window(self, env=None, *, phase: str, gate_index: int) -> dict:
        """Advance one slow reserve step strictly between candidate windows."""

        before = self.values()
        reasons = sorted(
            {
                reason
                for episode in self._window_episodes
                for reason in episode["reasons"]
            }
        )
        clean_episodes = sum(row["clean"] for row in self._window_episodes)
        update = "hold"
        if self.enabled and not self.locked and self._window_episodes:
            if reasons:
                update = "tighten"
            elif clean_episodes >= self.safe_window:
                update = "relax"
            self._apply_update(update)
        applied = self.apply(env) if env is not None else self.values()
        event = {
            "phase": phase,
            "gate_index": int(gate_index),
            "episodes": len(self._window_episodes),
            "clean_episodes": int(clean_episodes),
            "reasons": reasons,
            "update": (
                update
                if self.enabled and not self.locked
                else "locked" if self.enabled else "disabled"
            ),
            "before": before,
            "after": self.values(),
            "applied": applied,
            "filter_version": int(self.filter_version),
        }
        self._window_episodes.clear()
        self.history.append(event)
        return event

    def discard_partial_episode(self) -> None:
        self._episode_steps = 0
        self._episode_reasons.clear()

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "locked": self.locked,
            "defer_to_gate": self.defer_to_gate,
            "filter_version": int(self.filter_version),
            "safe_window": self.safe_window,
            "initial": {
                "projection_reserve": (
                    self.history[0]["before"]["projection_reserve"]
                    if self.history
                    else float(self.spatial)
                ),
                "projection_kinematic_reserve": (
                    self.history[0]["before"]["projection_kinematic_reserve"]
                    if self.history
                    else float(self.kinematic)
                ),
            },
            "final": self.values(),
            "relaxations": self.relaxations,
            "backoffs": self.backoffs,
            "completed_episodes": len(self.history),
            "history": self.history,
        }


def _run_online(
    algorithm,
    dataset,
    args,
    spec: MethodSpec,
    output_dir: Path,
    metrics_path: Path,
    key,
    rng: np.random.Generator,
) -> tuple[Any, dict[str, Any]]:
    import jax

    from relax.buffer.tree import TreeBuffer

    online_replay = TreeBuffer.from_guardian_calql_experience(
        dataset.summary.obs_dim,
        dataset.summary.action_dim,
        args.online_buffer_size,
        seed=args.seed + 101,
    )
    intervention_replay = TreeBuffer.from_guardian_calql_experience(
        dataset.summary.obs_dim,
        dataset.summary.action_dim,
        args.intervention_buffer_size,
        seed=args.seed + 102,
    )
    ordinary_anchor_replay = TreeBuffer.from_guardian_calql_experience(
        dataset.summary.obs_dim,
        dataset.summary.action_dim,
        max(
            int(args.ordinary_champion_anchor_episodes) * int(args.max_episode_steps),
            1,
        ),
        seed=args.seed + 103,
    )
    ordinary_anchor_summary = _collect_ordinary_champion_anchor(
        algorithm,
        args,
        ordinary_anchor_replay,
    )
    _write_json(
        output_dir / "ordinary_champion_anchor.json",
        ordinary_anchor_summary,
    )
    pending = _PendingDAgger(args.dagger_pre_steps, args.dagger_weight_decay)
    episode_experiences = []

    def store_completed(experiences):
        for experience in experiences:
            if not bool(experience.behavior_return_valid):
                raise AssertionError("unfinished online RTG cannot enter replay")
            if bool(experience.q_anchor_valid):
                raise AssertionError("fresh online state cannot be a Q anchor")
            online_replay.add(experience)
            # Keep both the exact raw/corrected boundary pair and the locally
            # relabelled pre-intervention context in the focused replay.  The
            # latter is what makes the DAgger signal visible at a meaningful
            # batch frequency instead of being diluted in the full rollout.
            if bool(experience.proxy_pair_valid) or (
                bool(experience.teacher_valid)
                and float(experience.teacher_weight) > 0.0
            ):
                intervention_replay.add(experience)

    def stage(experiences):
        episode_experiences.extend(experiences)

    def finalize_staged_episode() -> int:
        stage(pending.flush())
        finalized = _finalize_episode_returns(
            episode_experiences,
            gamma=args.discount,
            q_anchor_valid=False,
        )
        store_completed(finalized)
        episode_experiences.clear()
        return len(finalized)

    counters = {
        "warmup_environment_steps": 0,
        "online_environment_steps": 0,
        "episodes": 0,
        "interventions": 0,
        "audit_authorities": 0,
        "diagnostic_only_steps": 0,
        "actual_low_level_modifications": 0,
        "task_critic_updates": 0,
        "proxy_updates": 0,
        "actor_updates": 0,
        "filter_version_changes": 0,
        "candidate_evaluations": 0,
        "candidate_rejections": 0,
        "champion_promotions": 0,
        "performance_rejections": 0,
        "safety_rejections": 0,
        "critic_rejections": 0,
        "last_safe_updates": 0,
        "last_safe_rollbacks": 0,
        "learner_retained_after_rejection": 0,
        "actor_rollbacks": 0,
        "critic_rollbacks": 0,
        "counterfactual_teacher_queries": 0,
        "counterfactual_teacher_failures": 0,
        "low_level_authority_checks": 0,
        "hidden_low_level_interventions": 0,
        "inverse_teacher_queries": 0,
        "inverse_teacher_failures": 0,
        "rtg_finalized_transitions": 0,
        "discarded_partial_transitions": 0,
    }
    curriculum = _AuthorizationReserveCurriculum(
        args,
        enabled=(spec.use_authorization_curriculum and args.authorization_curriculum),
        defer_to_gate=True,
        start_locked=True,
    )
    post_filter_warmup_remaining = 0
    online_offline_ratio_initial = (
        args.offline_ratio_initial
        if spec.offline_ratio_initial_override is None
        else float(spec.offline_ratio_initial_override)
    )
    online_offline_ratio_final = (
        args.offline_ratio_final
        if spec.offline_ratio_final_override is None
        else float(spec.offline_ratio_final_override)
    )
    env = _make_environment(args, "hybrid", **curriculum.values())
    curriculum.apply(env)
    observation, _ = env.reset(seed=args.fixed_scene_seed)
    observation = np.asarray(observation, np.float32).reshape(-1)
    episode_step = 0
    monitor = _CalibrationMonitor(args.warmup_window, args.warmup_relative_tolerance)
    actor_hash_before_warmup = _policy_hash(algorithm.get_policy_params())
    warmup_stabilized = False
    latest = {}

    def reset_online_environment(current_env):
        if args.recreate_environment_each_episode:
            current_env.close()
            current_env = _make_environment(args, "hybrid", **curriculum.values())
            curriculum.apply(current_env)
        next_observation, _ = current_env.reset(seed=args.fixed_scene_seed)
        return current_env, np.asarray(next_observation, np.float32).reshape(-1)

    try:
        if spec.critic_warmup and args.warmup_max_steps > 0:
            while True:
                action = algorithm.get_deterministic_champion_action(observation)
                action = np.clip(
                    action + rng.normal(0.0, args.online_exploration_std, size=2),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                certificate = env.certify_target_action(action)
                authority = _audit_low_level_cbf_authority(
                    env,
                    action,
                    certificate,
                    trajectory_tolerance=args.cbf_trajectory_tolerance,
                    derive_teacher=True,
                )
                counters["low_level_authority_checks"] += 1
                counters["hidden_low_level_interventions"] += int(
                    authority["low_level_cbf_authority"]
                )
                counters["audit_authorities"] += int(
                    authority.get("audit_authority", authority["intervention"])
                )
                counters["diagnostic_only_steps"] += int(
                    authority.get("diagnostic_only", False)
                )
                counters["actual_low_level_modifications"] += int(
                    authority.get("actual_low_level_modified", False)
                )
                counters["inverse_teacher_queries"] += int(authority["intervention"])
                counters["inverse_teacher_failures"] += int(
                    authority["intervention"] and not authority["teacher_valid"]
                )
                query_teacher = bool(
                    args.counterfactual_dagger_teacher
                    and (spec.dagger_bc_only or not authority["intervention"])
                )
                (
                    teacher_action,
                    teacher_valid,
                    teacher_error,
                ) = _counterfactual_dagger_teacher(
                    env,
                    authority,
                    enabled=args.counterfactual_dagger_teacher,
                    trajectory_tolerance=args.cbf_trajectory_tolerance,
                    query_hard_margin=(args.counterfactual_dagger_query_hard_margin),
                    allow_intervention=spec.dagger_bc_only,
                    prefer_independent_recovery_action=spec.dagger_bc_only,
                    query_anchor_certificate=spec.dagger_bc_only,
                )
                counters["counterfactual_teacher_queries"] += int(query_teacher)
                counters["counterfactual_teacher_failures"] += int(
                    teacher_error is not None
                )
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.execute_certified_target(certificate)
                episode_step += 1
                truncated = bool(truncated or episode_step >= args.max_episode_steps)
                experience = _step_experience(
                    observation,
                    action,
                    reward,
                    terminated,
                    truncated,
                    next_observation,
                    certificate,
                    info,
                    soft_pair_weight=args.soft_pair_weight,
                    authority_audit=authority,
                    counterfactual_teacher_action=teacher_action,
                    counterfactual_teacher_valid=teacher_valid,
                    activate_counterfactual_teacher=spec.dagger_bc_only,
                    filter_version=curriculum.filter_version,
                    spatial_reserve=curriculum.spatial,
                    kinematic_reserve=curriculum.kinematic,
                )
                curriculum.observe(experience, info)
                stage(pending.push(experience))
                counters["warmup_environment_steps"] += 1
                counters["interventions"] += int(bool(experience.intervention))
                done = bool(terminated or truncated)
                observation = np.asarray(next_observation, np.float32).reshape(-1)
                if done:
                    counters["rtg_finalized_transitions"] += finalize_staged_episode()
                if len(online_replay) >= args.learning_starts:
                    batch = _sample_mixed_batch(
                        dataset,
                        online_replay,
                        intervention_replay,
                        ordinary_anchor_replay,
                        batch_size=args.batch_size,
                        offline_ratio=0.5,
                        intervention_ratio=args.intervention_ratio,
                        ordinary_anchor_ratio=(args.ordinary_champion_anchor_ratio),
                        canonical_only=args.bc_canonical_maneuver,
                        teacher_focus=spec.dagger_bc_only,
                        current_filter_version=curriculum.filter_version,
                        rng=rng,
                    )
                    key, update_key = jax.random.split(key)
                    latest = algorithm.online_critic_warmup(update_key, batch)
                    counters["task_critic_updates"] += 1
                    counters["proxy_updates"] += 1
                    monitor.add(latest)
                if done:
                    counters["episodes"] += 1
                    curriculum_event = curriculum.finish_episode(
                        env,
                        phase="online_critic_warmup",
                        episode=counters["episodes"],
                    )
                    _append_jsonl(
                        metrics_path,
                        {
                            **counters,
                            **curriculum_event,
                            "source_phase": curriculum_event.get("phase"),
                            "phase": "authorization_curriculum",
                        },
                    )
                    env, observation = reset_online_environment(env)
                    episode_step = 0
                step = counters["warmup_environment_steps"]
                if step % args.log_every == 0:
                    _append_jsonl(
                        metrics_path,
                        {
                            "phase": "online_critic_warmup",
                            **counters,
                            **latest,
                            **curriculum.values(),
                        },
                    )
                if done and step >= args.warmup_min_steps and monitor.stable():
                    warmup_stabilized = True
                    break
                if done and step >= args.warmup_max_steps:
                    break
        if pending.items or episode_experiences:
            raise AssertionError("critic warm-up ended with an incomplete episode")
        actor_hash_after_warmup = _policy_hash(algorithm.get_policy_params())
        if actor_hash_before_warmup != actor_hash_after_warmup:
            raise AssertionError("Actor moved during critic-only online warm-up")
        algorithm.refresh_champion_values()

        curriculum.discard_partial_episode()
        env.close()
        champion_suite = _evaluate_suite(
            algorithm,
            args,
            use_champion=True,
            **curriculum.values(),
        )
        champion_offline = _evaluate_offline(
            algorithm,
            dataset,
            "validation",
            batch_size=args.evaluation_batch_size,
            use_champion=True,
        )
        _write_json(
            output_dir / "online_initial_champion.json",
            {"suite": champion_suite, "offline": champion_offline},
        )
        initial_champion_failures = _closed_loop_safety_failures(
            champion_suite,
            args,
        )
        _write_json(
            output_dir / "online_initial_champion_gate.json",
            {
                "passed": not initial_champion_failures,
                "failures": initial_champion_failures,
            },
        )
        if initial_champion_failures:
            raise RuntimeError(
                "initial Champion failed the closed-loop safety gate: "
                + ", ".join(initial_champion_failures)
            )
        env = _make_environment(args, "hybrid", **curriculum.values())
        curriculum.apply(env)
        observation, _ = env.reset(seed=args.fixed_scene_seed)
        observation = np.asarray(observation, np.float32).reshape(-1)
        episode_step = 0

        next_candidate_step = max(int(args.candidate_every), 1)
        while (
            counters["online_environment_steps"] < args.online_steps
            or episode_experiences
            or pending.items
        ):
            action = algorithm.get_deterministic_action(observation)
            action = np.clip(
                action + rng.normal(0.0, args.online_exploration_std, size=2),
                -1.0,
                1.0,
            ).astype(np.float32)
            certificate = env.certify_target_action(action)
            authority = _audit_low_level_cbf_authority(
                env,
                action,
                certificate,
                trajectory_tolerance=args.cbf_trajectory_tolerance,
                derive_teacher=True,
            )
            counters["low_level_authority_checks"] += 1
            counters["hidden_low_level_interventions"] += int(
                authority["low_level_cbf_authority"]
            )
            counters["audit_authorities"] += int(
                authority.get("audit_authority", authority["intervention"])
            )
            counters["diagnostic_only_steps"] += int(
                authority.get("diagnostic_only", False)
            )
            counters["actual_low_level_modifications"] += int(
                authority.get("actual_low_level_modified", False)
            )
            counters["inverse_teacher_queries"] += int(authority["intervention"])
            counters["inverse_teacher_failures"] += int(
                authority["intervention"] and not authority["teacher_valid"]
            )
            query_teacher = bool(
                args.counterfactual_dagger_teacher
                and (spec.dagger_bc_only or not authority["intervention"])
            )
            (
                teacher_action,
                teacher_valid,
                teacher_error,
            ) = _counterfactual_dagger_teacher(
                env,
                authority,
                enabled=args.counterfactual_dagger_teacher,
                trajectory_tolerance=args.cbf_trajectory_tolerance,
                query_hard_margin=args.counterfactual_dagger_query_hard_margin,
                allow_intervention=spec.dagger_bc_only,
                prefer_independent_recovery_action=spec.dagger_bc_only,
                query_anchor_certificate=spec.dagger_bc_only,
            )
            counters["counterfactual_teacher_queries"] += int(query_teacher)
            counters["counterfactual_teacher_failures"] += int(
                teacher_error is not None
            )
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.execute_certified_target(certificate)
            episode_step += 1
            truncated = bool(truncated or episode_step >= args.max_episode_steps)
            experience = _step_experience(
                observation,
                action,
                reward,
                terminated,
                truncated,
                next_observation,
                certificate,
                info,
                soft_pair_weight=args.soft_pair_weight,
                authority_audit=authority,
                counterfactual_teacher_action=teacher_action,
                counterfactual_teacher_valid=teacher_valid,
                activate_counterfactual_teacher=spec.dagger_bc_only,
                filter_version=curriculum.filter_version,
                spatial_reserve=curriculum.spatial,
                kinematic_reserve=curriculum.kinematic,
            )
            curriculum.observe(experience, info)
            stage(pending.push(experience))
            counters["online_environment_steps"] += 1
            counters["interventions"] += int(bool(experience.intervention))
            done = bool(terminated or truncated)
            observation = np.asarray(next_observation, np.float32).reshape(-1)
            if done:
                counters["rtg_finalized_transitions"] += finalize_staged_episode()
            progress = min(
                counters["online_environment_steps"] / max(args.online_steps, 1),
                1.0,
            )
            offline_ratio = online_offline_ratio_initial + progress * (
                online_offline_ratio_final - online_offline_ratio_initial
            )
            if len(online_replay) >= args.learning_starts:
                for gradient_index in range(args.gradient_steps):
                    batch = _sample_mixed_batch(
                        dataset,
                        online_replay,
                        intervention_replay,
                        ordinary_anchor_replay,
                        batch_size=args.batch_size,
                        offline_ratio=offline_ratio,
                        intervention_ratio=args.intervention_ratio,
                        ordinary_anchor_ratio=(args.ordinary_champion_anchor_ratio),
                        canonical_only=args.bc_canonical_maneuver,
                        teacher_focus=spec.dagger_bc_only,
                        current_filter_version=curriculum.filter_version,
                        rng=rng,
                    )
                    key, update_key = jax.random.split(key)
                    update_actor = bool(
                        (counters["online_environment_steps"] + gradient_index)
                        % args.policy_delay
                        == 0
                    )
                    if spec.dagger_bc_only:
                        latest = algorithm.pretrain_bc(update_key, batch)
                        latest["bc_dagger_only"] = 1.0
                        counters["actor_updates"] += 1
                    elif post_filter_warmup_remaining > 0:
                        latest = algorithm.online_critic_warmup(update_key, batch)
                        post_filter_warmup_remaining -= 1
                        counters["task_critic_updates"] += 1
                        counters["proxy_updates"] += 1
                    else:
                        latest = algorithm.online_joint_update(
                            update_key, batch, update_actor=update_actor
                        )
                        counters["task_critic_updates"] += 1
                        counters["proxy_updates"] += 1
                        counters["actor_updates"] += int(update_actor)
            if done:
                counters["episodes"] += 1
                curriculum_event = curriculum.finish_episode(
                    env,
                    phase="online_joint",
                    episode=counters["episodes"],
                )
                _append_jsonl(
                    metrics_path,
                    {
                        **counters,
                        **curriculum_event,
                        "source_phase": curriculum_event.get("phase"),
                        "phase": "authorization_curriculum",
                    },
                )
                env, observation = reset_online_environment(env)
                episode_step = 0
            online_step = counters["online_environment_steps"]
            if online_step % args.log_every == 0:
                _append_jsonl(
                    metrics_path,
                    {
                        "phase": "online_joint",
                        **counters,
                        **latest,
                        "offline_ratio": offline_ratio,
                        "online_replay_size": len(online_replay),
                        "intervention_replay_size": len(intervention_replay),
                        "ordinary_anchor_replay_size": len(ordinary_anchor_replay),
                        **curriculum.values(),
                    },
                )
            if args.candidate_every > 0 and done and online_step >= next_candidate_step:
                while next_candidate_step <= online_step:
                    next_candidate_step += args.candidate_every
                curriculum.discard_partial_episode()
                env.close()
                env = None
                # Preserve the trained candidate before an acceptance failure
                # deliberately rolls deployment state back to the Champion.
                # This keeps diagnostics/retraining reproducible without
                # weakening the deployable-policy gate.
                algorithm.save(str(output_dir / f"candidate_{online_step:08d}.pkl"))
                candidate_suite = _evaluate_suite(
                    algorithm,
                    args,
                    use_champion=False,
                    **curriculum.values(),
                )
                candidate_offline = _evaluate_offline(
                    algorithm,
                    dataset,
                    "validation",
                    batch_size=args.evaluation_batch_size,
                )
                counters["candidate_evaluations"] += 1
                first_promotion = False
                candidate_score = _candidate_score(candidate_suite, args)
                champion_score = _candidate_score(champion_suite, args)
                score_improvement = candidate_score["score"] - champion_score["score"]
                passed, failures, critic_failure = _candidate_gate(
                    candidate_suite,
                    champion_suite,
                    candidate_offline,
                    champion_offline,
                    args,
                )
                learning_failures = _online_learning_gate_failures(
                    candidate_suite, counters, args
                )
                failures.extend(learning_failures)
                passed = bool(passed and not learning_failures)
                p_validation_delta = float(candidate_offline["p_policy_minus_champion"])
                if algorithm.config.proxy_objective != "dependency_regression":
                    p_validation_delta = -p_validation_delta
                trust_validation = algorithm.adapt_trust_from_validation(
                    q_policy_minus_champion=float(
                        candidate_offline["q_policy_minus_champion"]
                    ),
                    p_policy_minus_champion=p_validation_delta,
                )
                rejection_class = "promote"
                if passed:
                    algorithm.promote_champion()
                    champion_suite = candidate_suite
                    champion_offline = candidate_offline
                    counters["champion_promotions"] += 1
                    counters["last_safe_updates"] += 1
                    first_promotion = counters["champion_promotions"] == 1
                else:
                    counters["candidate_rejections"] += 1
                    rejection_class = _candidate_rejection_class(
                        failures,
                        critic_failure=critic_failure,
                    )
                    if rejection_class == "critic":
                        algorithm.rollback_to_champion(restore_critics=True)
                        counters["critic_rejections"] += 1
                        counters["critic_rollbacks"] += 1
                        counters["actor_rollbacks"] += 1
                    elif rejection_class == "safety":
                        algorithm.rollback_to_last_safe()
                        counters["safety_rejections"] += 1
                        counters["last_safe_rollbacks"] += 1
                        counters["actor_rollbacks"] += 1
                    else:
                        algorithm.retain_safe_learner_after_rejection()
                        counters["performance_rejections"] += 1
                        counters["last_safe_updates"] += 1
                        counters["learner_retained_after_rejection"] += 1
                previous_filter_version = curriculum.filter_version
                curriculum_event = curriculum.finish_candidate_window(
                    phase="candidate_window_boundary",
                    gate_index=counters["candidate_evaluations"],
                )
                if curriculum.filter_version != previous_filter_version:
                    counters["filter_version_changes"] += 1
                    post_filter_warmup_remaining = max(
                        post_filter_warmup_remaining,
                        args.post_filter_proxy_warmup_steps,
                    )
                if first_promotion:
                    # The first full Q/P/Actor Champion is established under a
                    # fixed filter.  Only now may later candidate windows alter
                    # soft reserves.
                    curriculum.unlock()
                _append_jsonl(
                    metrics_path,
                    {
                        "phase": "candidate_gate",
                        **counters,
                        "passed": passed,
                        "failures": failures,
                        "candidate": candidate_suite,
                        "candidate_offline": candidate_offline,
                        "candidate_score": candidate_score,
                        "champion_score_before_gate": champion_score,
                        "score_improvement": score_improvement,
                        "rejection_class": rejection_class,
                        "trust_validation": trust_validation,
                        "curriculum_boundary": curriculum_event,
                        "post_filter_warmup_remaining": (post_filter_warmup_remaining),
                        "authorization_curriculum": curriculum.summary(),
                    },
                )
                algorithm.save(str(output_dir / f"checkpoint_{online_step:08d}.pkl"))
                if first_promotion and args.stop_after_first_promotion:
                    break
                env = _make_environment(args, "hybrid", **curriculum.values())
                curriculum.apply(env)
                observation, _ = env.reset(seed=args.fixed_scene_seed)
                observation = np.asarray(observation, np.float32).reshape(-1)
                episode_step = 0
    finally:
        counters["discarded_partial_transitions"] += len(episode_experiences) + len(
            pending.items
        )
        episode_experiences.clear()
        pending.flush()
        if env is not None:
            env.close()

    # Only an accepted Champion is a deployable final result.
    algorithm.rollback_to_champion(restore_critics=False)
    # With zero online updates the just-audited initial Champion is bitwise
    # unchanged, so a duplicate MetaDrive suite adds runtime but no evidence.
    final_suite = (
        champion_suite
        if args.online_steps == 0
        else _evaluate_suite(
            algorithm,
            args,
            use_champion=True,
            **curriculum.values(),
        )
    )
    final_offline = _evaluate_offline(
        algorithm,
        dataset,
        "test",
        batch_size=args.evaluation_batch_size,
    )
    return key, {
        **counters,
        "warmup_stabilized": warmup_stabilized,
        "warmup_actor_hash_before": actor_hash_before_warmup,
        "warmup_actor_hash_after": actor_hash_after_warmup,
        "online_replay_size": len(online_replay),
        "intervention_replay_size": len(intervention_replay),
        "ordinary_anchor_replay_size": len(ordinary_anchor_replay),
        "ordinary_champion_anchor": ordinary_anchor_summary,
        "authorization_curriculum": curriculum.summary(),
        "online_offline_ratio_initial": online_offline_ratio_initial,
        "online_offline_ratio_final": online_offline_ratio_final,
        "champion_suite": champion_suite,
        "champion_validation": champion_offline,
        "final_suite": final_suite,
        "final_test": final_offline,
    }


def _configure_runtime(args) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.jax_memory_fraction)
    )
    # Do not hide the CPU backend with JAX_PLATFORMS=cuda.  The learning
    # networks still use JAX's default GPU and are asserted below, while
    # FlatSpaceNMPC deliberately selects jax.devices("cpu")[0].  Keeping both
    # backends visible prevents the simulator/control solver from competing
    # with Actor/Critic training for GPU memory.


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_runtime(args)
    if args.method == "v2_2_simple":
        from scripts.train_guardian_calql_v2_2_simple import main_from_args

        return main_from_args(args)
    import jax

    from relax.algorithm.guardian_calql import (
        GuardianCalQLConfig,
        GuardianCalQLSafeSAC,
    )
    from relax.network.safe_sac_jax import create_guardian_calql_net
    from relax.offline.metasafe_guardian_dataset import GuardianMetaSafeDataset

    started = time.time()
    backend = jax.default_backend()
    devices = [str(device) for device in jax.devices()]
    if args.require_gpu and backend != "gpu":
        raise RuntimeError(
            f"formal training requires a GPU, got backend={backend}, devices={devices}"
        )
    spec = METHODS[args.method]
    run_name = args.run_name or (
        f"fixed_seed{args.fixed_scene_seed}_{args.method}_seed{args.seed}"
    )
    output_dir = (args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"
    dataset = GuardianMetaSafeDataset(
        args.dataset,
        gamma=args.discount,
        split_seed=args.seed,
        soft_pair_weight=args.soft_pair_weight,
        soft_teacher_weight=args.soft_teacher_weight,
        exclude_legacy_reset_step=args.post_reset_warmup_steps > 0,
        expected_sha256=args.expected_dataset_sha256 or None,
    )
    hidden_sizes = tuple(
        int(value.strip()) for value in args.hidden_sizes.split(",") if value.strip()
    )
    if not hidden_sizes or any(value < 1 for value in hidden_sizes):
        raise ValueError("hidden sizes must be positive")
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
    proxy_enabled = spec.train_proxy or spec.use_proxy_actor
    config = GuardianCalQLConfig(
        gamma=args.discount,
        proxy_gamma=args.proxy_discount,
        tau=args.tau,
        critic_learning_rate=args.critic_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        proxy_learning_rate=args.proxy_learning_rate,
        alpha_learning_rate=args.alpha_learning_rate,
        q_lcb_beta=args.q_lcb_beta,
        calql_weight=args.calql_weight,
        online_critic_mode=args.online_critic_mode,
        q_mc_weight=args.q_mc_weight,
        q_mc_huber_delta=args.q_mc_huber_delta,
        q_anchor_weight_initial=(
            args.q_anchor_weight_initial if spec.use_q_anchor else 0.0
        ),
        q_anchor_weight_final=(
            args.q_anchor_weight_final if spec.use_q_anchor else 0.0
        ),
        q_anchor_decay_steps=args.q_anchor_decay_steps,
        proxy_objective=(
            "dependency_regression" if spec.dependency_proxy else "signed_td"
        ),
        proxy_td_weight=(args.proxy_td_loss_weight if proxy_enabled else 0.0),
        proxy_td_target_min=args.proxy_td_target_min,
        proxy_td_target_max=args.proxy_td_target_max,
        proxy_regression_weight=(
            args.proxy_regression_loss_weight if proxy_enabled else 0.0
        ),
        proxy_pair_loss_weight=(args.proxy_pair_loss_weight if proxy_enabled else 0.0),
        proxy_pair_margin=args.proxy_pair_margin,
        proxy_l2_weight=(args.proxy_l2_weight if proxy_enabled else 0.0),
        proxy_min_value=args.proxy_min_value,
        proxy_dependency_safe_threshold=args.proxy_dependency_safe_threshold,
        proxy_relative_margin=args.proxy_relative_margin,
        proxy_actor_weight_init=(
            args.proxy_actor_weight if spec.use_proxy_actor else 0.0
        ),
        proxy_dual_learning_rate=(
            args.proxy_dual_learning_rate if spec.use_proxy_actor else 0.0
        ),
        proxy_target=args.proxy_target,
        proxy_actor_boundary_only=args.proxy_actor_boundary_only,
        actor_sac_weight=args.actor_sac_weight,
        awac_weight=args.awac_beta if spec.use_awac else 0.0,
        awac_temperature=args.awac_temperature,
        awac_max_weight=args.awac_max_weight,
        teacher_bc_weight=args.teacher_bc_weight if spec.use_dagger else 0.0,
        champion_action_weight=(
            args.champion_action_weight if spec.use_trust_region else 0.0
        ),
        champion_boundary_action_weight=(
            args.champion_boundary_action_weight if spec.use_trust_region else 0.0
        ),
        champion_kl_weight=(args.champion_kl_weight if spec.use_trust_region else 0.0),
        champion_kl_min=(args.champion_kl_min if spec.use_trust_region else 0.0),
        champion_kl_target=(args.champion_kl_target if spec.use_trust_region else 0.0),
        champion_kl_dual_learning_rate=(
            args.champion_kl_dual_learning_rate if spec.use_trust_region else 0.0
        ),
        champion_kl_failure_multiplier=(
            args.champion_kl_failure_multiplier if spec.use_trust_region else 1.0
        ),
        champion_kl_validation_relax_multiplier=(
            args.champion_kl_validation_relax_multiplier
            if spec.use_trust_region
            else 1.0
        ),
        champion_kl_max=(args.champion_kl_max if spec.use_trust_region else 0.0),
        initial_policy_kl_weight=args.initial_policy_kl_weight,
    )
    algorithm = GuardianCalQLSafeSAC(network, params, config=config)
    rng = np.random.default_rng(args.seed + 77)
    online_initial_checkpoint_sha256 = None
    online_restart_champion_policy_sha256 = None
    online_loaded_proxy_dual_reset = None
    online_loaded_actor_optimizers_reset = None
    online_initial_policy_checkpoint_sha256 = None
    online_initial_policy_source_kind = None
    if args.online_initial_checkpoint is not None:
        checkpoint = args.online_initial_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not spec.online:
            raise ValueError(
                "--online-initial-checkpoint is only valid for an online method"
            )
        algorithm.load(str(checkpoint))
        online_initial_checkpoint_sha256 = _file_sha256(checkpoint)
        if args.online_restart_from_loaded_champion:
            online_restart_champion_policy_sha256 = _policy_hash(
                algorithm.champion_state.params.policy
            )
            algorithm.restart_from_champion()
            if (
                _policy_hash(algorithm.get_policy_params())
                != online_restart_champion_policy_sha256
            ):
                raise AssertionError(
                    "loaded Champion Actor changed during explicit restart"
                )
            algorithm.save(str(output_dir / "stage0_restarted_loaded_champion.pkl"))
        if args.online_reset_loaded_actor_optimizers:
            online_loaded_actor_optimizers_reset = algorithm.reset_actor_optimizers()
            algorithm.save(str(output_dir / "stage0_reset_loaded_actor_optimizers.pkl"))
        if args.online_reset_loaded_proxy_dual:
            online_loaded_proxy_dual_reset = algorithm.reset_proxy_actor_dual()
            algorithm.save(str(output_dir / "stage0_reset_loaded_proxy_dual.pkl"))

    if args.online_initial_policy_checkpoint is not None:
        policy_checkpoint = args.online_initial_policy_checkpoint.expanduser().resolve()
        if not policy_checkpoint.is_file():
            raise FileNotFoundError(policy_checkpoint)
        if not spec.online:
            raise ValueError(
                "--online-initial-policy-checkpoint is only valid for an "
                "online method"
            )
        (
            imported_policy,
            online_initial_policy_source_kind,
        ) = _load_compatible_actor_policy(
            policy_checkpoint, algorithm.get_policy_params()
        )
        algorithm.restore_policy(imported_policy)
        algorithm.mark_policy_as_initial()
        algorithm.save(str(output_dir / "stage0_imported_actor.pkl"))
        online_initial_policy_checkpoint_sha256 = _file_sha256(policy_checkpoint)
    _write_json(
        output_dir / "config.json",
        {
            "args": vars(args),
            "method_spec": asdict(spec),
            "algorithm_config": asdict(config),
            "dataset": asdict(dataset.summary),
            "split_episode_ids": dataset.split_episode_ids(),
            "jax_backend": backend,
            "jax_devices": devices,
            "jax_version": jax.__version__,
            "initial_checkpoint_sha256": online_initial_checkpoint_sha256,
            "restart_champion_policy_sha256": (online_restart_champion_policy_sha256),
            "loaded_proxy_dual_reset": online_loaded_proxy_dual_reset,
            "loaded_actor_optimizers_reset": (online_loaded_actor_optimizers_reset),
            "initial_policy_checkpoint_sha256": (
                online_initial_policy_checkpoint_sha256
            ),
            "initial_policy_source_kind": online_initial_policy_source_kind,
            "loaded_checkpoint_format": algorithm.loaded_checkpoint_format,
            "intervention_accounting_version": (INTERVENTION_ACCOUNTING_VERSION),
            "intervention_accounting_definition": (INTERVENTION_ACCOUNTING_DEFINITION),
        },
    )

    bc_initial_validation = _evaluate_offline(
        algorithm,
        dataset,
        "validation",
        batch_size=args.evaluation_batch_size,
    )
    best_bc = float("inf")
    best_bc_score = float("inf")
    best_policy = _copy_tree(algorithm.get_policy_params())
    bc_no_improvement = 0
    bc_actual_steps = 0
    bc_candidate_policies = []
    for step in range(1, args.bc_gradient_steps + 1):
        batch = dataset.sample(
            args.batch_size,
            split="train",
            teacher_fraction=args.bc_intervention_ratio,
            canonical_only=args.bc_canonical_maneuver,
            to_jax=True,
        )
        key, update_key = jax.random.split(key)
        latest = algorithm.pretrain_bc(update_key, batch)
        if step % args.log_every == 0:
            _append_jsonl(metrics_path, {"phase": "offline_bc", "step": step, **latest})
        if step % args.validation_every == 0 or step == args.bc_gradient_steps:
            validation = _evaluate_offline(
                algorithm,
                dataset,
                "validation",
                batch_size=args.evaluation_batch_size,
            )
            # Closed-loop safety is non-monotonic in BC loss.  Keep each
            # validation checkpoint so a later ordinary rollout can select
            # the earliest safe policy instead of being forced to use only
            # the final gradient step.
            algorithm.save(str(output_dir / f"bc_candidate_{step:08d}.pkl"))
            _append_jsonl(
                metrics_path,
                {"phase": "offline_bc_validation", "step": step, **validation},
            )
            controlled_metric = (
                "bc_mse_canonical_actor_controlled"
                if args.bc_canonical_maneuver
                else "bc_mse_actor_controlled"
            )
            intervention_metric = (
                "bc_mse_canonical_teacher_focus"
                if args.bc_canonical_maneuver
                else "bc_mse_teacher_focus"
            )
            bc_score = float(validation[controlled_metric]) + (
                args.bc_selection_intervention_weight
                * float(validation[intervention_metric])
            )
            bc_candidate_policies.append(
                {
                    "step": int(step),
                    "bc_mse": float(validation["bc_mse"]),
                    "bc_score": float(bc_score),
                    "policy": _copy_tree(algorithm.get_policy_params()),
                }
            )
            if bc_score < best_bc_score:
                best_bc = float(validation["bc_mse"])
                best_bc_score = bc_score
                best_policy = _copy_tree(algorithm.get_policy_params())
                bc_no_improvement = 0
            else:
                bc_no_improvement += 1
            bc_actual_steps = step
            if (
                step >= args.bc_min_gradient_steps
                and bc_no_improvement >= args.early_stop_patience
            ):
                break
        bc_actual_steps = step
    bc_closed_loop_selection = {
        "enabled": bool(
            args.closed_loop_bc_selection and not args.smoke and bc_actual_steps > 0
        ),
        "selected_step": None,
        "selected_bc_mse": None,
        "selected_bc_score": None,
        "rows": [],
    }
    if bc_closed_loop_selection["enabled"]:
        for candidate in bc_candidate_policies:
            algorithm.restore_policy(candidate["policy"])
            suite = _evaluate_suite(
                algorithm,
                args,
                use_champion=False,
            )
            failures = _closed_loop_safety_failures(suite, args)
            row = {
                "step": candidate["step"],
                "bc_mse": candidate["bc_mse"],
                "bc_score": candidate["bc_score"],
                "passed": not failures,
                "failures": failures,
                "suite": suite,
            }
            bc_closed_loop_selection["rows"].append(row)
            _write_json(
                output_dir / "bc_closed_loop_selection.json",
                bc_closed_loop_selection,
            )
            if not failures:
                best_policy = _copy_tree(candidate["policy"])
                best_bc = candidate["bc_mse"]
                best_bc_score = candidate["bc_score"]
                bc_closed_loop_selection.update(
                    {
                        "selected_step": candidate["step"],
                        "selected_bc_mse": candidate["bc_mse"],
                        "selected_bc_score": candidate["bc_score"],
                    }
                )
                break
        _write_json(
            output_dir / "bc_closed_loop_selection.json",
            bc_closed_loop_selection,
        )
        if bc_closed_loop_selection["selected_step"] is None:
            raise RuntimeError(
                "no BC validation checkpoint passed the closed-loop "
                "Hybrid/Ordinary safety gate"
            )
    else:
        _write_json(
            output_dir / "bc_closed_loop_selection.json",
            bc_closed_loop_selection,
        )
    if bc_actual_steps == 0:
        controlled_metric = (
            "bc_mse_canonical_actor_controlled"
            if args.bc_canonical_maneuver
            else "bc_mse_actor_controlled"
        )
        intervention_metric = (
            "bc_mse_canonical_teacher_focus"
            if args.bc_canonical_maneuver
            else "bc_mse_teacher_focus"
        )
        best_bc = float(bc_initial_validation["bc_mse"])
        best_bc_score = float(bc_initial_validation[controlled_metric]) + (
            args.bc_selection_intervention_weight
            * float(bc_initial_validation[intervention_metric])
        )
    # A complete online checkpoint already contains distinct
    # Champion/LastSafe/Learner actors.  With a zero BC budget this is an
    # online-only continuation, so do not collapse those three states by
    # re-declaring the current Learner as the initial Champion.
    if bc_actual_steps > 0:
        algorithm.restore_policy(best_policy)
        algorithm.mark_policy_as_initial()
    bc_validation = _evaluate_offline(
        algorithm,
        dataset,
        "validation",
        batch_size=args.evaluation_batch_size,
    )
    algorithm.save(str(output_dir / "stage1_bc.pkl"))

    critic_actual_steps = 0
    if spec.critic_mode:
        best_q_error = float("inf")
        best_q_state = _copy_tree(algorithm.state)
        q_no_improvement = 0
        for step in range(1, args.critic_gradient_steps + 1):
            batch = dataset.sample(args.batch_size, split="train", to_jax=True)
            key, update_key = jax.random.split(key)
            latest = algorithm.offline_critic_update(
                update_key, batch, mode=spec.critic_mode
            )
            if step % args.log_every == 0:
                _append_jsonl(
                    metrics_path,
                    {"phase": f"offline_{spec.critic_mode}", "step": step, **latest},
                )
            if step % args.validation_every == 0 or step == args.critic_gradient_steps:
                validation = _evaluate_offline(
                    algorithm,
                    dataset,
                    "validation",
                    batch_size=args.evaluation_batch_size,
                )
                _append_jsonl(
                    metrics_path,
                    {"phase": "offline_q_validation", "step": step, **validation},
                )
                if validation["q_calibration_mae"] < best_q_error:
                    best_q_error = float(validation["q_calibration_mae"])
                    best_q_state = _copy_tree(algorithm.state)
                    q_no_improvement = 0
                else:
                    q_no_improvement += 1
                critic_actual_steps = step
                if (
                    step >= args.critic_min_gradient_steps
                    and q_no_improvement >= args.early_stop_patience
                ):
                    break
            critic_actual_steps = step
        algorithm.state = jax.device_put(best_q_state)
        algorithm.save(str(output_dir / "stage2_task_critic.pkl"))

    proxy_actual_steps = 0
    if spec.train_proxy:
        best_proxy_score = float("-inf")
        best_proxy_state = _copy_tree(algorithm.state)
        proxy_no_improvement = 0
        for step in range(1, args.proxy_gradient_steps + 1):
            batch = dataset.sample(
                args.batch_size,
                split="train",
                pair_fraction=0.5,
                to_jax=True,
            )
            key, update_key = jax.random.split(key)
            latest = algorithm.offline_proxy_update(update_key, batch)
            if step % args.log_every == 0:
                _append_jsonl(
                    metrics_path,
                    {"phase": "offline_proxy", "step": step, **latest},
                )
            if step % args.validation_every == 0 or step == args.proxy_gradient_steps:
                validation = _evaluate_offline(
                    algorithm,
                    dataset,
                    "validation",
                    batch_size=args.evaluation_batch_size,
                )
                _append_jsonl(
                    metrics_path,
                    {"phase": "offline_proxy_validation", "step": step, **validation},
                )
                accuracy = float(validation["proxy_pair_accuracy"])
                auc = float(validation["proxy_future_intervention_auc_h20"])
                proxy_score = accuracy + (auc if np.isfinite(auc) else 0.0)
                if proxy_score > best_proxy_score:
                    best_proxy_score = proxy_score
                    best_proxy_state = _copy_tree(algorithm.state)
                    proxy_no_improvement = 0
                else:
                    proxy_no_improvement += 1
                proxy_actual_steps = step
                if (
                    step >= args.proxy_min_gradient_steps
                    and proxy_no_improvement >= args.early_stop_patience
                ):
                    break
            proxy_actual_steps = step
        algorithm.state = jax.device_put(best_proxy_state)
        algorithm.save(str(output_dir / "stage3_proxy.pkl"))

    offline_actor_actual_steps = 0
    if spec.offline_actor:
        for step in range(1, args.offline_actor_gradient_steps + 1):
            batch = dataset.sample(
                args.batch_size,
                split="train",
                pair_fraction=0.25 if spec.train_proxy else 0.0,
                canonical_only=args.bc_canonical_maneuver,
                to_jax=True,
            )
            key, update_key = jax.random.split(key)
            latest = algorithm.offline_actor_update(update_key, batch)
            if step % args.log_every == 0:
                _append_jsonl(
                    metrics_path,
                    {"phase": "offline_actor", "step": step, **latest},
                )
            offline_actor_actual_steps = step
        algorithm.save(str(output_dir / "stage4_offline_actor.pkl"))

    if spec.use_q_anchor and (
        critic_actual_steps > 0 or not algorithm.offline_q_reference_ready
    ):
        algorithm.mark_offline_q_reference()
        algorithm.save(str(output_dir / "stage5_online_q_reference.pkl"))

    offline_validation = _evaluate_offline(
        algorithm,
        dataset,
        "validation",
        batch_size=args.evaluation_batch_size,
    )
    offline_test = _evaluate_offline(
        algorithm,
        dataset,
        "test",
        batch_size=args.evaluation_batch_size,
    )
    online_summary = None
    if spec.online and args.online_steps >= 0:
        key, online_summary = _run_online(
            algorithm,
            dataset,
            args,
            spec,
            output_dir,
            metrics_path,
            key,
            rng,
        )
    elif not args.smoke:
        champion_suite = _evaluate_suite(algorithm, args, use_champion=True)
        if _policy_hash(algorithm.get_policy_params()) == _policy_hash(
            algorithm.champion_state.params.policy
        ):
            candidate_suite = champion_suite
        else:
            candidate_suite = _evaluate_suite(algorithm, args, use_champion=False)
        passed, failures, critic_failure = _candidate_gate(
            candidate_suite,
            champion_suite,
            offline_validation,
            bc_validation,
            args,
        )
        if passed:
            algorithm.promote_champion()
        else:
            algorithm.rollback_to_champion(restore_critics=critic_failure)
        online_summary = {
            "offline_only_candidate_gate": {
                "passed": passed,
                "failures": failures,
                "critic_rollback": critic_failure,
            },
            "offline_only_candidate_suite": candidate_suite,
            "offline_only_champion_suite": champion_suite,
        }
    else:
        # Smoke mode validates code paths without spending time in MetaDrive.
        algorithm.promote_champion()

    algorithm.save(str(output_dir / "final_champion.pkl"))
    summary = {
        "method": args.method,
        "seed": args.seed,
        "method_spec": asdict(spec),
        "algorithm_config": asdict(config),
        "fixed_scene_seed": args.fixed_scene_seed,
        "dataset": asdict(dataset.summary),
        "jax_backend": backend,
        "jax_devices": devices,
        "training_budget": {
            "online_initial_checkpoint": (
                str(args.online_initial_checkpoint.resolve())
                if args.online_initial_checkpoint is not None
                else None
            ),
            "online_initial_checkpoint_sha256": online_initial_checkpoint_sha256,
            "online_restart_from_loaded_champion": (
                args.online_restart_from_loaded_champion
            ),
            "online_restart_champion_policy_sha256": (
                online_restart_champion_policy_sha256
            ),
            "online_reset_loaded_proxy_dual": (args.online_reset_loaded_proxy_dual),
            "online_loaded_proxy_dual_reset": online_loaded_proxy_dual_reset,
            "online_reset_loaded_actor_optimizers": (
                args.online_reset_loaded_actor_optimizers
            ),
            "online_loaded_actor_optimizers_reset": (
                online_loaded_actor_optimizers_reset
            ),
            "online_initial_policy_checkpoint": (
                str(args.online_initial_policy_checkpoint.resolve())
                if args.online_initial_policy_checkpoint is not None
                else None
            ),
            "online_initial_policy_checkpoint_sha256": (
                online_initial_policy_checkpoint_sha256
            ),
            "online_initial_policy_source_kind": online_initial_policy_source_kind,
            "bc_gradient_steps_requested": args.bc_gradient_steps,
            "bc_gradient_steps": bc_actual_steps,
            "bc_effective_train_epochs": (
                bc_actual_steps
                * args.batch_size
                / max(dataset.split_transition_indices("train").size, 1)
            ),
            "critic_gradient_steps_requested": args.critic_gradient_steps,
            "critic_gradient_steps": critic_actual_steps,
            "critic_effective_train_epochs": (
                critic_actual_steps
                * args.batch_size
                / max(dataset.split_transition_indices("train").size, 1)
                if spec.critic_mode
                else 0.0
            ),
            "proxy_gradient_steps_requested": args.proxy_gradient_steps,
            "proxy_gradient_steps": proxy_actual_steps,
            "proxy_pair_oversampling_fraction": (0.5 if spec.train_proxy else 0.0),
            "offline_actor_gradient_steps_requested": (
                args.offline_actor_gradient_steps
            ),
            "offline_actor_gradient_steps": offline_actor_actual_steps,
            "online_environment_steps": args.online_steps if spec.online else 0,
        },
        "bc_initial_validation": bc_initial_validation,
        "bc_best_validation_mse": best_bc,
        "bc_best_selection_score": best_bc_score,
        "bc_closed_loop_selection": bc_closed_loop_selection,
        "bc_validation": bc_validation,
        "offline_validation": offline_validation,
        "offline_test": offline_test,
        "online": online_summary,
        "final_policy_sha256": _policy_hash(algorithm.get_policy_params()),
        "wall_time_seconds": time.time() - started,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            _json_sanitize(summary),
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
