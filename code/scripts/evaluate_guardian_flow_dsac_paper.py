#!/usr/bin/env python3
"""Evaluate a paper checkpoint or the non-learning Guardian/MPC teacher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.utils.flow_paper import PAPER_SCENE_SEEDS  # noqa: E402
from scripts.train_guardian_dsac_metasafe import (  # noqa: E402
    _run_isolated_evaluation,
    _sha256,
    _write_json,
)
from scripts.train_guardian_flow_dsac_paper import (  # noqa: E402
    _paper_evaluation_summary,
    _validate_paper_evaluation,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("learner", "teacher"), default="learner")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args(argv)
    if args.episodes != 1:
        parser.error("paper protocol uses exactly one episode per scene/run")
    return args


def _teacher_summary(evaluation):
    _validate_paper_evaluation(
        evaluation,
        required_modes=("hybrid",),
        require_value_network_metrics=False,
    )
    rows = {}
    returns = []
    costs = []
    successes = []
    safe_successes = []
    collisions = []
    margins = []
    safe_terminations = []
    inference_latency = []
    certification_latency = []
    certified_control_latency = []
    environment_steps = []
    for seed in PAPER_SCENE_SEEDS:
        hybrid = evaluation["per_seed"][str(seed)]["hybrid"]
        episodes = max(int(hybrid["episodes"]), 1)
        cost = float(hybrid["total_environment_cost"]) / episodes
        success = float(hybrid["success_rate"])
        collision = float(hybrid["collision_rate"])
        road = float(hybrid["road_departure_rate"])
        margin = float(hybrid["min_physical_body_sat_margin"])
        safe_termination = float(hybrid["safe_termination_rate"])
        safe_success = float(
            success == 1.0
            and collision == 0.0
            and road == 0.0
            and cost == 0.0
            and margin > 0.0
            and safe_termination == 0.0
            and hybrid.get("error") is None
        )
        row = {
            "hybrid_return": float(hybrid["mean_return"]),
            "hybrid_cost": cost,
            "hybrid_success": success,
            "hybrid_safe_success": safe_success,
            "hybrid_collision": collision,
            "hybrid_road_departure": road,
            "hybrid_min_physical_margin": margin,
            "hybrid_safe_termination": safe_termination,
            "policy_inference_latency_ms_mean": float(
                hybrid["inference_latency_ms_mean"]
            ),
            "guardian_certification_latency_ms_mean": float(
                hybrid["guardian_certification_latency_ms_mean"]
            ),
            "certified_control_latency_ms_mean": float(
                hybrid["certified_control_latency_ms_mean"]
            ),
            "environment_steps": int(hybrid["rows"][0]["steps"]),
        }
        rows[str(seed)] = row
        returns.append(row["hybrid_return"])
        costs.append(cost)
        successes.append(success)
        safe_successes.append(safe_success)
        collisions.append(collision)
        margins.append(margin)
        safe_terminations.append(safe_termination)
        inference_latency.append(row["policy_inference_latency_ms_mean"])
        certification_latency.append(row["guardian_certification_latency_ms_mean"])
        certified_control_latency.append(row["certified_control_latency_ms_mean"])
        environment_steps.append(row["environment_steps"])
    return {
        "scene_count": len(PAPER_SCENE_SEEDS),
        "teacher_mean_return": float(np.mean(returns)),
        "teacher_mean_cost": float(np.mean(costs)),
        "teacher_success_rate": float(np.mean(successes)),
        "teacher_safe_success_rate": float(np.mean(safe_successes)),
        "teacher_collision_rate": float(np.mean(collisions)),
        "teacher_safe_termination_rate": float(np.mean(safe_terminations)),
        "teacher_worst_physical_margin": float(np.min(margins)),
        "teacher_policy_inference_latency_ms_mean": float(np.mean(inference_latency)),
        "teacher_guardian_certification_latency_ms_mean": float(
            np.mean(certification_latency)
        ),
        "teacher_certified_control_latency_ms_mean": float(
            np.mean(certified_control_latency)
        ),
        "teacher_environment_steps_total": int(np.sum(environment_steps)),
        "ordinary_metrics": None,
        "hybrid_intervention_rate": None,
        "per_seed": rows,
        "all_scene_denominator": True,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("GUARDIAN_PAPER_EVAL_WORKERS", "4")
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    raw_output = output.with_name(output.stem + "_raw.json")
    evaluation = _run_isolated_evaluation(
        checkpoint=checkpoint,
        output=raw_output,
        seeds=PAPER_SCENE_SEEDS,
        modes=(("hybrid",) if args.policy == "teacher" else ("ordinary", "hybrid")),
        policy=args.policy,
    )
    summary = (
        _teacher_summary(evaluation)
        if args.policy == "teacher"
        else _paper_evaluation_summary(evaluation)
    )
    payload = {
        "format": "guardian_flow_dsac_paper_evaluation_v1",
        "policy": args.policy,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "scene_seeds": list(PAPER_SCENE_SEEDS),
        "episodes_per_scene": int(args.episodes),
        "raw_evaluation": str(raw_output),
        "raw_evaluation_sha256": _sha256(raw_output),
        "summary": summary,
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
