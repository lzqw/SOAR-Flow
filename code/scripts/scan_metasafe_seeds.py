#!/usr/bin/env python3
"""Predeclare online and held-out MetaSafe seeds without training feedback.

The scan evaluates only the initial Champion and the fixed [1, 0] Guardian
reference.  Selection is deterministic (ascending seed order), so no online
candidate or post-training metric can influence the scene split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.utils.action_semantics import SPEED_COUPLED_ACTION_SEMANTICS


SELECTION_RULE_VERSION = "metasafe_seed_selection_v2_2_predeclared_v1"


def _sanitize_nonfinite_json(value: Any, *, path: str = "$") -> tuple[Any, list[str]]:
    """Replace non-finite diagnostics with JSON null and return their paths."""

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        paths: list[str] = []
        for key, item in value.items():
            clean, item_paths = _sanitize_nonfinite_json(item, path=f"{path}.{key}")
            sanitized[str(key)] = clean
            paths.extend(item_paths)
        return sanitized, paths
    if isinstance(value, (list, tuple)):
        sanitized_items = []
        paths: list[str] = []
        for index, item in enumerate(value):
            clean, item_paths = _sanitize_nonfinite_json(item, path=f"{path}[{index}]")
            sanitized_items.append(clean)
            paths.extend(item_paths)
        return sanitized_items, paths
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None, [path]
    return value, []


def _finite_selection_metrics(row: Mapping[str, Any]) -> bool:
    required = (
        "initial_champion_hybrid_success",
        "initial_champion_ordinary_success",
        "initial_champion_hybrid_intervention",
        "initial_champion_hybrid_return",
        "initial_champion_ordinary_return",
        "guardian_success",
        "collision",
        "road_departure",
        "environment_cost",
        "solver_finite_rate",
        "physical_margin",
    )
    try:
        return all(np.isfinite(float(row[key])) for key in required)
    except (KeyError, TypeError, ValueError):
        return False


def _parse_seed_spec(value: str) -> list[int]:
    seeds: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending seed range: {part}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("candidate seed list is empty")
    return sorted(set(seeds))


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline-scene-seed", type=int, default=0)
    parser.add_argument("--candidate-seeds", default="1-20")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of isolated scene scans to run concurrently",
    )
    parser.add_argument("--action-semantics", default=SPEED_COUPLED_ACTION_SEMANTICS)
    parser.add_argument("--max-command-speed", type=float, default=4.8)
    parser.add_argument(
        "--input-scan",
        type=Path,
        help="select deterministically from existing scan rows instead of running MetaDrive",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not np.isfinite(args.max_command_speed) or args.max_command_speed <= 0.0:
        parser.error("--max-command-speed must be positive and finite")
    if args.action_semantics != SPEED_COUPLED_ACTION_SEMANTICS:
        parser.error("V2.2 seed scan requires target_point_speed_coupled_v1")
    if args.input_scan is None and args.checkpoint is None:
        parser.error("--checkpoint is required unless --input-scan is supplied")
    if args.worker and (args.worker_seed is None or args.worker_output is None):
        parser.error("worker mode requires seed and output")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _GuardianPolicy:
    def get_deterministic_action(self, observation):
        del observation
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def get_deterministic_champion_action(self, observation):
        return self.get_deterministic_action(observation)


def _safe_suite(suite: Mapping[str, Any], *, require_ordinary: bool) -> bool:
    modes = ("hybrid", "ordinary") if require_ordinary else ("hybrid",)
    for mode_name in modes:
        mode = suite[mode_name]
        if not (
            float(mode["success_rate"]) == 1.0
            and float(mode["collision_rate"]) == 0.0
            and float(mode["road_departure_rate"]) == 0.0
            and float(mode["total_environment_cost"]) == 0.0
            and float(mode["solver_finite_rate"]) == 1.0
            and float(mode["min_physical_body_sat_margin"]) > 0.0
        ):
            return False
    if require_ordinary:
        ordinary = suite["ordinary"]
        if (
            float(
                ordinary.get("actual_intervention_rate", ordinary["intervention_rate"])
            )
            > 0.0
        ):
            return False
    return True


def _worker(args) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from scripts.record_guardian_calql_metasafe_video import (
        _load_algorithm,
        _run_dir,
    )
    from scripts.train_guardian_calql_metasafe import (
        _evaluate_policy,
        _make_environment,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    config = json.loads(
        (_run_dir(checkpoint) / "config.json").read_text(encoding="utf-8")
    )
    algorithm, env_args, _obs_dim = _load_algorithm(checkpoint, config)
    env_args.fixed_scene_seed = int(args.worker_seed)
    env_args.action_semantics = args.action_semantics
    env_args.max_command_speed = float(args.max_command_speed)
    env_args.recreate_environment_each_episode = True
    env_args.include_step_trace = False

    metadata_env = _make_environment(env_args, "hybrid")
    try:
        _observation, reset_info = metadata_env.reset(seed=args.worker_seed)
    finally:
        metadata_env.close()
    scene = dict(reset_info.get("metasafe_scene", {}))
    objects = list(scene.get("objects", []))
    signature = str(reset_info.get("metasafe_scene_signature", ""))

    champion = {
        mode: _evaluate_policy(
            algorithm,
            env_args,
            mode=mode,
            use_champion=True,
            episodes=args.episodes,
        )
        for mode in ("hybrid", "ordinary")
    }
    guardian = {
        "hybrid": _evaluate_policy(
            _GuardianPolicy(),
            env_args,
            mode="hybrid",
            use_champion=True,
            episodes=args.episodes,
        )
    }
    row = {
        "scene_seed": int(args.worker_seed),
        "scene_signature": signature,
        "map_blocks": list(scene.get("map_blocks", [])),
        "obstacle_count": len(objects),
        "obstacle_types": sorted(
            {str(item.get("type", "unknown")) for item in objects}
        ),
        "initial_champion_hybrid_success": float(champion["hybrid"]["success_rate"]),
        "initial_champion_ordinary_success": float(
            champion["ordinary"]["success_rate"]
        ),
        "initial_champion_hybrid_intervention": float(
            champion["hybrid"].get(
                "actual_intervention_rate", champion["hybrid"]["intervention_rate"]
            )
        ),
        "initial_champion_hybrid_return": float(champion["hybrid"]["mean_return"]),
        "initial_champion_ordinary_return": float(champion["ordinary"]["mean_return"]),
        "guardian_success": float(guardian["hybrid"]["success_rate"]),
        "collision": float(guardian["hybrid"]["collision_rate"]),
        "road_departure": float(guardian["hybrid"]["road_departure_rate"]),
        "environment_cost": float(guardian["hybrid"]["total_environment_cost"]),
        "solver_finite_rate": float(guardian["hybrid"]["solver_finite_rate"]),
        "physical_margin": float(guardian["hybrid"]["min_physical_body_sat_margin"]),
        # The predeclared rule requires protected (Hybrid) safety.  Ordinary
        # is deliberately retained as an adaptation signal and may be weak.
        "initial_champion_safe": _safe_suite(champion, require_ordinary=False),
        "guardian_safe": _safe_suite(guardian, require_ordinary=False),
        "action_semantics_version": args.action_semantics,
        "max_command_speed": float(args.max_command_speed),
        "champion_suite": champion,
        "guardian_suite": guardian,
    }
    row["selection_metrics_finite"] = _finite_selection_metrics(row)
    serializable_row, nonfinite_paths = _sanitize_nonfinite_json(row)
    serializable_row["nonfinite_diagnostic_count"] = len(nonfinite_paths)
    serializable_row["nonfinite_diagnostic_paths"] = nonfinite_paths
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps(serializable_row, indent=2, allow_nan=False), encoding="utf-8"
    )
    return 0


def select_scene_split(
    rows: Iterable[Mapping[str, Any]], *, offline_scene_seed: int = 0
) -> dict[str, Any]:
    indexed = {int(row["scene_seed"]): dict(row) for row in rows}
    if int(offline_scene_seed) not in indexed:
        raise ValueError("scan rows do not contain the offline seed")
    offline = indexed[int(offline_scene_seed)]
    if not bool(offline.get("selection_metrics_finite", True)):
        raise ValueError("offline scan row contains non-finite selection metrics")
    offline_signature = str(offline["scene_signature"])
    baseline_i = float(offline["initial_champion_hybrid_intervention"])
    baseline_return = float(offline["initial_champion_ordinary_return"])

    selected_online: int | None = None
    online_reasons: dict[str, list[str]] = {}
    for seed in sorted(indexed):
        if seed == int(offline_scene_seed):
            continue
        row = indexed[seed]
        reasons: list[str] = []
        selection_metrics_finite = bool(row.get("selection_metrics_finite", True))
        if not selection_metrics_finite:
            reasons.append("nonfinite_selection_metrics")
        signature = str(row.get("scene_signature", ""))
        if not signature:
            reasons.append("scene_signature_missing")
        elif signature == offline_signature:
            reasons.append("same_signature_as_offline")
        if not bool(row.get("guardian_safe", False)):
            reasons.append("guardian_not_safe")
        if not bool(row.get("initial_champion_safe", False)):
            reasons.append("initial_champion_not_safe")
        if selection_metrics_finite:
            if float(row["initial_champion_hybrid_intervention"]) < 0.01:
                reasons.append("hybrid_intervention_below_1pct")
            harder = bool(
                float(row["initial_champion_hybrid_intervention"]) - baseline_i >= 0.005
                or baseline_return - float(row["initial_champion_ordinary_return"])
                >= 0.5
            )
            if not harder:
                reasons.append("no_predeclared_adaptation_space")
        online_reasons[str(seed)] = reasons
        if selected_online is None and not reasons:
            selected_online = seed
    if selected_online is None:
        raise RuntimeError("no candidate seed satisfies the predeclared online rule")

    selected_signature = str(indexed[selected_online]["scene_signature"])
    selected_heldout: int | None = None
    heldout_reasons: dict[str, list[str]] = {}
    for seed in sorted(indexed):
        if seed in (int(offline_scene_seed), selected_online):
            continue
        row = indexed[seed]
        reasons: list[str] = []
        if not bool(row.get("selection_metrics_finite", True)):
            reasons.append("nonfinite_selection_metrics")
        signature = str(row.get("scene_signature", ""))
        if not signature:
            reasons.append("scene_signature_missing")
        elif signature in {offline_signature, selected_signature}:
            reasons.append("signature_not_distinct")
        if not bool(row.get("guardian_safe", False)):
            reasons.append("guardian_not_safe")
        if not bool(row.get("initial_champion_safe", False)):
            reasons.append("initial_champion_not_safe")
        heldout_reasons[str(seed)] = reasons
        if selected_heldout is None and not reasons:
            selected_heldout = seed
    if selected_heldout is None:
        raise RuntimeError("no remaining safe, signature-distinct held-out seed")

    return {
        "offline_scene_seed": int(offline_scene_seed),
        "online_scene_seeds": [int(selected_online)],
        "retention_scene_seeds": [int(offline_scene_seed)],
        "heldout_scene_seeds": [int(selected_heldout)],
        "selection_diagnostics": {
            "online_rejection_reasons": online_reasons,
            "heldout_rejection_reasons": heldout_reasons,
        },
    }


def _orchestrate(args) -> int:
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.input_scan is not None:
        source = json.loads(
            args.input_scan.expanduser().resolve().read_text(encoding="utf-8")
        )
        rows = list(source.get("scan_rows", source.get("rows", [])))
        checkpoint_sha256 = source.get("initial_champion_checkpoint_sha256")
        candidate_seeds = sorted(int(row["scene_seed"]) for row in rows)
    else:
        checkpoint = args.checkpoint.expanduser().resolve()
        candidate_seeds = sorted(
            set(_parse_seed_spec(args.candidate_seeds)) | {int(args.offline_scene_seed)}
        )
        worker_dir = output.parent / f"{output.stem}_workers"
        worker_dir.mkdir(parents=True, exist_ok=True)

        def run_seed(seed: int) -> dict[str, Any]:
            worker_output = worker_dir / f"seed_{seed:03d}.json"
            worker_log = worker_dir / f"seed_{seed:03d}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--offline-scene-seed",
                str(args.offline_scene_seed),
                "--episodes",
                str(args.episodes),
                "--action-semantics",
                args.action_semantics,
                "--max-command-speed",
                str(args.max_command_speed),
                "--worker",
                "--worker-seed",
                str(seed),
                "--worker-output",
                str(worker_output),
            ]
            completed = subprocess.run(
                command,
                cwd=str(REPOSITORY_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            worker_log.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0 or not worker_output.is_file():
                raise RuntimeError(f"seed {seed} scan failed; see {worker_log}")
            return json.loads(worker_output.read_text(encoding="utf-8"))

        rows = []
        with ThreadPoolExecutor(
            max_workers=min(args.workers, len(candidate_seeds))
        ) as pool:
            futures = {pool.submit(run_seed, seed): seed for seed in candidate_seeds}
            for future in as_completed(futures):
                seed = futures[future]
                rows.append(future.result())
                print(
                    f"[seed scan] completed seed={seed} "
                    f"({len(rows)}/{len(candidate_seeds)})",
                    flush=True,
                )
        checkpoint_sha256 = _sha256(checkpoint)

    split = select_scene_split(rows, offline_scene_seed=args.offline_scene_seed)
    signatures = {
        str(int(row["scene_seed"])): str(row["scene_signature"]) for row in rows
    }
    manifest = {
        "format": SELECTION_RULE_VERSION,
        "created_before_online_training": True,
        "selection_uses_online_candidate_results": False,
        "candidate_seeds": candidate_seeds,
        "episodes_per_policy_mode": int(args.episodes),
        "action_semantics_version": args.action_semantics,
        "max_command_speed": float(args.max_command_speed),
        "initial_champion_checkpoint_sha256": checkpoint_sha256,
        "scene_signatures": signatures,
        **split,
        "predeclared_online_rule": {
            "ascending_first_match": True,
            "signature_differs_from_seed0": True,
            "guardian_success_episodes": int(args.episodes),
            "initial_champion_hybrid_success_episodes": int(args.episodes),
            "minimum_hybrid_intervention_rate": 0.01,
            "minimum_intervention_increase_rate": 0.005,
            "ordinary_return_drop_alternative": 0.5,
        },
        "scan_rows": sorted(rows, key=lambda row: int(row["scene_seed"])),
    }
    manifest["scene_selection_manifest_hash"] = _canonical_hash(manifest)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "offline": manifest["offline_scene_seed"],
                "online": manifest["online_scene_seeds"],
                "heldout": manifest["heldout_scene_seeds"],
                "manifest_hash": manifest["scene_selection_manifest_hash"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return _worker(args) if args.worker else _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
