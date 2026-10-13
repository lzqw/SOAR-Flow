#!/usr/bin/env python3
"""Collect and validate the five-Seed Guardian-DSAC main dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.utils.action_semantics import (  # noqa: E402
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)


OFFLINE_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_MAX_COMMAND_SPEED = 10.75


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/metasafe_cbf/guardian_dsac_multiseed_v1"),
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes-per-seed", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--collector-seed", type=int, default=20260805)
    parser.add_argument("--max-attempt-multiplier", type=float, default=5.0)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument(
        "--max-command-speed", type=float, default=DEFAULT_MAX_COMMAND_SPEED
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.seeds = _parse_seeds(args.seeds)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.episodes_per_seed < 1 or args.num_envs < 1:
        parser.error("episode and environment counts must be positive")
    if args.max_attempt_multiplier < 1.0:
        parser.error("--max-attempt-multiplier must be at least one")
    if args.max_episode_steps != 1200:
        parser.error("the multi-Seed main experiment keeps a 1200-step horizon")
    if not np.isfinite(args.max_command_speed) or args.max_command_speed <= 1.0:
        parser.error("--max-command-speed must be finite and above 1 m/s")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _collector_command(args, *, seed: int, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/collect_metasafe_cbf_dataset.py"),
        "--output-dir",
        str(output_dir),
        "--episodes",
        str(args.episodes_per_seed),
        "--num-envs",
        str(args.num_envs),
        "--scene-seed",
        str(seed),
        "--collector-seed",
        str(args.collector_seed + 100_003 * seed),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--minimum-attempt-success-rate",
        "0.000001",
        "--max-attempt-multiplier",
        str(args.max_attempt_multiplier),
        "--speed-sampling-profile",
        "fast_v231",
        "--action-semantics",
        TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        "--max-command-speed",
        str(args.max_command_speed),
        "--target-velocity",
        str(args.max_command_speed),
        # Keep only aggregate non-empty-bin coverage from the base collector.
        "--minimum-high-speed-bin-fraction",
        "0",
        "--maximum-high-speed-bin-fraction",
        "1",
        "--minimum-mid-high-bin-fraction",
        "0",
        "--minimum-low-speed-bins-fraction",
        "0",
    ]
    if not np.isclose(args.max_command_speed, DEFAULT_MAX_COMMAND_SPEED):
        command.append("--allow-recalibrated-fast-v231-speed")
    if args.overwrite_existing:
        command.append("--overwrite")
    return command


def _validate_seed_dataset(
    output_dir: Path,
    *,
    seed: int,
    episodes: int,
    max_episode_steps: int,
    max_command_speed: float,
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    dataset_path = output_dir / "metasafe_cbf_dataset.npz"
    if not summary_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(f"Seed{seed} dataset or summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures = []
    if int(summary.get("accepted_episodes", -1)) != episodes:
        failures.append("accepted_episode_count")
    if float(summary.get("accepted_collision_rate", np.inf)) != 0.0:
        failures.append("collision")
    if float(summary.get("accepted_road_departure_rate", np.inf)) != 0.0:
        failures.append("road_departure")
    if float(summary.get("accepted_total_environment_cost", np.inf)) != 0.0:
        failures.append("environment_cost")
    if int(summary.get("episode_steps_max", max_episode_steps + 1)) > max_episode_steps:
        failures.append("episode_step_limit")
    if int(summary.get("fixed_scene_seed", -1)) != seed or not bool(
        summary.get("fixed_scene_verified", False)
    ):
        failures.append("scene_binding")
    if summary.get("action_semantics_version") != TARGET_SPEED_LATERAL_ACTION_SEMANTICS:
        failures.append("action_semantics")
    counts = np.asarray(summary.get("speed_bin_counts", []), dtype=np.int64)
    if counts.shape != (4,) or np.any(counts <= 0):
        failures.append("speed_coverage")
    if not np.isclose(
        float(summary.get("max_command_speed", np.nan)), max_command_speed
    ):
        failures.append("max_command_speed")

    with np.load(dataset_path, allow_pickle=False) as archive:
        archive_seeds = np.unique(np.asarray(archive["scene_seed"], dtype=np.int64))
        signatures = np.unique(np.asarray(archive["scene_signature"]).astype(str))
        semantics = np.unique(
            np.asarray(archive["action_semantics_version"]).astype(str)
        )
        episode_ids = np.unique(np.asarray(archive["episode_id"], dtype=np.int64))
        step_limits = np.unique(
            np.asarray(archive["episode_step_limit"], dtype=np.int64)
        )
    if archive_seeds.tolist() != [seed]:
        failures.append("archive_scene_seed")
    if len(signatures) != 1 or not signatures[0]:
        failures.append("archive_scene_signature")
    if semantics.tolist() != [TARGET_SPEED_LATERAL_ACTION_SEMANTICS]:
        failures.append("archive_action_semantics")
    if episode_ids.size != episodes:
        failures.append("archive_episode_count")
    if step_limits.tolist() != [max_episode_steps]:
        failures.append("archive_episode_step_limit")
    observed_sha = _sha256(dataset_path)
    if summary.get("dataset_sha256") != observed_sha:
        failures.append("dataset_sha256")
    if failures:
        raise ValueError(
            f"Seed{seed} validation failed: {', '.join(sorted(set(failures)))}"
        )
    return {
        "seed": seed,
        "directory": str(output_dir.resolve()),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": observed_sha,
        "scene_signature": str(signatures[0]),
        "accepted_episodes": episodes,
        "attempts": int(summary["attempts"]),
        "attempt_success_rate": float(summary["attempt_success_rate"]),
        "transitions": int(summary["transitions"]),
        "episode_steps_max": int(summary["episode_steps_max"]),
        "speed_bin_counts": counts.astype(int).tolist(),
        "speed_bin_fractions": [
            float(value) for value in summary["speed_bin_fractions"]
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    for seed in args.seeds:
        output_dir = output_root / f"seed{seed}"
        try:
            row = _validate_seed_dataset(
                output_dir,
                seed=seed,
                episodes=args.episodes_per_seed,
                max_episode_steps=args.max_episode_steps,
                max_command_speed=args.max_command_speed,
            )
            row["reused"] = True
            print(f"[multiseed] Seed{seed}: reusing validated dataset", flush=True)
        except (FileNotFoundError, ValueError) as validation_error:
            command = _collector_command(args, seed=seed, output_dir=output_dir)
            if args.dry_run:
                print(" ".join(command), flush=True)
                continue
            if (
                output_dir.exists()
                and any(output_dir.iterdir())
                and not args.overwrite_existing
            ):
                raise RuntimeError(
                    f"Seed{seed} has an invalid partial dataset ({validation_error}); "
                    "rerun with --overwrite-existing"
                ) from validation_error
            log_path = logs_dir / f"seed{seed}.log"
            print(
                f"[multiseed] Seed{seed}: collecting {args.episodes_per_seed} episodes",
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=str(REPOSITORY_ROOT),
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise RuntimeError(
                    f"Seed{seed} collection failed:\n{tail}"
                ) from validation_error
            row = _validate_seed_dataset(
                output_dir,
                seed=seed,
                episodes=args.episodes_per_seed,
                max_episode_steps=args.max_episode_steps,
                max_command_speed=args.max_command_speed,
            )
            row["reused"] = False
        rows.append(row)
        _write_json(
            output_root / "manifest.partial.json",
            {"completed_seeds": rows},
        )

    if args.dry_run:
        return 0
    counts = np.sum(
        [np.asarray(row["speed_bin_counts"], dtype=np.int64) for row in rows], axis=0
    )
    total = max(int(np.sum(counts)), 1)
    manifest = {
        "format": "guardian_dsac_multiseed_collection_v1",
        "offline_seeds": list(args.seeds),
        "episodes_per_seed": args.episodes_per_seed,
        "total_episodes": sum(int(row["accepted_episodes"]) for row in rows),
        "num_envs_per_seed": args.num_envs,
        "action_semantics_version": TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        "speed_sampling_profile": "fast_v231",
        "max_episode_steps": args.max_episode_steps,
        "max_command_speed": args.max_command_speed,
        "speed_recalibrated": not np.isclose(
            args.max_command_speed, DEFAULT_MAX_COMMAND_SPEED
        ),
        "datasets": rows,
        "combined_speed_bin_counts": counts.astype(int).tolist(),
        "combined_speed_bin_fractions": (counts / total).astype(float).tolist(),
        "all_speed_bins_visited": bool(np.all(counts > 0)),
        "wall_time_seconds": time.time() - started,
    }
    if tuple(args.seeds) == OFFLINE_SEEDS and manifest["total_episodes"] != 1500:
        raise RuntimeError("formal Seed0--4 collection did not produce 1500 episodes")
    _write_json(output_root / "manifest.json", manifest)
    partial = output_root / "manifest.partial.json"
    if partial.exists():
        partial.unlink()
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
