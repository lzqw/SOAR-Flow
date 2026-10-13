#!/usr/bin/env python3
"""Fork a complete Full Flow 300k atomic state for proxy annealing."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

PARENT_STEP = 300_000
TARGET_STEP = 1_000_000
FORMAT = "guardian_flow_proxy_annealing_1m_extension_v1"
CHILD_PROFILE = "proxy_annealing_1m"
SCHEDULES = (
    "p_hard_off_300k",
    "p_decay_300k_500k",
    "p_decay_300k_700k",
    "p_decay_floor_003",
)
SCENE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)
EVALUATION_STEPS = tuple(range(0, PARENT_STEP + 1, 50_000))
IMMUTABLE_BUDGETS = {
    "flow_matching_steps": 80_000,
    "same_latent_distillation_steps": 40_000,
    "offline_q_guided_actor_steps": 0,
    "online_steps": PARENT_STEP,
    "batch_size": 512,
    "online_buffer_size": 500_000,
    "gradient_steps": 1,
    "policy_delay": 2,
    "offline_replay_ratio": 0.2,
    "correction_fraction": 0.25,
    "evaluation_every": 50_000,
    "checkpoint_every": 50_000,
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_parent(parent: Path) -> dict[str, Any]:
    paths = {
        name: parent / name
        for name in (
            "config.json",
            "summary.json",
            "resume_index.json",
            "offline_summary.json",
        )
    }
    config = _read_json(paths["config.json"])
    summary = _read_json(paths["summary.json"])
    resume = _read_json(paths["resume_index.json"])
    offline = _read_json(paths["offline_summary.json"])
    seed = int(config.get("seed", -1))
    online = summary.get("online") or {}
    actual_steps = int(
        online.get("actual_online_steps", online.get("online_environment_steps", -1))
    )
    if config.get("method_slug") != "full_flow":
        raise ValueError("proxy annealing requires a Full Flow parent")
    if seed not in (0, 1) or int(summary.get("algorithm_seed", -1)) != seed:
        raise ValueError("parent algorithm Seed must consistently be 0 or 1")
    if config.get("budgets") != IMMUTABLE_BUDGETS:
        raise ValueError("parent does not have the exact formal 300k budget")
    if actual_steps != PARENT_STEP or summary.get("completed_full_budget") is not True:
        raise ValueError("parent summary is not complete at 300k")
    if tuple(config.get("scene_seeds", ())) != SCENE_SEEDS:
        raise ValueError("parent does not use the fixed ten paper scenes")
    if config.get("actor_objective_terms") != [
        "task_q",
        "guardian_p",
        "correction",
        "teacher_anchor",
    ]:
        raise ValueError("parent is not the complete Full Flow actor objective")
    flow = config.get("flow_config") or {}
    if not (
        float(flow.get("q_actor_weight", -1.0)) == 1.0
        and float(flow.get("proxy_actor_weight", -1.0)) == 0.15
        and float(flow.get("correction_weight", -1.0)) == 2.0
        and float(flow.get("anchor_weight", -1.0)) == 0.05
    ):
        raise ValueError("parent Full Flow actor weights are not 1/0.15/2/0.05")
    environment = config.get("environment") or {}
    if not (
        int(environment.get("max_episode_steps", -1)) == 1_200
        and float(environment.get("max_command_speed", -1.0)) == 10.75
        and float(environment.get("target_velocity", -1.0)) == 10.75
    ):
        raise ValueError("parent horizon or calibrated speed differs")
    for flag in (
        "moving_safe_proximal",
        "safe_trajectory_memory",
        "seed_specific_logic",
        "performance_rollback",
        "conditional_launch",
        "evaluation_changes_training",
        "component_smoke",
        "composite_or_ensemble",
    ):
        if config.get(flag) is not False:
            raise ValueError(f"forbidden parent feature is enabled: {flag}")

    trainer_name = str(resume.get("trainer_state", ""))
    trainer = parent / trainer_name
    if (
        resume.get("format")
        not in (
            "guardian_flow_paper_resume_v2_atomic",
            "guardian_flow_paper_resume_v3_atomic",
        )
        or resume.get("replay_states_embedded_atomically") is not True
        or int(resume.get("step", -1)) != PARENT_STEP
        or Path(trainer_name).name != trainer_name
        or not trainer.is_file()
        or _sha256(trainer) != resume.get("trainer_state_sha256")
    ):
        raise ValueError("parent trainer state is not a valid 300k atomic generation")
    with trainer.open("rb") as stream:
        control = pickle.load(stream)
    evaluation_steps = tuple(
        int(row.get("step", -1)) for row in control.get("evaluations", ())
    )
    scheduler = control.get("scheduler") or {}
    scheduler_steps = {
        int(key): int(value) for key, value in (scheduler.get("steps") or {}).items()
    }
    online_replay = control.get("online_replay_state") or {}
    correction_replay = control.get("correction_replay_state") or {}
    if not (
        int(control.get("step", -1)) == PARENT_STEP
        and int(control.get("counters", {}).get("online_environment_steps", -1))
        == PARENT_STEP
        and evaluation_steps == EVALUATION_STEPS
        and set(scheduler_steps) == set(SCENE_SEEDS)
        and sum(scheduler_steps.values()) == PARENT_STEP
        and online_replay.get("format") == "seed_indexed_replay_v1"
        and correction_replay.get("format") == "seed_indexed_replay_v1"
        and int(online_replay.get("capacity", -1)) == 500_000
        and int(correction_replay.get("capacity", -1)) == 500_000
        and control.get("jax_key") is not None
        and control.get("numpy_rng_state") is not None
        and control.get("latent_counts") is not None
        and int(control.get("discard_incomplete_episode_rows_on_resume", -1)) == 0
    ):
        raise ValueError("parent replay/RNG/scheduler/latent state is incomplete")
    checkpoint_name = str(resume.get("algorithm_checkpoint", ""))
    backup_name = str(resume.get("algorithm_checkpoint_backup", ""))
    checkpoint = parent / checkpoint_name
    backup = parent / backup_name
    checkpoint_sha = str(resume.get("algorithm_checkpoint_sha256", ""))
    if not (
        checkpoint_name == f"checkpoint_{PARENT_STEP:08d}.pkl"
        and checkpoint.is_file()
        and backup.is_file()
        and _sha256(checkpoint) == checkpoint_sha
        and _sha256(backup) == checkpoint_sha
        and resume.get("algorithm_checkpoint_backup_sha256") == checkpoint_sha
    ):
        raise ValueError("parent algorithm checkpoint binding is invalid")
    if any(
        control.get(field) != resume.get(field)
        for field in (
            "algorithm_checkpoint",
            "algorithm_checkpoint_sha256",
            "algorithm_checkpoint_backup",
            "algorithm_checkpoint_backup_sha256",
        )
    ):
        raise ValueError("trainer state and resume index disagree")
    offline_checkpoint = parent / "offline_flow_bc.pkl"
    if not offline_checkpoint.is_file() or _sha256(offline_checkpoint) != str(
        offline.get("checkpoint_sha256", "")
    ):
        raise ValueError("parent frozen Flow-BC checkpoint binding is invalid")
    source = config.get("source_code") or {}
    return {
        "seed": seed,
        "parent_profile": config.get("experiment_profile"),
        "source_revision": source.get("git_revision"),
        "config_sha256": _sha256(paths["config.json"]),
        "summary_sha256": _sha256(paths["summary.json"]),
        "resume_index_sha256": _sha256(paths["resume_index.json"]),
        "trainer_state": trainer_name,
        "trainer_state_sha256": _sha256(trainer),
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha,
        "offline_checkpoint_sha256": _sha256(offline_checkpoint),
        "flow_bc_source_sha256": offline.get("source_sha256"),
        "shared_qp_sha256": offline.get("shared_qp_checkpoint_sha256"),
    }


def _assert_parent_unchanged(parent: Path, evidence: dict[str, Any]) -> None:
    expected = {
        "config.json": evidence["config_sha256"],
        "summary.json": evidence["summary_sha256"],
        "resume_index.json": evidence["resume_index_sha256"],
        evidence["trainer_state"]: evidence["trainer_state_sha256"],
        evidence["checkpoint"]: evidence["checkpoint_sha256"],
        "offline_flow_bc.pkl": evidence["offline_checkpoint_sha256"],
    }
    changed = [
        name
        for name, expected_sha in expected.items()
        if not (parent / name).is_file() or _sha256(parent / name) != expected_sha
    ]
    if changed:
        raise RuntimeError("300k parent changed during copy: " + ", ".join(changed))


def prepare(
    parent: Path,
    destination: Path,
    *,
    schedule: str,
    correction_weight: float = 2.0,
) -> dict[str, Any]:
    parent = parent.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if schedule not in SCHEDULES:
        raise ValueError(f"unsupported proxy schedule: {schedule}")
    if float(correction_weight) not in (1.0, 2.0, 3.0):
        raise ValueError("correction weight must be 1.0, 2.0, or 3.0")
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    if (
        parent == destination
        or parent in destination.parents
        or destination in parent.parents
    ):
        raise ValueError("parent and destination must be separate directory trees")
    evidence = _validate_parent(parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.prepare.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite prepared run: {destination}")
        staging: Path | None = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging.", dir=destination.parent
            )
        )
        try:
            shutil.copytree(
                parent, staging, dirs_exist_ok=True, copy_function=shutil.copy2
            )
            if _sha256(staging / "config.json") != evidence["config_sha256"]:
                raise RuntimeError("copied parent config failed SHA256 verification")
            if (
                _sha256(staging / evidence["trainer_state"])
                != evidence["trainer_state_sha256"]
            ):
                raise RuntimeError("copied trainer state failed SHA256 verification")
            shutil.copy2(staging / "config.json", staging / "parent_config_300k.json")
            manifest = {
                "format": FORMAT,
                "method_slug": "full_flow",
                "algorithm_seed": evidence["seed"],
                "proxy_weight_schedule": schedule,
                "effective_actor_correction_weight": float(correction_weight),
                "extended_from_step": PARENT_STEP,
                "target_online_steps": TARGET_STEP,
                "parent_run": str(parent),
                "destination_run": str(destination),
                "parent_profile_observed": evidence["parent_profile"],
                "child_profile": CHILD_PROFILE,
                "parent_source_revision": evidence["source_revision"],
                "parent_config_sha256": evidence["config_sha256"],
                "parent_summary_sha256": evidence["summary_sha256"],
                "parent_resume_index_sha256": evidence["resume_index_sha256"],
                "parent_trainer_state": str(parent / evidence["trainer_state"]),
                "parent_trainer_state_sha256": evidence["trainer_state_sha256"],
                "parent_checkpoint": str(parent / evidence["checkpoint"]),
                "parent_checkpoint_sha256": evidence["checkpoint_sha256"],
                "parent_offline_checkpoint_sha256": evidence[
                    "offline_checkpoint_sha256"
                ],
                "flow_bc_initialization_sha256": evidence["flow_bc_source_sha256"],
                "shared_qp_checkpoint_sha256": evidence["shared_qp_sha256"],
                "copied_atomic_resume_state": True,
                "replay_optimizer_rng_scheduler_and_latent_state_preserved": True,
            }
            _assert_parent_unchanged(parent, evidence)
            _atomic_json(staging / "extension_preparation.json", manifest)
            os.replace(staging, destination)
            staging = None
            return manifest
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--proxy-weight-schedule", choices=SCHEDULES, required=True)
    parser.add_argument("--correction-weight", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare(
        args.parent_run,
        args.destination,
        schedule=args.proxy_weight_schedule,
        correction_weight=args.correction_weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
