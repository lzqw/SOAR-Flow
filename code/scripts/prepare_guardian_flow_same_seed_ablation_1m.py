#!/usr/bin/env python3
"""Copy one complete Seed1 300k atomic run into a separate 1M extension run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any


METHODS = ("full_flow", "flow_no_p", "flow_no_correction")
PARENT_STEP = 300_000
TARGET_STEP = 1_000_000
PARENT_PROFILE = "same_seed_ablation"
CHILD_PROFILE = "same_seed_ablation_1m"
FORMAT = "guardian_flow_same_seed_ablation_1m_extension_v1"
FLOW_BC_SEED1_SHA256 = (
    "48da3e012811c9dae55a67a976db2a9681ea953a96115454250d9440a537595e"
)
SHARED_QP_SHA256 = "873c82c30599134dd7e99f320a6b1f521625cd34b2f2ed8bb8892ac812fb9067"
DATASET_SHA256 = "b79ed6b6921c05325b7ec159ece9df8631b3f8849ea55abb555eb0d36230d95b"
MANIFEST_SHA256 = "0873699b4a83483de312a7c3689c6e12cf96ccc3988a019c4a6285b469590e0e"
SOURCE_REVISION_BY_METHOD = {
    "full_flow": "7734b72f7d2035e8853adfd23b2009b9e3bff380",
    "flow_no_p": "389f26e78814854bc475497131d7fc9e3a5ab86d",
    "flow_no_correction": "389f26e78814854bc475497131d7fc9e3a5ab86d",
}
SOURCE_TRAINER_SHA256_BY_METHOD = {
    "full_flow": ("16e608763ad6a4c76af36d8aa5f0913193048f1083926b80ec74fdd1eebd618d"),
    "flow_no_p": ("faf726830a698b79e047b318499c3cb05cf7337de53eaa370c9a2155d8091f6b"),
    "flow_no_correction": (
        "faf726830a698b79e047b318499c3cb05cf7337de53eaa370c9a2155d8091f6b"
    ),
}
SCENE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)
EVALUATION_STEPS = tuple(range(0, PARENT_STEP + 1, 50_000))
OBJECTIVES = {
    "full_flow": ["task_q", "guardian_p", "correction", "teacher_anchor"],
    "flow_no_p": ["task_q", "correction", "teacher_anchor"],
    "flow_no_correction": ["task_q", "guardian_p", "teacher_anchor"],
}
EXPECTED_BUDGETS = {
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing required parent artifact: {path}") from error
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


def _completed_steps(summary: dict[str, Any]) -> int:
    online = summary.get("online") or {}
    return int(
        online.get("actual_online_steps", online.get("online_environment_steps", -1))
    )


def _validate_parent(parent: Path) -> dict[str, Any]:
    config_path = parent / "config.json"
    summary_path = parent / "summary.json"
    resume_path = parent / "resume_index.json"
    trainer_path = parent / "trainer_state_latest.pkl"
    offline_summary_path = parent / "offline_summary.json"
    config = _read_json(config_path)
    summary = _read_json(summary_path)
    resume = _read_json(resume_path)
    offline = _read_json(offline_summary_path)
    method = str(config.get("method_slug", ""))
    if method not in METHODS:
        raise ValueError(f"unsupported same-seed ablation method: {method!r}")
    if int(config.get("seed", -1)) != 1 or int(summary.get("algorithm_seed", -1)) != 1:
        raise ValueError("the parent run must use algorithm Seed1")
    if config.get("budgets") != EXPECTED_BUDGETS:
        raise ValueError("the parent config does not have the exact 300k budgets")
    if _completed_steps(summary) != PARENT_STEP:
        raise ValueError("the parent summary is not complete at 300k")
    if summary.get("completed_full_budget") is not True:
        raise ValueError("the parent did not mark its 300k budget complete")
    if config.get("actor_objective_terms") != OBJECTIVES[method]:
        raise ValueError("the parent actor objective is not the declared ablation")
    if tuple(config.get("scene_seeds", ())) != SCENE_SEEDS:
        raise ValueError("the parent scene set is not the formal ten-Seed set")
    if (
        tuple(config.get("offline_seeds", ())) != SCENE_SEEDS[:5]
        or tuple(config.get("online_novel_seeds", ())) != SCENE_SEEDS[5:]
    ):
        raise ValueError("the parent offline/novel Seed split is invalid")
    if config.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("the parent dataset SHA256 is invalid")
    if config.get("manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("the parent manifest SHA256 is invalid")
    source = config.get("source_code") or {}
    source_hashes = source.get("source_sha256") or {}
    if source.get("git_revision") != SOURCE_REVISION_BY_METHOD[method]:
        raise ValueError("the parent source revision is not the recorded formal run")
    if (
        source_hashes.get("scripts/train_guardian_flow_dsac_paper.py")
        != SOURCE_TRAINER_SHA256_BY_METHOD[method]
    ):
        raise ValueError("the parent trainer source SHA256 is invalid")
    environment = config.get("environment") or {}
    if (
        int(environment.get("max_episode_steps", -1)) != 1_200
        or float(environment.get("max_command_speed", -1.0)) != 10.75
        or float(environment.get("target_velocity", -1.0)) != 10.75
    ):
        raise ValueError("the parent horizon or calibrated speed is invalid")
    flow_config = config.get("flow_config") or {}
    expected_weights = {
        "full_flow": (0.15, 2.0),
        "flow_no_p": (0.0, 2.0),
        "flow_no_correction": (0.15, 0.0),
    }[method]
    if (
        float(flow_config.get("q_actor_weight", -1.0)) != 1.0
        or float(flow_config.get("proxy_actor_weight", -1.0)) != expected_weights[0]
        or float(flow_config.get("correction_weight", -1.0)) != expected_weights[1]
        or float(flow_config.get("anchor_weight", -1.0)) != 0.05
        or int(flow_config.get("actor_learning_rate_decay_steps", -1)) != 500_000
        or int(flow_config.get("critic_learning_rate_decay_steps", -1)) != 1_000_000
        or int(flow_config.get("proxy_learning_rate_decay_steps", -1)) != 1_000_000
    ):
        raise ValueError("the parent actor objective weights or schedules are invalid")
    observed_profile = config.get("experiment_profile")
    if method == "full_flow" and observed_profile != PARENT_PROFILE:
        raise ValueError("Full Flow parent must explicitly use same_seed_ablation")
    if method in ("flow_no_p", "flow_no_correction") and observed_profile not in (
        None,
        PARENT_PROFILE,
    ):
        raise ValueError("legacy ablation parent has an unexpected profile")
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
            raise ValueError(f"forbidden parent feature is enabled or missing: {flag}")
    if offline.get("source_sha256") != FLOW_BC_SEED1_SHA256:
        raise ValueError(
            "parent does not use the required Flow-BC Seed1 initialization"
        )
    if offline.get("shared_qp_checkpoint_sha256") != SHARED_QP_SHA256:
        raise ValueError("parent does not use the required shared Q/P checkpoint")
    if resume.get("format") != "guardian_flow_paper_resume_v2_atomic":
        raise ValueError("parent has no supported atomic resume generation")
    if resume.get("replay_states_embedded_atomically") is not True:
        raise ValueError("parent resume does not embed replay states atomically")
    if int(resume.get("step", -1)) != PARENT_STEP:
        raise ValueError("parent atomic state is not at 300k")
    if resume.get("trainer_state") != trainer_path.name or not trainer_path.is_file():
        raise ValueError("parent trainer state is missing")
    trainer_sha = _sha256(trainer_path)
    if resume.get("trainer_state_sha256") != trainer_sha:
        raise ValueError("parent trainer-state SHA256 binding is invalid")
    with trainer_path.open("rb") as stream:
        control = pickle.load(stream)
    evaluation_steps = tuple(
        int(row.get("step", -1)) for row in control.get("evaluations", ())
    )
    scheduler = control.get("scheduler") or {}
    scheduler_steps = {
        int(seed): int(value) for seed, value in (scheduler.get("steps") or {}).items()
    }
    online_replay = control.get("online_replay_state") or {}
    correction_replay = control.get("correction_replay_state") or {}
    if (
        control.get("format") != "guardian_flow_paper_resume_v2_atomic"
        or int(control.get("step", -1)) != PARENT_STEP
        or int(control.get("counters", {}).get("online_environment_steps", -1))
        != PARENT_STEP
        or evaluation_steps != EVALUATION_STEPS
        or set(scheduler_steps) != set(SCENE_SEEDS)
        or sum(scheduler_steps.values()) != PARENT_STEP
        or int(scheduler.get("current_seed", -1)) not in SCENE_SEEDS
        or online_replay.get("format") != "seed_indexed_replay_v1"
        or correction_replay.get("format") != "seed_indexed_replay_v1"
        or tuple(online_replay.get("seeds", ())) != SCENE_SEEDS
        or tuple(correction_replay.get("seeds", ())) != SCENE_SEEDS
        or int(online_replay.get("capacity", -1)) != 500_000
        or int(correction_replay.get("capacity", -1)) != 500_000
        or control.get("jax_key") is None
        or control.get("numpy_rng_state") is None
        or control.get("scheduler") is None
        or control.get("latent_counts") is None
        or int(control.get("discard_incomplete_episode_rows_on_resume", -1)) != 0
    ):
        raise ValueError("parent trainer state is not the complete 300k atomic state")
    for field in (
        "algorithm_checkpoint",
        "algorithm_checkpoint_sha256",
        "algorithm_checkpoint_backup",
        "algorithm_checkpoint_backup_sha256",
    ):
        if control.get(field) != resume.get(field):
            raise ValueError("parent trainer state does not match its resume index")
    checkpoint_path = parent / str(resume.get("algorithm_checkpoint", ""))
    backup_path = parent / str(resume.get("algorithm_checkpoint_backup", ""))
    checkpoint_sha = str(resume.get("algorithm_checkpoint_sha256", ""))
    if checkpoint_path.name != f"checkpoint_{PARENT_STEP:08d}.pkl":
        raise ValueError("parent atomic checkpoint is not checkpoint_00300000.pkl")
    if (
        not checkpoint_path.is_file()
        or _sha256(checkpoint_path) != checkpoint_sha
        or not backup_path.is_file()
        or _sha256(backup_path) != checkpoint_sha
        or resume.get("algorithm_checkpoint_backup_sha256") != checkpoint_sha
    ):
        raise ValueError("parent checkpoint or backup SHA256 binding is invalid")
    summary_steps = tuple(
        int(row.get("step", -1))
        for row in (summary.get("online") or {}).get("evaluations", ())
    )
    if summary_steps != EVALUATION_STEPS:
        raise ValueError("parent summary must contain exactly the seven 0--300k points")
    for step in EVALUATION_STEPS:
        for suffix in (".json", "_summary.json"):
            if not (parent / f"evaluation_{step:08d}{suffix}").is_file():
                raise ValueError(
                    f"parent evaluation evidence is missing at step {step}"
                )
    offline_checkpoint = parent / "offline_flow_bc.pkl"
    if not offline_checkpoint.is_file():
        raise ValueError("parent offline baseline checkpoint is missing")
    if _sha256(offline_checkpoint) != str(offline.get("checkpoint_sha256", "")):
        raise ValueError("parent offline baseline checkpoint SHA256 is invalid")
    return {
        "method": method,
        "config": config,
        "config_sha256": _sha256(config_path),
        "summary_sha256": _sha256(summary_path),
        "resume_index_sha256": _sha256(resume_path),
        "trainer_state_sha256": trainer_sha,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "offline_checkpoint": offline_checkpoint.name,
        "offline_checkpoint_sha256": _sha256(offline_checkpoint),
        "parent_profile_observed": observed_profile,
    }


def _assert_parent_unchanged(parent: Path, evidence: dict[str, Any]) -> None:
    expected = {
        "config.json": evidence["config_sha256"],
        "summary.json": evidence["summary_sha256"],
        "resume_index.json": evidence["resume_index_sha256"],
        "trainer_state_latest.pkl": evidence["trainer_state_sha256"],
        evidence["checkpoint"]: evidence["checkpoint_sha256"],
        evidence["offline_checkpoint"]: evidence["offline_checkpoint_sha256"],
    }
    changed = [
        name
        for name, expected_sha in expected.items()
        if not (parent / name).is_file() or _sha256(parent / name) != expected_sha
    ]
    if changed:
        raise RuntimeError(
            "300k parent changed while it was being copied: " + ", ".join(changed)
        )


def prepare(parent: Path, destination: Path) -> dict[str, Any]:
    parent = parent.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    if (
        parent == destination
        or parent in destination.parents
        or destination in parent.parents
    ):
        raise ValueError("parent and extension directories must be separate siblings")
    evidence = _validate_parent(parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.prepare.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            raise FileExistsError(
                "destination already exists; refusing to merge or overwrite: "
                f"{destination}"
            )
        staging: Path | None = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging.", dir=destination.parent
            )
        )
        try:
            shutil.copytree(
                parent, staging, dirs_exist_ok=True, copy_function=shutil.copy2
            )
            copied_config = staging / "config.json"
            if _sha256(copied_config) != evidence["config_sha256"]:
                raise RuntimeError("copied parent config failed SHA256 verification")
            copied_resume = staging / "resume_index.json"
            if _sha256(copied_resume) != evidence["resume_index_sha256"]:
                raise RuntimeError("copied resume index failed SHA256 verification")
            copied_trainer = staging / "trainer_state_latest.pkl"
            if _sha256(copied_trainer) != evidence["trainer_state_sha256"]:
                raise RuntimeError("copied trainer state failed SHA256 verification")
            copied_checkpoint = staging / evidence["checkpoint"]
            if _sha256(copied_checkpoint) != evidence["checkpoint_sha256"]:
                raise RuntimeError("copied 300k checkpoint failed SHA256 verification")
            parent_config_copy = staging / "parent_config_300k.json"
            shutil.copy2(copied_config, parent_config_copy)
            manifest = {
                "format": FORMAT,
                "method_slug": evidence["method"],
                "algorithm_seed": 1,
                "extended_from_step": PARENT_STEP,
                "target_online_steps": TARGET_STEP,
                "parent_run": str(parent),
                "destination_run": str(destination),
                "parent_profile_observed": evidence["parent_profile_observed"],
                "normalized_parent_profile": PARENT_PROFILE,
                "child_profile": CHILD_PROFILE,
                "parent_config": str(parent / "config.json"),
                "parent_config_sha256": evidence["config_sha256"],
                "parent_summary_sha256": evidence["summary_sha256"],
                "parent_resume_index_sha256": evidence["resume_index_sha256"],
                "parent_trainer_state": str(parent / "trainer_state_latest.pkl"),
                "parent_trainer_state_sha256": evidence["trainer_state_sha256"],
                "parent_checkpoint": str(parent / evidence["checkpoint"]),
                "parent_checkpoint_sha256": evidence["checkpoint_sha256"],
                "parent_offline_checkpoint_sha256": evidence[
                    "offline_checkpoint_sha256"
                ],
                "flow_bc_seed1_initialization_sha256": FLOW_BC_SEED1_SHA256,
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(prepare(args.parent_run, args.destination), indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
