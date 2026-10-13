#!/usr/bin/env python3
"""Run/resume the complete single-RTX4080 paper experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relax.utils.flow_paper import ALGORITHM_SEEDS  # noqa: E402
from scripts.train_guardian_dsac_metasafe import _sha256, _write_json  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-offline-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable), help="Bound srl100 Python"
    )
    parser.add_argument("--algorithm-seeds", default="0,1,2")
    parser.add_argument("--flow-q-seeds", default="0")
    parser.add_argument("--component-smoke", action="store_true")
    args = parser.parse_args(argv)
    algorithm_seeds = tuple(
        int(value) for value in args.algorithm_seeds.split(",") if value.strip()
    )
    seeds = tuple(int(value) for value in args.flow_q_seeds.split(",") if value.strip())
    if (
        not algorithm_seeds
        or len(algorithm_seeds) != len(set(algorithm_seeds))
        or any(seed not in ALGORITHM_SEEDS for seed in algorithm_seeds)
    ):
        parser.error("--algorithm-seeds must be a unique subset of 0,1,2")
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(seed not in ALGORITHM_SEEDS for seed in seeds)
    ):
        parser.error("--flow-q-seeds must be a unique subset of 0,1,2")
    if not args.component_smoke and algorithm_seeds != ALGORITHM_SEEDS:
        parser.error("formal paper suite fixes --algorithm-seeds to 0,1,2")
    if args.component_smoke and any(seed not in algorithm_seeds for seed in seeds):
        parser.error("component Flow-Q Seeds require matching Flow-BC smoke Seeds")
    args.algorithm_seeds = algorithm_seeds
    args.flow_q_seeds = seeds
    return args


def _completed(run_dir: Path, expected_steps: int) -> bool:
    path = run_dir / "summary.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if expected_steps == 0:
        return payload.get("online") is None and (run_dir / "final.pkl").is_file()
    online = payload.get("online") or {}
    return (
        int(online.get("actual_online_steps", -1)) == expected_steps
        and online.get("completed_full_budget") is True
        and (run_dir / "final.pkl").is_file()
    )


def _resumable(run_dir: Path, method: str) -> bool:
    if method == "flow_bc":
        return (run_dir / "offline_summary.json").is_file() and (
            run_dir / "offline_flow_bc.pkl"
        ).is_file()
    index_path = run_dir / "resume_index.json"
    trainer_state = run_dir / "trainer_state_latest.pkl"
    if not trainer_state.is_file() or not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        index.get("format") != "guardian_flow_paper_resume_v2_atomic"
        or index.get("replay_states_embedded_atomically") is not True
        or index.get("trainer_state") != trainer_state.name
    ):
        return False
    checkpoint = run_dir / str(index.get("algorithm_checkpoint", ""))
    backup = run_dir / str(index.get("algorithm_checkpoint_backup", ""))
    return checkpoint.is_file() or backup.is_file()


def _run(command, *, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(str(value) for value in command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=str(REPOSITORY_ROOT),
            env={
                **os.environ,
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "SDL_VIDEODRIVER": "dummy",
                "GUARDIAN_PAPER_EVAL_WORKERS": "4",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
        stream.write(f"[returncode={returncode} wall={time.time() - started:.1f}s]\n")
    if returncode != 0:
        raise RuntimeError(f"paper suite command failed ({returncode}): {command}")


def _training_command(args, method, seed, *, offline_checkpoint=None, resume_dir=None):
    command = [
        args.python,
        REPOSITORY_ROOT / "scripts/train_guardian_flow_dsac_paper.py",
        "--method",
        method,
        "--shared-offline-checkpoint",
        args.shared_offline_checkpoint,
        "--manifest",
        args.manifest,
        "--output-root",
        args.results_root,
        "--seed",
        str(seed),
    ]
    if method in ("gaussian", "flow_q_only", "full_flow"):
        command.extend(("--phase", "online"))
    if offline_checkpoint is not None:
        command.extend(("--flow-offline-checkpoint", offline_checkpoint))
    if resume_dir is not None:
        command.extend(("--resume-run", resume_dir))
    else:
        command.extend(("--run-name", f"{method}/train_seed_{seed}"))
    if args.component_smoke:
        command.append("--component-smoke")
    return command


def _ensure_run(args, method, seed, *, offline_checkpoint=None):
    run_dir = args.results_root / method / f"train_seed_{seed}"
    expected = (
        0 if method == "flow_bc" else (200 if args.component_smoke else 1_000_000)
    )
    if _completed(run_dir, expected):
        print(f"[paper-suite] already complete: {run_dir}", flush=True)
        return run_dir
    resume_dir = run_dir if _resumable(run_dir, method) else None
    if run_dir.exists() and resume_dir is None:
        raise RuntimeError(
            f"incomplete non-resumable run directory requires audit: {run_dir}"
        )
    command = _training_command(
        args,
        method,
        seed,
        offline_checkpoint=offline_checkpoint,
        resume_dir=resume_dir,
    )
    _run(command, log_path=args.results_root / "logs" / f"{method}_seed{seed}.log")
    if not _completed(run_dir, expected):
        raise RuntimeError(f"training command returned without complete run: {run_dir}")
    return run_dir


def main(argv=None) -> int:
    args = parse_args(argv)
    args.shared_offline_checkpoint = (
        args.shared_offline_checkpoint.expanduser().resolve()
    )
    args.manifest = args.manifest.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    if not args.component_smoke and (
        args.results_root == REPOSITORY_ROOT
        or REPOSITORY_ROOT in args.results_root.parents
    ):
        raise RuntimeError("formal suite results must be outside the Git checkout")
    args.results_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs = []

    flow_bc = {}
    for seed in args.algorithm_seeds:
        run = _ensure_run(args, "flow_bc", seed)
        flow_bc[seed] = run / "final.pkl"
        runs.append(run)

    teacher_dir = args.results_root / "guardian_mpc"
    teacher_output = teacher_dir / "teacher_evaluation.json"
    if not teacher_output.is_file():
        teacher_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.python,
            REPOSITORY_ROOT / "scripts/evaluate_guardian_flow_dsac_paper.py",
            "--checkpoint",
            flow_bc[0],
            "--output",
            teacher_output,
            "--policy",
            "teacher",
        ]
        _run(command, log_path=args.results_root / "logs" / "guardian_mpc.log")

    for seed in args.algorithm_seeds:
        runs.append(_ensure_run(args, "gaussian", seed))
    for seed in args.algorithm_seeds:
        runs.append(
            _ensure_run(
                args,
                "full_flow",
                seed,
                offline_checkpoint=flow_bc[seed],
            )
        )
    for seed in args.flow_q_seeds:
        runs.append(
            _ensure_run(
                args,
                "flow_q_only",
                seed,
                offline_checkpoint=flow_bc[seed],
            )
        )

    aggregate = [
        args.python,
        REPOSITORY_ROOT / "scripts/aggregate_guardian_flow_paper_results.py",
        "--results-root",
        args.results_root,
    ]
    if args.component_smoke:
        aggregate.append("--allow-partial")
    _run(aggregate, log_path=args.results_root / "logs" / "aggregate.log")
    _run(
        [
            args.python,
            REPOSITORY_ROOT / "scripts/plot_guardian_flow_paper_curves.py",
            "--input",
            args.results_root / "learning_curves.csv",
            "--output",
            args.results_root / "learning_curves.pdf",
        ],
        log_path=args.results_root / "logs" / "plot.log",
    )
    artifact_outputs = (
        "main_table.csv",
        "main_table.md",
        "per_seed_table.csv",
        "learning_curves.csv",
        "learning_curves.pdf",
        "learning_curves.png",
        "final_results.json",
        "EXPERIMENT_REPORT_ZH.md",
    )
    missing = [
        name for name in artifact_outputs if not (args.results_root / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"paper suite outputs are missing: {missing}")
    (args.results_root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(args.results_root / name)}  {name}\n"
            for name in artifact_outputs
        ),
        encoding="utf-8",
    )
    required_outputs = (*artifact_outputs, "SHA256SUMS")
    suite_summary = {
        "format": "guardian_flow_dsac_paper_suite_v1",
        "completed": True,
        "component_smoke": bool(args.component_smoke),
        "shared_offline_checkpoint": str(args.shared_offline_checkpoint),
        "shared_offline_checkpoint_sha256": _sha256(args.shared_offline_checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "flow_q_algorithm_seeds": list(args.flow_q_seeds),
        "algorithm_seeds_executed": list(args.algorithm_seeds),
        "runs": [str(path) for path in runs],
        "outputs": {
            name: {
                "path": str(args.results_root / name),
                "sha256": _sha256(args.results_root / name),
            }
            for name in required_outputs
        },
        "wall_time_seconds": time.time() - started,
    }
    _write_json(args.results_root / "suite_summary.json", suite_summary)
    print(json.dumps(suite_summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
