#!/usr/bin/env bash
set -u

usage() {
    echo "Usage: $0 PYTHON REPOSITORY RESULTS_ROOT SHARED_CHECKPOINT MANIFEST EXPECTED_REVISION METHOD SEED FLOW_OFFLINE_CHECKPOINT" >&2
    echo "Use '-' for FLOW_OFFLINE_CHECKPOINT when METHOD is gaussian." >&2
    exit 2
}

test "$#" -eq 9 || usage

PYTHON=$1
REPOSITORY=$2
RESULTS_ROOT=$3
SHARED_CHECKPOINT=$4
MANIFEST=$5
EXPECTED_REVISION=$6
METHOD=$7
SEED=$8
FLOW_OFFLINE_CHECKPOINT=$9
EXPERIMENT_PROFILE=${GUARDIAN_PAPER_EXPERIMENT_PROFILE:-main}
EXTENSION_PARENT_RUN=${GUARDIAN_PAPER_EXTENSION_PARENT_RUN:-}
PROXY_WEIGHT_SCHEDULE=${GUARDIAN_PROXY_WEIGHT_SCHEDULE:-fixed}
CORRECTION_WEIGHT=${GUARDIAN_CORRECTION_WEIGHT:-2.0}

case "$METHOD" in
    gaussian|full_flow|flow_q_only|flow_no_p|flow_no_correction) ;;
    *) usage ;;
esac
case "$EXPERIMENT_PROFILE" in
    main|same_seed_ablation|same_seed_ablation_1m|proxy_annealing_1m) ;;
    *) usage ;;
esac
if test "$EXPERIMENT_PROFILE" = "same_seed_ablation_1m"; then
    case "$METHOD" in
        full_flow|flow_no_p|flow_no_correction) ;;
        *) usage ;;
    esac
    test "$SEED" = "1" || usage
    test -n "$EXTENSION_PARENT_RUN" || usage
    test -d "$EXTENSION_PARENT_RUN" || usage
    EXPECTED_ONLINE_STEPS=1000000
elif test "$EXPERIMENT_PROFILE" = "proxy_annealing_1m"; then
    test "$METHOD" = "full_flow" || usage
    case "$SEED" in
        0|1) ;;
        *) usage ;;
    esac
    test -n "$EXTENSION_PARENT_RUN" || usage
    test -d "$EXTENSION_PARENT_RUN" || usage
    case "$PROXY_WEIGHT_SCHEDULE" in
        p_hard_off_300k|p_decay_300k_500k|p_decay_300k_700k|p_decay_floor_003) ;;
        *) usage ;;
    esac
    case "$CORRECTION_WEIGHT" in
        1|1.0|2|2.0|3|3.0) ;;
        *) usage ;;
    esac
    EXPECTED_ONLINE_STEPS=1000000
elif test "$EXPERIMENT_PROFILE" = "same_seed_ablation"; then
    test "$METHOD" = "full_flow" || usage
    case "$SEED" in
        0|1|2) ;;
        *) usage ;;
    esac
    EXPECTED_ONLINE_STEPS=300000
elif test "$METHOD" = "flow_no_p" || test "$METHOD" = "flow_no_correction"; then
    test "$SEED" = "1" || usage
    EXPECTED_ONLINE_STEPS=300000
else
    case "$SEED" in
        0|1|2) ;;
        *) usage ;;
    esac
    EXPECTED_ONLINE_STEPS=1000000
fi
if test "$METHOD" = "gaussian"; then
    test "$FLOW_OFFLINE_CHECKPOINT" = "-" || usage
else
    test "$FLOW_OFFLINE_CHECKPOINT" != "-" || usage
fi

RUN_NAME="$METHOD/train_seed_$SEED"
RUN_DIR="$RESULTS_ROOT/$RUN_NAME"
LOG_DIR="$RESULTS_ROOT/logs"
RUN_LOG="$LOG_DIR/${METHOD}_seed${SEED}.supplemental.log"
SUPERVISOR_LOG="$LOG_DIR/${METHOD}_seed${SEED}.supervisor.log"
MAX_ATTEMPTS=${GUARDIAN_PAPER_RUN_MAX_ATTEMPTS:-3}
RESTART_DELAY_SECONDS=${GUARDIAN_PAPER_RUN_RESTART_DELAY_SECONDS:-60}

mkdir -p "$LOG_DIR"

log_line() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >> "$SUPERVISOR_LOG"
}

run_complete() {
    "$PYTHON" - "$RUN_DIR" "$EXPECTED_ONLINE_STEPS" "$EXPERIMENT_PROFILE" "$PROXY_WEIGHT_SCHEDULE" "$CORRECTION_WEIGHT" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
expected_steps = int(sys.argv[2])
expected_profile = sys.argv[3]
expected_schedule = sys.argv[4]
expected_correction = float(sys.argv[5])
config = json.loads((run / "config.json").read_text(encoding="utf-8"))
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
online = summary.get("online") or {}
complete = (
    config.get("experiment_profile", "main") == expected_profile
    and int(config.get("budgets", {}).get("online_steps", -1)) == expected_steps
    and int(online.get("actual_online_steps", -1)) == expected_steps
    and online.get("completed_full_budget") is True
    and (run / "final.pkl").is_file()
)
if expected_profile == "proxy_annealing_1m":
    complete = complete and (
        (config.get("proxy_weight_schedule") or {}).get("name")
        == expected_schedule
        and float(config.get("effective_actor_correction_weight", -1.0))
        == expected_correction
    )
raise SystemExit(0 if complete else 1)
PY
}

run_resumable() {
    "$PYTHON" - "$RUN_DIR" <<'PY' >/dev/null 2>&1
import hashlib
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
index = json.loads((run / "resume_index.json").read_text(encoding="utf-8"))
trainer_name = str(index.get("trainer_state", ""))
trainer = run / trainer_name
checkpoint_name = str(index.get("algorithm_checkpoint", ""))
backup_name = str(index.get("algorithm_checkpoint_backup", ""))
checkpoint = run / checkpoint_name
backup = run / backup_name

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
expected_checkpoint_sha = str(index.get("algorithm_checkpoint_sha256", ""))
checkpoint_binding = (
    expected_checkpoint_sha
    and str(index.get("algorithm_checkpoint_backup_sha256", ""))
    == expected_checkpoint_sha
    and (
        (checkpoint.is_file() and sha256(checkpoint) == expected_checkpoint_sha)
        or (backup.is_file() and sha256(backup) == expected_checkpoint_sha)
    )
)
resumable = (
    (run / "config.json").is_file()
    and trainer.is_file()
    and index.get("format") in (
        "guardian_flow_paper_resume_v2_atomic",
        "guardian_flow_paper_resume_v3_atomic",
    )
    and index.get("replay_states_embedded_atomically") is True
    and pathlib.PurePath(trainer_name).name == trainer_name
    and pathlib.PurePath(checkpoint_name).name == checkpoint_name
    and pathlib.PurePath(backup_name).name == backup_name
    and sha256(trainer) == index.get("trainer_state_sha256")
    and checkpoint_binding
)
raise SystemExit(0 if resumable else 1)
PY
}

validate_host_and_source() {
    observed_revision=$(git -C "$REPOSITORY" rev-parse HEAD 2>/dev/null || true)
    dirty_source=$(git -C "$REPOSITORY" status --porcelain 2>/dev/null || true)
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
    if test "$observed_revision" != "$EXPECTED_REVISION"; then
        log_line "fatal_revision_mismatch expected=$EXPECTED_REVISION observed=$observed_revision"
        return 1
    fi
    if test -n "$dirty_source"; then
        log_line "fatal_dirty_source"
        return 1
    fi
    if test "$gpu_name" != "NVIDIA GeForce RTX 4080 SUPER"; then
        log_line "fatal_gpu_mismatch observed=$gpu_name"
        return 1
    fi
}

log_line "run_supervisor_started method=$METHOD seed=$SEED profile=$EXPERIMENT_PROFILE"
attempt=0
while true; do
    if run_complete; then
        log_line "run_complete method=$METHOD seed=$SEED"
        exit 0
    fi
    if ! validate_host_and_source; then
        exit 1
    fi

    command=(
        "$PYTHON"
        "$REPOSITORY/scripts/train_guardian_flow_dsac_paper.py"
        --method "$METHOD"
        --shared-offline-checkpoint "$SHARED_CHECKPOINT"
        --manifest "$MANIFEST"
        --output-root "$RESULTS_ROOT"
        --seed "$SEED"
        --phase online
        --online-steps "$EXPECTED_ONLINE_STEPS"
        --experiment-profile "$EXPERIMENT_PROFILE"
        --proxy-weight-schedule "$PROXY_WEIGHT_SCHEDULE"
        --correction-weight "$CORRECTION_WEIGHT"
    )
    if test "$METHOD" != "gaussian"; then
        command+=(--flow-offline-checkpoint "$FLOW_OFFLINE_CHECKPOINT")
    fi
    if test "$EXPERIMENT_PROFILE" = "same_seed_ablation_1m"; then
        command+=(
            --extend-ablation-from-300k
            --extension-parent-run "$EXTENSION_PARENT_RUN"
        )
    elif test "$EXPERIMENT_PROFILE" = "proxy_annealing_1m"; then
        command+=(
            --extend-proxy-annealing-from-300k
            --extension-parent-run "$EXTENSION_PARENT_RUN"
        )
    fi
    if test "$EXPERIMENT_PROFILE" = "same_seed_ablation_1m" || test "$EXPERIMENT_PROFILE" = "proxy_annealing_1m"; then
        if ! test -d "$RUN_DIR"; then
            log_line "fatal_missing_prepared_extension_run path=$RUN_DIR"
            exit 1
        fi
        if ! run_resumable; then
            log_line "fatal_incomplete_nonresumable_extension path=$RUN_DIR"
            exit 1
        fi
        command+=(--resume-run "$RUN_DIR")
        launch_mode=resume_extension
    elif test -d "$RUN_DIR"; then
        if ! run_resumable; then
            log_line "fatal_incomplete_nonresumable_run path=$RUN_DIR"
            exit 1
        fi
        command+=(--resume-run "$RUN_DIR")
        launch_mode=resume
    else
        command+=(--run-name "$RUN_NAME")
        launch_mode=new
    fi

    attempt=$((attempt + 1))
    if test "$attempt" -gt "$MAX_ATTEMPTS"; then
        log_line "fatal_restart_limit attempts=$attempt"
        exit 1
    fi
    log_line "run_launch attempt=$attempt mode=$launch_mode method=$METHOD seed=$SEED profile=$EXPERIMENT_PROFILE"
    (
        cd "$REPOSITORY" || exit 1
        if test "$EXPERIMENT_PROFILE" = "same_seed_ablation_1m"; then
            evaluation_lock=${GUARDIAN_PAPER_EVALUATION_LOCK:-/tmp/guardian_flow_ablation_1m_eval.lock}
            evaluation_workers=${GUARDIAN_PAPER_EVAL_WORKERS:-4}
        elif test "$EXPERIMENT_PROFILE" = "proxy_annealing_1m"; then
            evaluation_lock=${GUARDIAN_PAPER_EVALUATION_LOCK:-/tmp/guardian_flow_proxy_annealing_eval.lock}
            evaluation_workers=${GUARDIAN_PAPER_EVAL_WORKERS:-1}
        else
            evaluation_lock=${GUARDIAN_PAPER_EVALUATION_LOCK:-$RESULTS_ROOT/ablation_evaluation.lock}
            evaluation_workers=${GUARDIAN_PAPER_EVAL_WORKERS:-4}
        fi
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        SDL_VIDEODRIVER=dummy \
        GUARDIAN_PAPER_EVAL_WORKERS="$evaluation_workers" \
        GUARDIAN_PAPER_EVALUATION_LOCK="$evaluation_lock" \
        "${command[@]}"
    ) >> "$RUN_LOG" 2>&1
    returncode=$?
    if run_complete; then
        log_line "run_complete returncode=$returncode method=$METHOD seed=$SEED"
        exit 0
    fi
    log_line "run_exited returncode=$returncode attempt=$attempt restarting_in=${RESTART_DELAY_SECONDS}s"
    sleep "$RESTART_DELAY_SECONDS"
done
