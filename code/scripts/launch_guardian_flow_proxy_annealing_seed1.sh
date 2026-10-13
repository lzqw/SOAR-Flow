#!/usr/bin/env bash
set -eu

usage() {
    echo "Usage: $0 PYTHON REPOSITORY RESULTS_ROOT PARENT_300K SHARED_CHECKPOINT MANIFEST EXPECTED_REVISION FLOW_BC_SEED1" >&2
    exit 2
}

test "$#" -eq 8 || usage

PYTHON=$1
REPOSITORY=$2
RESULTS_ROOT=$3
PARENT_300K=$4
SHARED_CHECKPOINT=$5
MANIFEST=$6
EXPECTED_REVISION=$7
FLOW_BC_SEED1=$8
PREPARER="$REPOSITORY/scripts/prepare_guardian_flow_proxy_annealing_1m.py"
SUPERVISOR="$REPOSITORY/scripts/supervise_guardian_flow_paper_run.sh"
EVALUATION_LOCK=${GUARDIAN_PAPER_EVALUATION_LOCK:-/tmp/guardian_flow_proxy_annealing_eval.lock}

test -x "$PYTHON" || usage
test -d "$REPOSITORY" || usage
test -d "$PARENT_300K" || usage
test -f "$SHARED_CHECKPOINT" || usage
test -f "$MANIFEST" || usage
test -f "$FLOW_BC_SEED1" || usage
test "$(git -C "$REPOSITORY" rev-parse HEAD)" = "$EXPECTED_REVISION" || usage
test -z "$(git -C "$REPOSITORY" status --porcelain)" || usage

mkdir -p "$RESULTS_ROOT"
LAUNCH_LOG="$RESULTS_ROOT/proxy_annealing_launch.log"
PID_FILE="$RESULTS_ROOT/proxy_annealing_supervisors.tsv"
: > "$PID_FILE"

for schedule in \
    p_hard_off_300k \
    p_decay_300k_500k \
    p_decay_300k_700k \
    p_decay_floor_003
do
    branch_root="$RESULTS_ROOT/$schedule"
    destination="$branch_root/full_flow/train_seed_1"
    if ! test -d "$destination"; then
        "$PYTHON" "$PREPARER" \
            --parent-run "$PARENT_300K" \
            --destination "$destination" \
            --proxy-weight-schedule "$schedule" \
            --correction-weight 2.0 >> "$LAUNCH_LOG" 2>&1
    fi
done

for schedule in \
    p_hard_off_300k \
    p_decay_300k_500k \
    p_decay_300k_700k \
    p_decay_floor_003
do
    branch_root="$RESULTS_ROOT/$schedule"
    nohup env \
        GUARDIAN_PAPER_EXPERIMENT_PROFILE=proxy_annealing_1m \
        GUARDIAN_PAPER_EXTENSION_PARENT_RUN="$PARENT_300K" \
        GUARDIAN_PROXY_WEIGHT_SCHEDULE="$schedule" \
        GUARDIAN_CORRECTION_WEIGHT=2.0 \
        GUARDIAN_PAPER_EVALUATION_LOCK="$EVALUATION_LOCK" \
        GUARDIAN_PAPER_RUN_MAX_ATTEMPTS=10 \
        MALLOC_ARENA_MAX=2 \
        bash "$SUPERVISOR" \
            "$PYTHON" \
            "$REPOSITORY" \
            "$branch_root" \
            "$SHARED_CHECKPOINT" \
            "$MANIFEST" \
            "$EXPECTED_REVISION" \
            full_flow \
            1 \
            "$FLOW_BC_SEED1" >> "$LAUNCH_LOG" 2>&1 < /dev/null &
    supervisor_pid=$!
    printf '%s\t%s\n' "$schedule" "$supervisor_pid" >> "$PID_FILE"
done

printf 'launched four proxy-annealing supervisors; pids: %s\n' "$PID_FILE"
