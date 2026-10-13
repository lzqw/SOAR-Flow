#!/usr/bin/env bash
set -u

usage() {
    echo "Usage: $0 PYTHON ABLATION_REPOSITORY RESULTS_ROOT SHARED_CHECKPOINT MANIFEST EXPECTED_ABLATION_REVISION MAIN_REPOSITORY EXPECTED_MAIN_REVISION" >&2
    exit 2
}

test "$#" -eq 8 || usage

PYTHON=$1
REPOSITORY=$2
RESULTS_ROOT=$3
SHARED_CHECKPOINT=$4
MANIFEST=$5
EXPECTED_REVISION=$6
MAIN_REPOSITORY=$7
EXPECTED_MAIN_REVISION=$8
ABLATION_ROOT="$RESULTS_ROOT/ablations"
SUPERVISOR="$REPOSITORY/scripts/supervise_guardian_flow_paper_run.sh"
MAIN_TRAINER="$MAIN_REPOSITORY/scripts/train_guardian_flow_dsac_paper.py"
FLOW_OFFLINE_CHECKPOINT="$RESULTS_ROOT/flow_bc/train_seed_1/final.pkl"
DISPATCH_LOG="$ABLATION_ROOT/ablation_dispatch.log"
LAUNCH_SPACING_SECONDS=120
MIN_FREE_GPU_MIB=1600
MIN_AVAILABLE_RAM_MIB=18000

mkdir -p "$ABLATION_ROOT"

log_line() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >> "$DISPATCH_LOG"
}

source_and_gpu_valid() {
    test "$(git -C "$REPOSITORY" rev-parse HEAD 2>/dev/null)" = "$EXPECTED_REVISION" || return 1
    test -z "$(git -C "$REPOSITORY" status --porcelain 2>/dev/null)" || return 1
    test "$(git -C "$MAIN_REPOSITORY" rev-parse HEAD 2>/dev/null)" = "$EXPECTED_MAIN_REVISION" || return 1
    test -z "$(git -C "$MAIN_REPOSITORY" status --porcelain 2>/dev/null)" || return 1
    test "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" = "NVIDIA GeForce RTX 4080 SUPER"
}

run_complete() {
    local root=$1
    local method=$2
    local seed=$3
    local expected_steps=$4
    "$PYTHON" - "$root/$method/train_seed_$seed" "$method" "$seed" "$expected_steps" <<'PY' >/dev/null 2>&1
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
method = sys.argv[2]
seed = int(sys.argv[3])
expected_steps = int(sys.argv[4])
config = json.loads((run / "config.json").read_text(encoding="utf-8"))
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
online = summary.get("online") or {}
valid = (
    config.get("method_slug") == method
    and int(config.get("seed", -1)) == seed
    and int(config.get("budgets", {}).get("online_steps", -1)) == expected_steps
    and int(online.get("actual_online_steps", -1)) == expected_steps
    and online.get("completed_full_budget") is True
    and (run / "final.pkl").is_file()
)
raise SystemExit(0 if valid else 1)
PY
}

main_matrix_complete() {
    run_complete "$RESULTS_ROOT" gaussian 0 1000000 \
        && run_complete "$RESULTS_ROOT" gaussian 1 1000000 \
        && run_complete "$RESULTS_ROOT" gaussian 2 1000000 \
        && run_complete "$RESULTS_ROOT" full_flow 0 1000000 \
        && run_complete "$RESULTS_ROOT" full_flow 1 1000000 \
        && run_complete "$RESULTS_ROOT" full_flow 2 1000000 \
        && run_complete "$RESULTS_ROOT" flow_q_only 0 1000000
}

flow_source_ready() {
    "$PYTHON" - "$RESULTS_ROOT/flow_bc/train_seed_1" "$EXPECTED_MAIN_REVISION" <<'PY' >/dev/null 2>&1
import hashlib
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
revision = sys.argv[2]
config = json.loads((run / "config.json").read_text(encoding="utf-8"))
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
checkpoint = run / "final.pkl"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
offline = summary.get("offline") or {}
valid = (
    config.get("method_slug") == "flow_bc"
    and int(config.get("seed", -1)) == 1
    and config.get("source_code", {}).get("git_revision") == revision
    and int(offline.get("flow_matching_steps", -1)) == 80000
    and int(offline.get("same_latent_distillation_steps", -1)) == 40000
    and int(offline.get("offline_q_guided_actor_steps", -1)) == 0
    and summary.get("final_checkpoint_sha256") == digest
)
raise SystemExit(0 if valid else 1)
PY
}

main_trainers_active() {
    "$PYTHON" - "$MAIN_TRAINER" <<'PY'
from pathlib import Path
import os
import sys

target = str(Path(sys.argv[1]).resolve())
count = 0
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    if int(entry.name) == os.getpid():
        continue
    try:
        command = (entry / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    decoded = [value.decode("utf-8", "replace") for value in command if value]
    if target in decoded:
        count += 1
print(count)
PY
}

capacity_available() {
    local free_gpu_mib
    local available_ram_mib
    free_gpu_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    available_ram_mib=$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)
    test "$free_gpu_mib" -ge "$MIN_FREE_GPU_MIB" \
        && test "$available_ram_mib" -ge "$MIN_AVAILABLE_RAM_MIB"
}

ablation_complete() {
    run_complete "$ABLATION_ROOT" "$1" 1 300000
}

ablation_process_exists() {
    local method=$1
    "$PYTHON" - "$REPOSITORY/scripts/train_guardian_flow_dsac_paper.py" "$ABLATION_ROOT" "$method" <<'PY' >/dev/null 2>&1
from pathlib import Path
import sys

trainer = str(Path(sys.argv[1]).resolve())
root = str(Path(sys.argv[2]).resolve())
method = sys.argv[3]
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        command = [v.decode("utf-8", "replace") for v in (entry / "cmdline").read_bytes().split(b"\0") if v]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    joined = " ".join(command)
    if trainer in command and root in joined and f"--method {method}" in joined and "--seed 1" in joined:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

launch_one() {
    local method=$1
    local session="guardian_paper_ablation_${EXPECTED_REVISION:0:7}_${method}"
    if ablation_complete "$method"; then
        return 0
    fi
    if tmux has-session -t "$session" 2>/dev/null || ablation_process_exists "$method"; then
        return 0
    fi
    command="GUARDIAN_PAPER_RUN_MAX_ATTEMPTS=100 GUARDIAN_PAPER_RUN_RESTART_DELAY_SECONDS=60 GUARDIAN_PAPER_EVALUATION_LOCK=$ABLATION_ROOT/evaluation.lock bash $SUPERVISOR $PYTHON $REPOSITORY $ABLATION_ROOT $SHARED_CHECKPOINT $MANIFEST $EXPECTED_REVISION $method 1 $FLOW_OFFLINE_CHECKPOINT"
    tmux new-session -d -s "$session" "$command"
    log_line "ablation_launched session=$session method=$method seed=1 independent_single_model=true composite=false"
}

write_launch_manifest() {
    "$PYTHON" - "$ABLATION_ROOT/ABLATION_LAUNCH.json" "$EXPECTED_REVISION" "$EXPECTED_MAIN_REVISION" "$REPOSITORY" "$MAIN_REPOSITORY" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

path = Path(sys.argv[1])
payload = {
    "format": "guardian_flow_paper_ablation_launch_v1",
    "created_at_unix": time.time(),
    "ablation_training_revision": sys.argv[2],
    "main_training_revision": sys.argv[3],
    "ablation_repository": sys.argv[4],
    "main_repository": sys.argv[5],
    "gpu_required": "NVIDIA GeForce RTX 4080 SUPER",
    "shared_flow_bc_seed": 1,
    "methods": [
        {"method": "flow_no_p", "seed": 1, "online_steps": 300000},
        {"method": "flow_no_correction", "seed": 1, "online_steps": 300000},
    ],
    "independent_models": True,
    "shared_optimizer": False,
    "shared_online_replay": False,
    "composite_or_ensemble": False,
    "main_matrix_completed_before_launch": True,
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

log_line "ablation_dispatch_started revision=$EXPECTED_REVISION"
while ! main_matrix_complete; do
    if ! source_and_gpu_valid; then
        log_line "fatal_source_or_gpu_validation_failed"
        exit 1
    fi
    log_line "waiting_for_main_matrix active_main_trainers=$(main_trainers_active)"
    sleep 300
done

while test "$(main_trainers_active)" -ne 0; do
    log_line "waiting_for_main_trainers_to_exit active_main_trainers=$(main_trainers_active)"
    sleep 60
done

if ! source_and_gpu_valid || ! flow_source_ready; then
    log_line "fatal_prelaunch_validation_failed"
    exit 1
fi
write_launch_manifest

while ! ablation_complete flow_no_p || ! ablation_complete flow_no_correction; do
    if ! source_and_gpu_valid; then
        log_line "fatal_source_or_gpu_validation_failed"
        exit 1
    fi
    if ! ablation_complete flow_no_p; then
        if capacity_available; then
            launch_one flow_no_p
        else
            log_line "waiting_for_capacity_before_flow_no_p"
        fi
    fi
    sleep "$LAUNCH_SPACING_SECONDS"
    if ! ablation_complete flow_no_correction; then
        if capacity_available; then
            launch_one flow_no_correction
        else
            log_line "waiting_for_capacity_before_flow_no_correction"
        fi
    fi
    sleep 60
done
log_line "all_independent_ablations_complete"
