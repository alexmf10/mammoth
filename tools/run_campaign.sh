#!/usr/bin/env bash

# Sequential, resumable driver for the Split CIFAR-100 224 Pareto campaign.
# Production defaults are intentionally tied to the university server. The
# MAMMOTH_REPO_ROOT override exists only so this script can be tested safely
# from another checkout; it is not used by the server instructions.

set -uo pipefail

readonly DEFAULT_REPO_ROOT="/home/amf380/mammoth"
readonly OVERLAY_DIR="$HOME/.local/mammoth-pydeps"
readonly SAFETY_MARGIN_BYTES=$((15 * 1024 * 1024 * 1024))
readonly EXPECTED_RUNS=15

REPO_ROOT="${MAMMOTH_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
MODE="full"
DRY_RUN=0
CURRENT_CHILD_PID=""
DRIVER_PID_FILE=""

usage() {
    cat <<'EOF'
Usage:
  bash tools/run_campaign.sh             Run/resume the complete 15-run campaign.
  bash tools/run_campaign.sh --dry-run   Print all 15 commands without running them.
  bash tools/run_campaign.sh --smoke     Run/resume the single L2P acceptance smoke.

Required environment variable:
  PYTHON_BIN   Absolute path to the Python interpreter selected in Jupyter.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        --smoke)
            [[ "$MODE" == "full" && "$DRY_RUN" -eq 0 ]] || die "--smoke cannot be combined with another mode."
            MODE="smoke"
            ;;
        --dry-run)
            [[ "$MODE" == "full" && "$DRY_RUN" -eq 0 ]] || die "--dry-run cannot be combined with another mode."
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
    shift
done

[[ -n "${PYTHON_BIN:-}" ]] || die "PYTHON_BIN is not set. Copy the exact sys.executable path from Jupyter first."
[[ "$PYTHON_BIN" = /* ]] || die "PYTHON_BIN must be an absolute path: $PYTHON_BIN"
[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
[[ -d "$REPO_ROOT/.git" ]] || die "Mammoth checkout not found at $REPO_ROOT"
[[ -f "$REPO_ROOT/main.py" ]] || die "main.py not found at $REPO_ROOT/main.py"
[[ -f "$REPO_ROOT/tools/wrapper_peak.py" ]] || die "tools/wrapper_peak.py is missing."
[[ -f "$REPO_ROOT/tools/check_persistence.py" ]] || die "tools/check_persistence.py is missing."
[[ -d "$OVERLAY_DIR" ]] || die "Python overlay not found at $OVERLAY_DIR"

export PYTHONPATH="$OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT" || die "Cannot enter $REPO_ROOT"

if [[ "$MODE" == "smoke" ]]; then
    CAMPAIGN_ROOT="$REPO_ROOT/results/campaign-smoke"
else
    CAMPAIGN_ROOT="$REPO_ROOT/results/campaign"
fi

MANIFEST="$CAMPAIGN_ROOT/manifest.tsv"
CHECKPOINT_POLICY_FILE="$CAMPAIGN_ROOT/checkpoint_policy.txt"
readonly MANIFEST_HEADER=$'run_id\tmodel\tseed\tcommand\tpid\tstarted_at\tfinished_at\tduration_seconds\tprocess_exit_code\texit_code\tstatus\tlog_path\tpeak_json_path\tlogs_pyd_path\tcheckpoint_path\tdone_path'

cleanup_driver_pid() {
    if [[ -n "$DRIVER_PID_FILE" && -f "$DRIVER_PID_FILE" ]]; then
        local recorded_pid
        recorded_pid=$(<"$DRIVER_PID_FILE")
        if [[ "$recorded_pid" == "$$" ]]; then
            rm -f "$DRIVER_PID_FILE"
        fi
    fi
}

forward_signal_and_exit() {
    local signal_name="$1"
    local exit_code="$2"
    if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
        kill -s "$signal_name" "$CURRENT_CHILD_PID" 2>/dev/null || true
    fi
    exit "$exit_code"
}

trap cleanup_driver_pid EXIT
trap 'forward_signal_and_exit INT 130' INT
trap 'forward_signal_and_exit TERM 143' TERM

init_runtime_files() {
    mkdir -p "$CAMPAIGN_ROOT" || die "Cannot create $CAMPAIGN_ROOT"

    DRIVER_PID_FILE="$CAMPAIGN_ROOT/driver.pid"
    if [[ -f "$DRIVER_PID_FILE" ]]; then
        local previous_pid
        previous_pid=$(<"$DRIVER_PID_FILE")
        if [[ "$previous_pid" =~ ^[0-9]+$ ]] && [[ "$previous_pid" != "$$" ]] && kill -0 "$previous_pid" 2>/dev/null; then
            die "Another campaign driver is still alive with PID $previous_pid ($DRIVER_PID_FILE)."
        fi
    fi
    printf '%s\n' "$$" > "$DRIVER_PID_FILE" || die "Cannot write $DRIVER_PID_FILE"

    if [[ -f "$MANIFEST" ]]; then
        local current_header
        IFS= read -r current_header < "$MANIFEST" || true
        [[ "$current_header" == "$MANIFEST_HEADER" ]] || die "Existing manifest has an incompatible header: $MANIFEST"
    else
        printf '%s\n' "$MANIFEST_HEADER" > "$MANIFEST" || die "Cannot create $MANIFEST"
    fi
}

append_manifest_start() {
    local run_id="$1" model="$2" seed="$3" command_text="$4" pid="$5" started_at="$6"
    local log_path="$7" peak_path="$8" logs_pyd_path="$9" checkpoint_path="${10}" done_path="${11}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t\t\t\t\tRUNNING\t%s\t%s\t%s\t%s\t%s\n' \
        "$run_id" "$model" "$seed" "$command_text" "$pid" "$started_at" \
        "$log_path" "$peak_path" "$logs_pyd_path" "$checkpoint_path" "$done_path" >> "$MANIFEST"
}

finalize_manifest() {
    local run_id="$1" pid="$2" started_at="$3" finished_at="$4" duration="$5"
    local process_exit="$6" effective_exit="$7" status="$8"
    local temporary="$MANIFEST.tmp.$$"

    awk -F '\t' -v OFS='\t' \
        -v target_run="$run_id" -v target_pid="$pid" -v target_start="$started_at" \
        -v finished="$finished_at" -v duration="$duration" -v process_exit="$process_exit" \
        -v effective_exit="$effective_exit" -v final_status="$status" '
            NR == 1 { print; next }
            $1 == target_run && $5 == target_pid && $6 == target_start && $11 == "RUNNING" {
                $7 = finished
                $8 = duration
                $9 = process_exit
                $10 = effective_exit
                $11 = final_status
                updated = 1
            }
            { print }
            END { if (!updated) exit 42 }
        ' "$MANIFEST" > "$temporary"
    local awk_status=$?
    if [[ "$awk_status" -ne 0 ]]; then
        rm -f "$temporary"
        return 1
    fi
    mv -f "$temporary" "$MANIFEST"
}

line_count() {
    local path="$1"
    if [[ -f "$path" ]]; then
        awk 'END { print NR + 0 }' "$path"
    else
        printf '0\n'
    fi
}

quote_command() {
    local quoted=""
    printf -v quoted '%q ' "$@"
    quoted="${quoted% }"
    quoted="${quoted//$'\t'/ }"
    quoted="${quoted//$'\n'/ }"
    printf '%s\n' "$quoted"
}

largest_smoke_checkpoint() {
    local probe_dir="$REPO_ROOT/results/campaign-smoke/l2p-s0/checkpoints"
    local candidate size largest_size=0 largest_path=""

    [[ -d "$probe_dir" ]] || return 1
    while IFS= read -r -d '' candidate; do
        size=$(stat -c '%s' "$candidate" 2>/dev/null) || continue
        if ((size > largest_size)); then
            largest_size=$size
            largest_path="$candidate"
        fi
    done < <(find "$probe_dir" -maxdepth 1 -type f -name '*.pt' -print0 2>/dev/null)

    ((largest_size > 0)) || return 1
    printf '%s\t%s\n' "$largest_size" "$largest_path"
}

require_completed_smoke_gate() {
    local smoke_done="$REPO_ROOT/results/campaign-smoke/l2p-s0/.done"
    local probe_record=""

    [[ -f "$smoke_done" ]] || die "The full campaign is blocked: smoke gate marker is missing at $smoke_done. Run tools/run_campaign.sh --smoke first."
    probe_record=$(largest_smoke_checkpoint 2>/dev/null) || true
    [[ -n "$probe_record" ]] || die "The full campaign is blocked: no non-empty checkpoint probe exists under results/campaign-smoke/l2p-s0/checkpoints/. Rerun the smoke gate."
}

parse_quota_available_bytes() {
    # quota(1), without -s, reports 1 KiB blocks. Both the normal one-line
    # layout and the wrapped-filesystem layout are accepted. If several
    # filesystems have limits, use the smallest headroom (conservative).
    awk '
        function clean(value) {
            gsub(/\*/, "", value)
            return value
        }
        function numeric(value) {
            return value ~ /^[0-9]+$/
        }
        function consider(used, soft, hard, limit, remaining) {
            used = clean(used)
            soft = clean(soft)
            hard = clean(hard)
            if (!numeric(used) || !numeric(soft) || !numeric(hard)) return
            limit = 0
            if (soft + 0 > 0) limit = soft + 0
            if (hard + 0 > 0 && (limit == 0 || hard + 0 < limit)) limit = hard + 0
            if (limit == 0) return
            remaining = limit - (used + 0)
            if (remaining < 0) remaining = 0
            if (!found || remaining < minimum) minimum = remaining
            found = 1
        }
        {
            if (NF >= 4) consider($2, $3, $4)
            if (NF >= 3 && clean($1) ~ /^[0-9]+$/) consider($1, $2, $3)
        }
        END {
            if (found) printf "%.0f\n", minimum * 1024
        }
    '
}

CHECKPOINTS_ENABLED=0
CHECKPOINT_POLICY_REASON=""
PROBE_SIZE_BYTES=0
PROBE_PATH=""
ESTIMATED_CHECKPOINT_BYTES=0
FILESYSTEM_AVAILABLE_BYTES=0
QUOTA_AVAILABLE_BYTES=""
EFFECTIVE_AVAILABLE_BYTES=0
AVAILABLE_SOURCE="unknown"
QUOTA_REPORT=""
QUOTA_CHECK_RESULT="not_checked"

resolve_checkpoint_policy() {
    if [[ -f "$CHECKPOINT_POLICY_FILE" ]]; then
        local persisted
        persisted=$(awk -F '=' '$1 == "save_final_checkpoints" { print $2; exit }' "$CHECKPOINT_POLICY_FILE")
        if [[ "$persisted" == "0" || "$persisted" == "1" ]]; then
            CHECKPOINTS_ENABLED=$persisted
            CHECKPOINT_POLICY_REASON="reusing the decision persisted in $CHECKPOINT_POLICY_FILE"
            return
        fi
        die "Invalid checkpoint policy file: $CHECKPOINT_POLICY_FILE"
    fi

    local probe_record=""
    probe_record=$(largest_smoke_checkpoint 2>/dev/null) || true
    if [[ -z "$probe_record" ]]; then
        CHECKPOINT_POLICY_REASON="disabled: no smoke checkpoint probe was found"
        return
    fi
    IFS=$'\t' read -r PROBE_SIZE_BYTES PROBE_PATH <<< "$probe_record"
    ESTIMATED_CHECKPOINT_BYTES=$((PROBE_SIZE_BYTES * EXPECTED_RUNS))

    local filesystem_available_kib
    filesystem_available_kib=$(df -Pk "$HOME" 2>/dev/null | awk 'END { print $4 }')
    if [[ ! "$filesystem_available_kib" =~ ^[0-9]+$ ]]; then
        CHECKPOINT_POLICY_REASON="disabled: available filesystem space under HOME could not be measured"
        return
    fi
    FILESYSTEM_AVAILABLE_BYTES=$((filesystem_available_kib * 1024))
    EFFECTIVE_AVAILABLE_BYTES=$FILESYSTEM_AVAILABLE_BYTES
    AVAILABLE_SOURCE="df"

    if command -v quota >/dev/null 2>&1; then
        local quota_exit
        QUOTA_REPORT=$(quota -w 2>&1)
        quota_exit=$?
        QUOTA_AVAILABLE_BYTES=$(printf '%s\n' "$QUOTA_REPORT" | parse_quota_available_bytes)
        if [[ "$QUOTA_AVAILABLE_BYTES" =~ ^[0-9]+$ ]] && ((QUOTA_AVAILABLE_BYTES < EFFECTIVE_AVAILABLE_BYTES)); then
            EFFECTIVE_AVAILABLE_BYTES=$QUOTA_AVAILABLE_BYTES
            AVAILABLE_SOURCE="quota"
            QUOTA_CHECK_RESULT="parsed_limit"
        elif [[ "$QUOTA_AVAILABLE_BYTES" =~ ^[0-9]+$ ]]; then
            AVAILABLE_SOURCE="df (quota headroom is larger)"
            QUOTA_CHECK_RESULT="parsed_limit"
        elif ((quota_exit == 0)) && { [[ -z "$QUOTA_REPORT" ]] || printf '%s\n' "$QUOTA_REPORT" | grep -Eiq '(^|:)[[:space:]]*none[[:space:]]*$|no quota|quotas? (are )?not enabled'; }; then
            QUOTA_CHECK_RESULT="explicitly_no_quota"
        else
            QUOTA_CHECK_RESULT="unparseable"
            CHECKPOINT_POLICY_REASON="disabled: quota exists but its available space could not be proved safely"
            return
        fi
    else
        QUOTA_REPORT="quota command is not installed; df was used"
        QUOTA_CHECK_RESULT="command_missing"
        CHECKPOINT_POLICY_REASON="disabled: quota command is unavailable, so the HOME margin could not be proved safely"
        return
    fi

    local required_with_margin=$((ESTIMATED_CHECKPOINT_BYTES + SAFETY_MARGIN_BYTES))
    if ((EFFECTIVE_AVAILABLE_BYTES > required_with_margin)); then
        CHECKPOINTS_ENABLED=1
        CHECKPOINT_POLICY_REASON="enabled: estimated 15-checkpoint footprint leaves more than 15 GiB free"
    else
        CHECKPOINT_POLICY_REASON="disabled: estimated 15-checkpoint footprint would not leave more than 15 GiB free"
    fi
}

write_checkpoint_policy() {
    {
        printf 'save_final_checkpoints=%s\n' "$CHECKPOINTS_ENABLED"
        printf 'reason=%s\n' "$CHECKPOINT_POLICY_REASON"
        printf 'smoke_probe_path=%s\n' "$PROBE_PATH"
        printf 'smoke_probe_bytes=%s\n' "$PROBE_SIZE_BYTES"
        printf 'estimated_15_checkpoints_bytes=%s\n' "$ESTIMATED_CHECKPOINT_BYTES"
        printf 'safety_margin_bytes=%s\n' "$SAFETY_MARGIN_BYTES"
        printf 'filesystem_available_bytes=%s\n' "$FILESYSTEM_AVAILABLE_BYTES"
        printf 'quota_available_bytes=%s\n' "$QUOTA_AVAILABLE_BYTES"
        printf 'quota_check_result=%s\n' "$QUOTA_CHECK_RESULT"
        printf 'effective_available_bytes=%s\n' "$EFFECTIVE_AVAILABLE_BYTES"
        printf 'available_source=%s\n' "$AVAILABLE_SOURCE"
        printf '%s\n' 'quota_report_begin'
        printf '%s\n' "$QUOTA_REPORT"
        printf '%s\n' 'quota_report_end'
    } > "$CHECKPOINT_POLICY_FILE" || die "Cannot write $CHECKPOINT_POLICY_FILE"
}

print_checkpoint_policy() {
    printf 'Checkpoint policy: %s\n' "$CHECKPOINT_POLICY_REASON"
    printf '  save final checkpoints: %s\n' "$CHECKPOINTS_ENABLED"
    if ((PROBE_SIZE_BYTES > 0)); then
        printf '  smoke probe: %s bytes (%s)\n' "$PROBE_SIZE_BYTES" "$PROBE_PATH"
        printf '  estimate for 15 runs: %s bytes\n' "$ESTIMATED_CHECKPOINT_BYTES"
        printf '  effective free space: %s bytes (source: %s)\n' "$EFFECTIVE_AVAILABLE_BYTES" "$AVAILABLE_SOURCE"
        printf '  required safety margin after checkpoints: %s bytes (15 GiB)\n' "$SAFETY_MARGIN_BYTES"
    fi
}

expected_values() {
    local model="$1"
    if [[ "$MODE" == "smoke" ]]; then
        EXPECTED_TASKS_FOR_RUN=1
        EXPECTED_EPOCHS_FOR_RUN=1
        EXPECTED_LR_FOR_RUN="0.0075"
        return
    fi

    EXPECTED_TASKS_FOR_RUN=10
    case "$model" in
        l2p|dualprompt)
            EXPECTED_EPOCHS_FOR_RUN=5
            EXPECTED_LR_FOR_RUN="0.0075"
            ;;
        coda_prompt)
            EXPECTED_EPOCHS_FOR_RUN=20
            EXPECTED_LR_FOR_RUN="0.001"
            ;;
        *)
            die "Unsupported campaign model: $model"
            ;;
    esac
}

build_run_command() {
    local model="$1" seed="$2" run_id="$3" run_dir="$4"
    RUN_NOTES="pareto-$run_id"
    RUN_CHECKPOINT_PATH="-"
    TRAINING_ARGS=(
        --model "$model"
        --dataset seq-cifar100-224
        --model_config best
        --batch_size 64
        --device 0
        --seed "$seed"
        --base_path ./data/
        --non_verbose 1
    )

    if [[ "$MODE" == "smoke" ]]; then
        RUN_NOTES="gate-smoke-l2p-s0"
        RUN_CHECKPOINT_PATH="$run_dir/checkpoints"
        TRAINING_ARGS+=(
            --debug_mode 1
            --fitting_mode iters
            --n_iters 5
            --stop_after 1
            --savecheck last
            --checkpoint_path "$RUN_CHECKPOINT_PATH"
            --ckpt_name checkpoint-size-probe
        )
    else
        TRAINING_ARGS+=(--fitting_mode epochs)
        if ((CHECKPOINTS_ENABLED)); then
            RUN_CHECKPOINT_PATH="$run_dir/checkpoints"
            TRAINING_ARGS+=(
                --savecheck last
                --checkpoint_path "$RUN_CHECKPOINT_PATH"
                --ckpt_name "$run_id"
            )
        fi
    fi

    TRAINING_ARGS+=(--notes "$RUN_NOTES")
    RUN_COMMAND=("$PYTHON_BIN" -u "$REPO_ROOT/tools/wrapper_peak.py" "${TRAINING_ARGS[@]}")
}

checkpoint_exists() {
    local checkpoint_dir="$1"
    [[ "$checkpoint_dir" != "-" && -d "$checkpoint_dir" ]] || return 1
    [[ -n "$(find "$checkpoint_dir" -maxdepth 1 -type f -name '*.pt' -print -quit 2>/dev/null)" ]]
}

valid_peak_json() {
    local peak_path="$1" expected_pid="$2"
    "$PYTHON_BIN" - "$peak_path" "$expected_pid" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_pid = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))

assert payload.get("pid") == expected_pid
assert payload.get("status") == "success"
assert payload.get("exit_code") == 0
assert isinstance(payload.get("started_at"), str) and payload["started_at"]
assert isinstance(payload.get("finished_at"), str) and payload["finished_at"]
assert isinstance(payload.get("argv"), list) and payload["argv"]
assert isinstance(payload.get("git_sha"), str) and payload["git_sha"] not in {"", "unknown"}
for key in ("duration_seconds", "peak_allocated_mib", "peak_reserved_mib"):
    value = payload.get(key)
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value) and value > 0
assert "memory_measurement_error" not in payload
PY
}

run_one() {
    local model="$1" seed="$2"
    local run_id="$model-s$seed"
    local run_dir="$CAMPAIGN_ROOT/$run_id"
    local log_path="$run_dir/$run_id.log"
    local peak_path="$run_dir/peak.json"
    local done_path="$run_dir/.done"
    local process_pid_path="$run_dir/process.pid"
    local logs_pyd_path="$REPO_ROOT/data/results/class-il/seq-cifar100-224/$model/logs.pyd"

    expected_values "$model"
    build_run_command "$model" "$seed" "$run_id" "$run_dir"

    local display_command
    display_command=$(quote_command env "PEAK_FILE=$peak_path" "${RUN_COMMAND[@]}")

    if ((DRY_RUN)); then
        local planned_status="RUN"
        [[ -f "$done_path" ]] && planned_status="SKIP (.done already exists)"
        printf '\n[%s] %s\n' "$planned_status" "$run_id"
        printf '  %s\n' "$display_command"
        return 0
    fi

    if [[ -f "$done_path" ]]; then
        printf '[SKIP] %s already completed (%s).\n' "$run_id" "$done_path"
        return 0
    fi

    mkdir -p "$run_dir" || {
        printf '[FAIL] Cannot create %s\n' "$run_dir" >&2
        return 1
    }

    if [[ -f "$process_pid_path" ]]; then
        local old_process_pid
        old_process_pid=$(<"$process_pid_path")
        if [[ "$old_process_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_process_pid" 2>/dev/null; then
            printf '[BLOCKED] %s still has a live process with PID %s. Refusing to overlap runs.\n' "$run_id" "$old_process_pid" >&2
            return 75
        fi
        rm -f "$process_pid_path"
    fi

    local logs_before
    logs_before=$(line_count "$logs_pyd_path")
    rm -f "$peak_path" "$done_path"

    local started_at started_epoch
    started_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    started_epoch=$(date +'%s')
    printf '[START] %s at %s\n' "$run_id" "$started_at"
    printf '        %s\n' "$display_command"

    PEAK_FILE="$peak_path" "${RUN_COMMAND[@]}" > "$log_path" 2>&1 &
    local process_pid=$!
    CURRENT_CHILD_PID="$process_pid"
    printf '%s\n' "$process_pid" > "$process_pid_path"

    if ! append_manifest_start "$run_id" "$model" "$seed" "$display_command" "$process_pid" "$started_at" \
        "$log_path" "$peak_path" "$logs_pyd_path" "$RUN_CHECKPOINT_PATH" "$done_path"; then
        printf '[FAIL] Could not record the running process in %s; terminating PID %s.\n' "$MANIFEST" "$process_pid" >&2
        kill -TERM "$process_pid" 2>/dev/null || true
        wait "$process_pid" 2>/dev/null || true
        CURRENT_CHILD_PID=""
        rm -f "$process_pid_path"
        return 1
    fi

    wait "$process_pid"
    local process_exit=$?
    CURRENT_CHILD_PID=""
    rm -f "$process_pid_path"

    local finished_at finished_epoch duration_seconds
    finished_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    finished_epoch=$(date +'%s')
    duration_seconds=$((finished_epoch - started_epoch))

    local effective_exit=$process_exit
    local final_status="FAILED_PROCESS"

    if ((process_exit == 0)); then
        if [[ ! -s "$peak_path" ]] || ! valid_peak_json "$peak_path" "$process_pid" >/dev/null 2>&1; then
            effective_exit=90
            final_status="FAILED_PEAK_JSON"
            printf '\n[gate] Missing or invalid peak JSON: %s\n' "$peak_path" >> "$log_path"
        else
            printf '\n[gate] Verifying the new logs.pyd record (lines after %s).\n' "$logs_before" >> "$log_path"
            "$PYTHON_BIN" "$REPO_ROOT/tools/check_persistence.py" \
                --logs "$logs_pyd_path" \
                --model "$model" \
                --dataset seq-cifar100-224 \
                --seed "$seed" \
                --notes "$RUN_NOTES" \
                --expected-tasks "$EXPECTED_TASKS_FOR_RUN" \
                --expected-epochs "$EXPECTED_EPOCHS_FOR_RUN" \
                --expected-lr "$EXPECTED_LR_FOR_RUN" \
                --expected-batch-size 64 \
                --after-line "$logs_before" >> "$log_path" 2>&1
            local persistence_exit=$?
            if ((persistence_exit != 0)); then
                effective_exit=91
                final_status="FAILED_PERSISTENCE"
            elif [[ "$RUN_CHECKPOINT_PATH" != "-" ]] && ! checkpoint_exists "$RUN_CHECKPOINT_PATH"; then
                effective_exit=92
                final_status="FAILED_CHECKPOINT"
                printf '\n[gate] No final checkpoint was found under %s.\n' "$RUN_CHECKPOINT_PATH" >> "$log_path"
            else
                effective_exit=0
                final_status="SUCCESS"
            fi
        fi
    fi

    if ((effective_exit == 0)); then
        if ! touch "$done_path"; then
            effective_exit=93
            final_status="FAILED_DONE_MARKER"
        fi
    fi

    if ! finalize_manifest "$run_id" "$process_pid" "$started_at" "$finished_at" "$duration_seconds" \
        "$process_exit" "$effective_exit" "$final_status"; then
        rm -f "$done_path"
        printf '[FAIL] Could not finalize the manifest row for %s.\n' "$run_id" >&2
        return 1
    fi

    if ((effective_exit == 0)); then
        printf '[DONE] %s in %ss (PID %s).\n' "$run_id" "$duration_seconds" "$process_pid"
        return 0
    fi

    printf '[FAIL] %s: process_exit=%s gate_exit=%s status=%s. Continuing with the next run.\n' \
        "$run_id" "$process_exit" "$effective_exit" "$final_status" >&2
    return 1
}

if [[ "$MODE" == "full" && "$DRY_RUN" -eq 0 ]]; then
    require_completed_smoke_gate
fi

if [[ "$MODE" == "full" ]]; then
    resolve_checkpoint_policy
    print_checkpoint_policy
fi

if ((DRY_RUN)); then
    printf '\nComplete campaign plan (seed outer loop; model inner loop):\n'
else
    init_runtime_files
    if [[ "$MODE" == "full" && ! -f "$CHECKPOINT_POLICY_FILE" ]]; then
        write_checkpoint_policy
    fi
fi

failures=0
blocked=0

if [[ "$MODE" == "smoke" ]]; then
    run_one l2p 0
    run_status=$?
    if ((run_status == 75)); then
        blocked=1
    elif ((run_status != 0)); then
        failures=$((failures + 1))
    fi
else
    for seed in 0 1 2 3 4; do
        for model in l2p dualprompt coda_prompt; do
            run_one "$model" "$seed"
            run_status=$?
            if ((run_status == 75)); then
                blocked=1
                break 2
            elif ((run_status != 0)); then
                failures=$((failures + 1))
            fi
        done
    done
fi

if ((DRY_RUN)); then
    printf '\nDry run complete: no training process was started.\n'
    exit 0
fi

if ((blocked)); then
    printf 'Campaign stopped before launching another run because an earlier process is still alive.\n' >&2
    exit 75
fi

if ((failures > 0)); then
    printf 'Campaign attempted every uncompleted run; %s run(s) failed the process or metric gate.\n' "$failures" >&2
    exit 1
fi

printf 'All requested runs are complete.\n'
