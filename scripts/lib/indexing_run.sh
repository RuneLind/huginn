# shellcheck shell=bash
#
# indexing_run.sh — observational helpers for the daily indexing scripts.
#
# The daily scripts spend most of their wall clock OUTSIDE the reindex huginn
# can see (mimir: ~51 min tagging vs ~15 min reindex). These helpers let a
# script report its own phases, so the run ledger records the whole job instead
# of the last quarter of it.
#
# Sourced from the private sub-repos, all of which compute PROJECT_DIR as the
# huginn root:
#
#     HELPER="${PROJECT_DIR}/scripts/lib/indexing_run.sh"
#     if [ -f "${HELPER:-}" ] && . "$HELPER"; then :; else
#         run_begin(){ :; }; phase_begin(){ :; }; phase_end(){ :; }; run_end(){ :; }; RUN_ID=""
#     fi
#
# Three hazards this file is written around, all of which would otherwise abort
# an unattended job under `set -euo pipefail` — trading "no observability" for
# "no indexing", which is strictly worse:
#
#   set -e   Every exported helper swallows its own failures and returns 0, and
#            every call site adds `|| true`. run_end curls an API these jobs
#            routinely find down.
#   set -u   run_begin is what exports RUN_ID; with the no-op stub it never runs,
#            so call sites expand ${RUN_ID:-}, and the stub block sets RUN_ID="".
#   source   `.` exits with the status of the LAST COMMAND in this file, so this
#            file MUST end with an explicit `return 0`. Neither the `&&`/`||` nor
#            the `if/else` sourcing form fixes that on its own.
#
# EVERYTHING EXPORTED HERE IS OBSERVATIONAL. That is what makes stubbing each
# symbol to `:` safe. Functional logic (notably poll_update_status) deliberately
# stays duplicated in each script: stubbed to `:` it would make a script treat
# every reindex as instantly complete, and left unstubbed it would abort the job
# after the reindex was triggered but before it was awaited.

_IR_API_URL="${API_URL:-http://localhost:8321}"
_IR_ACTIVE=0
_IR_PHASES_FILE=""
_IR_PHASE_NAME=""
_IR_PHASE_START=0
_IR_PHASE_FATAL=0
_IR_COLLECTION=""
_IR_JOB=""
_IR_TRIGGER="scheduled"
_IR_VARIANT="incremental"
_IR_STARTED_AT=""
RUN_ID="${RUN_ID:-}"

_ir_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# All JSON is assembled by python3 reading the environment, never by string
# concatenation here — error strings and phase details are arbitrary text.
_ir_python() {
    if command -v python3 >/dev/null 2>&1; then
        python3 "$@"
        return $?
    fi
    return 1
}

# run_begin <collection> <job> <trigger> [variant]
#
# Exports RUN_ID and appends an OPENING partial record. The opening record is
# what distinguishes "the script is still running" from "the script died": a
# run group holding a begin with no matching end folds to status `incomplete`
# rather than to whatever partial happens to have survived. Without it, a script
# killed after the reindex but before run_end would fold to a clean 15-minute
# success — the exact under-reporting this ledger exists to prevent.
run_begin() {
    _IR_COLLECTION="${1:-}"
    _IR_JOB="${2:-}"
    _IR_TRIGGER="${3:-scheduled}"
    _IR_VARIANT="${4:-incremental}"
    [ -n "$_IR_COLLECTION" ] || return 0

    _IR_STARTED_AT="$(_ir_iso)" || return 0
    RUN_ID="${_IR_COLLECTION}-${_IR_STARTED_AT}"
    export RUN_ID
    _IR_PHASES_FILE="$(mktemp -t indexing_run.XXXXXX 2>/dev/null)" || _IR_PHASES_FILE=""
    _IR_ACTIVE=1

    _ir_emit "begin" "" "" || true
    return 0
}

# run_variant <incremental|rebuild>
#
# Reclassify a run already in flight. x-feed only learns which kind of run it is
# doing partway through: cleanup runs first, and only if it actually deleted
# something does the script drop the collection and do a full rebuild instead of
# an incremental update. The two differ by an order of magnitude, so a single
# median over both is meaningless — the dashboard medians them separately, which
# it can only do if the record says which one this was.
run_variant() {
    [ "$_IR_ACTIVE" = "1" ] || return 0
    case "${1:-}" in
        incremental|rebuild) _IR_VARIANT="$1" ;;
    esac
    return 0
}

# phase_begin <name> [fatal]
#
# `fatal` (1) means a failure of this phase fails the whole run; the default (0)
# only degrades it. The six scripts genuinely mix the two today, so each
# converted step must be classified explicitly rather than wrapped mechanically.
phase_begin() {
    [ "$_IR_ACTIVE" = "1" ] || return 0
    _IR_PHASE_NAME="${1:-phase}"
    _IR_PHASE_FATAL="${2:-0}"
    _IR_PHASE_START="$(date +%s)" || _IR_PHASE_START=0
    # Reset the call site's rc here as well as at the call site. Without the
    # reset, `rc` is assigned only on failure and never cleared — one failed
    # tagging phase would then mark every later phase (reindex included) failed,
    # rolling a degraded run up to failed.
    rc=0
    return 0
}

# phase_end <exit-code> [json-detail]
#
# Prescribed call shape, because under `set -e` an unguarded failing step aborts
# before phase_end is ever reached:
#
#     phase_begin tag; rc=0; tag_mimir || rc=$?; phase_end "$rc" || true
phase_end() {
    [ "$_IR_ACTIVE" = "1" ] || return 0
    [ -n "$_IR_PHASE_NAME" ] || return 0
    local rc_in="${1:-0}" detail="${2:-}" now duration status
    now="$(date +%s)" || now="$_IR_PHASE_START"
    duration=$((now - _IR_PHASE_START))
    [ "$duration" -ge 0 ] || duration=0
    if [ "$rc_in" = "0" ]; then
        status="succeeded"
    elif [ "$_IR_PHASE_FATAL" = "1" ]; then
        status="failed"
    else
        status="degraded"
    fi

    IR_PHASE_NAME="$_IR_PHASE_NAME" \
    IR_PHASE_STATUS="$status" \
    IR_PHASE_FATAL="$_IR_PHASE_FATAL" \
    IR_PHASE_DURATION="$duration" \
    IR_PHASE_DETAIL="$detail" \
    _ir_python -c '
import json, os, sys
phase = {
    "name": os.environ["IR_PHASE_NAME"],
    "status": os.environ["IR_PHASE_STATUS"],
    "durationSeconds": int(os.environ["IR_PHASE_DURATION"]),
}
if os.environ.get("IR_PHASE_FATAL") == "1":
    phase["fatal"] = True
raw = os.environ.get("IR_PHASE_DETAIL") or ""
if raw.strip():
    try:
        phase["detail"] = json.loads(raw)
    except ValueError:
        phase["detail"] = {"note": raw[:512]}
sys.stdout.write(json.dumps(phase, ensure_ascii=False) + "\n")
' >> "$_IR_PHASES_FILE" 2>/dev/null || true

    _IR_PHASE_NAME=""
    return 0
}

# run_end [status] [error]
#
# `status` may be empty, in which case the ledger rolls it up from the phases.
# Posts the closing record to the API; on any non-200 or an unreachable API the
# SAME json goes to the ledger module over stdin. Never `>>` the ledger file
# directly: macOS ships no flock(1), so a shell redirect cannot take the LOCK_EX
# every other writer holds, and it would be the one unlocked writer on exactly
# the path (API down) where writers converge.
run_end() {
    [ "$_IR_ACTIVE" = "1" ] || return 0
    _ir_emit "end" "${1:-}" "${2:-}" || true
    if [ -n "$_IR_PHASES_FILE" ]; then rm -f "$_IR_PHASES_FILE" || true; fi
    _IR_ACTIVE=0
    return 0
}

# Build one record and ship it. stage=begin writes the opening partial (no
# phases, no finishedAt); stage=end writes the closing one.
_ir_emit() {
    local stage="$1" status="$2" error="$3" payload http_code

    payload="$(
        IR_STAGE="$stage" \
        IR_RUN_ID="$RUN_ID" \
        IR_COLLECTION="$_IR_COLLECTION" \
        IR_JOB="$_IR_JOB" \
        IR_TRIGGER="$_IR_TRIGGER" \
        IR_VARIANT="$_IR_VARIANT" \
        IR_STARTED_AT="$_IR_STARTED_AT" \
        IR_FINISHED_AT="$(_ir_iso)" \
        IR_STATUS="$status" \
        IR_ERROR="$error" \
        IR_PHASES_FILE="$_IR_PHASES_FILE" \
        _ir_python -c '
import json, os, sys
stage = os.environ["IR_STAGE"]
record = {
    "runId": os.environ["IR_RUN_ID"],
    "collection": os.environ["IR_COLLECTION"],
    "job": os.environ.get("IR_JOB") or None,
    "trigger": os.environ.get("IR_TRIGGER") or "scheduled",
    "variant": os.environ.get("IR_VARIANT") or "incremental",
    "startedAt": os.environ["IR_STARTED_AT"],
    "source": "script",
    "stage": stage,
}
if stage == "end":
    record["finishedAt"] = os.environ["IR_FINISHED_AT"]
    record["error"] = os.environ.get("IR_ERROR") or None
    status = os.environ.get("IR_STATUS") or ""
    if status:
        record["status"] = status
    phases = []
    path = os.environ.get("IR_PHASES_FILE") or ""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    phases.append(json.loads(line))
                except ValueError:
                    continue
    record["phases"] = phases
sys.stdout.write(json.dumps(record, ensure_ascii=False))
'
    )" || return 0
    [ -n "$payload" ] || return 0

    http_code="$(
        printf '%s' "$payload" | curl -s -o /dev/null -w '%{http_code}' \
            -X POST -H 'Content-Type: application/json' --data-binary @- \
            --max-time 15 "${_IR_API_URL}/api/indexing/runs" 2>/dev/null
    )" || http_code="000"

    case "$http_code" in
        200|201|204) return 0 ;;
    esac

    # API down or rejecting: same JSON, same writer, through the module.
    printf '%s' "$payload" \
        | (cd "${PROJECT_DIR:-.}" && uv run python -m main.runtime.indexing_run_ledger append --file -) \
        >/dev/null 2>&1 || true
    return 0
}

# MUST be the last line: `.` returns the status of the last command run, so a
# non-zero final command here would send a perfectly loaded helper down the
# no-op stub branch at every call site.
return 0
