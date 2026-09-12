#!/usr/bin/env bash
# Overnight baseline sweep of the collections in privacy scope.
#
# The nightly `sensitivity` phase sweeps INCREMENTALLY: it re-asks the local
# model only about documents whose text changed. That is the right default for a
# nightly, and it is useless for the question that precedes a map bump — what
# does the local model think of this corpus as it stands. This script answers
# that: `--baseline` over each in-scope collection, one night, nothing mutated
# but reports, caches and ledger rows.
#
# It does NOT rebuild an index, swap a collection or reload the server. A map
# bump is what forces those, and a map bump is a decision taken after reading
# what this run produces — see mimir plans/huginn-privacy-sweep-recovery.mdx.
#
#   ./scripts/overnight_privacy_sweep.sh                 # every in-scope collection
#   ./scripts/overnight_privacy_sweep.sh --collection X  # just this one (repeatable)
#   ./scripts/overnight_privacy_sweep.sh --incremental   # changed documents only
#   ./scripts/overnight_privacy_sweep.sh --limit 5       # a priced sample per collection
#   ./scripts/overnight_privacy_sweep.sh --once          # unload the launchd job afterwards
#
# For unattended scheduling see com.huginn.privacy-sweep.plist.example alongside
# this script. `/bin/bash` is bash 3.2 on macOS and that is what launchd runs, so
# nothing here may use bash 4+ syntax; the contract tests run under /bin/bash for
# that reason.

set -euo pipefail

# --- Arguments ---------------------------------------------------------------
MODE="baseline"
LIMIT=""
ONCE=false
COLLECTIONS=()
# `shift 2` on a flag whose value is missing fails under `set -e` BEFORE the
# unknown-option arm can print anything: a console typo then looks exactly like a
# job that ran and found nothing. Each value-taking flag checks first.
require_value() {
    if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
        echo "Missing value for $1" >&2
        exit 1
    fi
}
while [ $# -gt 0 ]; do
    case "$1" in
        --collection)  require_value "$@"; COLLECTIONS+=("$2"); shift 2 ;;
        --limit)       require_value "$@"; LIMIT="$2"; shift 2 ;;
        --incremental) MODE="incremental"; shift ;;
        --baseline)    MODE="baseline"; shift ;;
        --once)        ONCE=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
if [ -n "$LIMIT" ]; then
    case "$LIMIT" in
        ''|*[!0-9]*) echo "Invalid --limit: '${LIMIT}' (expected a positive integer)" >&2; exit 1 ;;
    esac
    [ "$LIMIT" -ge 1 ] || { echo "Invalid --limit: '${LIMIT}' (expected a positive integer)" >&2; exit 1; }
fi

# --- Configuration -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"   # huginn root (server cwd)
JOB_LABEL="${JOB_LABEL:-overnight_privacy_sweep.sh}"
TRIGGER="${TRIGGER:-scheduled}"
# Its own ledger key, not `sensitivity-audit`: each sweep CLI call already writes
# a run under that key with the swept collection in its detail, and a second
# writer on the same key would double every row. This key is the JOB — one run,
# one phase per collection — the way `sensitivity-audit` is the audit.
#
# Deliberately NOT routed in a schedule_routing.json. Routing would buy a
# `nextRunAt` on the dashboard and a cadence-derived `incomplete` threshold of
# 48 h — for a job that disarms itself after one night, that is a next run that
# will never happen and a two-day window in which a killed run still reads as
# `running`. The flat 6 h threshold is the better fit: the measured range is
# 4.2-5.1 h, so a run still open at 6 h IS incomplete.
LEDGER_KEY="${LEDGER_KEY:-privacy-baseline}"
LAUNCHD_LABEL="${LAUNCHD_LABEL:-com.huginn.privacy-sweep}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"   # overridable so the contract tests do not write here
# ------------------------------------------------------------------------------

cd "$PROJECT_DIR"
# Both tolerate failure: an unwritable log directory is a reason to lose the log,
# never a reason to lose the night — and least of all a reason to exit under
# `set -e` before the ledger row that says the night happened.
mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG_FILE="${LOG_DIR}/overnight_privacy_sweep_$(date +%Y-%m-%d_%H%M%S).log"
# Every sweep redirects its own output here with `>> "$LOG_FILE"`, and a
# redirection bash cannot open fails the command before it starts — so an
# unwritable log directory does not cost the log, it costs every sweep in the
# night. Fall back to /dev/null and say so on stdout, which launchd captures.
#
# This check is a pre-flight, not a guarantee: a volume with room for one append
# passes it and then fails on a later write with ENOSPC, no permission having
# changed. That is why the `|| true` on the append below stays — it is the only
# guard against a disk that fills during a five-hour run.
if ! (: >> "$LOG_FILE") 2>/dev/null; then
    echo "Log directory ${LOG_DIR} is not writable — the run continues without a log file"
    LOG_FILE="/dev/null"
fi
log() {
    line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$line"
    printf '%s\n' "$line" >> "$LOG_FILE" 2>/dev/null || true
}

# --- Run-ledger helpers (observational only) ----------------------------------
# Stubbed to no-ops when the helper is missing, so a missing observability file
# can never stop the sweep. See scripts/lib/indexing_run.sh for the three hazards
# this shape is written around.
HELPER="${PROJECT_DIR}/scripts/lib/indexing_run.sh"
if [ -f "${HELPER:-}" ] && . "$HELPER"; then :; else
    run_begin(){ :; }; phase_begin(){ :; }; phase_end(){ :; }; run_end(){ :; }; RUN_ID=""
fi

# Phase details are built by python3 from the environment, never concatenated
# here — the same rule scripts/lib/indexing_run.sh states for its own payloads.
# A collection name carrying a quote would otherwise produce invalid JSON, which
# the helper stores as an opaque `note` string, silently dropping the structured
# fields a dashboard reads.
phase_detail() {
    IR_D_COLLECTION="${1:-}" IR_D_MODE="${2:-}" IR_D_EXIT="${3:-0}" \
    python3 -c 'import json, os; print(json.dumps({
        "collection": os.environ["IR_D_COLLECTION"],
        "mode": os.environ["IR_D_MODE"],
        "exit": int(os.environ["IR_D_EXIT"] or 0),
    }))' 2>/dev/null || printf '{}'
}

# --- Run ----------------------------------------------------------------------
# `rebuild` is the ledger's variant for a baseline sweep, matching what the
# sensitivity_sweep CLI records for the same run.
VARIANT="incremental"
[ "$MODE" = "baseline" ] && VARIANT="rebuild"
# Opened BEFORE discovery, so a night that discovers nothing still leaves a row.
# A job that sweeps nothing and records nothing is indistinguishable on the
# dashboard from a job that was never scheduled — which is exactly the state a
# missing private sub-repo, an absent .venv or a raising load_scope() produces.
run_begin "$LEDGER_KEY" "$JOB_LABEL" "$TRIGGER" "$VARIANT" || true

log "=== Overnight privacy sweep (${MODE}) ==="

# --- Targets ------------------------------------------------------------------
# Discovered, never hardcoded: `load_scope()` reads main/privacy/scope.json plus
# any private huginn-*/privacy/scope.json, so a collection whose NAME is not
# public is swept without that name appearing in this repo.
#
# A collection is swept when it is NAMED in scope, or when its reader basePath is
# in scope AND its manifest carries a privacy stamp. That second clause is what
# separates a live aliased sibling from a pre-alias backup: several stale copies
# on this machine share an in-scope basePath, carry no stamp, were built before
# aliasing existed, and would report their own people by construction. A
# collection NAMED in scope is swept whether or not it is stamped — an unstamped
# one there is the most important collection in the run, not the one to skip.
#
# Ordering is by document count ascending. That is a rough proxy for cost and not
# cost itself: per-window time varies more than window count does (measured
# 7.9 s/window on one collection against 4.5 s/window on another), so the
# cheapest verdict does not always land first.
discover_collections() {
    ./.venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from main.privacy.alias_registry import load_scope, path_in_scope

names, _ = load_scope()
swept, skipped = [], []
root = Path("data/collections")
built = set()
for directory in sorted(root.iterdir() if root.is_dir() else []):
    built.add(directory.name)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        if directory.name in names:
            skipped.append(("problem", directory.name, "named in scope but has no built index"))
        continue
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Unreadable is not "not in scope": say so rather than dropping it.
        skipped.append(("problem", directory.name, "unreadable manifest"))
        continue
    named = directory.name in names
    base = (manifest.get("reader") or {}).get("basePath")
    armed = named or bool(base and path_in_scope(base))
    if not armed:
        continue
    if not named and "privacy" not in manifest:
        skipped.append(("skip", directory.name, "no privacy stamp (pre-alias copy)"))
        continue
    documents = manifest.get("numberOfDocuments")
    swept.append((documents if isinstance(documents, int) else 0, directory.name))
for name in sorted(names - built):
    # In scope by name with no directory at all — invisible to the loop above.
    skipped.append(("problem", name, "named in scope but has no built index"))
for _, name in sorted(swept):
    print("sweep\t%s" % name)
for kind, name, reason in skipped:
    # `problem` is a skip that should NOT let the night call itself complete: a
    # collection in scope that nobody could read or nobody has built. `skip` is
    # the deliberate kind — a pre-alias copy the sweep is right to leave alone.
    print("%s\t%s\t%s" % (kind, name, reason))
PY
}

PROBLEMS=0
if [ "${#COLLECTIONS[@]}" -eq 0 ]; then
    log "Discovering collections in privacy scope"
    phase_begin discover 0; rc=0
    discovery="$(discover_collections 2>>"$LOG_FILE")" || rc=$?
    phase_end "$rc" "$(phase_detail "" "$MODE" "${rc:-0}")" || true
    [ "$rc" -eq 0 ] || log "Discovery exited ${rc} — see the log"
    while IFS="$(printf '\t')" read -r kind name reason; do
        case "$kind" in
            sweep)   COLLECTIONS+=("$name") ;;
            skip)    log "Not swept: ${name} — ${reason}" ;;
            problem) log "Not swept: ${name} — ${reason}"; PROBLEMS=$((PROBLEMS + 1)) ;;
        esac
    done <<EOF
$discovery
EOF
fi

FAILED=0
PRODUCED=0
if [ "${#COLLECTIONS[@]}" -eq 0 ]; then
    # A degraded phase rather than a silent exit: "nothing in scope" is a claim
    # about this machine, and it should have to show up somewhere.
    log "No collection to sweep — privacy scope is empty or unreadable here"
    phase_begin sweep 0
    phase_end 1 "$(phase_detail "" "$MODE" 1)" || true
    FAILED=1
else
    log "Order: ${COLLECTIONS[*]}"
    for collection in "${COLLECTIONS[@]}"; do
        log "--- ${collection}: sweeping (${MODE})"
        # Non-fatal per collection: an unknown person, or an unreachable Ollama,
        # must degrade this run without stopping the collections queued behind
        # it. Exit 2 is the finding the night exists to produce, not a reason to
        # abandon the rest of it.
        phase_begin "sweep:${collection}" 0; rc=0
        args=(--collection "$collection" --job "$JOB_LABEL" --trigger "$TRIGGER")
        [ "$MODE" = "baseline" ] && args+=(--baseline)
        [ -n "$LIMIT" ] && args+=(--limit "$LIMIT")
        uv run scripts/audit/sensitivity_sweep.py "${args[@]}" >> "$LOG_FILE" 2>&1 || rc=$?
        # Built first, so the phase_end call — and its `|| true` — stay on one line.
        detail="$(phase_detail "$collection" "$MODE" "${rc:-0}")"
        phase_end "$rc" "$detail" || true
        case "$rc" in
            0)   log "--- ${collection}: clean"; PRODUCED=$((PRODUCED + 1)) ;;
            2)   log "--- ${collection}: UNKNOWN PERSON(S) — triage the report in privacy/"
                 PRODUCED=$((PRODUCED + 1)); FAILED=1 ;;
            1)   log "--- ${collection}: exited 1 — no collection, no map, no Ollama, or a crash (see the log)"; FAILED=1 ;;
            127) log "--- ${collection}: exited 127 — uv not found on PATH"; FAILED=1 ;;
            *)   log "--- ${collection}: exited ${rc} — see the log"; FAILED=1 ;;
        esac
    done
fi

find "$LOG_DIR" -name 'overnight_privacy_sweep_*.log' -mtime +30 -delete 2>/dev/null || true
run_end "" || true
log "=== Overnight privacy sweep finished (degraded=${FAILED}) ==="

# A baseline is a night you decide to spend, not a recurring cost: at ~5 h of
# local-model time it is the most expensive thing this repo schedules. `--once`
# is what makes the installed job genuinely one-shot, so forgetting to unload by
# hand cannot turn tonight into every night.
#
# THE FILE GOES FIRST. `launchctl unload` leaves the plist in
# ~/Library/LaunchAgents, and every agent there is loaded again at the next
# login — so an unloaded one-shot re-arms itself at the next reboot, which is
# the same recurring 5-hour job by a slower route. Removing the file and then
# booting the label out by name disarms it in both senses; bootout needs no
# file, which is why the order works at all.
#
# Only after a night where EVERY discovered collection produced a verdict. One
# verdict out of three is the dangerous case, not the safe one: discovery orders
# smallest first, so the cheapest collection is exactly the one likeliest to
# finish before a model dies — disarming on it would leave the expensive two
# unswept with no schedule to retry them. A collection discovery could not read,
# or that is in scope with no built index, counts against the night too: it was
# not swept either, and its reason may be as transient as a file lock.
if [ "$ONCE" = true ] && [ "$PRODUCED" -eq "${#COLLECTIONS[@]}" ] && [ "$PRODUCED" -gt 0 ] \
     && [ "$PROBLEMS" -eq 0 ]; then
    # Both guards are about aiming the `rm` below. An empty HOME would point it
    # at /Library/LaunchAgents (a system path), and the label is interpolated
    # into a path, so a traversing one would delete somewhere else entirely.
    if [ -z "${HOME:-}" ]; then
        log "Not disarming: HOME is empty or unset, so the plist path cannot be resolved"
    elif ! case "$LAUNCHD_LABEL" in
            ""|[!A-Za-z0-9]*|*[!A-Za-z0-9._-]*) false ;;
            *) true ;;
         esac; then
        # `case`, not grep: grep matches any LINE, so a label carrying a newline
        # passed a `^…$` pattern and then went into a path.
        log "Not disarming: LAUNCHD_LABEL is not a plain launchd label"
    else
        PLIST="${HOME}/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"
        log "Disarming the one-shot schedule: ${LAUNCHD_LABEL}"
        rm -f "$PLIST" 2>/dev/null || true
        if [ -e "$PLIST" ]; then
            log "Could not remove ${PLIST} — the job will re-arm at the next login"
        fi
        # Under launchd this kills the process mid-call, so the success branch is
        # only ever reached by a hand-run. Both branches report what happened
        # rather than asserting it: the previous version logged a failure
        # unconditionally, which was a false report on every manual run.
        if launchctl bootout "gui/$(id -u)/${LAUNCHD_LABEL}" >/dev/null 2>&1; then
            log "Job ${LAUNCHD_LABEL} booted out"
        else
            log "Job ${LAUNCHD_LABEL} was not booted out (already unloaded, or launchctl refused)"
        fi
    fi
elif [ "$ONCE" = true ]; then
    log "Not disarming: ${PRODUCED} of ${#COLLECTIONS[@]} collections produced a verdict, so the schedule stays for the next attempt"
fi

# Exit 0 even when a collection reported someone: the finding lives in the report
# and in the ledger row, and a non-zero exit here would only make launchd's own
# log a third place to look. The gate that blocks a hand-off is
# package_collection.py reading the same report.
exit 0
