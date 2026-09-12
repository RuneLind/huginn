#!/usr/bin/env bash
# Overnight baseline sweep of every collection in privacy scope.
#
# The nightly `sensitivity` phase sweeps INCREMENTALLY: it re-asks the local
# model only about documents whose text changed. That is the right default for a
# nightly, and it is useless for the question "what does the model think of this
# corpus as it stands today", which is what you need before spending a map
# version on a name. This script answers that question: `--baseline` over every
# in-scope collection, one night, nothing mutated but reports and caches.
#
# It does NOT rebuild an index, swap a collection or reload the server. A map
# bump is what forces those, and a map bump is a decision taken after reading
# what this run produces — see mimir plans/huginn-privacy-sweep-recovery.mdx.
#
#   ./scripts/overnight_privacy_sweep.sh                 # every in-scope collection
#   ./scripts/overnight_privacy_sweep.sh --collection X  # just this one (repeatable)
#   ./scripts/overnight_privacy_sweep.sh --incremental    # changed documents only
#   ./scripts/overnight_privacy_sweep.sh --limit 5        # a priced sample per collection
#
# For unattended scheduling see com.huginn.privacy-sweep.plist.example alongside
# this script.

set -euo pipefail

# --- Arguments ---------------------------------------------------------------
MODE="baseline"
LIMIT=""
COLLECTIONS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --collection)  COLLECTIONS+=("${2:-}"); shift 2 ;;
        --incremental) MODE="incremental"; shift ;;
        --baseline)    MODE="baseline"; shift ;;
        --limit)       LIMIT="${2:-}"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
if [ -n "$LIMIT" ] && { ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [ "$LIMIT" -lt 1 ]; }; then
    echo "Invalid --limit: '${LIMIT}' (expected a positive integer)" >&2
    exit 1
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
LEDGER_KEY="${LEDGER_KEY:-privacy-baseline}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"   # overridable so the contract tests do not write here
# ------------------------------------------------------------------------------

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/overnight_privacy_sweep_$(date +%Y-%m-%d_%H%M%S).log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# --- Run-ledger helpers (observational only) ----------------------------------
# Stubbed to no-ops when the helper is missing, so a missing observability file
# can never stop the sweep. See scripts/lib/indexing_run.sh for the three hazards
# this shape is written around.
HELPER="${PROJECT_DIR}/scripts/lib/indexing_run.sh"
if [ -f "${HELPER:-}" ] && . "$HELPER"; then :; else
    run_begin(){ :; }; phase_begin(){ :; }; phase_end(){ :; }; run_end(){ :; }; RUN_ID=""
fi

# --- Targets ------------------------------------------------------------------
# Discovered, never hardcoded: `load_scope()` reads main/privacy/scope.json plus
# any private huginn-*/privacy/scope.json, so a collection whose NAME is not
# public is swept without that name appearing in this repo. Smallest first, so
# the cheap verdicts land early in the night; a collection in scope with no built
# index on this machine is skipped rather than failing the run.
discover_collections() {
    ./.venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from main.privacy.alias_registry import load_scope

collections, _ = load_scope()
sized = []
for name in collections:
    manifest = Path("data/collections") / name / "manifest.json"
    if not manifest.exists():
        continue
    try:
        documents = json.loads(manifest.read_text(encoding="utf-8")).get("numberOfDocuments")
    except (OSError, ValueError):
        documents = None
    sized.append((documents if isinstance(documents, int) else 0, name))
for _, name in sorted(sized):
    print(name)
PY
}

if [ "${#COLLECTIONS[@]}" -eq 0 ]; then
    log "Discovering collections in privacy scope"
    while IFS= read -r name; do
        [ -n "$name" ] && COLLECTIONS+=("$name")
    done < <(discover_collections || true)
fi

if [ "${#COLLECTIONS[@]}" -eq 0 ]; then
    log "No in-scope collection with a built index — nothing to sweep"
    exit 0
fi

# --- Run ----------------------------------------------------------------------
log "=== Overnight privacy sweep (${MODE}) — ${#COLLECTIONS[@]} collection(s) ==="
log "Order: ${COLLECTIONS[*]}"

# `rebuild` is the ledger's variant for a baseline sweep, matching what the
# sensitivity_sweep CLI records for the same run.
VARIANT="incremental"
[ "$MODE" = "baseline" ] && VARIANT="rebuild"
run_begin "$LEDGER_KEY" "$JOB_LABEL" "$TRIGGER" "$VARIANT" || true

FAILED=0
for collection in "${COLLECTIONS[@]}"; do
    log "--- ${collection}: sweeping (${MODE})"
    # Non-fatal per collection: an unknown person, or an unreachable Ollama, must
    # degrade this run without stopping the collections after it. Exit 2 (someone
    # unmapped) is the finding the night exists to produce, not a reason to stop.
    phase_begin "sweep:${collection}" 0; rc=0
    args=(--collection "$collection" --job "$JOB_LABEL" --trigger "$TRIGGER")
    [ "$MODE" = "baseline" ] && args+=(--baseline)
    [ -n "$LIMIT" ] && args+=(--limit "$LIMIT")
    uv run scripts/audit/sensitivity_sweep.py "${args[@]}" >> "$LOG_FILE" 2>&1 || rc=$?
    # Built first, so the phase_end call — and its `|| true` — stay on one line.
    detail="$(printf '{"collection": "%s", "mode": "%s", "exit": %s}' \
        "$collection" "$MODE" "${rc:-0}")"
    phase_end "$rc" "$detail" || true
    case "$rc" in
        0) log "--- ${collection}: clean" ;;
        2) log "--- ${collection}: UNKNOWN PERSON(S) — triage the report in privacy/"; FAILED=1 ;;
        *) log "--- ${collection}: sweep exited ${rc} (no collection / no map / no Ollama)"; FAILED=1 ;;
    esac
done

find "$LOG_DIR" -name 'overnight_privacy_sweep_*.log' -mtime +30 -delete 2>/dev/null || true
run_end "" || true
log "=== Overnight privacy sweep finished (degraded=${FAILED}) ==="

# Exit 0 even when a collection reported someone: the finding lives in the report
# and in the ledger row, and a non-zero exit here would only make launchd's own
# log the third place to look. The gate that blocks a hand-off is
# package_collection.py reading the same report.
exit 0
