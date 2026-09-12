#!/usr/bin/env bash
#
# indexing_jobs.sh — pause and resume the scheduled indexing jobs.
#
# For running on battery: the daily jobs each pull a corpus and then reindex it,
# which spins the GPU for minutes at a time. Unloading them from launchd stops
# them from firing; loading them back restores the schedule in the plists.
#
# Unloading does NOT interrupt a job already in flight (launchd only stops
# spawning new ones), so `stop` reports any live run and `stop --kill` ends it.
#
#   ./scripts/indexing_jobs.sh stop [--kill]
#   ./scripts/indexing_jobs.sh start
#   ./scripts/indexing_jobs.sh status
#
# The job set is every ~/Library/LaunchAgents/com.huginn.*.plist — no job names
# are hardcoded, so this stays correct as jobs come and go.

set -euo pipefail

AGENT_DIR="${HOME}/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

plists() {
    local found=0
    for f in "${AGENT_DIR}"/com.huginn.*.plist; do
        [ -e "$f" ] || continue
        found=1
        printf '%s\n' "$f"
    done
    [ "$found" = 1 ] || { echo "no com.huginn.*.plist in ${AGENT_DIR}" >&2; return 1; }
}

label_of() { basename "$1" .plist; }

is_loaded() { launchctl print "${DOMAIN}/$1" >/dev/null 2>&1; }

# PIDs of jobs launchd currently has running (a loaded-but-idle job has no PID).
running_pids() {
    local label pid
    for f in $(plists); do
        label=$(label_of "$f")
        pid=$(launchctl print "${DOMAIN}/${label}" 2>/dev/null | awk '/^\tpid = /{print $3}')
        [ -n "${pid:-}" ] && printf '%s %s\n' "$label" "$pid"
    done
    return 0
}

cmd_stop() {
    local kill_running="${1:-}"

    local live
    live=$(running_pids)
    if [ -n "$live" ]; then
        echo "in flight right now:"
        echo "$live" | awk '{printf "  %s  pid %s\n", $1, $2}'
        if [ "$kill_running" = "--kill" ]; then
            while read -r label _pid; do
                [ -n "$label" ] || continue
                echo "  killing ${label}"
                launchctl kill SIGTERM "${DOMAIN}/${label}" 2>/dev/null || true
            done <<< "$live"
        else
            echo "  (left alone — re-run with --kill to end them)"
        fi
    fi

    for f in $(plists); do
        local label; label=$(label_of "$f")
        if is_loaded "$label"; then
            launchctl bootout "${DOMAIN}/${label}" 2>/dev/null || true
            echo "stopped  ${label}"
        else
            echo "already stopped  ${label}"
        fi
    done
    echo
    echo "Indexing paused. Resume with: $0 start"
}

cmd_start() {
    for f in $(plists); do
        local label; label=$(label_of "$f")
        if is_loaded "$label"; then
            echo "already running  ${label}"
        else
            launchctl bootstrap "$DOMAIN" "$f"
            echo "started  ${label}"
        fi
    done
    echo
    echo "Schedules restored from the plists."
}

cmd_status() {
    local sched
    for f in $(plists); do
        local label; label=$(label_of "$f")
        sched=$(/usr/libexec/PlistBuddy -c "Print :StartCalendarInterval" "$f" 2>/dev/null \
                | awk '/Hour/{h=$3} /Minute/{m=$3} END{if (h!="") printf "%02d:%02d", h, m}')
        [ -n "$sched" ] || sched=$(/usr/libexec/PlistBuddy -c "Print :StartInterval" "$f" 2>/dev/null \
                | awk '{printf "every %ss", $1}')
        if is_loaded "$label"; then
            printf 'loaded   %-40s %s\n' "$label" "${sched:-?}"
        else
            printf 'STOPPED  %-40s %s\n' "$label" "${sched:-?}"
        fi
    done
    local live; live=$(running_pids)
    [ -n "$live" ] && { echo; echo "in flight:"; echo "$live" | awk '{printf "  %s  pid %s\n", $1, $2}'; }
    return 0
}

case "${1:-status}" in
    stop)   cmd_stop "${2:-}" ;;
    start)  cmd_start ;;
    status) cmd_status ;;
    *) echo "usage: $0 {stop [--kill]|start|status}" >&2; exit 2 ;;
esac
