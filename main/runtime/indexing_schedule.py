"""launchd schedule discovery for the indexing jobs.

Reads installed ``~/Library/LaunchAgents/com.huginn.*.plist`` files and maps each
job to the collections it indexes. Every failure mode here degrades to ``None``;
the jobs endpoint must never fail because a plist is missing or malformed.

The job → collection mapping is an explicit table keyed on the script BASENAME,
not derived from the plist. Installed plists carry only ``/bin/bash <script>`` —
no collection appears anywhere in them — and the scripts themselves declare their
collections in three different shell syntaxes (``COLLECTION="mimir"``,
``COLLECTIONS=("wiki" "wiki-life")``, ``COLLECTIONS="${COLLECTIONS:-jira-issues}"``).
Parsing that is more fragile than a ten-line table. Basename-keyed because the
script directories are private sub-repos and gitignored.
"""
import glob
import logging
import os
import plistlib

logger = logging.getLogger(__name__)

LAUNCH_AGENTS_GLOB = os.path.expanduser("~/Library/LaunchAgents/com.huginn.*.plist")

# script basename -> collections it reindexes
SCRIPT_COLLECTIONS = {
    "daily_mimir_update.sh": ["mimir"],
    "daily_wiki_update.sh": ["wiki", "wiki-life"],
    "daily_nav_wiki_update.sh": ["nav-wiki"],
    "daily_capra_wiki_update.sh": ["capra-wiki"],
    "daily_melosys_kode_wiki_update.sh": ["melosys-kode-wiki"],
    "daily_anthropic_update.sh": ["anthropic-knowledge"],
}


def _schedule_from_plist(data):
    calendar = data.get("StartCalendarInterval")
    if isinstance(calendar, dict):
        return {
            "kind": "calendar",
            "hour": calendar.get("Hour"),
            "minute": calendar.get("Minute"),
            "weekday": calendar.get("Weekday"),
        }
    if isinstance(calendar, list) and calendar:
        first = calendar[0]
        return {
            "kind": "calendar",
            "hour": first.get("Hour"),
            "minute": first.get("Minute"),
            "weekday": first.get("Weekday"),
        }
    interval = data.get("StartInterval")
    if isinstance(interval, int):
        # No installed job uses this yet; supported so an interval-scheduled job
        # gets a schedule instead of a null. "Next run" for these derives from
        # lastRun.finishedAt + seconds rather than a wall-clock time.
        return {"kind": "interval", "seconds": interval}
    return None


def load_schedules(pattern=LAUNCH_AGENTS_GLOB):
    """collection name -> {job, schedule}, best effort.

    Returns an empty mapping rather than raising if the LaunchAgents directory is
    unreadable or every plist is malformed.
    """
    schedules = {}
    try:
        paths = sorted(glob.glob(pattern))
    except OSError:
        return schedules

    for path in paths:
        try:
            with open(path, "rb") as handle:
                data = plistlib.load(handle)
        except Exception:
            logger.debug("Could not read launchd plist %s", path, exc_info=True)
            continue

        arguments = data.get("ProgramArguments") or []
        script = next(
            (os.path.basename(a) for a in arguments
             if isinstance(a, str) and a.endswith(".sh")),
            None,
        )
        collections = SCRIPT_COLLECTIONS.get(script)
        if not collections:
            continue

        entry = {
            "job": data.get("Label") or os.path.basename(path),
            "schedule": _schedule_from_plist(data),
        }
        for collection in collections:
            schedules[collection] = entry
    return schedules
