"""launchd schedule discovery for the indexing jobs.

Two things here are load-bearing and were not previously covered: where the
script → collection table comes from (private routing files, not this repo), and
how an hourly calendar array is reported.
"""
import json
import os
import plistlib

from main.runtime.indexing_schedule import (
    SCRIPT_COLLECTIONS,
    _schedule_from_plist,
    load_schedules,
    load_script_collections,
)


def _write_plist(path, label, script, schedule):
    data = {
        "Label": label,
        "ProgramArguments": ["/bin/bash", f"/somewhere/{script}"],
    }
    data.update(schedule)
    with open(path, "wb") as handle:
        plistlib.dump(data, handle)


def _write_routing(path, mapping):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"scriptCollections": mapping}, handle)


class TestRoutingLivesOutsideThisRepo:
    """Five of the seven scheduled collection names were never public and one is
    customer-adjacent, which this repo's conventions ban outright. The names
    belong in each private sub-repo's schedule_routing.json — the graph_routing
    precedent — so the table compiled into the public repo stays empty."""

    def test_the_public_table_is_empty(self):
        assert SCRIPT_COLLECTIONS == {}

    def test_routing_files_supply_the_names(self, tmp_path):
        _write_routing(str(tmp_path / "huginn-x" / "scripts" / "schedule_routing.json"),
                       {"daily_x_update.sh": ["x"]})
        mapping = load_script_collections(
            globs=(str(tmp_path / "huginn-*" / "scripts" / "schedule_routing.json"),))
        assert mapping == {"daily_x_update.sh": ["x"]}

    def test_several_sub_repos_merge(self, tmp_path):
        _write_routing(str(tmp_path / "huginn-a" / "scripts" / "schedule_routing.json"),
                       {"a.sh": ["ca"]})
        _write_routing(str(tmp_path / "huginn-b" / "scripts" / "schedule_routing.json"),
                       {"b.sh": ["cb1", "cb2"]})
        mapping = load_script_collections(
            globs=(str(tmp_path / "huginn-*" / "scripts" / "schedule_routing.json"),))
        assert mapping == {"a.sh": ["ca"], "b.sh": ["cb1", "cb2"]}

    def test_a_malformed_routing_file_costs_only_itself(self, tmp_path):
        good = tmp_path / "huginn-a" / "scripts" / "schedule_routing.json"
        _write_routing(str(good), {"a.sh": ["ca"]})
        bad = tmp_path / "huginn-b" / "scripts" / "schedule_routing.json"
        os.makedirs(os.path.dirname(str(bad)), exist_ok=True)
        bad.write_text("{not json", encoding="utf-8")
        mapping = load_script_collections(
            globs=(str(tmp_path / "huginn-*" / "scripts" / "schedule_routing.json"),))
        assert mapping == {"a.sh": ["ca"]}

    def test_no_routing_file_degrades_to_no_schedules(self, tmp_path):
        """The designed degradation is `schedule: null`, never a failure."""
        _write_plist(str(tmp_path / "com.huginn.x.plist"), "com.huginn.x",
                     "daily_x_update.sh", {"StartCalendarInterval": {"Hour": 9, "Minute": 0}})
        assert load_schedules(pattern=str(tmp_path / "com.huginn.*.plist")) == {}


class TestScheduleShape:
    def test_a_single_calendar_entry(self):
        assert _schedule_from_plist({"StartCalendarInterval": {"Hour": 9, "Minute": 15}}) == {
            "kind": "calendar", "hour": 9, "minute": 15, "weekday": None}

    def test_twentyfour_entries_at_one_minute_is_hourly(self):
        """x-feed. Reporting only calendar[0] would render this as "daily at
        00:35" — understating the cadence 24-fold and putting "next run" 23
        hours out for all but one hour of the day."""
        entries = [{"Hour": h, "Minute": 35} for h in range(24)]
        assert _schedule_from_plist({"StartCalendarInterval": entries}) == {
            "kind": "hourly", "minute": 35}

    def test_twentyfour_entries_each_with_a_weekday_is_not_hourly(self):
        """24 entries at one minute each pinning a Weekday run 24 times on ONE
        day of the week, not 168 times a week. Collapsing to `hourly` would
        advertise a cadence 7x the real one, so it falls through to calendar and
        preserves the weekday like the single-dict branch does."""
        entries = [{"Hour": h, "Minute": 35, "Weekday": 1} for h in range(24)]
        schedule = _schedule_from_plist({"StartCalendarInterval": entries})
        assert schedule["kind"] == "calendar"
        assert schedule["weekday"] == 1
        assert schedule["entries"] == 24

    def test_twentyfour_weekday_free_entries_are_still_hourly(self):
        """The unpinned form is the real x-feed shape and must stay hourly."""
        entries = [{"Hour": h, "Minute": 35} for h in range(24)]
        assert _schedule_from_plist({"StartCalendarInterval": entries}) == {
            "kind": "hourly", "minute": 35}

    def test_a_partial_array_stays_a_calendar(self):
        entries = [{"Hour": 9, "Minute": 0}, {"Hour": 21, "Minute": 0}]
        schedule = _schedule_from_plist({"StartCalendarInterval": entries})
        assert schedule["kind"] == "calendar"
        assert schedule["entries"] == 2

    def test_an_interval_job_still_parses(self):
        assert _schedule_from_plist({"StartInterval": 3600}) == {
            "kind": "interval", "seconds": 3600}

    def test_an_unscheduled_plist_has_no_schedule(self):
        assert _schedule_from_plist({"RunAtLoad": True}) is None


class TestScheduleCache:
    """load_schedules is called per request and polled every 15s, sharing a
    process with search. It caches on an mtime SIGNATURE — a stat per file, far
    cheaper than open-and-parse — so it re-reads the instant a plist or routing
    file changes and never on a staleness timer."""

    def _setup(self, tmp_path):
        plist = tmp_path / "com.huginn.x.plist"
        _write_plist(str(plist), "com.huginn.x", "daily_x_update.sh",
                     {"StartCalendarInterval": {"Hour": 9, "Minute": 0}})
        routing = tmp_path / "huginn-x" / "scripts" / "schedule_routing.json"
        _write_routing(str(routing), {"daily_x_update.sh": ["x"]})
        pattern = str(tmp_path / "com.huginn.*.plist")
        routing_globs = (str(tmp_path / "huginn-*" / "scripts" / "schedule_routing.json"),)
        return pattern, routing_globs, plist, routing

    def test_a_repeat_call_with_no_change_is_a_cache_hit(self, tmp_path):
        pattern, routing_globs, _, _ = self._setup(tmp_path)
        first = load_schedules(pattern=pattern, routing_globs=routing_globs)
        second = load_schedules(pattern=pattern, routing_globs=routing_globs)
        assert first == {"x": {"job": "com.huginn.x",
                               "schedule": {"kind": "calendar", "hour": 9,
                                            "minute": 0, "weekday": None}}}
        # Same object identity ⇒ nothing was re-globbed or re-parsed.
        assert second is first

    def test_touching_a_plist_invalidates_the_cache(self, tmp_path):
        pattern, routing_globs, plist, _ = self._setup(tmp_path)
        first = load_schedules(pattern=pattern, routing_globs=routing_globs)
        # Force a distinct mtime rather than rewriting (coarse-resolution clocks
        # could land in the same second and hide the change).
        stamp = os.stat(str(plist)).st_mtime_ns + 1_000_000_000
        os.utime(str(plist), ns=(stamp, stamp))
        second = load_schedules(pattern=pattern, routing_globs=routing_globs)
        assert second is not first
        assert second == first  # content unchanged, only re-read

    def test_touching_a_routing_file_invalidates_the_cache(self, tmp_path):
        pattern, routing_globs, _, routing = self._setup(tmp_path)
        first = load_schedules(pattern=pattern, routing_globs=routing_globs)
        _write_routing(str(routing), {"daily_x_update.sh": ["x", "x-extra"]})
        second = load_schedules(pattern=pattern, routing_globs=routing_globs)
        assert second is not first
        # The second routing target now also has a schedule.
        assert set(second) == {"x", "x-extra"}

    def test_adding_a_plist_invalidates_the_cache(self, tmp_path):
        pattern, routing_globs, _, _ = self._setup(tmp_path)
        first = load_schedules(pattern=pattern, routing_globs=routing_globs)
        # New job + routing entry: the matched SET changes, so the signature does.
        _write_plist(str(tmp_path / "com.huginn.y.plist"), "com.huginn.y",
                     "daily_y_update.sh", {"StartCalendarInterval": {"Hour": 10, "Minute": 0}})
        _write_routing(str(tmp_path / "huginn-y" / "scripts" / "schedule_routing.json"),
                       {"daily_y_update.sh": ["y"]})
        second = load_schedules(pattern=pattern, routing_globs=routing_globs)
        assert second is not first
        assert set(second) == {"x", "y"}

    def test_removing_a_plist_invalidates_the_cache(self, tmp_path):
        pattern, routing_globs, plist, _ = self._setup(tmp_path)
        first = load_schedules(pattern=pattern, routing_globs=routing_globs)
        assert set(first) == {"x"}
        os.remove(str(plist))
        second = load_schedules(pattern=pattern, routing_globs=routing_globs)
        assert second is not first
        assert second == {}
