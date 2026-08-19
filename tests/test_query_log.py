"""Tests for main.core.query_log — the persistent per-query JSONL log.

Unit tests hit ``log_search_request`` directly; the integration test goes
through ``run_search_request`` to pin the seam (one record per request, for
every transport that shares the pipeline).
"""
import json
import os
import threading

import pytest

from main.core.query_log import (
    DEFAULT_QUERY_LOG_PATH,
    _resolve_max_bytes,
    _resolve_path,
    log_search_request,
)
from main.core.search_pipeline import run_search_request
from main.core.search_trace import create_trace

from tests.test_search_pipeline import _FakeSearcher, _RecordingAugmenter, _raw_response, _search_kwargs


def _response(**overrides):
    response = {
        "results": [
            {"id": "wiki/muninn/summaries.md", "relevance": 0.87, "url": "https://x"},
            {"id": "doc-2", "relevance": 0.41},
        ],
    }
    response.update(overrides)
    return response


class TestLogSearchRequest:

    def test_appends_one_jsonl_record(self, tmp_path, monkeypatch):
        log_file = tmp_path / "query-log.jsonl"
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(log_file))

        log_search_request(collections=["mimir"], query="how does tracing work", response=_response())

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["collections"] == ["mimir"]
        assert record["query"] == "how does tracing work"
        assert record["resultCount"] == 2
        assert record["bestScore"] == 0.87
        assert record["topDoc"] == "wiki/muninn/summaries.md"
        assert record["lowConfidence"] is False
        assert record["rescued"] is False
        assert "ts" in record

    def test_appends_not_truncates(self, tmp_path, monkeypatch):
        log_file = tmp_path / "query-log.jsonl"
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(log_file))

        log_search_request(collections=["a"], query="q1", response=_response())
        log_search_request(collections=["b"], query="q2", response=_response())

        assert len(log_file.read_text(encoding="utf-8").splitlines()) == 2

    def test_flags_low_confidence_and_rescue(self, tmp_path, monkeypatch):
        log_file = tmp_path / "query-log.jsonl"
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(log_file))

        log_search_request(
            collections=["mimir"],
            query="q",
            response=_response(lowConfidence=True, corrective={"rescued": True}),
        )

        record = json.loads(log_file.read_text(encoding="utf-8"))
        assert record["lowConfidence"] is True
        assert record["rescued"] is True

    def test_empty_results_logged_without_top_doc(self, tmp_path, monkeypatch):
        log_file = tmp_path / "query-log.jsonl"
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(log_file))

        log_search_request(collections=["mimir"], query="no hits", response={"results": []})

        record = json.loads(log_file.read_text(encoding="utf-8"))
        assert record["resultCount"] == 0
        assert record["bestScore"] is None
        assert record["topDoc"] is None

    def test_long_query_truncated(self, tmp_path, monkeypatch):
        log_file = tmp_path / "query-log.jsonl"
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(log_file))

        log_search_request(collections=["c"], query="x" * 1000, response=_response())

        record = json.loads(log_file.read_text(encoding="utf-8"))
        assert len(record["query"]) == 300

    def test_disabled_writes_nothing(self, tmp_path, monkeypatch):
        # "off" comes from the autouse fixture; assert nothing appears anywhere obvious
        monkeypatch.setenv("HUGINN_QUERY_LOG", "off")
        log_search_request(collections=["c"], query="q", response=_response())
        assert _resolve_path() is None
        assert not (tmp_path / "query-log.jsonl").exists()

    def test_default_path_is_repo_logs_dir(self, monkeypatch):
        monkeypatch.delenv("HUGINN_QUERY_LOG", raising=False)
        assert _resolve_path() == DEFAULT_QUERY_LOG_PATH
        assert DEFAULT_QUERY_LOG_PATH.parts[-2:] == ("logs", "query-log.jsonl")

    def test_never_raises_on_unwritable_path(self, tmp_path, monkeypatch):
        # Point the log at a directory: open() fails, but the search must not.
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(tmp_path))
        log_search_request(collections=["c"], query="q", response=_response())


class TestRotation:

    def _log(self, tmp_path, monkeypatch, max_bytes=None):
        log_file = tmp_path / "query-log.jsonl"
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(log_file))
        if max_bytes is not None:
            monkeypatch.setenv("HUGINN_QUERY_LOG_MAX_BYTES", max_bytes)
        return log_file

    def test_below_threshold_does_not_rotate(self, tmp_path, monkeypatch):
        log_file = self._log(tmp_path, monkeypatch, max_bytes="100000")

        log_search_request(collections=["a"], query="q1", response=_response())
        log_search_request(collections=["b"], query="q2", response=_response())

        assert len(log_file.read_text(encoding="utf-8").splitlines()) == 2
        assert not (tmp_path / "query-log.jsonl.1").exists()

    def test_over_threshold_rotates_to_generation_one(self, tmp_path, monkeypatch):
        log_file = self._log(tmp_path, monkeypatch, max_bytes="200")
        log_file.write_text("old\n" * 100, encoding="utf-8")

        log_search_request(collections=["a"], query="after rotation", response=_response())

        assert (tmp_path / "query-log.jsonl.1").read_text(encoding="utf-8") == "old\n" * 100
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["query"] == "after rotation"

    def test_second_rotation_overwrites_generation_one(self, tmp_path, monkeypatch):
        log_file = self._log(tmp_path, monkeypatch, max_bytes="200")
        log_file.write_text("gen-a\n" * 100, encoding="utf-8")

        log_search_request(collections=["a"], query="q1", response=_response())
        log_file.write_text("gen-b\n" * 100, encoding="utf-8")
        log_search_request(collections=["a"], query="q2", response=_response())

        assert (tmp_path / "query-log.jsonl.1").read_text(encoding="utf-8") == "gen-b\n" * 100
        assert not (tmp_path / "query-log.jsonl.2").exists()
        assert not (tmp_path / "query-log.jsonl.1.1").exists()

    def test_zero_disables_rotation(self, tmp_path, monkeypatch):
        log_file = self._log(tmp_path, monkeypatch, max_bytes="0")
        log_file.write_text("old\n" * 100, encoding="utf-8")

        log_search_request(collections=["a"], query="q", response=_response())

        assert not (tmp_path / "query-log.jsonl.1").exists()
        assert log_file.read_text(encoding="utf-8").startswith("old\n")
        assert len(log_file.read_text(encoding="utf-8").splitlines()) == 101

    def test_invalid_value_falls_back_to_default(self, tmp_path, monkeypatch):
        log_file = self._log(tmp_path, monkeypatch, max_bytes="not-a-number")
        log_file.write_text("old\n" * 100, encoding="utf-8")

        log_search_request(collections=["a"], query="q", response=_response())

        # default is 10 MiB, so a 600-byte file stays put — and nothing raised
        assert not (tmp_path / "query-log.jsonl.1").exists()
        assert len(log_file.read_text(encoding="utf-8").splitlines()) == 101


class TestRunSearchRequestLogging:

    def test_one_record_per_request(self, tmp_path, monkeypatch):
        log_file = tmp_path / "query-log.jsonl"
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(log_file))

        run_search_request(
            {"mimir": _FakeSearcher(_raw_response(doc_id="a", url="https://example.com/a"))},
            raw_query="the raw query",
            search_query="the expanded query",
            augmenter=_RecordingAugmenter(),
            detected_entities=[],
            graph_answer=None,
            trace=create_trace(False),
            search_kwargs=_search_kwargs(),
            shape_kwargs={"limit": 10},
            min_relevance=None,
            corrective_mode="off",
        )

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["collections"] == ["mimir"]
        # the log records the user's query, not the graph-expanded one
        assert record["query"] == "the raw query"
        assert record["resultCount"] == 1


class TestResolveMaxBytes:
    """Direct tests of the threshold parser.

    The rotation tests above exercise it only through observable rotation, which
    leaves the negative space green under mutation (``ValueError`` → 0 and
    negatives → 0 both keep them passing while silently disabling the bound).
    """

    _DEFAULT = 10 * 1024 * 1024

    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, _DEFAULT),
            ("", _DEFAULT),
            ("   ", _DEFAULT),
            ("abc", _DEFAULT),
            ("-5", _DEFAULT),
            ("0", 0),
            ("  500  ", 500),
            ("1", 1),
        ],
    )
    def test_parses_env_value(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv("HUGINN_QUERY_LOG_MAX_BYTES", raising=False)
        else:
            monkeypatch.setenv("HUGINN_QUERY_LOG_MAX_BYTES", value)

        assert _resolve_max_bytes() == expected


class TestConcurrentRotation:

    def test_racing_writer_does_not_overwrite_the_retained_generation(self, tmp_path, monkeypatch):
        """Three transports write this log; an unsynchronized stat→replace loses history.

        Deterministic interleave: writer-1 is held between its size reading and
        its rename — the race window — while writer-2 runs a full rotation. Under
        a shared lock writer-2 blocks instead, re-stats, and finds nothing to
        rotate.
        """
        log_file = tmp_path / "query-log.jsonl"
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(log_file))
        monkeypatch.setenv("HUGINN_QUERY_LOG_MAX_BYTES", "500")
        history = "history\n" * 100  # 800 bytes: over the threshold
        log_file.write_text(history, encoding="utf-8")

        in_race_window = threading.Event()
        second_writer_done = threading.Event()
        real_replace = os.replace
        held = []

        def replace_hook(src, dst, *args, **kwargs):
            if threading.current_thread().name == "writer-1" and not held:
                held.append(True)
                in_race_window.set()
                second_writer_done.wait(timeout=0.5)
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", replace_hook)

        def write(query):
            log_search_request(collections=["a"], query=query, response=_response())

        first = threading.Thread(target=write, args=("first",), name="writer-1")
        first.start()
        assert in_race_window.wait(timeout=10), "writer-1 never reached the rotation"

        def second():
            write("second")
            second_writer_done.set()

        runner = threading.Thread(target=second, name="writer-2")
        runner.start()
        runner.join(timeout=10)
        first.join(timeout=10)
        assert not first.is_alive() and not runner.is_alive()

        assert (tmp_path / "query-log.jsonl.1").read_text(encoding="utf-8") == history
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert sorted(json.loads(line)["query"] for line in lines) == ["first", "second"]

    def test_lock_file_sits_beside_the_log(self, tmp_path, monkeypatch):
        log_file = tmp_path / "query-log.jsonl"
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(log_file))

        log_search_request(collections=["a"], query="q", response=_response())

        assert (tmp_path / "query-log.jsonl.lock").exists()
