"""Tests for huginn-side corrective rescue (Path D).

Covers ``run_corrective_search`` + ``merge_search_results`` in
``main.core.search_response_formatter``, the trace's ``set_corrective``
extension, and the MCP adapter's ``_format_retry_hints`` footer suppression on
rescue success.

The unit under test is the rescue decision (weak signal + usable hint → call
``rerun_search_fn``, merge results, recompute signal). Tests construct
already-shaped result dicts to keep the focus on the rescue control flow rather
than the search pipeline.
"""
import json
from unittest.mock import patch

import pytest

from main.core.search_response_formatter import (
    WEAK_RESULT_RELEVANCE,
    merge_search_results,
    run_corrective_search,
)
from main.core.search_trace import SearchTrace, create_trace


class _FakeAugmenter:
    """Augmenter stub: returns a fixed retry-hints dict for any query."""

    def __init__(self, hints=None):
        self._hints = hints
        self.calls = []

    def get_retry_hints(self, q, detected_entities):
        self.calls.append((q, list(detected_entities)))
        return self._hints


def _shaped(doc_id, *, collection="wiki", relevance=0.85):
    """Build a minimal shaped result dict — the shape ``shape_search_results``
    produces after stripping internal fields."""
    return {
        "collection": collection,
        "id": doc_id,
        "title": f"Doc {doc_id}",
        "url": f"https://example.com/{doc_id}",
        "relevance": relevance,
        "confidenceBand": "high" if relevance >= 0.65 else "medium" if relevance >= 0.4 else "low",
        "matchedChunks": [{"content": f"body of {doc_id}", "relevance": relevance, "heading": None}],
    }


class TestNoRescueOnConfident:
    """Case 1: confident first search → no rescue, response shape unchanged."""

    def test_confident_results_skip_rescue_and_omit_corrective_field(self):
        confident = [_shaped("doc-1", relevance=0.92), _shaped("doc-2", relevance=0.80)]
        augmenter = _FakeAugmenter(hints=None)
        trace = SearchTrace()
        rerun_calls = []

        def rerun_search_fn(q):
            rerun_calls.append(q)
            return []

        results, response = run_corrective_search(
            confident,
            query="what is X",
            augmenter=augmenter,
            detected_entities=[],
            min_relevance=None,
            trace=trace,
            reranked=True,
            mode="auto",
            rerun_search_fn=rerun_search_fn,
            limit=10,
        )

        assert rerun_calls == []
        assert "corrective" not in response
        assert results == confident
        assert response["bestScore"] == 0.92
        assert trace.to_dict()["response"]["corrective"]["rescueFired"] is False
        assert trace.to_dict()["response"]["corrective"]["verdict"] == "confident"


class TestRescueOnWeakWithBroaderHint:
    """Case 2: weak signal + broaderQuery → rescue fires, merge widens result set."""

    def test_weak_response_with_broader_hint_fires_rescue_and_adds_corrective(self):
        weak = [_shaped("orig-1", relevance=0.30)]
        rescue = [_shaped("rescue-1", relevance=0.88), _shaped("rescue-2", relevance=0.72)]
        augmenter = _FakeAugmenter(hints={"broaderQuery": "X", "narrowerQuery": "X foo"})
        trace = SearchTrace()
        rerun_calls = []

        def rerun_search_fn(q):
            rerun_calls.append(q)
            return rescue

        results, response = run_corrective_search(
            weak,
            query="X bar baz",
            augmenter=augmenter,
            detected_entities=[],
            min_relevance=None,
            trace=trace,
            reranked=True,
            mode="auto",
            rerun_search_fn=rerun_search_fn,
            limit=10,
        )

        assert rerun_calls == ["X"], "broaderQuery preferred over narrowerQuery"
        assert response["corrective"]["rescueFired"] is True
        assert response["corrective"]["verdict"] == "rescued"
        assert response["corrective"]["queriesTried"] == ["X bar baz", "X"]
        assert response["bestScore"] >= WEAK_RESULT_RELEVANCE
        # Merged: rescue results win on relevance ordering.
        ids = [r["id"] for r in results]
        assert ids[0] == "rescue-1"
        assert "orig-1" in ids


class TestRescueFallsBackToNarrower:
    """Case 3: weak + only narrowerQuery available → rescue uses narrower."""

    def test_uses_narrower_when_broader_absent(self):
        weak = [_shaped("orig-1", relevance=0.30)]
        rescue = [_shaped("rescue-1", relevance=0.85)]
        augmenter = _FakeAugmenter(hints={"narrowerQuery": "X foo entity"})
        trace = SearchTrace()
        rerun_calls = []

        def rerun_search_fn(q):
            rerun_calls.append(q)
            return rescue

        _, response = run_corrective_search(
            weak,
            query="X foo",
            augmenter=augmenter,
            detected_entities=[],
            min_relevance=None,
            trace=trace,
            reranked=True,
            mode="auto",
            rerun_search_fn=rerun_search_fn,
            limit=10,
        )

        assert rerun_calls == ["X foo entity"]
        assert response["corrective"]["rescueFired"] is True
        assert response["corrective"]["queriesTried"][1] == "X foo entity"


class TestWeakButNoHint:
    """Case 4: weak signal but augmenter returns nothing → no rescue."""

    def test_weak_no_hint_records_verdict_and_skips_rescue(self):
        weak = [_shaped("orig-1", relevance=0.30)]
        augmenter = _FakeAugmenter(hints=None)
        trace = SearchTrace()
        rerun_calls = []

        results, response = run_corrective_search(
            weak,
            query="X",
            augmenter=augmenter,
            detected_entities=[],
            min_relevance=None,
            trace=trace,
            reranked=True,
            mode="auto",
            rerun_search_fn=lambda q: rerun_calls.append(q) or [],
            limit=10,
        )

        assert rerun_calls == []
        # No-rescue path: response shape unchanged (no corrective key).
        assert "corrective" not in response
        assert results == weak
        assert trace.to_dict()["response"]["corrective"]["verdict"] == "weak_no_hint"
        assert trace.to_dict()["response"]["corrective"]["rescueFired"] is False


class TestModeOff:
    """Case 5: ``mode="off"`` skips rescue and matches today's response shape."""

    def test_mode_off_byte_identical_to_apply_corrective_signal(self):
        from main.core.search_response_formatter import apply_corrective_signal

        weak = [_shaped("orig-1", relevance=0.30)]
        augmenter = _FakeAugmenter(hints={"broaderQuery": "X"})
        trace_a, trace_b = SearchTrace(), SearchTrace()

        baseline_results, baseline_response = apply_corrective_signal(
            list(weak),
            query="X foo",
            augmenter=augmenter,
            detected_entities=[],
            min_relevance=None,
            trace=trace_a,
            reranked=True,
        )

        # Reset augmenter call history so comparison isn't sensitive to it.
        augmenter.calls = []
        run_results, run_response = run_corrective_search(
            list(weak),
            query="X foo",
            augmenter=augmenter,
            detected_entities=[],
            min_relevance=None,
            trace=trace_b,
            reranked=True,
            mode="off",
            rerun_search_fn=lambda q: [],  # would-be rescue source, never called
            limit=10,
        )

        assert run_results == baseline_results
        assert run_response == baseline_response
        # Trace's response block is also identical except for `corrective` (absent in both on off).
        assert "corrective" not in trace_b.to_dict()["response"]


class TestModeForce:
    """Case 6: ``mode="force"`` fires rescue even on confident first search."""

    def test_force_mode_fires_rescue_when_hint_exists(self):
        confident = [_shaped("doc-1", relevance=0.92)]
        rescue = [_shaped("rescue-1", relevance=0.85)]
        augmenter = _FakeAugmenter(hints={"broaderQuery": "X"})
        trace = SearchTrace()
        rerun_calls = []

        _, response = run_corrective_search(
            confident,
            query="X foo",
            augmenter=augmenter,
            detected_entities=[],
            min_relevance=None,
            trace=trace,
            reranked=True,
            mode="force",
            rerun_search_fn=lambda q: rerun_calls.append(q) or rescue,
            limit=10,
        )

        assert rerun_calls == ["X"]
        assert response["corrective"]["rescueFired"] is True
        assert response["corrective"]["mode"] == "force"


class TestMergeDedup:
    """Case 7: ``merge_search_results`` dedupes by (collection, id), rescue wins."""

    def test_dedupe_keeps_one_per_collection_doc_pair(self):
        originals = [
            _shaped("dup", collection="wiki", relevance=0.30),
            _shaped("only-orig", collection="wiki", relevance=0.25),
        ]
        rescue = [
            _shaped("dup", collection="wiki", relevance=0.88),  # same key as original
            _shaped("only-rescue", collection="wiki", relevance=0.70),
        ]

        merged = merge_search_results(originals, rescue, limit=10)
        ids = [r["id"] for r in merged]

        assert ids.count("dup") == 1
        # Rescue's higher-relevance "dup" copy wins (it's iterated first and seen-set blocks the original).
        dup_in_merged = [r for r in merged if r["id"] == "dup"][0]
        assert dup_in_merged["relevance"] == 0.88
        assert set(ids) == {"dup", "only-orig", "only-rescue"}
        # Re-sorted by relevance desc.
        assert merged[0]["id"] == "dup"

    def test_dedupe_distinguishes_collections(self):
        originals = [_shaped("doc", collection="a", relevance=0.20)]
        rescue = [_shaped("doc", collection="b", relevance=0.85)]

        merged = merge_search_results(originals, rescue, limit=10)

        assert len(merged) == 2

    def test_limit_caps_output(self):
        originals = [_shaped(f"o{i}", relevance=0.5) for i in range(5)]
        rescue = [_shaped(f"r{i}", relevance=0.9) for i in range(5)]

        merged = merge_search_results(originals, rescue, limit=3)

        assert len(merged) == 3
        assert all(r["id"].startswith("r") for r in merged)


class TestTraceMetadata:
    """Case 8: trace's response block carries corrective metadata."""

    def test_trace_records_corrective_meta_on_rescue(self):
        weak = [_shaped("orig-1", relevance=0.30)]
        rescue = [_shaped("rescue-1", relevance=0.85)]
        augmenter = _FakeAugmenter(hints={"broaderQuery": "X"})
        trace = SearchTrace()

        run_corrective_search(
            weak,
            query="X foo",
            augmenter=augmenter,
            detected_entities=[],
            min_relevance=None,
            trace=trace,
            reranked=True,
            mode="auto",
            rerun_search_fn=lambda q: rescue,
            limit=10,
        )

        trace_resp = trace.to_dict()["response"]
        assert trace_resp["corrective"]["rescueFired"] is True
        assert trace_resp["corrective"]["verdict"] == "rescued"
        assert trace_resp["corrective"]["queriesTried"] == ["X foo", "X"]
        assert trace_resp["corrective"]["retries"] == 1
        # bestScore reflects the post-rescue state.
        assert trace_resp["bestScore"] >= WEAK_RESULT_RELEVANCE

    def test_null_trace_set_corrective_is_noop(self):
        from main.core.search_trace import NULL_TRACE

        # Should not raise and should not flip any state.
        NULL_TRACE.set_corrective({"any": "thing"})
        assert NULL_TRACE.to_dict() is None

    def test_create_trace_disabled_does_not_break_run(self):
        weak = [_shaped("orig-1", relevance=0.30)]
        augmenter = _FakeAugmenter(hints={"broaderQuery": "X"})
        trace = create_trace(False)

        results, response = run_corrective_search(
            weak,
            query="X foo",
            augmenter=augmenter,
            detected_entities=[],
            min_relevance=None,
            trace=trace,
            reranked=True,
            mode="auto",
            rerun_search_fn=lambda q: [_shaped("rescue", relevance=0.85)],
            limit=10,
        )

        assert response["corrective"]["rescueFired"] is True
        assert trace.to_dict() is None


class TestMcpFooterSuppression:
    """Case 9: MCP rendered text omits weak-match footer on rescued verdict."""

    def test_footer_suppressed_on_rescued_verdict(self):
        from knowledge_api_mcp_adapter import _format_retry_hints

        data = {
            "retryHints": {"broaderQuery": "X"},
            "corrective": {"rescueFired": True, "verdict": "rescued"},
        }

        assert _format_retry_hints(data) == ""

    def test_footer_kept_on_still_weak_verdict(self):
        from knowledge_api_mcp_adapter import _format_retry_hints

        data = {
            "retryHints": {"broaderQuery": "X"},
            "noConfidentResults": True,
            "corrective": {"rescueFired": True, "verdict": "still_weak"},
        }

        out = _format_retry_hints(data)
        assert "No confident match" in out
        assert "X" in out

    def test_footer_kept_when_rescue_did_not_fire(self):
        from knowledge_api_mcp_adapter import _format_retry_hints

        data = {
            "retryHints": {"broaderQuery": "X"},
            "corrective": {"rescueFired": False, "verdict": "weak_no_hint"},
        }

        out = _format_retry_hints(data)
        assert out  # non-empty

    def test_footer_kept_when_no_corrective_block(self):
        from knowledge_api_mcp_adapter import _format_retry_hints

        data = {
            "retryHints": {"broaderQuery": "X"},
            "noConfidentResults": True,
        }

        out = _format_retry_hints(data)
        assert "No confident match" in out


class TestEndToEndViaBuildSearchToolFn:
    """The MCP closure built by ``build_search_tool_fn`` should call
    ``searcher.search`` twice on a weak rescue (original + rescue query) and
    suppress the ``corrective`` field on confident queries."""

    def _make(self, response_by_query, *, corrective_default="auto"):
        from main.core.mcp_search_tool import build_search_tool_fn
        from main.graph.graph_search_augmenter import GraphSearchAugmenter

        class _Searcher:
            def __init__(self):
                self.calls = []

            def search(self, query, **kwargs):
                self.calls.append({"query": query, **kwargs})
                return response_by_query.get(query, response_by_query["__default__"])

        searcher = _Searcher()
        augmenter = GraphSearchAugmenter(None)
        fn = build_search_tool_fn(
            searcher,
            "wiki",
            augmenter,
            max_number_of_chunks=20,
            max_number_of_documents=10,
            include_full_text=False,
            corrective_default=corrective_default,
        )
        return fn, searcher, augmenter

    def _strong_raw(self):
        return {
            "results": [
                {
                    "id": "doc-strong",
                    "url": "https://example.com/strong",
                    "path": "wiki/Strong-page.json",
                    "matchedChunks": [
                        {"content": {"indexedData": "Highly relevant body.", "heading": None}, "score": -0.45},
                    ],
                }
            ],
            "reranked": True,
        }

    def _weak_raw(self):
        return {
            "results": [
                {
                    "id": "doc-weak",
                    "url": "https://example.com/weak",
                    "path": "wiki/Weakish-page.json",
                    "matchedChunks": [
                        {"content": {"indexedData": "Tangentially related.", "heading": None}, "score": -0.02},
                    ],
                }
            ],
            "reranked": True,
        }

    def test_confident_first_search_does_not_invoke_second_search(self):
        fn, searcher, _ = self._make({"__default__": self._strong_raw()})

        result = json.loads(fn("anything"))

        assert len(searcher.calls) == 1
        assert "corrective" not in result

    def test_weak_first_search_with_hint_triggers_second_search(self):
        # Augmenter with no graph returns None for get_retry_hints — patch it
        # to return a hint so we can drive the rescue path purely through the
        # closure.
        fn, searcher, augmenter = self._make({"__default__": self._weak_raw()})
        with patch.object(augmenter, "get_retry_hints", return_value={"broaderQuery": "broader"}):
            searcher.calls.clear()  # reset
            result = json.loads(fn("weak query"))

        assert len(searcher.calls) == 2, "rescue path should call searcher.search twice"
        assert searcher.calls[1]["query"] == "broader"
        assert searcher.calls[1]["title_boost_query"] == "broader"
        assert result["corrective"]["rescueFired"] is True

    def test_mode_off_passed_per_call_disables_rescue(self):
        fn, searcher, augmenter = self._make({"__default__": self._weak_raw()})
        with patch.object(augmenter, "get_retry_hints", return_value={"broaderQuery": "broader"}):
            result = json.loads(fn("weak query", corrective="off"))

        assert len(searcher.calls) == 1
        assert "corrective" not in result
