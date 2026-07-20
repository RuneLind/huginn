"""Rendering contract for the MCP adapter.

The whole point of this test is to guard against *two disconnected universes*:
``main/core/search_response_formatter.py`` emits a set of per-result fields, and
the MCP adapter renders those fields into markdown for the model. Nothing
structurally links the two — the formatter can grow a field (``confidenceBand``
was one such addition) that the adapter silently never renders, and no test
would notice.

This test pins the coupling by *deriving* the field universe from the formatter
itself: it runs the real shaping code (``_shape_doc`` / ``shape_search_results``)
on a maximally-populated searcher document, then asserts that every display
field the formatter actually emits is surfaced by the adapter's rendering. The
field set is no longer a hand-maintained list — if someone adds a display field
to the formatter, this test FAILS until they either render it (add a
``COVERAGE`` entry) or consciously mark it internal (add it to
``INTERNAL_ONLY``). The ``COVERAGE`` map still carries the adapter-regression
value: dropping a rendered field makes its sentinel value vanish from the output
and the corresponding assertion fires.
"""
from unittest.mock import MagicMock, patch

import knowledge_api_mcp_adapter as adapter
from main.core import search_response_formatter as fmt
from main.graph.graph_search_augmenter import GraphSearchAugmenter
from mcp_adapter.formatting import _format_date, _format_relevance

GRAPH_CONTEXT_KEY = GraphSearchAugmenter.GRAPH_CONTEXT_KEY


# Fields the formatter attaches that are internal and deliberately NOT rendered
# (scoring scaffolding the formatter pops before the public response, or per-chunk
# ordering fields). Documented here so a reviewer adding a field makes a conscious
# choice. ``test_internal_only_is_subset_of_formatter_emitted`` guards that every
# name here is still a field the formatter actually emits — a rename can't leave a
# stale entry lingering unnoticed.
INTERNAL_ONLY = {
    "_score",       # popped by the formatter itself
    "_reranked",    # popped by the formatter itself
    "score",        # per-chunk, popped by the formatter itself
}


def _mock_get(mock, payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    mock.return_value = resp


def _max_searcher_doc():
    """A raw searcher document populating every field ``_shape_doc`` reads, so the
    shaped output exercises every display field the formatter can emit.

    This is the *input* to the formatter — feeding real formatter output to the
    adapter is what links the two universes (as opposed to a hand-written shaped
    fixture that can silently diverge from what the formatter produces)."""
    return {
        "id": "sentinel-id",
        "path": "some/dir/SENTINEL_TITLE.json",  # title_from_doc_path → SENTINEL_TITLE
        "url": "https://sentinel.example/doc",
        "modifiedTime": "2025-06-15T10:00:00Z",
        "matchedChunks": [
            {
                "content": {
                    # separate_metadata() lifts the [a > b] line into a breadcrumb.
                    "indexedData": "[SENTINEL > BREADCRUMB]\nSENTINEL_BODY_CONTENT",
                    "heading": "SENTINEL_HEADING",
                    "metadata": {"MetaKey": "SENTINEL_META_VAL"},
                },
                "score": -0.9,  # → high-confidence relevance band
            }
        ],
    }


# For each display field the formatter can emit, how to derive the substring the
# adapter's render is expected to contain. A NEW formatter display field with no
# entry here fails ``test_render_covers_display_fields_*`` at the ``in COVERAGE``
# check — the deliberate "render it or declare it internal" gate.
COVERAGE = {
    "collection": lambda r: r["collection"],
    "id": lambda r: str(r["id"]),
    "title": lambda r: r["title"],
    "url": lambda r: r["url"],
    "snippet": lambda r: r["snippet"],
    "heading": lambda r: r["heading"],
    "breadcrumb": lambda r: r["breadcrumb"],
    "modifiedTime": lambda r: _format_date(r["modifiedTime"]),
    "relevance": lambda r: _format_relevance(r["relevance"]),
    "confidenceBand": lambda r: r["confidenceBand"],
    "metadata": lambda r: next(iter(r["metadata"].values())),
    "matchedChunks": lambda r: r["matchedChunks"][0]["content"],
}


def _shaped_result(brief):
    """Real formatter output for one maximally-populated doc, plus a graph_context
    sentinel (graph_context is augmenter-provided, not a formatter field). Returns
    (public_result, emitted_display_fields)."""
    per_collection = [("sentinel-col", {"results": [_max_searcher_doc()], "reranked": True})]
    public, _ = fmt.shape_search_results(per_collection, limit=10, brief=brief)
    r = public[0]
    # Raw _shape_doc still carries the internal fields (_score/_reranked, per-chunk
    # score) that shape_search_results pops — union them in so INTERNAL_ONLY has a
    # real field set to be a subset of.
    raw = fmt._shape_doc(_max_searcher_doc(), "sentinel-col", True, brief, None, 3)
    emitted = set(r) | set(raw)
    display_fields = emitted - INTERNAL_ONLY
    r[GRAPH_CONTEXT_KEY] = ["SENTINEL_GRAPH_CTX"]  # augmenter field, checked separately
    return r, display_fields


def _assert_covers(display_fields, r, out):
    for key in sorted(display_fields):
        assert key in COVERAGE, (
            f"formatter emits display field {key!r} that this contract does not "
            f"verify the adapter renders. Render it (add a COVERAGE entry that "
            f"checks its value appears in the output) or, if it is deliberately "
            f"not surfaced, add it to INTERNAL_ONLY."
        )
        expected = COVERAGE[key](r)
        assert expected and expected in out, (
            f"adapter dropped display field {key!r} (expected {expected!r} in output)"
        )


class TestRenderContract:
    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_render_covers_display_fields_brief(self, mock_get):
        r, display_fields = _shaped_result(brief=True)
        _mock_get(mock_get, {"results": [r]})
        out = adapter.search_knowledge("q", brief=True)
        _assert_covers(display_fields, r, out)
        assert "SENTINEL_GRAPH_CTX" in out  # graph_context (augmenter, not formatter)

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_render_covers_display_fields_full(self, mock_get):
        r, display_fields = _shaped_result(brief=False)
        _mock_get(mock_get, {"results": [r]})
        out = adapter.search_knowledge("q", brief=False)
        _assert_covers(display_fields, r, out)
        assert "SENTINEL_GRAPH_CTX" in out  # graph_context (augmenter, not formatter)

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_top_level_display_fields(self, mock_get):
        r, _ = _shaped_result(brief=True)
        _mock_get(
            mock_get,
            {
                "results": [r],
                "graph_answer": "SENTINEL_GRAPH_ANSWER",
                "lowConfidence": True,
            },
        )
        out = adapter.search_knowledge("q", brief=True)
        assert "SENTINEL_GRAPH_ANSWER" in out   # graph_answer
        assert "Low confidence" in out          # lowConfidence

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_retry_hints_and_no_confident_results_rendered(self, mock_get):
        _mock_get(
            mock_get,
            {
                "results": [],
                "noConfidentResults": True,
                "retryHints": {
                    "relatedTerms": ["SENTINEL_RELATED"],
                    "narrowerQuery": "SENTINEL_NARROWER",
                    "broaderQuery": "SENTINEL_BROADER",
                },
            },
        )
        out = adapter.search_knowledge("q")
        assert "No confident match" in out
        assert "SENTINEL_RELATED" in out
        assert "SENTINEL_NARROWER" in out
        assert "SENTINEL_BROADER" in out

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "TRACE_DEFAULT", True)
    @patch.object(adapter, "_api_get")
    def test_trace_pointer_rendered_when_enabled(self, mock_get):
        r, _ = _shaped_result(brief=True)
        _mock_get(mock_get, {"results": [r], "traceId": "SENTINEL_TRACE_ID"})
        out = adapter.search_knowledge("q", brief=True)
        assert "huginn-trace-url" in out
        assert "SENTINEL_TRACE_ID" in out

    def test_internal_only_is_subset_of_formatter_emitted(self):
        # The internal set is the escape hatch: fields here are consciously not
        # surfaced. This asserts every name is a field the formatter *actually*
        # emits — so a renamed/removed internal field can't linger in the list
        # unnoticed (the old tautology `== {...}` could never catch that).
        raw = fmt._shape_doc(_max_searcher_doc(), "sentinel-col", True, False, None, 3)
        emitted = set(raw)
        for chunk in raw.get("matchedChunks", []):
            emitted |= set(chunk)
        missing = INTERNAL_ONLY - emitted
        assert not missing, (
            f"INTERNAL_ONLY lists fields the formatter no longer emits: {missing} "
            f"— a rename/removal left a stale entry; drop it or fix the name."
        )
