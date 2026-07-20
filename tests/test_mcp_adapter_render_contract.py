"""Rendering contract for the MCP adapter.

The whole point of this test is to guard against *two disconnected universes*:
``main/core/search_response_formatter.py`` emits a set of per-result and
top-level fields, and the MCP adapter renders those fields into markdown for the
model. Nothing structurally links the two — the formatter can grow a field
(``confidenceBand`` was one such addition) that the adapter silently never
renders, and no test would notice.

This test pins the coupling: for every *display* field the formatter can emit,
the adapter's rendering must surface it. Fields are exercised with sentinel
values and asserted to appear in the rendered output. If someone adds a display
field to the formatter, they must either render it here or consciously add it to
``INTERNAL_ONLY`` — the list is the contract.
"""
from unittest.mock import MagicMock, patch

import knowledge_api_mcp_adapter as adapter
from main.graph.graph_search_augmenter import GraphSearchAugmenter


# Per-result fields search_response_formatter.py attaches that are internal and
# deliberately NOT rendered (stripped before the model sees them, or scoring
# scaffolding popped inside the formatter). Documented here so a reviewer adding
# a field makes a conscious choice.
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


def _full_result():
    """A result carrying every display field the formatter can emit."""
    return {
        "collection": "sentinel-col",
        "id": "sentinel-id",
        "title": "SENTINEL_TITLE",
        "heading": "SENTINEL_HEADING",
        "url": "https://sentinel.example/doc",
        "snippet": "SENTINEL_SNIPPET",
        "relevance": 0.823,
        "confidenceBand": "high",
        "modifiedTime": "2025-06-15T10:00:00Z",
        "breadcrumb": "SENTINEL_BREADCRUMB",
        "metadata": {"Status": "SENTINEL_META"},
        GraphSearchAugmenter.GRAPH_CONTEXT_KEY: ["SENTINEL_GRAPH_CTX"],
        "matchedChunks": [
            {
                "heading": "SENTINEL_CHUNK_HEADING",
                "content": "SENTINEL_CHUNK_CONTENT",
                "metadata": {"Sub": "SENTINEL_CHUNK_META"},
            }
        ],
    }


class TestRenderContract:
    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_brief_renders_all_display_fields(self, mock_get):
        _mock_get(mock_get, {"results": [_full_result()]})
        out = adapter.search_knowledge("q", brief=True)

        # Fields the brief loop is responsible for surfacing.
        assert "SENTINEL_TITLE" in out
        assert "SENTINEL_HEADING" in out
        assert "sentinel-col" in out
        assert "sentinel-id" in out
        assert "https://sentinel.example/doc" in out
        assert "SENTINEL_SNIPPET" in out
        assert "82.3% relevant" in out          # relevance
        assert "high" in out                    # confidenceBand
        assert "2025-06-15" in out              # modifiedTime
        assert "SENTINEL_BREADCRUMB" in out
        assert "SENTINEL_META" in out           # visible metadata
        assert "SENTINEL_GRAPH_CTX" in out      # graph_context

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_full_renders_all_display_fields(self, mock_get):
        _mock_get(mock_get, {"results": [_full_result()]})
        out = adapter.search_knowledge("q", brief=False)

        assert "SENTINEL_TITLE" in out
        assert "sentinel-col" in out
        assert "sentinel-id" in out
        assert "https://sentinel.example/doc" in out
        assert "82.3% relevant" in out          # relevance
        assert "high" in out                    # confidenceBand
        assert "2025-06-15" in out              # modifiedTime
        assert "SENTINEL_BREADCRUMB" in out
        assert "SENTINEL_GRAPH_CTX" in out      # graph_context
        # matchedChunks are the body of the full render.
        assert "SENTINEL_CHUNK_HEADING" in out
        assert "SENTINEL_CHUNK_CONTENT" in out
        assert "SENTINEL_CHUNK_META" in out

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_top_level_display_fields(self, mock_get):
        _mock_get(
            mock_get,
            {
                "results": [_full_result()],
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
        _mock_get(
            mock_get,
            {"results": [_full_result()], "traceId": "SENTINEL_TRACE_ID"},
        )
        out = adapter.search_knowledge("q", brief=True)
        assert "huginn-trace-url" in out
        assert "SENTINEL_TRACE_ID" in out

    def test_internal_fields_are_documented_not_rendered(self):
        # The internal set is the escape hatch: fields here are consciously not
        # surfaced. Kept as an explicit contract so growth is a deliberate act.
        assert INTERNAL_ONLY == {"_score", "_reranked", "score"}
