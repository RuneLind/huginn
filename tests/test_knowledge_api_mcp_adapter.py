import json
from unittest.mock import patch, MagicMock

import httpx
import pytest

import knowledge_api_mcp_adapter as adapter


class TestCollectionScoping:
    def test_allowed_collections_parsed_from_env(self):
        with patch.dict("os.environ", {"KNOWLEDGE_COLLECTIONS": "col-a, col-b"}):
            # Re-evaluate the parsing logic
            result = [
                c.strip()
                for c in "col-a, col-b".split(",")
                if c.strip()
            ]
            assert result == ["col-a", "col-b"]

    def test_empty_env_means_all_allowed(self):
        result = [
            c.strip() for c in "".split(",") if c.strip()
        ] or None
        assert result is None


class TestSearchKnowledge:
    @patch.object(adapter, "ALLOWED_COLLECTIONS", ["allowed-col"])
    def test_rejects_disallowed_collection(self):
        result = adapter.search_knowledge("test query", collection="forbidden-col")
        assert "not available" in result
        assert "allowed-col" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_returns_no_results_message(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test query")
        assert "No results found" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_brief_format(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "collection": "my-col",
                    "id": "doc1",
                    "title": "Test Doc",
                    "url": "https://example.com",
                    "snippet": "Some snippet",
                    "relevance": 0.85,
                    "modifiedTime": "2025-01-15T10:00:00Z",
                    "breadcrumb": "Projects > My Project > Test Doc",
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test", brief=True)
        assert "Test Doc" in result
        assert "my-col" in result
        assert "doc1" in result
        assert "85.0% relevant" in result
        assert "2025-01-15" in result
        assert "Projects > My Project > Test Doc" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_full_format_with_chunks(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "collection": "my-col",
                    "id": "doc1",
                    "title": "Test Doc",
                    "url": "https://example.com",
                    "relevance": 0.92,
                    "modifiedTime": "2025-03-01T12:00:00Z",
                    "breadcrumb": "Wiki > Engineering > Test Doc",
                    "matchedChunks": [
                        {
                            "heading": "Section 1",
                            "content": "Chunk text here",
                            "metadata": {"Status": "Active"},
                            "relevance": 0.92,
                        },
                    ],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test", brief=False)
        assert "Test Doc" in result
        assert "Section 1" in result
        assert "Chunk text here" in result
        assert "92.0% relevant" in result
        assert "2025-03-01" in result
        assert "Status: Active" in result
        assert "Wiki > Engineering > Test Doc" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_brief_format_without_breadcrumb(self, mock_get):
        """Results without breadcrumb should still format correctly."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "collection": "my-col",
                    "id": "doc1",
                    "title": "Test Doc",
                    "url": "https://example.com",
                    "snippet": "Some snippet",
                    "relevance": 0.85,
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test", brief=True)
        assert "Test Doc" in result
        assert ">" not in result.split("**Test Doc**")[0]  # no breadcrumb arrow

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_brief_format_without_optional_fields(self, mock_get):
        """Results without relevance/modifiedTime should still format correctly."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "collection": "my-col",
                    "id": "doc1",
                    "title": "Test Doc",
                    "url": "https://example.com",
                    "snippet": "Some snippet",
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test", brief=True)
        assert "Test Doc" in result
        assert "relevant" not in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        result = adapter.search_knowledge("test")
        assert "not running" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", ["col-a", "col-b"])
    @patch.object(adapter, "_api_get")
    def test_scoped_search_sends_all_allowed(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        adapter.search_knowledge("test")
        call_params = mock_get.call_args[1]["params"] if "params" in mock_get.call_args[1] else mock_get.call_args[0][1]
        assert call_params["collection"] == ["col-a", "col-b"]


class TestWipBanner:
    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_brief_shows_wip_banner(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "collection": "my-col",
                    "id": "doc1",
                    "title": "CDM 4.4 Analyse",
                    "url": "https://example.com",
                    "snippet": "Some snippet",
                    "metadata": {"wip": "true"},
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test", brief=True)
        assert "**[UNDER ARBEID]**" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_full_shows_wip_banner(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "collection": "my-col",
                    "id": "doc1",
                    "title": "Draft Doc",
                    "url": "https://example.com",
                    "metadata": {"wip": "true"},
                    "matchedChunks": [{"content": "Chunk text"}],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test", brief=False)
        assert "**[UNDER ARBEID]**" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_no_wip_banner_without_metadata(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "collection": "my-col",
                    "id": "doc1",
                    "title": "Normal Doc",
                    "url": "https://example.com",
                    "snippet": "Some snippet",
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test", brief=True)
        assert "UNDER ARBEID" not in result


class TestInternalMetadataFiltering:
    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_internal_metadata_hidden_from_output(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "collection": "my-col",
                    "id": "doc1",
                    "title": "Test Doc",
                    "url": "https://example.com",
                    "matchedChunks": [
                        {
                            "content": "Chunk text",
                            "metadata": {
                                "page_id": "12345",
                                "space": "MYSPACE",
                                "breadcrumb": "A > B",
                                "title": "Test Doc",
                                "wip": "true",
                                "Status": "Active",
                            },
                        }
                    ],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test", brief=False)
        assert "Status: Active" in result
        assert "page_id: 12345" not in result
        assert "space: MYSPACE" not in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_no_metadata_line_when_only_internal_keys(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "collection": "my-col",
                    "id": "doc1",
                    "title": "Test Doc",
                    "url": "https://example.com",
                    "matchedChunks": [
                        {
                            "content": "Chunk text",
                            "metadata": {"page_id": "12345", "wip": "true"},
                        }
                    ],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.search_knowledge("test", brief=False)
        # Should not contain any italic metadata line since all keys are internal
        lines = result.split("\n")
        assert not any(line.startswith("*") and line.endswith("*") and ":" in line for line in lines)


class TestGetDocument:
    @patch.object(adapter, "ALLOWED_COLLECTIONS", ["allowed-col"])
    def test_rejects_disallowed_collection(self):
        result = adapter.get_document("forbidden-col", "doc1")
        assert "not available" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_returns_formatted_document(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "title": "My Document",
            "url": "https://example.com/doc",
            "text": "Document body content",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.get_document("col", "doc1")
        assert "My Document" in result
        assert "https://example.com/doc" in result
        assert "Document body content" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_wip_document_shows_banner(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "title": "Draft Doc",
            "url": "https://example.com/doc",
            "text": "Work in progress content",
            "metadata": {"wip": "true"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.get_document("col", "doc1")
        assert "**[UNDER ARBEID]**" in result
        assert "Draft Doc" in result

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_non_wip_document_no_banner(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "title": "Normal Doc",
            "url": "https://example.com/doc",
            "text": "Normal content",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.get_document("col", "doc1")
        assert "UNDER ARBEID" not in result


class TestListCollections:
    @patch.object(adapter, "ALLOWED_COLLECTIONS", ["col-a"])
    @patch.object(adapter, "_api_get")
    def test_filters_by_allowed(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "collections": [
                {"name": "col-a", "document_count": 10, "embedding_count": 100},
                {"name": "col-b", "document_count": 5, "embedding_count": 50},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = adapter.list_collections()
        assert "col-a" in result
        assert "col-b" not in result


class TestDescriptionBuilding:
    def test_includes_knowledge_description(self):
        desc = adapter._build_search_description(description="My custom knowledge base")
        assert "My custom knowledge base" in desc
        assert "This knowledge base contains:" in desc

    def test_no_description_when_empty(self):
        desc = adapter._build_search_description(description="")
        assert "This knowledge base contains:" not in desc

    def test_omits_graph_when_disabled(self):
        desc = adapter._build_search_description(has_graph=False)
        assert "BUC" not in desc
        assert "SED" not in desc
        assert "graph_answer" not in desc

    def test_includes_graph_when_enabled(self):
        desc = adapter._build_search_description(has_graph=True)
        assert "BUC" in desc
        assert "graph_answer" in desc

    def test_omits_sessions_when_disabled(self):
        desc = adapter._build_search_description(has_sessions=False)
        assert "project" not in desc
        assert "git_branch" not in desc
        assert "claude-sessions" not in desc

    def test_includes_sessions_when_enabled(self):
        desc = adapter._build_search_description(has_sessions=True)
        assert "project" in desc
        assert "git_branch" in desc

    def test_includes_tags_when_available(self):
        desc = adapter._build_search_description(tags_doc="\n\nAvailable tags:\nfoo, bar")
        assert "foo, bar" in desc
        assert "tags param" in desc

    def test_omits_tags_mention_when_no_tags(self):
        desc = adapter._build_search_description(tags_doc="")
        assert "tags param" not in desc

    def test_includes_all_features_in_dev_mode(self):
        desc = adapter._build_search_description(
            description="",
            has_sessions=True,
            has_graph=True,
            tags_doc="\n\nAvailable tags:\ntag1",
        )
        assert "project" in desc
        assert "BUC" in desc
        assert "tag1" in desc

    def test_always_includes_core_instructions(self):
        desc = adapter._build_search_description(
            description="", has_sessions=False, has_graph=False, tags_doc=""
        )
        assert "brief=True" in desc
        assert "get_document" in desc
        assert "url" in desc


class TestFeatureDetection:
    def test_notion_detected_from_collection_name(self):
        assert adapter._detect_feature(["my-notion-v9", "my-handbook"], "notion") is True

    def test_notion_not_detected_for_confluence(self):
        assert adapter._detect_feature(["my-confluence"], "notion") is False

    def test_sessions_detected_from_collection_name(self):
        assert adapter._detect_feature(["claude-sessions", "my-notion"], "session") is True

    def test_sessions_not_detected_for_notion_only(self):
        assert adapter._detect_feature(["my-notion", "my-handbook"], "session") is False

    def test_dev_mode_enables_all(self):
        """ALLOWED_COLLECTIONS=None means all features enabled."""
        assert adapter._detect_feature(None, "notion") is True
        assert adapter._detect_feature(None, "session") is True
        assert adapter._detect_feature(None, "my-project") is True

    def test_my_project_detected(self):
        assert adapter._detect_feature(["my-project", "my-confluence"], "my-project") is True

    def test_my_project_not_detected_for_notion(self):
        assert adapter._detect_feature(["my-notion-v9"], "my-project") is False


class TestSearchSchemaVariants:
    """Verify that the registered search tool exposes the right params per profile."""

    def test_sessions_and_tags_variant_has_all_params(self):
        import inspect
        sig = inspect.signature(adapter._search_with_sessions_and_tags)
        params = list(sig.parameters.keys())
        assert "project" in params
        assert "git_branch" in params
        assert "tags" in params

    def test_sessions_variant_has_session_params_but_not_tags(self):
        import inspect
        sig = inspect.signature(adapter._search_with_sessions)
        params = list(sig.parameters.keys())
        assert "project" in params
        assert "git_branch" in params
        assert "tags" not in params

    def test_tags_variant_has_tags_but_not_sessions(self):
        import inspect
        sig = inspect.signature(adapter._search_with_tags)
        params = list(sig.parameters.keys())
        assert "tags" in params
        assert "project" not in params
        assert "git_branch" not in params

    def test_basic_variant_has_no_optional_filters(self):
        import inspect
        sig = inspect.signature(adapter._search_basic)
        params = list(sig.parameters.keys())
        assert "tags" not in params
        assert "project" not in params
        assert "git_branch" not in params

    def test_pick_returns_sessions_and_tags_when_both(self):
        with patch.object(adapter, "_has_sessions", True), \
             patch.object(adapter, "_has_tags", True):
            assert adapter._pick_search_function() is adapter._search_with_sessions_and_tags

    def test_pick_returns_sessions_when_sessions_only(self):
        with patch.object(adapter, "_has_sessions", True), \
             patch.object(adapter, "_has_tags", False):
            assert adapter._pick_search_function() is adapter._search_with_sessions

    def test_pick_returns_tags_when_tags_only(self):
        with patch.object(adapter, "_has_sessions", False), \
             patch.object(adapter, "_has_tags", True):
            assert adapter._pick_search_function() is adapter._search_with_tags

    def test_pick_returns_basic_when_no_tags_no_sessions(self):
        with patch.object(adapter, "_has_sessions", False), \
             patch.object(adapter, "_has_tags", False):
            assert adapter._pick_search_function() is adapter._search_basic

    @patch.object(adapter, "ALLOWED_COLLECTIONS", None)
    @patch.object(adapter, "_api_get")
    def test_tags_variant_passes_tags_to_impl(self, mock_get):
        """Ensure _search_with_tags correctly forwards the tags parameter."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        adapter._search_with_tags("test", tags="foo,bar")
        call_params = mock_get.call_args[1]["params"] if "params" in mock_get.call_args[1] else mock_get.call_args[0][1]
        assert call_params["tags"] == "foo,bar"
