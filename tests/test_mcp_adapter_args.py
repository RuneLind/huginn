"""CLI argument parsing tests for the two stdio MCP adapters.

Covers the surface unique to each adapter: required flags, defaults, and the
new --graphPaths arg added when the adapters were wired through the shared
search pipeline. The shared search-tool behaviour is covered in
test_mcp_search_tool.py.
"""
import pytest

import collection_search_mcp_stdio_adapter as single
import multi_collection_search_mcp_adapter as multi


class TestSingleCollectionArgs:

    def test_collection_is_required(self):
        with pytest.raises(SystemExit):
            single._parse_args([])

    def test_defaults(self):
        args = single._parse_args(["--collection", "wiki"])
        assert args["collection"] == "wiki"
        assert args["maxNumberOfDocuments"] == 10
        assert args["maxNumberOfChunks"] == 100
        assert args["includeFullText"] is False
        assert args["graphPaths"] is None

    def test_graph_paths_accepts_multiple(self):
        args = single._parse_args([
            "--collection", "wiki",
            "--graphPaths", "/a/g.json", "/b/g.json",
        ])
        assert args["graphPaths"] == ["/a/g.json", "/b/g.json"]

    def test_include_full_text_flag(self):
        args = single._parse_args(["--collection", "wiki", "--includeFullText"])
        assert args["includeFullText"] is True


class TestMultiCollectionArgs:

    def test_collections_is_required(self):
        with pytest.raises(SystemExit):
            multi._parse_args([])

    def test_defaults(self):
        args = multi._parse_args(["--collections", "wiki", "jira"])
        assert args["collections"] == ["wiki", "jira"]
        assert args["maxNumberOfDocuments"] == 10
        assert args["maxNumberOfChunks"] == 100
        assert args["includeFullText"] is False
        assert args["graphPaths"] is None

    def test_graph_paths_accepts_multiple(self):
        args = multi._parse_args([
            "--collections", "wiki",
            "--graphPaths", "/a/g.json", "/b/g.json",
        ])
        assert args["graphPaths"] == ["/a/g.json", "/b/g.json"]
