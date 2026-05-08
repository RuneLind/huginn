"""Direct unit tests for the shared stdio search-tool args helper.

The two stdio MCP adapters compose this helper into their own parsers; their
end-to-end behaviour is covered in tests/test_mcp_adapter_args.py.
"""
import argparse

import pytest

from main.runtime.stdio_search_args import add_search_tool_args


def _build_parser():
    parser = argparse.ArgumentParser()
    add_search_tool_args(parser)
    return parser


class TestSharedSearchToolArgs:

    def test_defaults(self):
        args = vars(_build_parser().parse_args([]))
        assert args["index"] is None
        assert args["maxNumberOfChunks"] == 100
        assert args["maxNumberOfDocuments"] == 10
        assert args["includeFullText"] is False
        assert args["graphPaths"] is None

    def test_max_chunks_and_documents_are_int(self):
        args = vars(_build_parser().parse_args([
            "--maxNumberOfChunks", "42",
            "--maxNumberOfDocuments", "7",
        ]))
        assert args["maxNumberOfChunks"] == 42
        assert args["maxNumberOfDocuments"] == 7

    def test_include_full_text_flag(self):
        args = vars(_build_parser().parse_args(["--includeFullText"]))
        assert args["includeFullText"] is True

    def test_graph_paths_accepts_multiple(self):
        args = vars(_build_parser().parse_args([
            "--graphPaths", "/a/g.json", "/b/g.json",
        ]))
        assert args["graphPaths"] == ["/a/g.json", "/b/g.json"]

    def test_graph_paths_accepts_empty(self):
        args = vars(_build_parser().parse_args(["--graphPaths"]))
        assert args["graphPaths"] == []

    def test_index_can_be_set(self):
        args = vars(_build_parser().parse_args(["--index", "faiss-l2"]))
        assert args["index"] == "faiss-l2"

    def test_short_form_dashes(self):
        args = vars(_build_parser().parse_args([
            "-maxNumberOfChunks", "5",
            "-includeFullText",
        ]))
        assert args["maxNumberOfChunks"] == 5
        assert args["includeFullText"] is True

    def test_unknown_flag_rejected(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--bogus"])
