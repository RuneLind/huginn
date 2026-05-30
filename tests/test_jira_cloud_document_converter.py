"""Tests for recursive ADF text extraction (Phase 7b / H8)."""

from main.sources.jira.jira_cloud_document_converter import JiraCloudDocumentConverter


convert = JiraCloudDocumentConverter._convert_adf_to_text


def _doc(*blocks):
    return {"type": "doc", "version": 1, "content": list(blocks)}


def _para(text):
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}


class TestAdfExtraction:
    def test_flat_paragraph_text(self):
        assert convert(_doc(_para("Hello world"))) == "Hello world"

    def test_inline_text_nodes_concatenate_within_paragraph(self):
        adf = _doc({"type": "paragraph", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]})
        assert convert(adf) == "Hello world"

    def test_bullet_list_text_is_captured(self):
        # 3 levels deep (bulletList -> listItem -> paragraph -> text): the old
        # two-level walk dropped these entirely.
        adf = _doc(
            _para("Intro"),
            {"type": "bulletList", "content": [
                {"type": "listItem", "content": [_para("First bullet")]},
                {"type": "listItem", "content": [_para("Second bullet")]},
            ]},
        )
        result = convert(adf)
        assert "Intro" in result
        assert "First bullet" in result
        assert "Second bullet" in result

    def test_table_cell_text_is_captured(self):
        adf = _doc({"type": "table", "content": [
            {"type": "tableRow", "content": [
                {"type": "tableCell", "content": [_para("Cell A")]},
                {"type": "tableCell", "content": [_para("Cell B")]},
            ]},
        ]})
        result = convert(adf)
        assert "Cell A" in result
        assert "Cell B" in result

    def test_nested_blockquote_panel(self):
        adf = _doc({"type": "panel", "content": [
            {"type": "blockquote", "content": [_para("Quoted inside panel")]},
        ]})
        assert "Quoted inside panel" in convert(adf)

    def test_blocks_do_not_run_together(self):
        result = convert(_doc(_para("Line one"), _para("Line two")))
        assert result.splitlines() == ["Line one", "Line two"]

    def test_empty_or_missing_content(self):
        assert convert(None) == ""
        assert convert({}) == ""
        assert convert({"type": "doc", "content": []}) == ""
