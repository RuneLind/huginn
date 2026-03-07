from main.sources.notion.notion_block_to_markdown import (
    convert_blocks_to_markdown,
    extract_page_properties,
    extract_page_properties_structured,
)


def _block(block_type, rich_text=None, **extra):
    """Helper to build a Notion block dict."""
    data = {}
    if rich_text is not None:
        data["rich_text"] = rich_text
    data.update(extra)
    block = {"type": block_type, block_type: data}
    return block


def _text(content, bold=False, italic=False, code=False, strikethrough=False, href=None):
    """Helper to build a Notion rich_text element."""
    rt = {
        "plain_text": content,
        "annotations": {
            "bold": bold,
            "italic": italic,
            "code": code,
            "strikethrough": strikethrough,
        },
    }
    if href:
        rt["href"] = href
    return rt


class TestParagraph:
    def test_plain_text(self):
        blocks = [_block("paragraph", [_text("Hello world")])]
        assert convert_blocks_to_markdown(blocks) == "Hello world"

    def test_empty_paragraph(self):
        blocks = [_block("paragraph", [])]
        assert convert_blocks_to_markdown(blocks) == ""

    def test_multiple_paragraphs(self):
        blocks = [
            _block("paragraph", [_text("First")]),
            _block("paragraph", [_text("Second")]),
        ]
        assert convert_blocks_to_markdown(blocks) == "First\nSecond"


class TestHeadings:
    def test_heading_1(self):
        blocks = [_block("heading_1", [_text("Title")])]
        assert convert_blocks_to_markdown(blocks) == "# Title"

    def test_heading_2(self):
        blocks = [_block("heading_2", [_text("Subtitle")])]
        assert convert_blocks_to_markdown(blocks) == "## Subtitle"

    def test_heading_3(self):
        blocks = [_block("heading_3", [_text("Section")])]
        assert convert_blocks_to_markdown(blocks) == "### Section"


class TestRichTextAnnotations:
    def test_bold(self):
        blocks = [_block("paragraph", [_text("bold", bold=True)])]
        assert convert_blocks_to_markdown(blocks) == "**bold**"

    def test_italic(self):
        blocks = [_block("paragraph", [_text("italic", italic=True)])]
        assert convert_blocks_to_markdown(blocks) == "*italic*"

    def test_code(self):
        blocks = [_block("paragraph", [_text("code", code=True)])]
        assert convert_blocks_to_markdown(blocks) == "`code`"

    def test_strikethrough(self):
        blocks = [_block("paragraph", [_text("deleted", strikethrough=True)])]
        assert convert_blocks_to_markdown(blocks) == "~~deleted~~"

    def test_link(self):
        blocks = [_block("paragraph", [_text("click", href="https://example.com")])]
        assert convert_blocks_to_markdown(blocks) == "[click](https://example.com)"

    def test_multiple_segments(self):
        blocks = [_block("paragraph", [
            _text("normal "),
            _text("bold", bold=True),
            _text(" end"),
        ])]
        assert convert_blocks_to_markdown(blocks) == "normal **bold** end"


class TestLists:
    def test_bulleted_list(self):
        blocks = [
            _block("bulleted_list_item", [_text("Item 1")]),
            _block("bulleted_list_item", [_text("Item 2")]),
        ]
        assert convert_blocks_to_markdown(blocks) == "- Item 1\n- Item 2"

    def test_numbered_list(self):
        blocks = [
            _block("numbered_list_item", [_text("First")]),
            _block("numbered_list_item", [_text("Second")]),
            _block("numbered_list_item", [_text("Third")]),
        ]
        assert convert_blocks_to_markdown(blocks) == "1. First\n2. Second\n3. Third"

    def test_numbered_list_resets_after_other_block(self):
        blocks = [
            _block("numbered_list_item", [_text("One")]),
            _block("paragraph", [_text("Break")]),
            _block("numbered_list_item", [_text("One again")]),
        ]
        result = convert_blocks_to_markdown(blocks)
        assert result == "1. One\nBreak\n1. One again"

    def test_nested_bulleted_list(self):
        child = _block("bulleted_list_item", [_text("Nested")])
        parent = _block("bulleted_list_item", [_text("Parent")])
        parent["children"] = [child]
        result = convert_blocks_to_markdown([parent])
        assert "- Parent" in result
        assert "  - Nested" in result

    def test_todo(self):
        blocks = [
            _block("to_do", [_text("Done")], checked=True),
            _block("to_do", [_text("Not done")], checked=False),
        ]
        result = convert_blocks_to_markdown(blocks)
        assert "- [x] Done" in result
        assert "- [ ] Not done" in result


class TestCodeBlock:
    def test_code_with_language(self):
        blocks = [_block("code", [_text("print('hi')")], language="python")]
        result = convert_blocks_to_markdown(blocks)
        assert result == "```python\nprint('hi')\n```"

    def test_code_without_language(self):
        blocks = [_block("code", [_text("foo")], language="")]
        result = convert_blocks_to_markdown(blocks)
        assert result == "```\nfoo\n```"


class TestQuoteAndCallout:
    def test_quote(self):
        blocks = [_block("quote", [_text("Wise words")])]
        assert convert_blocks_to_markdown(blocks) == "> Wise words"

    def test_callout_with_emoji(self):
        blocks = [_block("callout", [_text("Note")], icon={"type": "emoji", "emoji": "💡"})]
        assert "💡" in convert_blocks_to_markdown(blocks)
        assert "Note" in convert_blocks_to_markdown(blocks)

    def test_callout_with_null_icon(self):
        block = _block("callout", [_text("No icon")])
        block["callout"]["icon"] = None
        result = convert_blocks_to_markdown([block])
        assert "> No icon" == result


class TestDivider:
    def test_divider(self):
        blocks = [_block("divider")]
        assert convert_blocks_to_markdown(blocks) == "---"


class TestMedia:
    def test_image(self):
        block = {
            "type": "image",
            "image": {
                "type": "external",
                "external": {"url": "https://example.com/img.png"},
                "caption": [],
            },
        }
        result = convert_blocks_to_markdown([block])
        assert "![image](https://example.com/img.png)" == result

    def test_image_with_caption(self):
        block = {
            "type": "image",
            "image": {
                "type": "external",
                "external": {"url": "https://example.com/img.png"},
                "caption": [_text("My caption")],
            },
        }
        result = convert_blocks_to_markdown([block])
        assert "![My caption](https://example.com/img.png)" == result

    def test_bookmark(self):
        block = _block("bookmark", url="https://example.com")
        block["bookmark"]["caption"] = []
        result = convert_blocks_to_markdown([block])
        assert "[https://example.com](https://example.com)" == result

    def test_embed(self):
        block = _block("embed", url="https://example.com/widget")
        result = convert_blocks_to_markdown([block])
        assert "https://example.com/widget" in result


class TestTable:
    def test_simple_table(self):
        rows = [
            {"type": "table_row", "table_row": {"cells": [[_text("A")], [_text("B")]]}},
            {"type": "table_row", "table_row": {"cells": [[_text("1")], [_text("2")]]}},
        ]
        block = {"type": "table", "table": {}, "children": rows}
        result = convert_blocks_to_markdown([block])
        lines = result.split("\n")
        assert "| A | B |" == lines[0]
        assert "| --- | --- |" == lines[1]
        assert "| 1 | 2 |" == lines[2]


class TestToggle:
    def test_toggle_with_children(self):
        child = _block("paragraph", [_text("Hidden content")])
        block = _block("toggle", [_text("Click me")])
        block["children"] = [child]
        result = convert_blocks_to_markdown([block])
        assert "**Click me**" in result
        assert "Hidden content" in result


class TestStructuralBlocks:
    def test_synced_block(self):
        child = _block("paragraph", [_text("Synced")])
        block = {"type": "synced_block", "synced_block": {}, "children": [child]}
        result = convert_blocks_to_markdown([block])
        assert result == "Synced"

    def test_column_list(self):
        col1 = {"type": "column", "column": {}, "children": [_block("paragraph", [_text("Col 1")])]}
        col2 = {"type": "column", "column": {}, "children": [_block("paragraph", [_text("Col 2")])]}
        block = {"type": "column_list", "column_list": {}, "children": [col1, col2]}
        result = convert_blocks_to_markdown([block])
        assert "Col 1" in result
        assert "Col 2" in result


class TestSpecialBlocks:
    def test_child_page(self):
        block = _block("child_page", title="My Page")
        result = convert_blocks_to_markdown([block])
        assert "[Child page: My Page]" == result

    def test_child_database(self):
        block = _block("child_database", title="My DB")
        result = convert_blocks_to_markdown([block])
        assert "[Child database: My DB]" == result

    def test_equation(self):
        block = _block("equation", expression="E = mc^2")
        result = convert_blocks_to_markdown([block])
        assert "$$E = mc^2$$" == result

    def test_table_of_contents_returns_none(self):
        blocks = [{"type": "table_of_contents", "table_of_contents": {}}]
        # None items are filtered out in convert_blocks_to_markdown
        result = convert_blocks_to_markdown(blocks)
        assert result == ""

    def test_breadcrumb_returns_none(self):
        blocks = [{"type": "breadcrumb", "breadcrumb": {}}]
        result = convert_blocks_to_markdown(blocks)
        assert result == ""


class TestDepthLimit:
    def test_respects_max_depth(self):
        result = convert_blocks_to_markdown(
            [_block("paragraph", [_text("deep")])],
            depth=10,
            max_depth=10,
        )
        assert result == ""

    def test_renders_within_depth(self):
        result = convert_blocks_to_markdown(
            [_block("paragraph", [_text("ok")])],
            depth=9,
            max_depth=10,
        )
        assert result == "ok"


class TestExtractPageProperties:
    def test_empty_properties(self):
        assert extract_page_properties({}) == ""
        assert extract_page_properties(None) == ""

    def test_skips_title(self):
        props = {
            "Name": {"type": "title", "title": [{"plain_text": "My Page"}]},
        }
        assert extract_page_properties(props) == ""

    def test_rich_text(self):
        props = {
            "Svar": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "This is the answer", "annotations": {}}],
            },
        }
        result = extract_page_properties(props)
        assert "**Svar:**" in result
        assert "This is the answer" in result

    def test_multi_select(self):
        props = {
            "Tags": {
                "type": "multi_select",
                "multi_select": [{"name": "Python"}, {"name": "AI"}],
            },
        }
        result = extract_page_properties(props)
        assert "Python, AI" in result

    def test_select(self):
        props = {
            "Category": {"type": "select", "select": {"name": "Engineering"}},
        }
        result = extract_page_properties(props)
        assert "Engineering" in result

    def test_status(self):
        props = {
            "Status": {"type": "status", "status": {"name": "Done"}},
        }
        assert "Done" in extract_page_properties(props)

    def test_number(self):
        props = {"Score": {"type": "number", "number": 42}}
        assert "42" in extract_page_properties(props)

    def test_checkbox(self):
        props = {"Published": {"type": "checkbox", "checkbox": True}}
        assert "Yes" in extract_page_properties(props)

        props_false = {"Published": {"type": "checkbox", "checkbox": False}}
        assert "No" in extract_page_properties(props_false)

    def test_date(self):
        props = {
            "Due": {"type": "date", "date": {"start": "2025-01-15", "end": None}},
        }
        assert "2025-01-15" in extract_page_properties(props)

    def test_date_range(self):
        props = {
            "Period": {"type": "date", "date": {"start": "2025-01-01", "end": "2025-06-30"}},
        }
        result = extract_page_properties(props)
        assert "2025-01-01 - 2025-06-30" in result

    def test_url(self):
        props = {
            "Link": {"type": "url", "url": "https://example.com"},
        }
        result = extract_page_properties(props)
        assert "[https://example.com](https://example.com)" in result

    def test_email(self):
        props = {"Email": {"type": "email", "email": "test@example.com"}}
        assert "test@example.com" in extract_page_properties(props)

    def test_phone_number(self):
        props = {"Phone": {"type": "phone_number", "phone_number": "+47 12345678"}}
        assert "+47 12345678" in extract_page_properties(props)

    def test_created_by(self):
        props = {
            "Created by": {"type": "created_by", "created_by": {"name": "Alice"}},
        }
        assert "Alice" in extract_page_properties(props)

    def test_last_edited_by(self):
        props = {
            "Edited by": {"type": "last_edited_by", "last_edited_by": {"name": "Bob"}},
        }
        assert "Bob" in extract_page_properties(props)

    def test_created_time(self):
        props = {
            "Created": {"type": "created_time", "created_time": "2025-01-15T10:00:00.000Z"},
        }
        assert "2025-01-15T10:00:00.000Z" in extract_page_properties(props)

    def test_people(self):
        props = {
            "Assignees": {
                "type": "people",
                "people": [{"name": "Alice"}, {"name": "Bob"}],
            },
        }
        assert "Alice, Bob" in extract_page_properties(props)

    def test_files(self):
        props = {
            "Attachments": {
                "type": "files",
                "files": [
                    {"name": "doc.pdf", "type": "external", "external": {"url": "https://example.com/doc.pdf"}},
                ],
            },
        }
        result = extract_page_properties(props)
        assert "[doc.pdf](https://example.com/doc.pdf)" in result

    def test_relation_with_resolved_titles(self):
        props = {
            "Events": {
                "type": "relation",
                "relation": [
                    {"id": "abc-123", "title": "Meeting A"},
                    {"id": "def-456", "title": "Meeting B"},
                ],
            },
        }
        result = extract_page_properties(props)
        assert "Meeting A, Meeting B" in result

    def test_relation_without_titles_returns_empty(self):
        props = {
            "Related": {"type": "relation", "relation": [{"id": "abc-123"}]},
        }
        assert extract_page_properties(props) == ""

    def test_unique_id_with_prefix(self):
        props = {
            "ID": {"type": "unique_id", "unique_id": {"prefix": "TASK", "number": 42}},
        }
        assert "TASK-42" in extract_page_properties(props)

    def test_unique_id_without_prefix(self):
        props = {
            "ID": {"type": "unique_id", "unique_id": {"prefix": "", "number": 7}},
        }
        assert "7" in extract_page_properties(props)

    def test_formula_string(self):
        props = {
            "Full Name": {"type": "formula", "formula": {"type": "string", "string": "Alice Smith"}},
        }
        assert "Alice Smith" in extract_page_properties(props)

    def test_formula_number(self):
        props = {
            "Total": {"type": "formula", "formula": {"type": "number", "number": 100}},
        }
        assert "100" in extract_page_properties(props)

    def test_rollup_number(self):
        props = {
            "Sum": {"type": "rollup", "rollup": {"type": "number", "number": 55}},
        }
        assert "55" in extract_page_properties(props)

    def test_multiple_properties(self):
        props = {
            "Name": {"type": "title", "title": [{"plain_text": "Page"}]},
            "Tags": {"type": "multi_select", "multi_select": [{"name": "A"}]},
            "Score": {"type": "number", "number": 10},
        }
        result = extract_page_properties(props)
        assert "**Tags:**" in result
        assert "**Score:**" in result
        assert "Name" not in result  # title skipped

    def test_null_select(self):
        props = {"Category": {"type": "select", "select": None}}
        assert extract_page_properties(props) == ""

    def test_null_date(self):
        props = {"Due": {"type": "date", "date": None}}
        assert extract_page_properties(props) == ""


class TestExtractPagePropertiesStructured:
    def test_returns_dict(self):
        props = {
            "Tags": {"type": "multi_select", "multi_select": [{"name": "Python"}]},
            "Name": {"type": "title", "title": [{"plain_text": "Page"}]},
        }
        result = extract_page_properties_structured(props)
        assert result == {"Tags": "Python"}

    def test_empty_returns_empty_dict(self):
        assert extract_page_properties_structured({}) == {}
        assert extract_page_properties_structured(None) == {}
