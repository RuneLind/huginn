import os
import tempfile

import pytest

from main.sources.files.files_document_converter import FilesDocumentConverter


def _make_document(relative_path="folder/page.md", content_texts=None, file_content=None):
    """Build a minimal files document dict as produced by the reader."""
    if content_texts is None:
        content_texts = ["This is some real body content for testing purposes."]

    # Write a temp file so __build_url can read it
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
    tmp.write(file_content or "")
    tmp.close()

    return {
        "fileRelativePath": relative_path,
        "fileFullPath": tmp.name,
        "modifiedTime": "2026-01-01T00:00:00Z",
        "content": [{"text": t} for t in content_texts],
    }


@pytest.fixture
def converter():
    return FilesDocumentConverter()


@pytest.fixture
def make_doc():
    """Factory fixture that tracks temp files and cleans up after the test."""
    created = []

    def _factory(**kwargs):
        doc = _make_document(**kwargs)
        created.append(doc["fileFullPath"])
        return doc

    yield _factory

    for path in created:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


class TestStripFrontmatter:
    def test_removes_frontmatter(self, converter):
        text = "---\nurl: https://example.com\ntitle: Test\n---\nActual content here."
        assert converter._strip_frontmatter(text) == "Actual content here."

    def test_no_frontmatter(self, converter):
        text = "Just plain content."
        assert converter._strip_frontmatter(text) == "Just plain content."

    def test_malformed_frontmatter_no_closing(self, converter):
        text = "---\nurl: https://example.com\nActual content here."
        assert converter._strip_frontmatter(text) == text

    def test_frontmatter_only_at_start(self, converter):
        text = "Some text\n---\nkey: value\n---\nMore text"
        assert converter._strip_frontmatter(text) == text


class TestStripCodeBlocks:
    def test_strips_fenced_code_block(self, converter):
        text = "Before\n```python\ndef foo():\n    return 42\n```\nAfter"
        result = converter._clean_chunk_text(text)
        assert result == "Before\n\nAfter"
        assert "def foo" not in result

    def test_strips_code_block_without_language(self, converter):
        text = "Before\n```\nsome code\n```\nAfter"
        result = converter._clean_chunk_text(text)
        assert result == "Before\n\nAfter"

    def test_strips_multiple_code_blocks(self, converter):
        text = "A\n```js\nconst x = 1;\n```\nB\n```py\ny = 2\n```\nC"
        result = converter._clean_chunk_text(text)
        assert result == "A\n\nB\n\nC"

    def test_strips_code_block_with_special_language_hint(self, converter):
        text = "Before\n```c++\nint main() { return 0; }\n```\nAfter"
        result = converter._clean_chunk_text(text)
        assert result == "Before\n\nAfter"
        assert "int main" not in result

    def test_preserves_inline_backticks(self, converter):
        text = "Use `foo()` to call the function"
        assert converter._clean_chunk_text(text) == text

    def test_preserves_text_without_code(self, converter):
        text = "Just regular text with no code blocks"
        assert converter._clean_chunk_text(text) == text


class TestCleanChunkText:
    def test_replaces_s3_url(self, converter):
        text = "See file: https://prod-files-secure.s3.us-west-2.amazonaws.com/abc/def/image.png?X-Amz-Sig=xyz here"
        result = converter._clean_chunk_text(text)
        assert result == "See file: [file] here"

    def test_replaces_multiple_s3_urls(self, converter):
        text = "A https://bucket.s3.eu-west-1.amazonaws.com/a/b B https://other.s3.us-east-1.amazonaws.com/c/d C"
        result = converter._clean_chunk_text(text)
        assert result == "A [file] B [file] C"

    def test_preserves_non_s3_urls(self, converter):
        text = "Link: https://example.com/page"
        assert converter._clean_chunk_text(text) == text

    def test_strips_markdown_image_with_base64(self, converter):
        text = "Before ![diagram](data:image/png;base64,iVBORw0KGgoAAAAN==) after"
        result = converter._clean_chunk_text(text)
        assert result == "Before  after"
        assert "base64" not in result

    def test_strips_markdown_image_with_s3_url(self, converter):
        text = "Before ![image](https://prod-files-secure.s3.us-west-2.amazonaws.com/abc/img.png?X-Amz-Sig=xyz) after"
        result = converter._clean_chunk_text(text)
        assert result == "Before  after"
        assert "amazonaws" not in result

    def test_strips_markdown_image_with_regular_url(self, converter):
        text = "See ![logo](https://example.com/logo.png) here"
        result = converter._clean_chunk_text(text)
        assert result == "See  here"

    def test_preserves_regular_markdown_links(self, converter):
        text = "See [this page](https://example.com/page) for details"
        assert converter._clean_chunk_text(text) == text

    def test_s3_url_in_markdown_link_preserves_link_text(self, converter):
        text = "[PDF document](https://prod-files-secure.s3.us-west-2.amazonaws.com/abc/doc.pdf?X-Amz=xyz)"
        result = converter._clean_chunk_text(text)
        assert result == "[PDF document]([file])"


class TestNoEmptyChunkZero:
    def test_first_chunk_has_real_content(self, converter, make_doc):
        doc = make_doc(content_texts=["Hello world, this is real content."])
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        assert "Hello world" in chunks[0]["indexedData"]

    def test_breadcrumb_prepended_to_all_chunks(self, converter, make_doc):
        doc = make_doc(content_texts=["Some body content here."])
        results = converter.convert(doc)
        for chunk in results[0]["chunks"]:
            assert chunk["indexedData"].startswith("[")

    def test_empty_content_gets_breadcrumb_fallback(self, converter, make_doc):
        doc = make_doc(content_texts=["", "   "])
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        assert len(chunks) == 1
        assert chunks[0]["indexedData"] == "[folder > page]"


class TestBreadcrumbTruncation:
    def test_shallow_path_unchanged(self, converter, make_doc):
        doc = make_doc(relative_path="Teams/Nysalg/Rammeavtaler/NAV.md")
        results = converter.convert(doc)
        breadcrumb_part = results[0]["chunks"][0]["indexedData"].split("\n")[0]
        assert breadcrumb_part == "[Teams > Nysalg > Rammeavtaler > NAV]"

    def test_deep_path_truncated(self, converter, make_doc):
        doc = make_doc(
            relative_path="Teams/Avdeling/Gruppe/HR/Personal/Fravær/Ferie/Regler.md"
        )
        results = converter.convert(doc)
        breadcrumb_part = results[0]["chunks"][0]["indexedData"].split("\n")[0]
        assert breadcrumb_part == "[Teams > ... > Ferie > Regler]"

    def test_exactly_four_parts_unchanged(self, converter, make_doc):
        doc = make_doc(relative_path="A/B/C/D.md")
        results = converter.convert(doc)
        breadcrumb_part = results[0]["chunks"][0]["indexedData"].split("\n")[0]
        assert breadcrumb_part == "[A > B > C > D]"

    def test_five_parts_truncated(self, converter, make_doc):
        doc = make_doc(relative_path="A/B/C/D/E.md")
        results = converter.convert(doc)
        breadcrumb_part = results[0]["chunks"][0]["indexedData"].split("\n")[0]
        assert breadcrumb_part == "[A > ... > D > E]"


class TestNormalChunking:
    def test_metadata_preserved(self, converter, make_doc):
        doc = make_doc()
        doc["content"] = [{"text": "Some text content.", "metadata": {"page": 1}}]
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        assert chunks[0].get("metadata") == {"page": 1}

    def test_frontmatter_stripped_before_chunking(self, converter, make_doc):
        doc = make_doc(
            content_texts=["---\ntitle: Test\nurl: https://x.com\n---\nReal content after frontmatter."]
        )
        results = converter.convert(doc)
        all_chunk_text = " ".join(c["indexedData"] for c in results[0]["chunks"])
        assert "title: Test" not in all_chunk_text
        assert "Real content after frontmatter" in all_chunk_text

    def test_s3_urls_cleaned_in_chunks(self, converter, make_doc):
        doc = make_doc(
            content_texts=["Check this https://prod-files-secure.s3.us-west-2.amazonaws.com/abc/img.png out"]
        )
        results = converter.convert(doc)
        all_chunk_text = " ".join(c["indexedData"] for c in results[0]["chunks"])
        assert "amazonaws.com" not in all_chunk_text
        assert "[file]" in all_chunk_text

    def test_document_text_also_cleaned(self, converter, make_doc):
        doc = make_doc(
            content_texts=["---\ntitle: Test\n---\nBody https://x.s3.us-west-2.amazonaws.com/a end"]
        )
        results = converter.convert(doc)
        text = results[0]["text"]
        assert "title: Test" not in text
        assert "amazonaws.com" not in text
        assert "[file]" in text
        assert "Body" in text


class TestHeadingAwareChunking:
    def test_markdown_with_headings_produces_heading_key(self, converter, make_doc):
        doc = make_doc(content_texts=["# Overview\nThis is the overview.\n\n## Details\nHere are details."])
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        assert len(chunks) == 2
        assert chunks[0]["heading"] == "Overview"
        assert chunks[1]["heading"] == "Details"

    def test_indexed_data_contains_heading_text(self, converter, make_doc):
        doc = make_doc(content_texts=["# My Section\nSection body content."])
        results = converter.convert(doc)
        chunk = results[0]["chunks"][0]
        assert "## My Section" in chunk["indexedData"]
        assert "Section body content" in chunk["indexedData"]

    def test_breadcrumb_prepended_with_headings(self, converter, make_doc):
        doc = make_doc(
            relative_path="Teams/Engineering/Onboarding.md",
            content_texts=["# Getting Started\nRead the handbook."],
        )
        results = converter.convert(doc)
        chunk = results[0]["chunks"][0]
        assert chunk["indexedData"].startswith("[Teams > Engineering > Onboarding]")
        assert "## Getting Started" in chunk["indexedData"]

    def test_preamble_chunk_has_no_heading_key(self, converter, make_doc):
        doc = make_doc(content_texts=["Intro text.\n\n# Section\nSection body."])
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        assert "heading" not in chunks[0]
        assert "Intro text" in chunks[0]["indexedData"]
        assert chunks[1]["heading"] == "Section"

    def test_plain_text_without_headings_same_as_before(self, converter, make_doc):
        doc = make_doc(content_texts=["Just plain text without any markdown headings."])
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        assert "heading" not in chunks[0]
        assert "plain text" in chunks[0]["indexedData"]

    def test_all_heading_levels_normalized_to_h2_in_indexed_data(self, converter, make_doc):
        doc = make_doc(content_texts=["# H1 Title\nH1 body.\n\n## H2 Title\nH2 body.\n\n### H3 Title\nH3 body."])
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        assert "## H1 Title" in chunks[0]["indexedData"]
        assert "## H2 Title" in chunks[1]["indexedData"]
        assert "## H3 Title" in chunks[2]["indexedData"]

    def test_metadata_preserved_with_headings(self, converter, make_doc):
        doc = make_doc()
        doc["content"] = [{"text": "# Section\nBody text.", "metadata": {"source": "notion"}}]
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        assert chunks[0].get("metadata") == {"source": "notion"}
        assert chunks[0]["heading"] == "Section"


class TestFrontmatterMetadata:
    def test_extracts_wip_metadata(self, converter, make_doc):
        doc = make_doc(
            content_texts=["---\ntitle: My Page\nwip: true\nurl: https://x.com\n---\nBody content."]
        )
        results = converter.convert(doc)
        assert results[0].get("metadata") == {"title": "My Page", "wip": "true", "url": "https://x.com"}

    def test_extracts_all_known_fields(self, converter, make_doc):
        doc = make_doc(
            content_texts=[
                "---\ntitle: Test\nbreadcrumb: A > B > C\nspace: MYSPACE\npage_id: 12345\nwip: true\nurl: https://x.com\n---\nBody."
            ]
        )
        results = converter.convert(doc)
        meta = results[0]["metadata"]
        assert meta["title"] == "Test"
        assert meta["breadcrumb"] == "A > B > C"
        assert meta["space"] == "MYSPACE"
        assert meta["page_id"] == "12345"
        assert meta["wip"] == "true"

    def test_no_metadata_when_no_known_fields(self, converter, make_doc):
        doc = make_doc(
            content_texts=["---\nmodifiedTime: 2025-01-01\ncustomField: something\n---\nBody."]
        )
        results = converter.convert(doc)
        assert "metadata" not in results[0]

    def test_no_metadata_without_frontmatter(self, converter, make_doc):
        doc = make_doc(content_texts=["Just plain content, no frontmatter."])
        results = converter.convert(doc)
        assert "metadata" not in results[0]

    def test_metadata_propagated_to_chunks(self, converter, make_doc):
        doc = make_doc(
            content_texts=["---\ntitle: Test Page\nwip: true\n---\nSome body content."]
        )
        results = converter.convert(doc)
        for chunk in results[0]["chunks"]:
            assert chunk.get("metadata", {}).get("wip") == "true"
            assert chunk.get("metadata", {}).get("title") == "Test Page"

    def test_frontmatter_metadata_merged_with_content_metadata(self, converter, make_doc):
        doc = make_doc()
        doc["content"] = [
            {"text": "---\nwip: true\n---\nBody text.", "metadata": {"source": "notion"}}
        ]
        results = converter.convert(doc)
        chunk_meta = results[0]["chunks"][0]["metadata"]
        assert chunk_meta["source"] == "notion"
        assert chunk_meta["wip"] == "true"

    def test_frontmatter_metadata_overrides_content_metadata(self, converter, make_doc):
        doc = make_doc()
        doc["content"] = [
            {"text": "---\ntitle: FM Title\n---\nBody.", "metadata": {"title": "Old Title"}}
        ]
        results = converter.convert(doc)
        assert results[0]["chunks"][0]["metadata"]["title"] == "FM Title"

    def test_quoted_values_stripped(self, converter, make_doc):
        doc = make_doc(
            content_texts=['---\ntitle: "My Quoted Title"\n---\nBody.']
        )
        results = converter.convert(doc)
        assert results[0]["metadata"]["title"] == "My Quoted Title"


class TestSessionChunking:
    """Tests for session-aware chunking (dispatched when session_id is in frontmatter)."""

    def test_session_uses_multi_turn_splitter(self, converter, make_doc):
        doc = make_doc(
            content_texts=[
                "---\nsession_id: \"abc123\"\nproject: \"my-proj\"\nurl: claude --resume abc123\n---\n"
                "# Session title\n\n"
                "## User\n\nHow do I fix the auth bug?\n\n"
                "## Assistant\n\n<details><summary>Thinking</summary>\nLet me check...\n</details>\n\n"
                "## Assistant\n\n- [Tool: Read] /src/auth.py\n\n"
                "## Assistant\n\nThe bug is in the token validation logic. Fix by checking expiry.\n\n"
                "## User\n\nCan you also add tests?\n\n"
                "## Assistant\n\nSure, here are the tests for the auth module.\n"
            ]
        )
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        # Session splitter groups turns — should produce fewer chunks than heading splitter
        # (heading splitter would create one chunk per ## heading = many chunks)
        assert len(chunks) <= 4
        assert len(chunks) >= 1

    def test_session_chunks_have_metadata(self, converter, make_doc):
        doc = make_doc(
            content_texts=[
                "---\nsession_id: \"abc123\"\nproject: \"my-proj\"\n---\n"
                "## User\n\nQuestion here.\n\n## Assistant\n\nAnswer here.\n"
            ]
        )
        results = converter.convert(doc)
        for chunk in results[0]["chunks"]:
            assert chunk.get("metadata", {}).get("session_id") == "abc123"
            assert chunk.get("metadata", {}).get("project") == "my-proj"

    def test_session_noise_stripped_in_chunks(self, converter, make_doc):
        doc = make_doc(
            content_texts=[
                "---\nsession_id: \"abc123\"\n---\n"
                "## User\n\nWhat's the config?\n\n"
                "## Assistant\n\n<details><summary>Thinking</summary>\nInternal thoughts.\n</details>\n\n"
                "## Assistant\n\n- [Tool: Read] /config.py\n\n"
                "## Assistant\n\nThe config uses environment variables.\n"
            ]
        )
        results = converter.convert(doc)
        all_text = " ".join(c["indexedData"] for c in results[0]["chunks"])
        assert "Internal thoughts" not in all_text
        assert "[Tool:" not in all_text
        assert "environment variables" in all_text

    def test_session_preserves_code_blocks(self, converter, make_doc):
        """Code blocks in session assistant responses should NOT be stripped."""
        doc = make_doc(
            content_texts=[
                "---\nsession_id: \"abc123\"\n---\n"
                "## User\n\nShow me how to do it.\n\n"
                "## Assistant\n\nHere's the code:\n\n```python\ndef hello():\n    print('hi')\n```\n\nThat should work.\n"
            ]
        )
        results = converter.convert(doc)
        all_text = " ".join(c["indexedData"] for c in results[0]["chunks"])
        assert "def hello():" in all_text
        assert "print('hi')" in all_text

    def test_non_session_not_affected(self, converter, make_doc):
        """Documents without session_id use the normal heading splitter."""
        doc = make_doc(
            content_texts=[
                "---\ntitle: Regular Page\n---\n"
                "## User\n\nThis heading happens to be 'User'.\n\n"
                "## Assistant\n\nAnd this happens to be 'Assistant'.\n"
            ]
        )
        results = converter.convert(doc)
        chunks = results[0]["chunks"]
        # Normal heading splitter: each ## heading creates a chunk
        assert any(c.get("heading") == "User" for c in chunks)
        assert any(c.get("heading") == "Assistant" for c in chunks)


class TestBuildUrl:
    def test_extracts_url_from_frontmatter(self, converter, make_doc):
        doc = make_doc(
            file_content="---\ntitle: My Page\nurl: https://notion.so/my-page\n---\nContent"
        )
        results = converter.convert(doc)
        assert results[0]["url"] == "https://notion.so/my-page"

    def test_falls_back_to_file_url(self, converter, make_doc):
        doc = make_doc(file_content="No frontmatter here")
        results = converter.convert(doc)
        assert results[0]["url"].startswith("file://")

    def test_no_url_in_frontmatter_falls_back(self, converter, make_doc):
        doc = make_doc(file_content="---\ntitle: No URL\n---\nContent")
        results = converter.convert(doc)
        assert results[0]["url"].startswith("file://")
