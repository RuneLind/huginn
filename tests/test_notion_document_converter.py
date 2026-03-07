from main.sources.notion.notion_document_converter import NotionDocumentConverter


def _make_document(title="Test Page", page_id="abc-123", blocks=None, breadcrumb="Root -> Test Page"):
    """Build a minimal Notion document dict as yielded by the reader."""
    page = {
        "id": page_id,
        "last_edited_time": "2025-01-15T10:00:00.000Z",
        "properties": {
            "title": {
                "type": "title",
                "title": [{"plain_text": title}],
            }
        },
    }
    if blocks is None:
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"plain_text": "Hello world", "annotations": {}}],
                },
            }
        ]
    return {"page": page, "blocks": blocks, "breadcrumb": breadcrumb}


class TestConvert:
    def test_basic_conversion(self):
        converter = NotionDocumentConverter()
        results = converter.convert(_make_document())

        assert len(results) == 1
        doc = results[0]
        assert doc["id"] == "abc-123"
        assert doc["title"] == "Test Page"
        assert doc["modifiedTime"] == "2025-01-15T10:00:00.000Z"
        assert "notion.so" in doc["url"]
        assert "Hello world" in doc["text"]
        assert doc["breadcrumb"] == "Root -> Test Page"

    def test_url_strips_dashes(self):
        converter = NotionDocumentConverter()
        results = converter.convert(_make_document(page_id="abc-def-123"))
        assert "abcdef123" in results[0]["url"]

    def test_chunks_include_breadcrumb(self):
        converter = NotionDocumentConverter()
        results = converter.convert(_make_document(breadcrumb="A -> B -> C"))
        chunks = results[0]["chunks"]
        assert chunks[0]["indexedData"] == "A -> B -> C"

    def test_chunks_include_body(self):
        converter = NotionDocumentConverter()
        results = converter.convert(_make_document())
        chunks = results[0]["chunks"]
        # Should have breadcrumb chunk + at least one body chunk
        assert len(chunks) >= 2
        assert "Hello world" in chunks[1]["indexedData"]

    def test_empty_blocks_still_produces_result(self):
        converter = NotionDocumentConverter()
        results = converter.convert(_make_document(blocks=[]))
        assert len(results) == 1
        # Only breadcrumb chunk when body is empty
        assert len(results[0]["chunks"]) == 1

    def test_on_convert_callback_called(self):
        callback_calls = []

        def on_convert(document, result):
            callback_calls.append((document, result))

        converter = NotionDocumentConverter(on_convert=on_convert)
        doc = _make_document()
        converter.convert(doc)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] is doc
        assert callback_calls[0][1]["id"] == "abc-123"

    def test_no_callback_when_not_set(self):
        converter = NotionDocumentConverter()
        # Should not raise
        converter.convert(_make_document())

    def test_properties_included_in_text(self):
        doc = _make_document()
        doc["page"]["properties"]["Svar"] = {
            "type": "rich_text",
            "rich_text": [{"plain_text": "This is the answer", "annotations": {}}],
        }
        converter = NotionDocumentConverter()
        results = converter.convert(doc)
        assert "This is the answer" in results[0]["text"]

    def test_properties_included_in_chunks(self):
        doc = _make_document()
        doc["page"]["properties"]["Tags"] = {
            "type": "multi_select",
            "multi_select": [{"name": "Python"}, {"name": "AI"}],
        }
        converter = NotionDocumentConverter()
        results = converter.convert(doc)
        all_chunk_text = " ".join(c["indexedData"] for c in results[0]["chunks"])
        assert "Python, AI" in all_chunk_text

    def test_empty_properties_no_change(self):
        doc = _make_document()
        doc["page"]["properties"] = {}
        converter = NotionDocumentConverter()
        results = converter.convert(doc)
        assert "Hello world" in results[0]["text"]
