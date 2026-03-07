from langchain_text_splitters import RecursiveCharacterTextSplitter

from .notion_block_to_markdown import convert_blocks_to_markdown, extract_page_properties
from .notion_document_reader import NotionDocumentReader


class NotionDocumentConverter:
    def __init__(self, on_convert=None):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        self.on_convert = on_convert

    def convert(self, document):
        page = document["page"]
        blocks = document["blocks"]
        breadcrumb = document["breadcrumb"]

        page_id = page["id"]
        title = NotionDocumentReader.get_page_title(page)
        last_edited = page.get("last_edited_time", "")
        url = self._build_url(page_id)

        properties_md = extract_page_properties(page.get("properties", {}))
        markdown_body = convert_blocks_to_markdown(blocks)

        content_parts = [p for p in [properties_md, markdown_body] if p]
        content = "\n\n".join(content_parts)

        full_text = f"{breadcrumb}\n\n{content}" if content else breadcrumb

        chunks = [{"indexedData": breadcrumb}]

        if content:
            for chunk in self.text_splitter.split_text(content):
                chunks.append({"indexedData": chunk})

        results = [{
            "id": page_id,
            "url": url,
            "modifiedTime": last_edited,
            "text": full_text,
            "chunks": chunks,
            "title": title,
            "breadcrumb": breadcrumb,
        }]

        if self.on_convert:
            for result in results:
                self.on_convert(document, result)

        return results

    @staticmethod
    def _build_url(page_id):
        clean_id = page_id.replace("-", "")
        return f"https://www.notion.so/{clean_id}"
