import os
import re

from main.sources.files.markdown_heading_splitter import MarkdownHeadingSplitter
from main.sources.files.session_markdown_splitter import SessionMarkdownSplitter

# Frontmatter fields to preserve as document metadata (key in frontmatter -> key in metadata)
_FRONTMATTER_METADATA_FIELDS = {"wip", "title", "breadcrumb", "space", "page_id", "session_id", "project", "gitBranch", "tags", "category", "date", "url",
                                "issue_key", "status", "issue_type", "epic_link", "epic_summary", "labels"}


class FilesDocumentConverter:
    _FRONTMATTER_RE = re.compile(r'^---\n.*?\n---\n?', re.DOTALL)
    _FRONTMATTER_FIELD_RE = re.compile(r'^([^:\n]+):\s*(.+)$', re.MULTILINE)
    _MD_IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^)]+\)')
    _S3_URL_RE = re.compile(r'https://[a-zA-Z0-9._-]+\.s3\.[a-zA-Z0-9-]+\.amazonaws\.com/[^\s)]*')
    _CODE_BLOCK_RE = re.compile(r'```[^\n]*\n.*?```', re.DOTALL)

    def __init__(self):
        self.heading_splitter = MarkdownHeadingSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        self.session_splitter = SessionMarkdownSplitter(
            target_chars=2500,
            min_chars=400,
        )

    def convert(self, document):
        breadcrumb = self.__build_breadcrumb(document['fileRelativePath'])
        fm_metadata = self.__extract_frontmatter_metadata(document)
        is_session = bool(fm_metadata and fm_metadata.get('session_id'))
        result = {
            "id": document['fileRelativePath'],
            "url": self.__build_url(document),
            "modifiedTime": document['modifiedTime'],
            "text": self.__build_document_text(document, breadcrumb, is_session),
            "chunks": self.__split_to_chunks(document, breadcrumb, fm_metadata, is_session)
        }
        if fm_metadata:
            result["metadata"] = fm_metadata
        return [result]
    
    def __build_breadcrumb(self, file_relative_path):
        """Convert file path to breadcrumb format: [Part1 > Part2 > PageTitle]

        Paths deeper than 4 levels are truncated: [First > ... > Parent > Page]
        """
        parts = file_relative_path.replace("\\", "/").split("/")
        # Strip file extension from the last part (the page title)
        if parts:
            name, _ = os.path.splitext(parts[-1])
            parts[-1] = name
        if len(parts) > 4:
            parts = [parts[0], "...", parts[-2], parts[-1]]
        return "[" + " > ".join(parts) + "]"

    def __build_document_text(self, document, breadcrumb, is_session=False):
        content = self.__convert_to_text(
            [self._strip_frontmatter(content_part['text']) if is_session
             else self._clean_chunk_text(self._strip_frontmatter(content_part['text']))
             for content_part in document['content']], "")
        return self.__convert_to_text([breadcrumb, content])
    
    def __convert_to_text(self, elements, delimiter="\n\n"):
        return delimiter.join([element for element in elements if element]).strip()
    
    def _strip_frontmatter(self, text):
        return self._FRONTMATTER_RE.sub('', text)

    def __extract_frontmatter_metadata(self, document):
        """Extract selected fields from YAML frontmatter as document metadata."""
        for content_part in document['content']:
            match = self._FRONTMATTER_RE.match(content_part['text'])
            if match:
                fm_block = match.group(0)
                metadata = {}
                for field_match in self._FRONTMATTER_FIELD_RE.finditer(fm_block):
                    key = field_match.group(1).strip()
                    if key in _FRONTMATTER_METADATA_FIELDS:
                        metadata[key] = field_match.group(2).strip().strip('"')
                return metadata if metadata else None
        return None

    def _clean_chunk_text(self, text):
        text = self._CODE_BLOCK_RE.sub('', text)
        text = self._MD_IMAGE_RE.sub('', text)
        return self._S3_URL_RE.sub('[file]', text)

    def __split_to_chunks(self, document, breadcrumb, fm_metadata=None, is_session=False):
        splitter = self.session_splitter if is_session else self.heading_splitter

        chunks = []

        for content_part in document['content']:
            stripped = self._strip_frontmatter(content_part['text'])
            # Sessions: skip _clean_chunk_text (preserves code blocks); session splitter handles noise
            cleaned = stripped if is_session else self._clean_chunk_text(stripped)
            if cleaned.strip():
                for section in splitter.split(cleaned):
                    heading = section["heading"]

                    # Merge content-part metadata (e.g. from Unstructured) with frontmatter metadata
                    chunk_meta = {}
                    if "metadata" in content_part:
                        chunk_meta.update(content_part['metadata'])
                    if fm_metadata:
                        chunk_meta.update(fm_metadata)

                    # Inject tags and epic context into indexed text for embedding + BM25 enrichment
                    tags_line = f"tags: {chunk_meta['tags']}\n" if chunk_meta.get('tags') else ""
                    epic_line = f"epic: {chunk_meta['epic_summary']}\n" if chunk_meta.get('epic_summary') else ""
                    context_lines = tags_line + epic_line
                    if heading:
                        indexed_data = f"{breadcrumb}\n{context_lines}## {heading}\n{section['text']}"
                    else:
                        indexed_data = f"{breadcrumb}\n{context_lines}{section['text']}"

                    chunk = {"indexedData": indexed_data}
                    if chunk_meta:
                        chunk["metadata"] = chunk_meta
                    if heading:
                        chunk["heading"] = heading
                    chunks.append(chunk)

        if not chunks:
            chunks.append({"indexedData": breadcrumb})

        return chunks

    _FRONTMATTER_URL_RE = re.compile(r'^url:\s*(.+)$', re.MULTILINE)

    def __build_url(self, document):
        file_path = document['fileFullPath']

        if file_path.endswith('.md'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                match = self._FRONTMATTER_RE.match(content)
                if match:
                    url_match = self._FRONTMATTER_URL_RE.search(match.group(0))
                    if url_match:
                        return url_match.group(1).strip().strip('"')
            except Exception:
                pass

        return f"file://{document['fileFullPath']}"