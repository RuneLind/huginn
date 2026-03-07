import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


class MarkdownHeadingSplitter:
    """Splits markdown text into chunks at H1-H3 heading boundaries.

    Stage 1: Split at heading boundaries into logical sections.
    Stage 2: Sub-split sections exceeding chunk_size with RecursiveCharacterTextSplitter.
    Fallback: Content with no headings uses plain RecursiveCharacterTextSplitter.
    """

    _HEADING_RE = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    _ATX_CLOSING_RE = re.compile(r'\s+#+\s*$')

    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _split_by_headings(self, text):
        """Split text into sections at H1-H3 heading boundaries.

        Returns list of {"heading": str or None, "text": str}.
        """
        matches = list(self._HEADING_RE.finditer(text))
        if not matches:
            return [{"heading": None, "text": text}]

        sections = []

        # Preamble: text before the first heading
        preamble = text[:matches[0].start()]
        if preamble.strip():
            sections.append({"heading": None, "text": preamble.strip()})

        for i, match in enumerate(matches):
            heading_text = self._ATX_CLOSING_RE.sub('', match.group(2).strip())
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            sections.append({"heading": heading_text, "text": body})

        return sections

    def split(self, text):
        """Split markdown text into chunks, respecting heading boundaries.

        Returns list of {"text": str, "heading": str or None}.
        """
        sections = self._split_by_headings(text)

        # Fallback: no headings found (single section with heading=None)
        if len(sections) == 1 and sections[0]["heading"] is None:
            return [{"text": chunk, "heading": None}
                    for chunk in self.text_splitter.split_text(text)]

        chunks = []
        for section in sections:
            body = section["text"]
            heading = section["heading"]

            if not body:
                continue

            if len(body) <= self.chunk_size:
                chunks.append({"text": body, "heading": heading})
            else:
                for sub_chunk in self.text_splitter.split_text(body):
                    chunks.append({"text": sub_chunk, "heading": heading})

        return chunks
