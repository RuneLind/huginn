import os
import re

from main.privacy.alias_registry import ALIAS_CHANGED_KEY
from main.sources.files.markdown_heading_splitter import MarkdownHeadingSplitter
from main.sources.files.session_markdown_splitter import SessionMarkdownSplitter
from main.utils.frontmatter import parse_tags, read_frontmatter, strip_frontmatter

# Frontmatter fields to preserve as document metadata (key in frontmatter -> key in metadata)
#
# GLOBAL, not per source: every markdown collection is converted here, so a key
# added below surfaces on any document that happens to carry it. That is why
# this is an allowlist in the first place — a field an ingest writer emits and
# this set does not name reaches no API consumer at all, however carefully it was
# written.
#
# It is also why the vimeo id is `vimeo_video_id` and not `video_id`: `video_id`
# is ALREADY WRITTEN, by main/fetchers/youtube/youtube_channel_fetcher.py, into
# every file under data/sources/youtube-transcripts/markdown/EmmaHubbard (75 on
# disk 2026-09-04) — the basePath of the on-disk `emma-hubbard-transcripts`
# collection, which is not in start.sh's --collections today but would surface
# them on any reload — so
# allowlisting the bare key would serve a YouTube id under the name a Vimeo
# consumer reads. The other three were greped the same day: no writer in `main/`
# emits `caption_lang`, `caption_kind` or `duration_sec` as a frontmatter key
# except the vimeo ingest, and the only files under `data/sources/` carrying any
# of them are the two vimeo captures. (`duration_sec` also appears in the
# YouTube fetcher as a local variable spelling a mm:ss line — not a key.)
_FRONTMATTER_METADATA_FIELDS = {"wip", "title", "breadcrumb", "space", "page_id", "session_id", "project", "gitBranch", "tags", "category", "date", "url",
                                "issue_key", "status", "issue_type", "epic_link", "epic_summary", "labels",
                                "relevance_score", "combined_score", "engagement_score", "author_score",
                                "vimeo_video_id", "caption_lang", "caption_kind", "duration_sec",
                                # The summary's own provenance — which summary KIND wrote the
                                # body and the language it is written in. Deliberately NOT
                                # namespaced under vimeo: `write_summary` is shared by six
                                # verticals, and any of them growing a kind/language picker
                                # writes these two keys with this meaning. Greped 2026-09-05:
                                # only the vimeo ingest emits either, and zero of the 809
                                # markdown files under data/ and the sister wiki tree carry one
                                # — on the mini; the x/tiktok/article source trees live on the
                                # laptop and were not in that grep, and their `author:` is the
                                # one this allowlist now serves.
                                "summary_kind", "summary_lang",
                                # Vimeo v2 PR 2: what oEmbed knew. `author` is ALREADY
                                # written by the x_articles/tiktok/articles ingests and
                                # was never served; allowlisting it surfaces theirs too,
                                # with the same meaning (who published the source).
                                "author", "upload_date", "speaker", "thumbnail_url"}

#: Metadata fields served as INTEGERS rather than as the strings
#: ``read_frontmatter`` hands back (it is a line parser, not YAML — every value
#: arrives as text). Same rule as the collection routes' score coercion
#: (``main/routes/collections.py``'s ``_resolve_doc_scores``): a value that is
#: not a whole number is OMITTED rather than served as text or as a NaN, so
#: "key missing" stays the single no-value signal for consumers.
_FRONTMATTER_INT_FIELDS = {"duration_sec"}


def _coerce_int(value):
    """``value`` as an ``int``, or ``None`` when it does not parse as one.

    "Parses as an int" is `int()`'s rule, which is wider than "is a whole
    number": it accepts surrounding whitespace, a sign, underscores between
    digits and any Unicode decimal digit (`٣٢٢٠` is 3220). Nothing narrows it,
    because the input is a line from a frontmatter block a human may have typed
    and the alternative to accepting those is serving them as text.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


class FilesDocumentConverter:
    _MD_IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^)]+\)')
    _S3_URL_RE = re.compile(r'https://[a-zA-Z0-9._-]+\.s3\.[a-zA-Z0-9-]+\.amazonaws\.com/[^\s)]*')
    _CODE_BLOCK_RE = re.compile(r'```[^\n]*\n.*?```', re.DOTALL)

    def __init__(self, alias_registry=None):
        # None => byte-identical behaviour to before build-time aliasing existed.
        # Set only for collections main.privacy.resolve_registry puts in scope.
        self.alias_registry = alias_registry
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
        # Aliasing happens HERE, before the chunk prefixer runs on the result
        # (documents_collection_creator calls convert() first), so no contextual
        # prefix is ever generated from a real name.
        if self.alias_registry is not None and self.alias_registry.apply_document(result):
            result[ALIAS_CHANGED_KEY] = True
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
        return strip_frontmatter(text)

    def __extract_frontmatter_metadata(self, document):
        """Extract selected fields from YAML frontmatter as document metadata."""
        for content_part in document['content']:
            fm = read_frontmatter(content_part['text'])
            if fm:
                metadata = {}
                for key, value in fm.items():
                    if key not in _FRONTMATTER_METADATA_FIELDS:
                        continue
                    if key in _FRONTMATTER_INT_FIELDS:
                        number = _coerce_int(value)
                        if number is not None:
                            metadata[key] = number
                        continue
                    metadata[key] = value
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
                    # parse_tags normalizes bracketed (`[a, b]`) or bare (`a, b`) forms;
                    # re-join to a clean comma string so the indexed line never carries
                    # brackets or a stray Python list repr.
                    tags_line = f"tags: {', '.join(parse_tags(chunk_meta['tags']))}\n" if chunk_meta.get('tags') else ""
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

    def __build_url(self, document):
        file_path = document['fileFullPath']

        if file_path.endswith(('.md', '.mdx')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                url = read_frontmatter(content).get("url")
                if url:
                    return url
            except Exception:
                pass

        return f"file://{document['fileFullPath']}"