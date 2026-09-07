import os
import re
import unicodedata
from urllib.parse import urlsplit

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
                                # writes these two keys with this meaning — which is no
                                # longer hypothetical: the YOUTUBE ingest emits
                                # `summary_kind` too (2026-09-07), the second vertical to,
                                # and it is why the key is not namespaced. `summary_lang`
                                # is still vimeo-only. The rest of the 2026-09-05 grep
                                # stands: zero of the 809 markdown files under data/ and
                                # the sister wiki tree carried either — on the mini; the
                                # x/tiktok/article source trees live on the laptop and were
                                # not in that grep, and their `author:` is the one this
                                # allowlist now serves.
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


#: Unicode categories a stored image destination may not contain. Cc and Cf
#: are the load-bearing two; Zl/Zp (U+2028/9) and the C1 control U+0085 are
#: ``\s`` to ``re`` and already truncate the destination in
#: ``_MD_IMAGE_PARTS_RE`` before this gate runs — listed for the reader, not
#: for the check.
_NON_PRINTABLE = frozenset({"Cc", "Cf", "Zl", "Zp"})


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
             else self._clean_document_text(self._strip_frontmatter(content_part['text']))
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

    #: A markdown image, split: alt text, destination (``<dest>`` or the first
    #: token), and whatever follows (a title) — which is never re-emitted.
    _MD_IMAGE_PARTS_RE = re.compile(r'!\[([^\]]*)\]\(\s*(?:<([^>]*)>|([^)\s]*))')
    #: Query keys that carry a credential: AWS SigV4/V2, CloudFront, Google
    #: Cloud Storage and Azure SAS. Matched as ``key=`` after ``?``, ``&`` or ``;``.
    _SIGNED_QUERY_RE = re.compile(r'(?:^|[&;])(?:x-amz-[^=&;]*|signature|sig|x-goog-[^=&;]*|key-pair-id)=', re.IGNORECASE)
    #: Alt text the document-level text keeps: caption WORDS. Each
    #: whitespace-separated token is at most 20 characters from an enumerated
    #: set — unicode word characters and caption punctuation — and the whole
    #: alt at most 80; no token is scheme-shaped. No ``/``, ``?``, ``&``,
    #: ``;``, ``=``, ``#``, tab or line break is in the set, so no url or blob
    #: can be spelled. The RESIDUAL, stated exactly (separators count toward
    #: the 80): at most 77 characters of that alphabet survive, as four words
    #: of 20+20+20+17 — an AWS access key id (20 chars) or a 16-hex secret
    #: fits one word; a JWT, ``ghp_…`` or ``sk-…`` token (40+ chars, unsplit)
    #: does not. Accepted: a caption is free text and cannot be told from a
    #: short token by shape. ``Slide at 00:01:33`` and ``Diagram: the 3-step
    #: loop`` pass.
    _ALT_TOKEN_RE = re.compile(r"\A(?![a-z][a-z0-9+.\-]*:\S)[\w.,:!'\u2019()\-\u2013\u2014\u2026]{1,20}\Z", re.IGNORECASE)
    _ALT_MAX = 80
    #: IDNA label separators besides ``.`` — a host spelled with one resolves
    #: like the ASCII dot and must be judged as one.
    _IDNA_DOTS = str.maketrans({"\u3002": ".", "\uff0e": ".", "\uff61": "."})
    _DEST_MAX = 2048

    @classmethod
    def _plain_alt(cls, alt):
        alt = " ".join(alt.split())
        if len(alt) > cls._ALT_MAX:
            return ""
        return alt if all(cls._ALT_TOKEN_RE.match(t) for t in alt.split(" ") if t) else ""

    @classmethod
    def _document_text_image(cls, image_markdown):
        """What the document-level text keeps of a markdown image: ``None`` to
        drop it, else a NORMALIZED ``![alt](dest)`` — the title is never
        re-emitted, the alt is kept only as caption words (``_plain_alt``:
        an enumerated charset, tokens ≤20, whole ≤80) so it can spell neither
        a url nor a token, and
        the destination must pass a WHITELIST. Every channel of the image is
        therefore bounded, not just the destination: a round that judged the
        destination and re-emitted the whole match let a signature ride in
        through the title and a base64 blob through the alt.

        The destination whitelist, decided over scheme × authority (userinfo
        ⇒ drop) × host × path-params × query with ``urlsplit`` (a ``ValueError``
        from it drops the image), over a destination ≤2048 printable chars:

        * no scheme, no authority (``img/a.png``, ``/api/x.jpg``) — kept unless
          the query is signed;
        * an authority — protocol-relative ``//host/…`` or ``http(s)://host/…``
          — dropped when it carries userinfo, or the host (IDNA dots folded,
          trailing dot stripped) is ``amazonaws.com`` or under it, or the
          query or path-params are signed;
        * ``http(s):`` with NO authority (``https:example.com/k.png`` — generic
          URI syntax, not an http url) — dropped; a previous parser crashed on it;
        * any other scheme (``data:``, ``javascript:``, ``ftp:``) — dropped; a
          ``data:`` blob would ride into every contextual-prefix prompt and
          sensitivity-sweep window.

        A signature in the FRAGMENT is not a credential the server ever sees
        and is kept."""
        m = cls._MD_IMAGE_PARTS_RE.match(image_markdown)
        if not m:
            return None
        alt = cls._plain_alt(m.group(1))
        dest = (m.group(2) if m.group(2) is not None else m.group(3)).strip()
        # A destination is bounded and printable: a 100 KB base64 path, a NUL
        # or DEL, a C1 control, a bidi override or a zero-width character
        # (categories Cc/Cf/Zl/Zp) is dropped, not stored.
        if len(dest) > cls._DEST_MAX or any(unicodedata.category(c) in _NON_PRINTABLE for c in dest):
            return None
        # After the destination only a title or the close may follow: a stray
        # ``>`` (``<a>b.png>``) or a bare token (``<a.png> extra``) is dropped
        # rather than re-emitted as a destination the source never named.
        if not dest or not re.match(r'\s*(?:[")\']|$)', image_markdown[m.end():]):
            return None
        try:
            parts = urlsplit(dest)
        except ValueError:
            return None
        scheme = parts.scheme.lower()
        if scheme and scheme not in ("http", "https"):
            return None
        if scheme and not parts.netloc:
            return None
        if parts.netloc:
            # Userinfo is the authority's credential slot: never re-emitted.
            if "@" in parts.netloc:
                return None
            # NFKC folds fullwidth letters. The format characters IDNA ignores
            # (soft hyphen, ZWSP, BOM) never reach here: the printable gate
            # above drops every Cf character in the destination. A percent-
            # encoded dot (``amazonaws%2ecom``) is NOT folded — urlsplit does
            # not decode the host — and slips this UNSIGNED-host rule; a signed
            # url is dropped by the query check whatever the host spelling.
            host = unicodedata.normalize("NFKC", parts.hostname or "").translate(cls._IDNA_DOTS).rstrip(".").lower()
            if host == "amazonaws.com" or host.endswith(".amazonaws.com"):
                return None
        # ``;params`` ride in the PATH for urlsplit; a credential there counts.
        if cls._SIGNED_QUERY_RE.search(parts.query) is not None \
                or cls._SIGNED_QUERY_RE.search(parts.path.partition(";")[2]) is not None:
            return None
        # The angle-bracket form is what lets a destination carry a space.
        return f"![{alt}](<{dest}>)" if m.group(2) is not None else f"![{alt}]({dest})"

    def _clean_document_text(self, text):
        """The document-level ``text`` — what ``/api/document`` serves and a
        reader renders — keeps ordinary markdown images; the chunk text that
        feeds the embeddings drops them. Muninn's Vimeo captures quote slides as
        ``![Slide at HH:MM:SS](/api/vimeo/frames/...)``, and the chunk rule
        erased every one from the stored copy while the source .md still had
        them (measured 2026-09-06). Which images, and re-emitted how:
        ``_document_text_image``. Fenced code is still dropped here: ``text``
        also feeds the contextual-prefix prompt, the sensitivity sweep, the
        search dedup hash and the graph blurb, and widening what those see is a
        separate decision."""
        text = self._CODE_BLOCK_RE.sub('', text)
        text = self._MD_IMAGE_RE.sub(
            lambda m: self._document_text_image(m.group(0)) or '', text)
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