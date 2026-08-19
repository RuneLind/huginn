import logging
import time
from typing import Iterable

from main.core.contextual_prefix.cache import ContextualCache
from main.core.contextual_prefix.prefix_generator import PrefixGenerator


logger = logging.getLogger(__name__)


MIN_CHUNK_CHARS_FOR_PREFIX = 80


class ChunkPrefixer:
    """Orchestrates contextual-prefix generation for a single converted document.

    Takes a converted_document of the shape produced by *_document_converter.py:
        {"id": ..., "text": ..., "chunks": [{"indexedData": str, ...}, ...]}

    Mutates each chunk in place:
        chunk["contextualPrefix"] = prefix
        chunk["indexedData"] = prefix + "\\n\\n" + chunk["indexedData"]

    Skips chunks shorter than MIN_CHUNK_CHARS_FOR_PREFIX (typically breadcrumb-only).
    """

    def __init__(self, generator: PrefixGenerator, cache: ContextualCache):
        self.generator = generator
        self.cache = cache

    def prefix_document(self, converted_document: dict) -> None:
        start_time = time.monotonic()
        doc_id = converted_document["id"]
        doc_text = converted_document.get("text") or self._reconstruct_doc_text(converted_document["chunks"])
        chunks = converted_document.get("chunks", [])

        if not chunks:
            return

        indices_to_generate: list[int] = []
        chunk_texts_to_generate: list[str] = []
        cached_prefixes: dict[int, str] = {}

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("indexedData", "")
            if len(chunk_text) < MIN_CHUNK_CHARS_FOR_PREFIX:
                continue
            cached = self.cache.get(doc_id, chunk_text, self.generator.model_id)
            if cached is not None:
                cached_prefixes[i] = cached
            else:
                indices_to_generate.append(i)
                chunk_texts_to_generate.append(chunk_text)

        cache_hits = len(cached_prefixes)
        requested = len(chunk_texts_to_generate)

        new_prefixes: list[str] = []
        if chunk_texts_to_generate:
            new_prefixes = self._generate_with_retry(doc_text, chunk_texts_to_generate, doc_id)

        for j, idx in enumerate(indices_to_generate):
            if j >= len(new_prefixes):
                break
            prefix = (new_prefixes[j] or "").strip()
            if not prefix:
                continue
            cached_prefixes[idx] = prefix
            self.cache.put(doc_id, chunks[idx]["indexedData"], self.generator.model_id, prefix,
                           doc_path=converted_document.get("url"))

        for idx, prefix in cached_prefixes.items():
            chunk = chunks[idx]
            chunk["contextualPrefix"] = prefix
            chunk["indexedData"] = f"{prefix}\n\n{chunk['indexedData']}"

        generated = len(cached_prefixes) - cache_hits
        duration = time.monotonic() - start_time
        logger.info(
            "prefix doc=%s chunks=%d cached=%d generated=%d requested=%d duration=%.1fs",
            doc_id, len(chunks), cache_hits, generated, requested, duration,
        )

    def _generate_with_retry(self, doc_text: str, chunks_to_generate: list[str], doc_id: str) -> list[str]:
        """Generate prefixes, retrying once if the backend returns the wrong count.

        Backends stochastically drop or invent an item (Ollama also returns [] on its own
        per-batch mismatch); an A/B run lost every prefix on 1 of 7 docs that way. One retry
        with the identical prompt recovers those. Exceptions are NOT retried — the backends
        already retry transport errors themselves.
        """
        for attempt in (1, 2):
            try:
                prefixes = self.generator.generate(doc_text, chunks_to_generate)
            except Exception:
                logger.exception("Prefix generation failed for doc %s; skipping prefixes for %d chunks",
                                 doc_id, len(chunks_to_generate))
                return []

            if len(prefixes) == len(chunks_to_generate):
                if attempt == 2:
                    logger.info("Prefix retry succeeded for doc %s after first attempt mismatched", doc_id)
                return prefixes

            logger.warning("Generator returned %d prefixes for %d chunks (doc %s, attempt %d)%s",
                           len(prefixes), len(chunks_to_generate), doc_id, attempt,
                           "; retrying once" if attempt == 1 else "; skipping doc")
        return []

    def prefix_documents(self, converted_documents: Iterable[dict]) -> None:
        for converted_document in converted_documents:
            self.prefix_document(converted_document)
        self.cache.flush()

    @staticmethod
    def _reconstruct_doc_text(chunks: list[dict]) -> str:
        return "\n\n".join(c.get("indexedData", "") for c in chunks)
