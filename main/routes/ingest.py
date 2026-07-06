"""Ingest routes — YouTube, X articles, TikTok, Anthropic, Jira.

The POST /ingest handlers are all the same shape (validate path config → write →
find similar → optionally reindex → shape response). Rather than repeat that per
source, one generic handler is generated from each ``IngestSource`` in the
registry. Source-specific behavior (which result keys to surface, the similarity
query, self-link exclusion, whether to reindex) lives in the registry config.
"""
import logging
from contextlib import contextmanager

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from main.core.search_response_formatter import extract_chunk_text, truncate_snippet
from main.ingest.registry import INGEST_SOURCES, IngestSource
from main.ingest.youtube import fetch_transcript, list_categories
from main.runtime.knowledge_store import KnowledgeStore, get_store, run_collection_update
from main.utils.filename import title_from_doc_path

router = APIRouter()
logger = logging.getLogger(__name__)


@contextmanager
def _ingest_errors(operation: str):
    """Convert unexpected ingest failures into a structured 500.

    The write/sanitize/index paths (file I/O, frontmatter escaping, similarity
    search) can raise arbitrary errors; without this the client got a bare 500
    and the only record was a stack trace. Here we log the full traceback
    server-side and return a clear JSON error. Deliberate HTTPExceptions (e.g.
    503 unconfigured, 422 validation) pass through untouched.
    """
    try:
        yield
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("%s failed", operation)
        raise HTTPException(status_code=500, detail=f"{operation} failed: {exc}") from exc


def _find_similar_documents(searcher, query: str, exclude_match) -> list[dict]:
    """Run a similarity search and return up to 5 {title, url, snippet} results.

    exclude_match is called on each result; truthy means skip (e.g. self-link).
    """
    search_result = searcher.search(
        query,
        max_number_of_chunks=30,
        max_number_of_documents=5,
        include_matched_chunks_content=True,
    )
    similar = []
    for doc in search_result.get("results", []):
        if exclude_match(doc):
            continue
        doc_title = title_from_doc_path(doc.get("path", ""))
        chunks = doc.get("matchedChunks", [])
        snippet = ""
        if chunks:
            raw = chunks[0].get("content", "")
            snippet = truncate_snippet(extract_chunk_text(raw))
        similar.append({
            "title": doc_title,
            "url": doc.get("url", ""),
            "snippet": snippet,
        })
    return similar[:5]


def _similar_for_collection(store, collection_name, query, exclude_match) -> list[dict]:
    """Look up the searcher for a configured collection and run a similarity search."""
    if not (collection_name and store.has_collection(collection_name)):
        return []
    searcher = store.get_searchers([collection_name]).get(collection_name)
    if not searcher:
        return []
    return _find_similar_documents(searcher, query=query, exclude_match=exclude_match)


def _maybe_enqueue_reindex(store, background_tasks, collection_name) -> str:
    """Enqueue a background reindex unless one is already running for the collection.

    Ingest succeeds regardless of reindex state, so a busy collection skips the
    duplicate rebuild (avoiding the concurrent read-modify-write clobber, H4)
    rather than failing the ingest. Returns a status the caller surfaces.
    """
    if not (collection_name and store.has_collection(collection_name)):
        return "not_configured"
    if not store.try_begin_update(collection_name):
        return "skipped_already_running"
    background_tasks.add_task(run_collection_update, collection_name, store)
    return "started"


def _make_ingest_handler(src: IngestSource):
    """Build a FastAPI POST handler for one push source from its registry config.

    The request body model is injected via ``__annotations__`` so FastAPI parses
    and validates it exactly as a statically-declared handler would.
    """

    def handler(
        req,
        background_tasks: BackgroundTasks,
        request: Request,
        store: KnowledgeStore = Depends(get_store),
    ):
        path = getattr(request.app.state, src.path_attr)
        collection = getattr(request.app.state, src.collection_attr)
        if not path:
            raise HTTPException(status_code=503, detail=src.not_configured_detail)

        with _ingest_errors(src.operation):
            result = src.ingest_fn(req, **{src.path_kwarg: path})

            similar = _similar_for_collection(
                store, collection,
                query=src.similar_query(req, result),
                exclude_match=lambda doc: src.exclude_match(req, doc),
            )

            response = {"status": "ingested"}
            for key in src.response_fields:
                response[key] = result[key]
            response["similar"] = similar
            if src.do_reindex:
                response["reindex"] = _maybe_enqueue_reindex(store, background_tasks, collection)
            return response

    handler.__name__ = f"{src.name}_ingest"
    handler.__annotations__["req"] = src.request_model
    return handler


for _src in INGEST_SOURCES:
    router.add_api_route(
        _src.route_path,
        _make_ingest_handler(_src),
        methods=["POST"],
        name=f"{_src.name}_ingest",
    )


@router.get("/api/youtube/transcript/{video_id}")
def youtube_transcript(video_id: str):
    """Fetch raw YouTube transcript without summarizing. Used by javrvis to get transcript for its own Claude call."""
    text = fetch_transcript(video_id)
    return {"video_id": video_id, "transcript": text, "char_count": len(text)}


@router.get("/api/youtube/categories")
def youtube_categories(request: Request):
    """List available YouTube transcript categories."""
    yt_path = request.app.state.youtube_transcripts_path
    if not yt_path:
        raise HTTPException(status_code=503, detail="YouTube transcripts path not configured")
    return {"categories": list_categories(yt_path)}
