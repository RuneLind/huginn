"""Ingest routes — YouTube, X articles, Jira (writes to source dirs, then reindexes)."""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from main.core.search_response_formatter import extract_chunk_text, truncate_snippet
from main.ingest.jira import JiraIngestRequest, ingest_jira
from main.ingest.x_articles import XArticleIngestRequest, ingest_x_article
from main.ingest.youtube import (
    YouTubeIngestRequest,
    fetch_transcript,
    ingest_youtube,
    list_categories,
)
from main.runtime.knowledge_store import KnowledgeStore, get_store, run_collection_update
from main.utils.filename import title_from_doc_path

router = APIRouter()


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


@router.post("/api/youtube/ingest")
def youtube_ingest(
    req: YouTubeIngestRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    store: KnowledgeStore = Depends(get_store),
):
    """Ingest a YouTube transcript: summarize via Claude, auto-categorize, save, index, return similar."""
    yt_path = request.app.state.youtube_transcripts_path
    yt_collection = request.app.state.youtube_collection
    if not yt_path:
        raise HTTPException(status_code=503, detail="YouTube transcripts path not configured")

    result = ingest_youtube(req, transcripts_path=yt_path)

    similar = _similar_for_collection(
        store, yt_collection,
        query=result["summary"][:2000],
        exclude_match=lambda doc: doc.get("url", "") == req.url,
    )
    reindex = _maybe_enqueue_reindex(store, background_tasks, yt_collection)

    return {
        "status": "ingested",
        "file_path": result["file_path"],
        "category": result["category"],
        "summary": result["summary"],
        "similar": similar,
        "reindex": reindex,
    }


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


@router.post("/api/x-articles/ingest")
def x_article_ingest(
    req: XArticleIngestRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    store: KnowledgeStore = Depends(get_store),
):
    """Ingest an X/Twitter article: save summary as markdown, find similar, reindex."""
    xa_path = request.app.state.x_articles_sources_path
    xa_collection = request.app.state.x_articles_collection
    if not xa_path:
        raise HTTPException(status_code=503, detail="X articles sources path not configured (--x-articles-sources-path)")

    result = ingest_x_article(req, sources_path=xa_path)

    similar = _similar_for_collection(
        store, xa_collection,
        query=req.summary[:2000],
        exclude_match=lambda doc: doc.get("url", "") == req.url,
    )
    reindex = _maybe_enqueue_reindex(store, background_tasks, xa_collection)

    return {
        "status": "ingested",
        "file_path": result["file_path"],
        "author": result["author"],
        "category": result["category"],
        "summary": result["summary"],
        "similar": similar,
        "reindex": reindex,
    }


@router.post("/api/jira/ingest")
def jira_ingest(
    req: JiraIngestRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    store: KnowledgeStore = Depends(get_store),
):
    """Ingest a Jira issue from DOM-scraped content: save as markdown, find similar, reindex.

    If an existing file for this issue_key is found, merges metadata to preserve
    epic_summary, project, and other fields the Chrome extension doesn't capture.
    """
    jira_path = request.app.state.jira_sources_path
    jira_collection = request.app.state.jira_collection

    if not jira_path:
        raise HTTPException(status_code=503, detail="Jira sources path not configured (--jira-sources-path)")

    result = ingest_jira(req, sources_path=jira_path)

    similar = _similar_for_collection(
        store, jira_collection,
        query=f"{req.issueKey} {result['summary']}",
        exclude_match=lambda doc: req.issueKey in doc.get("url", ""),
    )

    # Automatic reindex skipped — the daily update script handles both
    # collection reindexing and knowledge graph rebuild in one pass.
    # Use POST /api/collections/{name}/update to trigger manually if needed.

    return {
        "status": "ingested",
        "issue_key": result["issue_key"],
        "file_path": result["file_path"],
        "summary": result["summary"],
        "similar": similar,
    }
