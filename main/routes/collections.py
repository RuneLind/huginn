"""Collection-level routes — listing, tags, document lookup, manual update."""
import json
import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from main.runtime.knowledge_store import KnowledgeStore, get_store, run_collection_update

router = APIRouter()


@router.get("/api/collections")
def list_collections(store: KnowledgeStore = Depends(get_store)):
    result = []
    for name, searcher in store.get_searchers().items():
        try:
            manifest_text = store.disk_persister.read_text_file(f"{name}/manifest.json")
            manifest = json.loads(manifest_text)
        except FileNotFoundError:
            manifest = {}
        result.append({
            "name": name,
            "document_count": manifest.get("numberOfDocuments", 0),
            "chunk_count": manifest.get("numberOfChunks", 0),
            "embedding_count": searcher.indexer.get_size(),
            "updatedTime": manifest.get("updatedTime"),
        })
    return {"collections": result}


@router.get("/api/tags")
def list_tags(
    collection: str = Query(None, description="Collection name (all if omitted)"),
    store: KnowledgeStore = Depends(get_store),
):
    """Return tag distribution for a collection (or all collections). Cached at startup."""
    target_names = [collection] if collection else store.collection_names()
    result = {}
    for name in target_names:
        if not store.has_collection(name):
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        tags = store.get_tag_counts([name]).get(name, {})
        result[name] = {
            "unique_tags": len(tags),
            "tags": tags,
        }
    return result


def _resolve_doc_date(doc: dict) -> str | None:
    """Best-effort 'added' date for a document.

    Prefers the frontmatter ``date`` (day-precision, set at ingest) and falls
    back to ``modifiedTime`` (file mtime, which can be reset by bulk reindexing).
    """
    metadata = doc.get("metadata") or {}
    return metadata.get("date") or doc.get("modifiedTime")


def _read_doc_date(store: KnowledgeStore, doc_path: str) -> str | None:
    """Read a single document JSON and return its added date, or None on error."""
    try:
        doc = json.loads(store.disk_persister.read_text_file(doc_path))
    except Exception:
        return None
    return _resolve_doc_date(doc)


@router.get("/api/collection/{name}/documents")
def list_collection_documents(
    name: str,
    include_dates: bool = Query(
        False,
        description="Attach each document's added date. Slower — reads every document file.",
    ),
    store: KnowledgeStore = Depends(get_store),
):
    """List all documents in a collection with their IDs and URLs.

    When ``include_dates`` is set, each entry also carries a ``date`` field
    (frontmatter date, falling back to file mtime) so callers can sort/group by
    recency. This reads every document file, so it is opt-in to keep the default
    listing (used by hot paths like duplicate checks) cheap.
    """
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        mapping_text = store.disk_persister.read_text_file(
            f"{name}/indexes/index_document_mapping.json"
        )
        mapping = json.loads(mapping_text)
    except Exception:
        return {"documents": []}

    seen_ids = set()
    documents = []
    for entry in mapping.values():
        doc_id = entry.get("documentId", "")
        doc_url = entry.get("documentUrl", "")
        if doc_id in seen_ids or not doc_url:
            continue
        seen_ids.add(doc_id)
        doc = {"id": doc_id, "url": doc_url}
        if include_dates:
            doc["date"] = _read_doc_date(store, entry.get("documentPath", ""))
        documents.append(doc)

    return {"documents": documents}


@router.get("/api/document/{collection}/{doc_id:path}")
def get_document(collection: str, doc_id: str, store: KnowledgeStore = Depends(get_store)):
    if not store.has_collection(collection):
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

    if doc_id.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    doc_path = f"{collection}/documents/{doc_id}"
    if not doc_id.endswith(".json"):
        doc_path += ".json"

    base_dir = os.path.realpath(store.disk_persister.base_path)
    resolved = os.path.realpath(os.path.join(base_dir, doc_path))
    if not resolved.startswith(base_dir + os.sep):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    try:
        doc_text = store.disk_persister.read_text_file(doc_path)
        return json.loads(doc_text)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")


@router.post("/api/collections/{name}/update")
def update_collection(
    name: str,
    background_tasks: BackgroundTasks,
    store: KnowledgeStore = Depends(get_store),
):
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    background_tasks.add_task(run_collection_update, name, store)
    return {"status": "update_started", "collection": name}
