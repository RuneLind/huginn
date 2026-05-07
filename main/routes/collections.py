"""Collection-level routes — listing, tags, document lookup, manual update."""
import json
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query


def make_collections_router(store, run_collection_update) -> APIRouter:
    router = APIRouter()

    @router.get("/api/collections")
    def list_collections():
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

    @router.get("/api/collection/{name}/documents")
    def list_collection_documents(name: str):
        """List all documents in a collection with their IDs and URLs."""
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
            documents.append({"id": doc_id, "url": doc_url})

        return {"documents": documents}

    @router.get("/api/document/{collection}/{doc_id:path}")
    def get_document(collection: str, doc_id: str):
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
    def update_collection(name: str, background_tasks: BackgroundTasks):
        if not store.has_collection(name):
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

        background_tasks.add_task(run_collection_update, name)
        return {"status": "update_started", "collection": name}

    return router
