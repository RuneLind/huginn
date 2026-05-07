"""Notion page route — live API fetch with local-index fallback."""
import json
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Query

from main.runtime.knowledge_store import KnowledgeStore, get_store
from main.sources.notion.notion_document_reader import NotionDocumentReader

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_relation_titles(notion, page):
    """Resolve relation property IDs to titles for rendering."""
    for prop in page.get("properties", {}).values():
        if prop.get("type") != "relation":
            continue
        for rel in prop.get("relation", []):
            if "id" in rel and "title" not in rel:
                try:
                    related = notion.pages.retrieve(page_id=rel["id"])
                    rel["title"] = NotionDocumentReader.get_page_title(related)
                except Exception:
                    pass


def _fetch_all_blocks(notion, block_id, depth=0):
    """Recursively fetch all blocks for a page."""
    if depth > 5:
        return []
    blocks = []
    cursor = None
    while True:
        response = notion.blocks.children.list(block_id=block_id, start_cursor=cursor)
        for block in response.get("results", []):
            blocks.append(block)
            if block.get("has_children"):
                block["children"] = _fetch_all_blocks(notion, block["id"], depth + 1)
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return blocks


def _find_local_page_by_notion_id(store: KnowledgeStore, notion_id: str):
    """Look up locally indexed content by Notion page ID."""
    normalized = notion_id.replace("-", "")
    entry = store.notion_id_to_doc.get(normalized)
    if not entry:
        return None
    try:
        doc = json.loads(store.disk_persister.read_text_file(entry["doc_path"]))
        return {
            "id": notion_id,
            "title": entry["doc_path"].rsplit("/", 1)[-1].replace(".json", ""),
            "url": entry["url"],
            "content": doc.get("text", ""),
            "source": "local_index",
        }
    except Exception:
        return None


@router.get("/api/notion/page/{notion_id}")
def get_notion_page(
    notion_id: str,
    source: str = Query("auto", description="Source: auto (live→local fallback), live (API only), local (index only)"),
    store: KnowledgeStore = Depends(get_store),
):
    """Fetch page content from Notion API and/or local index."""
    if source not in ("auto", "live", "local"):
        raise HTTPException(status_code=400, detail=f"Invalid source '{source}'. Must be one of: auto, live, local")
    if source == "local":
        local_content = _find_local_page_by_notion_id(store, notion_id)
        if local_content:
            return local_content
        raise HTTPException(status_code=404, detail=f"Page '{notion_id}' not found in local index")

    local_content = _find_local_page_by_notion_id(store, notion_id) if source == "auto" else None

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        if source == "live":
            raise HTTPException(status_code=503, detail="NOTION_TOKEN not configured")
        if local_content:
            return local_content
        raise HTTPException(status_code=503, detail="NOTION_TOKEN not configured and no local content found")

    try:
        from notion_client import Client
        from main.sources.notion.notion_block_to_markdown import convert_blocks_to_markdown, extract_page_properties

        notion = Client(auth=token)
        page = notion.pages.retrieve(page_id=notion_id)
        _resolve_relation_titles(notion, page)

        all_blocks = _fetch_all_blocks(notion, notion_id)
        properties_md = extract_page_properties(page.get("properties", {}))
        blocks_md = convert_blocks_to_markdown(all_blocks)

        content_parts = [p for p in [properties_md, blocks_md] if p]
        markdown = "\n\n".join(content_parts)

        return {
            "id": notion_id,
            "title": NotionDocumentReader.get_page_title(page),
            "url": page.get("url", ""),
            "lastEdited": page.get("last_edited_time", ""),
            "content": markdown,
        }
    except Exception as e:
        logger.error(f"Notion API error for page {notion_id}: {e}")
        if source == "live":
            raise HTTPException(status_code=502, detail=f"Notion API error: {e}")
        if local_content:
            return local_content
        raise HTTPException(status_code=502, detail=f"Notion API error: {e}")
