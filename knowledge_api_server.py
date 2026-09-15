#!/usr/bin/env python3
"""
Knowledge API Server — long-running HTTP API for vector search.

Loads embedding model and FAISS indexes once at startup, serves search
results via HTTP. Designed for low-latency responses (<50ms after warmup).

Usage:
    uv run knowledge_api_server.py --collections my-notion --port 8321
"""
import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from main.routes.collections import router as collections_router
from main.routes.graph import router as graph_router
from main.routes.ingest import router as ingest_router
from main.routes.notion import router as notion_router
from main.routes.search import router as search_router
from main.runtime.knowledge_store import KnowledgeStore, get_store
from main.runtime.server_config import ServerConfig
from main.utils.logger import setup_root_logger

setup_root_logger()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    config: ServerConfig = app.state.config
    store.load_collections(config.collections, data_path=config.data_path)
    yield


app = FastAPI(title="Knowledge API", lifespan=lifespan)
# Deliberately a loose attribute, not a ServerConfig field: a derived static
# path (repo root), not runtime configuration.
app.state.huginn_root = Path(__file__).parent
# A default (env-only) config so the module-level ``app`` is usable before main()
# runs — e.g. under TestClient. Replaced with the fully-resolved one in main().
app.state.config = ServerConfig.default()

# CORS for Chrome extension and local dev access
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|http://localhost(:\d+)?)$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    store = get_store()
    return {
        "status": "ok",
        "collections": store.collection_names(),
        "totalEmbeddings": store.total_embeddings(),
    }


@app.get("/ready")
def ready(store: KnowledgeStore = Depends(get_store)):
    """503 unless every ``--collections`` entry is served with at least one chunk.

    ``/health`` answers 200 even when a collection failed to load (the store
    logs and skips it), so it cannot gate a readiness probe.
    """
    sizes = store.collection_sizes()
    requested = app.state.config.collections
    missing = [c for c in requested if c not in sizes]
    empty = [c for c in requested if sizes.get(c) == 0]
    ok = bool(requested) and not missing and not empty
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not_ready", "collections": sizes,
                 "missing": missing, "empty": empty},
    )


app.include_router(search_router)
app.include_router(graph_router)
app.include_router(notion_router)
app.include_router(collections_router)
app.include_router(ingest_router)


def main():
    ap = argparse.ArgumentParser(description="Knowledge API Server")
    ServerConfig.add_arguments(ap)
    args = ap.parse_args()

    config = ServerConfig.from_args(args)
    app.state.config = config

    # access_log=False: the access line carries the raw `?q=` before the privacy
    # seam runs, and start.sh redirects stdout into logs/ — a typed real name
    # would persist there verbatim.
    uvicorn.run(app, host=config.host, port=config.port, access_log=False)


if __name__ == "__main__":
    main()
