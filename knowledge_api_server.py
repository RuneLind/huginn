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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from main.routes.collections import router as collections_router
from main.routes.graph import router as graph_router
from main.routes.ingest import router as ingest_router
from main.routes.notion import router as notion_router
from main.routes.search import router as search_router
from main.runtime.knowledge_store import get_store
from main.runtime.server_config import ServerConfig
from main.utils.logger import setup_root_logger

setup_root_logger()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    config: ServerConfig = app.state.config
    store.load_collections(
        config.collections,
        data_path=config.data_path,
        extra_graph_paths=config.graph_paths,
    )
    yield


app = FastAPI(title="Knowledge API", lifespan=lifespan)
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

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
