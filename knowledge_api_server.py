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
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from main.ingest.registry import INGEST_SOURCES
from main.routes.collections import router as collections_router
from main.routes.graph import router as graph_router
from main.routes.ingest import router as ingest_router
from main.routes.notion import router as notion_router
from main.routes.search import router as search_router
from main.runtime.knowledge_store import get_store
from main.utils.logger import setup_root_logger

setup_root_logger()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    store.load_collections(app.state.collection_names, data_path=app.state.data_path)
    yield


app = FastAPI(title="Knowledge API", lifespan=lifespan)
app.state.huginn_root = Path(__file__).parent

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
    ap.add_argument(
        "--collections", nargs="+", required=True,
        help="Collections to load (e.g., my-notion)",
    )
    ap.add_argument(
        "--data-path", default=os.environ.get("HUGINN_DATA_PATH", "./data/collections"),
        help="Base path for collection data (default: ./data/collections)",
    )
    ap.add_argument("--port", type=int, default=8321, help="Port to listen on")
    ap.add_argument("--host", default="127.0.0.1", help="Host to bind to")

    # Per push-ingest source: --*-sources-path/--*-collection args from the registry.
    for src in INGEST_SOURCES:
        ap.add_argument(
            src.path_arg,
            default=os.environ.get(src.path_env),
            help=src.path_help,
        )
        ap.add_argument(
            src.collection_arg,
            default=os.environ.get(src.collection_env, src.collection_default),
            help=src.collection_help,
        )
    args = ap.parse_args()

    app.state.data_path = args.data_path
    app.state.collection_names = args.collections

    # Mirror each source's resolved path/collection onto app.state for the routes.
    for src in INGEST_SOURCES:
        setattr(app.state, src.path_attr, getattr(args, src.path_attr))
        setattr(app.state, src.collection_attr, getattr(args, src.collection_attr))

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
