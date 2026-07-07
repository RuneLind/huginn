"""Single source of truth for HTTP API server configuration.

Built once in ``knowledge_api_server.main()`` from argparse + environment, then
stored on ``app.state.config`` and read by the routes. Replaces the previous
mechanism of copying ~10 CLI flags field-by-field onto ``app.state`` and reading
them back via scattered ``getattr`` — including the loose per-ingest-source
attributes generated from the ``INGEST_SOURCES`` registry (now an
``ingest_sources`` dict keyed by source name).

HARD CONTRACT: every flag name, env var name, default, and precedence is
preserved exactly. ``add_arguments`` / ``from_args`` mirror what
``knowledge_api_server.main()`` used to do inline; the registry loop still drives
the per-source args and their resolution — it just targets this dataclass.

Deliberately NOT in here: knowledge-graph path resolution (KNOWLEDGE_GRAPH_PATH /
JIRA_GRAPH_PATH env vars + auto-glob) stays owned by ``main/graph/graph_loader.py``,
which the KnowledgeStore invokes at load/reload time; a snapshot here would be a
stale pass-through. Likewise the HUGINN_TRACE_* flags are read live per request
by the search route (``main/core/trace_store.py``).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from main.ingest.registry import INGEST_SOURCES

DEFAULT_DATA_PATH = "./data/collections"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8321


@dataclass(frozen=True)
class IngestSourceConfig:
    """Resolved write-root path + target collection for one push-ingest source."""

    path: str | None
    collection: str | None


@dataclass
class ServerConfig:
    """Resolved configuration for one ``knowledge_api_server`` process."""

    collections: list[str]
    data_path: str
    host: str
    port: int
    # source name -> resolved path/collection, populated via the INGEST_SOURCES registry
    ingest_sources: dict[str, IngestSourceConfig]

    def ingest(self, name: str) -> IngestSourceConfig:
        """Resolved config for a registered ingest source (KeyError if unknown)."""
        return self.ingest_sources[name]

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Register every server CLI flag on ``parser`` (exact names/defaults preserved)."""
        parser.add_argument(
            "--collections", nargs="+", required=True,
            help="Collections to load (e.g., my-notion)",
        )
        parser.add_argument(
            "--data-path", default=os.environ.get("HUGINN_DATA_PATH", DEFAULT_DATA_PATH),
            help="Base path for collection data (default: ./data/collections)",
        )
        parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
        parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")

        # Per push-ingest source: --*-sources-path/--*-collection args from the registry.
        for src in INGEST_SOURCES:
            parser.add_argument(
                src.path_arg,
                default=os.environ.get(src.path_env),
                help=src.path_help,
            )
            parser.add_argument(
                src.collection_arg,
                default=os.environ.get(src.collection_env, src.collection_default),
                help=src.collection_help,
            )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ServerConfig":
        """Build a config from a parsed argparse namespace (see ``add_arguments``)."""
        ingest_sources = {
            src.name: IngestSourceConfig(
                path=getattr(args, src.path_attr),
                collection=getattr(args, src.collection_attr),
            )
            for src in INGEST_SOURCES
        }
        return cls(
            collections=args.collections,
            data_path=args.data_path,
            host=args.host,
            port=args.port,
            ingest_sources=ingest_sources,
        )

    @classmethod
    def default(cls) -> "ServerConfig":
        """Env-only config used before ``main()`` runs (module import / TestClient).

        Derived from the very same ``add_arguments`` registrations a CLI boot
        uses — no duplicated default/env-fallback literals to drift — with
        ``collections=[]`` standing in for the required ``--collections`` flag.
        """
        parser = argparse.ArgumentParser(add_help=False)
        cls.add_arguments(parser)
        defaults = {action.dest: action.default for action in parser._actions}
        defaults["collections"] = []
        return cls.from_args(argparse.Namespace(**defaults))
