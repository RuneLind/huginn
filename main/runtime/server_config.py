"""Single source of truth for HTTP API server configuration.

Built once in ``knowledge_api_server.main()`` from argparse + environment, then
stored on ``app.state.config`` and read by the routes. Replaces the previous
mechanisms it consolidates:

  1. the ~10 CLI flags copied field-by-field onto ``app.state``,
  2. the env-only ``KNOWLEDGE_GRAPH_PATH`` / ``JIRA_GRAPH_PATH`` graph resolution
     (now folded in via ``discover_graph_paths`` at construction), and
  3. the loose per-ingest-source ``app.state`` attributes generated from the
     ``INGEST_SOURCES`` registry (now a ``ingest_sources`` dict keyed by name).

HARD CONTRACT: every flag name, env var name, default, and precedence is
preserved exactly. ``add_arguments`` / ``from_args`` mirror what
``knowledge_api_server.main()`` used to do inline; the registry loop still drives
the per-source args and their resolution — it just targets this dataclass.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from main.core.trace_store import any_trace_enabled, pointer_mode_enabled
from main.graph.graph_loader import discover_graph_paths
from main.ingest.registry import INGEST_SOURCES

DEFAULT_DATA_PATH = "./data/collections"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8321


@dataclass(frozen=True)
class IngestSourceConfig:
    """Resolved write-root path + target collection for one push-ingest source."""

    path: str | None
    collection: str | None


def _resolve_ingest_sources(path_for, collection_for) -> dict[str, IngestSourceConfig]:
    """Build the per-source config dict from the registry.

    ``path_for(src)`` / ``collection_for(src)`` supply the resolved values — from
    argparse (``from_args``) or straight from the environment (``default``).
    """
    return {
        src.name: IngestSourceConfig(path=path_for(src), collection=collection_for(src))
        for src in INGEST_SOURCES
    }


@dataclass
class ServerConfig:
    """Resolved configuration for one ``knowledge_api_server`` process."""

    collections: list[str]
    data_path: str
    host: str
    port: int
    # source name -> resolved path/collection, populated via the INGEST_SOURCES registry
    ingest_sources: dict[str, IngestSourceConfig]
    # fully resolved knowledge-graph JSON paths (env vars + auto-glob + extras),
    # resolved once here so config is the single owner of graph-path resolution.
    graph_paths: list[Path]
    # boot-time snapshot of the HUGINN_TRACE_* env flags.
    # trace_default mirrors any_trace_enabled() (env forces tracing on);
    # trace_pointer mirrors pointer_mode_enabled() (emit a traceId pointer).
    trace_default: bool
    trace_pointer: bool

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
        ingest_sources = _resolve_ingest_sources(
            path_for=lambda src: getattr(args, src.path_attr),
            collection_for=lambda src: getattr(args, src.collection_attr),
        )
        return cls(
            collections=args.collections,
            data_path=args.data_path,
            host=args.host,
            port=args.port,
            ingest_sources=ingest_sources,
            graph_paths=discover_graph_paths(),
            trace_default=any_trace_enabled(),
            trace_pointer=pointer_mode_enabled(),
        )

    @classmethod
    def default(cls) -> "ServerConfig":
        """Env-only config used before ``main()`` runs (module import / TestClient).

        Mirrors ``add_arguments`` defaults so the module-level ``app`` always has a
        config even when no CLI is parsed. Precedence matches ``from_args``: env
        var when set, else the registry's ``collection_default``.
        """
        ingest_sources = _resolve_ingest_sources(
            path_for=lambda src: os.environ.get(src.path_env),
            collection_for=lambda src: os.environ.get(src.collection_env, src.collection_default),
        )
        return cls(
            collections=[],
            data_path=os.environ.get("HUGINN_DATA_PATH", DEFAULT_DATA_PATH),
            host=DEFAULT_HOST,
            port=DEFAULT_PORT,
            ingest_sources=ingest_sources,
            graph_paths=discover_graph_paths(),
            trace_default=any_trace_enabled(),
            trace_pointer=pointer_mode_enabled(),
        )
