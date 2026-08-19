"""Tests for ServerConfig resolution: flags, env fallbacks, and precedence."""
import argparse
import sys
from dataclasses import replace

import pytest

from main.ingest.registry import INGEST_SOURCES
from main.runtime.server_config import DEFAULT_PORT, ServerConfig


_CONFIG_ENV_VARS = (
    "HUGINN_DATA_PATH",
    *[src.path_env for src in INGEST_SOURCES],
    *[src.collection_env for src in INGEST_SOURCES],
)


def _build(argv, monkeypatch, env=None):
    """Parse ``argv`` through ServerConfig's own arg registration and build a config.

    All config-relevant env vars are cleared first (hermetic against a polluted
    dev shell), then ``env`` (a dict) is applied before ``add_arguments`` so that
    argparse defaults (which capture ``os.environ`` at registration time) see it,
    mirroring how the real process boots.
    """
    for name in _CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    parser = argparse.ArgumentParser()
    ServerConfig.add_arguments(parser)
    return ServerConfig.from_args(parser.parse_args(argv))


def test_explicit_flags_resolve_onto_fields(monkeypatch):
    cfg = _build([
        "--collections", "wiki", "jira-issues",
        "--port", "9999",
        "--host", "0.0.0.0",
        "--data-path", "/data/x",
        "--youtube-transcripts-path", "/yt",
        "--youtube-collection", "yt-coll",
    ], monkeypatch)
    assert cfg.collections == ["wiki", "jira-issues"]
    assert cfg.port == 9999
    assert cfg.host == "0.0.0.0"
    assert cfg.data_path == "/data/x"
    assert cfg.ingest("youtube").path == "/yt"
    assert cfg.ingest("youtube").collection == "yt-coll"


def test_port_and_host_defaults(monkeypatch):
    cfg = _build(["--collections", "wiki"], monkeypatch)
    assert cfg.port == DEFAULT_PORT
    assert cfg.host == "127.0.0.1"
    assert cfg.data_path == "./data/collections"


def test_ingest_collection_default_from_registry(monkeypatch):
    cfg = _build(["--collections", "wiki"], monkeypatch)
    # No flag, no env → each source falls back to its registry collection_default.
    for src in INGEST_SOURCES:
        assert cfg.ingest(src.name).collection == src.collection_default
        assert cfg.ingest(src.name).path is None  # path has no default


def test_env_fallback_when_flag_absent(monkeypatch):
    cfg = _build(
        ["--collections", "wiki"],
        monkeypatch,
        env={
            "JIRA_SOURCES_PATH": "/env/jira",
            "JIRA_COLLECTION": "env-jira-coll",
            "HUGINN_DATA_PATH": "/env/data",
        },
    )
    assert cfg.ingest("jira").path == "/env/jira"
    assert cfg.ingest("jira").collection == "env-jira-coll"
    assert cfg.data_path == "/env/data"


def test_flag_beats_env(monkeypatch):
    cfg = _build(
        ["--collections", "wiki", "--jira-sources-path", "/flag/jira", "--data-path", "/flag/data"],
        monkeypatch,
        env={"JIRA_SOURCES_PATH": "/env/jira", "HUGINN_DATA_PATH": "/env/data"},
    )
    assert cfg.ingest("jira").path == "/flag/jira"  # flag wins
    assert cfg.data_path == "/flag/data"


def test_all_registry_sources_present(monkeypatch):
    cfg = _build(["--collections", "wiki"], monkeypatch)
    assert set(cfg.ingest_sources) == {src.name for src in INGEST_SOURCES}


class TestDefault:
    """ServerConfig.default() — env-only boot for the module-level app."""

    def _clear(self, monkeypatch):
        for name in _CONFIG_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

    def test_matches_cli_defaults(self, monkeypatch):
        """default() must resolve identically to a bare CLI boot (minus collections)."""
        self._clear(monkeypatch)
        cli = _build(["--collections", "wiki"], monkeypatch)
        dflt = ServerConfig.default()
        assert dflt.collections == []
        assert dflt.data_path == cli.data_path
        assert dflt.host == cli.host
        assert dflt.port == cli.port
        # Whole-dataclass compare: guards that default() blanks only collections.
        assert dflt == replace(cli, collections=[])

    def test_matches_cli_boot_with_env_fallbacks(self, monkeypatch):
        """Same equality with env vars set — proves the env path flows through too."""
        cli = _build(
            ["--collections", "wiki"],
            monkeypatch,
            env={
                "HUGINN_DATA_PATH": "/env/data",
                "TIKTOK_SOURCES_PATH": "/env/tiktok",
                "YOUTUBE_COLLECTION": "env-yt-coll",
            },
        )
        assert cli.data_path == "/env/data"  # guard: the env really was in play
        assert cli.ingest("tiktok").path == "/env/tiktok"
        assert cli.ingest("youtube").collection == "env-yt-coll"
        assert ServerConfig.default() == replace(cli, collections=[])

    def test_default_survives_a_new_required_flag(self, monkeypatch):
        """A flag required for CLI boot must not make default() SystemExit at import.

        knowledge_api_server calls default() at module import, so any argv
        rejection there is an ImportError for every consumer.
        """
        self._clear(monkeypatch)
        cfg = _RequiredTenantConfig.default()
        assert cfg.collections == []
        assert cfg.data_path == "./data/collections"
        # Guard: --tenant really is required on the CLI surface being relaxed.
        cli_parser = argparse.ArgumentParser(add_help=False)
        _RequiredTenantConfig.add_arguments(cli_parser)
        with pytest.raises(SystemExit):
            cli_parser.parse_args([])

    def test_ignores_process_argv(self, monkeypatch):
        """default() parses its own empty argv, so a junk sys.argv cannot reach it."""
        self._clear(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["prog", "--bogus", "x"])
        assert ServerConfig.default().collections == []

    def test_env_fallbacks_apply(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("HUGINN_DATA_PATH", "/env/data")
        monkeypatch.setenv("TIKTOK_SOURCES_PATH", "/env/tiktok")
        monkeypatch.setenv("TIKTOK_COLLECTION", "env-tiktok-coll")
        cfg = ServerConfig.default()
        assert cfg.data_path == "/env/data"
        assert cfg.ingest("tiktok").path == "/env/tiktok"
        assert cfg.ingest("tiktok").collection == "env-tiktok-coll"


class _RequiredTenantConfig(ServerConfig):
    """Stand-in for a future flag that is mandatory on the CLI but not for default()."""

    @staticmethod
    def add_arguments(parser, *, collections_required: bool = True) -> None:
        ServerConfig.add_arguments(parser, collections_required=collections_required)
        parser.add_argument("--tenant", required=collections_required, default="acme")
