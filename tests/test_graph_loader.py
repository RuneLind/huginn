"""Tests for knowledge-graph path discovery (M17)."""

from pathlib import Path

import pytest

from main.graph.graph_loader import discover_graph_paths


def _write_graph(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"nodes": [], "edges": []}')
    return path


@pytest.fixture(autouse=True)
def _isolate_discovery(monkeypatch):
    # Clear the recognised env vars and neutralise the auto-glob dirs so the
    # discovery result depends only on what each test sets up (the real private
    # sub-repos would otherwise leak graph files into the result).
    for var in ("KNOWLEDGE_GRAPH_PATH", "JIRA_GRAPH_PATH", "LLM_GRAPH_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("main.graph.graph_loader._AUTO_GLOB_DIRS", ())


class TestDiscoverGraphPaths:
    def test_env_var_path_is_picked_up(self, tmp_path, monkeypatch):
        g = _write_graph(tmp_path / "graph.json")
        monkeypatch.setenv("KNOWLEDGE_GRAPH_PATH", str(g))
        assert discover_graph_paths() == [g]

    def test_nonexistent_env_path_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_GRAPH_PATH", str(tmp_path / "missing.json"))
        assert discover_graph_paths() == []

    def test_extra_paths_appended(self, tmp_path):
        g = _write_graph(tmp_path / "extra.json")
        assert discover_graph_paths(extra_paths=[str(g)]) == [g]

    def test_missing_extra_path_skipped(self, tmp_path):
        assert discover_graph_paths(extra_paths=[str(tmp_path / "nope.json")]) == []

    def test_env_vars_resolved_in_declaration_order(self, tmp_path, monkeypatch):
        gk = _write_graph(tmp_path / "k.json")
        gj = _write_graph(tmp_path / "j.json")
        monkeypatch.setenv("KNOWLEDGE_GRAPH_PATH", str(gk))
        monkeypatch.setenv("JIRA_GRAPH_PATH", str(gj))
        assert discover_graph_paths() == [gk, gj]

    def test_duplicates_removed_order_preserved(self, tmp_path, monkeypatch):
        g1 = _write_graph(tmp_path / "a.json")
        g2 = _write_graph(tmp_path / "b.json")
        monkeypatch.setenv("KNOWLEDGE_GRAPH_PATH", str(g1))
        # g1 supplied both via env and extras → must appear once, env order first.
        paths = discover_graph_paths(extra_paths=[str(g1), str(g2)])
        assert paths == [g1, g2]
