"""Tests for knowledge-graph path discovery (M17)."""

import json
import logging
from pathlib import Path

import pytest

from main.graph.graph_loader import (
    check_graph_staleness,
    discover_graph_paths,
    resolve_graph_output_path,
)


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

    def test_set_but_missing_env_path_warns(self, tmp_path, monkeypatch, caplog):
        # An explicitly-set env path that doesn't exist should warn (not silently
        # vanish), while still being dropped from the result.
        monkeypatch.setenv("KNOWLEDGE_GRAPH_PATH", str(tmp_path / "missing.json"))
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            assert discover_graph_paths() == []
        assert any("KNOWLEDGE_GRAPH_PATH" in r.message for r in caplog.records)

    def test_unset_env_path_does_not_warn(self, caplog):
        # Fresh clone: env vars unset → no warning spam.
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            assert discover_graph_paths() == []
        assert caplog.records == []


class TestResolveGraphOutputPath:
    def test_explicit_output_wins(self, tmp_path):
        out = tmp_path / "custom.json"
        assert resolve_graph_output_path("anything", str(out)) == out

    def test_explicit_output_wins_even_with_routing(self, tmp_path, monkeypatch):
        rdir = tmp_path / "nav" / "scripts" / "knowledge_graph"
        rdir.mkdir(parents=True)
        (rdir / "graph_routing.json").write_text(json.dumps({"collections": ["jira"]}))
        monkeypatch.setattr("main.graph.graph_loader._AUTO_GLOB_DIRS", (str(rdir),))
        out = tmp_path / "custom.json"
        assert resolve_graph_output_path("jira", str(out)) == out

    def test_routing_config_maps_collection_to_its_dir(self, tmp_path, monkeypatch):
        rdir = tmp_path / "nav" / "scripts" / "knowledge_graph"
        rdir.mkdir(parents=True)
        (rdir / "graph_routing.json").write_text(json.dumps({"collections": ["jira-issues"]}))
        monkeypatch.setattr("main.graph.graph_loader._AUTO_GLOB_DIRS", (str(rdir),))
        assert resolve_graph_output_path("jira-issues") == rdir / "jira-issues_llm_graph.json"

    def test_default_dir_claims_unlisted_collection(self, tmp_path, monkeypatch):
        nav = tmp_path / "nav"
        nav.mkdir()
        (nav / "graph_routing.json").write_text(json.dumps({"collections": ["jira-issues"]}))
        jarvis = tmp_path / "jarvis"
        jarvis.mkdir()
        (jarvis / "graph_routing.json").write_text(json.dumps({"default": True}))
        monkeypatch.setattr(
            "main.graph.graph_loader._AUTO_GLOB_DIRS", (str(nav), str(jarvis))
        )
        # Listed collection → nav; unlisted → the default (jarvis) dir.
        assert resolve_graph_output_path("jira-issues") == nav / "jira-issues_llm_graph.json"
        assert resolve_graph_output_path("youtube") == jarvis / "youtube_llm_graph.json"

    def test_explicit_listing_wins_over_default(self, tmp_path, monkeypatch):
        nav = tmp_path / "nav"
        nav.mkdir()
        (nav / "graph_routing.json").write_text(json.dumps({"collections": ["jira-issues"]}))
        jarvis = tmp_path / "jarvis"
        jarvis.mkdir()
        (jarvis / "graph_routing.json").write_text(json.dumps({"default": True}))
        monkeypatch.setattr(
            "main.graph.graph_loader._AUTO_GLOB_DIRS", (str(nav), str(jarvis))
        )
        assert resolve_graph_output_path("jira-issues") == nav / "jira-issues_llm_graph.json"

    def test_no_routing_raises(self, tmp_path, monkeypatch):
        # Fresh public clone: no routing configs anywhere → clear failure telling
        # the user to pass --output. No implicit fallback.
        monkeypatch.setattr(
            "main.graph.graph_loader._AUTO_GLOB_DIRS", (str(tmp_path / "absent"),)
        )
        with pytest.raises(ValueError, match="--output"):
            resolve_graph_output_path("some-collection")

    def test_unreadable_routing_config_skipped(self, tmp_path, monkeypatch):
        rdir = tmp_path / "nav"
        rdir.mkdir()
        (rdir / "graph_routing.json").write_text("{ not json")
        monkeypatch.setattr("main.graph.graph_loader._AUTO_GLOB_DIRS", (str(rdir),))
        # Corrupt config is ignored, and with no other route we fail cleanly.
        with pytest.raises(ValueError):
            resolve_graph_output_path("x")

    def test_custom_filename_overrides_default_name(self, tmp_path, monkeypatch):
        # The Jira extractor writes jira_graph.json instead of *_llm_graph.json.
        rdir = tmp_path / "nav"
        rdir.mkdir()
        (rdir / "graph_routing.json").write_text(json.dumps({"collections": ["jira-issues"]}))
        monkeypatch.setattr("main.graph.graph_loader._AUTO_GLOB_DIRS", (str(rdir),))
        out = resolve_graph_output_path("jira-issues", filename="jira_graph.json")
        assert out == rdir / "jira_graph.json"

    def test_second_default_warns_first_wins(self, tmp_path, monkeypatch, caplog):
        # Two configs both claiming "default": true is ambiguous — resolution
        # follows _AUTO_GLOB_DIRS order, but the ambiguity must be surfaced.
        a = tmp_path / "a"
        a.mkdir()
        (a / "graph_routing.json").write_text(json.dumps({"default": True}))
        b = tmp_path / "b"
        b.mkdir()
        (b / "graph_routing.json").write_text(json.dumps({"default": True}))
        monkeypatch.setattr("main.graph.graph_loader._AUTO_GLOB_DIRS", (str(a), str(b)))
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            out = resolve_graph_output_path("x")
        assert out == a / "x_llm_graph.json"
        assert any("default" in r.message for r in caplog.records)


def _stamped_graph(stamp):
    """Return a (path, parsed-graph) pair as check_graph_staleness expects."""
    payload = {"nodes": [], "edges": []}
    if stamp is not None:
        payload["source_stamp"] = stamp
    return (Path("g.json"), payload)


def _write_manifest(data_path: Path, collection: str, **fields):
    d = data_path / collection
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(fields))


class TestCheckGraphStaleness:
    def test_unstamped_graph_never_warns(self, tmp_path, caplog):
        g = _stamped_graph(stamp=None)
        data_path = tmp_path / "collections"
        _write_manifest(data_path, "c", numberOfDocuments=10)
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            check_graph_staleness([g], str(data_path))
        assert caplog.records == []

    def test_matching_stamp_does_not_warn(self, tmp_path, caplog):
        g = _stamped_graph(
            {"collection": "c", "document_count": 10,
             "last_modified_document_time": "2026-01-01T00:00:00"},
        )
        data_path = tmp_path / "collections"
        _write_manifest(data_path, "c", numberOfDocuments=10,
                        lastModifiedDocumentTime="2026-01-01T00:00:00")
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            check_graph_staleness([g], str(data_path))
        assert caplog.records == []

    def test_document_count_divergence_warns(self, tmp_path, caplog):
        g = _stamped_graph({"collection": "c", "document_count": 10})
        data_path = tmp_path / "collections"
        _write_manifest(data_path, "c", numberOfDocuments=42)
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            check_graph_staleness([g], str(data_path))
        assert any("stale" in r.message and "c" in r.message for r in caplog.records)

    def test_string_vs_int_count_does_not_warn(self, tmp_path, caplog):
        # "10" vs 10 is a type quirk, not staleness — must not warn forever.
        g = _stamped_graph({"collection": "c", "document_count": "10"})
        data_path = tmp_path / "collections"
        _write_manifest(data_path, "c", numberOfDocuments=10)
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            check_graph_staleness([g], str(data_path))
        assert caplog.records == []

    def test_content_change_divergence_warns(self, tmp_path, caplog):
        g = _stamped_graph(
            {"collection": "c", "document_count": 10,
             "last_modified_document_time": "2026-01-01T00:00:00"},
        )
        data_path = tmp_path / "collections"
        _write_manifest(data_path, "c", numberOfDocuments=10,
                        lastModifiedDocumentTime="2026-06-01T00:00:00")
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            check_graph_staleness([g], str(data_path))
        assert any("documents changed" in r.message for r in caplog.records)

    def test_updated_time_only_bump_does_not_warn(self, tmp_path, caplog):
        # A no-op reindex rewrites the manifest's updatedTime (a daily launchd job
        # does exactly that); staleness must key on lastModifiedDocumentTime — real
        # content change — or the warning would fire permanently on every start.
        g = _stamped_graph(
            {"collection": "c", "document_count": 10,
             "last_modified_document_time": "2026-01-01T00:00:00"},
        )
        data_path = tmp_path / "collections"
        _write_manifest(data_path, "c", numberOfDocuments=10,
                        lastModifiedDocumentTime="2026-01-01T00:00:00",
                        updatedTime="2026-07-07T09:00:00")
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            check_graph_staleness([g], str(data_path))
        assert caplog.records == []

    def test_missing_manifest_skipped(self, tmp_path, caplog):
        # Collection not present in this deployment → cannot compare, stay quiet.
        g = _stamped_graph({"collection": "c", "document_count": 10})
        with caplog.at_level(logging.WARNING, logger="main.graph.graph_loader"):
            check_graph_staleness([g], str(tmp_path / "collections"))
        assert caplog.records == []


def _load_extractor_module():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "knowledge_graph" / "extract_entities_llm.py"
    spec = importlib.util.spec_from_file_location("extract_entities_llm_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBuildSourceStamp:
    """The extractor's source_stamp must stay comparable to the manifest."""

    def test_full_run_stamps_manifest_count_and_content_time(self, tmp_path):
        _write_manifest(tmp_path, "c", numberOfDocuments=42,
                        lastModifiedDocumentTime="2026-01-01T00:00:00",
                        updatedTime="2026-07-07T09:00:00")
        stamp = _load_extractor_module().build_source_stamp("c", str(tmp_path))
        assert stamp == {
            "collection": "c",
            "document_count": 42,
            "last_modified_document_time": "2026-01-01T00:00:00",
        }
        # updatedTime is deliberately not stamped — it moves on no-op reindexes.
        assert "updated_time" not in stamp

    def test_unreadable_manifest_omits_document_count(self, tmp_path):
        # A raw file count is not comparable to a later manifest's post-exclude
        # numberOfDocuments — omit rather than stamp an incomparable number.
        stamp = _load_extractor_module().build_source_stamp("c", str(tmp_path))
        assert stamp == {"collection": "c"}

    def test_limited_run_stamps_processed_count(self, tmp_path):
        # `--limit 20` writes a partial graph; stamping the manifest's full count
        # would hide that forever. The processed count must win.
        _write_manifest(tmp_path, "c", numberOfDocuments=1000,
                        lastModifiedDocumentTime="2026-01-01T00:00:00")
        stamp = _load_extractor_module().build_source_stamp("c", str(tmp_path), processed_doc_count=20)
        assert stamp["document_count"] == 20
