"""Tests for scripts/knowledge_graph/extract_entities_llm.call_ollama —
the entity-extraction wrapper around the shared Ollama transport.

The extractor is manual-only (needs a live Ollama + collection), so these
mocked-transport unit tests stand in for it: they pin the request payload
(temperature=0, num_predict, think:false, format:json, timeout, system prompt)
and the swallow-and-parse contract (dict on success, None on any failure).
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "knowledge_graph"))

import extract_entities_llm  # noqa: E402
from extract_entities_llm import call_ollama, SYSTEM_PROMPT  # noqa: E402


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _resp(payload: dict) -> _FakeResp:
    return _FakeResp(json.dumps(payload).encode("utf-8"))


class TestCallOllamaPayload:
    def test_request_payload_shape(self):
        """Payload must match the pre-consolidation extractor request exactly:
        temperature=0 (deterministic), num_predict=3000, think:false,
        format:json, the system prompt, and the caller's timeout."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data)
            captured["timeout"] = timeout
            return _resp({"message": {"content": '{"entities": [], "relationships": []}'}})

        with patch("main.utils.ollama_cli.urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("model-x", "some document text", "Doc Title", timeout=300)

        data = captured["data"]
        assert data["model"] == "model-x"
        assert data["stream"] is False
        assert data["think"] is False
        assert data["format"] == "json"
        # temperature MUST be 0 — the shared default is 0.2, which would make
        # entity extraction non-deterministic.
        assert data["options"] == {"temperature": 0, "num_predict": 3000}
        assert data["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert data["messages"][1]["role"] == "user"
        assert "Doc Title" in data["messages"][1]["content"]
        assert captured["timeout"] == 300

    def test_text_truncated_to_3000_chars(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data)
            return _resp({"message": {"content": "{}"}})

        with patch("main.utils.ollama_cli.urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("m", "x" * 5000, "T")

        # user content wraps the (truncated) body in a template; assert the body
        # was clipped to exactly 3000 consecutive x's, not the raw 5000.
        content = captured["data"]["messages"][1]["content"]
        assert "x" * 3000 in content
        assert "x" * 3001 not in content


class TestCallOllamaReturn:
    def test_returns_parsed_dict(self):
        graph = {"entities": [{"name": "FAISS", "type": "Technology"}], "relationships": []}
        with patch("main.utils.ollama_cli.urllib.request.urlopen",
                   return_value=_resp({"message": {"content": json.dumps(graph)}})):
            assert call_ollama("m", "text", "T") == graph

    def test_strips_markdown_fences(self):
        fenced = "```json\n{\"entities\": [], \"relationships\": []}\n```"
        with patch("main.utils.ollama_cli.urllib.request.urlopen",
                   return_value=_resp({"message": {"content": fenced}})):
            assert call_ollama("m", "text", "T") == {"entities": [], "relationships": []}

    def test_empty_content_returns_none(self):
        with patch("main.utils.ollama_cli.urllib.request.urlopen",
                   return_value=_resp({"message": {"content": ""}})):
            assert call_ollama("m", "text", "T") is None


class TestCallOllamaErrorPaths:
    def test_transport_error_returns_none(self):
        import urllib.error

        with patch("main.utils.ollama_cli.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            assert call_ollama("m", "text", "T") is None

    def test_response_error_field_returns_none(self):
        """A response ``error`` field makes the shared transport raise
        RuntimeError; the wrapper must swallow it to None, as before."""
        with patch("main.utils.ollama_cli.urllib.request.urlopen",
                   return_value=_resp({"error": "model not found"})):
            assert call_ollama("m", "text", "T") is None

    def test_bad_model_json_returns_none(self):
        """Content that is not valid JSON (post-fence-strip) returns None —
        the wrapper's own json.loads guard, not the transport's."""
        with patch("main.utils.ollama_cli.urllib.request.urlopen",
                   return_value=_resp({"message": {"content": "not json at all"}})):
            assert call_ollama("m", "text", "T") is None


class TestLimitDryRunPath:
    def test_limit_run_end_to_end_with_mocked_transport(self, tmp_path):
        """Drive main() with --limit against a tmp collection and a mocked
        Ollama transport (no live server). Exercises the --limit slicing,
        the truncated source_stamp, and per-doc call_ollama routing through
        the shared transport — all without touching the network."""
        docs_dir = tmp_path / "coll" / "documents"
        docs_dir.mkdir(parents=True)
        for i in range(3):
            (docs_dir / f"doc{i}.json").write_text(json.dumps({
                "id": f"doc{i}",
                "text": "FAISS is a vector search library. " * 10,
                "metadata": {"title": f"Doc {i}"},
            }), encoding="utf-8")

        out_path = tmp_path / "graph.json"

        graph = json.dumps({
            "entities": [{"name": "FAISS", "type": "Technology"}],
            "relationships": [],
        })

        # Both the /api/tags liveness ping (a direct urlopen in the extractor)
        # and the /api/chat call (via the shared transport) hit the SAME
        # urllib.request.urlopen object, so one dispatching mock serves both.
        def fake_urlopen(target, timeout=None):
            url = target if isinstance(target, str) else target.full_url
            if url.endswith("/api/tags"):
                return _FakeResp(b"{}")
            return _resp({"message": {"content": graph}})

        argv = [
            "extract_entities_llm.py",
            "--collection", "coll",
            "--data-path", str(tmp_path),
            "--output", str(out_path),
            "--limit", "2",
        ]

        with patch.object(sys, "argv", argv), \
             patch("urllib.request.urlopen", side_effect=fake_urlopen):
            extract_entities_llm.main()

        assert out_path.exists()
        written = json.loads(out_path.read_text(encoding="utf-8"))
        # --limit 2 of 3 docs => truncated stamp reports the processed count so
        # the loader flags the partial graph as stale.
        assert written["source_stamp"]["document_count"] == 2
        # FAISS extracted from both processed docs.
        labels = {n["label"] for n in written["nodes"]}
        assert "FAISS" in labels


class TestExtractionCache:
    """The graph cache is keyed by doc id alone, so a cache written from
    PRE-alias documents replays real names into the graph JSON after a rebuild.
    The privacy policy version gates that — but only for a collection the
    registry is actually armed for."""

    def _write(self, path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_envelope_round_trips_doc_ids_that_start_with_underscore(self, tmp_path):
        cache_path = tmp_path / "g.cache.json"
        entries = {"_internal/notes.md": {"entities": []}, "a.md": {"entities": []}}
        extract_entities_llm.write_extraction_cache(cache_path, entries)
        assert extract_entities_llm.load_extraction_cache(cache_path, aliased=True) == entries

    def test_current_policy_version_is_kept(self, tmp_path):
        cache_path = tmp_path / "g.cache.json"
        extract_entities_llm.write_extraction_cache(cache_path, {"a.md": {"entities": []}})
        assert extract_entities_llm.load_extraction_cache(cache_path, aliased=True)

    def test_legacy_flat_cache_is_discarded_for_an_in_scope_collection(self, tmp_path, capsys):
        cache_path = self._write(tmp_path / "g.cache.json",
                                 {"a.md": {"entities": []}, "b.md": {"entities": []}})
        assert extract_entities_llm.load_extraction_cache(cache_path, aliased=True) == {}
        # off-by-one: the old sentinel-key form counted the sentinel as an entry
        assert "2 stale" in capsys.readouterr().out

    def test_legacy_flat_cache_is_kept_for_an_out_of_scope_collection(self, tmp_path):
        entries = {"a.md": {"entities": []}, "b.md": {"entities": []}}
        cache_path = self._write(tmp_path / "g.cache.json", entries)
        assert extract_entities_llm.load_extraction_cache(cache_path, aliased=False) == entries

    def test_wrong_policy_version_is_discarded(self, tmp_path):
        cache_path = self._write(tmp_path / "g.cache.json",
                                 {"policy_version": 999, "entries": {"a.md": {}}})
        assert extract_entities_llm.load_extraction_cache(cache_path, aliased=True) == {}

    def test_missing_or_unreadable_cache_is_empty(self, tmp_path):
        assert extract_entities_llm.load_extraction_cache(tmp_path / "absent.json", aliased=True) == {}
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        assert extract_entities_llm.load_extraction_cache(broken, aliased=True) == {}


class TestPrivacyArming:
    """The extractor writes a graph JSON straight out of the collection's own
    documents, so it must make the SAME scoping decision the build made.

    Two ways it silently made a different one: it never passed
    `armed_by_manifest`, so a collection kept aliased only by its manifest stamp
    (the scope files having drifted) extracted with no registry and re-derived
    real names into the graph; and a `PrivacyMapMissing` propagated as a
    traceback, which under a nightly wrapper reads as a crash rather than as
    "refused to extract".
    """

    def _collection(self, tmp_path, manifest):
        docs_dir = tmp_path / "coll" / "documents"
        docs_dir.mkdir(parents=True)
        (docs_dir / "doc0.json").write_text(json.dumps({
            "id": "doc0", "text": "FAISS is a vector search library. " * 10,
            "metadata": {"title": "Doc 0"},
        }), encoding="utf-8")
        (tmp_path / "coll" / "manifest.json").write_text(json.dumps(manifest),
                                                         encoding="utf-8")

    def _argv(self, tmp_path):
        return ["extract_entities_llm.py", "--collection", "coll",
                "--data-path", str(tmp_path), "--output", str(tmp_path / "graph.json"),
                "--limit", "1"]

    def _run(self, tmp_path, resolve):
        graph = json.dumps({"entities": [{"name": "FAISS", "type": "Technology"}],
                            "relationships": []})

        def fake_urlopen(target, timeout=None):
            url = target if isinstance(target, str) else target.full_url
            if url.endswith("/api/tags"):
                return _FakeResp(b"{}")
            return _resp({"message": {"content": graph}})

        with patch.object(sys, "argv", self._argv(tmp_path)), \
             patch("urllib.request.urlopen", side_effect=fake_urlopen), \
             patch.object(extract_entities_llm, "resolve_registry", resolve):
            extract_entities_llm.main()

    def test_manifest_privacy_stamp_arms_the_registry(self, tmp_path):
        seen = {}

        def resolve(collection, base_path, **kwargs):
            seen.update(kwargs)
            return None

        self._collection(tmp_path, {"reader": {"basePath": "./data/sources/demo"},
                                    "privacy": {"policy_version": 1, "map_version": 7}})
        self._run(tmp_path, resolve)
        assert seen["armed_by_manifest"] is True

    def test_a_collection_without_a_stamp_is_not_armed_by_the_manifest(self, tmp_path):
        seen = {}

        def resolve(collection, base_path, **kwargs):
            seen.update(kwargs)
            return None

        self._collection(tmp_path, {"reader": {"basePath": "./data/sources/demo"}})
        self._run(tmp_path, resolve)
        assert seen["armed_by_manifest"] is False

    def test_a_missing_alias_map_exits_2_with_one_line(self, tmp_path, capsys):
        def resolve(collection, base_path, **kwargs):
            raise extract_entities_llm.PrivacyMapMissing("no alias map found")

        self._collection(tmp_path, {"reader": {"basePath": "./data/sources/demo"},
                                    "privacy": {"policy_version": 1, "map_version": 7}})
        with pytest.raises(SystemExit) as excinfo:
            self._run(tmp_path, resolve)
        assert excinfo.value.code == 2
        out = capsys.readouterr().out
        assert "no alias map found" in out
        assert not (tmp_path / "graph.json").exists()
