import json
import os
import tempfile

import pytest

from main.core.contextual_prefix import ChunkPrefixer, ContextualCache, make_backend
from main.core.contextual_prefix.backends import BackendSpec
from main.core.contextual_prefix.backends.echo import EchoBackend
from main.core.contextual_prefix.backends.ollama_backend import _parse_prefix_array
from main.core.contextual_prefix.cache import chunk_fingerprint
from main.core.contextual_prefix.chunk_prefixer import MIN_CHUNK_CHARS_FOR_PREFIX


# ---------- BackendSpec.parse ----------

def test_backend_spec_parses_none():
    assert BackendSpec.parse("none").kind == "none"
    assert BackendSpec.parse("").kind == "none"
    assert BackendSpec.parse(None).kind == "none"  # type: ignore[arg-type]


def test_backend_spec_parses_ollama_with_colon_model():
    spec = BackendSpec.parse("ollama:qwen3.6:35b-a3b-nvfp4")
    assert spec.kind == "ollama"
    assert spec.model == "qwen3.6:35b-a3b-nvfp4"
    assert spec.model_id == "ollama:qwen3.6:35b-a3b-nvfp4"


def test_backend_spec_parses_claude_code():
    spec = BackendSpec.parse("claude-code:claude-haiku-4-5")
    assert spec.kind == "claude-code"
    assert spec.model == "claude-haiku-4-5"


def test_make_backend_returns_none_for_none_spec():
    assert make_backend("none") is None
    assert make_backend(BackendSpec(kind="none", model="")) is None


def test_make_backend_returns_echo_instance():
    backend = make_backend("echo:test")
    assert isinstance(backend, EchoBackend)
    assert backend.model_id == "echo:test"


def test_make_backend_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_backend("voodoo:42")


def test_make_backend_requires_model_for_real_backends():
    with pytest.raises(ValueError):
        make_backend("ollama")
    with pytest.raises(ValueError):
        make_backend("claude-code")


# ---------- ContextualCache ----------

def test_cache_get_returns_none_when_empty():
    with tempfile.TemporaryDirectory() as td:
        cache = ContextualCache(os.path.join(td, "cache.json"))
        assert cache.get("doc1", "some chunk text", "echo:v1") is None


def test_cache_put_then_get_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        cache = ContextualCache(os.path.join(td, "cache.json"))
        cache.put("doc1", "chunk A text", "echo:v1", "anchor A")
        assert cache.get("doc1", "chunk A text", "echo:v1") == "anchor A"
        # Different doc with same chunk text gets a separate entry.
        assert cache.get("doc2", "chunk A text", "echo:v1") is None
        # Different model is a different key.
        assert cache.get("doc1", "chunk A text", "echo:v2") is None


def test_cache_persists_across_instances():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cache.json")
        c1 = ContextualCache(path)
        c1.put("doc1", "chunk", "m1", "prefix")
        c1.flush()

        c2 = ContextualCache(path)
        assert c2.get("doc1", "chunk", "m1") == "prefix"


def test_cache_flush_is_atomic_via_rename():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cache.json")
        cache = ContextualCache(path)
        cache.put("doc", "chunk", "m1", "prefix")
        cache.flush()
        # No leftover tmp file
        assert not os.path.exists(path + ".tmp")
        # File is valid JSON with the expected shape
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["entries"]) == 1


def test_cache_flush_is_noop_when_clean():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cache.json")
        cache = ContextualCache(path)
        cache.flush()
        assert not os.path.exists(path)


def test_chunk_fingerprint_changes_with_doc_id_and_text():
    a = chunk_fingerprint("doc1", "text")
    b = chunk_fingerprint("doc2", "text")
    c = chunk_fingerprint("doc1", "different text")
    assert len({a, b, c}) == 3


# ---------- ChunkPrefixer ----------

def _converted_doc(chunk_texts, doc_id="docX", doc_text=None):
    return {
        "id": doc_id,
        "url": f"https://example/{doc_id}",
        "modifiedTime": "2026-05-16T00:00:00+00:00",
        "text": doc_text or " ".join(chunk_texts),
        "chunks": [{"indexedData": t} for t in chunk_texts],
    }


def test_prefixer_prepends_prefix_to_long_chunks():
    long_chunk = "x" * (MIN_CHUNK_CHARS_FOR_PREFIX + 50)
    doc = _converted_doc([long_chunk])

    with tempfile.TemporaryDirectory() as td:
        cache = ContextualCache(os.path.join(td, "cache.json"))
        prefixer = ChunkPrefixer(generator=EchoBackend(), cache=cache)
        prefixer.prefix_document(doc)

    assert doc["chunks"][0]["contextualPrefix"].startswith("[echo prefix for chunk 1")
    assert doc["chunks"][0]["indexedData"].startswith("[echo prefix for chunk 1")
    assert doc["chunks"][0]["indexedData"].endswith(long_chunk)


def test_prefixer_skips_short_chunks():
    short_chunk = "a small breadcrumb"  # < MIN_CHUNK_CHARS_FOR_PREFIX
    long_chunk = "x" * (MIN_CHUNK_CHARS_FOR_PREFIX + 10)
    doc = _converted_doc([short_chunk, long_chunk])

    with tempfile.TemporaryDirectory() as td:
        cache = ContextualCache(os.path.join(td, "cache.json"))
        ChunkPrefixer(generator=EchoBackend(), cache=cache).prefix_document(doc)

    assert "contextualPrefix" not in doc["chunks"][0]
    assert doc["chunks"][0]["indexedData"] == short_chunk

    assert doc["chunks"][1]["contextualPrefix"].startswith("[echo prefix")
    assert doc["chunks"][1]["indexedData"].endswith(long_chunk)


def test_prefixer_uses_cache_on_second_run():
    long_chunk = "x" * (MIN_CHUNK_CHARS_FOR_PREFIX + 10)

    class CountingBackend:
        model_id = "echo:counting"

        def __init__(self):
            self.calls = 0

        def generate(self, document_text, chunks):
            self.calls += 1
            return [f"prefix-{i}" for i in range(len(chunks))]

    backend = CountingBackend()
    with tempfile.TemporaryDirectory() as td:
        cache_path = os.path.join(td, "cache.json")

        doc1 = _converted_doc([long_chunk])
        ChunkPrefixer(backend, ContextualCache(cache_path)).prefix_documents([doc1])
        assert backend.calls == 1

        # Cache file written; second run should hit it.
        doc2 = _converted_doc([long_chunk])
        ChunkPrefixer(backend, ContextualCache(cache_path)).prefix_documents([doc2])
        assert backend.calls == 1
        assert doc2["chunks"][0]["contextualPrefix"] == "prefix-0"


def test_prefixer_continues_when_backend_returns_empty():
    """A backend failure must not crash the pipeline — chunks just stay un-prefixed."""
    class FailingBackend:
        model_id = "echo:failing"

        def generate(self, document_text, chunks):
            return []  # backend returned nothing — simulating a parse error

    long_chunk = "x" * (MIN_CHUNK_CHARS_FOR_PREFIX + 10)
    doc = _converted_doc([long_chunk])

    with tempfile.TemporaryDirectory() as td:
        cache = ContextualCache(os.path.join(td, "cache.json"))
        ChunkPrefixer(FailingBackend(), cache).prefix_document(doc)

    assert "contextualPrefix" not in doc["chunks"][0]
    assert doc["chunks"][0]["indexedData"] == long_chunk


def test_prefixer_continues_when_backend_raises():
    class ExplodingBackend:
        model_id = "echo:exploding"

        def generate(self, document_text, chunks):
            raise RuntimeError("LLM exploded")

    long_chunk = "x" * (MIN_CHUNK_CHARS_FOR_PREFIX + 10)
    doc = _converted_doc([long_chunk])

    with tempfile.TemporaryDirectory() as td:
        cache = ContextualCache(os.path.join(td, "cache.json"))
        ChunkPrefixer(ExplodingBackend(), cache).prefix_document(doc)

    assert "contextualPrefix" not in doc["chunks"][0]


def test_prefixer_count_mismatch_is_treated_as_failure():
    """Defensive: if the model returns the wrong number of prefixes we drop all of them
    rather than risk pairing them to the wrong chunks."""
    class WrongCountBackend:
        model_id = "echo:wrong"

        def generate(self, document_text, chunks):
            return ["only one"] if len(chunks) > 1 else []

    long_chunk = "x" * (MIN_CHUNK_CHARS_FOR_PREFIX + 10)
    doc = _converted_doc([long_chunk, long_chunk + "Y"])

    with tempfile.TemporaryDirectory() as td:
        cache = ContextualCache(os.path.join(td, "cache.json"))
        ChunkPrefixer(WrongCountBackend(), cache).prefix_document(doc)

    for chunk in doc["chunks"]:
        assert "contextualPrefix" not in chunk


# ---------- OllamaBackend._parse_prefix_array ----------

def test_parse_prefix_array_handles_plain_json_list():
    assert _parse_prefix_array('["a", "b"]', expected_count=2) == ["a", "b"]


def test_parse_prefix_array_strips_markdown_fences():
    raw = "```json\n[\"a\", \"b\"]\n```"
    assert _parse_prefix_array(raw, expected_count=2) == ["a", "b"]


def test_parse_prefix_array_handles_object_wrapper():
    raw = '{"prefixes": ["a", "b"]}'
    assert _parse_prefix_array(raw, expected_count=2) == ["a", "b"]


def test_parse_prefix_array_returns_empty_on_bad_json():
    assert _parse_prefix_array("not json", expected_count=2) == []


def test_parse_prefix_array_returns_empty_on_non_list():
    assert _parse_prefix_array('"just a string"', expected_count=2) == []


def test_parse_prefix_array_tolerates_trailing_comma():
    assert _parse_prefix_array('["a", "b",]', expected_count=2) == ["a", "b"]
    assert _parse_prefix_array('[\n  "a",\n  "b",\n]', expected_count=2) == ["a", "b"]


# ---------- Parallel prefixing (ChunkPrefixer is thread-safe via cache lock) ----------

def test_ollama_backend_batches_chunks_to_bounded_calls(monkeypatch):
    """A doc with 25 chunks at chunks_per_call=10 must make 3 calls (10, 10, 5),
    not one 25-chunk call that risks exceeding num_predict."""
    from main.core.contextual_prefix.backends.ollama_backend import OllamaBackend

    backend = OllamaBackend(model="test-model", chunks_per_call=10)

    call_log: list[int] = []

    def fake_generate_batch(document_text, batch_chunks):
        call_log.append(len(batch_chunks))
        return [f"prefix-{i}" for i in range(len(batch_chunks))]

    monkeypatch.setattr(backend, "_generate_batch", fake_generate_batch)
    chunks = [f"chunk{i}" for i in range(25)]
    prefixes = backend.generate("doc text", chunks)

    assert call_log == [10, 10, 5]
    assert len(prefixes) == 25
    assert prefixes[0] == "prefix-0"
    assert prefixes[10] == "prefix-0"  # batch 2 starts a new sequence
    assert prefixes[20] == "prefix-0"  # batch 3 too


def test_ollama_backend_aborts_doc_if_any_batch_returns_wrong_count(monkeypatch):
    from main.core.contextual_prefix.backends.ollama_backend import OllamaBackend

    backend = OllamaBackend(model="test-model", chunks_per_call=5)

    def fake_generate_batch(document_text, batch_chunks):
        # second batch returns one less prefix than asked for
        if batch_chunks[0] == "chunk5":
            return [f"p-{i}" for i in range(len(batch_chunks) - 1)]
        return [f"p-{i}" for i in range(len(batch_chunks))]

    monkeypatch.setattr(backend, "_generate_batch", fake_generate_batch)
    chunks = [f"chunk{i}" for i in range(12)]
    prefixes = backend.generate("doc text", chunks)

    assert prefixes == []  # aborted — don't try to pair mismatched prefixes


def test_parallel_prefixing_handles_concurrent_calls_safely():
    """ChunkPrefixer.prefix_document must be safe to call from many threads at once."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    long_chunk = "x" * (MIN_CHUNK_CHARS_FOR_PREFIX + 10)
    doc_count = 8
    docs = [_converted_doc([long_chunk], doc_id=f"doc{i}") for i in range(doc_count)]

    threads_seen: set = set()
    lock = threading.Lock()

    class TrackingBackend:
        model_id = "echo:tracking"

        def generate(self, document_text, chunks):
            with lock:
                threads_seen.add(threading.get_ident())
            return [f"prefix-{i}" for i in range(len(chunks))]

    with tempfile.TemporaryDirectory() as td:
        cache = ContextualCache(os.path.join(td, "cache.json"))
        prefixer = ChunkPrefixer(TrackingBackend(), cache)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(prefixer.prefix_document, docs))

        for doc in docs:
            assert doc["chunks"][0]["contextualPrefix"] == "prefix-0"
            assert doc["chunks"][0]["indexedData"].startswith("prefix-0")

        assert len(cache) == doc_count
        # Actually used multiple threads (defensively: at least 2; depending on scheduling,
        # could be up to 4). If this only ever sees 1 thread, the parallel path isn't doing
        # anything useful.
        assert len(threads_seen) > 1, f"Expected concurrent threads, saw {len(threads_seen)}"
