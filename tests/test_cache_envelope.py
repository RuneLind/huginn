"""Guards on main/privacy/cache_envelope.py — the shared LLM-cache envelope.

Two callers (the knowledge-graph extractor and the sensitivity sweep) had two
copies of the same three decisions, so the tests that matter are the ones about
the decisions rather than about JSON round-tripping: a legacy flat file is
distinguishable from an envelope, a crashed write leaves the previous cache
intact, and a run that dies mid-write does not leave a temp file behind that a
later glob would mistake for a cache.
"""
import json
import os

import pytest

from main.privacy import cache_envelope


def test_a_round_trip_keeps_metadata_and_entries(tmp_path):
    path = tmp_path / "c.json"
    cache_envelope.write_envelope(path, {"policy_version": 1}, {"a.md": {"n": 1}})
    assert cache_envelope.load_envelope(path) == ({"policy_version": 1},
                                                  {"a.md": {"n": 1}})


def test_a_doc_id_starting_with_an_underscore_is_an_entry_not_metadata(tmp_path):
    """The reason the envelope exists. With metadata as sibling `_`-prefixed
    keys, `_index.md` is indistinguishable from a version marker."""
    path = tmp_path / "c.json"
    cache_envelope.write_envelope(path, {"policy_version": 1}, {"_index.md": {"n": 1}})
    metadata, entries = cache_envelope.load_envelope(path)
    assert metadata == {"policy_version": 1} and "_index.md" in entries


def test_a_legacy_flat_file_comes_back_with_no_metadata(tmp_path):
    """So a caller that requires a metadata match rejects it, and the graph
    extractor — which still has pre-envelope caches for ~30 out-of-scope
    collections — can accept it deliberately."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"a.md": {"n": 1}}), encoding="utf-8")
    assert cache_envelope.load_envelope(path) == ({}, {"a.md": {"n": 1}})


@pytest.mark.parametrize("content", ["{ truncated", "[]", '"a string"', ""])
def test_an_unreadable_file_is_a_cold_cache_not_a_crash(tmp_path, content):
    """Raising here turns a truncated cache into a crashed nightly job."""
    path = tmp_path / "c.json"
    path.write_text(content, encoding="utf-8")
    assert cache_envelope.load_envelope(path) == ({}, {})


def test_a_missing_file_is_a_cold_cache(tmp_path):
    assert cache_envelope.load_envelope(tmp_path / "absent.json") == ({}, {})


def test_the_write_is_atomic(tmp_path, monkeypatch):
    """A crash between the temp write and the rename must leave the PREVIOUS
    cache readable. `write_text` straight onto the path left a truncated file,
    and the next run re-asked the model about the whole collection."""
    path = tmp_path / "c.json"
    cache_envelope.write_envelope(path, {"policy_version": 1}, {"a.md": {"n": 1}})

    def exploding(src, dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", exploding)
    with pytest.raises(OSError):
        cache_envelope.write_envelope(path, {"policy_version": 1}, {"b.md": {"n": 2}})
    assert cache_envelope.load_envelope(path)[1] == {"a.md": {"n": 1}}
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".c.json")] == []


def test_the_lock_is_a_sidecar_not_the_data_file(tmp_path):
    """The data file is replaced by inode swap, so a lock taken on it protects
    nothing after the first write — the same reasoning the indexing-run ledger
    documents for its own JSONL."""
    path = tmp_path / "c.json"
    cache_envelope.write_envelope(path, {}, {})
    assert (tmp_path / "c.json.lock").exists()


def test_a_nested_directory_is_created(tmp_path):
    path = tmp_path / "state" / "sensitivity" / "demo.json"
    cache_envelope.write_envelope(path, {"model": "m"}, {})
    assert path.exists()
