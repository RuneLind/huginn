"""Guards on scripts/audit/verify_aliased_collection.py.

The sweep is the only thing standing between a rebuilt collection and a copy of
it on someone else's laptop, so the ways it can pass *vacuously* — a truncated
map, a stamp from another build, an empty needle list, a name the needle regex
cannot see — matter more than the ways it can fail. Every test here therefore
builds a collection the sweep would otherwise wave through.

Every name is invented ("Ada Example", "Zylphia Quorndal"). The real map lives
in a gitignored private sub-repo and is never read by the test suite.
"""
import json
import pickle
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "audit"))

import verify_aliased_collection as verify  # noqa: E402

MAP_VERSION = 7
GOOD_STAMP = {"policy_version": 1, "map_version": MAP_VERSION,
              "aliasedAt": "2026-01-02T00:00:00+00:00"}


def _map(entry_count=verify.MIN_MAP_ENTRIES):
    """A schema-valid map with enough entries to clear the floor.

    The generated variants carry a digit on purpose: a single all-alphabetic
    token is a bare given name and AliasRegistry refuses to compile one.
    """
    entries = []
    for i in range(entry_count):
        name = f"Ada Example{i:02d}"
        entries.append({
            "alias": f"dev-{i:02d}",
            "name": name,
            "role": "dev",
            "variants": [name, f"Example{i:02d}, Ada", f"ada.example{i:02d}"],
            "require_full_name": False,
            "idents": [],
            "departed": False,
            "confirmed": True,
            "extra_variants": [],
        })
    return {
        "version": MAP_VERSION,
        "entries": entries,
        "ident_policy": "redact",
        "non_person_labels": ["saksbehandler"],
        "unmapped_people": ["Zylphia Quorndal"],
        "unmapped_people_variants": {
            "Zylphia Quorndal": ["Zylphia Quorndal", "Quorndal, Zylphia",
                                 "zylphia.quorndal"],
        },
        "person_redaction_token": "[~ukjent-person]",
        "retired_aliases": [],
    }


def _write_collection(root, name, *, documents, stamp=GOOD_STAMP,
                      documents_count=None, chunks_count=None):
    """A minimal on-disk collection the sweep can walk end to end."""
    collection = root / name
    (collection / "documents").mkdir(parents=True)
    (collection / "indexes" / "indexer_BM25").mkdir(parents=True)

    for index, document in enumerate(documents):
        (collection / "documents" / f"doc{index}.json").write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "collectionName": name,
        "numberOfDocuments": len(documents) if documents_count is None else documents_count,
        "numberOfChunks": len(documents) if chunks_count is None else chunks_count,
        "reader": {"type": "localFiles", "basePath": "./data/sources/demo"},
    }
    if stamp is not None:
        manifest["privacy"] = stamp
    (collection / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mapping = {str(i): d["id"] for i, d in enumerate(documents)}
    (collection / "indexes" / "index_document_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8")
    (collection / "indexes" / "reverse_index_document_mapping.json").write_text(
        json.dumps({v: k for k, v in mapping.items()}), encoding="utf-8")

    corpus = [d["text"].lower().split() for d in documents]
    with open(collection / "indexes" / "indexer_BM25" / "indexer", "wb") as f:
        pickle.dump({"corpus_tokens": corpus, "ids": list(mapping)}, f)
    return collection


def _clean_document(index):
    return {"id": f"notat-{index}.md", "url": f"file:///srv/notat-{index}.md",
            "text": "dev-01 skrev dette sammen med en saksbehandler.",
            "chunks": [{"indexedData": "dev-01 skrev dette."}]}


@pytest.fixture
def map_file(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(_map()), encoding="utf-8")
    return path


def _run(argv, monkeypatch, repo_root):
    """Drive main() the way the campaign does, with no private files in reach."""
    monkeypatch.setattr(verify, "REPO_ROOT", repo_root)
    monkeypatch.setattr(sys, "argv", ["verify_aliased_collection.py", *argv])
    with pytest.raises(SystemExit) as excinfo:
        verify.main()
    return excinfo.value.code


# --- check 0: the map itself ------------------------------------------------

def test_truncated_map_is_rejected_by_the_entry_floor(tmp_path):
    """A decoy map with three names certifies everything it does not know."""
    collection = _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)])
    assert verify.check_map_stamp(tmp_path, "demo-aliased", _map(3)) is False
    assert verify.check_map_stamp(tmp_path, "demo-aliased", _map()) is True
    assert collection.exists()


@pytest.mark.parametrize("stamp", [
    {**GOOD_STAMP, "map_version": MAP_VERSION - 1},
    {**GOOD_STAMP, "policy_version": verify.POLICY_VERSION + 1},
    None,
])
def test_stamp_from_another_build_is_rejected(tmp_path, stamp):
    # The sweep verifying with map v7 against an index built from v6 reports a
    # clean collection for names v6 never knew.
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)], stamp=stamp)
    assert verify.check_map_stamp(tmp_path, "demo-aliased", _map()) is False


# --- needles ----------------------------------------------------------------

def test_empty_needle_list_exits_rather_than_passing_vacuously():
    with pytest.raises(SystemExit):
        verify.needle_pattern([])


def test_needle_pattern_sees_a_percent_encoded_name():
    """The boundary is shared with the registry: a form the substituter removes
    but the sweep cannot see would certify a collection nobody aliased."""
    pattern = verify.needle_pattern(["Ada Example"])
    assert pattern.search("ovuser=abc%2CAda%20Example%40nav.no")
    assert pattern.search("?q=fra%20Ada%20Example")
    assert not pattern.search("AdaExample")


def test_contains_sequence_catches_a_hyphenated_name_in_a_document_id():
    """Ids are never aliased by design, so a name in a path is a real leak; a
    plain substring test misses it because the separator is a hyphen."""
    sequences = verify.token_sequences(["Ada Example"])
    tokens = [t.lower() for t in verify.WORD_SPLIT_RE.split("Team/Ada-Example.md") if t]
    assert verify.contains_sequence(tokens, sequences) is True
    assert verify.contains_sequence(["team", "notat", "md"], sequences) is False


# --- counts against the pre-alias twin --------------------------------------

def test_document_count_drift_fails_by_default(tmp_path, map_file, monkeypatch):
    """A rebuild that indexed a different set of documents is not a rebuild of
    the same collection — and its extra documents were never swept before."""
    _write_collection(tmp_path, "demo", documents=[_clean_document(0)])
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_clean_document(0), _clean_document(1)])
    code = _run(["--collection", "demo-aliased", "--compare", "demo",
                 "--collections-dir", str(tmp_path), "--map", str(map_file)],
                monkeypatch, tmp_path)
    assert code == 1


def test_allow_count_drift_downgrades_the_count_check_to_a_warning(
        tmp_path, map_file, monkeypatch, capsys):
    _write_collection(tmp_path, "demo", documents=[_clean_document(0)])
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_clean_document(0), _clean_document(1)])
    code = _run(["--collection", "demo-aliased", "--compare", "demo",
                 "--collections-dir", str(tmp_path), "--map", str(map_file),
                 "--allow-count-drift"],
                monkeypatch, tmp_path)
    assert code == 0
    assert "WARN" in capsys.readouterr().out


def test_matching_counts_pass(tmp_path, map_file, monkeypatch):
    _write_collection(tmp_path, "demo", documents=[_clean_document(0)])
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)])
    code = _run(["--collection", "demo-aliased", "--compare", "demo",
                 "--collections-dir", str(tmp_path), "--map", str(map_file)],
                monkeypatch, tmp_path)
    assert code == 0


def test_a_real_name_in_a_document_still_fails(tmp_path, map_file, monkeypatch):
    """The whole sweep, end to end, on a collection that leaks — otherwise the
    three tests above only prove the count check works."""
    leaky = {"id": "notat-0.md", "url": "file:///srv/notat-0.md",
             "text": "Ada Example00 skrev dette.",
             "chunks": [{"indexedData": "Ada Example00 skrev dette."}]}
    _write_collection(tmp_path, "demo", documents=[_clean_document(0)])
    _write_collection(tmp_path, "demo-aliased", documents=[leaky])
    code = _run(["--collection", "demo-aliased", "--compare", "demo",
                 "--collections-dir", str(tmp_path), "--map", str(map_file)],
                monkeypatch, tmp_path)
    assert code == 1
