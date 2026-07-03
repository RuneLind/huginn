"""Tests for orphan-document pruning during incremental collection updates.

An incremental update only re-reads documents inside its time window, so a
document whose source file was deleted, renamed, or moved to .excluded/ is never
revisited. Without pruning, its chunks linger in the index and keep surfacing in
search. These tests cover the reader's full-id enumeration and the updater's
reconciliation.
"""
import datetime
import json

import numpy as np

from main.core.documents_collection_creator import DocumentCollectionCreator
from main.persisters.disk_persister import DiskPersister
from main.sources.files.files_document_reader import FilesDocumentReader


# --- FilesDocumentReader.get_all_document_ids ---------------------------------

def test_get_all_document_ids_ignores_time_window(tmp_path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    # A future cutoff hides everything from read_all_documents ...
    reader = FilesDocumentReader(
        base_path=str(tmp_path),
        include_patterns=[".*"],
        start_from_time=datetime.datetime(2999, 1, 1),
    )
    assert sum(1 for _ in reader.read_all_documents()) == 0
    # ... but get_all_document_ids still returns the full current set.
    assert reader.get_all_document_ids() == {"a.md", "b.md"}


def test_get_all_document_ids_respects_exclude_patterns(tmp_path):
    (tmp_path / "keep.md").write_text("k", encoding="utf-8")
    excluded = tmp_path / ".excluded"
    excluded.mkdir()
    (excluded / "gone.md").write_text("g", encoding="utf-8")
    reader = FilesDocumentReader(
        base_path=str(tmp_path),
        include_patterns=[".*"],
        exclude_patterns=[r"^\..*"],
    )
    assert reader.get_all_document_ids() == {"keep.md"}


# --- DocumentCollectionCreator.__prune_orphaned_documents ---------------------

class FakeIndexer:
    def __init__(self, name="indexer_BM25"):
        self._name = name
        self.removed = []

    def get_name(self):
        return self._name

    def remove_ids(self, ids):
        self.removed.extend(int(i) for i in np.array(ids).tolist())


class FakeReader:
    def __init__(self, valid_ids):
        self._valid_ids = set(valid_ids)

    def get_all_document_ids(self):
        return self._valid_ids


def _prune(creator, index_mapping, reverse_index_mapping):
    creator._DocumentCollectionCreator__prune_orphaned_documents(
        index_mapping, reverse_index_mapping)


def _make_creator(persister, reader, indexer, collection="col"):
    return DocumentCollectionCreator(
        collection_name=collection,
        document_reader=reader,
        document_converter=None,
        document_indexers=[indexer],
        persister=persister,
    )


def test_prunes_orphan_and_deletes_its_document_json(tmp_path):
    persister = DiskPersister(str(tmp_path))
    persister.save_text_file(json.dumps({"id": "valid.md"}), "col/documents/valid.md.json")
    persister.save_text_file(json.dumps({"id": "orphan.md"}), "col/documents/orphan.md.json")

    indexer = FakeIndexer()
    creator = _make_creator(persister, FakeReader({"valid.md"}), indexer)

    index_mapping = {"0": {"documentId": "valid.md"}, "1": {"documentId": "orphan.md"}}
    reverse = {"valid.md": [0], "orphan.md": [1]}

    _prune(creator, index_mapping, reverse)

    assert reverse == {"valid.md": [0]}
    assert index_mapping == {"0": {"documentId": "valid.md"}}
    assert indexer.removed == [1]
    assert persister.is_path_exists("col/documents/valid.md.json")
    assert not persister.is_path_exists("col/documents/orphan.md.json")


def test_no_orphans_is_a_noop(tmp_path):
    persister = DiskPersister(str(tmp_path))
    indexer = FakeIndexer()
    creator = _make_creator(persister, FakeReader({"valid.md"}), indexer)

    reverse = {"valid.md": [0]}
    index_mapping = {"0": {"documentId": "valid.md"}}
    _prune(creator, index_mapping, reverse)

    assert reverse == {"valid.md": [0]}
    assert indexer.removed == []


def test_empty_valid_set_does_not_wipe_index(tmp_path):
    # A transient empty read (FS hiccup / mistyped pattern) must NOT be treated
    # as "every document was deleted" — that would nuke the whole index.
    persister = DiskPersister(str(tmp_path))
    persister.save_text_file(json.dumps({"id": "valid.md"}), "col/documents/valid.md.json")
    indexer = FakeIndexer()
    creator = _make_creator(persister, FakeReader(set()), indexer)

    reverse = {"valid.md": [0]}
    index_mapping = {"0": {"documentId": "valid.md"}}
    _prune(creator, index_mapping, reverse)

    assert reverse == {"valid.md": [0]}
    assert indexer.removed == []
    assert persister.is_path_exists("col/documents/valid.md.json")


def test_reader_without_enumeration_never_prunes(tmp_path):
    # Query-based readers (Jira/Confluence/Notion) return only the incremental
    # window and have no get_all_document_ids — pruning must be skipped entirely
    # so an out-of-window document is not mistaken for a deleted one.
    persister = DiskPersister(str(tmp_path))
    indexer = FakeIndexer()

    class QueryReader:
        pass

    creator = _make_creator(persister, QueryReader(), indexer)
    reverse = {"some-doc": [0], "another": [1]}
    index_mapping = {"0": {"documentId": "some-doc"}, "1": {"documentId": "another"}}
    _prune(creator, index_mapping, reverse)

    assert reverse == {"some-doc": [0], "another": [1]}
    assert indexer.removed == []
