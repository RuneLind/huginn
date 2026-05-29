"""Regression tests for atomic index-artifact commit in DocumentCollectionCreator (Phase 2b / H2).

A crash or error while persisting the index pickle + the three sidecar JSONs must leave
the prior on-disk artifacts intact rather than a half-written, mutually-inconsistent set.
"""

import json

import pytest

from main.core.documents_collection_creator import DocumentCollectionCreator
from main.persisters.disk_persister import DiskPersister


class FakeIndexer:
    def __init__(self, name, payload, size=1):
        self._name = name
        self._payload = payload
        self._size = size
        self.serialize_error = None

    def get_name(self):
        return self._name

    def get_size(self):
        return self._size

    def serialize(self):
        if self.serialize_error is not None:
            raise self.serialize_error
        return self._payload


def make_creator(persister, indexers, collection="col"):
    return DocumentCollectionCreator(
        collection_name=collection,
        document_reader=None,
        document_converter=None,
        document_indexers=indexers,
        persister=persister,
    )


def persist(creator, index_info, mapping, reverse):
    # exercise the private set-commit method directly
    creator._DocumentCollectionCreator__atomically_persist_index_artifacts(index_info, mapping, reverse)


@pytest.fixture
def persister(tmp_path):
    return DiskPersister(str(tmp_path))


def test_commits_all_artifacts_on_success(persister, tmp_path):
    indexer = FakeIndexer("indexer_BM25", {"tokens": [1, 2]})
    creator = make_creator(persister, [indexer])

    persist(creator, {"lastIndexItemId": 4}, {"0": {"documentId": "d0"}}, {"d0": [0]})

    assert persister.read_bin_file("col/indexes/indexer_BM25/indexer") == {"tokens": [1, 2]}
    assert json.loads(persister.read_text_file("col/indexes/index_info.json")) == {"lastIndexItemId": 4}
    assert json.loads(persister.read_text_file("col/indexes/index_document_mapping.json")) == {"0": {"documentId": "d0"}}
    assert json.loads(persister.read_text_file("col/indexes/reverse_index_document_mapping.json")) == {"d0": [0]}
    # no staging temp files left behind anywhere under the collection
    assert not _staging_files(tmp_path)


def test_failure_mid_persist_leaves_prior_artifacts_intact(persister, tmp_path):
    good = FakeIndexer("indexer_FAISS", {"v": 1})
    bad = FakeIndexer("indexer_BM25", {"v": 1})
    creator = make_creator(persister, [good, bad])

    # first, a clean commit establishes the prior on-disk state
    persist(creator, {"lastIndexItemId": 1}, {"0": {"documentId": "old"}}, {"old": [0]})

    # now a second persist fails while serializing the second indexer
    good._payload = {"v": 2}
    bad._payload = {"v": 2}
    bad.serialize_error = RuntimeError("disk full mid-write")
    with pytest.raises(RuntimeError):
        persist(creator, {"lastIndexItemId": 99}, {"0": {"documentId": "new"}}, {"new": [0]})

    # every artifact must still be the prior version — nothing half-committed
    assert persister.read_bin_file("col/indexes/indexer_FAISS/indexer") == {"v": 1}
    assert persister.read_bin_file("col/indexes/indexer_BM25/indexer") == {"v": 1}
    assert json.loads(persister.read_text_file("col/indexes/index_info.json")) == {"lastIndexItemId": 1}
    assert json.loads(persister.read_text_file("col/indexes/index_document_mapping.json")) == {"0": {"documentId": "old"}}
    assert json.loads(persister.read_text_file("col/indexes/reverse_index_document_mapping.json")) == {"old": [0]}
    # the aborted attempt must not leak staged files
    assert not _staging_files(tmp_path)


def _staging_files(root):
    return [str(p) for p in root.rglob(DiskPersister.TEMP_PREFIX + "*")]
