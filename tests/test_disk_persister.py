"""Regression tests for atomic + staged writes in DiskPersister (Phase 2b / H2)."""

import os

import pytest

from main.persisters.disk_persister import DiskPersister


@pytest.fixture
def persister(tmp_path):
    return DiskPersister(str(tmp_path))


class TestAtomicSave:
    def test_text_round_trip(self, persister):
        persister.save_text_file("hælló æøå", "sub/dir/file.txt")
        assert persister.read_text_file("sub/dir/file.txt") == "hælló æøå"

    def test_bin_round_trip(self, persister):
        persister.save_bin_file({"a": [1, 2, 3]}, "blob/indexer")
        assert persister.read_bin_file("blob/indexer") == {"a": [1, 2, 3]}

    def test_save_leaves_no_temp_files(self, persister, tmp_path):
        persister.save_text_file("data", "indexes/info.json")
        leftovers = [n for n in os.listdir(tmp_path / "indexes") if n.startswith(DiskPersister.TEMP_PREFIX)]
        assert leftovers == []

    def test_save_honours_umask_permissions(self, persister, tmp_path):
        persister.save_text_file("data", "indexes/info.json")
        mode = (tmp_path / "indexes" / "info.json").stat().st_mode & 0o777
        # an in-place open(path, 'w') would have produced 0o666 & ~umask, not mkstemp's 0o600
        assert mode == persister._file_mode

    def test_failed_replace_preserves_existing_file_and_cleans_temp(self, persister, tmp_path, monkeypatch):
        persister.save_text_file("original", "indexes/info.json")

        def boom(src, dst):
            raise OSError("simulated crash before rename completes")

        monkeypatch.setattr("main.persisters.disk_persister.os.replace", boom)
        with pytest.raises(OSError):
            persister.save_text_file("new", "indexes/info.json")

        # the pre-existing file must survive an interrupted overwrite
        assert persister.read_text_file("indexes/info.json") == "original"
        # and the failed write must not leak a temp file
        leftovers = [n for n in os.listdir(tmp_path / "indexes") if n.startswith(DiskPersister.TEMP_PREFIX)]
        assert leftovers == []


class TestReadFolderFiles:
    def test_skips_interrupted_temp_files(self, persister, tmp_path):
        persister.save_text_file("real", "docs/0.json")
        # simulate a temp left behind by a hard-killed atomic write
        (tmp_path / "docs" / (DiskPersister.TEMP_PREFIX + "leftover")).write_text("partial")

        files = persister.read_folder_files("docs")
        assert files == ["0.json"]


class TestAtomicWriteSet:
    def test_commits_all_files_together_on_success(self, persister):
        with persister.atomic_write_set() as write_set:
            write_set.add_text_file("info", "indexes/info.json")
            write_set.add_bin_file({"k": 7}, "indexes/BM25/indexer")

        assert persister.read_text_file("indexes/info.json") == "info"
        assert persister.read_bin_file("indexes/BM25/indexer") == {"k": 7}

    def test_targets_untouched_until_block_exits(self, persister, tmp_path):
        persister.save_text_file("v1", "indexes/info.json")

        write_set = persister.atomic_write_set()
        write_set.__enter__()
        write_set.add_text_file("v2", "indexes/info.json")
        # staged but not yet committed
        assert persister.read_text_file("indexes/info.json") == "v1"

        write_set.__exit__(None, None, None)
        assert persister.read_text_file("indexes/info.json") == "v2"

    def test_error_in_block_rolls_back_and_leaves_targets_intact(self, persister, tmp_path):
        persister.save_text_file("v1", "indexes/info.json")
        persister.save_bin_file({"v": 1}, "indexes/BM25/indexer")

        with pytest.raises(RuntimeError):
            with persister.atomic_write_set() as write_set:
                write_set.add_text_file("v2", "indexes/info.json")
                write_set.add_bin_file({"v": 2}, "indexes/BM25/indexer")
                raise RuntimeError("indexing failed before commit")

        # nothing committed; prior versions intact
        assert persister.read_text_file("indexes/info.json") == "v1"
        assert persister.read_bin_file("indexes/BM25/indexer") == {"v": 1}
        # and no staged temp files leak (read_folder_files filters them, so check the fs directly)
        leftovers = list((tmp_path / "indexes").rglob(DiskPersister.TEMP_PREFIX + "*"))
        assert leftovers == []
