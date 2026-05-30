"""Characterization tests for the update-factory reader/converter builders (M13).

These pin the exact reader class, converter class, env vars, query addition,
and validation behaviour for every reader type, so the if/elif → registry
refactor can be proven behaviour-preserving (this code has no other coverage
and constructs live-API clients from credentials, so it can't be exercised
end-to-end in CI).
"""
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

import main.factories.update_collection_factory as uf


LAST_MODIFIED = "2026-01-10T00:00:00"
UPDATE_DATE = "2026-01-09"  # LAST_MODIFIED - 1 day
UPDATE_TIME = datetime.fromisoformat(LAST_MODIFIED) - timedelta(days=1)


def _build(manifest):
    """Call the reader/converter dispatcher under whichever name it has."""
    fn = getattr(uf, "_create_reader_and_converter", None) or getattr(
        uf, "__create_reader_and_converter"
    )
    return fn(manifest)


def _manifest(reader):
    return {"reader": reader, "lastModifiedDocumentTime": LAST_MODIFIED}


class TestJira:
    def test_builds_jira_reader_and_converter(self, monkeypatch):
        monkeypatch.setenv("JIRA_TOKEN", "tok")
        monkeypatch.setenv("JIRA_LOGIN", "user")
        monkeypatch.setenv("JIRA_PASSWORD", "pw")
        m = _manifest({"type": "jira", "baseUrl": "https://j", "query": "project=X", "batchSize": 50})
        with patch.object(uf, "JiraDocumentReader") as Reader, \
             patch.object(uf, "JiraDocumentConverter") as Converter:
            reader, converter = _build(m)
        Reader.assert_called_once_with(
            base_url="https://j",
            query=f'project=X AND (created >= "{UPDATE_DATE}" OR updated >= "{UPDATE_DATE}")',
            token="tok", login="user", password="pw", batch_size=50,
        )
        assert reader is Reader.return_value
        assert converter is Converter.return_value


class TestJiraCloud:
    def test_builds_with_atlassian_creds(self, monkeypatch):
        monkeypatch.setenv("ATLASSIAN_EMAIL", "e@x")
        monkeypatch.setenv("ATLASSIAN_TOKEN", "atok")
        m = _manifest({"type": "jiraCloud", "baseUrl": "https://jc", "query": "project=Y", "batchSize": 25})
        with patch.object(uf, "JiraCloudDocumentReader") as Reader, \
             patch.object(uf, "JiraCloudDocumentConverter"):
            _build(m)
        Reader.assert_called_once_with(
            base_url="https://jc",
            query=f'project=Y AND (created >= "{UPDATE_DATE}" OR updated >= "{UPDATE_DATE}")',
            email="e@x", api_token="atok", batch_size=25,
        )

    def test_missing_creds_raises(self, monkeypatch):
        monkeypatch.delenv("ATLASSIAN_EMAIL", raising=False)
        monkeypatch.delenv("ATLASSIAN_TOKEN", raising=False)
        m = _manifest({"type": "jiraCloud", "baseUrl": "https://jc", "query": "q", "batchSize": 10})
        with pytest.raises(ValueError):
            _build(m)


class TestConfluence:
    def test_builds_with_token(self, monkeypatch):
        monkeypatch.setenv("CONF_TOKEN", "ctok")
        monkeypatch.delenv("CONF_LOGIN", raising=False)
        monkeypatch.delenv("CONF_PASSWORD", raising=False)
        m = _manifest({"type": "confluence", "baseUrl": "https://c", "query": "space=Z",
                       "batchSize": 50, "readAllComments": True})
        with patch.object(uf, "ConfluenceDocumentReader") as Reader, \
             patch.object(uf, "ConfluenceDocumentConverter"):
            _build(m)
        Reader.assert_called_once_with(
            base_url="https://c",
            query=f'space=Z AND (created >= "{UPDATE_DATE}" OR lastModified >= "{UPDATE_DATE}")',
            token="ctok", login=None, password=None, batch_size=50, read_all_comments=True,
        )

    def test_missing_all_creds_raises(self, monkeypatch):
        for v in ("CONF_TOKEN", "CONF_LOGIN", "CONF_PASSWORD"):
            monkeypatch.delenv(v, raising=False)
        m = _manifest({"type": "confluence", "baseUrl": "https://c", "query": "q",
                       "batchSize": 1, "readAllComments": False})
        with pytest.raises(ValueError):
            _build(m)


class TestConfluenceCloud:
    def test_builds_with_atlassian_creds(self, monkeypatch):
        monkeypatch.setenv("ATLASSIAN_EMAIL", "e@x")
        monkeypatch.setenv("ATLASSIAN_TOKEN", "atok")
        m = _manifest({"type": "confluenceCloud", "baseUrl": "https://cc", "query": "space=W",
                       "batchSize": 30, "readAllComments": False})
        with patch.object(uf, "ConfluenceCloudDocumentReader") as Reader, \
             patch.object(uf, "ConfluenceCloudDocumentConverter"):
            _build(m)
        Reader.assert_called_once_with(
            base_url="https://cc",
            query=f'space=W AND (created >= "{UPDATE_DATE}" OR lastModified >= "{UPDATE_DATE}")',
            email="e@x", api_token="atok", batch_size=30, read_all_comments=False,
        )

    def test_missing_creds_raises(self, monkeypatch):
        monkeypatch.delenv("ATLASSIAN_EMAIL", raising=False)
        monkeypatch.delenv("ATLASSIAN_TOKEN", raising=False)
        m = _manifest({"type": "confluenceCloud", "baseUrl": "x", "query": "q",
                       "batchSize": 1, "readAllComments": False})
        with pytest.raises(ValueError):
            _build(m)


class TestNotion:
    def test_builds_with_token_and_time(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "ntok")
        m = _manifest({"type": "notion", "rootPageId": "root-1", "requestDelay": 0.5})
        with patch.object(uf, "NotionDocumentReader") as Reader, \
             patch.object(uf, "NotionDocumentConverter"):
            _build(m)
        Reader.assert_called_once_with(
            token="ntok", root_page_id="root-1", request_delay=0.5, start_from_time=UPDATE_TIME,
        )

    def test_request_delay_defaults(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "ntok")
        m = _manifest({"type": "notion"})
        with patch.object(uf, "NotionDocumentReader") as Reader, \
             patch.object(uf, "NotionDocumentConverter"):
            _build(m)
        _, kwargs = Reader.call_args
        assert kwargs["request_delay"] == 0.35
        assert kwargs["root_page_id"] is None

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("NOTION_TOKEN", raising=False)
        with pytest.raises(ValueError):
            _build(_manifest({"type": "notion"}))


class TestLocalFiles:
    def test_builds_files_reader(self):
        m = _manifest({"type": "localFiles", "basePath": "/data/src",
                       "includePatterns": ["a.*"], "excludePatterns": ["b.*"], "failFast": True})
        with patch.object(uf, "FilesDocumentReader") as Reader, \
             patch.object(uf, "FilesDocumentConverter"):
            _build(m)
        Reader.assert_called_once_with(
            base_path="/data/src", include_patterns=["a.*"], exclude_patterns=["b.*"],
            fail_fast=True, start_from_time=UPDATE_TIME,
        )

    def test_defaults(self):
        m = _manifest({"type": "localFiles", "basePath": "/d"})
        with patch.object(uf, "FilesDocumentReader") as Reader, \
             patch.object(uf, "FilesDocumentConverter"):
            _build(m)
        _, kwargs = Reader.call_args
        assert kwargs["include_patterns"] == [".*"]
        assert kwargs["exclude_patterns"] == []
        assert kwargs["fail_fast"] is False


class TestUnknownType:
    def test_unknown_reader_type_raises(self):
        with pytest.raises(Exception):
            _build(_manifest({"type": "mystery"}))
