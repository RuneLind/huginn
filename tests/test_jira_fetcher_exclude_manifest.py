"""Tests for JiraFetcher.load_exclude_manifest permanent-vs-dynamic filtering.

Content-based (low_word_count/empty_stub/minimal_content) and age-based
(too_old) exclusions are DYNAMIC: a stub can grow into a real issue and an old
issue can be revived. Those must NOT be permanently skipped on re-fetch, or the
issue is trapped in .excluded/ forever. Only stable noise_* reasons are
permanent skips.
"""

import json

from scripts.jira.fetchers.jira_fetcher import JiraFetcher


def _write_manifest(tmp_path, entries):
    path = tmp_path / "excluded_manifest.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def test_content_exclusions_are_not_permanently_skipped(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"issue_key": "PROJ-1", "reason": "low_word_count"},
        {"issue_key": "PROJ-2", "reason": "empty_stub"},
        {"issue_key": "PROJ-3", "reason": "minimal_content"},
    ])
    assert JiraFetcher.load_exclude_manifest(manifest) == set()


def test_age_exclusions_are_not_permanently_skipped(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"issue_key": "PROJ-1", "reason": "too_old: last updated 2024-06-10"},
        {"issue_key": "PROJ-2", "reason": "last updated 2024-05-21"},
    ])
    assert JiraFetcher.load_exclude_manifest(manifest) == set()


def test_noise_exclusions_are_permanently_skipped(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"issue_key": "PROJ-1", "reason": "noise_status: rejected"},
        {"issue_key": "PROJ-2", "reason": "noise_title: spike"},
        {"issue_key": "PROJ-3", "reason": "noise_label: wontfix"},
        {"issue_key": "PROJ-4", "reason": "noise_type: epic"},
    ])
    assert JiraFetcher.load_exclude_manifest(manifest) == {
        "PROJ-1", "PROJ-2", "PROJ-3", "PROJ-4",
    }


def test_mixed_manifest_returns_only_noise(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"issue_key": "PROJ-1", "reason": "low_word_count"},
        {"issue_key": "PROJ-2", "reason": "noise_status: rejected"},
        {"issue_key": "PROJ-3", "reason": "too_old: last updated 2024-06-10"},
    ])
    assert JiraFetcher.load_exclude_manifest(manifest) == {"PROJ-2"}


def test_missing_reason_is_not_permanent(tmp_path):
    # Legacy entries without a reason field default to dynamic (re-fetch).
    manifest = _write_manifest(tmp_path, [
        {"issue_key": "PROJ-1"},
    ])
    assert JiraFetcher.load_exclude_manifest(manifest) == set()


def test_missing_manifest_returns_empty_set(tmp_path):
    assert JiraFetcher.load_exclude_manifest(str(tmp_path / "nope.json")) == set()
