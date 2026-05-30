"""Tests for the shared excluded-manifest merge helper (M7)."""

from main.utils.manifest import merge_manifest_entries


class TestMergeManifestEntries:
    def test_appends_new_unseen_entries(self):
        existing = [{"page_id": "1", "title": "A"}]
        new = [{"page_id": "2", "title": "B"}]
        result = merge_manifest_entries(existing, new, "page_id")
        assert [e["page_id"] for e in result] == ["1", "2"]

    def test_dedups_already_present_id(self):
        existing = [{"page_id": "1", "title": "A"}]
        new = [{"page_id": "1", "title": "A-dup"}, {"page_id": "2", "title": "B"}]
        result = merge_manifest_entries(existing, new, "page_id")
        assert [e["page_id"] for e in result] == ["1", "2"]

    def test_existing_entry_missing_id_does_not_raise(self):
        # An on-disk manifest entry lacking the id field must not KeyError.
        existing = [{"title": "legacy entry with no page_id"}]
        new = [{"page_id": "2", "title": "B"}]
        result = merge_manifest_entries(existing, new, "page_id")
        assert len(result) == 2
        assert result[0] == {"title": "legacy entry with no page_id"}

    def test_empty_id_entries_are_not_collapsed(self):
        # Two new entries with empty ids carry no identity and must both survive
        # rather than collapse into one.
        existing = []
        new = [
            {"page_id": "", "title": "first empty"},
            {"page_id": "", "title": "second empty"},
        ]
        result = merge_manifest_entries(existing, new, "page_id")
        assert [e["title"] for e in result] == ["first empty", "second empty"]

    def test_missing_id_on_new_entry_is_kept(self):
        existing = [{"page_id": "1"}]
        new = [{"title": "no id field at all"}]
        result = merge_manifest_entries(existing, new, "page_id")
        assert len(result) == 2

    def test_empty_existing_keeps_all_new(self):
        new = [{"issue_key": "X-1"}, {"issue_key": "X-2"}]
        result = merge_manifest_entries([], new, "issue_key")
        assert result == new

    def test_does_not_mutate_inputs(self):
        existing = [{"notion_id": "a"}]
        new = [{"notion_id": "b"}]
        merge_manifest_entries(existing, new, "notion_id")
        assert existing == [{"notion_id": "a"}]
        assert new == [{"notion_id": "b"}]
