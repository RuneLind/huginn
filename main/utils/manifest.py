"""Shared helper for merging excluded-manifest entries across cleanup adapters.

The confluence/jira/notion cleanup scripts each append entries to an
`excluded_manifest.json` and dedup against the existing file by an id field
(`page_id` / `issue_key` / `notion_id`). This centralises the merge so the
dedup is correct in one place: it tolerates on-disk entries missing the id key,
and never collapses entries that have a missing or empty id into one another.
"""


def merge_manifest_entries(existing, new_entries, id_field):
    """Return existing entries plus the new ones not already present by id_field.

    Dedup is keyed on a non-empty id_field value only. An entry whose id is
    missing or empty is always kept — such entries carry no identity, so
    collapsing them by an empty key would silently drop distinct records.

    Args:
        existing: list of entry dicts already on disk (may lack id_field).
        new_entries: list of freshly produced entry dicts.
        id_field: the key used as the dedup identity (e.g. "page_id").
    """
    seen_ids = {e.get(id_field, "") for e in existing}
    seen_ids.discard("")
    additions = [
        e for e in new_entries
        if not e.get(id_field) or e.get(id_field) not in seen_ids
    ]
    return list(existing) + additions
