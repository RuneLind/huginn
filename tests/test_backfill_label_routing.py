"""Label routing for the log backfill.

The backfill maps a log marker label ("Daily Jira") to the collections its run
wrote. Labels whose collection this repo's public CLAUDE.md already names are
compiled in; every other one arrives from a gitignored ``huginn-*/scripts/
schedule_routing.json``, the same files and the same precedent
``indexing_schedule.load_script_collections`` uses.

That input file is gitignored, so it can never be reviewed in a PR — which is
exactly why the merge logic around it is tested here. Mirrors
``tests/test_indexing_schedule.py::TestRoutingLivesOutsideThisRepo``; the two
loaders are deliberately kept behaviourally identical, so a divergence found
here is a bug in one of them rather than a difference of intent.
"""

import importlib.util
import json
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_backfill_module():
    """Import the script by path — ``scripts/`` is not an importable package."""
    path = os.path.join(_REPO_ROOT, "scripts", "backfill_indexing_runs.py")
    spec = importlib.util.spec_from_file_location("backfill_indexing_runs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backfill = _load_backfill_module()


def _write_routing(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"labelCollections": labels}, handle)


def _globs(tmp_path):
    return (str(tmp_path / "huginn-*" / "scripts" / "schedule_routing.json"),)


class TestNoPrivateNameIsCompiledIn:
    def test_the_public_table_names_only_publicly_documented_collections(self):
        """huginn is a public repo; an employer- or customer-derived collection
        name must not be readable in it. The compiled table is the one place
        that could regress silently, so it is pinned."""
        assert backfill.LABEL_COLLECTIONS == {
            "Daily Jira": ["jira-issues"],
            "Daily Confluence": ["melosys-confluence-v3"],
        }


class TestRoutingLivesOutsideThisRepo:
    def test_routing_files_supply_extra_labels(self, tmp_path):
        _write_routing(str(tmp_path / "huginn-x" / "scripts" / "schedule_routing.json"),
                       {"Daily Notion": ["some-notion"]})
        mapping = backfill.load_label_collections(globs=_globs(tmp_path))
        assert mapping["Daily Notion"] == ["some-notion"]

    def test_public_defaults_survive_alongside_routed_labels(self, tmp_path):
        _write_routing(str(tmp_path / "huginn-x" / "scripts" / "schedule_routing.json"),
                       {"Daily Notion": ["some-notion"]})
        mapping = backfill.load_label_collections(globs=_globs(tmp_path))
        assert mapping["Daily Jira"] == ["jira-issues"]

    def test_several_sub_repos_merge(self, tmp_path):
        _write_routing(str(tmp_path / "huginn-a" / "scripts" / "schedule_routing.json"),
                       {"Label A": ["ca"]})
        _write_routing(str(tmp_path / "huginn-b" / "scripts" / "schedule_routing.json"),
                       {"Label B": ["cb1", "cb2"]})
        mapping = backfill.load_label_collections(globs=_globs(tmp_path))
        assert mapping["Label A"] == ["ca"]
        assert mapping["Label B"] == ["cb1", "cb2"]

    def test_a_routed_label_overrides_a_public_default(self, tmp_path):
        """Deliberate: a deployment can repoint a public label at its own
        collection without editing this repo."""
        _write_routing(str(tmp_path / "huginn-a" / "scripts" / "schedule_routing.json"),
                       {"Daily Jira": ["other-jira"]})
        mapping = backfill.load_label_collections(globs=_globs(tmp_path))
        assert mapping["Daily Jira"] == ["other-jira"]

    def test_a_malformed_routing_file_costs_only_itself(self, tmp_path, capsys):
        good = tmp_path / "huginn-a" / "scripts" / "schedule_routing.json"
        _write_routing(str(good), {"Label A": ["ca"]})
        bad = tmp_path / "huginn-b" / "scripts" / "schedule_routing.json"
        os.makedirs(os.path.dirname(str(bad)), exist_ok=True)
        bad.write_text("{not json", encoding="utf-8")

        mapping = backfill.load_label_collections(globs=_globs(tmp_path))

        assert mapping["Label A"] == ["ca"]
        assert mapping["Daily Jira"] == ["jira-issues"]
        # Silence here would mean a whole label family missing from the backfill
        # while the run still exits 0.
        assert "warning" in capsys.readouterr().err

    def test_no_routing_file_degrades_to_the_public_defaults(self, tmp_path):
        assert backfill.load_label_collections(globs=_globs(tmp_path)) == \
            backfill.LABEL_COLLECTIONS

    @pytest.mark.parametrize("labels", [
        {"Daily Jira": []},
        {"Daily Jira": [""]},
        {"Daily Jira": ["   "]},
    ])
    def test_an_emptied_entry_cannot_clobber_a_public_default(self, tmp_path, labels):
        """An accidentally-blanked list would otherwise drop every Jira start
        marker from the backfill with no error at all — silent data loss.
        ``load_script_collections`` guards the same way."""
        _write_routing(str(tmp_path / "huginn-a" / "scripts" / "schedule_routing.json"),
                       labels)
        mapping = backfill.load_label_collections(globs=_globs(tmp_path))
        assert mapping["Daily Jira"] == ["jira-issues"]

    def test_non_string_members_are_skipped_not_coerced(self, tmp_path):
        """str()-coercing a nested value produces a plausible-looking name that
        becomes a real ledger filename downstream (``['nested'].jsonl``)."""
        _write_routing(str(tmp_path / "huginn-a" / "scripts" / "schedule_routing.json"),
                       {"Label A": [["nested"], {"a": 1}, "real"]})
        mapping = backfill.load_label_collections(globs=_globs(tmp_path))
        assert mapping["Label A"] == ["real"]

    def test_a_non_dict_label_block_is_ignored(self, tmp_path):
        path = tmp_path / "huginn-a" / "scripts" / "schedule_routing.json"
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        path.write_text(json.dumps({"labelCollections": ["not", "a", "dict"]}),
                        encoding="utf-8")
        assert backfill.load_label_collections(globs=_globs(tmp_path)) == \
            backfill.LABEL_COLLECTIONS


class TestCollectionsForUsesTheMergedMapping:
    def test_an_explicit_marker_argument_still_wins(self):
        """``collection=`` in the marker text outranks any routing."""
        assert backfill._collections_for("Daily Jira", "collection=explicit") == ["explicit"]

    def test_a_label_with_no_mapping_yields_nothing(self):
        assert backfill._collections_for("Totally Unknown Label", None) == []
