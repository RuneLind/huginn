"""Unit tests for build-time people aliasing.

Every name here is invented ("Ada Example", "Bo Tester"). The real map lives in
a gitignored private sub-repo and is never touched by the test suite; the
fixture below reproduces its schema exactly (v7), including the fully expanded,
longest-first `variants` list the runtime consumes verbatim.
"""
import json

import pytest

from main.privacy.alias_registry import AliasRegistry, PrivacyMapMissing, resolve_registry


def _entry(alias, name, variants, role="dev", require_full_name=False, extra=()):
    return {
        "alias": alias,
        "name": name,
        "role": role,
        "variants": sorted(set(variants) | set(extra), key=lambda s: (-len(s), s)),
        "require_full_name": require_full_name,
        "idents": [],
        "departed": False,
        "confirmed": True,
        "extra_variants": list(extra),
    }


MAP = {
    "version": 7,
    "entries": [
        _entry("dev-01", "Ada Example",
               ["Ada Example", "Ada Example [X]", "Example, Ada",
                "ada.example", "ada_example", "example.ada", "example_ada"]),
        _entry("fag-01", "Bo Tester",
               ["Bo Tester", "Bo Tester [X]", "Tester, Bo",
                "bo.tester", "bo_tester", "tester.bo", "tester_bo"], role="fag"),
        _entry("dev-02", "Åse Øygard",
               ["Åse Øygard", "Åse Øygard [X]", "Øygard, Åse",
                "åse.øygard", "åse_øygard", "øygard.åse", "øygard_åse",
                "Ase Oygard", "Ase Oygard [X]", "ase.oygard", "ase_oygard",
                "oygard.ase", "oygard_ase"]),
    ],
    "ident_policy": "redact",
    "non_person_labels": [
        "utsendt arbeidstaker",
        "Saksbehandler",
        "saksbehandler",
        "Testbruker Q000456",
        # Overlaps the *start* of a mapped person variant and is longer than the
        # bare surname; longest-first must let it win.
        "Tester, Bo og andre",
        # Contains a mapped person variant outright, so the exemption tests
        # below fail if exempt labels ever stop entering the alternation.
        "Bo Tester-rutinen",
        "srvtestbruker",
        "Sikkerhet",
    ],
    "unmapped_people": ["Kari Ukjent", "Zylphia Quorndal", "Zylphia"],
    "unmapped_people_variants": {
        "Kari Ukjent": ["Kari Ukjent [X]", "Ukjent, Kari", "Kari Ukjent",
                        "kari.ukjent", "kari_ukjent", "ukjent.kari", "ukjent_kari"],
        "Zylphia Quorndal": ["Zylphia Quorndal [X]", "Quorndal, Zylphia",
                             "Zylphia Quorndal", "zylphia.quorndal", "quorndal.zylphia"],
        "Zylphia": ["Zylphia [X]", "Zylphia"],
    },
    "person_redaction_token": "[~ukjent-person]",
    "retired_aliases": ["dev-99"],
}


@pytest.fixture
def registry():
    return AliasRegistry(MAP, ident_exceptions=["Q000456", "f000111"])


@pytest.fixture
def map_file(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(MAP), encoding="utf-8")
    return path


# --- substitution classes ---------------------------------------------------

def test_longest_first_prefers_the_bracket_form(registry):
    # "Ada Example" also matches inside "Ada Example [X]"; the longer variant
    # must win or the alias is followed by an orphaned "[X]".
    assert registry.apply("Skrevet av Ada Example [X] i går") == "Skrevet av dev-01 i går"


def test_case_insensitive(registry):
    assert registry.apply("ADA EXAMPLE og ada example") == "dev-01 og dev-01"


def test_comma_form(registry):
    assert registry.apply("Assignee: Tester, Bo") == "Assignee: fag-01"


def test_slug_and_dotted_handle_forms(registry):
    assert registry.apply("kontakt bo.tester eller bo_tester") == "kontakt fag-01 eller fag-01"
    assert registry.apply("kontakt @ada.example") == "kontakt @dev-01"


def test_dotted_name_inside_url_encoding(registry):
    # A `\w` lookbehind here (the `C` of `%2C`) left a real full name in the clear.
    assert "dev-01" in registry.apply("ovuser=abc%2CAda.Example%40nav.no&OR=Teams")


def test_slug_not_matched_inside_a_longer_dotted_path(registry):
    # Matching here would weld the surviving tail onto the alias — the r9 lesson.
    assert registry.apply("no.nav.ada.example.Klasse") == "no.nav.ada.example.Klasse"
    assert registry.apply("bo.tester.example sa") == "bo.tester.example sa"


def test_sentence_final_period_does_not_block_a_slug(registry):
    # A `.` that ends a sentence is not another path segment.
    assert registry.apply("Kommentar fra bo.tester.") == "Kommentar fra fag-01."
    assert registry.apply("Kommentar fra bo_tester.") == "Kommentar fra fag-01."


def test_folded_scandinavian_forms(registry):
    assert registry.apply("Åse Øygard og ase.oygard") == "dev-02 og dev-02"
    assert registry.apply("Øygard, Åse") == "dev-02"


def test_sentence_final_period_does_not_block_a_bare_token(registry):
    assert registry.apply("Jeg sjekker med Zylphia.") == "Jeg sjekker med [~ukjent-person]."


def test_unmapped_person_is_redacted_not_aliased(registry):
    assert registry.apply("Kari Ukjent svarte") == "[~ukjent-person] svarte"
    assert registry.apply("ukjent.kari svarte") == "[~ukjent-person] svarte"


def test_non_person_label_is_exempt(registry):
    # Non-vacuous on purpose: "Bo Tester-rutinen" CONTAINS a mapped person
    # variant, so dropping exempt labels from the alternation aliases inside it.
    text = ("En utsendt arbeidstaker og en saksbehandler følger "
            "Bo Tester-rutinen, jf. Saksbehandler.")
    assert registry.apply(text) == text


def test_non_person_label_wins_over_shorter_overlapping_person_variant(registry):
    # "Tester, Bo" (person) starts at the same offset as the longer exempt label.
    assert registry.apply("Tester, Bo og andre møtte") == "Tester, Bo og andre møtte"


def test_word_boundaries_do_not_eat_substrings(registry):
    assert registry.apply("srvtestbrukerbruker") == "srvtestbrukerbruker"


@pytest.mark.parametrize("separator", ["\n", "  ", "\t", " ", " \n  ", "%20"])
def test_multi_token_variant_matches_across_any_whitespace_run(registry, separator):
    # A single literal space between the tokens was the only form that matched;
    # a hard-wrapped comment, a double space or a percent-encoded URL left the
    # full name in the clear.
    assert registry.apply(f"av Ada{separator}Example her") == "av dev-01 her"


def test_bare_token_variant_is_not_welded_onto_a_dotted_path(registry):
    # The slug alternative `zylphia.quorndal` is blocked by the preceding dot;
    # without the same left boundary on the bare token the engine falls through
    # to `Zylphia` and produces "no.nav.[~ukjent-person].quorndal.Klasse".
    assert registry.apply("no.nav.zylphia.quorndal.Klasse") == "no.nav.zylphia.quorndal.Klasse"
    assert registry.apply("pakke-Zylphia") == "pakke-Zylphia"
    # …but a sentence boundary still redacts (the case the left boundary must not eat).
    assert registry.apply("Jeg sjekket med Zylphia") == "Jeg sjekket med [~ukjent-person]"


def test_exempt_label_survives_a_unicode_case_fold(registry):
    # `ſ` matches `s` under IGNORECASE but str.lower() leaves it alone, so a
    # lower()-keyed replacement table missed the label and redacted a role noun.
    assert registry.apply("en ſaksbehandler skrev") == "en ſaksbehandler skrev"


def test_unresolvable_match_is_left_alone_rather_than_redacted(registry):
    # Dotless ı matches `i` under IGNORECASE and casefold() does not fold it, so
    # the lookup misses. Passing the text through is the only safe direction:
    # redacting turns an unknown word into a person claim.
    assert registry.apply("Sıkkerhet er viktig") == "Sıkkerhet er viktig"


# --- idents -----------------------------------------------------------------

def test_bare_ident_redacted(registry):
    assert registry.apply("BrukerId Q000124 mangler") == "BrukerId [~person] mangler"


def test_wrapped_ident_consumes_the_wrapper(registry):
    assert registry.apply("Beklager [~Q000124] jeg glemte") == "Beklager [~person] jeg glemte"


def test_ident_exception_token_untouched(registry):
    # `f000111` is deliberately NOT part of any non_person_label: if it were, the
    # exempt-label alternative would keep it whole and the assertion would say
    # nothing about the ident-exception mechanism it is here to pin.
    assert registry.apply("commit: f000111") == "commit: f000111"
    assert "f000111" not in " ".join(MAP["non_person_labels"])
    assert registry.apply("Testbruker Q000456 er opprettet") == "Testbruker Q000456 er opprettet"


def test_ident_exception_matching_is_case_insensitive(registry):
    # The corpus writes git SHAs lowercase and document-type ids uppercase; one
    # spelling in the exceptions file must cover both.
    assert registry.apply("commit: F000111") == "commit: F000111"
    # A token that is NOT exempt still redacts, whatever its case.
    assert registry.apply("BrukerId x123456") == "BrukerId [~person]"


def test_ident_shape_is_letter_plus_six_digits(registry):
    assert registry.apply("versjon Q00012 og Q0001234") == "versjon Q00012 og Q0001234"


# --- dotted handles ---------------------------------------------------------

def test_unmapped_dotted_handle_becomes_person(registry):
    assert registry.apply("ping @ola.nordmann.example") == "ping @person"


def test_tld_terminated_handles_are_not_people(registry):
    assert registry.apply("send til @nav.no og @trygdeetaten.no") == "send til @nav.no og @trygdeetaten.no"
    assert registry.apply("konto @nav.no-konto") == "konto @nav.no-konto"


def test_package_and_annotation_paths_excluded(registry):
    text = "@org.springframework.web.bind.annotation.RequestParam og @Abac.Attr"
    assert registry.apply(text) == text


def test_versions_are_not_handles(registry):
    assert registry.apply("pakke@1.0.0") == "pakke@1.0.0"
    assert registry.apply("bruk @v1.2.3 taggen") == "bruk @v1.2.3 taggen"


@pytest.mark.parametrize("text", [
    "@org.junit.Test",                       # two-segment package path, was redacted
    "@com.example.Foo",
    "@nav.no",
    "kontakt ola@firma.internal her",        # unknown TLD, and @ is preceded by \w
])
def test_handle_shapes_that_are_not_people(registry, text):
    assert registry.apply(text) == text


@pytest.mark.parametrize("text,expected", [
    ("ping @Ola.Nordmann", "ping @person"),          # capitalised: was not matched at all
    ("ping @ola.nordmann2", "ping @person"),         # digit inside a segment
    ("skriv til ola.nordmann@nav.no", "skriv til person@nav.no"),
])
def test_handles_and_dotted_email_locals_are_redacted(registry, text, expected):
    assert registry.apply(text) == expected


# --- apply_document ---------------------------------------------------------

def _document():
    return {
        "id": "Team/Ada Example/notat.md",
        "url": "file:///srv/data/Team/Ada Example/notat.md",
        "modifiedTime": "2026-01-01T00:00:00",
        "text": "[Team] Ada Example skrev dette. Ident Q000124.",
        "metadata": {"title": "Notat fra Ada Example",
                     "tags": ["rapportering", "Bo Tester"],
                     "relevance_score": 0.42},
        "chunks": [
            {"indexedData": "[Team]\n## Ada Example\nTester, Bo kommenterte.",
             "heading": "Ada Example",
             "metadata": {"title": "Notat fra Ada Example", "labels": ["Kari Ukjent"]}},
        ],
    }


def test_apply_document_never_touches_id_or_url(registry):
    document = _document()
    assert registry.apply_document(document) is True
    assert document["id"] == "Team/Ada Example/notat.md"
    assert document["url"] == "file:///srv/data/Team/Ada Example/notat.md"
    assert document["modifiedTime"] == "2026-01-01T00:00:00"


def test_apply_document_aliases_text_chunks_and_metadata(registry):
    document = _document()
    registry.apply_document(document)
    assert document["text"] == "[Team] dev-01 skrev dette. Ident [~person]."
    assert document["metadata"]["title"] == "Notat fra dev-01"
    assert document["metadata"]["tags"] == ["rapportering", "fag-01"]
    assert document["metadata"]["relevance_score"] == 0.42
    chunk = document["chunks"][0]
    assert chunk["indexedData"] == "[Team]\n## dev-01\nfag-01 kommenterte."
    assert chunk["heading"] == "dev-01"
    assert chunk["metadata"]["labels"] == ["[~ukjent-person]"]


def test_apply_document_reports_no_change_for_clean_text(registry):
    document = {"id": "a.md", "url": "file://a.md", "text": "Ingen navn her.",
                "chunks": [{"indexedData": "Ingen navn her."}]}
    assert registry.apply_document(document) is False


def test_apply_document_walks_nested_metadata(registry):
    """Metadata is arbitrary JSON, not just str and list-of-str.

    The all-str list guard skipped a mixed list wholesale, and neither a nested
    dict nor a list of dicts was ever visited — the Jira reader puts comment
    dicts there.
    """
    document = {
        "id": "a.md", "url": "file://a.md", "text": "ingenting",
        "metadata": {
            "mixed": ["Ada Example", 7, None, "Bo Tester"],
            "comments": [{"author": "Ada Example", "score": 3}],
            "nested": {"owner": {"name": "Bo Tester"}, "count": 2},
            "Ada Example": "Bo Tester",
        },
        "chunks": [],
    }
    assert registry.apply_document(document) is True
    metadata = document["metadata"]
    assert metadata["mixed"] == ["dev-01", 7, None, "fag-01"]
    assert metadata["comments"] == [{"author": "dev-01", "score": 3}]
    assert metadata["nested"] == {"owner": {"name": "fag-01"}, "count": 2}
    # keys are join keys for downstream consumers; values only
    assert metadata["Ada Example"] == "fag-01"


@pytest.mark.parametrize("text", [
    "Ada Example og Bo Tester og Kari Ukjent",
    "Ident [~Q000124] og Q000124 og f000111",
    "ping @ola.nordmann.example og ola.nordmann@nav.no",
    "dev-01 fag-01 [~person] [~ukjent-person] @person",
    "no.nav.zylphia.quorndal.Klasse og en saksbehandler",
])
def test_apply_is_idempotent(registry, text):
    once = registry.apply(text)
    assert registry.apply(once) == once


def test_manifest_stamp(registry):
    stamp = registry.manifest_stamp()
    assert stamp["policy_version"] == 1
    assert stamp["map_version"] == 7
    assert stamp["aliasedAt"]


# --- loading and scoping ----------------------------------------------------

def test_missing_ident_exceptions_file_redacts_everything(map_file, tmp_path):
    # No exceptions file (the normal state of a clone without private sub-repos):
    # every ident-shaped token redacts. Fail-safe direction.
    registry = AliasRegistry.load(str(map_file),
                                  ident_exceptions_path=str(tmp_path / "absent.json"))
    assert registry.apply("commit: f000111") == "commit: [~person]"


def test_load_reads_ident_exceptions_file(map_file, tmp_path):
    exceptions = tmp_path / "ident_exceptions.json"
    exceptions.write_text(json.dumps({"version": 1, "tokens": ["f000111"]}), encoding="utf-8")
    registry = AliasRegistry.load(str(map_file), ident_exceptions_path=str(exceptions))
    assert registry.apply("commit: f000111") == "commit: f000111"


def _mutated(**changes):
    data = json.loads(json.dumps(MAP))
    for key, value in changes.items():
        data[key] = value
    return data


def _entries_with_variant(variant):
    entries = json.loads(json.dumps(MAP["entries"]))
    entries[0]["variants"].append(variant)
    return entries


@pytest.mark.parametrize("bad_map", [
    # (i) blank literals: an empty alternative matches at every position.
    _mutated(entries=_entries_with_variant("")),
    _mutated(entries=_entries_with_variant("   ")),
    _mutated(non_person_labels=[*MAP["non_person_labels"], ""]),
    _mutated(unmapped_people_variants={**MAP["unmapped_people_variants"], "X": [""]}),
    # (ii) an emptied map must not build a stamped, name-free-looking index.
    _mutated(entries=[]),
    # (iii) a bare given name as an entry variant substitutes half the corpus.
    _mutated(entries=_entries_with_variant("Ada")),
    # (iv) schema drift, caught as a privacy failure rather than AttributeError.
    _mutated(unmapped_people_variants=["Kari Ukjent"]),
    _mutated(entries=[{"alias": "dev-01"}]),
    _mutated(entries=[{"variants": ["Ada Example"]}]),
])
def test_invalid_map_refuses_to_compile(bad_map):
    with pytest.raises(PrivacyMapMissing):
        AliasRegistry(bad_map)


def test_invalid_map_fails_closed_through_resolve_registry(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(_mutated(entries=[])), encoding="utf-8")
    with pytest.raises(PrivacyMapMissing):
        resolve_registry("jira-issues", None, map_path=str(path))


def test_two_discovered_maps_are_ambiguous_rather_than_first_wins(tmp_path, monkeypatch):
    for name in ("huginn-a", "huginn-b"):
        privacy = tmp_path / name / "privacy"
        privacy.mkdir(parents=True)
        (privacy / "aliases.json").write_text(json.dumps(MAP), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PrivacyMapMissing, match="ambiguous"):
        resolve_registry("jira-issues", None)


def test_out_of_scope_collection_returns_none(tmp_path):
    assert resolve_registry("some-other-collection", str(tmp_path)) is None


def test_in_scope_collection_without_a_map_raises(map_file):
    with pytest.raises(PrivacyMapMissing):
        resolve_registry("jira-issues", None, map_path=str(map_file.parent / "missing.json"))


def test_in_scope_collection_with_a_map_arms(map_file):
    registry = resolve_registry("jira-issues", None, map_path=str(map_file))
    assert registry is not None and registry.map_version == 7


def test_manifest_stamp_arms_an_otherwise_out_of_scope_collection(map_file):
    registry = resolve_registry("not-in-any-scope-file", "/nowhere",
                                armed_by_manifest=True, map_path=str(map_file))
    assert registry is not None


def test_public_scope_lists_the_three_campaign_collections():
    from main.privacy.alias_registry import load_scope
    collections, _ = load_scope()
    assert {"melosys-confluence-v3", "jira-issues", "nav-wiki"} <= collections
