"""Unit tests for build-time people aliasing.

Every name here is invented ("Ada Example", "Bo Tester"). The real map lives in
a gitignored private sub-repo and is never touched by the test suite; the
fixture below reproduces its schema exactly (v7), including the fully expanded,
longest-first `variants` list the runtime consumes verbatim.
"""
import json

import pytest

from main.privacy import alias_registry
from main.privacy.alias_registry import (
    AliasRegistry, PrivacyMapInvalid, PrivacyMapMissing, resolve_registry,
)


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
    # …but a sentence boundary still redacts (the case the left boundary must not eat).
    assert registry.apply("Jeg sjekket med Zylphia") == "Jeg sjekket med [~ukjent-person]"


def test_single_token_left_boundary_blocks_only_a_dot_and_word_characters(registry):
    """A hyphen or a bracket in front of a mononym is punctuation, not a path.

    The boundary exists to stop `no.nav.zylphia.…` welding a surviving path tail
    onto the redaction token; blocking `-` as well made a leading hyphen (a list
    bullet, a diff marker, a compound with a preceding word broken across a line)
    a free pass for a real name.
    """
    assert registry.apply("-Zylphia") == "-[~ukjent-person]"
    assert registry.apply("(Zylphia") == "([~ukjent-person]"
    assert registry.apply("team.Zylphia") == "team.Zylphia"


@pytest.mark.parametrize("text", [
    "https://x/?f=%2CAda%20Example%40nav.no",
    "https://x/?q=fra%20Ada%20Example",
])
def test_multi_token_variant_matches_across_percent_encoded_punctuation(registry, text):
    """`%2C`/`%40` around a percent-encoded name are not word characters to a
    reader, but they are to `\\w` — `(?<!\\w)` saw the `C` of `%2C` and refused."""
    result = registry.apply(text)
    assert "dev-01" in result
    assert "Ada" not in result and "Example" not in result


def test_variant_separator_spans_at_most_one_newline(registry):
    # One hard wrap is the same name; a blank line is a paragraph break, and
    # welding across it joined the end of one sentence to the start of the next.
    assert registry.apply("Ada\nExample") == "dev-01"
    assert registry.apply("Ada\n\nExample") == "Ada\n\nExample"


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
    text = "@org.springframework.web.bind.annotation.RequestParam"
    assert registry.apply(text) == text


@pytest.mark.parametrize("text", [
    "@lombok.Setter",
    "@mockito.Captor",
    "@dagger.Provides",
    "@org.junit.Test",
])
def test_code_annotations_are_not_people(registry, text):
    """A lowercase root followed by a CamelCase segment is a Java/Kotlin
    annotation, whatever the root is — the annotation vocabulary is open-ended
    (`@lombok.Setter`, `@dagger.Provides`) and a root allow-list cannot track it."""
    assert registry.apply(text) == text


@pytest.mark.parametrize("root", ["jakarta", "kotlin", "android", "net"])
def test_package_roots_are_honoured_even_when_fully_lowercase(registry, root):
    # No CamelCase segment here, so the annotation rule does not fire and only
    # the root list keeps these out of the person branch.
    text = f"@{root}.internal.pakkenavn"
    assert registry.apply(text) == text


@pytest.mark.parametrize("host", ["nav.no", "acme.se", "acme.dk", "acme.fi", "acme.de",
                                  "acme.uk", "acme.eu", "acme.fr", "acme.nl",
                                  "svc.internal", "svc.local", "svc.test", "svc.dev",
                                  "svc.localhost"])
def test_domain_terminated_handles_are_not_people(registry, host):
    assert registry.apply(f"send til @{host} her") == f"send til @{host} her"


@pytest.mark.parametrize("text,expected", [
    ("ping @Ola.Nordmann", "ping @person"),
    ("ping @ola.nordmann2", "ping @person"),
    # The three the removed "last segment <=4 alphabetic chars is a TLD"
    # heuristic silently classified as domains — Norwegian surnames are short.
    ("ping @ola.berg", "ping @person"),
    ("ping @kari.moe", "ping @person"),
    ("ping @per.aas", "ping @person"),
])
def test_short_surname_handles_are_people_not_domains(registry, text, expected):
    assert registry.apply(text) == expected


@pytest.mark.parametrize("text", [
    "np.eye@vec",          # numpy matmul, not an address
    "A.T@B",
    "self.w@x",
])
def test_matrix_multiplication_is_not_an_email(registry, text):
    """`@` between two dotted expressions is Python's matmul operator. Requiring
    a real domain on the right is what tells the two apart."""
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
    ("skriv til ola.nordmann@nav.no", "skriv til person@nav.no"),
    ("skriv til ola.nordmann@firma.internal", "skriv til person@firma.internal"),
])
def test_dotted_email_locals_are_redacted(registry, text, expected):
    assert registry.apply(text) == expected


# --- stranded handle tails --------------------------------------------------
#
# A SINGLE-token variant leading a dotted handle used to be substituted in
# place, and the remainder — `@[~ukjent-person].mellomnavn.til` — no longer
# matched _HANDLE_RE, whose first segment must be alphanumeric. So the pass that
# exists to collapse handles never saw it and the rest of that person's name
# shipped. The registry now declines to substitute there, but ONLY when it has
# just confirmed that the handle pass collapses the whole handle instead.
#
# The probe list below is written from FAILURE CLASSES, not from remembered
# shapes: the five defects `/code-review` found in the first design (exempt
# labels, ident exceptions, prose with a missing space, email domains, greedy
# tails eating extensions/dates/another person's alias), plus the ordering hole
# that a substitution inside the tail opens. Twenty tests written from shapes
# missed all five.


@pytest.mark.parametrize("text,expected", [
    # THE shape, both channels. Nothing of the name survives: the head is not
    # substituted, and the handle pass then eats head and tail together.
    ("ping @Zylphia.mellomnavn.til", "ping @person"),
    ("ping @Zylphia.mellomnavn", "ping @person"),
    ("eg trur @Zylphia.mellomnavn.til har ein prosess", "eg trur @person har ein prosess"),
])
def test_a_bare_token_leading_a_handle_collapses_whole(registry, text, expected):
    assert registry.apply(text) == expected
    assert "Zylphia" not in registry.apply(text)
    assert "mellomnavn" not in registry.apply(text)


def test_only_the_redaction_channels_can_reach_the_guard(registry):
    """A mapped ENTRY cannot have a bare single-token variant — `_validate`
    refuses one outright — so the head of a dotted handle is only ever an
    unmapped-person variant or an ident. A hyphenated compound, the one bare
    entry-variant shape the map does allow, is unreachable from the other side:
    `_HANDLE_RE` has no `-` in its segment class, so no handle is found and the
    substitution happens exactly as before.
    """
    with pytest.raises(PrivacyMapInvalid):
        AliasRegistry({**MAP, "entries": [_entry("dev-09", "Zed Quorndalsen",
                                                 ["Zed Quorndalsen", "Quorndalsen"])]})
    data = json.loads(json.dumps(MAP))
    data["unmapped_people_variants"]["Nord-Hansen"] = ["Nord-Hansen"]
    reg = AliasRegistry(data, ident_exceptions=["Q000456", "f000111"])
    assert reg.apply("@Nord-Hansen.mellomnavn") == "@[~ukjent-person].mellomnavn"


def test_a_literal_in_the_TAIL_defeats_the_collapse_so_both_are_substituted(registry):
    # The ordering hole: the decline is decided on the text as it is, but this
    # same pass is still scanning and will rewrite `Ada Example` in the tail.
    # `@Zylphia.dev-01` ends in the label `dev`, which _HANDLE_TLDS calls a
    # host, so the handle pass would NOT collapse it and `Zylphia` would ship.
    out = registry.apply("@Zylphia.Ada Example")
    assert out == "@[~ukjent-person].dev-01"
    assert "Zylphia" not in out


def test_a_redacted_ident_in_the_tail_collapses_with_the_handle(registry):
    # Both outcomes are redactions, so the whole handle is simply the stronger
    # one. An ident EXCEPTION is the opposite case and must survive — below.
    assert registry.apply("@Zylphia.Q000124") == "@person"


def test_an_ident_exception_in_the_tail_refuses_the_collapse(registry):
    # Gate check 3b. `Q000456` is an ident exception in the fixture, and
    # collapsing the handle would delete it.
    assert registry.apply("@Zylphia.Q000456") == "@[~ukjent-person].Q000456"


def test_an_exempt_label_in_the_tail_refuses_the_collapse(registry):
    """Gate check 5, and the defect that killed the first version of this fix.

    `boundaried()`'s bare branch blocks a preceding `.`, so a tail segment can
    never be matched by the pattern — the label is invisible from inside the
    substituter while check 5's own needle sees it perfectly. Looked up segment
    by segment for exactly that reason.
    """
    for label in ("saksbehandler", "srvtestbruker", "Sikkerhet"):
        assert registry.apply(f"@Zylphia.{label}") == f"@[~ukjent-person].{label}"


def test_a_variant_ending_in_punctuation_before_the_at_does_not_leak(registry):
    r"""The boundary flip. `_HANDLE_RE`'s `(?<![\w.])` was read against the text
    as it was and would have been enforced against the text as it became:
    `Ada Example [X]` becomes `dev-01`, whose `1` blocks the lookbehind, so the
    collapse never fired and the name shipped in full. 203 of the live map's
    1203 variants end in a non-word character.
    """
    for lead, alias in (("Ada Example [X]", "dev-01"), ("Bo Tester [X]", "fag-01"),
                        ("Åse Øygard [X]", "dev-02")):
        out = registry.apply(f"{lead}@Zylphia.mellomnavn")
        assert out == f"{alias}@person"
        assert "Zylphia" not in out
    # ...and the result is stable, which it was not: applying twice used to
    # redact what the first pass left behind.
    text = "Ada Example [X]@Zylphia.mellomnavn"
    assert registry.apply(registry.apply(text)) == registry.apply(text)


@pytest.mark.parametrize("text", [
    # `_HANDLE_RE`'s segment class is `[a-zæøå0-9]`, so the match stops inside
    # the token and collapsing it would weld `@person` onto the rest.
    "@Zylphia.müller",
    "@Zylphia.mellomnavn.MÜLLER",
])
def test_a_handle_the_regex_truncated_is_never_collapsed(registry, text):
    out = registry.apply(text)
    assert out.startswith("@[~ukjent-person].")
    assert "@person" not in out


def _registry_with(labels=(), exceptions=("Q000456", "f000111")):
    data = json.loads(json.dumps(MAP))
    data["non_person_labels"] = sorted({*data["non_person_labels"], *labels})
    return AliasRegistry(data, ident_exceptions=list(exceptions))


@pytest.mark.parametrize("tail", ["nord-hansen", "Nord-Hansen", "o'brien",
                                  "mellomnavn-til", "müller"])
def test_a_segment_the_handle_regex_cannot_finish_is_never_collapsed(registry, tail):
    """`_HANDLE_RE`'s `[a-zæøå0-9]` stops at a hyphen and an apostrophe as surely
    as at `ü`, and a Norwegian surname contains both. Collapsing there welds
    `@person` onto the rest and leaves half the surname in the clear — the exact
    corruption the truncation guard exists to refuse, in the shapes its first
    version did not cover."""
    out = registry.apply(f"@Zylphia.{tail}")
    assert out == f"@[~ukjent-person].{tail}"
    assert "@person" not in out


@pytest.mark.parametrize("tail,why", [
    ("nord\u2011hansen", "non-breaking hyphen"),
    ("nord\u2013hansen", "en dash, what a word processor autocorrects a hyphen to"),
    ("mu\u0308ller", "NFD, which is what macOS hands back"),
    ("o\u2018brien", "typographic quote"),
    ("nord\u00adhansen", "SOFT HYPHEN, i.e. &shy; from any HTML source"),
    ("nord\u200chansen", "zero-width non-joiner"),
    ("nord\u02dahansen", "RING ABOVE, a spacing diacritic"),
])
def test_a_token_continuing_in_a_character_class_no_literal_set_covers(registry, tail, why):
    r"""`_TOKEN_CONTINUES` as a hand-typed string covered `-` and `'` and missed
    every combining mark, every non-ASCII dash, every other quote form and every
    invisible — each of which continues a name token where `[a-zæøå0-9]` stops,
    and none of which `\w` recognises. Deciding by Unicode category does not
    prove the set closed; it moves the default to refusing, which is the
    behaviour that already exists, so a missing entry costs a weld and a
    surplus one costs nothing.
    """
    out = registry.apply(f"@zylphia.{tail} er her.")
    assert out.startswith("@[~ukjent-person].")
    assert "@person" not in out


def test_a_multi_token_ident_exception_crossing_the_edge_refuses_the_collapse():
    """An exempt label crossing the handle's right edge is caught by the name
    loop, because labels are in `_pattern`. Ident exceptions are not in
    `_pattern` at all, so a `search` bounded at `end` saw nothing and check 3b
    broke silently. The exceptions file is documented to hold test users."""
    reg = AliasRegistry(MAP, ident_exceptions=["Q000456", "srv testbruker"])
    assert (reg.apply("@zylphia.srv testbruker kjorte.")
            == "@[~ukjent-person].srv testbruker kjorte.")


def test_an_ident_exception_that_is_not_ident_shaped_still_refuses_the_collapse():
    r"""Gate check 3b needles every exception literally. `_load_ident_exceptions`
    accepts any string and its docstring says the file holds git SHAs and test
    users, so keying the guard on `[A-Za-z]\d{6}` asked a narrower question than
    the check does and deleted the rest."""
    reg = _registry_with(exceptions=("Q000456", "srvtestuser", "deadbeef"))
    for token in ("srvtestuser", "deadbeef", "Q000456"):
        assert reg.apply(f"@Zylphia.{token}") == f"@[~ukjent-person].{token}"


def test_an_exempt_label_spelled_across_a_dot_still_refuses_the_collapse():
    r"""Check 5's needle is `(?<!\w)label(?!\w)`, which spans dot boundaries; a
    segment-by-segment lookup cannot see such a label at all."""
    reg = _registry_with(labels=("srv.testbruker",))
    assert (reg.apply("meld til @Zylphia.srv.testbruker i dag")
            == "meld til @[~ukjent-person].srv.testbruker i dag")


@pytest.mark.parametrize("gap", [1, 5, 40, 200, 5000])
def test_a_variant_crossing_the_edge_is_seen_at_any_separator_length(registry, gap):
    r"""Why the scan is anchored to the handle rather than bounded by a constant
    reach past it. `VARIANT_SEPARATOR` is `[^\S\n]+` and the ident wrapper
    carries `\s*` — both unbounded — so no constant covers a variant that
    crosses the handle's right edge. A 200-space run defeated a
    `longest * 3 + 32` window: the crossing match went unseen, the handle
    collapsed, and the surname shipped in the clear.
    """
    out = registry.apply("@Zylphia.Ada" + " " * gap + "Example")
    assert out.startswith("@[~ukjent-person].dev-01")
    assert "Example" not in out


def test_the_guard_is_bounded_by_the_handle_not_by_the_document(registry):
    r"""`finditer(text, at)` with a `break` still scans forward to the next match
    anywhere, so one mention-dense document made the name pass quadratic —
    measured at 97x on a 100 KB page, unbounded above it, with no timeout and no
    error. The window is now a constant past the handle.
    """
    import time
    filler = "lorem ipsum dolor sit amet " * 400
    short = "@olav.hansen " * 40 + filler
    long_ = "@olav.hansen " * 40 + filler * 40
    def elapsed(text):
        start = time.perf_counter()
        registry.apply(text)
        return time.perf_counter() - start
    elapsed(short)                                    # warm
    ratio = elapsed(long_) / max(elapsed(short), 1e-6)
    # Linear in document length would be ~40x; quadratic in handles x length was
    # far worse. Generous ceiling so this measures the class, not the machine.
    assert ratio < 120, f"apply() grew {ratio:.0f}x for 40x the text"


def test_the_probe_does_not_consume_the_warn_once_slot(registry):
    """`_handle_would_strand` asks a hypothetical question; running `_substitute`
    to answer it flipped the once-per-registry unresolved-match warning."""
    registry.apply("@Zylphia.mellomnavn.til")
    assert registry._warned_unresolved is False


def test_scandinavian_segments_are_not_treated_as_truncation(registry):
    assert registry.apply("@Zylphia.øygard.til") == "@person"


# --- the five defects the deleting design shipped ---------------------------

def test_exempt_labels_are_never_deleted(registry):
    # Defect 1: gate check 5 asserts exempt labels survive. Nothing is deleted
    # here at all, and the exempt path returned `matched` before this change too.
    assert registry.apply("feltet Saksbehandler er tomt") == "feltet Saksbehandler er tomt"
    assert registry.apply("se Zylphia.Saksbehandler") == "se [~ukjent-person].Saksbehandler"


def test_ident_exceptions_are_never_deleted(registry):
    # Defect 2: gate check 3b. `Q000456` is an ident exception in the fixture.
    assert registry.apply("bruker Q000456 kjørte jobben") == "bruker Q000456 kjørte jobben"
    assert registry.apply("se Zylphia.Q000456") == "se [~ukjent-person].Q000456"


@pytest.mark.parametrize("text,expected", [
    # Defect 3: a full stop with no space after it must not eat the next word.
    ("Se Zylphia. Deretter gjør vi X.", "Se [~ukjent-person]. Deretter gjør vi X."),
    ("Se Zylphia.Deretter gjør vi X.", "Se [~ukjent-person].Deretter gjør vi X."),
    ("Kontakt dev-01. Han svarer.", "Kontakt dev-01. Han svarer."),
])
def test_prose_after_a_substituted_name_is_untouched(registry, text, expected):
    assert registry.apply(text) == expected


def test_an_email_domain_is_never_at_risk(registry):
    # Defect 4: the lookbehind that kept a domain out. _HANDLE_RE's own
    # `(?<![\w.])` does it here — the `@` is preceded by a word character, so no
    # handle is found and the head is substituted exactly as before.
    assert (registry.apply("noreply@Zylphia.example.com")
            == "noreply@[~ukjent-person].example.com")
    assert (registry.apply("skriv til Zylphia.mellomnavn@nav.no")
            == "skriv til [~ukjent-person].mellomnavn@nav.no")


@pytest.mark.parametrize("text,expected", [
    # Defect 5: extensions, dates and other people's aliases were swallowed.
    ("vedlegg Zylphia.pdf", "vedlegg [~ukjent-person].pdf"),
    ("Zylphia.2024.01.15 kl 12", "[~ukjent-person].2024.01.15 kl 12"),
    ("Zylphia.Ada Example", "[~ukjent-person].dev-01"),
])
def test_tails_that_are_not_names_survive_intact(registry, text, expected):
    assert registry.apply(text) == expected


# --- shapes the handle pass refuses: substitute, exactly as before ----------

@pytest.mark.parametrize("text,expected", [
    # An annotation shape (lowercase head, CamelCase tail), a version and a
    # host. `is_person_handle` refuses all three, so declining would leave the
    # NAME in the clear to protect a tail that is not a person. Substitute.
    # The annotation rule needs a LOWERCASE head, so this is the shape that
    # reaches it — `@Zylphia.Setter` reads as a person and collapses.
    ("@zylphia.Setter", "@[~ukjent-person].Setter"),
    ("@Zylphia.2", "@[~ukjent-person].2"),
    ("@Zylphia.no", "@[~ukjent-person].no"),
    ("@Zylphia.internal", "@[~ukjent-person].internal"),
])
def test_non_person_handle_shapes_are_still_substituted(registry, text, expected):
    assert registry.apply(text) == expected


@pytest.mark.parametrize("text", [
    "@org.junit.Test",
    "bruk @v1.2.3 taggen",
    "@lombok.Setter over feltet",
    "np.eye@vec og A.T@B",
    "feltet person.noe er tomt",
    "no.nav.ada.example.Klasse",
])
def test_unrelated_shapes_are_byte_identical(registry, text):
    assert registry.apply(text) == text


# --- behaviour that must NOT move -------------------------------------------

def test_a_handle_wholly_covered_by_a_slug_variant_keeps_its_alias(registry):
    # Nothing is stranded, so the which-person signal is kept rather than
    # traded for @person. This is why the guard needs `handle.end() > end`.
    assert registry.apply("kontakt @ada.example") == "kontakt @dev-01"
    assert registry.apply("kontakt @bo.tester.") == "kontakt @fag-01."


def test_a_slug_inside_a_longer_dotted_handle_still_goes_through_the_handle_pass(registry):
    assert registry.apply("ping @ada.example.mellomnavn") == "ping @person"


def test_percent_encoded_name_in_a_url_is_unaffected(registry):
    assert "dev-01" in registry.apply("ovuser=abc%2CAda.Example%40nav.no&OR=Teams")


def test_a_multi_token_variant_cannot_reach_the_guard(registry):
    # A space breaks _HANDLE_RE, so no handle is ever found spanning one.
    assert registry.apply("@Ada Example svarte") == "@dev-01 svarte"


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


# --- /Users/<name>/ — the pasted-stack-trace fingerprint ----------------------

@pytest.mark.parametrize("text,expected", [
    ("File \"/Users/someone/source/huginn/main/x.py\", line 3",
     "File \"/Users/<user>/source/huginn/main/x.py\", line 3"),
    ("cd /Users/some.one/src && ls", "cd /Users/<user>/src && ls"),
    ("file:///Users/someone/notes/x.md", "file:///Users/<user>/notes/x.md"),
    # A home directory with nothing under it is check 10's business, not a
    # substitution: there is no `/` to bound the account name against.
    ("bygget under /Users/someone", "bygget under /Users/someone"),
    # Not a home path at all: the segment before `/Users/` is a word character.
    ("s3://bucket/Users/someone/x", "s3://bucket/Users/someone/x"),
])
def test_a_home_directory_path_loses_the_account_name(registry, text, expected):
    """A pasted stack trace or shell transcript carries the BUILDER's home
    directory, and the account name in it is a person — usually the one person
    the alias map is least likely to list, because they are the operator rather
    than a document author. Every fingerprint check 10 found on the real
    collections was this shape.
    """
    assert registry.apply(text) == expected


@pytest.mark.parametrize("text", [
    "Ada Example og Bo Tester og Kari Ukjent",
    "Ident [~Q000124] og Q000124 og f000111",
    "ping @ola.nordmann.example og ola.nordmann@nav.no",
    "dev-01 fag-01 [~person] [~ukjent-person] @person",
    "no.nav.zylphia.quorndal.Klasse og en saksbehandler",
    "/Users/someone/src og /Users/<user>/src",
])
def test_apply_is_idempotent(registry, text):
    once = registry.apply(text)
    assert registry.apply(once) == once


def test_manifest_stamp(registry):
    stamp = registry.manifest_stamp()
    assert stamp["policy_version"] == alias_registry.POLICY_VERSION
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


def test_two_distinct_literals_sharing_a_casefold_key_are_invalid():
    """`Ada Weiss` and `Ada Weiß` casefold to one table key but belong to two
    DIFFERENT people.

    The table is keyed on the casefold, so the second literal's replacement is
    dropped and both spellings get the first one's alias — a wrong substitution
    that no output inspection reveals. Pure case variation of the SAME literal
    (`Saksbehandler`/`saksbehandler`, both in MAP) is not this and must still
    compile, and neither is the same collision under one alias.
    """
    entries = json.loads(json.dumps(MAP["entries"]))
    entries[0]["variants"].append("Ada Weiss")
    entries[1]["variants"].append("Ada Weiß")
    with pytest.raises(PrivacyMapInvalid, match="casefold"):
        AliasRegistry(_mutated(entries=entries))


def test_casefold_collision_with_the_same_replacement_still_compiles():
    """Two spellings of one person under ONE alias collide harmlessly.

    The collision matters only because the losing literal's replacement is
    silently dropped. When both literals resolve to the same replacement there
    is nothing to drop, and refusing made the whole map unloadable over an entry
    that was simply thorough about a Unicode spelling.
    """
    entries = json.loads(json.dumps(MAP["entries"]))
    entries[0]["variants"] += ["Ada Weiss", "Ada Weiß"]
    registry = AliasRegistry(_mutated(entries=entries))
    assert registry.apply("av Ada Weiss og Ada Weiß") == "av dev-01 og dev-01"


def test_casefold_collision_across_classes_is_still_invalid():
    """…but the same collision between an exempt label and a person is not: one
    of the two replacements is real and gets silently thrown away."""
    entries = json.loads(json.dumps(MAP["entries"]))
    entries[0]["variants"].append("Ada Weiß")
    bad = _mutated(entries=entries, non_person_labels=[*MAP["non_person_labels"], "Ada Weiss"])
    with pytest.raises(PrivacyMapInvalid, match="casefold"):
        AliasRegistry(bad)


@pytest.mark.parametrize("variant", ["Ada", " Ada ", "\tAda\n"])
def test_a_whitespace_padded_bare_given_name_is_still_refused(variant):
    """`fullmatch` against the padded literal fails, so the guard waved it
    through and compiled a pattern that substitutes a bare given name — the one
    substitution the campaign explicitly decided never to make."""
    with pytest.raises(PrivacyMapInvalid, match="bare given name"):
        AliasRegistry(_mutated(entries=_entries_with_variant(variant)))


@pytest.mark.parametrize("text", [
    "https://x/?f=%3BAda%20Example%3B",       # %3B — a semicolon separator
    "https://x/?p=%2FAda%20Example%2F",       # %2F — a path separator
    "https://x/?q=%3AAda%20Example%3A",       # %3A — a colon
])
def test_multi_token_variant_matches_across_any_percent_escape(registry, text):
    """The boundary knows the SHAPE of a percent escape, not three instances.

    `%2C`, `%20` and `%40` were enumerated from one corpus sample. Every other
    escape a query string uses fenced a name the substituter then walked past.
    """
    assert "dev-01" in registry.apply(text)


def test_case_only_duplicate_literals_still_compile():
    # MAP already carries "Saksbehandler" and "saksbehandler".
    assert AliasRegistry(MAP).apply("en Saksbehandler og en saksbehandler") == \
        "en Saksbehandler og en saksbehandler"


def test_hyphenated_single_token_variant_is_allowed():
    """A hyphenated compound surname is a full name, not a bare given name; the
    old "no whitespace, `.`, `_` or `,`" rule rejected the whole map over it."""
    registry = AliasRegistry(_mutated(entries=_entries_with_variant("Ada-Example")))
    assert registry.apply("skrevet av Ada-Example") == "skrevet av dev-01"


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
    _point_discovery_at(monkeypatch, tmp_path)
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


# --- path_in_scope ----------------------------------------------------------

def _point_discovery_at(monkeypatch, root):
    """Point the private-file globs and the relative scope paths at a tmp root.

    They resolve against `alias_registry.REPO_ROOT`, not the process CWD — a
    guard that only arms when the caller happens to be standing in the repo is
    not a guard. `chdir` stays so a genuinely CWD-relative read would still be
    caught by the tests that assert a refusal.
    """
    from main.privacy import alias_registry
    monkeypatch.setattr(alias_registry, "REPO_ROOT", str(root))
    monkeypatch.chdir(root)


@pytest.fixture
def scoped_tree(tmp_path, monkeypatch):
    """A private scope file naming one tree, discovered the usual way."""
    tree = tmp_path / "sources" / "in-scope"
    (tree / "sub").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    privacy = tmp_path / "huginn-x" / "privacy"
    privacy.mkdir(parents=True)
    (privacy / "scope.json").write_text(
        json.dumps({"collections": [], "basePaths": ["./sources/in-scope"]}), encoding="utf-8")
    _point_discovery_at(monkeypatch, tmp_path)
    return tmp_path, tree


def test_path_in_scope_matches_the_tree_and_everything_under_it(scoped_tree):
    from main.privacy.alias_registry import path_in_scope
    root, tree = scoped_tree
    assert path_in_scope(str(tree)) is True
    assert path_in_scope(str(tree / "sub")) is True
    assert path_in_scope("./sources/in-scope/sub") is True


def test_path_in_scope_rejects_a_sibling_and_a_walk_out(scoped_tree):
    from main.privacy.alias_registry import path_in_scope
    root, tree = scoped_tree
    assert path_in_scope(str(root / "outside")) is False
    # A prefix match on the string would accept this sibling directory.
    (root / "sources" / "in-scope-other").mkdir()
    assert path_in_scope(str(root / "sources" / "in-scope-other")) is False
    assert path_in_scope(str(tree / ".." / "in-scope-other")) is False
    assert path_in_scope("") is False


def test_path_in_scope_follows_a_symlink_into_the_tree(scoped_tree):
    """Realpath on both sides: a link is not a way out of scope."""
    from main.privacy.alias_registry import path_in_scope
    root, tree = scoped_tree
    link = root / "outside" / "link"
    link.symlink_to(tree / "sub")
    assert path_in_scope(str(link)) is True


def test_two_entries_sharing_an_alias_are_refused(tmp_path):
    import json, pytest
    from main.privacy import alias_registry
    data = {"version": 1, "entries": [
        {"alias": "dev-01", "name": "Ada Example", "variants": ["Ada Example"]},
        {"alias": "dev-01", "name": "Bo Tester", "variants": ["Bo Tester"]},
    ] + [{"alias": f"dev-{i:02d}", "name": f"Kari Tester{i}", "variants": [f"Kari Tester{i}"]} for i in range(2, 60)],
        "non_person_labels": [], "unmapped_people_variants": {}}
    p = tmp_path / "m.json"; p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(alias_registry.PrivacyMapInvalid):
        alias_registry.AliasRegistry.load(p)


def test_given_names_strip_the_comma_of_a_surname_first_label(tmp_path):
    import json
    from main.privacy import alias_registry
    data = {"version": 1, "entries": [
        {"alias": f"dev-{i:02d}", "name": f"Kari Tester{i}", "variants": [f"Kari Tester{i}"]} for i in range(60)],
        "non_person_labels": [], "unmapped_people_variants": {"Example, Ada": ["Example, Ada"]}}
    p = tmp_path / "m.json"; p.write_text(json.dumps(data), encoding="utf-8")
    reg = alias_registry.AliasRegistry.load(p)
    assert "example" in reg.given_names and "example," not in reg.given_names
