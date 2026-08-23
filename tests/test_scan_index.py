"""Guards on the distribution gate (main/privacy/index_scan.py + its CLI).

The gate is the only thing standing between a built collection and a copy of it
on someone else's laptop, so the ways it can pass *vacuously* — a truncated map,
a stamp from another build, an empty needle list, a name the needle regex cannot
see, a candidate filter pointed the wrong way — matter more than the ways it can
fail. Every test here builds a collection the gate would otherwise wave through.

Every name is invented ("Ada Example", "Zylphia Quorndal", "Kari Ukjent"). The
real map lives in a gitignored private sub-repo and is never read by the suite.
"""
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "audit"))

import scan_index as cli  # noqa: E402
from main.indexes.indexers.bm25_indexer import _tokenize  # noqa: E402
from main.privacy import index_scan  # noqa: E402
from main.privacy.alias_registry import PrivacyMapInvalid, boundaried  # noqa: E402

MAP_VERSION = 7
GOOD_STAMP = {"policy_version": 1, "map_version": MAP_VERSION,
              "aliasedAt": "2026-01-02T00:00:00+00:00"}


def _map(entry_count=index_scan.MIN_MAP_ENTRIES):
    """A schema-valid map with enough entries to clear the floor.

    The generated variants carry a digit on purpose: a single all-alphabetic
    token is a bare given name and AliasRegistry refuses to compile one.
    """
    entries = []
    for i in range(entry_count):
        name = f"Ada Example{i:02d}"
        entries.append({
            "alias": f"dev-{i:02d}",
            "name": name,
            "role": "dev",
            "variants": [name, f"Example{i:02d}, Ada", f"ada.example{i:02d}"],
            "require_full_name": False,
            "idents": [],
            "departed": False,
            "confirmed": True,
            "extra_variants": [],
        })
    return {
        "version": MAP_VERSION,
        "entries": entries,
        "ident_policy": "redact",
        "non_person_labels": ["saksbehandler", "Bo Kommune"],
        "unmapped_people": ["Zylphia Quorndal"],
        "unmapped_people_variants": {
            "Zylphia Quorndal": ["Zylphia Quorndal", "Quorndal, Zylphia",
                                 "zylphia.quorndal"],
        },
        "bare_given_name_residual": {"Zylphia": 3},
        "person_redaction_token": "[~ukjent-person]",
        "retired_aliases": [],
    }


def _write_collection(root, name, *, documents, stamp=GOOD_STAMP,
                      documents_count=None, chunks_count=None, drop_chunk_count=False,
                      bm25_texts=None):
    """A minimal on-disk collection the gate can walk end to end."""
    collection = root / name
    (collection / "documents").mkdir(parents=True)
    (collection / "indexes" / "indexer_BM25").mkdir(parents=True)

    for index, document in enumerate(documents):
        (collection / "documents" / f"doc{index}.json").write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "collectionName": name,
        "numberOfDocuments": len(documents) if documents_count is None else documents_count,
        "numberOfChunks": len(documents) if chunks_count is None else chunks_count,
        "reader": {"type": "localFiles", "basePath": "./data/sources/demo"},
    }
    if drop_chunk_count:
        del manifest["numberOfChunks"]
    if stamp is not None:
        manifest["privacy"] = stamp
    (collection / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mapping = {str(i): d["id"] for i, d in enumerate(documents)}
    (collection / "indexes" / "index_document_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8")
    (collection / "indexes" / "reverse_index_document_mapping.json").write_text(
        json.dumps({v: k for k, v in mapping.items()}), encoding="utf-8")

    corpus = ([_tokenize(d["text"]) for d in documents] if bm25_texts is None
              else [_tokenize(t) for t in bm25_texts])
    with open(collection / "indexes" / "indexer_BM25" / "indexer", "wb") as f:
        pickle.dump({"corpus_tokens": corpus, "ids": list(mapping)}, f)
    return collection


def _document(index, text):
    return {"id": f"notat-{index}.md", "url": f"file:///srv/notat-{index}.md",
            "text": text, "chunks": [{"indexedData": text}]}


def _clean_document(index):
    return _document(index, "dev-01 skrev dette sammen med en saksbehandler.")


@pytest.fixture
def map_file(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(_map()), encoding="utf-8")
    return path


def _scan(tmp_path, map_file, name="demo-aliased", **kwargs):
    return index_scan.scan_collection(tmp_path / name, map_file, **kwargs)


def _run(argv, monkeypatch, repo_root):
    """Drive the CLI the way the campaign does, with no private files in reach."""
    from main.privacy import alias_registry
    monkeypatch.setattr(cli, "REPO_ROOT", repo_root)
    monkeypatch.setattr(cli, "DEFAULT_CANDIDATES_DIR", repo_root / "privacy-out")
    # The map and the ident exceptions come from alias_registry's own globs now,
    # so the tmp root has to be the one it looks in too.
    monkeypatch.setattr(alias_registry, "REPO_ROOT", str(repo_root))
    monkeypatch.setattr(sys, "argv", ["scan_index.py", *argv])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return excinfo.value.code


# --- check 0: the map itself ------------------------------------------------

def test_truncated_map_is_rejected_by_the_entry_floor(tmp_path, map_file):
    """A decoy map with three names certifies everything it does not know."""
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)])
    small = tmp_path / "small.json"
    small.write_text(json.dumps(_map(3)), encoding="utf-8")
    assert _scan(tmp_path, small).check("map_stamp").passed is False
    assert _scan(tmp_path, map_file).check("map_stamp").passed is True


@pytest.mark.parametrize("stamp", [
    {**GOOD_STAMP, "map_version": MAP_VERSION - 1},
    {**GOOD_STAMP, "policy_version": index_scan.POLICY_VERSION + 1},
    None,
])
def test_stamp_from_another_build_is_rejected(tmp_path, map_file, stamp):
    # The gate verifying with map v7 against an index built from v6 reports a
    # clean collection for names v6 never knew.
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)], stamp=stamp)
    assert _scan(tmp_path, map_file).check("map_stamp").passed is False


# --- needles ----------------------------------------------------------------

def test_empty_needle_list_raises_rather_than_passing_vacuously():
    with pytest.raises(ValueError):
        index_scan.needle_pattern([])


def test_needle_pattern_sees_a_percent_encoded_name():
    """The boundary is shared with the registry: a form the substituter removes
    but the gate cannot see would certify a collection nobody aliased."""
    pattern = index_scan.needle_pattern(["Ada Example"])
    assert pattern.search("ovuser=abc%2CAda%20Example%40nav.no")
    assert pattern.search("?q=fra%20Ada%20Example")
    assert not pattern.search("AdaExample")


def test_needle_boundaries_are_per_shape_not_all_multi_token():
    """A mononym needle must use the registry's bare-token boundary.

    Under the multi-token boundary (which allows a preceding `.`) the gate
    reported `no.nav.zylphia` as a leak — a form the substituter deliberately
    leaves alone — and blocked distribution over a non-leak.
    """
    pattern = index_scan.needle_pattern(["Zylphia", "zylphia.quorndal"])
    assert not pattern.search("pakken no.nav.zylphia er intern")
    assert pattern.search("skrevet av Zylphia i går")
    # …and a slug needle must not fire inside a longer dotted path.
    assert not pattern.search("no.nav.zylphia.quorndal.impl")


# 14 characters that can sit against a needle in the corpora: percent escapes,
# digits and letters welded onto a slug, path/query punctuation, quotes, a
# non-breaking space, and the two whitespace characters a variant may span.
FENCES = ["%", "4", "z", ".", "-", "_", "/", "=", ",", "(", '"', "\u00a0", "\n", "\t"]


def test_the_bucketed_scan_never_hides_a_hit_the_alternation_finds():
    """The bucketing is a speed trick; a miss there is a silent PASS.

    The old prefilter tokenised the text and compared whole words, which is not
    a superset: a slug welded onto a preceding word (`xyzada.example00`) and a
    name behind a percent escape whose hex digits eat the first letters
    (`50%Beate…`) both matched the alternation and were rejected by the filter.
    Buckets are keyed on the first word and selected by SUBSTRING, which is what
    makes the test below hold for a welded slug.
    """
    random.seed(20260823)
    needles = index_scan.build_needles(_map(40))
    full = index_scan.needle_pattern(needles)
    scanner = index_scan.NeedleScanner(needles)
    checked = 0
    for _ in range(500):
        text = "".join(
            random.choice(FENCES) + needle + random.choice(FENCES)
            for needle in random.sample(needles, 3))
        expected = set(full.findall(text))
        if not expected:
            continue
        checked += 1
        found = set(scanner.findall(text))
        # Superset, not equality: each bucket scans independently, so where the
        # single alternation consumes one of two overlapping forms the buckets
        # may report both. Missing one is the failure mode that matters.
        assert expected <= found, (sorted(expected - found), text)
        assert scanner.search(text) is not None
    assert checked > 400, f"only {checked} of 500 fixtures produced a hit"


def test_the_bucketed_scan_sees_a_welded_slug_and_a_percent_eaten_first_word():
    """The two shapes the tokenising prefilter dropped, named explicitly."""
    scanner = index_scan.NeedleScanner(["ada.example00", "Beate Example"])
    assert scanner.findall("xyzada.example00") == ["ada.example00"]
    assert scanner.findall("id4711ada.example00") == ["ada.example00"]
    assert scanner.findall("50%Beate Example") == ["Beate Example"]
    assert scanner.findall("en helt vanlig setning") == []


def test_contains_sequence_catches_a_hyphenated_name_in_a_document_id():
    """Ids are never aliased by design, so a name in a path is a real leak; a
    plain substring test misses it because the separator is a hyphen."""
    sequences = index_scan.token_sequences(["Ada Example"])
    tokens = [t.lower() for t in index_scan.WORD_SPLIT_RE.split("Team/Ada-Example.md") if t]
    assert index_scan.contains_sequence(tokens, sequences) is True
    assert index_scan.contains_sequence(["team", "notat", "md"], sequences) is False


# --- check 9: capitalised-bigram candidates ---------------------------------

def test_an_unmapped_person_fails_the_scan(tmp_path, map_file):
    """The headline category: nobody ever added this person to the map, so every
    map-derived check is blind to them."""
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Notatet ble skrevet av Kari Ukjent i mars.")])
    report = _scan(tmp_path, map_file)
    check = report.check("bigram_candidates")
    assert check.passed is False and check.count == 1
    assert report.candidates[0]["text"] == "Kari Ukjent"


def test_retention_keeps_a_candidate_rather_than_filtering_it_out(tmp_path, map_file):
    """The direction of the filter IS the check.

    An earlier draft subtracted anything that looked like a name, which removes
    precisely the population this check exists to find. Retention means: a
    capitalised pair is a candidate only when its first token is a plausible
    given name, and everything else is ignored.
    """
    _write_collection(tmp_path, "demo-aliased", documents=[
        _document(0, "Decision on Applicable Legislation ble vedtatt av Kari Ukjent."),
    ])
    report = _scan(tmp_path, map_file)
    texts = [c["text"] for c in report.candidates]
    assert texts == ["Kari Ukjent"]
    assert "Applicable Legislation" not in texts


def test_a_reviewed_non_person_pair_is_allow_listed(tmp_path, map_file):
    allowlist = tmp_path / "non_person_bigrams.json"
    allowlist.write_text(json.dumps({"bigrams": ["Kari Ukjent"]}), encoding="utf-8")
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Skrevet av Kari Ukjent.")])
    report = _scan(tmp_path, map_file, allowed_bigrams_path=allowlist)
    assert report.check("bigram_candidates").passed is True
    assert report.candidates_before_allowlist == 1
    assert report.candidates == []


def test_a_non_person_label_is_not_a_candidate(tmp_path, map_file):
    """`non_person_labels` are reviewed already; re-reporting them trains the
    reader to skim the candidate list."""
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Saken ligger hos Bo Kommune nå.")])
    assert _scan(tmp_path, map_file).check("bigram_candidates").passed is True


def test_a_bigram_spans_a_single_line_break_but_not_a_paragraph(tmp_path, map_file):
    """A hard-wrapped full name is the same name.

    This used to refuse to cross a newline at all, on the theory that the last
    word of one line and the first of the next invent candidates. They do — but
    a wrapped `First\nLast` is exactly the form the substituter's own
    VARIANT_SEPARATOR accepts, and refusing it made the one check that can see
    an unmapped person blind to every line-wrapped one. A BLANK line is still a
    paragraph break and does not join.
    """
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Sendt til Kari\nUkjentsen mottok den.")])
    report = _scan(tmp_path, map_file)
    assert report.check("bigram_candidates").passed is False
    assert [c["text"] for c in report.candidates] == ["Kari Ukjentsen"]

    _write_collection(tmp_path, "para-aliased",
                      documents=[_document(0, "Sendt til Kari\n\nUkjentsen mottok den.")])
    assert _scan(tmp_path, map_file, name="para-aliased").check(
        "bigram_candidates").passed is True


def test_the_gazetteer_union_includes_the_maps_own_given_names():
    """A rare given name lives only in the private map, and is unioned at
    runtime — the public file must never grow corpus-specific names."""
    alias_map = _map()
    alias_map["bare_given_name_residual"] = {"Zylphia": 3}
    names = index_scan.map_given_names(alias_map)
    assert "zylphia" in names and "ada" in names


# --- check 10: distributor fingerprints -------------------------------------

@pytest.mark.parametrize("text", [
    "se /Users/someone/source/huginn/data for detaljer",
    "gjenopprett fra manifest.json.bak før du kjører",
])
def test_a_distributor_fingerprint_fails_the_scan(tmp_path, map_file, text):
    _write_collection(tmp_path, "demo-aliased", documents=[_document(0, text)])
    assert _scan(tmp_path, map_file).check("fingerprints").passed is False


def test_a_relative_path_is_not_a_fingerprint(tmp_path, map_file):
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "se ./data/sources/demo for detaljer")])
    assert _scan(tmp_path, map_file).check("fingerprints").passed is True


# --- check 11: sensitive tokens ---------------------------------------------

def test_an_organisasjonsnummer_is_advisory_not_blocking(tmp_path, map_file):
    """An organisation number is public register data, not personal data.

    It is the one category that fires on every real collection (6/18/20) and
    every hit so far has been a legitimate employer reference in a Jira issue.
    Blocking on it would make the gate unpassable for content it is not there to
    stop, which is how a gate gets switched off. Still reported.
    """
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Arbeidsgiver med orgnr 987654325.")])
    check = _scan(tmp_path, map_file).check("sensitive_tokens")
    assert check.passed is True
    # Twice: the fixture document carries the same sentence as `text` and as the
    # single chunk's `indexedData`, and both are artifacts that ship.
    assert "'organisasjonsnummer': 2" in check.notes[0]


def test_a_bank_account_still_blocks(tmp_path, map_file):
    """The categories that still block, so the advisory move above is narrow."""
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "kontonummer 1234.56.78903 for utbetaling")])
    assert _scan(tmp_path, map_file).check("sensitive_tokens").passed is False


def test_a_page_id_that_passes_mod11_is_not_an_organisasjonsnummer(tmp_path, map_file):
    """The precision requirement, as a test: this exact shape produced hundreds
    of findings on the real corpora before the anchor and leading-digit rules."""
    _write_collection(tmp_path, "demo-aliased", documents=[
        _document(0, "https://confluence.example.no/pages/viewpage.action?pageId=987654325"),
    ])
    assert _scan(tmp_path, map_file).check("sensitive_tokens").passed is True


def test_an_advisory_category_does_not_fail_the_scan(tmp_path, map_file):
    """A phone number is reported and does not block: the categories that block
    are the ones measured precise enough to block on."""
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "tlf: 22 33 44 55 ved spørsmål")])
    check = _scan(tmp_path, map_file).check("sensitive_tokens")
    assert check.passed is True
    assert "telefon" in check.notes[0]


# --- counts against the pre-alias twin --------------------------------------

def test_document_count_drift_fails_by_default(tmp_path, map_file, monkeypatch):
    """A rebuild that indexed a different set of documents is not a rebuild of
    the same collection — and its extra documents were never scanned before."""
    _write_collection(tmp_path, "demo", documents=[_clean_document(0)])
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_clean_document(0), _clean_document(1)])
    code = _run(["--collection", "demo-aliased", "--compare", "demo",
                 "--collections-dir", str(tmp_path), "--map", str(map_file)],
                monkeypatch, tmp_path)
    assert code == 1


def test_allow_count_drift_downgrades_the_count_check_to_a_warning(
        tmp_path, map_file, monkeypatch, capsys):
    _write_collection(tmp_path, "demo", documents=[_clean_document(0)])
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_clean_document(0), _clean_document(1)])
    code = _run(["--collection", "demo-aliased", "--compare", "demo",
                 "--collections-dir", str(tmp_path), "--map", str(map_file),
                 "--allow-count-drift"],
                monkeypatch, tmp_path)
    assert code == 0
    assert "WARN" in capsys.readouterr().out


def test_a_count_the_twin_does_not_record_is_skipped_not_failed(tmp_path, map_file):
    """An older twin manifest predates `numberOfChunks`; comparing a real count
    against None reported a drift that does not exist."""
    _write_collection(tmp_path, "demo", documents=[_clean_document(0)], drop_chunk_count=True)
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)])
    check = _scan(tmp_path, map_file, compare_dir=tmp_path / "demo").check(
        "manifest_and_prefixes")
    assert check.passed is True
    assert any("numberOfChunks: not recorded" in note for note in check.notes)


def test_matching_counts_pass(tmp_path, map_file, monkeypatch):
    _write_collection(tmp_path, "demo", documents=[_clean_document(0)])
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)])
    code = _run(["--collection", "demo-aliased", "--compare", "demo",
                 "--collections-dir", str(tmp_path), "--map", str(map_file)],
                monkeypatch, tmp_path)
    assert code == 0


def test_a_real_name_in_a_document_still_fails(tmp_path, map_file, monkeypatch):
    """The whole gate, end to end, on a collection that leaks — otherwise the
    tests above only prove the individual checks work."""
    leaky = _document(0, "Ada Example00 skrev dette.")
    _write_collection(tmp_path, "demo", documents=[_clean_document(0)])
    _write_collection(tmp_path, "demo-aliased", documents=[leaky])
    code = _run(["--collection", "demo-aliased", "--compare", "demo",
                 "--collections-dir", str(tmp_path), "--map", str(map_file)],
                monkeypatch, tmp_path)
    assert code == 1


# --- report shape -----------------------------------------------------------

def test_the_json_report_carries_no_literals(tmp_path, map_file):
    """The JSON report is meant for a PR body and a package stamp; the
    candidates (the one output with real text in it) must not be in it."""
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Skrevet av Kari Ukjent på /Users/someone/x.")])
    report = _scan(tmp_path, map_file)
    blob = json.dumps(report.to_json(), ensure_ascii=False)
    assert "Kari" not in blob and "Ukjent" not in blob and "someone" not in blob
    assert report.to_json()["candidatesAfterAllowlist"] == 1
    assert "Kari Ukjent" in json.dumps(report.candidates, ensure_ascii=False)


def test_printed_lines_carry_no_literals(tmp_path, map_file):
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Skrevet av Kari Ukjent.")])
    printed = "\n".join(_scan(tmp_path, map_file).format_lines())
    assert "Kari" not in printed and "Ukjent" not in printed
    assert "xxxx xxxxxx" in printed


# --- A2: the scan boundary is WIDER than the substituter's --------------------

def test_a_slug_needle_is_seen_before_a_dotted_label():
    """`scan >= substituter` is the contract, not `scan == substituter`.

    The registry deliberately refuses to substitute a slug followed by another
    dotted label — `no.nav.ada.example00.impl` is a package path, and rewriting
    it breaks code. But `ada.example00.md` in a document id IS the person, and
    the old shared boundary made the gate blind to it.
    """
    pattern = index_scan.needle_pattern(["ada.example00"])
    assert pattern.search("Team/ada.example00.md")
    assert pattern.search("ada.example00.html")
    assert pattern.search("ada.example00.quorndal")
    # …and still not welded onto a longer path on the LEFT.
    assert not pattern.search("no.nav.ada.example00")
    # The substituter's own boundary is the narrower one, unchanged.
    assert not re.search(boundaried("ada.example00"), "Team/ada.example00.md", re.I)


def test_a_mononym_needle_is_seen_after_a_percent_escape():
    pattern = index_scan.needle_pattern(["Zylphia"])
    assert pattern.search("?q=%20Zylphia")
    assert pattern.search("f=%2CZylphia")
    # A mononym welded onto a surviving dotted path is still not a hit.
    assert not pattern.search("pakken no.nav.zylphia er intern")
    assert not re.search(boundaried("Zylphia"), "?q=%20Zylphia", re.I)


# --- A3: the bigram detector -------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("NAV Kari Ukjentsen skrev det", "Kari Ukjentsen"),          # pair inside a run
    ("## Referat Kari Ukjentsen", "Kari Ukjentsen"),             # head is a heading word
    ("Zorblax Kari Ukjentsen", "Kari Ukjentsen"),                # head is nothing at all
    ("se ref.Kari Ukjentsen for detaljer", "Kari Ukjentsen"),    # a dot immediately before
    ("Anne-Marie Ukjentsen ringte", "Anne-Marie Ukjentsen"),     # hyphenated given name
    ("Kari\nUkjentsen mottok den", "Kari Ukjentsen"),            # one newline inside
])
def test_every_adjacent_pair_in_a_capitalised_run_is_evaluated(
        tmp_path, map_file, text, expected):
    """Only the HEAD pair used to be probed, so a name behind any capitalised
    word — an acronym, a heading, another name — was invisible."""
    _write_collection(tmp_path, "demo-aliased", documents=[_document(0, text)])
    report = _scan(tmp_path, map_file)
    assert expected in [c["text"] for c in report.candidates], report.candidates


@pytest.mark.parametrize("text,expected", [
    ("Áilu Ukjentsen kom innom", "Áilu Ukjentsen"),
    ("Łukasz Ukjentsen kom innom", "Łukasz Ukjentsen"),
])
def test_a_non_ascii_initial_capital_is_a_name_token(tmp_path, map_file, text, expected):
    """`[A-ZÆØÅ]` is a Norwegian-keyboard assumption; the corpora are not."""
    given = tmp_path / "given.txt"
    given.write_text("\n".join(["Áilu", "Łukasz"] + [f"Filler{i:03d}" for i in range(600)]),
                     encoding="utf-8")
    _write_collection(tmp_path, "demo-aliased", documents=[_document(0, text)])
    report = _scan(tmp_path, map_file, given_names_file=given)
    assert expected in [c["text"] for c in report.candidates], report.candidates


def test_a_gazetteer_below_the_floor_refuses_to_certify(tmp_path, map_file):
    """A truncated gazetteer makes check 9 retain almost nothing, which reads as
    a clean collection. Same failure mode, and same floor, as the alias map."""
    thin = tmp_path / "thin.txt"
    thin.write_text("Kari\nAda\n", encoding="utf-8")
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)])
    with pytest.raises(ValueError, match="gazetteer"):
        _scan(tmp_path, map_file, given_names_file=thin)


def test_map_given_names_reads_the_entry_name_too():
    """`variants` may spell a person only as `Last, First` or a slug; `name` is
    the one field guaranteed to be `First Last`."""
    alias_map = _map(3)
    for entry in alias_map["entries"]:
        entry["variants"] = [v.replace("Ada ", "Bo ") for v in entry["variants"]]
    alias_map["entries"][0]["name"] = "Zylphia Example00"
    alias_map["entries"][0]["variants"] = ["Example00, Zylphia"]
    alias_map["bare_given_name_residual"] = {}
    assert "zylphia" in index_scan.map_given_names(alias_map)


# --- A4: guards on checks 5 and 6 --------------------------------------------

def test_an_empty_exempt_and_exception_set_does_not_compile_a_blank_pattern(
        tmp_path, map_file):
    """`"|".join([])` is the empty regex, which matches at every position — the
    invariant would then compare a count of *characters* on both sides."""
    bare = json.loads(map_file.read_text(encoding="utf-8"))
    bare["non_person_labels"] = []
    bare_path = tmp_path / "bare.json"
    bare_path.write_text(json.dumps(bare), encoding="utf-8")
    # The pre-alias twin has to CHANGE under the registry, or a blank pattern
    # counts the same number of empty matches on both sides and passes anyway.
    _write_collection(tmp_path, "demo",
                      documents=[_document(0, "Ada Example00 skrev dette.")])
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)])
    check = _scan(tmp_path, bare_path, compare_dir=tmp_path / "demo").check(
        "exempt_labels_unmoved")
    assert check.passed is True and check.count == 0


def test_check_six_does_not_report_a_prefix_count_for_a_stamp_failure(tmp_path, map_file):
    """`count` is read as "how many prefixes named a person". Reporting 0 next
    to FAIL for a missing stamp says the opposite of what happened."""
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)], stamp=None)
    check = _scan(tmp_path, map_file).check("manifest_and_prefixes")
    assert check.passed is False and check.count == 1
    assert "privacy stamp" in check.detail


def test_a_map_entry_without_a_name_is_refused(tmp_path):
    """`build_needles` reads `entry["name"]` for the permutation forms; without
    it the gate raises KeyError halfway through instead of refusing the map."""
    alias_map = _map(3)
    del alias_map["entries"][0]["name"]
    with pytest.raises(PrivacyMapInvalid, match="name"):
        index_scan.AliasRegistry(alias_map)


# --- A5/A6: check 8, the "I could not read this" check ------------------------

def test_a_truncated_document_json_fails_check_eight(tmp_path, map_file):
    """A document the scan cannot parse is a document it cannot certify. It used
    to raise JSONDecodeError out of the gate, which reads as a crashed tool."""
    collection = _write_collection(tmp_path, "demo-aliased",
                                   documents=[_clean_document(0)])
    (collection / "documents" / "truncated.json").write_text('{"id": "x", "text"',
                                                             encoding="utf-8")
    check = _scan(tmp_path, map_file).check("unreadable_files")
    assert check.passed is False and "truncated.json" in check.detail


def test_an_unreadable_manifest_fails_rather_than_raising(tmp_path, map_file):
    collection = _write_collection(tmp_path, "demo-aliased",
                                   documents=[_clean_document(0)])
    (collection / "manifest.json").write_text("{not json", encoding="utf-8")
    report = _scan(tmp_path, map_file)
    assert report.passed is False
    assert report.check("unreadable_files").passed is False
    assert report.check("map_stamp").passed is False


def test_a_symlinked_file_fails_check_eight(tmp_path, map_file):
    """A symlink is archived as a link that resolves on the RECIPIENT's machine,
    and following it would scan a file that is not in the collection at all."""
    collection = _write_collection(tmp_path, "demo-aliased",
                                   documents=[_clean_document(0)])
    (tmp_path / "elsewhere.json").write_text('{"id": "e", "text": "Ada Example00"}',
                                             encoding="utf-8")
    (collection / "documents" / "link.json").symlink_to(tmp_path / "elsewhere.json")
    check = _scan(tmp_path, map_file).check("unreadable_files")
    assert check.passed is False and "link.json" in check.detail


def test_a_symlinked_directory_fails_check_eight(tmp_path, map_file):
    collection = _write_collection(tmp_path, "demo-aliased",
                                   documents=[_clean_document(0)])
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "leak.json").write_text('{"id": "e", "text": "Ada Example00"}',
                                                    encoding="utf-8")
    (collection / "extra").symlink_to(tmp_path / "outside", target_is_directory=True)
    check = _scan(tmp_path, map_file).check("unreadable_files")
    assert check.passed is False and "extra" in check.detail


def test_a_bak_file_fails_check_eight(tmp_path, map_file):
    collection = _write_collection(tmp_path, "demo-aliased",
                                   documents=[_clean_document(0)])
    (collection / "manifest.json.bak").write_text("{}", encoding="utf-8")
    check = _scan(tmp_path, map_file).check("unreadable_files")
    assert check.passed is False and "manifest.json.bak" in check.detail


def test_an_undecodable_binary_fails_check_eight(tmp_path, map_file):
    """The scan can only certify what it has read."""
    collection = _write_collection(tmp_path, "demo-aliased",
                                   documents=[_clean_document(0)])
    (collection / "stray.dat").write_bytes(b"\xff\xfe\x00\x01Ada Example00")
    check = _scan(tmp_path, map_file).check("unreadable_files")
    assert check.passed is False and "stray.dat" in check.detail


def test_a_vector_index_carrying_strings_fails_check_eight(tmp_path, map_file):
    """A vector index is allowed to be a bare ndarray and nothing else — a
    pickle that also carries the source strings would ship them."""
    collection = _write_collection(tmp_path, "demo-aliased",
                                   documents=[_clean_document(0)])
    faiss_dir = collection / "indexes" / "indexer_e5"
    faiss_dir.mkdir(parents=True)
    with open(faiss_dir / "indexer", "wb") as f:
        pickle.dump({"texts": ["Ada Example00"]}, f)
    check = _scan(tmp_path, map_file).check("unreadable_files")
    assert check.passed is False and "indexer_e5" in check.detail


def test_a_bare_ndarray_vector_index_is_accepted(tmp_path, map_file):
    collection = _write_collection(tmp_path, "demo-aliased",
                                   documents=[_clean_document(0)])
    faiss_dir = collection / "indexes" / "indexer_e5"
    faiss_dir.mkdir(parents=True)
    with open(faiss_dir / "indexer", "wb") as f:
        pickle.dump(np.zeros((2, 3), dtype="float32"), f)
    assert _scan(tmp_path, map_file).check("unreadable_files").passed is True


# --- A7: the gazetteer union, end to end -------------------------------------

def test_a_given_name_only_the_private_map_knows_is_caught_end_to_end(tmp_path, map_file):
    """`Zylphia` is not in the public gazetteer and never will be — it is a rare
    name that only this corpus uses. The union has to happen at RUNTIME, and the
    unit test on `map_given_names` does not prove the wiring."""
    public = tmp_path / "public.txt"
    public.write_text("\n".join(f"Filler{i:03d}" for i in range(600)), encoding="utf-8")
    assert "zylphia" not in index_scan.load_public_given_names(public)
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Møtet ble ledet av Zylphia Ukjentsen.")])
    report = _scan(tmp_path, map_file, given_names_file=public)
    assert [c["text"] for c in report.candidates] == ["Zylphia Ukjentsen"]


# --- A8: check 2, the BM25 corpus --------------------------------------------

def test_a_name_in_the_bm25_token_list_fails_check_two(tmp_path, map_file):
    """A byte-grep of the pickle is vacuous: `ada example00` matches zero bytes
    while both tokens sit side by side in `corpus_tokens`."""
    _write_collection(
        tmp_path, "demo-aliased",
        documents=[_clean_document(0)],
        bm25_texts=["dev-01 skrev dette", "referatet fra Ada Example00 i mars"])
    check = _scan(tmp_path, map_file).check("bm25_sequences")
    assert check.passed is False and check.count == 1
    assert "Ada" not in check.detail and "Example00" not in check.detail


def test_a_clean_bm25_corpus_passes_check_two(tmp_path, map_file):
    _write_collection(tmp_path, "demo-aliased", documents=[_clean_document(0)],
                      bm25_texts=["dev-01 skrev dette sammen med en saksbehandler"])
    check = _scan(tmp_path, map_file).check("bm25_sequences")
    assert check.passed is True and check.count == 0


# --- A10: observability on a check-1 failure ---------------------------------

def test_a_person_form_failure_prints_masked_examples(tmp_path, map_file):
    """A bare count says a name survived somewhere in 543 files. The shape says
    which form it was, which is what tells the operator where to look."""
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Ada Example00 skrev dette.")])
    check = _scan(tmp_path, map_file).check("person_forms")
    assert check.passed is False
    joined = " ".join(check.notes)
    assert "xxx xxxxxxx99" in joined
    assert "Ada" not in joined and "Example" not in joined


# --- the scanned member list the packager tars -------------------------------

def test_the_report_lists_exactly_the_files_it_scanned(tmp_path, map_file):
    """The packager tars this list, not a fresh walk: anything that appeared
    between the scan and the tar was never certified."""
    collection = _write_collection(tmp_path, "demo-aliased",
                                   documents=[_clean_document(0)])
    report = _scan(tmp_path, map_file)
    on_disk = {p.relative_to(collection).as_posix()
               for p in collection.rglob("*") if p.is_file()}
    assert {m["path"] for m in report.scanned_members} == on_disk
    assert all(isinstance(m["size"], int) and m["mtime"] is not None
               for m in report.scanned_members)
    assert "scannedMembers" not in report.to_json()


def test_the_scan_boundary_never_narrows_the_substituters():
    """`scan >= substituter`, held mechanically over every needle shape.

    Two independently-written boundary builders drift, and the direction that
    drifts silently is the gate becoming NARROWER than the substituter: a form
    the build removes but the scan cannot see certifies a collection that still
    carries it. So every literal that matches the registry's own pattern must
    also match the gate's.
    """
    literals = ["Ada Example00", "Zylphia", "ada.example00", "ada_example00",
                "Example00, Ada", "Nord-Hansen", "O'Brien"]
    fenced = [f"{left}{literal}{right}"
              for literal in literals
              for left in ("", " ", "(", "%2C", ".", "-")
              for right in ("", " ", ")", "%40", ".", "-")]
    for literal in literals:
        registry = re.compile(boundaried(literal), re.I)
        gate = re.compile(index_scan.scan_boundaried(literal), re.I)
        for text in fenced:
            if registry.search(text):
                assert gate.search(text), (literal, text)
