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
import pickle
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "audit"))

import scan_index as cli  # noqa: E402
from main.privacy import index_scan  # noqa: E402

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
                      documents_count=None, chunks_count=None, drop_chunk_count=False):
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

    corpus = [d["text"].lower().split() for d in documents]
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
    monkeypatch.setattr(cli, "REPO_ROOT", repo_root)
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


def test_prefilter_never_hides_a_hit():
    """The word prefilter is a speed trick; a miss there is a silent pass."""
    needles = ["Ada Example", "zylphia.quorndal"]
    prefilter = index_scan.needle_prefilter(needles)
    assert index_scan.may_contain_needle("skrevet av Ada Example", prefilter)
    assert index_scan.may_contain_needle("?f=%2CAda%20Example%40x.no", prefilter)
    assert index_scan.may_contain_needle("se zylphia.quorndal", prefilter)
    assert not index_scan.may_contain_needle("en helt vanlig setning", prefilter)


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


def test_a_bigram_does_not_span_a_line_break(tmp_path, map_file):
    """The last word of one line and the first of the next are both capitalised
    often enough that crossing the break invents candidates."""
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Sendt til Kari\nUkjent mottok den.")])
    assert _scan(tmp_path, map_file).check("bigram_candidates").passed is True


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

def test_an_organisasjonsnummer_fails_the_scan(tmp_path, map_file):
    _write_collection(tmp_path, "demo-aliased",
                      documents=[_document(0, "Arbeidsgiver med orgnr 987654325.")])
    check = _scan(tmp_path, map_file).check("sensitive_tokens")
    # Twice: the fixture document carries the same sentence as `text` and as the
    # single chunk's `indexedData`, and both are artifacts that ship.
    assert check.passed is False and check.count == 2


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
