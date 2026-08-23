"""Guards on the LOCAL sensitivity sweep.

Library (`main/privacy/sensitivity_sweep.py`) and CLI
(`scripts/audit/sensitivity_sweep.py`). The sweep asks a local model who is
still named in a built collection, so the only parts that can be tested
deterministically are the parts that matter: what survives the model's answer,
what the cache lets it skip, what it tells the ledger, and what the packager
does with the verdict. The model itself is an injected callable throughout — a
test that needed a GPU would not run.

Every name here is invented ("Ada Example00", "Zylphia Quorndal", "Kari
Ukjent"), and the map fixture is the one tests/test_scan_index.py builds. The
real map lives in a gitignored private sub-repo and is never read by the suite.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from main.privacy import sensitivity_sweep as sweep  # noqa: E402
from main.privacy.alias_registry import AliasRegistry  # noqa: E402
from main.runtime.indexing_run_ledger import IndexingRunLedger  # noqa: E402
from scripts.audit import sensitivity_sweep as cli  # noqa: E402
from tests.test_scan_index import MAP_VERSION, _map  # noqa: E402


@pytest.fixture
def classifier():
    data = _map()
    return sweep.ReferenceClassifier(AliasRegistry(data), data,
                                     allowed_bigrams={"bo kommune sentrum"})


# --- parsing ------------------------------------------------------------------
#
# One function, two answers that must never be confused: `[]` is "the model says
# no one is named here", `None` is "the model said something unreadable". A
# document of the second kind is not clean, is not cached, and counts towards the
# run being INCONCLUSIVE.

class TestParseReferences:
    def test_the_documented_shape(self):
        raw = '{"references": [{"text": "Ada Example00", "kind": "full_name"}]}'
        assert sweep.parse_references(raw) == [
            {"text": "Ada Example00", "kind": "full_name"}]

    def test_no_references_is_an_empty_list_not_a_failure(self):
        assert sweep.parse_references('{"references": []}') == []

    @pytest.mark.parametrize("raw", [
        '```json\n{"references": [{"text": "Kari Ukjent", "kind": "full_name"}]}\n```',
        '```JSON\n{"references": [{"text": "Kari Ukjent", "kind": "full_name"}]}\n```',
        '```\n{"references": [{"text": "Kari Ukjent", "kind": "full_name"}]}\n```',
    ])
    def test_a_fenced_answer_is_still_read(self, raw):
        """`format:"json"` makes this rare, not impossible, and a fence costing a
        whole document's findings would be a silent hole in the sweep. The
        language tag is matched case-insensitively — models emit ```JSON."""
        assert sweep.parse_references(raw)[0]["text"] == "Kari Ukjent"

    def test_a_bare_list_and_bare_strings_are_accepted(self):
        assert sweep.parse_references('["Kari Ukjent"]') == [
            {"text": "Kari Ukjent", "kind": "other"}]

    def test_an_unknown_kind_is_coerced_not_dropped(self):
        """The model's label is a hint; the classification below is the answer.
        Dropping the reference over its label would lose a real finding."""
        raw = '{"references": [{"text": "Kari Ukjent", "kind": "menneske"}]}'
        assert sweep.parse_references(raw) == [{"text": "Kari Ukjent", "kind": "other"}]

    def test_whitespace_in_a_reference_is_collapsed(self):
        raw = '{"references": [{"text": "Kari\\n  Ukjent", "kind": "full_name"}]}'
        assert sweep.parse_references(raw)[0]["text"] == "Kari Ukjent"

    @pytest.mark.parametrize("raw", [
        "", "not json", "42", "null", '"a string"',
        '{"references": null}',
        '{"references": {"text": "Kari Ukjent"}}',
        '{"people": [{"text": "Kari Ukjent"}]}',
        '{"text": "Kari Ukjent", "kind": "full_name"}',
    ])
    def test_an_answer_without_a_reference_list_is_unreadable(self, raw):
        """Schema drift is a parse FAILURE, not an empty result. A model that
        answers `{"people": …}` has stopped following the contract, and reading
        that as "nobody is named" is the vacuous pass this whole counter exists
        to catch."""
        assert sweep.parse_references(raw) is None

    def test_an_item_without_text_is_skipped_inside_a_readable_answer(self):
        raw = '{"references": [{"kind": "full_name"}, {"text": "Kari Ukjent"}]}'
        assert sweep.parse_references(raw) == [{"text": "Kari Ukjent", "kind": "other"}]


# --- classification -----------------------------------------------------------

class TestClassification:
    def test_a_full_name_nobody_mapped_is_an_unknown_person(self, classifier):
        text = "Referatet ble skrevet av Kari Ukjent i går."
        assert classifier.classify("Kari Ukjent", text) == sweep.UNKNOWN_PERSON

    def test_an_alias_is_bucketed_not_reported(self, classifier):
        """The substituter working. Counted so the report says so, rather than
        staying silent about the majority of what the model sees."""
        assert classifier.classify("dev-06", "Saken ble tatt av dev-06.") == sweep.ALIAS

    def test_a_capitalised_alias_is_still_an_alias(self, classifier):
        """`AliasRegistry.to_names` is deliberately case-sensitive (it feeds a
        query rewrite). A model quoting back a sentence-initial `Dev-06` was
        therefore reported as an unknown person — the substituter's own output,
        top of the triage list."""
        assert classifier.classify("Dev-06", "Dev-06 tok saken.") == sweep.ALIAS

    @pytest.mark.parametrize("token", ["[~person]", "[~ukjent-person]", "@person"])
    def test_a_redaction_token_is_an_alias(self, classifier, token):
        assert classifier.classify(token, f"Kontakt {token} for mer.") == sweep.ALIAS

    def test_a_handle_token_is_matched_with_a_boundary(self, classifier):
        """`@person` was matched as a bare substring, so `@personalt` — a handle
        that is not the redaction token — was filed as the substituter working
        and never reported."""
        assert classifier.classify("@personalt", "Skriv til @personalt om det.") == \
            sweep.UNKNOWN_PERSON

    @pytest.mark.parametrize("fragment", ["dev", "person", "ukjent-person"])
    def test_a_fragment_of_a_token_is_dropped(self, classifier, fragment):
        """The model returns the head of `dev-06` and the middle of `[~person]`
        often enough to matter. Neither is a person nobody mapped; both are the
        tokenizer showing through, and reported they are the noise that makes a
        triage list unreadable."""
        assert classifier.classify(fragment, f"En {fragment} gjorde det.") is None

    def test_a_finding_shorter_than_three_characters_is_dropped(self, classifier):
        assert classifier.classify("Ka", "Ka skjer?") is None

    def test_an_exempt_label_is_dropped(self, classifier):
        """`non_person_labels` is the list the campaign already adjudicated;
        re-litigating it nightly is how a gate gets switched off."""
        assert classifier.classify("saksbehandler", "En saksbehandler tar saken.") is None

    def test_the_reviewed_bigram_allowlist_is_dropped(self, classifier):
        assert classifier.classify("Bo Kommune Sentrum",
                                   "Møtet var i Bo Kommune Sentrum.") is None

    def test_a_bare_mapped_given_name_is_a_residual_not_an_unknown(self, classifier):
        """The residual is a deliberate decision — `Ada` would alias half the
        corpus — so it is counted, never alarmed on."""
        assert classifier.classify("Ada", "Ada svarte på det i går.") == sweep.MAPPED_RESIDUAL

    def test_a_residual_given_name_from_the_map_counts_too(self, classifier):
        assert classifier.classify("Zylphia", "Zylphia kommer i morgen.") == \
            sweep.MAPPED_RESIDUAL

    def test_a_multi_token_residual_key_counts_as_a_residual(self):
        """`bare_given_name_residual` keys are counted from the corpus, not
        constructed as single tokens. The single-token guard belongs to
        `registry.given_names`, and applying it here reported every multi-word
        residual key as an unknown person, every night."""
        data = _map()
        data["bare_given_name_residual"]["Zylphia Quorndal"] = 2
        classifier = sweep.ReferenceClassifier(AliasRegistry(data), data)
        assert classifier.classify("Zylphia Quorndal", "Zylphia Quorndal kommer.") == \
            sweep.MAPPED_RESIDUAL

    def test_a_full_name_starting_with_a_mapped_given_name_is_still_unknown(self, classifier):
        """The `given_names` carve-out is for a name standing ALONE. `Ada
        Nyansatt` is a full name nobody aliased, which is what the sweep is
        for."""
        assert classifier.classify("Ada Nyansatt", "Ada Nyansatt begynte i dag.") == \
            sweep.UNKNOWN_PERSON

    def test_a_role_phrase_is_reported_but_never_blocking(self, classifier):
        """"the case worker who signed" identifies someone to a reader who
        already knows the case. That is a judgement call, so it goes in the
        report as `role` and stays out of `unknownCount`, which refuses a
        hand-off."""
        assert classifier.classify("den nye teamlederen", "Vi spurte den nye teamlederen.",
                                   kind="role") == sweep.ROLE

    @pytest.mark.parametrize("kind", ["full_name", "surname", "handle", "initials",
                                      "other", None])
    def test_every_other_kind_is_an_unknown_person(self, classifier, kind):
        """`other` is the coercion for a label the model invented — treated as a
        MISSING kind, i.e. as a person. Only an explicit `role` downgrades."""
        assert classifier.classify("Kari Ukjent", "Skrevet av Kari Ukjent.", kind=kind) == \
            sweep.UNKNOWN_PERSON

    def test_a_reference_the_document_does_not_contain_is_dropped(self, classifier):
        """The prompt demands a verbatim substring. An invented name in a privacy
        report is worse than a missed one: it is the finding a human spends the
        whole triage budget on."""
        assert classifier.classify("Kari Ukjent", "Dokumentet nevner ingen.") is None

    def test_the_containment_check_ignores_case(self, classifier):
        assert classifier.classify("KARI UKJENT", "signert av Kari Ukjent") == \
            sweep.UNKNOWN_PERSON

    def test_a_name_the_document_wraps_across_a_line_is_still_found(self, classifier):
        """Both sides of the containment test are whitespace-normalised. The
        model quotes the name back on one line; the document has a newline in the
        middle of it, and a raw `in` test called the real finding a
        hallucination — the sweep failing in the one direction nobody notices."""
        assert classifier.classify("Kari Ukjent", "Skrevet av Kari\nUkjent i går.") == \
            sweep.UNKNOWN_PERSON


# --- windowing ------------------------------------------------------------------

class TestWindows:
    def test_a_short_document_is_one_window(self):
        assert sweep._windows("kort") == ["kort"]

    def test_an_empty_document_is_no_window_at_all(self):
        """Otherwise the sweep spends an 11-second model call asking who is named
        in the empty string, once per empty document in the collection."""
        assert sweep._windows("") == []

    def test_no_trailing_window_is_shorter_than_the_overlap(self):
        """Such a window is a strict suffix of the one before it — every
        character in it was already sent — so it is a duplicate model call."""
        step = sweep.WINDOW_CHARS - sweep.WINDOW_OVERLAP
        windows = sweep._windows("x" * (2 * step + 50))
        assert len(windows) == 2
        assert all(len(w) >= sweep.WINDOW_OVERLAP for w in windows)

    @pytest.mark.parametrize("length", [1, 6000, 6001, 11650, 11900, 20000, 30000])
    def test_every_character_is_inside_some_window(self, length):
        """Dropping the short trailing window must not open a hole at the end —
        it is a suffix of its predecessor, so the predecessor already covers it,
        and this is the assertion that keeps that true."""
        step = sweep.WINDOW_CHARS - sweep.WINDOW_OVERLAP
        text = "".join(chr(0x61 + i % 26) for i in range(length))
        windows = sweep._windows(text)
        covered = 0
        for index, window in enumerate(windows):
            start = index * step
            assert window == text[start:start + sweep.WINDOW_CHARS]
            covered = max(covered, start + len(window))
        assert covered == length


def _straddling_document(offset_from_boundary=5):
    """A document whose only name sits across the first window boundary."""
    start = sweep.WINDOW_CHARS - offset_from_boundary
    filler = "x" * start
    return {"id": "a.md", "text": filler + "Kari Ukjent" + "y" * 6000}


class TestWindowOverlap:
    """The overlap is load-bearing, so a test has to fail when it is removed."""

    def test_a_name_across_a_boundary_is_found(self, classifier):
        result = sweep.sweep_document(_straddling_document(), classifier, model="m",
                                      timeout=1, call=_answers({"Kari Ukjent":
                                                                ["Kari Ukjent"]}))
        assert [f["text"] for f in result[0]] == ["Kari Ukjent"]

    def test_without_the_overlap_it_is_found_by_neither_window(self, classifier,
                                                               monkeypatch):
        """Both halves of the name are in the corpus and neither window holds it
        whole, so the model cannot quote it and the classifier's verbatim rule
        would drop it if it did. This is the failure the overlap prevents."""
        monkeypatch.setattr(sweep, "WINDOW_OVERLAP", 0)
        result = sweep.sweep_document(_straddling_document(), classifier, model="m",
                                      timeout=1, call=_answers({"Kari Ukjent":
                                                                ["Kari Ukjent"]}))
        assert result[0] == []

    def test_a_name_inside_the_overlap_is_reported_once(self, classifier):
        """It is genuinely in two windows, so the model returns it twice."""
        start = sweep.WINDOW_CHARS - sweep.WINDOW_OVERLAP + 20
        document = {"id": "a.md", "text": "x" * start + "Kari Ukjent" + "y" * 6000}
        findings, failures, windows = sweep.sweep_document(
            document, classifier, model="m", timeout=1,
            call=_answers({"Kari Ukjent": ["Kari Ukjent"]}))
        assert windows == 3 and [f["text"] for f in findings] == ["Kari Ukjent"]


# --- cache --------------------------------------------------------------------

class TestCache:
    def test_the_key_covers_everything_that_changes_the_verdict(self, monkeypatch):
        base = sweep.cache_key("hei", "model-a", map_version=7, allowlist_sha="a" * 64)
        assert base != sweep.cache_key("hei igjen", "model-a", map_version=7,
                                       allowlist_sha="a" * 64)
        assert base != sweep.cache_key("hei", "model-b", map_version=7,
                                       allowlist_sha="a" * 64)
        assert base != sweep.cache_key("hei", "model-a", map_version=8,
                                       allowlist_sha="a" * 64)
        assert base != sweep.cache_key("hei", "model-a", map_version=7,
                                       allowlist_sha="b" * 64)
        monkeypatch.setattr(sweep, "POLICY_VERSION", sweep.POLICY_VERSION + 1)
        assert base != sweep.cache_key("hei", "model-a", map_version=7,
                                       allowlist_sha="a" * 64)

    def test_a_round_trip_keeps_the_entries(self, tmp_path):
        path = tmp_path / "c.json"
        sweep.write_cache(path, "m", {"doc": {"hash": "h", "unknown": []}}, 7, "sha")
        assert sweep.load_cache(path, "m", 7, "sha")["doc"]["hash"] == "h"

    @pytest.mark.parametrize("other", [
        {"model": "other", "map_version": 7, "allowlist_sha": "sha"},
        {"model": "m", "map_version": 8, "allowlist_sha": "sha"},
        {"model": "m", "map_version": 7, "allowlist_sha": "other-sha"},
    ])
    def test_a_changed_input_invalidates_the_whole_cache(self, tmp_path, other):
        """Wholesale rather than per entry: the composite key would reject every
        entry one at a time anyway, and saying so once is clearer in a log."""
        path = tmp_path / "c.json"
        sweep.write_cache(path, "m", {"doc": {"hash": "h"}}, 7, "sha")
        assert sweep.load_cache(path, other["model"], other["map_version"],
                                other["allowlist_sha"]) == {}

    def test_another_policy_version_invalidates_the_whole_cache(self, tmp_path, monkeypatch):
        path = tmp_path / "c.json"
        sweep.write_cache(path, "m", {"doc": {"hash": "h"}}, 7, "sha")
        monkeypatch.setattr(sweep, "POLICY_VERSION", sweep.POLICY_VERSION + 1)
        assert sweep.load_cache(path, "m", 7, "sha") == {}

    def test_a_flat_legacy_file_is_not_read_as_entries(self, tmp_path):
        """The envelope exists because a doc id may itself start with `_`, so a
        sibling metadata key is not safely distinguishable from an entry. A flat
        file has no metadata, so it cannot match — this cache has no
        pre-envelope generation to keep."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"doc": {"hash": "h"}}), encoding="utf-8")
        assert sweep.load_cache(path, "m", 7, "sha") == {}

    def test_a_corrupt_file_is_a_cold_cache_not_a_crash(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{ truncated", encoding="utf-8")
        assert sweep.load_cache(path, "m", 7, "sha") == {}

    def test_the_allowlist_sha_follows_the_file(self, tmp_path):
        path = tmp_path / "bigrams.json"
        path.write_text('{"bigrams": ["bo kommune"]}', encoding="utf-8")
        first = sweep.allowlist_sha256(path)
        path.write_text('{"bigrams": ["bo kommune", "kari ukjent"]}', encoding="utf-8")
        assert first != sweep.allowlist_sha256(path)
        assert sweep.allowlist_sha256(tmp_path / "absent.json") is None


# --- the sweep itself ---------------------------------------------------------

def _collection(root: Path, documents, *, expected=None) -> Path:
    """The two files run_sweep reads: a manifest and documents/*.json."""
    collection = root / "demo"
    (collection / "documents").mkdir(parents=True)
    for index, (doc_id, text) in enumerate(documents):
        (collection / "documents" / f"doc{index}.json").write_text(
            json.dumps({"id": doc_id, "url": f"file:///{doc_id}", "text": text,
                        "chunks": [{"indexedData": text}]}), encoding="utf-8")
    (collection / "manifest.json").write_text(json.dumps({
        "collectionName": "demo",
        "numberOfDocuments": len(documents) if expected is None else expected,
        "updatedTime": "2026-08-01T00:00:00+00:00",
        "lastModifiedDocumentTime": "2026-07-31T10:00:00",
        "privacy": {"policy_version": 1, "map_version": MAP_VERSION},
        "reader": {"type": "localFiles", "basePath": "./data/sources/demo"},
    }), encoding="utf-8")
    return collection


def _answers(mapping):
    """A fake model that answers per prompt substring, and counts its calls."""
    calls = []

    def call(prompt, **kwargs):
        calls.append(prompt)
        for needle, names in mapping.items():
            if needle in prompt:
                return json.dumps({"references": [{"text": n, "kind": "full_name"}
                                                  for n in names]})
        return '{"references": []}'

    call.calls = calls
    return call


def _key(text, model="m"):
    return sweep.cache_key(text, model, map_version=None, allowlist_sha=None)


class TestRunSweep:
    def test_it_buckets_and_counts(self, tmp_path, classifier):
        collection = _collection(tmp_path, [
            ("a.md", "Skrevet av Kari Ukjent."),
            ("b.md", "Saken tas av dev-06 og Ada svarer."),
        ])
        call = _answers({"Kari Ukjent": ["Kari Ukjent"],
                         "dev-06": ["dev-06", "Ada"]})
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True, call=call)
        assert result["counts"] == {sweep.ALIAS: 1, sweep.MAPPED_RESIDUAL: 1,
                                    sweep.ROLE: 0, sweep.UNKNOWN_PERSON: 1}
        assert result["unknownCount"] == 1
        assert {f["documentId"] for f in result["findings"]} == {"a.md", "b.md"}

    def test_a_long_document_is_windowed_rather_than_truncated(self, tmp_path, classifier):
        """The long tail runs to 30 KB. Truncating it would make the sweep
        quietly blind to the second half of the pages most likely to name
        someone."""
        tail = "x" * (sweep.WINDOW_CHARS * 2) + " Kari Ukjent"
        collection = _collection(tmp_path, [("a.md", tail)])
        call = _answers({"Kari Ukjent": ["Kari Ukjent"]})
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True, call=call)
        assert len(call.calls) == result["windows"] == 3
        # Seen in more than one window, reported once.
        assert result["counts"][sweep.UNKNOWN_PERSON] == 1

    def test_incremental_skips_an_unchanged_document(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        cache = {"a.md": {"hash": _key("Skrevet av Kari Ukjent."),
                          "findings_count": 1, "unknown_count": 0, "unknown": []}}
        call = _answers({})
        result = sweep.run_sweep(collection, classifier, model="m", baseline=False,
                                 cache=cache, call=call)
        assert call.calls == []
        assert result["documentsCached"] == 1 and result["documentsAsked"] == 0

    def test_baseline_ignores_the_cache(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        cache = {"a.md": {"hash": _key("Skrevet av Kari Ukjent."),
                          "findings_count": 0, "unknown_count": 0, "unknown": []}}
        call = _answers({"Kari Ukjent": ["Kari Ukjent"]})
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                                 cache=cache, call=call)
        assert result["documentsAsked"] == 1 and result["unknownCount"] == 1

    def test_a_changed_document_is_re_asked(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        cache = {"a.md": {"hash": _key("noe helt annet"), "unknown": []}}
        call = _answers({"Kari Ukjent": ["Kari Ukjent"]})
        result = sweep.run_sweep(collection, classifier, model="m", baseline=False,
                                 cache=cache, call=call)
        assert result["documentsAsked"] == 1 and result["unknownCount"] == 1

    def test_a_cached_unknown_still_counts_and_still_names_itself(self, tmp_path,
                                                                  classifier):
        """A nightly incremental that skipped the one dirty document and reported
        zero unknowns would flip the packaging gate from refuse to pass with
        nothing fixed. And the gate's refusal says "the report has the strings",
        so the cached entry has to carry them — a count alone tells a triager a
        name exists somewhere in 538 documents."""
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        cache = {"a.md": {"hash": _key("Skrevet av Kari Ukjent."),
                          "findings_count": 1, "unknown_count": 1,
                          "unknown": ["Kari Ukjent"]}}
        result = sweep.run_sweep(collection, classifier, model="m", baseline=False,
                                 cache=cache, call=_answers({}))
        assert result["unknownCount"] == 1 and result["unknownCountCached"] == 1
        assert [f["text"] for f in result["findings"]] == ["Kari Ukjent"]
        assert result["findings"][0]["fromCache"] is True

    def test_a_transport_failure_is_counted_not_raised(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])

        def call(prompt, **kwargs):
            raise RuntimeError("Ollama request failed")

        result = sweep.run_sweep(collection, classifier, model="m", baseline=True, call=call)
        assert result["parseFailures"] == 1 and result["unknownCount"] == 0

    @pytest.mark.parametrize("error", [
        OSError("connection reset"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ])
    def test_every_transport_failure_mode_is_counted(self, tmp_path, classifier, error):
        """A refused socket and a non-UTF-8 body are "this window went unread"
        just as much as the RuntimeError the wrapper raises. Uncaught, either one
        stops the run and the other 500 documents go unjudged."""
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])

        def call(prompt, **kwargs):
            raise error

        result = sweep.run_sweep(collection, classifier, model="m", baseline=True, call=call)
        assert result["parseFailures"] == 1

    def test_an_unparseable_answer_is_counted_as_a_failure(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                                 call=lambda prompt, **kw: "I am afraid I cannot")
        assert result["parseFailures"] == 1

    def test_an_empty_answer_is_a_failure_not_a_clean_document(self, tmp_path, classifier):
        """`call_ollama` returns "" when the response carries no message. Silence
        is the one answer that looks exactly like "nobody is named here"."""
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                                 call=lambda prompt, **kw: "")
        assert result["parseFailures"] == 1

    def test_a_fenced_empty_answer_is_not_a_failure(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Ingen navn.")])
        result = sweep.run_sweep(
            collection, classifier, model="m", baseline=True,
            call=lambda prompt, **kw: '```json\n{"references": []}\n```')
        assert result["parseFailures"] == 0

    def test_the_cache_entry_records_the_hash_the_counts_and_the_strings(
            self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                                 call=_answers({"Kari Ukjent": ["Kari Ukjent"]}))
        assert result["entries"]["a.md"] == {
            "hash": _key("Skrevet av Kari Ukjent."), "findings_count": 1,
            "unknown_count": 1, "unknown": ["Kari Ukjent"]}

    def test_a_document_with_an_unread_window_is_not_cached(self, tmp_path, classifier):
        """Caching it would make the gap permanent: the hash matches on every
        later run, so the window that went unread is never asked about again."""
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                                 call=lambda prompt, **kw: "I cannot help")
        assert result["entries"] == {}

    def test_a_previously_cached_document_that_goes_unread_loses_its_entry(
            self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        cache = {"a.md": {"hash": _key("Skrevet av Kari Ukjent."), "unknown": []}}
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                                 cache=cache, call=lambda prompt, **kw: "nope")
        assert result["entries"] == {}

    def test_entries_for_documents_that_are_gone_are_pruned(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Ingen navn.")])
        cache = {"a.md": {"hash": _key("Ingen navn."), "unknown": []},
                 "deleted.md": {"hash": "whatever", "unknown": ["Kari Ukjent"]}}
        result = sweep.run_sweep(collection, classifier, model="m", baseline=False,
                                 cache=cache, call=_answers({}))
        assert set(result["entries"]) == {"a.md"}

    def test_a_limited_run_does_not_prune_what_it_did_not_look_at(self, tmp_path,
                                                                  classifier):
        """Pruning against the SLICED list would delete the cache for every
        document the limit skipped — the whole cache, on the next `--limit 5`."""
        collection = _collection(tmp_path, [("a.md", "Ingen navn."), ("b.md", "Heller ikke.")])
        cache = {"b.md": {"hash": _key("Heller ikke."), "unknown": []}}
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                                 limit=1, cache=cache, call=_answers({}))
        assert set(result["entries"]) == {"a.md", "b.md"}

    def test_an_empty_document_costs_no_model_call(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "")])
        call = _answers({})
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True, call=call)
        assert call.calls == [] and result["windows"] == 0


# --- stdout safety ------------------------------------------------------------

def test_the_stdout_summary_never_prints_a_name(tmp_path, classifier):
    """Everything printed is a shape or a count; the report file is the only
    output that carries real strings."""
    collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
    result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                             call=_answers({"Kari Ukjent": ["Kari Ukjent"]}))
    printed = "\n".join(sweep.summary_lines(result))
    assert "Kari" not in printed and "Ukjent" not in printed
    assert "xxxx xxxxxx" in printed


# --- ledger -------------------------------------------------------------------

class TestLedgerRecord:
    def _folded(self, tmp_path, record):
        ledger = IndexingRunLedger(runs_dir=str(tmp_path))
        ledger.append(record)
        return ledger.recent(sweep.LEDGER_COLLECTION, limit=5)[-1]

    def _record(self, status, **kwargs):
        kwargs.setdefault("started_at", "2026-08-23T10:00:00Z")
        kwargs.setdefault("finished_at", "2026-08-23T10:05:00Z")
        kwargs.setdefault("detail", {})
        kwargs.setdefault("collection", "demo")
        kwargs.setdefault("baseline", False)
        return sweep.ledger_record(status, **kwargs)

    def test_a_clean_sweep_succeeds(self, tmp_path):
        run = self._folded(tmp_path, self._record("succeeded", detail={"unknown": 0}))
        assert run["collection"] == sweep.LEDGER_COLLECTION
        assert run["status"] == "succeeded"
        assert [p["name"] for p in run["phases"]] == [sweep.LEDGER_PHASE]
        assert run["durationSeconds"] == 300

    def test_an_unknown_person_degrades_the_run(self, tmp_path):
        assert self._folded(tmp_path, self._record(
            "degraded", detail={"unknown": 3}))["status"] == "degraded"

    def test_an_unreachable_model_fails_the_run(self, tmp_path):
        """A sweep that judged nothing must not read like a sweep that found
        nothing — hence the phase is marked fatal."""
        assert self._folded(tmp_path, self._record(
            "failed", error="ollama down"))["status"] == "failed"

    def test_the_record_carries_no_opening_partial(self, tmp_path):
        """One foreground process either writes its record or died; an unmatched
        opener would fold to `incomplete` forever whenever someone Ctrl-Cs a
        manual sweep."""
        record = self._record("succeeded")
        assert "stage" not in record
        assert self._folded(tmp_path, record)["status"] == "succeeded"

    def test_the_swept_collection_stays_in_the_phase_detail(self, tmp_path):
        """The ledger row is keyed `sensitivity-audit`; WHICH collection was
        swept is only legible here."""
        run = self._folded(tmp_path, self._record("succeeded", collection="nav-demo"))
        assert run["phases"][0]["detail"]["collection"] == "nav-demo"

    @pytest.mark.parametrize("baseline,variant", [(True, "rebuild"), (False, "incremental")])
    def test_the_variant_follows_the_mode(self, tmp_path, baseline, variant):
        """A baseline re-reads every document and an incremental reads the
        changed ones; they differ by an order of magnitude, and one median over
        both describes neither."""
        assert self._folded(tmp_path, self._record(
            "succeeded", baseline=baseline))["variant"] == variant

    def test_two_sweeps_in_the_same_second_are_two_runs(self, tmp_path):
        """Every sweep is recorded under ONE collection key, so a runId keyed on
        a whole-second stamp folded two collections swept back to back into one
        run — and the second verdict vanished."""
        ledger = IndexingRunLedger(runs_dir=str(tmp_path))
        for collection in ("demo-a", "demo-b"):
            ledger.append(self._record("succeeded", collection=collection))
        runs = ledger.recent(sweep.LEDGER_COLLECTION, limit=10)
        assert len(runs) == 2
        assert len({run["runId"] for run in runs}) == 2

    def test_the_run_id_names_the_swept_collection(self):
        assert self._record("succeeded", collection="demo")["runId"].startswith(
            f"{sweep.LEDGER_COLLECTION}-demo-")


def test_report_run_falls_back_to_the_ledger_when_the_api_is_down(tmp_path, monkeypatch):
    """The same two-step the shell helper uses: the API is routinely down when an
    unattended job runs, and the ledger file must never be written by anything
    that cannot take the flock."""
    monkeypatch.setenv("HUGINN_RUNS_DIR", str(tmp_path))
    record = sweep.ledger_record("succeeded", started_at="2026-08-23T10:00:00Z",
                                 finished_at="2026-08-23T10:00:01Z", detail={},
                                 collection="demo", baseline=False)
    assert sweep.report_run(record, "http://127.0.0.1:59999") == "ledger"
    assert IndexingRunLedger(runs_dir=str(tmp_path)).recent(sweep.LEDGER_COLLECTION)


# --- the report file ------------------------------------------------------------

class TestReportPath:
    def test_the_name_carries_the_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sweep, "report_dirs", lambda: [tmp_path])
        assert sweep.report_path("demo", mode="incremental").name.endswith(
            "_incremental.json")

    def test_a_limited_run_cannot_land_on_a_full_baseline(self, tmp_path, monkeypatch):
        """Two runs on one day wrote the same filename, so a `--limit 50` spot
        check silently replaced the evidence the packaging gate reads."""
        monkeypatch.setattr(sweep, "report_dirs", lambda: [tmp_path])
        full = sweep.report_path("demo", mode="baseline")
        limited = sweep.report_path("demo", mode="baseline", limit=50)
        assert full != limited and limited.name.endswith("_baseline-limit50.json")


# --- the packaging gate --------------------------------------------------------

FULL = {"unknownCount": 0, "documents": 10, "documentsExpected": 10, "limit": None,
        "mapVersion": MAP_VERSION, "policyVersion": sweep.POLICY_VERSION, "model": "m",
        "collectionLastModifiedDocumentTime": "2026-08-20T13:02:01.631639",
        "findings": []}

MANIFEST = {"lastModifiedDocumentTime": "2026-08-20T13:02:01.631639",
            "numberOfDocuments": 10}


def _report(directory: Path, collection: str, *, generated_at="2026-08-22T00:00:00Z",
            name=None, **fields):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"collection": collection, "generatedAt": generated_at, **FULL, **fields}
    path = directory / (name or f"sweep_{collection}_2026-08-23_baseline.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate(tmp_path, **kwargs):
    manifest = kwargs.pop("manifest", MANIFEST)
    return sweep.sweep_gate("demo", manifest, dirs=[tmp_path], map_version=MAP_VERSION,
                            model="m", **kwargs)


class TestSweepGate:
    def test_no_report_warns_rather_than_refuses(self, tmp_path):
        """The sweep is a SECOND opinion. Making a hand-off depend on a local GPU
        being up is how a gate gets routed around."""
        status, message = _gate(tmp_path)
        assert status == "warn" and "no local sensitivity sweep" in message

    def test_a_clean_full_report_passes(self, tmp_path):
        _report(tmp_path, "demo")
        status, message = _gate(tmp_path)
        assert status == "pass" and "10/10" in message

    def test_an_unknown_person_refuses(self, tmp_path):
        _report(tmp_path, "demo", unknownCount=2)
        status, message = _gate(tmp_path)
        assert status == "refuse" and "2 unknown" in message
        # The count travels, the strings stay in the gitignored report.
        assert "Kari" not in message

    def test_an_unknown_person_in_a_LIMITED_report_still_refuses(self, tmp_path):
        """Coverage bounds what a CLEAN verdict proves. It does not soften a
        finding: a person the model actually found is found whatever else went
        unread."""
        _report(tmp_path, "demo", unknownCount=1, limit=5, documents=5,
                name="sweep_demo_2026-08-23_baseline-limit5.json")
        assert _gate(tmp_path)[0] == "refuse"

    def test_a_limited_report_cannot_certify(self, tmp_path):
        _report(tmp_path, "demo", limit=5, documents=5,
                name="sweep_demo_2026-08-23_baseline-limit5.json")
        status, message = _gate(tmp_path)
        assert status == "warn" and "5/10" in message

    def test_a_partial_report_without_a_limit_cannot_certify_either(self, tmp_path):
        """A run interrupted, or one against a collection that has grown since:
        `documents != documentsExpected` says so without any flag being set."""
        _report(tmp_path, "demo", documents=7)
        status, message = _gate(tmp_path)
        assert status == "warn" and "7/10" in message

    def test_a_limited_report_never_supersedes_a_full_one(self, tmp_path):
        """A 50-document sample certifying a 538-document hand-off is the exact
        failure this ordering prevents — and the newer file is the sample."""
        _report(tmp_path, "demo", generated_at="2026-08-21T00:00:00Z")
        _report(tmp_path, "demo", generated_at="2026-08-24T00:00:00Z", limit=5,
                documents=5, name="sweep_demo_2026-08-24_baseline-limit5.json")
        assert _gate(tmp_path)[0] == "pass"

    def test_a_pre_coverage_report_is_treated_as_partial(self, tmp_path):
        """Reports written before the coverage fields cannot prove they read the
        whole collection. Warn — an unprovable clean verdict must not certify."""
        path = tmp_path / "sweep_demo_2026-08-23.json"
        path.write_text(json.dumps({"collection": "demo",
                                    "generatedAt": "2026-08-22T00:00:00Z",
                                    "unknownCount": 0}), encoding="utf-8")
        assert _gate(tmp_path)[0] == "warn"

    def test_the_newest_full_report_wins_by_its_own_stamp(self, tmp_path):
        """By the stamp inside the file, not the filename date or the mtime: a
        report copied between machines keeps the content and loses both."""
        _report(tmp_path, "demo", generated_at="2026-08-21T00:00:00Z", unknownCount=5,
                name="sweep_demo_2026-08-21_baseline.json")
        _report(tmp_path, "demo", generated_at="2026-08-22T00:00:00Z")
        assert _gate(tmp_path)[0] == "pass"

    def test_another_collections_report_does_not_count(self, tmp_path):
        _report(tmp_path, "other", name="sweep_other_2026-08-23_baseline.json")
        assert _gate(tmp_path)[0] == "warn"

    def test_a_report_without_an_unknown_count_is_refused(self, tmp_path):
        _report(tmp_path, "demo", unknownCount=None)
        assert _gate(tmp_path)[0] == "refuse"

    def test_a_changed_collection_warns(self, tmp_path):
        """`lastModifiedDocumentTime`, NOT `updatedTime`: the latter moves on
        every no-op reindex, so comparing it would warn every night on a
        collection nothing touched — the graph source stamp's own reasoning."""
        _report(tmp_path, "demo", collectionLastModifiedDocumentTime="2026-08-01T00:00:00")
        status, message = _gate(tmp_path)
        assert status == "warn" and "has since changed" in message

    def test_a_no_op_reindex_does_not_invalidate_the_sweep(self, tmp_path):
        _report(tmp_path, "demo", collectionUpdatedTime="2026-08-20T00:00:00+00:00")
        assert _gate(tmp_path, manifest={**MANIFEST,
                                         "updatedTime": "2026-08-23T04:00:00+00:00"})[0] \
            == "pass"

    def test_a_naive_stamp_matches_its_utc_twin(self, tmp_path):
        """The manifest writes the stamp without an offset and the report with a
        `Z`. Compared as strings, the two forms of one instant never match and
        every collection reads as changed."""
        _report(tmp_path, "demo",
                collectionLastModifiedDocumentTime="2026-08-20T13:02:01.631639")
        assert _gate(tmp_path, manifest={**MANIFEST,
                                         "lastModifiedDocumentTime":
                                             "2026-08-20T13:02:01.631639Z"})[0] == "pass"

    @pytest.mark.parametrize("stamp", ["whenever", "", None])
    def test_an_unreadable_stamp_warns_rather_than_matching(self, tmp_path, stamp):
        """Two unreadable stamps are not evidence that nothing changed. Letting
        them compare equal would make a manifest with no stamp at all the
        easiest way to silence this check."""
        _report(tmp_path, "demo", collectionLastModifiedDocumentTime=stamp)
        assert _gate(tmp_path, manifest={**MANIFEST,
                                         "lastModifiedDocumentTime": stamp})[0] == "warn"

    @pytest.mark.parametrize("drift", [{"mapVersion": MAP_VERSION - 1},
                                       {"policyVersion": sweep.POLICY_VERSION + 1},
                                       {"model": "some-other-model"}])
    def test_a_verdict_produced_under_other_inputs_warns(self, tmp_path, drift):
        """The map, the policy and the model each decide what survives the
        filter. A clean verdict from a different one is about a different
        filter."""
        _report(tmp_path, "demo", **drift)
        status, message = _gate(tmp_path)
        assert status == "warn" and "different inputs" in message

    def test_the_model_defaults_to_the_machines_general_purpose_one(self, tmp_path):
        from main.utils.ollama_cli import DEFAULT_MODEL
        _report(tmp_path, "demo", model=DEFAULT_MODEL)
        assert sweep.sweep_gate("demo", MANIFEST, dirs=[tmp_path],
                                map_version=MAP_VERSION)[0] == "pass"


# --- a run nobody could read is not a clean run -------------------------------
#
# `unknownCount: 0` from a run where the model answered nothing readable looks,
# in the report, exactly like a genuinely clean collection. That is the shape a
# vacuous pass takes here — the same failure `index_scan`'s map-entry and
# gazetteer floors exist to prevent.

class TestUnreadableRuns:
    def test_a_run_within_the_failure_budget_is_readable(self):
        assert sweep.answers_are_readable(117, 5)

    def test_a_run_past_the_budget_is_not(self):
        assert not sweep.answers_are_readable(117, 38)

    def test_a_run_that_asked_nothing_is_vacuously_readable(self):
        """Every document cached: the cached verdicts are what carry the run."""
        assert sweep.answers_are_readable(0, 0)

    def test_the_gate_will_not_call_an_unreadable_report_clean(self, tmp_path):
        _report(tmp_path, "demo", windows=100, parseFailures=90)
        status, message = _gate(tmp_path)
        assert status == "warn" and "not evidence" in message

    def test_an_unknown_person_still_outranks_unreadability(self, tmp_path):
        """A refusal must not be downgraded to a warning by a bad answer rate."""
        _report(tmp_path, "demo", unknownCount=2, windows=100, parseFailures=90)
        assert _gate(tmp_path)[0] == "refuse"

    def test_a_report_without_the_fields_is_still_readable(self, tmp_path):
        """Absent counters are not a bad answer rate; the coverage check is what
        judges an old report, not this one."""
        _report(tmp_path, "demo")
        assert _gate(tmp_path)[0] == "pass"


# --- the CLI ------------------------------------------------------------------

@pytest.fixture
def cli_workspace(tmp_path, monkeypatch):
    """A tmp repo root whose map, report directory and ledger are all scratch.

    `alias_registry.REPO_ROOT` is what `discover_map_path` resolves the private
    glob against, so repointing it is what keeps the run off the real map.
    """
    from main.privacy import alias_registry
    privacy = tmp_path / "huginn-x" / "privacy"
    privacy.mkdir(parents=True)
    (privacy / "aliases.json").write_text(json.dumps(_map()), encoding="utf-8")
    monkeypatch.setattr(alias_registry, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(sweep, "report_dirs", lambda: [tmp_path / "reports"])
    monkeypatch.setattr(cli, "ollama_reachable", lambda *a, **k: True)
    monkeypatch.setattr(cli, "resolve_allowlist", lambda explicit: None)
    monkeypatch.setenv("HUGINN_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _cli(workspace, monkeypatch, answers, extra=("--baseline",)):
    monkeypatch.setattr(cli, "call_ollama", _answers(answers))
    return cli.main([
        "--collection", "demo",
        "--collections-dir", str(workspace / "collections"),
        "--cache-dir", str(workspace / "cache"),
        "--api-url", "http://127.0.0.1:59999",
        *extra,
    ])


def _written_report(workspace):
    return json.loads(next((workspace / "reports").glob("sweep_demo_*.json"))
                      .read_text(encoding="utf-8"))


class TestCli:
    def test_a_clean_collection_exits_zero_and_writes_a_report(
            self, cli_workspace, monkeypatch, capsys):
        _collection(cli_workspace / "collections", [("a.md", "Ingen navn her.")])
        assert _cli(cli_workspace, monkeypatch, {}) == 0
        report = _written_report(cli_workspace)
        assert report["unknownCount"] == 0 and report["mode"] == "baseline"
        assert "RESULT: clean" in capsys.readouterr().out

    def test_the_report_carries_what_the_gate_reads(self, cli_workspace, monkeypatch):
        """Coverage, the inputs the verdict was produced under, and the stamp of
        the collection it read. Without all four the gate cannot tell a full
        clean run from an unprovable one."""
        _collection(cli_workspace / "collections", [("a.md", "Ingen navn her.")])
        _cli(cli_workspace, monkeypatch, {})
        report = _written_report(cli_workspace)
        assert report["documents"] == report["documentsExpected"] == 1
        assert report["limit"] is None
        assert report["mapVersion"] == MAP_VERSION
        assert report["policyVersion"] == sweep.POLICY_VERSION
        assert report["collectionLastModifiedDocumentTime"] == "2026-07-31T10:00:00"
        assert report["collectionUpdatedTime"] == "2026-08-01T00:00:00+00:00"

    def test_a_limited_run_writes_a_report_the_gate_will_not_certify(
            self, cli_workspace, monkeypatch):
        _collection(cli_workspace / "collections",
                    [("a.md", "Ingen navn."), ("b.md", "Heller ikke.")])
        assert _cli(cli_workspace, monkeypatch, {},
                    extra=("--baseline", "--limit", "1")) == 0
        report = _written_report(cli_workspace)
        assert report["limit"] == 1 and report["documents"] == 1
        assert report["documentsExpected"] == 2
        assert sweep.is_full_report(report) is False

    def test_a_limit_is_allowed_in_incremental_mode_too(self, cli_workspace, monkeypatch):
        """It used to be documented as `--baseline`-only while working in both;
        a priced sample of the changed documents is a reasonable thing to want."""
        _collection(cli_workspace / "collections",
                    [("a.md", "Ingen navn."), ("b.md", "Heller ikke.")])
        assert _cli(cli_workspace, monkeypatch, {}, extra=("--limit", "1")) == 0
        report = _written_report(cli_workspace)
        assert report["mode"] == "incremental" and report["limit"] == 1

    def test_the_filename_says_which_run_it_was(self, cli_workspace, monkeypatch):
        _collection(cli_workspace / "collections", [("a.md", "Ingen navn.")])
        _cli(cli_workspace, monkeypatch, {}, extra=("--baseline", "--limit", "1"))
        name = next((cli_workspace / "reports").glob("sweep_demo_*.json")).name
        assert name.endswith("_baseline-limit1.json")

    def test_an_unknown_person_exits_two(self, cli_workspace, monkeypatch, capsys):
        _collection(cli_workspace / "collections", [("a.md", "Skrevet av Kari Ukjent.")])
        assert _cli(cli_workspace, monkeypatch, {"Kari Ukjent": ["Kari Ukjent"]}) == 2
        out = capsys.readouterr().out
        # Exit 2 is the signal; the name itself stays in the gitignored report.
        assert "unknown person reference" in out and "Kari" not in out

    def test_an_unreachable_model_exits_one_and_records_a_failure(
            self, cli_workspace, monkeypatch):
        _collection(cli_workspace / "collections", [("a.md", "Ingen navn her.")])
        monkeypatch.setattr(cli, "ollama_reachable", lambda *a, **k: False)
        assert _cli(cli_workspace, monkeypatch, {}) == 1
        run = IndexingRunLedger(runs_dir=str(cli_workspace / "runs")).recent(
            sweep.LEDGER_COLLECTION)[-1]
        assert run["status"] == "failed"

    def test_a_crash_mid_run_still_leaves_a_ledger_record(self, cli_workspace, monkeypatch):
        """No record at all reads on the dashboard exactly like a night the sweep
        was never scheduled. The record is the only durable trace an unattended
        run leaves."""
        _collection(cli_workspace / "collections", [("a.md", "Ingen navn her.")])

        def exploding(*args, **kwargs):
            raise MemoryError("out of memory")

        monkeypatch.setattr(cli, "run_sweep", exploding)
        with pytest.raises(MemoryError):
            _cli(cli_workspace, monkeypatch, {})
        run = IndexingRunLedger(runs_dir=str(cli_workspace / "runs")).recent(
            sweep.LEDGER_COLLECTION)[-1]
        assert run["status"] == "failed" and "MemoryError" in run["error"]

    def test_the_run_is_recorded_under_the_audit_collection(
            self, cli_workspace, monkeypatch):
        _collection(cli_workspace / "collections", [("a.md", "Skrevet av Kari Ukjent.")])
        _cli(cli_workspace, monkeypatch, {"Kari Ukjent": ["Kari Ukjent"]})
        run = IndexingRunLedger(runs_dir=str(cli_workspace / "runs")).recent(
            sweep.LEDGER_COLLECTION)[-1]
        assert run["status"] == "degraded"
        assert run["variant"] == "rebuild"
        assert run["phases"][0]["detail"]["unknown"] == 1
        assert run["phases"][0]["detail"]["collection"] == "demo"

    def test_the_cache_is_written_and_then_honoured(self, cli_workspace, monkeypatch):
        _collection(cli_workspace / "collections", [("a.md", "Skrevet av Kari Ukjent.")])
        _cli(cli_workspace, monkeypatch, {"Kari Ukjent": ["Kari Ukjent"]})
        cached = sweep.load_cache(cli_workspace / "cache" / "demo.json",
                                  cli.DEFAULT_MODEL, MAP_VERSION, None)
        assert cached["a.md"]["unknown_count"] == 1
        # A second, incremental run asks nothing, still reports the unknown, and
        # still carries the string a triager needs.
        call = _answers({})
        monkeypatch.setattr(cli, "call_ollama", call)
        assert cli.main([
            "--collection", "demo",
            "--collections-dir", str(cli_workspace / "collections"),
            "--cache-dir", str(cli_workspace / "cache"),
            "--api-url", "http://127.0.0.1:59999",
        ]) == 2
        assert call.calls == []
        incremental = sorted((cli_workspace / "reports").glob("*_incremental.json"))
        payload = json.loads(incremental[-1].read_text(encoding="utf-8"))
        assert [f["text"] for f in payload["findings"]] == ["Kari Ukjent"]

    def test_a_tracked_report_path_is_refused(self, cli_workspace, monkeypatch):
        """The report is the one output with real names in it; a
        `--report-out report.json` typo is the mistake this campaign already made
        once, with a shorter fuse."""
        _collection(cli_workspace / "collections", [("a.md", "Ingen navn her.")])
        with pytest.raises(SystemExit) as exit_info:
            _cli(cli_workspace, monkeypatch, {},
                 extra=("--baseline", "--report-out", str(REPO_ROOT / "docs" / "sweep.json")))
        assert "REFUSED" in str(exit_info.value)

    def test_a_missing_collection_exits_one(self, cli_workspace, monkeypatch):
        assert _cli(cli_workspace, monkeypatch, {}) == 1

    def test_the_cli_reports_inconclusive_and_degrades_the_run(
            self, cli_workspace, monkeypatch, capsys):
        _collection(cli_workspace / "collections", [("a.md", "x" * 20000)])
        monkeypatch.setattr(cli, "call_ollama", lambda prompt, **kw: "I cannot help")
        code = cli.main([
            "--collection", "demo",
            "--collections-dir", str(cli_workspace / "collections"),
            "--cache-dir", str(cli_workspace / "cache"),
            "--api-url", "http://127.0.0.1:59999", "--baseline",
        ])
        assert code == 0 and "INCONCLUSIVE" in capsys.readouterr().out
        run = IndexingRunLedger(runs_dir=str(cli_workspace / "runs")).recent(
            sweep.LEDGER_COLLECTION)[-1]
        assert run["status"] == "degraded"


# --- orchestrator inline round: an unreadable run must not supersede a refusal ---

class TestRefusalsSurviveLaterNoise:
    def test_an_unreadable_newer_report_does_not_supersede_a_refusing_one(self, tmp_path):
        _report(tmp_path, "demo", generated_at="2026-08-20T00:00:00Z",
                name="sweep_demo_2026-08-20T000000Z_baseline.json", unknownCount=2)
        _report(tmp_path, "demo", generated_at="2026-08-21T00:00:00Z",
                name="sweep_demo_2026-08-21T000000Z_baseline.json",
                unknownCount=0, windows=4, parseFailures=4)
        status, message = _gate(tmp_path)
        assert status == "refuse" and "2 unknown" in message

    def test_the_report_name_carries_a_time_so_a_same_day_rerun_cannot_overwrite(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sweep, "report_dirs", lambda: [tmp_path])
        name = sweep.report_path("demo", mode="baseline").name
        assert "T" in name and name.endswith("Z_baseline.json")

    def test_a_span_that_merely_contains_an_alias_is_still_a_person(self, classifier):
        doc = "Skrevet av Ola Nordmann (dev-01) i dag."
        assert classifier.classify("Ola Nordmann (dev-01)", doc) == sweep.UNKNOWN_PERSON
        assert classifier.classify("Dev-01", doc.replace("dev-01", "Dev-01")) == sweep.ALIAS

    def test_a_clean_full_report_with_role_phrases_warns(self, tmp_path):
        _report(tmp_path, "demo", counts={sweep.ROLE: 2})
        status, message = _gate(tmp_path)
        assert status == "warn" and "role" in message
