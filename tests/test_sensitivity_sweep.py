"""Guards on the LOCAL sensitivity sweep (scripts/audit/sensitivity_sweep.py).

The sweep asks a local model who is still named in a built collection, so the
only parts that can be tested deterministically are the parts that matter: what
survives the model's answer, what the cache lets it skip, what it tells the
ledger, and what the packager does with the verdict. The model itself is a
monkeypatched callable throughout — a test that needed a GPU would not run.

Every name here is invented ("Ada Example00", "Zylphia Quorndal", "Kari
Ukjent"), and the map fixture is the one tests/test_scan_index.py builds. The
real map lives in a gitignored private sub-repo and is never read by the suite.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# By its package path, so this is the SAME module object package_collection
# imports — a bare `import sensitivity_sweep` off a sys.path entry would be a
# second copy with its own globals, and a monkeypatch of one would not reach the
# other (tests/test_package_collection.py depends on that).
from scripts.audit import sensitivity_sweep as sweep  # noqa: E402
from main.privacy.alias_registry import AliasRegistry  # noqa: E402
from main.runtime.indexing_run_ledger import IndexingRunLedger  # noqa: E402
from tests.test_scan_index import _map  # noqa: E402


@pytest.fixture
def classifier():
    data = _map()
    return sweep.ReferenceClassifier(AliasRegistry(data), data,
                                     allowed_bigrams={"bo kommune sentrum"})


# --- parsing ------------------------------------------------------------------

class TestParseReferences:
    def test_the_documented_shape(self):
        raw = '{"references": [{"text": "Ada Example00", "kind": "full_name"}]}'
        assert sweep.parse_references(raw) == [
            {"text": "Ada Example00", "kind": "full_name"}]

    def test_a_fenced_answer_is_still_read(self):
        """`format:"json"` makes this rare, not impossible, and a fence costing a
        whole document's findings would be a silent hole in the sweep."""
        raw = '```json\n{"references": [{"text": "Kari Ukjent", "kind": "full_name"}]}\n```'
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

    @pytest.mark.parametrize("raw", ["", "not json", "42", '{"references": null}',
                                     '{"references": [{"kind": "full_name"}]}'])
    def test_unusable_answers_yield_nothing(self, raw):
        assert sweep.parse_references(raw) == []


# --- classification -----------------------------------------------------------

class TestClassification:
    def test_a_full_name_nobody_mapped_is_an_unknown_person(self, classifier):
        text = "Referatet ble skrevet av Kari Ukjent i går."
        assert classifier.classify("Kari Ukjent", text) == sweep.UNKNOWN_PERSON

    def test_an_alias_is_bucketed_not_reported(self, classifier):
        """The substituter working. Counted so the report says so, rather than
        staying silent about the majority of what the model sees."""
        assert classifier.classify("dev-06", "Saken ble tatt av dev-06.") == sweep.ALIAS

    @pytest.mark.parametrize("token", ["[~person]", "[~ukjent-person]", "@person"])
    def test_a_redaction_token_is_an_alias(self, classifier, token):
        assert classifier.classify(token, f"Kontakt {token} for mer.") == sweep.ALIAS

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
        """`bare_given_name_residual` is the same population as
        `registry.given_names`, counted from the corpus. A residual bucketed as
        an unknown person would block a hand-off over the one thing the campaign
        explicitly decided not to substitute."""
        assert classifier.classify("Zylphia", "Zylphia kommer i morgen.") == \
            sweep.MAPPED_RESIDUAL

    def test_a_full_name_starting_with_a_mapped_given_name_is_still_unknown(self, classifier):
        """The residual carve-out is for a name standing ALONE. `Ada Nyansatt`
        is a full name nobody aliased, which is exactly what the sweep is for."""
        assert classifier.classify("Ada Nyansatt", "Ada Nyansatt begynte i dag.") == \
            sweep.UNKNOWN_PERSON

    def test_a_reference_the_document_does_not_contain_is_dropped(self, classifier):
        """The prompt demands a verbatim substring. An invented name in a privacy
        report is worse than a missed one: it is the finding a human spends the
        whole triage budget on."""
        assert classifier.classify("Kari Ukjent", "Dokumentet nevner ingen.") is None

    def test_the_containment_check_ignores_case(self, classifier):
        assert classifier.classify("KARI UKJENT", "signert av Kari Ukjent") == \
            sweep.UNKNOWN_PERSON


# --- cache --------------------------------------------------------------------

class TestCache:
    def test_the_key_covers_text_policy_and_model(self, monkeypatch):
        base = sweep.cache_key("hei", "model-a")
        assert base != sweep.cache_key("hei igjen", "model-a")
        assert base != sweep.cache_key("hei", "model-b")
        monkeypatch.setattr(sweep, "POLICY_VERSION", sweep.POLICY_VERSION + 1)
        assert base != sweep.cache_key("hei", "model-a")

    def test_a_round_trip_keeps_the_entries(self, tmp_path):
        path = tmp_path / "c.json"
        sweep.write_cache(path, "m", {"doc": {"hash": "h", "findings_count": 1,
                                              "unknown_count": 0}})
        assert sweep.load_cache(path, "m")["doc"]["hash"] == "h"

    def test_another_model_invalidates_the_whole_cache(self, tmp_path):
        path = tmp_path / "c.json"
        sweep.write_cache(path, "m", {"doc": {"hash": "h"}})
        assert sweep.load_cache(path, "other") == {}

    def test_another_policy_version_invalidates_the_whole_cache(self, tmp_path, monkeypatch):
        path = tmp_path / "c.json"
        sweep.write_cache(path, "m", {"doc": {"hash": "h"}})
        monkeypatch.setattr(sweep, "POLICY_VERSION", sweep.POLICY_VERSION + 1)
        assert sweep.load_cache(path, "m") == {}

    def test_a_flat_legacy_file_is_not_read_as_entries(self, tmp_path):
        """The envelope exists because a doc id may itself start with `_`, so a
        sibling metadata key is not safely distinguishable from an entry."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"doc": {"hash": "h"}}), encoding="utf-8")
        assert sweep.load_cache(path, "m") == {}

    def test_a_corrupt_file_is_a_cold_cache_not_a_crash(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{ truncated", encoding="utf-8")
        assert sweep.load_cache(path, "m") == {}


# --- the sweep itself ---------------------------------------------------------

def _collection(root: Path, documents) -> Path:
    """The two files run_sweep reads: a manifest and documents/*.json."""
    collection = root / "demo"
    (collection / "documents").mkdir(parents=True)
    for index, (doc_id, text) in enumerate(documents):
        (collection / "documents" / f"doc{index}.json").write_text(
            json.dumps({"id": doc_id, "url": f"file:///{doc_id}", "text": text,
                        "chunks": [{"indexedData": text}]}), encoding="utf-8")
    (collection / "manifest.json").write_text(json.dumps({
        "collectionName": "demo", "numberOfDocuments": len(documents),
        "updatedTime": "2026-08-01T00:00:00+00:00",
        "privacy": {"policy_version": 1, "map_version": 7},
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
                                    sweep.UNKNOWN_PERSON: 1}
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
        assert len(call.calls) == 3
        # Seen in more than one window, reported once.
        assert result["counts"][sweep.UNKNOWN_PERSON] == 1

    def test_incremental_skips_an_unchanged_document(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        cache = {"a.md": {"hash": sweep.cache_key("Skrevet av Kari Ukjent.", "m"),
                          "findings_count": 1, "unknown_count": 0}}
        call = _answers({})
        result = sweep.run_sweep(collection, classifier, model="m", baseline=False,
                                 cache=cache, call=call)
        assert call.calls == []
        assert result["documentsCached"] == 1 and result["documentsAsked"] == 0

    def test_baseline_ignores_the_cache(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        cache = {"a.md": {"hash": sweep.cache_key("Skrevet av Kari Ukjent.", "m"),
                          "findings_count": 0, "unknown_count": 0}}
        call = _answers({"Kari Ukjent": ["Kari Ukjent"]})
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                                 cache=cache, call=call)
        assert result["documentsAsked"] == 1 and result["unknownCount"] == 1

    def test_a_changed_document_is_re_asked(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        cache = {"a.md": {"hash": sweep.cache_key("noe helt annet", "m"),
                          "findings_count": 0, "unknown_count": 0}}
        call = _answers({"Kari Ukjent": ["Kari Ukjent"]})
        result = sweep.run_sweep(collection, classifier, model="m", baseline=False,
                                 cache=cache, call=call)
        assert result["documentsAsked"] == 1 and result["unknownCount"] == 1

    def test_a_cached_unknown_still_counts_towards_the_run(self, tmp_path, classifier):
        """A nightly incremental that skipped the one dirty document and then
        reported zero unknowns would flip the packaging gate from refuse to pass
        with nothing having been fixed."""
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        cache = {"a.md": {"hash": sweep.cache_key("Skrevet av Kari Ukjent.", "m"),
                          "findings_count": 1, "unknown_count": 1}}
        result = sweep.run_sweep(collection, classifier, model="m", baseline=False,
                                 cache=cache, call=_answers({}))
        assert result["unknownCount"] == 1 and result["unknownCountCached"] == 1

    def test_a_transport_failure_is_counted_not_raised(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])

        def call(prompt, **kwargs):
            raise RuntimeError("Ollama request failed")

        result = sweep.run_sweep(collection, classifier, model="m", baseline=True, call=call)
        assert result["parseFailures"] == 1 and result["unknownCount"] == 0

    def test_an_unparseable_answer_is_counted_as_a_failure(self, tmp_path, classifier):
        """Counted so a run that was mostly the model failing to answer cannot
        report zero findings and read as a pass."""
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

    def test_the_cache_entry_records_the_new_hash_and_counts(self, tmp_path, classifier):
        collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
        result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                                 call=_answers({"Kari Ukjent": ["Kari Ukjent"]}))
        entry = result["entries"]["a.md"]
        assert entry == {"hash": sweep.cache_key("Skrevet av Kari Ukjent.", "m"),
                         "findings_count": 1, "unknown_count": 1}


# --- stdout safety ------------------------------------------------------------

def test_the_stdout_summary_never_prints_a_name(tmp_path, classifier):
    """Everything printed is a shape or a count; the report file is the only
    output that carries real strings."""
    collection = _collection(tmp_path, [("a.md", "Skrevet av Kari Ukjent.")])
    result = sweep.run_sweep(collection, classifier, model="m", baseline=True,
                             call=_answers({"Kari Ukjent": ["Kari Ukjent"]}))
    printed = "\n".join(sweep._summary_lines(result))
    assert "Kari" not in printed and "Ukjent" not in printed
    assert "xxxx xxxxxx" in printed


# --- ledger -------------------------------------------------------------------

class TestLedgerRecord:
    def _folded(self, tmp_path, record):
        ledger = IndexingRunLedger(runs_dir=str(tmp_path))
        ledger.append(record)
        return ledger.recent(sweep.LEDGER_COLLECTION, limit=5)[-1]

    def test_a_clean_sweep_succeeds(self, tmp_path):
        run = self._folded(tmp_path, sweep.ledger_record(
            "succeeded", started_at="2026-08-23T10:00:00Z",
            finished_at="2026-08-23T10:05:00Z", detail={"unknown": 0}))
        assert run["collection"] == sweep.LEDGER_COLLECTION
        assert run["status"] == "succeeded"
        assert [p["name"] for p in run["phases"]] == [sweep.LEDGER_PHASE]
        assert run["durationSeconds"] == 300

    def test_an_unknown_person_degrades_the_run(self, tmp_path):
        run = self._folded(tmp_path, sweep.ledger_record(
            "degraded", started_at="2026-08-23T10:00:00Z",
            finished_at="2026-08-23T10:05:00Z", detail={"unknown": 3}))
        assert run["status"] == "degraded"

    def test_an_unreachable_model_fails_the_run(self, tmp_path):
        """A sweep that judged nothing must not read like a sweep that found
        nothing — hence the phase is marked fatal."""
        run = self._folded(tmp_path, sweep.ledger_record(
            "failed", started_at="2026-08-23T10:00:00Z",
            finished_at="2026-08-23T10:00:05Z", detail={}, error="ollama down"))
        assert run["status"] == "failed"

    def test_the_record_carries_no_opening_partial(self, tmp_path):
        """One foreground process either writes its record or died; an unmatched
        opener would fold to `incomplete` forever whenever someone Ctrl-Cs a
        manual sweep."""
        record = sweep.ledger_record("succeeded", started_at="2026-08-23T10:00:00Z",
                                     finished_at="2026-08-23T10:05:00Z", detail={})
        assert "stage" not in record
        assert self._folded(tmp_path, record)["status"] == "succeeded"


def test_report_run_falls_back_to_the_ledger_when_the_api_is_down(tmp_path, monkeypatch):
    """The same two-step the shell helper uses: the API is routinely down when an
    unattended job runs, and the ledger file must never be written by anything
    that cannot take the flock."""
    monkeypatch.setenv("HUGINN_RUNS_DIR", str(tmp_path))
    record = sweep.ledger_record("succeeded", started_at="2026-08-23T10:00:00Z",
                                 finished_at="2026-08-23T10:00:01Z", detail={})
    assert sweep.report_run(record, "http://127.0.0.1:59999") == "ledger"
    assert IndexingRunLedger(runs_dir=str(tmp_path)).recent(sweep.LEDGER_COLLECTION)


# --- the packaging gate --------------------------------------------------------

def _report(directory: Path, collection: str, *, generated_at, unknown=0, day="2026-08-23"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sweep_{collection}_{day}.json"
    path.write_text(json.dumps({"collection": collection, "generatedAt": generated_at,
                                "unknownCount": unknown, "findings": []}), encoding="utf-8")
    return path


class TestSweepGate:
    MANIFEST = {"updatedTime": "2026-08-20T00:00:00+00:00"}

    def test_no_report_warns_rather_than_refuses(self, tmp_path):
        """The sweep is a SECOND opinion. Making a hand-off depend on a local GPU
        being up is how a gate gets routed around."""
        status, message = sweep.sweep_gate("demo", self.MANIFEST, dirs=[tmp_path])
        assert status == "warn" and "no local sensitivity sweep" in message

    def test_a_clean_recent_report_passes(self, tmp_path):
        _report(tmp_path, "demo", generated_at="2026-08-22T00:00:00Z")
        assert sweep.sweep_gate("demo", self.MANIFEST, dirs=[tmp_path])[0] == "pass"

    def test_an_unknown_person_refuses(self, tmp_path):
        _report(tmp_path, "demo", generated_at="2026-08-22T00:00:00Z", unknown=2)
        status, message = sweep.sweep_gate("demo", self.MANIFEST, dirs=[tmp_path])
        assert status == "refuse" and "2 unknown" in message

    def test_a_report_older_than_the_rebuild_refuses(self, tmp_path):
        """A clean verdict about text that has since been replaced certifies
        nothing."""
        _report(tmp_path, "demo", generated_at="2026-08-19T00:00:00Z")
        status, message = sweep.sweep_gate("demo", self.MANIFEST, dirs=[tmp_path])
        assert status == "refuse" and "older than" in message

    def test_the_newest_report_wins_by_its_own_stamp(self, tmp_path):
        """By the stamp inside the file, not the filename date or the mtime: a
        report copied between machines keeps the content and loses both."""
        _report(tmp_path, "demo", generated_at="2026-08-21T00:00:00Z", unknown=5,
                day="2026-08-21")
        _report(tmp_path, "demo", generated_at="2026-08-22T00:00:00Z", day="2026-08-22")
        assert sweep.sweep_gate("demo", self.MANIFEST, dirs=[tmp_path])[0] == "pass"

    def test_another_collections_report_does_not_count(self, tmp_path):
        _report(tmp_path, "other", generated_at="2026-08-22T00:00:00Z")
        assert sweep.sweep_gate("demo", self.MANIFEST, dirs=[tmp_path])[0] == "warn"

    def test_a_report_without_an_unknown_count_is_refused(self, tmp_path):
        path = tmp_path / "sweep_demo_2026-08-23.json"
        path.write_text(json.dumps({"collection": "demo",
                                    "generatedAt": "2026-08-22T00:00:00Z"}),
                        encoding="utf-8")
        assert sweep.sweep_gate("demo", self.MANIFEST, dirs=[tmp_path])[0] == "refuse"


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
    monkeypatch.setattr(sweep, "ollama_reachable", lambda *a, **k: True)
    monkeypatch.setattr(sweep, "resolve_allowlist", lambda explicit: None)
    monkeypatch.setenv("HUGINN_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _cli(workspace, monkeypatch, answers, extra=()):
    monkeypatch.setattr(sweep, "call_ollama", _answers(answers))
    return sweep.main([
        "--collection", "demo",
        "--collections-dir", str(workspace / "collections"),
        "--cache-dir", str(workspace / "cache"),
        "--api-url", "http://127.0.0.1:59999",
        "--baseline", *extra,
    ])


class TestCli:
    def test_a_clean_collection_exits_zero_and_writes_a_report(
            self, cli_workspace, monkeypatch, capsys):
        _collection(cli_workspace / "collections", [("a.md", "Ingen navn her.")])
        assert _cli(cli_workspace, monkeypatch, {}) == 0
        report = json.loads(next((cli_workspace / "reports").glob("sweep_demo_*.json"))
                            .read_text(encoding="utf-8"))
        assert report["unknownCount"] == 0 and report["mode"] == "baseline"
        assert "RESULT: clean" in capsys.readouterr().out

    def test_an_unknown_person_exits_two(self, cli_workspace, monkeypatch, capsys):
        _collection(cli_workspace / "collections", [("a.md", "Skrevet av Kari Ukjent.")])
        assert _cli(cli_workspace, monkeypatch, {"Kari Ukjent": ["Kari Ukjent"]}) == 2
        out = capsys.readouterr().out
        # Exit 2 is the signal; the name itself stays in the gitignored report.
        assert "unknown person reference" in out and "Kari" not in out

    def test_an_unreachable_model_exits_one_and_records_a_failure(
            self, cli_workspace, monkeypatch):
        _collection(cli_workspace / "collections", [("a.md", "Ingen navn her.")])
        monkeypatch.setattr(sweep, "ollama_reachable", lambda *a, **k: False)
        assert _cli(cli_workspace, monkeypatch, {}) == 1
        run = IndexingRunLedger(runs_dir=str(cli_workspace / "runs")).recent(
            sweep.LEDGER_COLLECTION)[-1]
        assert run["status"] == "failed"

    def test_the_run_is_recorded_under_the_audit_collection(
            self, cli_workspace, monkeypatch):
        _collection(cli_workspace / "collections", [("a.md", "Skrevet av Kari Ukjent.")])
        _cli(cli_workspace, monkeypatch, {"Kari Ukjent": ["Kari Ukjent"]})
        run = IndexingRunLedger(runs_dir=str(cli_workspace / "runs")).recent(
            sweep.LEDGER_COLLECTION)[-1]
        assert run["status"] == "degraded"
        assert run["phases"][0]["detail"]["unknown"] == 1

    def test_the_cache_is_written_and_then_honoured(self, cli_workspace, monkeypatch):
        _collection(cli_workspace / "collections", [("a.md", "Skrevet av Kari Ukjent.")])
        _cli(cli_workspace, monkeypatch, {"Kari Ukjent": ["Kari Ukjent"]})
        cached = sweep.load_cache(cli_workspace / "cache" / "demo.json",
                                  sweep.DEFAULT_MODEL)
        assert cached["a.md"]["unknown_count"] == 1
        # A second, incremental run asks nothing and still reports the unknown.
        call = _answers({})
        monkeypatch.setattr(sweep, "call_ollama", call)
        assert sweep.main([
            "--collection", "demo",
            "--collections-dir", str(cli_workspace / "collections"),
            "--cache-dir", str(cli_workspace / "cache"),
            "--api-url", "http://127.0.0.1:59999",
        ]) == 2
        assert call.calls == []

    def test_a_tracked_report_path_is_refused(self, cli_workspace, monkeypatch):
        """The report is the one output with real names in it; a
        `--report-out report.json` typo is the mistake this campaign already made
        once, with a shorter fuse."""
        _collection(cli_workspace / "collections", [("a.md", "Ingen navn her.")])
        with pytest.raises(SystemExit) as exit_info:
            _cli(cli_workspace, monkeypatch, {},
                 extra=["--report-out", str(REPO_ROOT / "docs" / "sweep.json")])
        assert "REFUSED" in str(exit_info.value)
