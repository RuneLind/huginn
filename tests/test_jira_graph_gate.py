"""The write gate on the Jira graph extractor.

``scripts/knowledge_graph/extract_jira_graph.py`` is the only graph step that
reads a RAW pre-alias source tree — the built ``documents/`` drop ``parent`` on
30 % of issues, which is a third of the subtask edges, so repointing it was
measured and rejected — and it writes what it builds into a repo. So the WRITE
is gated: same needles, same boundaries, same NUL join as the distribution
gate's check 1.

On today's corpus the gate is a no-op; every test here builds the corpus where
it is not, plus the three ways it could pass vacuously (out of scope, no map,
a truncated map).

Every name is invented ("Ada Example00", "Zylphia Quorndal"). The real map lives
in a gitignored private sub-repo and is never read by the suite.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACTOR = REPO_ROOT / "scripts" / "knowledge_graph" / "extract_jira_graph.py"

from main.privacy import index_scan  # noqa: E402

# A public entry in main/privacy/scope.json's basePaths, so a source tree at
# <root>/data/sources/jira-issues is in scope on the public file alone.
IN_SCOPE_SOURCE = Path("data") / "sources" / "jira-issues"

REFUSED = 3          # the gate's exit code; deliberately not argparse's 2

MAPPED_NAME = "Ada Example00"
UNMAPPED_NAME = "Zylphia Quorndal"


def _map(entry_count=index_scan.MIN_MAP_ENTRIES):
    """A map with enough entries to clear the gate's floor.

    Only the fields the needle builder reads; ``person_forms_in_payload`` never
    compiles an ``AliasRegistry``, so the full build-time schema is not needed.
    """
    entries = [
        {"alias": f"dev-{i:02d}", "name": f"Ada Example{i:02d}",
         "variants": [f"Ada Example{i:02d}", f"Example{i:02d}, Ada", f"ada.example{i:02d}"]}
        for i in range(entry_count)
    ]
    return {
        "version": 7,
        "entries": entries,
        "non_person_labels": ["saksbehandler"],
        "unmapped_people_variants": {
            UNMAPPED_NAME: [UNMAPPED_NAME, "Quorndal, Zylphia", "zylphia.quorndal"],
        },
    }


def _issue(directory: Path, key: str, title: str, **frontmatter):
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f'title: "{title}"', f"issue_key: {key}", "status: Ferdig",
             "issue_type: Historie"]
    lines += [f'{name}: "{value}"' for name, value in frontmatter.items()]
    body = "\n".join(["---", *lines, "---", "", f"Body of {key}.", ""])
    (directory / f"{key}.md").write_text(body, encoding="utf-8")


def _run(root: Path, source: Path, output: Path, extra_env=None):
    """Drive the extractor as a subprocess with the privacy root relocated.

    HUGINN_PRIVACY_ROOT is what keeps a test from discovering the operator's
    real map and real scope: every private glob and every relative scope path
    resolves against it.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "HUGINN_PRIVACY_ROOT": str(root)}
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(EXTRACTOR), "--source", str(source), "--output", str(output)],
        cwd=root, env=env, capture_output=True, text=True, timeout=300,
    )


@pytest.fixture
def corpus(tmp_path):
    """An in-scope source tree with one clean issue, and a map that knows names."""
    source = tmp_path / IN_SCOPE_SOURCE
    _issue(source, "MELOSYS-1", "Avklare regelverket for utsendte")
    (tmp_path / "huginn-fixture" / "privacy").mkdir(parents=True)
    (tmp_path / "huginn-fixture" / "privacy" / "aliases.json").write_text(
        json.dumps(_map()), encoding="utf-8")
    return tmp_path, source


# --- the gate fires ---------------------------------------------------------

def test_a_mapped_name_in_an_issue_title_refuses_the_write(corpus):
    """The one field that carries free text into the graph is the title.

    Nothing else the extractor copies is prose — which is why the corpus
    measures clean and why this is the shape the gate exists for.
    """
    root, source = corpus
    _issue(source, "MELOSYS-2", f"Avklaring med {MAPPED_NAME} om regelverket")
    output = root / "graph.json"
    output.write_text("PREVIOUS GRAPH", encoding="utf-8")

    result = _run(root, source, output)

    assert result.returncode == REFUSED
    assert "REFUSED" in result.stderr
    # A refusal names shapes, never the person it refused on.
    assert MAPPED_NAME not in result.stderr and "Ada" not in result.stderr
    assert MAPPED_NAME not in result.stdout and "Ada" not in result.stdout
    assert "xxx" in result.stderr
    # Scanned in memory, so the previous graph is still the previous graph.
    assert output.read_text(encoding="utf-8") == "PREVIOUS GRAPH"


def test_an_unmapped_person_refuses_too(corpus):
    """`unmapped_people_variants` are needles as much as mapped entries are."""
    root, source = corpus
    _issue(source, "MELOSYS-2", f"Referat fra møte med {UNMAPPED_NAME}")
    output = root / "graph.json"

    result = _run(root, source, output)

    assert result.returncode == REFUSED
    assert not output.exists()


def test_an_epic_summary_is_scanned_as_well_as_a_title(corpus):
    """Epic nodes get their label from `epic_summary`, a second free-text field."""
    root, source = corpus
    _issue(source, "MELOSYS-2", "Ordinær tittel",
           epic_link="PK-1", epic_summary=f"Epos eid av {MAPPED_NAME}")
    output = root / "graph.json"

    result = _run(root, source, output)

    assert result.returncode == REFUSED
    assert not output.exists()


def test_the_gate_sees_the_permutations_the_scan_sees(corpus):
    """The needles are built the gate's way, not re-derived here.

    `Example00, Ada` is a permutation form: it is in no entry's `variants`, and
    a hand-rolled check comparing against the listed literals would let it
    through. Asserting it directly is what pins the gate to `build_needles`.
    """
    root, source = corpus
    map_path = root / "huginn-fixture" / "privacy" / "aliases.json"
    graph = {"nodes": [{"label": "MELOSYS-2: Avklaring med Example00, Ada om saken"}]}

    assert index_scan.person_forms_in_payload(graph, map_path)


# --- the gate does not fire -------------------------------------------------

def test_a_clean_in_scope_corpus_writes_normally(corpus):
    root, source = corpus
    output = root / "graph.json"

    result = _run(root, source, output)

    assert result.returncode == 0, result.stderr
    graph = json.loads(output.read_text(encoding="utf-8"))
    assert [n["id"] for n in graph["nodes"]] == ["issue:MELOSYS-1"]


def test_an_out_of_scope_source_is_not_gated_at_all(tmp_path):
    """A clone with no NAV source tree must behave exactly as it did before.

    Note there is no map anywhere under this root: out of scope, the gate never
    looks for one, so its absence cannot fail the run.
    """
    source = tmp_path / "data" / "sources" / "some-other-jira"
    _issue(source, "OTHER-1", f"Avklaring med {MAPPED_NAME} om regelverket")
    output = tmp_path / "graph.json"

    result = _run(tmp_path, source, output)

    assert result.returncode == 0, result.stderr
    assert MAPPED_NAME in output.read_text(encoding="utf-8")


# --- the ways it could pass vacuously ---------------------------------------

def test_an_in_scope_source_with_no_map_refuses(tmp_path):
    """Fail closed, the same way the index build refuses to build unaliased."""
    source = tmp_path / IN_SCOPE_SOURCE
    _issue(source, "MELOSYS-1", "Avklare regelverket for utsendte")
    output = tmp_path / "graph.json"

    result = _run(tmp_path, source, output)

    assert result.returncode == REFUSED
    assert "REFUSED" in result.stderr
    assert not output.exists()


def test_a_truncated_map_refuses_instead_of_certifying(corpus):
    """Few needles and a clean report look identical from the outside."""
    root, source = corpus
    (root / "huginn-fixture" / "privacy" / "aliases.json").write_text(
        json.dumps(_map(entry_count=3)), encoding="utf-8")
    _issue(source, "MELOSYS-2", f"Avklaring med {MAPPED_NAME} om regelverket")
    output = root / "graph.json"

    result = _run(root, source, output)

    assert result.returncode == REFUSED
    assert "floor" in result.stderr
    assert not output.exists()


def test_two_maps_are_ambiguous_rather_than_first_wins(corpus):
    root, source = corpus
    (root / "huginn-second" / "privacy").mkdir(parents=True)
    (root / "huginn-second" / "privacy" / "aliases.json").write_text(
        json.dumps(_map()), encoding="utf-8")
    output = root / "graph.json"

    result = _run(root, source, output)

    assert result.returncode == REFUSED
    assert not output.exists()


# --- the widened category set -----------------------------------------------

@pytest.mark.parametrize("title, category", [
    ("Feilsøk saken for A123456 i morgen", "nav_ident"),
    ("Avklart i tråden med @kari.moe om regelverket", "dotted_handle"),
    ("Overfør til konto 1234.56.78903 før fristen", "bankkonto"),
])
def test_the_other_blocking_categories_refuse_too(corpus, title, category):
    """A graph committed to a repo is a distribution surface.

    Check 1 alone answers only "did a LISTED person survive". An ident, a dotted
    handle or a bank account in an issue title is as unshippable as a name, and
    the title is the one free-text field this gate protects.
    """
    root, source = corpus
    _issue(source, "MELOSYS-2", title)
    output = root / "graph.json"

    result = _run(root, source, output)

    assert result.returncode == REFUSED, result.stdout + result.stderr
    assert category in result.stderr
    assert not output.exists()


def test_an_advisory_category_does_not_refuse(corpus):
    """An organisasjonsnummer identifies a company in a public register.

    It is legitimate content in an issue about an employer, it is the category
    that fires most, and the collection gate reports rather than blocks on it.
    """
    root, source = corpus
    _issue(source, "MELOSYS-2", "Arbeidsgiver med organisasjonsnummer 889640782")
    output = root / "graph.json"

    result = _run(root, source, output)

    assert result.returncode == 0, result.stderr
    assert output.exists()


# --- the boundaries, which are the whole reason not to re-derive -------------

@pytest.mark.parametrize("text", [
    # A dotted slug followed by another dotted label. The SUBSTITUTER leaves this
    # alone (it must not rewrite `no.nav.ada.example.impl`), so a gate sharing its
    # boundary verbatim would be blind to the person in `ada.example00.md`.
    "se ada.example00.md for detaljer",
    # A mononym after a percent escape: `%20Ada Example00` in a query string is
    # the name in the clear, and the substituter's narrowing hides it.
    "https://example.test/s?q=%20Ada%20Example00",
])
def test_the_scan_boundary_widenings_are_inherited_not_re_derived(corpus, text):
    root, _ = corpus
    map_path = root / "huginn-fixture" / "privacy" / "aliases.json"

    assert index_scan.person_forms_in_payload({"nodes": [{"label": text}]}, map_path)


# --- a map the build would refuse ------------------------------------------

@pytest.mark.parametrize("mutate", [
    lambda m: m.pop("entries"),
    lambda m: m.pop("unmapped_people_variants"),
    lambda m: m.pop("non_person_labels"),
])
def test_a_schema_drifted_map_refuses_rather_than_crashing(corpus, mutate):
    """One exit path, not a traceback the daily logs as an ordinary crash."""
    root, source = corpus
    map_path = root / "huginn-fixture" / "privacy" / "aliases.json"
    payload = _map()
    mutate(payload)
    map_path.write_text(json.dumps(payload), encoding="utf-8")
    output = root / "graph.json"

    result = _run(root, source, output)

    assert result.returncode == REFUSED, result.stdout + result.stderr
    assert "REFUSED" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_an_unreadable_map_refuses_rather_than_crashing(corpus):
    root, source = corpus
    map_path = root / "huginn-fixture" / "privacy" / "aliases.json"
    map_path.chmod(0o000)
    try:
        result = _run(root, source, root / "graph.json")
    finally:
        map_path.chmod(0o644)

    assert result.returncode == REFUSED
    assert "Traceback" not in result.stderr


# --- the silent exit-0 paths the ledger phase would call `succeeded` ---------

def test_a_missing_source_directory_exits_non_zero(tmp_path):
    result = _run(tmp_path, tmp_path / "nope", tmp_path / "graph.json")
    assert result.returncode != 0
    assert not (tmp_path / "graph.json").exists()


# --- the output is stable across processes ----------------------------------

def test_cross_reference_edges_are_emitted_in_a_stable_order(corpus):
    """The graph is a tracked file, so its byte order is part of its contract.

    ``cross_refs`` is a set, so unsorted its iteration order varies per process:
    two runs of the extractor over the real corpus differed by ~700 lines and
    churned the tracked graph in git nightly. PYTHONHASHSEED is pinned to two
    DIFFERENT values here to force divergence deterministically — the opposite
    of pinning it to hide the problem, which is how the byte-identical claim in
    huginn #122 had to be stated.
    """
    root, source = corpus
    refs = ["MELOSYS-%d" % n for n in (91, 17, 55, 3, 78, 26, 64, 40, 82, 9, 71, 33)]
    for key in refs:
        _issue(source, key, "Referenced issue")
    (source / "MELOSYS-1.md").write_text(
        "\n".join(["---", 'title: "Referring issue"', "issue_key: MELOSYS-1",
                    "status: Ferdig", "issue_type: Historie", "---", "",
                    "See " + ", ".join(refs) + ".", ""]),
        encoding="utf-8")

    outputs = []
    for seed in ("1", "2"):
        output = root / f"graph-{seed}.json"
        result = _run(root, source, output, extra_env={"PYTHONHASHSEED": seed})
        assert result.returncode == 0, result.stderr
        outputs.append(output.read_bytes())

    assert outputs[0] == outputs[1]

    # Stated directly too, so the test cannot pass by two seeds coinciding.
    graph = json.loads(outputs[0].decode("utf-8"))
    emitted = [e["target"] for e in graph["edges"]
               if e["type"] == "refererer_til" and e["source"] == "issue:MELOSYS-1"]
    assert emitted == sorted(f"issue:{key}" for key in refs)
