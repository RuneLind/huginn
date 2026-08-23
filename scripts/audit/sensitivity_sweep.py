#!/usr/bin/env python3
"""A LOCAL second opinion on a built collection: ask a local model who is named.

    .venv/bin/python scripts/audit/sensitivity_sweep.py --collection nav-wiki
    .venv/bin/python scripts/audit/sensitivity_sweep.py --collection nav-wiki --baseline
    .venv/bin/python scripts/audit/sensitivity_sweep.py --collection nav-wiki --baseline --limit 50

``main/privacy/index_scan.py`` is the deterministic gate and stays the thing that
blocks a hand-off. Two questions it cannot answer:

* a person the alias map has never heard of, spelled in any form other than the
  capitalised-bigram shape check 9 retains (a mononym, a surname on its own, an
  initials-plus-surname byline, a role phrase that identifies exactly one
  individual);
* whether a bare given name standing in prose is a person reference at all — the
  residual the campaign decided not to substitute is five figures of
  occurrences, and a regex cannot read them.

So this sweep reads the collection's DERIVED documents (``documents/*.json`` —
the aliased text, never ``data/sources/``) and asks a model to list the person
references it can see. Everything it says is then filtered deterministically:
aliases and redaction tokens are dropped, the map's ``non_person_labels`` and the
private reviewed bigram allow-list are dropped, and a reference the document does
not literally contain is dropped as a hallucination. What survives is bucketed
into ``alias`` / ``mapped_residual`` / ``unknown_person``.

**LOCAL BY REQUIREMENT.** The transport is ``main/utils/ollama_cli.py`` and
nothing else. A sensitivity check that ships document text to a hosted model IS
the leak it is looking for, so there is no ``--backend`` flag here and there
never should be — the point of the whole privacy campaign is that this corpus
does not leave the machine.

Two modes:

* ``--baseline`` — every document (``--limit`` bounds a priced sample).
* default (incremental) — only documents whose text hash changed since the cache.

Outputs:

* a **gitignored** JSON report (real strings, for a human to triage) at
  ``<private-sub-repo>/privacy/sweep_<collection>_<date>.json``, falling back to
  ``data/privacy/`` when no private sub-repo carries a ``privacy/`` directory;
* a stdout summary of counts and SHAPES only, safe to paste;
* a run record under the ledger collection key ``sensitivity-audit``.

Exit codes: ``0`` clean, ``2`` when an ``unknown_person`` survived the filters,
``1`` when the model could not be reached (nothing was judged).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.privacy.alias_registry import (  # noqa: E402
    HANDLE_TOKEN, IDENT_TOKEN, POLICY_VERSION, USER_PATH_TOKEN,
    AliasRegistry, PrivacyMapMissing, _private_glob, discover_map_path, shape as sanitize,
)
from main.privacy.index_scan import documents_of, load_allowed_bigrams  # noqa: E402
from main.runtime.indexing_run_ledger import (  # noqa: E402
    IndexingRunLedger, mint_run_id, now_iso,
)
from main.utils.ollama_cli import DEFAULT_MODEL, call_ollama  # noqa: E402
from scripts.audit.scan_index import (  # noqa: E402
    DEFAULT_CANDIDATES_DIR, DEFAULT_COLLECTIONS_DIR, refuse_if_tracked, resolve_allowlist,
)

# The ledger collection key every sweep run is recorded under. Not the swept
# collection's own key: a sweep is not an indexing run of it, and folding the two
# together would put a `degraded` sweep on the row a reader uses to answer "is the
# index fresh". `_COLLECTION_RE` accepts it, and `/api/indexing/jobs` shows it with
# `loaded: false` by design — no server serves a collection by this name.
LEDGER_COLLECTION = "sensitivity-audit"
LEDGER_PHASE = "sweep"

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "state" / "sensitivity"
PRIVATE_REPORT_GLOB = "huginn-*/privacy"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# Prompt window. Most documents are far shorter than this (nav-wiki's median is
# ~240 characters), but the long tail runs to 30 KB and truncating it would make
# the sweep quietly blind to the second half of exactly the pages most likely to
# name someone. Windows overlap so a name straddling a boundary is still seen
# whole in one of them; duplicates are collapsed per document.
WINDOW_CHARS = 6000
WINDOW_OVERLAP = 200

# Kinds the model may return. Anything else is coerced to `other` rather than
# dropped — the classification below is what decides, not the model's label.
KINDS = ("full_name", "given_name", "surname", "handle", "initials", "role")

SYSTEM_PROMPT = """You are a privacy reviewer for an internal knowledge base.

The text you are given has ALREADY had its known people replaced by pseudonyms
(tokens shaped like "dev-06") and by redaction markers (tokens in square
brackets, "@person", "<user>"). Your job is to find the person references that
are STILL in the clear.

List every reference to a specific human being that appears in the text:
- a full name
- a given name used on its own to refer to someone
- a family name used on its own
- a chat handle, username or email local part that identifies a person
- initials plus a family name
- a role phrase that identifies exactly ONE specific individual in context

Do NOT list:
- organisations, teams, systems, products, projects, places or countries
- generic role nouns that do not point at one specific person
- the pseudonyms and redaction markers described above
- anything you cannot find verbatim in the text

Return ONLY JSON in this exact form:
{"references": [{"text": "<verbatim substring of the text>", "kind": "full_name|given_name|surname|handle|initials|role"}]}

Return {"references": []} when the text names no one. Copy each "text" exactly
as it is spelled in the document; do not normalise, translate or complete it."""

USER_PROMPT_TEMPLATE = """Document title: {title}

---
{text}
---

List the person references still in the clear in the text above."""


# --- model output parsing ----------------------------------------------------

def _strip_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def answered_in_json(raw: str) -> bool:
    """Did the model answer at all? Distinguishes "no one is named here" from
    "the model said something this script cannot read" — which is not a clean
    document and must not be counted as one."""
    try:
        json.loads(_strip_fences(raw))
    except ValueError:
        return False
    return True


def parse_references(raw: str) -> list:
    """References out of one model response, or [] for anything unusable.

    Tolerant on purpose: ``format:"json"`` makes a fenced or prose-wrapped answer
    rare but not impossible, and a sweep that raises on one malformed response
    stops judging the other 500 documents. A response that cannot be read is a
    parse failure the caller counts (see ``answered_in_json``), never a clean
    document.
    """
    if not raw:
        return []
    try:
        payload = json.loads(_strip_fences(raw))
    except ValueError:
        return []
    if isinstance(payload, dict):
        items = payload.get("references")
        if items is None:
            # Some responses answer with a bare object per reference.
            items = [payload] if payload.get("text") else []
    elif isinstance(payload, list):
        items = payload
    else:
        return []
    references = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        value = item.get("text") or item.get("name") or item.get("reference")
        if not isinstance(value, str) or not value.strip():
            continue
        kind = item.get("kind") if item.get("kind") in KINDS else "other"
        references.append({"text": " ".join(value.split()), "kind": kind})
    return references


# --- deterministic classification --------------------------------------------

ALIAS = "alias"
MAPPED_RESIDUAL = "mapped_residual"
UNKNOWN_PERSON = "unknown_person"

# Every literal the substituter can leave behind, matched as a SUBSTRING — which
# is why `EMAIL_LOCAL_TOKEN` ("person") is deliberately not in the list: it would
# swallow "personnummer" and every Norwegian compound built on the same word,
# turning a real finding into an `alias`. The four here are distinctive enough
# that a substring match cannot mean anything else.
_REDACTION_TOKENS = (IDENT_TOKEN, HANDLE_TOKEN, USER_PATH_TOKEN, "<user>")

_SINGLE_TOKEN_RE = re.compile(r"^[^\W\d_]+(?:['’\-][^\W\d_]+)*$")


class ReferenceClassifier:
    """The deterministic half of the sweep. Nothing here asks the model anything.

    Order matters, and it is the order of increasing doubt:

    1. **not in the document** — dropped. The prompt demands a verbatim
       substring; a reference that is not one was invented, and an invented name
       in a privacy report is worse than a missed one because it is the finding a
       human will spend the triage budget on.
    2. **an alias or a redaction token** — bucketed ``alias``. This is the
       substituter working, reported so the report says so rather than staying
       silent about the majority of what the model sees.
    3. **an exempt label or a reviewed non-person bigram** — dropped. These are
       the two lists the campaign already adjudicated by hand; re-litigating them
       every night is how a gate gets switched off.
    4. **a bare given name the map knows** — bucketed ``mapped_residual``. The
       residual is a deliberate decision (`Ada` would alias half the corpus), so
       it is counted, not alarmed on.
    5. everything else — ``unknown_person``, which fails the sweep.
    """

    def __init__(self, registry: AliasRegistry, alias_map: dict, allowed_bigrams=frozenset()):
        self._registry = registry
        self._redaction = {t.lower() for t in (*_REDACTION_TOKENS, registry.redaction_token)}
        self._exempt = {label.strip().lower()
                        for label in alias_map.get("non_person_labels", [])
                        if isinstance(label, str) and label.strip()}
        self._allowed = {b.lower() for b in allowed_bigrams}
        # `registry.given_names` is the first token of every entry name and every
        # unmapped label. The map's `bare_given_name_residual` keys are the same
        # population counted from the corpus, and a residual bucketed as
        # `unknown_person` would block a hand-off over the one thing the campaign
        # explicitly decided not to substitute.
        self._given_names = set(registry.given_names) | {
            name.strip().lower()
            for name in alias_map.get("bare_given_name_residual", {})
            if isinstance(name, str) and name.strip()}

    def _contains_alias(self, text: str) -> bool:
        lowered = text.lower()
        if any(token and token in lowered for token in self._redaction):
            return True
        # `to_names` rewrites `dev-06` back to a name; a text it changes is a text
        # holding an alias. Cheaper and more exact than a second `dev-\d+` regex,
        # and it cannot drift from the alias set the registry actually compiled.
        return self._registry.to_names(text) != text

    def classify(self, text: str, document_text: str) -> str | None:
        """The bucket for one reference, or None when it is dropped."""
        candidate = " ".join((text or "").split())
        if not candidate:
            return None
        if candidate.lower() not in document_text.lower():
            return None
        if self._contains_alias(candidate):
            return ALIAS
        lowered = candidate.lower()
        if lowered in self._exempt or lowered in self._allowed:
            return None
        if _SINGLE_TOKEN_RE.match(candidate) and lowered in self._given_names:
            return MAPPED_RESIDUAL
        return UNKNOWN_PERSON


# --- cache -------------------------------------------------------------------

CACHE_POLICY_KEY = "policy_version"
CACHE_MODEL_KEY = "model"
CACHE_ENTRIES_KEY = "entries"


def cache_key(text: str, model: str) -> str:
    """``sha256(text + policy version + model)``.

    All three, because all three change the answer: the text obviously, the
    policy version because a changed substitution rule changes what is left in
    the clear, and the model because a different reviewer is a different opinion.
    A composite digest rather than three stored fields so one comparison decides.
    """
    digest = hashlib.sha256()
    digest.update((text or "").encode("utf-8"))
    digest.update(f"\x00policy={POLICY_VERSION}\x00model={model}".encode("utf-8"))
    return digest.hexdigest()


def cache_path_for(collection: str, cache_dir=None) -> Path:
    return Path(cache_dir or DEFAULT_CACHE_DIR) / f"{collection}.json"


def load_cache(path: Path, model: str) -> dict:
    """Cached per-document verdicts, or {} when the envelope no longer applies.

    Envelope, not a flat dict keyed by doc id, for the reason the graph
    extractor's cache carries one: a doc id may itself start with ``_``, so a
    sibling metadata key is not safely distinguishable from an entry. A cache
    written under another policy version or another model is discarded wholesale
    — the per-entry key would reject every entry one at a time anyway, and saying
    so once is clearer in a log.
    """
    if not Path(path).exists():
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or not isinstance(raw.get(CACHE_ENTRIES_KEY), dict):
        return {}
    if raw.get(CACHE_POLICY_KEY) != POLICY_VERSION or raw.get(CACHE_MODEL_KEY) != model:
        return {}
    return {k: v for k, v in raw[CACHE_ENTRIES_KEY].items() if isinstance(v, dict)}


def write_cache(path: Path, model: str, entries: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({CACHE_POLICY_KEY: POLICY_VERSION,
                                CACHE_MODEL_KEY: model,
                                CACHE_ENTRIES_KEY: entries},
                               ensure_ascii=False, indent=2), encoding="utf-8")


# --- report discovery and the packaging gate ---------------------------------

REPORT_PREFIX = "sweep_"


def report_dirs() -> list:
    """Where sweep reports may live, most private first.

    A private sub-repo's ``privacy/`` directory when there is exactly one (the
    same discovery convention as the alias map and the bigram allow-list), then
    ``data/privacy/``. Both are gitignored; the report is the one output of this
    script that contains real names.
    """
    found = [Path(p) for p in _private_glob(PRIVATE_REPORT_GLOB) if os.path.isdir(p)]
    return [*found, Path(DEFAULT_CANDIDATES_DIR)]


def report_path(collection: str, out_dir=None, today=None) -> Path:
    directory = Path(out_dir) if out_dir else report_dirs()[0]
    stamp = today or datetime.now(timezone.utc).date().isoformat()
    return directory / f"{REPORT_PREFIX}{collection}_{stamp}.json"


def _read_report(path: Path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def latest_report(collection: str, dirs=None):
    """The newest sweep report for a collection, by its own ``generatedAt``.

    By the stamp inside the file rather than the filename's date or the file
    mtime: a report copied between machines keeps its content and loses both of
    the others, and the packaging gate compares this value against the
    collection's ``updatedTime``.
    """
    best = None
    for directory in (dirs if dirs is not None else report_dirs()):
        for path in sorted(Path(directory).glob(f"{REPORT_PREFIX}{collection}_*.json")):
            payload = _read_report(path)
            if not payload or payload.get("collection") != collection:
                continue
            if best is None or (payload.get("generatedAt") or "") > (best[1].get("generatedAt") or ""):
                best = (path, payload)
    return best


def sweep_gate(collection: str, manifest: dict, dirs=None):
    """``(status, message)`` for the packager: ``pass`` / ``warn`` / ``refuse``.

    The sweep is a SECOND opinion, so a collection that has never been swept is a
    warning, not a refusal — the deterministic gate is still the thing that
    certifies, and making a hand-off depend on a local GPU being up would be a
    gate people route around. But a sweep that DID run and found an unknown
    person is a refusal, and so is one that ran before the collection was last
    rebuilt: a clean verdict about text that has since been replaced certifies
    nothing.
    """
    found = latest_report(collection, dirs)
    if found is None:
        return "warn", (f"no local sensitivity sweep report for {collection} — the deterministic "
                        f"gate passed, but nothing has read the text for an unmapped person. "
                        f"Run scripts/audit/sensitivity_sweep.py --collection {collection} "
                        f"--baseline")
    path, payload = found
    unknown = payload.get("unknownCount")
    if not isinstance(unknown, int):
        return "refuse", f"{path.name} carries no unknownCount — it is not a readable sweep report"
    if unknown > 0:
        return "refuse", (f"the local sensitivity sweep ({path.name}) found {unknown} unknown "
                          f"person reference(s). Triage them before shipping — the report has "
                          f"the strings, this message deliberately does not")
    generated, updated = payload.get("generatedAt") or "", manifest.get("updatedTime") or ""
    if updated and generated < _normalise_stamp(updated):
        return "refuse", (f"the local sensitivity sweep ({path.name}) is older than the "
                          f"collection's last rebuild — it judged text that is no longer there")
    return "pass", f"local sensitivity sweep {path.name}: clean"


def _normalise_stamp(value: str) -> str:
    """A manifest ``updatedTime`` as the same fixed-width UTC form the report uses."""
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the sweep ---------------------------------------------------------------

def _windows(text: str):
    if len(text) <= WINDOW_CHARS:
        return [text]
    step = WINDOW_CHARS - WINDOW_OVERLAP
    return [text[start:start + WINDOW_CHARS] for start in range(0, len(text), step)]


def _title_of(document: dict) -> str:
    metadata = document.get("metadata") or {}
    title = metadata.get("title") if isinstance(metadata, dict) else None
    return title or (document.get("id") or "").rsplit("/", 1)[-1]


def ollama_reachable(url: str = OLLAMA_TAGS_URL, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def sweep_document(document: dict, classifier: ReferenceClassifier, *, model: str,
                   timeout: int, call=call_ollama):
    """``(findings, parse_failures)`` for one document. Never raises on the model.

    ``call`` is injected so the tests drive the whole pipeline — windowing,
    parsing, classification, dedupe — without an Ollama on the machine.
    """
    text = document.get("text") or ""
    title = _title_of(document)
    findings, seen, failures = [], set(), 0
    for window in _windows(text):
        prompt = USER_PROMPT_TEMPLATE.format(title=title, text=window)
        try:
            raw = call(prompt, model=model, timeout=timeout, temperature=0,
                       system=SYSTEM_PROMPT, options={"num_predict": 1200})
        except RuntimeError:
            failures += 1
            continue
        references = parse_references(raw)
        if not answered_in_json(raw):
            # An unparseable answer is not a clean document. Counted so a run
            # that was mostly the model failing to answer cannot report zero
            # findings and read as a pass.
            failures += 1
        for reference in references:
            bucket = classifier.classify(reference["text"], text)
            if bucket is None:
                continue
            key = (reference["text"].lower(), bucket)
            if key in seen:
                continue
            seen.add(key)
            findings.append({"text": reference["text"], "kind": reference["kind"],
                             "classification": bucket})
    return findings, failures


def run_sweep(collection_dir: Path, classifier: ReferenceClassifier, *, model: str,
              baseline: bool, limit: int = 0, cache: dict | None = None,
              timeout: int = 300, call=call_ollama, progress=None):
    """Sweep a built collection. Returns a result dict the report is written from.

    In incremental mode a document whose composite hash is already in the cache is
    NOT re-asked — but its cached ``unknown_count`` still lands in the run total.
    A nightly incremental that skipped the one dirty document and then reported
    zero unknowns would flip the packaging gate from refuse to pass without
    anything having been fixed.
    """
    cache = {} if cache is None else cache
    documents = list(documents_of(Path(collection_dir)))
    documents.sort(key=lambda d: str(d.get("id") or ""))
    if limit and limit > 0:
        documents = documents[:limit]

    findings, counts = [], Counter()
    asked = cached = parse_failures = windows = 0
    cached_unknown = 0
    started = time.time()
    new_entries = dict(cache)

    for index, document in enumerate(documents, 1):
        text = document.get("text") or ""
        doc_id = str(document.get("id") or f"#{index}")
        key = cache_key(text, model)
        entry = cache.get(doc_id)
        if not baseline and entry and entry.get("hash") == key:
            cached += 1
            cached_unknown += int(entry.get("unknown_count") or 0)
            continue

        doc_findings, failures = sweep_document(
            document, classifier, model=model, timeout=timeout, call=call)
        asked += 1
        windows += len(_windows(text))
        parse_failures += failures
        for finding in doc_findings:
            counts[finding["classification"]] += 1
            findings.append({"documentId": doc_id, **finding})
        new_entries[doc_id] = {
            "hash": key,
            "findings_count": len(doc_findings),
            "unknown_count": sum(1 for f in doc_findings
                                 if f["classification"] == UNKNOWN_PERSON),
        }
        if progress:
            progress(index, len(documents), doc_id, doc_findings)

    elapsed = time.time() - started
    return {
        "documents": len(documents),
        "documentsAsked": asked,
        "documentsCached": cached,
        "windows": windows,
        "parseFailures": parse_failures,
        "counts": {ALIAS: counts[ALIAS], MAPPED_RESIDUAL: counts[MAPPED_RESIDUAL],
                   UNKNOWN_PERSON: counts[UNKNOWN_PERSON]},
        "unknownCount": counts[UNKNOWN_PERSON] + cached_unknown,
        "unknownCountCached": cached_unknown,
        "findings": findings,
        "elapsedSeconds": round(elapsed, 1),
        "entries": new_entries,
    }


# --- ledger ------------------------------------------------------------------

def ledger_record(status: str, *, started_at: str, finished_at: str, detail: dict,
                  job=None, trigger="cli", error=None, duration=None) -> dict:
    """One self-contained closing record for the sweep.

    No opening ``stage: "begin"`` partial: this is a single foreground process
    that either writes its record or died, and an unmatched opener would fold to
    `incomplete` forever on the (common) path where someone Ctrl-Cs a manual
    sweep. The phase is marked ``fatal`` so an unreachable model rolls the run up
    to ``failed`` rather than to ``degraded`` — a sweep that judged nothing must
    not read like a sweep that found something.
    """
    return {
        "runId": mint_run_id(LEDGER_COLLECTION, started_at),
        "collection": LEDGER_COLLECTION,
        "job": job,
        "trigger": trigger,
        "variant": "incremental",
        "source": "script",
        "startedAt": started_at,
        "finishedAt": finished_at,
        "error": error,
        "phases": [{"name": LEDGER_PHASE, "status": status, "fatal": True,
                    "startedAt": started_at,
                    "durationSeconds": duration if duration is not None else 0,
                    "detail": detail}],
    }


def report_run(record: dict, api_url: str) -> str:
    """POST the record, falling back to the ledger module. Returns what happened.

    The same two-step ``scripts/lib/indexing_run.sh`` uses, for the same reason:
    the API is routinely down when an unattended job runs, and the ledger file
    must never be written by anything that cannot take the flock. Here the
    fallback is an in-process ``IndexingRunLedger().append`` rather than a
    subprocess — same writer, same lock, one less shell.
    """
    payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/indexing/runs", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if 200 <= response.status < 300:
                return "api"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass
    try:
        IndexingRunLedger().append(record)
        return "ledger"
    except Exception as e:                                   # noqa: BLE001
        print(f"WARN: could not record the sweep run: {e}", file=sys.stderr)
        return "lost"


# --- CLI ---------------------------------------------------------------------

def _summary_lines(result: dict) -> list:
    """Counts and SHAPES only. This is the half that gets pasted somewhere."""
    counts = result["counts"]
    lines = [
        f"documents: {result['documents']} "
        f"({result['documentsAsked']} asked, {result['documentsCached']} cached), "
        f"{result['windows']} model calls in {result['elapsedSeconds']}s",
        f"alias/redaction: {counts[ALIAS]}   mapped residual: {counts[MAPPED_RESIDUAL]}   "
        f"unknown persons: {result['unknownCount']} "
        f"({result['unknownCountCached']} from cached documents)",
        f"unparseable model answers: {result['parseFailures']}",
    ]
    unknown = Counter(sanitize(f["text"]) for f in result["findings"]
                      if f["classification"] == UNKNOWN_PERSON)
    if unknown:
        lines.append(f"unknown shapes: {[s for s, _ in unknown.most_common(8)]}")
    residual = Counter(sanitize(f["text"]) for f in result["findings"]
                       if f["classification"] == MAPPED_RESIDUAL)
    if residual:
        lines.append(f"residual shapes: {[s for s, _ in residual.most_common(5)]}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--collection", required=True)
    ap.add_argument("--collections-dir", default=str(DEFAULT_COLLECTIONS_DIR))
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Local Ollama model (default: {DEFAULT_MODEL}). There is no hosted "
                         f"backend here and there must not be one.")
    ap.add_argument("--baseline", action="store_true",
                    help="Ask about every document, ignoring the cache. The default mode asks "
                         "only about documents whose text hash changed.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Bound a --baseline run to the first N documents (0 = all)")
    ap.add_argument("--map", default=None, help="Alias map path (default: the discovered one)")
    ap.add_argument("--allowed-bigrams", default=None,
                    help="Reviewed non-person bigram allow-list (default: the discovered one)")
    ap.add_argument("--report-out", default=None,
                    help="Where the report goes. THIS FILE CONTAINS REAL TEXT; an explicit "
                         "path is refused unless git already ignores it.")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-cache", action="store_true", help="Neither read nor write the cache")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8321"))
    ap.add_argument("--job", default=None, help="launchd job label, for the run record")
    ap.add_argument("--trigger", default="cli", help="scheduled/manual/cli")
    ap.add_argument("--no-ledger", action="store_true", help="Do not record a run")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    collection_dir = Path(args.collections_dir).resolve() / args.collection
    if not (collection_dir / "manifest.json").exists():
        print(f"REFUSED: no such collection: {collection_dir}", file=sys.stderr)
        return 1

    manifest = json.loads((collection_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        map_path = discover_map_path(args.map)
        registry = AliasRegistry.load(map_path)
    except (PrivacyMapMissing, OSError, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    alias_map = json.loads(Path(map_path).read_text(encoding="utf-8"))
    classifier = ReferenceClassifier(
        registry, alias_map, load_allowed_bigrams(resolve_allowlist(args.allowed_bigrams)))

    destination = (refuse_if_tracked(Path(args.report_out)) if args.report_out
                   else report_path(args.collection))
    destination.parent.mkdir(parents=True, exist_ok=True)

    cache_file = cache_path_for(args.collection, args.cache_dir)
    cache = {} if args.no_cache else load_cache(cache_file, args.model)

    started_at = now_iso()
    started = time.time()
    if not ollama_reachable():
        message = "Ollama is not reachable at localhost:11434 — nothing was judged"
        print(f"FAILED: {message}", file=sys.stderr)
        if not args.no_ledger:
            report_run(ledger_record("failed", started_at=started_at, finished_at=now_iso(),
                                     detail={"collection": args.collection,
                                             "model": args.model},
                                     job=args.job, trigger=args.trigger, error=message),
                       args.api_url)
        return 1

    def progress(index, total, doc_id, findings):
        unknown = sum(1 for f in findings if f["classification"] == UNKNOWN_PERSON)
        print(f"  [{index}/{total}] {len(findings)} refs, {unknown} unknown", flush=True)

    # `call=call_ollama` explicitly rather than relying on the default argument:
    # a default is bound at def time, so a test monkeypatching the module symbol
    # would drive the real transport. Passing it here resolves the global on
    # every call, which is what makes the CLI path testable at all.
    result = run_sweep(collection_dir, classifier, model=args.model,
                       baseline=args.baseline, limit=args.limit, cache=cache,
                       timeout=args.timeout, call=call_ollama, progress=progress)

    if not args.no_cache:
        write_cache(cache_file, args.model, result.pop("entries"))
    else:
        result.pop("entries", None)

    generated_at = now_iso()
    report = {
        "collection": args.collection,
        "generatedAt": generated_at,
        "model": args.model,
        "policyVersion": POLICY_VERSION,
        "mapVersion": registry.map_version,
        "mode": "baseline" if args.baseline else "incremental",
        "limit": args.limit or None,
        "collectionUpdatedTime": manifest.get("updatedTime"),
        **{k: v for k, v in result.items() if k != "findings"},
        "findings": result["findings"],
    }
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ncollection: {args.collection}  model: {args.model}  "
          f"policy v{POLICY_VERSION}  map v{registry.map_version}\n")
    for line in _summary_lines(result):
        print(line)
    print(f"\nreport: {destination}")

    unknown = result["unknownCount"]
    status = "degraded" if unknown else "succeeded"
    if not args.no_ledger:
        where = report_run(
            ledger_record(status, started_at=started_at, finished_at=generated_at,
                          duration=int(time.time() - started),
                          detail={"collection": args.collection, "model": args.model,
                                  "mode": report["mode"],
                                  "documents": result["documents"],
                                  "documentsAsked": result["documentsAsked"],
                                  "unknown": unknown,
                                  "mappedResidual": result["counts"][MAPPED_RESIDUAL],
                                  "parseFailures": result["parseFailures"]},
                          job=args.job, trigger=args.trigger),
            args.api_url)
        print(f"run recorded under {LEDGER_COLLECTION} via {where}")

    if unknown:
        print(f"\nRESULT: {unknown} unknown person reference(s) — triage the report before "
              f"packaging this collection.")
        return 2
    print("\nRESULT: clean — the local model found no person reference the map does not "
          "already account for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
