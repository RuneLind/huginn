"""A LOCAL second opinion on a built collection: ask a local model who is named.

``main/privacy/index_scan.py`` is the deterministic gate and stays the thing that
blocks a hand-off. Two questions it cannot answer:

* a person the alias map has never heard of, spelled in any form other than the
  capitalised-bigram shape check 9 retains (a mononym, a surname on its own, an
  initials-plus-surname byline, a role phrase that identifies exactly one
  individual);
* whether a bare given name standing in prose is a person reference at all — the
  residual the campaign decided not to substitute is five figures of
  occurrences, and a regex cannot read them.

So the sweep reads the collection's DERIVED documents (``documents/*.json`` —
the aliased text, never ``data/sources/``) and asks a model to list the person
references it can see. Everything the model says is then filtered
deterministically: aliases and redaction tokens are dropped, the map's
``non_person_labels`` and the private reviewed bigram allow-list are dropped, and
a reference the document does not contain is dropped as a hallucination. What
survives is bucketed ``alias`` / ``mapped_residual`` / ``role`` /
``unknown_person``.

**LOCAL BY REQUIREMENT.** The transport is ``main/utils/ollama_cli.py`` and
nothing else. A sensitivity check that ships document text to a hosted model IS
the leak it is looking for, so there is no ``--backend`` flag and there never
should be.

This module is the library half — classification, windowing, cache, report
discovery, the packaging gate, the ledger record.
``scripts/audit/sensitivity_sweep.py`` is the CLI over it, and
``scripts/audit/package_collection.py`` imports :func:`sweep_gate` from here.
They were one file, which meant the packager pulled a ``scripts/`` module into a
``main/`` import graph and the tests had to reason about which of two module
objects a monkeypatch reached.
"""
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from main.privacy import DEFAULT_PRIVACY_DIR
from main.privacy.alias_registry import (
    HANDLE_TOKEN, IDENT_TOKEN, POLICY_VERSION, USER_PATH_TOKEN,
    AliasRegistry, _private_glob, shape as sanitize,
)
from main.privacy.cache_envelope import load_envelope, write_envelope
from main.privacy.index_scan import documents_of
from main.runtime.indexing_run_ledger import IndexingRunLedger
from main.utils.ollama_cli import DEFAULT_MODEL, call_ollama

# The ledger collection key every sweep run is recorded under. Not the swept
# collection's own key: a sweep is not an indexing run of it, and folding the two
# together would put a `degraded` sweep on the row a reader uses to answer "is the
# index fresh". `_COLLECTION_RE` accepts it, and `/api/indexing/jobs` shows it with
# `loaded: false` by design — no server serves a collection by this name.
LEDGER_COLLECTION = "sensitivity-audit"
LEDGER_PHASE = "sweep"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "state" / "sensitivity"
PRIVATE_REPORT_GLOB = "huginn-*/privacy"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_TAGS_TIMEOUT = 5

# Prompt window. Most documents are far shorter than this (nav-wiki's median is
# ~240 characters), but the long tail runs to 30 KB and truncating it would make
# the sweep quietly blind to the second half of exactly the pages most likely to
# name someone. Windows overlap so a name straddling a boundary is still seen
# whole in one of them; duplicates are collapsed per document.
WINDOW_CHARS = 6000
WINDOW_OVERLAP = 200

# The share of model calls that may go unread before the run's verdict stops
# meaning anything. `unknownCount: 0` from a run where the model answered
# nothing readable is indistinguishable, in the report, from a genuinely clean
# collection — and it is the shape a vacuous pass takes here, exactly as an
# empty needle list is in `index_scan`. So a run past this ratio is `degraded`
# in the ledger and the packaging gate declines to read it as clean.
# 0.2 sits well above the measured rate on the real corpus (5 of 117 calls,
# 4.3%, on the nav-wiki --limit 50 baseline) and well below a run that has
# stopped working.
MAX_PARSE_FAILURE_RATIO = 0.2

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

The text is mostly Norwegian. Names carry æ, ø and å, and a name is often
written surname-first in a byline ("Etternavn, Fornavn") — list that byline as
one reference, spelled exactly as it stands, comma included.

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

_FENCE_OPEN_RE = re.compile(r"^```[a-z]*\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```$")


def _strip_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", text))
    return text


def parse_references(raw: str):
    """The references in one model answer, or ``None`` when it is unreadable.

    THE DISTINCTION THIS FUNCTION EXISTS FOR: ``[]`` means the model said "no one
    is named here", ``None`` means it said something this script cannot read.
    Those are not the same document, and collapsing them is how a run where the
    model answered nothing usable reports ``unknownCount: 0`` and reads as clean.
    One function rather than a parse plus a separate readability probe, because
    two parses of the same string are two chances to disagree about it.

    Readable means a references LIST was recovered — a top-level list, or an
    object with a list under ``references``. Schema drift (``{"people": …}``, a
    scalar, ``null``, an object where the list should be) is a parse failure
    rather than a silent zero: the model was asked for one shape, and answering
    in another is exactly the failure mode the counter is watching for.
    """
    if not raw:
        return None
    try:
        payload = json.loads(_strip_fences(raw))
    except ValueError:
        return None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("references"), list):
        items = payload["references"]
    else:
        return None

    references = []
    for item in items:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        value = item.get("text") or item.get("name") or item.get("reference")
        if not isinstance(value, str) or not value.strip():
            continue
        kind = item.get("kind") if item.get("kind") in KINDS else "other"
        references.append({"text": normalise_whitespace(value), "kind": kind})
    return references


def normalise_whitespace(text: str) -> str:
    """Collapse every run of whitespace to one space.

    Applied to BOTH sides of the containment check. A name the document wraps
    across a line (`Ola\\nNordmann`) is quoted back by the model on one line, and
    a raw `in` test then calls the real finding a hallucination and drops it —
    the sweep failing silently in the one direction nobody would notice.
    """
    return " ".join((text or "").split())


# --- deterministic classification --------------------------------------------

ALIAS = "alias"
MAPPED_RESIDUAL = "mapped_residual"
ROLE = "role"
UNKNOWN_PERSON = "unknown_person"

# Internal-only verdicts for one part of a compound. They never reach a report:
# EXEMPT means "an adjudicated non-person", DROP means "this whole candidate is
# adjudicated non-person and is not counted at all".
EXEMPT = "__exempt__"
DROP = "__drop__"

BUCKETS = (ALIAS, MAPPED_RESIDUAL, ROLE, UNKNOWN_PERSON)

# Every literal the substituter can leave behind. `EMAIL_LOCAL_TOKEN`
# ("person") is deliberately not here: matched with a word boundary it is a
# real Norwegian word, and matched without one it swallows "personnummer" and
# every compound built on it.
_REDACTION_TOKENS = (IDENT_TOKEN, HANDLE_TOKEN, USER_PATH_TOKEN, "<user>")

# A finding this short is never a person reference and always a fragment of one
# — "Bo", "Li", initials the model split. Reported, they are the noise that
# makes a triage list unreadable.
MIN_FINDING_CHARS = 3

_SINGLE_TOKEN_RE = re.compile(r"^[^\W\d_]+(?:['’\-][^\W\d_]+)*$")

# Separators a corpus uses to name several people at once. `og` (Norwegian "and")
# is word-bounded so it cannot eat the `og` inside a name; `&` and `+` appear in
# the same attribution columns. The strip set is the punctuation that clings to a
# part once the run is split — including the `?` of an uncertain attribution,
# which is exactly the shape a Confluence table produces.
#
# NOT whitespace: splitting on a space would make any two-token name whose parts
# are both known given names decompose into two accounted parts and vanish — the
# same failure the gazetteer's concatenation test exists to prevent. A run is
# only a run when something explicitly joins it.
_COMPOUND_SPLIT_RE = re.compile(r"\s*(?:[,/&+;|]|\bog\b)\s*", re.IGNORECASE)
_COMPOUND_STRIP = " \t?.,:;()[]{}<>\"'’"
# The separators actually PRESENT in a candidate, used to refuse the ambiguous
# two-part comma form. Kept in step with _COMPOUND_SPLIT_RE.
_COMPOUND_SEPARATORS_RE = re.compile(r"[,/&+;|]|\bog\b", re.IGNORECASE)


def _boundaried(token: str) -> str:
    """A token as a regex that cannot fire inside a longer word.

    Only where a boundary is meaningful: `@person` ends in a letter, so
    `@personalt` must not match it, while `[~person]` ends in `]` and needs
    nothing. Leading boundaries are added on the same test, so `/Users/<user>/`
    is matched wherever it appears.
    """
    if not token:
        return ""
    prefix = r"(?<![\w-])" if (token[0].isalnum() or token[0] == "_") else ""
    suffix = r"(?![\w-])" if (token[-1].isalnum() or token[-1] == "_") else ""
    return f"{prefix}{re.escape(token)}{suffix}"


def _alias_literals(alias_map: dict) -> set:
    """Every alias the map can have put into the text, retired ones included.

    Read from the map rather than from ``AliasRegistry.to_names``: that inverse
    is deliberately case-SENSITIVE (it feeds a query rewrite where case carries
    meaning), and a model quoting `Dev-06` back at us from a sentence-initial
    position must still be recognised as the substituter working.
    """
    literals = set()
    for entry in alias_map.get("entries", []):
        if isinstance(entry, dict) and isinstance(entry.get("alias"), str):
            literals.add(entry["alias"].strip())
    for retired in alias_map.get("retired_aliases", []):
        if isinstance(retired, str):
            literals.add(retired.strip())
        elif isinstance(retired, dict) and isinstance(retired.get("alias"), str):
            literals.add(retired["alias"].strip())
    return {literal for literal in literals if literal}


class ReferenceClassifier:
    """The deterministic half of the sweep. Nothing here asks the model anything.

    Order matters, and it is the order of increasing doubt:

    1. **too short, or not in the document** — dropped. The prompt demands a
       verbatim substring; a reference that is not one was invented, and an
       invented name in a privacy report is worse than a missed one because it
       is the finding a human will spend the triage budget on.
    2. **an alias or a redaction token** — bucketed ``alias``. This is the
       substituter working, reported so the report says so rather than staying
       silent about the majority of what the model sees.
    3. **a FRAGMENT of one** — dropped. The model returns "dev" out of `dev-06`
       and "person" out of `@person` often enough to matter, and neither is a
       person nobody mapped; they are the tokenizer showing through.
    4. **an exempt label or a reviewed non-person bigram** — dropped. These are
       the two lists the campaign already adjudicated by hand; re-litigating them
       every night is how a gate gets switched off.
    5. **a given name or residual key the map knows** — ``mapped_residual``. The
       residual is a deliberate decision (`Ada` would alias half the corpus), so
       it is counted, not alarmed on.
    6. **a role phrase** — ``role``. Reported, never blocking: "the case worker
       who signed" identifies someone to a reader who already knows the case,
       and it is a judgement call no gate should make unattended. It stays ahead
       of 7 and 8 so that a run or a genitive cannot silently delete the
       ``sweep_gate`` warning a role phrase raises.
    7. **a genitive of a residual name** — ``mapped_residual``. The stem must
       itself be accounted for, so this can only re-label something already
       destined for that bucket.
    8. **a RUN of already-accounted references** — ``mapped_residual``, or
       dropped when every part is an exempt label. One reference naming several
       people (`A/B`, `A, B og C`) matches nothing as a whole candidate. EVERY
       part must be accounted for, and a part is never accepted on the strength
       of being some entry's given name — see ``_part_bucket``.
    9. everything else — ``unknown_person``, which fails the sweep.
    """

    def __init__(self, registry: AliasRegistry, alias_map: dict, allowed_bigrams=frozenset()):
        self._registry = registry
        tokens = {t for t in (*_REDACTION_TOKENS, registry.redaction_token) if t}
        aliases = _alias_literals(alias_map)
        pattern = "|".join(_boundaried(t) for t in
                           sorted(tokens | aliases, key=len, reverse=True))
        self._alias_re = re.compile(pattern, re.IGNORECASE) if pattern else None
        # Lowercased, for the fragment test. A finding that is a strict
        # substring of one of these is the model quoting part of a token.
        self._token_text = {t.lower() for t in (tokens | aliases)}

        self._exempt = {label.strip().lower()
                        for label in alias_map.get("non_person_labels", [])
                        if isinstance(label, str) and label.strip()}
        self._allowed = {b.lower() for b in allowed_bigrams}
        # `registry.given_names` is the first token of every entry name and every
        # unmapped label — single tokens by construction, and only meaningful as
        # a whole finding when the finding is a single token too (`Ada Nyansatt`
        # is a full name nobody aliased).
        self._given_names = set(registry.given_names)
        # The map's `bare_given_name_residual` keys are the same population
        # counted from the corpus. They are NOT single-token by construction —
        # the counter records whatever stands in the corpus — so the single-token
        # guard must not apply to them, or a multi-word residual key is reported
        # as an unknown person every night.
        self._residual = {name.strip().lower()
                          for name in alias_map.get("bare_given_name_residual", {})
                          if isinstance(name, str) and name.strip()}

    def _is_alias(self, text: str) -> bool:
        # Whole-candidate match: a span that merely CONTAINS an alias
        # (`Ola Nordmann (dev-06)`) is not the substituter working.
        return bool(self._alias_re and self._alias_re.fullmatch(text.strip()))

    def _is_token_fragment(self, lowered: str) -> bool:
        return any(lowered != token and lowered in token for token in self._token_text)

    def _part_bucket(self, part: str):
        """What one part of a compound is, or None when nothing accounts for it.

        Returns ``EXEMPT`` (an adjudicated non-person), ``MAPPED_RESIDUAL`` (a
        known person reference), or None.

        `self._given_names` is deliberately NOT consulted. A comma is both a list
        separator and a NAME-INVERSION separator: `Surname, Given` is one person,
        and in the live map 105 of 291 surname tokens are also somebody's given
        name, so accepting a given-name-only match would let an unknown person
        written in inverted form decompose into two "known" halves and stop
        blocking. The residual register is the safe channel — it is the corpus-
        attested bare given names of people the campaign already adjudicated.

        The alias test runs on the RAW piece, before stripping: the redaction
        tokens are bracketed (`[~ukjent-person]`) and the strip set eats brackets,
        so stripping first destroyed exactly the token this needs to recognise.
        """
        if self._is_alias(part):
            return ALIAS
        lowered = part.strip(_COMPOUND_STRIP).lower()
        if not lowered:
            return None
        if lowered in self._exempt or lowered in self._allowed:
            return EXEMPT
        if lowered in self._residual or self._is_genitive(lowered):
            return MAPPED_RESIDUAL
        return None

    def _is_genitive(self, lowered: str) -> bool:
        """`Adas` for a residual `Ada`. Norwegian genitive, no apostrophe.

        Bounded deliberately: the stem must itself be a name the map already
        accounts for, so this can only ever re-label something that was going to
        be `mapped_residual` under its own name. It cannot absorb a new person.
        """
        if len(lowered) < MIN_FINDING_CHARS + 1 or not lowered.endswith("s"):
            return False
        stem = lowered[:-1]
        # RESIDUAL ONLY, never `given_names`. Norwegian surnames ending in -s are
        # common, so accepting `<any entry's given name> + s` would re-open the
        # inverted-name hole one character at a time: `Berg, Ada` blocks while
        # `Bergs, Ada` would not. The residual register is corpus-attested bare
        # given names of people already adjudicated; a given name alone is not.
        return stem in self._residual

    def _compound_verdict(self, candidate: str):
        """`Ada/Bea`, `Ada, Bea og Cec` — a RUN of already-known names.

        Returns ``MAPPED_RESIDUAL``, ``DROP`` (every part is an exempt label), or
        None when any part is unaccounted for.

        Confluence attribution columns and Jira comments name several people at
        once, joined by a slash, a comma or `og`. The model quotes the whole run
        as one reference, and every other test here is a whole-candidate lookup,
        so a run matches nothing and lands in `unknown_person` — 12 of the 44
        strings in the 2026-08-23 triage, none of them a person the map had not
        already adjudicated.

        EVERY part must be accounted for. A run containing one unmapped name
        still fails, which is the property that makes this safe: it narrows what
        counts as news, it does not stop the sweep noticing news.
        """
        parts = [piece.strip() for piece in _COMPOUND_SPLIT_RE.split(candidate)]
        parts = [p for p in parts if p.strip(_COMPOUND_STRIP)]
        if len(parts) < 2:
            return None
        # `Surname, Given` is ONE person written inverted, and a bare comma is the
        # only separator a corpus uses for BOTH that and a list. Two parts joined
        # by nothing but a comma are therefore refused outright — no real
        # attribution run in the 2026-08-23 triage had that shape (they used a
        # slash, `og`, or three or more parts), so this costs nothing and closes
        # the inversion case instead of merely narrowing it.
        if len(parts) == 2 and set(_COMPOUND_SEPARATORS_RE.findall(candidate)) <= {","}:
            return None
        buckets = [self._part_bucket(part) for part in parts]
        if any(bucket is None for bucket in buckets):
            return None
        # A run of nothing but adjudicated non-persons is DROPPED, not counted:
        # the whole-candidate path drops a lone exempt label, and counting the
        # pair as a residual would inflate the very number triage reads.
        if all(bucket == EXEMPT for bucket in buckets):
            return DROP
        # A run of nothing but aliases and redaction tokens IS the substituter
        # working, and belongs in the bucket that says so rather than inflating
        # the residual count triage reads.
        if all(bucket == ALIAS for bucket in buckets):
            return ALIAS
        return MAPPED_RESIDUAL

    def classify(self, text: str, document_text: str, kind=None) -> str | None:
        """The bucket for one reference, or None when it is dropped."""
        candidate = normalise_whitespace(text)
        if len(candidate) < MIN_FINDING_CHARS:
            return None
        if candidate.lower() not in normalise_whitespace(document_text).lower():
            return None
        if self._is_alias(candidate):
            return ALIAS
        lowered = candidate.lower()
        if self._is_token_fragment(lowered):
            return None
        # Edge punctuation is never part of a name, and a corpus attaches plenty
        # of it — `Frid?` is a Confluence cell marking an uncertain attribution.
        # Testing the stripped form as well keeps those out of `unknown_person`
        # without loosening any of the lookups themselves.
        bare = lowered.strip(_COMPOUND_STRIP)
        if lowered in self._exempt or lowered in self._allowed:
            return None
        if bare in self._exempt or bare in self._allowed:
            return None
        if lowered in self._residual or bare in self._residual:
            return MAPPED_RESIDUAL
        if _SINGLE_TOKEN_RE.match(candidate.strip(_COMPOUND_STRIP)) and bare in self._given_names:
            return MAPPED_RESIDUAL
        # `other` — the coercion for a kind the model invented — is treated as a
        # missing kind, i.e. as a person. Only an explicit `role` downgrades.
        #
        # This stays AHEAD of the compound and genitive rules on purpose: a role
        # phrase the model explicitly flagged drives a `sweep_gate` warning, and
        # letting a compound re-bucket it as `mapped_residual` would delete that
        # warning silently.
        if kind == "role":
            return ROLE
        if self._is_genitive(bare):
            return MAPPED_RESIDUAL
        compound = self._compound_verdict(candidate)
        if compound == DROP:
            return None
        if compound is not None:
            return compound
        return UNKNOWN_PERSON


# --- cache -------------------------------------------------------------------

CACHE_POLICY_KEY = "policy_version"
CACHE_MODEL_KEY = "model"
CACHE_MAP_KEY = "map_version"
CACHE_ALLOWLIST_KEY = "allowlist_sha256"


def allowlist_sha256(path) -> str | None:
    """The sha of the reviewed bigram allow-list, or None when there is none.

    Part of the cache identity because the allow-list is the one input that can
    turn an `unknown_person` into a drop without the document or the map moving.
    A cache written against a longer allow-list was written against a weaker
    filter — the same argument `PACKAGE-STAMP.json` hashes it for.
    """
    if not path or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_key(text: str, model: str, *, map_version=None, allowlist_sha=None) -> str:
    """``sha256`` over everything that can change the verdict for one document.

    The text obviously; the policy version because a changed substitution rule
    changes what is left in the clear; the model because a different reviewer is
    a different opinion; the map version and the allow-list sha because both
    decide which findings survive the filter. A composite digest rather than
    five stored fields, so one comparison decides.
    """
    digest = hashlib.sha256()
    digest.update((text or "").encode("utf-8"))
    digest.update(f"\x00policy={POLICY_VERSION}\x00model={model}"
                  f"\x00map={map_version}\x00allow={allowlist_sha}".encode("utf-8"))
    return digest.hexdigest()


def cache_path_for(collection: str, cache_dir=None) -> Path:
    return Path(cache_dir or DEFAULT_CACHE_DIR) / f"{collection}.json"


def cache_metadata(model: str, map_version=None, allowlist_sha=None) -> dict:
    return {CACHE_POLICY_KEY: POLICY_VERSION, CACHE_MODEL_KEY: model,
            CACHE_MAP_KEY: map_version, CACHE_ALLOWLIST_KEY: allowlist_sha}


def load_cache(path, model: str, map_version=None, allowlist_sha=None) -> dict:
    """Cached per-document verdicts, or {} when the envelope no longer applies.

    A cache written under another policy version, model, map version or
    allow-list is discarded wholesale — the per-entry key would reject every
    entry one at a time anyway, and saying so once is clearer in a log.
    """
    metadata, entries = load_envelope(path)
    if metadata != cache_metadata(model, map_version, allowlist_sha):
        return {}
    return {k: v for k, v in entries.items() if isinstance(v, dict)}


def write_cache(path, model: str, entries: dict, map_version=None,
                allowlist_sha=None) -> None:
    write_envelope(path, cache_metadata(model, map_version, allowlist_sha), entries)


# --- report discovery and the packaging gate ---------------------------------

REPORT_PREFIX = "sweep_"


def report_dirs() -> list:
    """Where sweep reports may live, most private first.

    A private sub-repo's ``privacy/`` directory when there is one (the same
    discovery convention as the alias map and the bigram allow-list), then
    ``data/privacy/``. Both are gitignored; the report is the one output of this
    tool that contains real names.
    """
    found = [Path(p) for p in _private_glob(PRIVATE_REPORT_GLOB) if Path(p).is_dir()]
    return [*found, Path(DEFAULT_PRIVACY_DIR)]


def report_path(collection: str, *, mode: str, limit: int = 0) -> Path:
    """``sweep_<collection>_<date>_<mode>[-limitN].json`` in the most private dir.

    The mode and the limit are IN THE NAME because a limited sample must never
    land on top of a full baseline: two runs on one day used to write the same
    filename, so a `--limit 50` spot check silently replaced the evidence the
    packaging gate reads. The gate distinguishes them by content too
    (:func:`is_full_report`); the filename is what stops the overwrite.
    """
    # Date AND time: a same-day re-run must not overwrite a report that refused.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    suffix = f"-limit{limit}" if limit and limit > 0 else ""
    return report_dirs()[0] / f"{REPORT_PREFIX}{collection}_{stamp}_{mode}{suffix}.json"


def answers_are_readable(windows, parse_failures) -> bool:
    """Did enough of the run come back readable for its verdict to mean anything?

    A run that asked nothing at all (every document cached) is vacuously fine —
    the cached verdicts are what carry it. A run that asked and could not read
    the answers is not.
    """
    if not isinstance(windows, int) or not isinstance(parse_failures, int) or windows <= 0:
        return True
    return parse_failures <= windows * MAX_PARSE_FAILURE_RATIO


def is_full_report(payload: dict) -> bool:
    """Did this report read the WHOLE collection?

    Requires both halves: no ``--limit``, and a document count that matches the
    manifest's ``numberOfDocuments`` as the run recorded it. A report that
    carries neither field is a pre-coverage report and cannot prove it — treated
    as partial, which downgrades a hand-off to a warning rather than letting an
    unprovable clean verdict certify one.
    """
    if payload.get("limit"):
        return False
    documents, expected = payload.get("documents"), payload.get("documentsExpected")
    return isinstance(documents, int) and documents == expected


def _read_report(path: Path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _reports(collection: str, dirs=None):
    """Every readable report for a collection, oldest first by ``generatedAt``.

    By the stamp inside the file rather than the filename's date or the file
    mtime: a report copied between machines keeps its content and loses both.
    """
    found = []
    for directory in (dirs if dirs is not None else report_dirs()):
        for path in sorted(Path(directory).glob(f"{REPORT_PREFIX}{collection}_*.json")):
            payload = _read_report(path)
            if payload and payload.get("collection") == collection:
                found.append((path, payload))
    return sorted(found, key=lambda pair: pair[1].get("generatedAt") or "")


def latest_report(collection: str, dirs=None):
    """The newest FULL report, else the newest report of any coverage, else None.

    Returns ``(path, payload, full)``. A limited run is deliberately not allowed
    to supersede a full baseline: it is a spot check, and reading one as the
    collection's standing verdict is how a 50-document sample came to certify a
    538-document hand-off.
    """
    found = _reports(collection, dirs)
    if not found:
        return None
    # A full report that could not read its answers is not evidence and must
    # not supersede an older readable one — least of all a refusing one.
    for path, payload in reversed(found):
        if is_full_report(payload) and answers_are_readable(
                payload.get("windows"), payload.get("parseFailures")):
            return path, payload, True
    for path, payload in reversed(found):
        if is_full_report(payload):
            return path, payload, True
    path, payload = found[-1]
    return path, payload, False


def _coverage(payload: dict) -> str:
    documents = payload.get("documents")
    expected = payload.get("documentsExpected")
    return f"{documents if documents is not None else '?'}/" \
           f"{expected if expected is not None else '?'}"


def sweep_gate(collection: str, manifest: dict, dirs=None, *, map_version=None,
               model=None, policy_version=None):
    """``(status, message)`` for the packager: ``pass`` / ``warn`` / ``refuse``.

    The sweep is a SECOND opinion, so anything short of a clean full report is a
    warning rather than a refusal — the deterministic gate is still the thing
    that certifies, and making a hand-off depend on a local GPU being up is a
    gate people route around. What DOES refuse is a report that found an unknown
    person: that is positive evidence, and it holds whatever the coverage was.

    Everything else that makes a clean verdict unreliable warns and says which:
    no report, only a limited one, too many unreadable answers, a map /
    policy / model the report was not produced under, or a collection whose
    documents have changed since. The staleness comparison is
    ``lastModifiedDocumentTime``, NOT ``updatedTime`` — the latter moves on every
    no-op reindex, exactly the reasoning the graph source-stamp uses.
    """
    found = latest_report(collection, dirs)
    if found is None:
        return "warn", (f"no local sensitivity sweep report for {collection} — the deterministic "
                        f"gate passed, but nothing has read the text for an unmapped person. "
                        f"Run scripts/audit/sensitivity_sweep.py --collection {collection} "
                        f"--baseline")
    path, payload, full = found
    coverage = _coverage(payload)

    unknown = payload.get("unknownCount")
    if not isinstance(unknown, int):
        return "refuse", (f"{path.name} carries no unknownCount — it is not a readable sweep "
                          f"report")
    if unknown > 0:
        return "refuse", (f"the local sensitivity sweep ({path.name}, {coverage} documents) "
                          f"found {unknown} unknown person reference(s). Triage them before "
                          f"shipping — the report has the strings, this message deliberately "
                          f"does not")
    if not full:
        return "warn", (f"the only local sensitivity sweep for {collection} ({path.name}) "
                        f"covered {coverage} documents. A sample cannot certify the rest — "
                        f"run it with --baseline and no --limit")
    if not answers_are_readable(payload.get("windows"), payload.get("parseFailures")):
        return "warn", (f"the local sensitivity sweep ({path.name}, {coverage} documents) could "
                        f"not read {payload.get('parseFailures')} of {payload.get('windows')} "
                        f"model answers — its clean verdict is not evidence. Re-run it")

    drifted = []
    if map_version is not None and payload.get("mapVersion") != map_version:
        drifted.append(f"map v{payload.get('mapVersion')} != v{map_version}")
    expected_policy = POLICY_VERSION if policy_version is None else policy_version
    if payload.get("policyVersion") != expected_policy:
        drifted.append(f"policy v{payload.get('policyVersion')} != v{expected_policy}")
    expected_model = DEFAULT_MODEL if model is None else model
    if payload.get("model") != expected_model:
        drifted.append(f"model {payload.get('model')} != {expected_model}")
    if drifted:
        return "warn", (f"the local sensitivity sweep ({path.name}, {coverage} documents) was "
                        f"produced under different inputs ({'; '.join(drifted)}) — its verdict "
                        f"is about a different filter than the one shipping now")

    swept = _normalise_stamp(payload.get("collectionLastModifiedDocumentTime"))
    current = _normalise_stamp(manifest.get("lastModifiedDocumentTime"))
    # UNKNOWN on either side warns even when both are unknown. Two unreadable
    # stamps are not evidence that nothing changed — they are the absence of the
    # evidence, and letting `"unknown" == "unknown"` pass would make a manifest
    # with no `lastModifiedDocumentTime` at all the easiest way to silence this.
    if swept == STAMP_UNKNOWN or current == STAMP_UNKNOWN or swept != current:
        return "warn", (f"the local sensitivity sweep ({path.name}, {coverage} documents) read "
                        f"the collection at lastModifiedDocumentTime {swept}, and it is now "
                        f"{current} — a clean verdict about text that has since changed "
                        f"certifies nothing. Re-run it")
    roles = (payload.get("counts") or {}).get(ROLE, 0)
    if isinstance(roles, int) and roles > 0:
        # The model's own `role` label is the one place it can downgrade a
        # person without blocking; a clean verdict with role phrases is a read.
        return "warn", (f"the local sensitivity sweep ({path.name}, {coverage} documents) is "
                        f"clean but the model labelled {roles} reference(s) as role phrases — "
                        f"read them before shipping")
    return "pass", f"local sensitivity sweep {path.name}: clean ({coverage} documents)"


STAMP_UNKNOWN = "unknown"


def _normalise_stamp(value) -> str:
    """A timestamp as one fixed-width UTC form, or ``STAMP_UNKNOWN``.

    Naive stamps are assumed UTC — the manifest writes
    ``lastModifiedDocumentTime`` without an offset while the report writes ``Z``,
    so without this the two forms of the same instant never compare equal and
    every collection reads as stale. Anything that is not a timestamp at all
    (missing, a sentinel, a hand-edited string) becomes ``STAMP_UNKNOWN``, which
    the caller treats as "cannot tell" — a warning — rather than as a value that
    can match.
    """
    if not isinstance(value, str) or not value.strip():
        return STAMP_UNKNOWN
    try:
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return STAMP_UNKNOWN
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the sweep ---------------------------------------------------------------

def _windows(text: str) -> list:
    """Overlapping slices of one document, or [] for an empty one.

    Two properties the naive `range(0, len, step)` did not have:

    * an EMPTY document produces no window, so the sweep does not spend an
      11-second model call asking who is named in "";
    * no trailing window shorter than the overlap. Such a window is a strict
      suffix of the one before it — every character in it was already sent —
      so it is a duplicate call, and on the real corpus it fired on roughly one
      long document in six.
    """
    if not text:
        return []
    if len(text) <= WINDOW_CHARS:
        return [text]
    step = WINDOW_CHARS - WINDOW_OVERLAP
    windows = [text[start:start + WINDOW_CHARS] for start in range(0, len(text), step)]
    if len(windows) > 1 and len(windows[-1]) < WINDOW_OVERLAP:
        windows.pop()
    return windows


def _title_of(document: dict) -> str:
    metadata = document.get("metadata") or {}
    title = metadata.get("title") if isinstance(metadata, dict) else None
    return title or (document.get("id") or "").rsplit("/", 1)[-1]


def ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=OLLAMA_TAGS_TIMEOUT):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def sweep_document(document: dict, classifier: ReferenceClassifier, *, model: str,
                   timeout: int, call=call_ollama):
    """``(findings, parse_failures, windows)`` for one document.

    Never raises on the model: a sweep that stops at the first unreachable
    window stops judging the other 500 documents. The window count is returned
    rather than recomputed by the caller — slicing the text twice is how the
    reported call count and the actual one drift apart.
    """
    text = document.get("text") or ""
    title = _title_of(document)
    windows = _windows(text)
    findings, seen, failures = [], set(), 0
    for window in windows:
        prompt = USER_PROMPT_TEMPLATE.format(title=title, text=window)
        try:
            raw = call(prompt, model=model, timeout=timeout, temperature=0,
                       system=SYSTEM_PROMPT, options={"num_predict": 1200})
        except (RuntimeError, OSError, UnicodeDecodeError):
            # Every way the transport can fail without the process being wrong:
            # a refused socket, a truncated read, a response body that is not
            # UTF-8. All three are "this window went unread", which is what the
            # failure counter means.
            failures += 1
            continue
        references = parse_references(raw)
        if references is None:
            failures += 1
            continue
        for reference in references:
            bucket = classifier.classify(reference["text"], text, kind=reference["kind"])
            if bucket is None:
                continue
            key = (reference["text"].lower(), bucket)
            if key in seen:
                continue
            seen.add(key)
            findings.append({"text": reference["text"], "kind": reference["kind"],
                             "classification": bucket})
    return findings, failures, len(windows)


def run_sweep(collection_dir, classifier: ReferenceClassifier, *, model: str,
              baseline: bool, limit: int = 0, cache: dict | None = None,
              timeout: int = 300, call=call_ollama, progress=None,
              map_version=None, allowlist_sha=None):
    """Sweep a built collection. Returns a result dict the report is written from.

    In incremental mode a document whose composite hash is already in the cache
    is NOT re-asked — but its cached findings still land in the run's totals AND
    in its findings list. A nightly incremental that skipped the one dirty
    document and then reported zero unknowns would flip the packaging gate from
    refuse to pass without anything having been fixed, and one that reported the
    count without the strings would tell a triager a name exists somewhere in
    538 documents.
    """
    cache = {} if cache is None else cache
    documents = list(documents_of(Path(collection_dir)))
    documents.sort(key=lambda d: str(d.get("id") or ""))
    # Every id the collection HAS, before --limit slices it. Pruning against the
    # sliced list would delete the cache for every document a limited run did
    # not look at, which is the whole cache on the next `--limit 5`.
    present = {str(document.get("id") or f"#{index}")
               for index, document in enumerate(documents, 1)}
    if limit and limit > 0:
        documents = documents[:limit]

    findings, counts = [], Counter()
    asked = cached = parse_failures = windows = 0
    started = time.time()
    new_entries = {doc_id: entry for doc_id, entry in cache.items() if doc_id in present}

    for index, document in enumerate(documents, 1):
        text = document.get("text") or ""
        doc_id = str(document.get("id") or f"#{index}")
        key = cache_key(text, model, map_version=map_version, allowlist_sha=allowlist_sha)
        entry = cache.get(doc_id)
        if not baseline and entry and entry.get("hash") == key:
            cached += 1
            for text_value in entry.get("unknown") or []:
                counts[UNKNOWN_PERSON] += 1
                findings.append({"documentId": doc_id, "text": text_value,
                                 "kind": "other", "classification": UNKNOWN_PERSON,
                                 "fromCache": True})
            continue

        doc_findings, failures, doc_windows = sweep_document(
            document, classifier, model=model, timeout=timeout, call=call)
        asked += 1
        windows += doc_windows
        parse_failures += failures
        for finding in doc_findings:
            counts[finding["classification"]] += 1
            findings.append({"documentId": doc_id, **finding})
        if failures:
            # NOT cached. A document with an unread window has a verdict nobody
            # produced; caching it would make the gap permanent — the hash
            # matches on every later run, so the window is never asked again.
            new_entries.pop(doc_id, None)
        else:
            unknown = [f["text"] for f in doc_findings
                       if f["classification"] == UNKNOWN_PERSON]
            new_entries[doc_id] = {"hash": key, "findings_count": len(doc_findings),
                                   "unknown_count": len(unknown), "unknown": unknown}
        if progress:
            progress(index, len(documents), doc_id, doc_findings)

    elapsed = time.time() - started
    cached_unknown = sum(1 for f in findings if f.get("fromCache"))
    return {
        "documents": len(documents),
        "documentsAsked": asked,
        "documentsCached": cached,
        "windows": windows,
        "parseFailures": parse_failures,
        "counts": {bucket: counts[bucket] for bucket in BUCKETS},
        "unknownCount": counts[UNKNOWN_PERSON],
        "unknownCountCached": cached_unknown,
        "findings": findings,
        "elapsedSeconds": round(elapsed, 1),
        "entries": new_entries,
    }


# --- ledger ------------------------------------------------------------------

def mint_sweep_run_id(collection: str) -> str:
    """``sensitivity-audit-<collection>-<ns>``.

    Not the ledger's own ``mint_run_id``, which keys on a whole-second ISO
    stamp: every sweep is recorded under ONE collection key, so two collections
    swept in the same second folded into a single run and the second one's
    verdict disappeared. Nanoseconds plus the swept collection make the id
    unique on both axes.
    """
    return f"{LEDGER_COLLECTION}-{collection}-{time.time_ns()}"


def ledger_record(status: str, *, started_at: str, finished_at: str, detail: dict,
                  collection: str, baseline: bool, job=None, trigger="cli",
                  error=None, duration=None) -> dict:
    """One self-contained closing record for the sweep.

    No opening ``stage: "begin"`` partial: this is a single foreground process
    that either writes its record or died, and an unmatched opener would fold to
    `incomplete` forever on the (common) path where someone Ctrl-Cs a manual
    sweep. The phase is marked ``fatal`` so an unreachable model rolls the run up
    to ``failed`` rather than to ``degraded`` — a sweep that judged nothing must
    not read like a sweep that found something.

    ``variant`` follows the mode for the same reason the indexing jobs set it: a
    baseline re-reads every document and an incremental reads the changed ones,
    they differ by an order of magnitude in duration, and one median over both
    describes neither.
    """
    return {
        "runId": mint_sweep_run_id(collection),
        "collection": LEDGER_COLLECTION,
        "job": job,
        "trigger": trigger,
        "variant": "rebuild" if baseline else "incremental",
        "source": "script",
        "startedAt": started_at,
        "finishedAt": finished_at,
        "error": error,
        "phases": [{"name": LEDGER_PHASE, "status": status, "fatal": True,
                    "startedAt": started_at,
                    "durationSeconds": duration if duration is not None else 0,
                    "detail": {"collection": collection, **detail}}],
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


# --- the pasteable summary ----------------------------------------------------

def summary_lines(result: dict) -> list:
    """Counts and SHAPES only. This is the half that gets pasted somewhere."""
    counts = result["counts"]
    lines = [
        f"documents: {result['documents']} "
        f"({result['documentsAsked']} asked, {result['documentsCached']} cached), "
        f"{result['windows']} model calls in {result['elapsedSeconds']}s",
        f"alias/redaction: {counts[ALIAS]}   mapped residual: {counts[MAPPED_RESIDUAL]}   "
        f"role phrases: {counts[ROLE]}   "
        f"unknown persons: {result['unknownCount']} "
        f"({result['unknownCountCached']} from cached documents)",
        f"unparseable model answers: {result['parseFailures']}",
    ]
    for label, bucket, top in (("unknown", UNKNOWN_PERSON, 8),
                               ("residual", MAPPED_RESIDUAL, 5),
                               ("role", ROLE, 5)):
        shapes = Counter(sanitize(f["text"]) for f in result["findings"]
                         if f["classification"] == bucket)
        if shapes:
            lines.append(f"{label} shapes: {[s for s, _ in shapes.most_common(top)]}")
    return lines
