"""The distribution gate: prove a BUILT collection is safe to hand to someone.

``scan_collection(collection_dir, map_path, ...)`` returns a :class:`ScanReport`.
``scripts/audit/scan_index.py`` is the CLI over it and
``scripts/audit/package_collection.py`` calls it directly — packaging writes a
tarball only when the report passes. A scanner someone has to remember to run is
advisory; a scanner the packager calls is a gate.

The checks: **0** map/stamp agreement, **1** listed person forms in the DECODED
strings of every artifact, **2** the same forms as consecutive BM25 corpus
tokens, **3** NAV idents and **3b** their exceptions, **4** dotted handles,
**5** the exemption invariant against the pre-alias twin, **6** manifest,
privacy stamp and contextual prefixes, **7** document ids and urls (never
aliased by design), **8** every file read or the scan fails, **9** capitalised
bigram candidates, **10** distributor fingerprints, **11**
``sensitivity_scanner`` categories, **12** the index mapping still resolving to
the documents on disk. ``benchmarks/pii/RESULTS.md`` carries the
measurements; each check's own docstring carries its reasoning.

SCOPE. Checks 1-8 certify that no *listed* form survives; they do not cover bare
given names, which the campaign decided not to substitute. Check 9 covers the
pairs those appear in, and is the only check that can see a person nobody ever
added to the map. Real names are read from the gitignored map and NEVER printed:
everything reported is a shape (letters -> x, digits -> 9) or a count.
``ScanReport.candidates`` is the one exception, returned in memory for a caller
to write to a gitignored file — it is not part of ``to_json()``.
"""

import itertools
import json
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from main.graph.graph_loader import get_collection_manifest
from main.indexes.indexers.bm25_indexer import _tokenize
from main.persisters.disk_persister import DiskPersister
from main.privacy.alias_registry import (
    POLICY_VERSION, VARIANT_LEFT_BOUNDARY, VARIANT_RIGHT_BOUNDARY, VARIANT_SEPARATOR,
    AliasRegistry, boundaried, shape as sanitize,
)
from main.privacy.sensitivity_scanner import (
    ALL_CATEGORIES, BLOCKING_CATEGORIES, SensitivityScanner,
)

# The real map has 90 entries. A floor rejects a truncated or decoy map before it
# can certify a collection it knows almost no names from.
MIN_MAP_ENTRIES = 50

# The same argument for the gazetteer check 9 retains against. A truncated one
# retains nothing, and "no candidates" is exactly what a clean collection looks
# like — the failure is silent in the direction that matters.
MIN_GAZETTEER_ENTRIES = 500

GIVEN_NAMES_FILE = Path(__file__).with_name("given_names.txt")

WORD_SPLIT_RE = re.compile(r"[^\w]+")

# A distributor fingerprint: an absolute macOS home path, or a reference to a
# backup file. `.bak` as a FILENAME is check 8; this is `.bak` inside content.
FINGERPRINT_RE = re.compile(r"/Users/[\w.\-]+|(?<![\w])[\w.\-]*\.bak(?![\w])")

INDEX_MAPPING = "indexes/index_document_mapping.json"
JSON_ARTIFACTS = (INDEX_MAPPING,
                  "indexes/reverse_index_document_mapping.json",
                  "manifest.json")
BM25_INDEX = "indexes/indexer_BM25/indexer"

# Every check this module can produce, in report order. The packager stamps a
# row for each so a reader can tell "passed" from "never ran": 3b and 5 only run
# with a pre-alias twin, and a stamp that simply omits them is indistinguishable
# from one where they passed.
CHECKS = (
    ("0", "map_stamp"),
    ("1", "person_forms"),
    ("2", "bm25_sequences"),
    ("3", "nav_idents"),
    ("3b", "ident_exceptions_unmoved"),
    ("4", "dotted_handles"),
    ("5", "exempt_labels_unmoved"),
    ("6", "manifest_and_prefixes"),
    ("7", "id_url_names"),
    ("8", "unreadable_files"),
    ("9", "bigram_candidates"),
    ("10", "fingerprints"),
    ("11", "sensitive_tokens"),
    ("12", "document_paths"),
)
CHECK_NAMES = tuple(name for _, name in CHECKS)

# Checks 3 and 4 own idents and dotted handles, so check 11 does not report them
# again — but the scanner is still asked for every category, and the findings are
# split by category here rather than re-implementing two of the matchers inline.
IDENT_CATEGORY, HANDLE_CATEGORY = "nav_ident", "dotted_handle"
_OWNED_ELSEWHERE = {IDENT_CATEGORY, HANDLE_CATEGORY}
CHECK_11_BLOCKING = BLOCKING_CATEGORIES - _OWNED_ELSEWHERE
CHECK_11_ADVISORY = ALL_CATEGORIES - BLOCKING_CATEGORIES - _OWNED_ELSEWHERE

# What separates the strings of one file when they are scanned as one blob (see
# `_check_artifacts`). NUL is not `\w`, not `\s` and not `.`, so no needle
# boundary, variant separator or name-run gap can span it: joining cannot invent
# a match that the strings did not carry individually.
STRING_JOIN = "\x00"


@dataclass
class Check:
    """One gate check. ``detail`` is safe to print: shapes and counts only."""
    number: str
    name: str
    passed: bool
    count: int
    detail: str = ""
    notes: list = field(default_factory=list)


@dataclass
class ScanReport:
    collection: str
    map_version: object
    policy_version: int
    map_entries: int
    checks: list = field(default_factory=list)
    # Bigram candidates that survived retention and the allow-list. REAL TEXT —
    # the caller writes these to a gitignored file for triage. Never in to_json().
    candidates: list = field(default_factory=list)
    candidates_before_allowlist: int = 0
    # Every file the walk saw, with its size and mtime. The packager tars THIS
    # list rather than re-walking: a file that appeared or changed between the
    # scan and the tar was never certified. Real ids, so not in to_json().
    scanned_members: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def check(self, name: str):
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def counts(self) -> dict:
        return {check.name: check.count for check in self.checks}

    def check_summary(self) -> dict:
        """Every KNOWN check, whether it ran or not. See ``CHECKS``."""
        summary = {}
        for _, name in CHECKS:
            check = self.check(name)
            summary[name] = {"passed": check.passed if check else True,
                             "count": check.count if check else 0,
                             "ran": check is not None}
        return summary

    def to_json(self) -> dict:
        """Machine-readable report. Shapes and counts only, no literals."""
        return {
            "collection": self.collection,
            "passed": self.passed,
            "mapVersion": self.map_version,
            "policyVersion": self.policy_version,
            "mapEntries": self.map_entries,
            "candidatesBeforeAllowlist": self.candidates_before_allowlist,
            "candidatesAfterAllowlist": len(self.candidates),
            "checks": [
                {"number": c.number, "name": c.name, "passed": c.passed,
                 "count": c.count, "detail": c.detail, "notes": c.notes}
                for c in self.checks
            ],
        }

    def format_lines(self) -> list:
        lines = []
        for check in self.checks:
            flag = "ok" if check.passed else "FAIL"
            lines.append(f"{check.number}. {check.name}: {check.count} [{flag}] {check.detail}")
            lines.extend(f"   {note}" for note in check.notes)
        return lines


# --- needles ----------------------------------------------------------------

def permutation_forms(name: str) -> set:
    tokens = name.split()
    forms = set()
    for perm in itertools.permutations(tokens):
        lower = [t.lower() for t in perm]
        forms.add(" ".join(perm))
        forms.add(".".join(lower))
        forms.add("_".join(lower))
        if len(perm) >= 2:
            forms.add(f"{perm[-1]}, {' '.join(perm[:-1])}")
    return forms


def build_needles(alias_map: dict) -> list:
    needles = set()
    for entry in alias_map["entries"]:
        needles |= set(entry["variants"]) | permutation_forms(entry["name"])
    for label, variants in alias_map["unmapped_people_variants"].items():
        needles |= set(variants) | permutation_forms(label)
    exempt = {x.lower() for x in alias_map["non_person_labels"]}
    return sorted({n for n in needles if n and n.lower() not in exempt}, key=len, reverse=True)


def scan_boundaried(literal: str) -> str:
    r"""``boundaried`` widened wherever the SUBSTITUTER's narrowing hides a leak.

    **The contract is scan >= substituter, not scan == substituter.** Sharing the
    registry's boundaries verbatim was half right: a form the substituter removes
    but the scan cannot see (a name fenced by percent escapes) lets the gate
    certify a collection that still carries it, so the scan must be at least as
    wide. But the two narrowings the registry applies exist to protect *code*
    from being rewritten, not to describe what is safe to ship:

    * a slug followed by another dotted label. `no.nav.ada.example.impl` is a
      package path the substituter must not touch — but `ada.example.md` is the
      person, in a document id, which is never aliased by design (check 7). The
      registry's ``(?!\.\w)`` made the gate blind to every one of those.
    * a mononym after a percent escape. `%20Zylphia` in a query string is the
      name in the clear; the substituter leaves it because a mononym welded onto
      a path segment is not a person, and the percent case is the one exception.

    Everything else is `boundaried` exactly, and
    ``test_the_scan_boundary_never_narrows_the_substituters`` holds the two
    together.
    """
    tokens = literal.split()
    body = VARIANT_SEPARATOR.join(re.escape(t) for t in tokens) if len(tokens) > 1 \
        else re.escape(literal)
    slug = ("." in literal or "_" in literal) and len(tokens) == 1
    if slug:
        left, right = r"(?<![.\-])", r"(?![\w\-])"
    elif len(tokens) == 1:
        left, right = r"(?:(?<![.\w])|(?<=%[0-9A-Fa-f]{2}))", r"(?!\w)"
    else:
        left, right = VARIANT_LEFT_BOUNDARY, VARIANT_RIGHT_BOUNDARY
    prefix = left if literal[:1].isalnum() or literal[:1] == "_" else ""
    suffix = right if literal[-1:].isalnum() or literal[-1:] == "_" else ""
    return f"{prefix}{body}{suffix}"


def needle_pattern(needles: list) -> re.Pattern:
    """One alternation over every needle, with the per-shape scan boundaries."""
    if not needles:
        raise ValueError("No needles built from the alias map — the scan would pass vacuously.")
    return re.compile("|".join(scan_boundaried(n) for n in needles), re.I)


_WORD_RE = re.compile(r"[^\W_]+")


class NeedleScanner:
    r"""The needle alternation, split into buckets keyed on the first word.

    1 974 branches over the real map, each with its own lookarounds, and
    Python's engine tries every branch at every position: 170 s over the 4.0 M
    characters of one collection, and still 62 s behind the token prefilter this
    replaced — which makes a gate the packager calls unusable and a gate nobody
    runs. Bucketed it is 0.67 s (``benchmarks/pii/RESULTS.md`` §5).

    Every needle begins with an alphanumeric run, so a text that does not contain
    that word cannot match any branch of that needle's bucket. The word is looked
    for as a **substring** of ``text.lower()``, not as a token — that is what
    keeps the filter a superset. The predecessor tokenised the text and compared
    whole words, which is not:

    * `xyzada.example` — the slug boundary allows a preceding word character (a
      dotted name appears percent-encoded in URLs), so the needle matches while
      the first word is not a token of the text;
    * `50%Beate Example` — the old filter stripped percent escapes first, and
      `%Be` is a syntactically valid escape, so it ate the first two letters of
      the name it was supposed to find.

    100 of 100 fuzz cases over the real map went missing that way, and a miss
    here is a silent PASS.
    """

    def __init__(self, needles: list):
        if not needles:
            raise ValueError("No needles built from the alias map — the scan would pass "
                             "vacuously.")
        buckets: dict[str, list] = {}
        for needle in needles:
            match = _WORD_RE.search(needle)
            if match is None:
                # No alphanumeric run means no bucket to put it in, and the
                # predecessor simply left it out — i.e. never scanned for it,
                # which reads as "clean" in the report. There is no correct
                # bucket for such a needle, so the map is refused instead.
                raise ValueError(
                    f"a needle (shape {sanitize(needle)!r}) has no alphanumeric run and "
                    f"therefore no bucket; the scan would silently never look for it")
            if len(match.group(0).lower()) != len(match.group(0)):
                # `İ` lowercases to two chars; an anchor built from the lowered
                # word would not match the needle's own spelling (silent PASS).
                raise ValueError(
                    f"a needle (shape {sanitize(needle)!r}) starts with a character whose "
                    f"lowercase form is longer than itself; the bucket anchor cannot be trusted")
            buckets.setdefault(match.group(0).lower(), []).append(needle)
        self._patterns = []
        for word in sorted(buckets):
            group = buckets[word]
            pattern = re.compile("|".join(scan_boundaried(n) for n in group), re.I)
            # Every needle in this bucket begins WITH the bucket word, so every
            # match must begin at an occurrence of it: the alternation can be
            # tried at those positions only, instead of at all four million
            # positions of a collection. `Pattern.match(text, pos)` still
            # evaluates lookbehinds against the text before `pos`, which is what
            # keeps the boundaries meaningful. A bucket whose needles do not all
            # start with the word (a literal opening with punctuation) falls back
            # to a full scan rather than to a wrong answer.
            anchorable = all(n.lower().startswith(word) for n in group)
            # The anchor regex uses the SAME case fold as the needle pattern
            # (re.IGNORECASE), so a text spelling the bucket word with `ı`, `ſ`
            # or `İ` is found where the pattern would match it; `str.lower()`
            # folds differently and once hid such hits (see _iter).
            # Zero-width lookahead so overlapping occurrences (`adada` holds
            # `ada` at 0 AND 2) each become an anchor, like the old find() loop.
            word_re = re.compile("(?=" + re.escape(word) + ")", re.IGNORECASE)
            self._patterns.append((word_re, pattern, anchorable))

    @property
    def buckets(self) -> int:
        return len(self._patterns)

    def _iter(self, text: str):
        """Every match in `text`, bucketed where that is sound and in full where
        it is not.

        Bucket membership and anchors come from a case-insensitive literal
        regex run over the ORIGINAL text — never from `text.lower()`. Two
        silent-PASS bugs lived in the lowered-copy approach: `İ` (U+0130)
        lowercases to two characters and shifted every later anchor offset, and
        `re.IGNORECASE` folds `ı`, `ſ`, `µ` and a dozen Greek/Cyrillic variants
        onto letters that `str.lower()` leaves alone, so the filter rejected a
        bucket whose pattern matched. Using the same fold as the pattern for
        the filter makes the buckets a pure optimisation of the alternation.
        """
        for word_re, pattern, anchorable in self._patterns:
            if not anchorable:
                if word_re.search(text):
                    yield from pattern.finditer(text)
                continue
            for anchor in word_re.finditer(text):
                match = pattern.match(text, anchor.start())
                if match:
                    yield match

    def findall(self, text: str) -> list:
        return [match.group(0) for match in self._iter(text)]

    def search(self, text: str):
        for match in self._iter(text):
            return match
        return None


def token_sequences(needles: list) -> dict:
    """Needle -> its `[^\\w]+`-split token tuple, for path-shaped text."""
    sequences = {}
    for needle in needles:
        tokens = tuple(t.lower() for t in WORD_SPLIT_RE.split(needle) if t)
        if tokens:
            sequences.setdefault(tokens, needle)
    return sequences


def contains_sequence(tokens, sequences) -> bool:
    # The set of sequence lengths is a property of the needle set, not of the
    # position being probed; rebuilding it inside the loop rebuilt it once per
    # token of every document id in the collection.
    lengths = {len(s) for s in sequences}
    for position in range(len(tokens)):
        for length in lengths:
            if tuple(tokens[position:position + length]) in sequences:
                return True
    return False


# --- given names ------------------------------------------------------------

def load_public_given_names(path: Path = GIVEN_NAMES_FILE) -> set:
    if not Path(path).exists():
        return set()
    names = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        # Strip FIRST, then test for a comment. Testing the raw line let an
        # indented `# …` through as a name, and disagreed with the structural
        # test in tests/test_public_hygiene.py, which strips first.
        entry = line.strip()
        if entry and not entry.startswith("#"):
            names.add(entry.lower())
    return names


def map_given_names(alias_map: dict) -> set:
    """The given names the PRIVATE map knows, loaded at runtime and never shipped.

    Four sources: the first token of every multi-token entry variant, the first
    token of the entry's ``name`` (the only field guaranteed to spell the person
    `First Last` — a map whose variants are all `Last, First` and slugs
    contributed nothing), the same for the unmapped people, and every key of
    ``bare_given_name_residual``. That last set is exactly the population this
    check has to be able to retain: a residual given name followed by a surname
    is a full name nobody aliased.
    """
    names = set()

    def add_first_token(literal):
        if isinstance(literal, str) and " " in literal.strip():
            names.add(literal.split()[0].lower())

    for entry in alias_map.get("entries", []):
        add_first_token(entry.get("name"))
        for variant in entry.get("variants", []):
            add_first_token(variant)
    for label, variants in alias_map.get("unmapped_people_variants", {}).items():
        add_first_token(label)
        for variant in variants:
            add_first_token(variant)
    names |= {name.lower() for name in alias_map.get("bare_given_name_residual", {})}
    return {name for name in names if name}


def load_allowed_bigrams(path) -> set:
    """Explicitly reviewed non-person capitalised pairs, from a gitignored file.

    Seeded from a real corpus run and reviewed one by one: place names, design
    system components whose first word is also a given name, sentence-initial
    Norwegian words that happen to be given names, and the national synthetic
    test persons. Missing file => nothing is allow-listed, which fails towards
    reporting.
    """
    if not path or not Path(path).exists():
        return set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {b.strip().lower() for b in data.get("bigrams", []) if isinstance(b, str) and b.strip()}


# --- check 9: capitalised name runs -----------------------------------------

# A name token: letters only, at least two characters, an internal hyphen or
# apostrophe allowed (`Nord-Hansen`, `O'Brien`). No digits and no underscore, so
# an issue key (`ABC-12345`) and an underscored identifier are not name tokens. The initial capital is tested
# with `str.isupper()` rather than an `[A-ZÆØÅ]` class: that class is a Norwegian
# keyboard's alphabet, and the corpora are not — `Áilu` and `Łukasz` fell out of
# check 9 entirely, which is the one check that can see an unmapped person.
_NAME_TOKEN_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['’\-][^\W\d_])*")

# What may separate two tokens of one name. Horizontal whitespace, or ONE
# newline: a hard-wrapped `First\nLast` is the same name, and it is exactly the
# form the substituter's own VARIANT_SEPARATOR accepts. A blank line is a
# paragraph break and does not join.
_NAME_GAP_RE = re.compile(r"(?:[^\S\n]+|[^\S\n]*\n[^\S\n]*)")

_GIVEN_NAME_PART_RE = re.compile(r"[^\W\d_]{2,}")


def _name_tokens(text: str):
    """(token, start, end) for every token that can take part in a name run."""
    length = len(text)
    for match in _NAME_TOKEN_RE.finditer(text):
        token = match.group(0)
        if len(token) < 2 or not token[0].isupper():
            continue
        start, end = match.start(), match.end()
        # A digit or `_` welded to either side means this is an identifier
        # fragment, not a word: `id4711Ada`, `Ada_02`.
        if start and (text[start - 1].isdigit() or text[start - 1] == "_"):
            continue
        if end < length and (text[end].isdigit() or text[end] == "_"):
            continue
        yield token, start, end


def name_runs(text: str):
    """Maximal runs of two or more consecutive capitalised name tokens."""
    run, previous_end = [], 0
    for token, start, end in _name_tokens(text):
        if run and _NAME_GAP_RE.fullmatch(text, previous_end, start):
            run.append(token)
        else:
            if len(run) > 1:
                yield run
            run = [token]
        previous_end = end
    if len(run) > 1:
        yield run


def _is_given_name(token: str, given_names: set) -> bool:
    """True when the token, or any hyphen/apostrophe-separated part of it, is a
    plausible given name. `Anne-Marie` is one name to a reader and two entries to
    a gazetteer, and probing only the whole token dropped every compound."""
    lowered = token.lower()
    if lowered in given_names:
        return True
    return any(part in given_names for part in _GIVEN_NAME_PART_RE.findall(lowered))


def bigram_candidates(texts, given_names: set, exempt: set, allowed: set):
    """(surviving Counter, count before the allow-list was applied).

    EVERY adjacent pair inside a capitalised run is evaluated, not just the one
    at its head. Probing only the head meant a name behind any other capitalised
    word was invisible — an acronym (`NAV Kari Example`), a heading word
    (`## Referat Kari Example`), or simply another name.
    """
    retained, before = Counter(), Counter()
    for text in texts:
        for run in name_runs(text):
            for first, second in zip(run, run[1:]):
                if not _is_given_name(first, given_names):
                    continue
                candidate = f"{first} {second}"
                lowered = candidate.lower()
                if lowered in exempt:
                    continue
                before[candidate] += 1
                if lowered not in allowed:
                    retained[candidate] += 1
    return retained, before


# --- file walk --------------------------------------------------------------

def classify(relative: Path) -> str:
    posix = relative.as_posix()
    if posix.endswith(".bak"):
        return "bak"
    if posix.startswith("documents/") and posix.endswith(".json"):
        return "json"
    if posix in JSON_ARTIFACTS:
        return "json"
    if posix == BM25_INDEX:
        return "bm25"          # check 2 reads this one
    if re.fullmatch(r"indexes/indexer_[^/]+/indexer", posix):
        return "vectors"
    return "other"


# Directory names a collection's own layout puts there. Everything else in a
# path came from a document id or from whoever dropped a stray file in the tree.
_STRUCTURAL_DIRS = frozenset({"documents", "indexes"})
_INDEXER_DIR_RE = re.compile(r"indexer_[\w.\-]+")


def safe_location(relative: str) -> str:
    """Where an uncertifiable file sits, with no corpus-controlled text in it.

    Check 8's detail is printed and written to ``--json-report``, and a file
    under ``documents/`` is named after its document id — which is never
    aliased, by design, which is the entire reason check 7 exists. Reporting
    `documents/Ada Example.md.json: unreadable` would put a real name in the one
    output that is supposed to be safe to paste.

    So the FILENAME is dropped (`…`) and every directory segment is shaped
    (letters -> x) unless it is part of the collection's own layout. What is
    left — the directory, the reason, and the count — is what the operator needs
    to find the file on the disk they are already standing on.
    """
    segments = relative.split("/")
    directories = [segment if segment in _STRUCTURAL_DIRS
                   or _INDEXER_DIR_RE.fullmatch(segment) else sanitize(segment)
                   for segment in segments[:-1]]
    return "/".join([*directories, "…"])


def walk_strings(value):
    """Every string in a decoded JSON structure, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)


def _scan_vectors(path: Path):
    """A vector index is allowed to be a bare ndarray and nothing else."""
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception as e:                                  # noqa: BLE001
        return False, f"not loadable ({type(e).__name__})"
    if type(payload).__module__.startswith("numpy") and not list(walk_strings(payload)):
        return True, ""
    return False, f"vector index is not a bare ndarray ({type(payload).__name__})"


def collection_files(root: Path):
    """(path, kind) for everything under `root`, symlinks flagged, never followed.

    ``os.walk(followlinks=False)`` plus an explicit ``is_symlink()`` on each
    entry. A symlinked FILE would be scanned as its target — a file that is not
    part of the collection at all, and that the recipient will not receive — and
    a symlinked DIRECTORY would have the whole subtree walked and certified while
    the tarball carries a dangling link. Either one means the directory is not
    what the gate thinks it is, so both are check-8 failures.
    """
    entries = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        linked_dirs = [name for name in dirnames if (here / name).is_symlink()]
        entries.extend((here / name, "symlink") for name in linked_dirs)
        dirnames[:] = sorted(name for name in dirnames if name not in linked_dirs)
        for name in filenames:
            path = here / name
            entries.append((path, "symlink" if path.is_symlink()
                            else classify(path.relative_to(root))))
    return sorted(entries, key=lambda entry: entry[0].as_posix())


def _read_json(path: Path):
    """(payload, problem). A file the gate cannot decode is a check-8 failure
    with a path on it, never a traceback out of the middle of the scan."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        return None, f"unreadable ({type(e).__name__})"


def documents_of(root: Path):
    """Decoded documents. Unreadable ones are skipped — check 8 reports them."""
    for path in sorted((root / "documents").rglob("*.json")):
        if path.is_symlink():
            continue
        payload, problem = _read_json(path)
        if problem is None:
            yield payload


# --- the checks -------------------------------------------------------------

def _check_map_stamp(alias_map: dict, manifest) -> Check:
    entries = len(alias_map["entries"])
    problems = []
    if entries < MIN_MAP_ENTRIES:
        problems.append(f"map has {entries} entries, below the {MIN_MAP_ENTRIES} floor")
    if manifest is None:
        problems.append("manifest.json is missing or unreadable")
        stamp = {}
    else:
        stamp = manifest.get("privacy") or {}
    if manifest is not None:
        if stamp.get("map_version") != alias_map.get("version"):
            problems.append(f"manifest map_version {stamp.get('map_version')} "
                            f"!= {alias_map.get('version')}")
        if stamp.get("policy_version") != POLICY_VERSION:
            problems.append(f"manifest policy_version {stamp.get('policy_version')} "
                            f"!= {POLICY_VERSION}")
    return Check("0", "map_stamp", not problems, len(problems),
                 f"({entries} entries, map v{alias_map.get('version')}, "
                 f"policy v{POLICY_VERSION}) {'; '.join(problems) or 'ok'}")


def _check_artifacts(root: Path, scanner: NeedleScanner, sequences, sensitivity,
                     given_names, exempt, allowed):
    """Checks 1, 3, 4, 7, 8, 9, 10 and 11 — one walk over every file."""
    per_artifact, examples = Counter(), {}
    idents, handles = Counter(), Counter()
    fingerprints, sensitive = Counter(), Counter()
    id_url_hits, unreadable, texts, members = [], [], [], []
    scanned = 0

    def scan_strings(strings):
        """The cheap per-string detectors."""
        for text in strings:
            for finding in sensitivity.detect(text):
                sensitive[finding.category] += 1
                if finding.category == IDENT_CATEGORY:
                    idents[finding.shape] += 1
                elif finding.category == HANDLE_CATEGORY:
                    handles[finding.shape] += 1
            for match in FINGERPRINT_RE.finditer(text):
                fingerprints[sanitize(match.group(0))] += 1

    for path, kind in collection_files(root):
        relative = path.relative_to(root).as_posix()
        label = "documents" if relative.startswith("documents/") else relative
        # NEVER the filename: see `safe_location`.
        location = safe_location(relative)
        try:
            stat = os.lstat(path)
            members.append({"path": relative, "size": stat.st_size,
                            "mtime": stat.st_mtime_ns})
        except OSError as e:
            unreadable.append(f"{location}: unstattable ({type(e).__name__})")
            continue

        if kind == "symlink":
            unreadable.append(f"{location}: symlink (not followed)")
            continue
        if kind == "bak":
            unreadable.append(f"{location}: .bak file")
            continue
        if kind == "bm25":
            continue
        if kind == "vectors":
            ok, problem = _scan_vectors(path)
            if not ok:
                unreadable.append(f"{location}: {problem}")
            scanned += 1
            continue

        scanned += 1
        if kind == "json":
            payload, problem = _read_json(path)
            if problem:
                unreadable.append(f"{location}: {problem}")
                continue
            strings = list(walk_strings(payload))
            if relative.startswith("documents/") and isinstance(payload, dict):
                for field_name in ("id", "url"):
                    value = payload.get(field_name) or ""
                    tokens = [t.lower() for t in WORD_SPLIT_RE.split(value) if t]
                    if contains_sequence(tokens, sequences):
                        id_url_hits.append(sanitize(value))
        else:
            try:
                strings = [path.read_text(encoding="utf-8")]
            except (UnicodeDecodeError, OSError) as e:
                unreadable.append(f"{location}: unreadable ({type(e).__name__})")
                continue

        scan_strings(strings)
        # The needle alternation and the name-run walk run ONCE per file over the
        # joined strings rather than once per string. STRING_JOIN is NUL, which
        # no boundary, separator or gap in either pattern can cross, so the
        # result is identical and the per-string overhead (a `.lower()` and a
        # bucket sweep each) is paid once.
        blob = STRING_JOIN.join(strings)
        hits = scanner.findall(blob)
        if hits:
            per_artifact[label] += len(hits)
            examples.setdefault(label, []).extend(sanitize(h) for h in hits[:3])
        texts.append(blob)

    retained, before = bigram_candidates(texts, given_names, exempt, allowed)
    blocking = {c: n for c, n in sensitive.items() if c in CHECK_11_BLOCKING}
    advisory = {c: n for c, n in sensitive.items() if c in CHECK_11_ADVISORY}

    checks = [
        Check("1", "person_forms", not per_artifact, sum(per_artifact.values()),
              f"in {scanned} scanned files {dict(per_artifact) or '(clean)'}",
              # A bare count says a listed name survived somewhere in 543 files.
              # The masked shapes say which FORM it was, which is what tells the
              # operator whether to look at a slug, a permutation or prose.
              notes=[f"{label}: {sorted(set(shapes))[:3]}"
                     for label, shapes in sorted(examples.items())]),
        Check("3", "nav_idents", not idents, sum(idents.values()),
              f"non-exempt ident tokens {list(idents)[:10]}"),
        Check("4", "dotted_handles", not handles, sum(handles.values()),
              f"after version/package/domain exclusions {list(handles)[:5]}"),
        Check("7", "id_url_names", not id_url_hits, len(id_url_hits),
              f"ids/urls containing a mapped name {id_url_hits[:3]}"),
        Check("8", "unreadable_files", not unreadable, len(unreadable),
              f".bak / symlinked / uncertifiable files {unreadable[:5]}"),
        Check("9", "bigram_candidates", not retained, len(retained),
              f"unreviewed capitalised pairs with a plausible given name "
              f"({len(before)} before the allow-list, "
              f"{sum(retained.values())} occurrences)",
              notes=[f"shapes: {[sanitize(c) for c, _ in retained.most_common(5)]}"]
              if retained else []),
        Check("10", "fingerprints", not fingerprints, sum(fingerprints.values()),
              f"absolute /Users/ paths or .bak references "
              f"{[s for s, _ in fingerprints.most_common(3)]}"),
        Check("11", "sensitive_tokens", not blocking, sum(blocking.values()),
              f"blocking categories {blocking or '(clean)'}",
              notes=[f"advisory (not blocking): {advisory}"] if advisory else []),
    ]
    return checks, retained, len(before), members


def _discriminative(tokens) -> bool:
    """A token sequence that could only be a name, not prose.

    `['ola', 'k']` (a short given name plus an initial) matches ordinary
    Norwegian text; counting it as a leak produced six false positives whose
    contexts were all prose, and which occur *more* often in the pre-alias
    collection than in the aliased one.
    """
    return (len(tokens) >= 2
            and all(len(t) > 1 for t in tokens)
            and sum(len(t) for t in tokens) >= 8)


def _corpus_tokens(root: Path):
    """The BM25 corpus, read the way the searcher reads it.

    ``DiskPersister`` owns the unpickling and turns a truncated artifact into a
    ``CorruptArtifactError`` naming the file, instead of an ``EOFError`` from the
    middle of the gate.
    """
    return DiskPersister(str(root.parent)).read_bin_file(
        f"{root.name}/{BM25_INDEX}")["corpus_tokens"]


def _check_bm25(root: Path, needles, compare_root: Path | None) -> Check:
    """Check 2. A byte-grep of the pickle is vacuous: `ada example` matches zero
    bytes while both tokens sit side by side in ``corpus_tokens``."""
    if not (root / BM25_INDEX).exists():
        return Check("2", "bm25_sequences", True, 0, "no index, skipped")

    sequences, weak = {}, {}
    for needle in needles:
        tokens = tuple(_tokenize(needle))
        if not tokens:
            continue
        (sequences if _discriminative(tokens) else weak).setdefault(tokens, needle)

    by_first, weak_by_first = {}, {}
    for tokens in sequences:
        by_first.setdefault(tokens[0], []).append(tokens)
    for tokens in weak:
        weak_by_first.setdefault(tokens[0], []).append(tokens)

    def count(target: Path):
        corpus = _corpus_tokens(target)
        strong_hits, weak_hits = Counter(), 0
        for doc_tokens in corpus:
            for position, token in enumerate(doc_tokens):
                for candidate in by_first.get(token, ()):
                    if tuple(doc_tokens[position:position + len(candidate)]) == candidate:
                        strong_hits[sequences[candidate]] += 1
                for candidate in weak_by_first.get(token, ()):
                    if tuple(doc_tokens[position:position + len(candidate)]) == candidate:
                        weak_hits += 1
        return len(corpus), strong_hits, weak_hits

    chunks, hits, weak_hits = count(root)
    note = (f"{len(weak)} non-discriminative sequences excluded "
            f"(single token or a one-character token): {weak_hits} prose collisions")
    if compare_root is not None and (compare_root / BM25_INDEX).exists():
        note += f" (pre-alias control: {count(compare_root)[2]})"
    return Check("2", "bm25_sequences", not hits, sum(hits.values()),
                 f"over {chunks} indexed chunks, {len(sequences)} discriminative sequences "
                 f"{[sanitize(h) for h in list(hits)[:3]]}", notes=[note])


def _check_exemption_invariant(compare_root: Path, alias_map, exceptions, registry) -> list:
    """Checks 5 and 3b: aliasing must move no exempt label or exception token.

    Exact, because both sides are the SAME text: the pre-alias collection's own
    decoded documents, before and after `registry.apply`.
    """
    terms = sorted({*alias_map["non_person_labels"], *exceptions}, key=len, reverse=True)
    if not terms:
        # `"|".join([])` is the EMPTY pattern, which matches at every position:
        # the invariant would then compare a count of characters on each side and
        # report a "moved exempt label" for any text the registry shortened.
        detail = "no exempt labels and no ident exceptions to hold invariant"
        return [Check("5", "exempt_labels_unmoved", True, 0, detail),
                Check("3b", "ident_exceptions_unmoved", True, 0, detail)]

    before_text = []
    for document in documents_of(compare_root):
        before_text.extend(walk_strings(document))
    joined_before = "\n".join(before_text)
    joined_after = registry.apply(joined_before)

    pattern = re.compile("|".join(r"(?<!\w)" + re.escape(t) + r"(?!\w)" for t in terms), re.I)

    def counts(text):
        return Counter(m.group(0).lower() for m in pattern.finditer(text))

    before, after = counts(joined_before), counts(joined_after)
    exception_keys = {t.lower() for t in exceptions}
    moved = [(sanitize(term), before[term], after[term])
             for term in sorted(set(before) | set(after))
             if before[term] != after[term] and term not in exception_keys]
    ident_moved = [(sanitize(term), before[term], after[term])
                   for term in sorted(exception_keys)
                   if before[term] != after[term]]
    return [
        Check("5", "exempt_labels_unmoved", not moved, len(moved),
              f"over {len(joined_before)} chars of the pre-alias twin {moved[:5]}"),
        Check("3b", "ident_exceptions_unmoved", not ident_moved, len(ident_moved),
              f"{ident_moved}"),
    ]


def _check_manifest_and_prefixes(root: Path, manifest, compare_root: Path | None,
                                 scanner: NeedleScanner, allow_count_drift: bool) -> Check:
    problems, notes = [], []
    if manifest is None:
        return Check("6", "manifest_and_prefixes", False, 1,
                     "manifest.json is missing or unreadable")

    privacy = manifest.get("privacy")
    prefix_block = manifest.get("contextualPrefix")
    if privacy is None:
        problems.append("no privacy stamp in the manifest")
    notes.append(f"privacy stamp: {privacy}")

    if compare_root is not None:
        before = get_collection_manifest(compare_root.parent, compare_root.name) or {}
        expected_model = (before.get("contextualPrefix") or {}).get("model")
        if expected_model:
            actual_model = (prefix_block or {}).get("model")
            if actual_model != expected_model:
                problems.append("contextualPrefix model differs from the twin's")
            notes.append(f"contextualPrefix model: {actual_model} (was {expected_model})")
        for key in ("numberOfDocuments", "numberOfChunks"):
            new, old = manifest.get(key), before.get(key)
            if new is None or old is None:
                # One side simply does not record this number. Older manifests
                # predate `numberOfChunks`, and comparing a real count against
                # None reported a drift that does not exist — failing a rebuild
                # for the age of the twin's manifest rather than for its content.
                notes.append(f"{key}: not recorded on both sides, skipped")
                continue
            if new == old:
                notes.append(f"{key}: {new} (was {old})")
                continue
            notes.append(f"{'WARN' if allow_count_drift else 'FAIL'} {key}: {new} (was {old}) "
                         f"— the rebuild did not index the same set of documents")
            if not allow_count_drift:
                problems.append(f"{key} drifted from the twin")

    dirty = 0
    if prefix_block:
        prefixed = 0
        for document in documents_of(root):
            for chunk in document.get("chunks", []):
                prefix = chunk.get("contextualPrefix")
                if not prefix:
                    continue
                prefixed += 1
                if scanner.search(prefix):
                    dirty += 1
        notes.append(f"chunks carrying a contextual prefix: {prefixed}; "
                     f"prefixes naming a mapped person: {dirty}")
        if dirty:
            problems.append(f"{dirty} cached prefixes replayed a mapped name")

    # `count` used to be `dirty` unconditionally, so a missing stamp reported
    # "0 [FAIL]" — a count of the one thing that did NOT go wrong.
    return Check("6", "manifest_and_prefixes", not problems, len(problems),
                 "; ".join(problems), notes=notes)


def _path_prefix(document_path) -> str:
    """The collection directory a ``documentPath`` names, for the report.

    Only when there is one: a path with no ``/`` after its first segment has no
    directory name to report, and printing what is left would be printing a
    document filename — which IS a document id, and ids are never aliased
    (check 7's whole premise).
    """
    head, separator, _ = (document_path or "").partition("/")
    return head if (separator and head) else "(unrecognised)"


def _check_document_paths(root: Path, manifest) -> Check:
    """Check 12 — every mapped chunk still resolves to a document on disk.

    The searcher reads chunk text through the mapping's ``documentPath``, and
    resolves it against the collections directory — so the name that has to
    match is the **directory's**, which is also what ``KnowledgeStore`` loads
    the collection by. This check exists because a rebuild swap left the temp
    prefix on every path and three collections served empty chunk texts for
    weeks with no error anywhere (CLAUDE.md, "Build-time people aliasing").

    Keying off ``manifest.collectionName`` instead made the check
    self-consistent rather than true: a directory renamed or copied under
    another name (a backup, a scratch copy, a hand-made "staged" tree) agrees
    with its own manifest and serves nothing. So the manifest must ALSO name the
    directory — that equality is part of the check.

    No other check here could see any of this: they all read ``documents/*.json``
    directly and never ask whether the index still points at them. This one
    walks the join. Nothing corpus-controlled reaches the detail — the offending
    *prefix* (a collection directory name) and counts, never a path.
    """
    collection = root.name
    declared = (manifest or {}).get("collectionName")
    if declared != collection:
        return Check("12", "document_paths", False, 1,
                     f"the manifest calls this collection {declared!r} but the "
                     f"directory is {collection!r}; the searcher resolves every "
                     f"documentPath against the directory, so the two must agree")

    mapping, problem = _read_json(root / INDEX_MAPPING)
    if problem is not None or not isinstance(mapping, dict):
        return Check("12", "document_paths", False, 1,
                     f"{INDEX_MAPPING} is missing or undecodable — the index cannot "
                     f"be joined to the documents it claims to hold")
    if not mapping:
        # Every other check reads documents/*.json directly, so an index that
        # maps nothing passes all of them — and this one vacuously too.
        return Check("12", "document_paths", False, 1,
                     f"{INDEX_MAPPING} maps no chunks at all; there is no join to "
                     f"verify and nothing this collection can answer with")

    expected = f"{collection}/documents/"
    wrong_prefix, missing = Counter(), 0
    for entry in mapping.values():
        document_path = (entry.get("documentPath") or "") if isinstance(entry, dict) else ""
        if not document_path.startswith(expected):
            wrong_prefix[_path_prefix(document_path)] += 1
            continue
        relative = document_path[len(collection) + 1:]
        if ".." in Path(relative).parts or not (root / relative).is_file():
            missing += 1

    count = sum(wrong_prefix.values()) + missing
    if not count:
        return Check("12", "document_paths", True, 0,
                     f"{len(mapping)} mapped chunks, all under {expected} and all "
                     f"resolving to a file")
    return Check("12", "document_paths", False, count,
                 f"of {len(mapping)} mapped chunks: {sum(wrong_prefix.values())} under a "
                 f"foreign prefix {list(wrong_prefix)[:3]}, {missing} pointing at a file "
                 f"that is not there — chunk texts unreadable, reranked queries would "
                 f"return nothing")


# --- entry point ------------------------------------------------------------

def scan_collection(collection_dir, map_path, *, compare_dir=None, exceptions=(),
                    allow_count_drift=False, allowed_bigrams_path=None,
                    given_names_file=GIVEN_NAMES_FILE) -> ScanReport:
    """Scan one built collection directory. Returns a :class:`ScanReport`.

    `collection_dir` is the directory itself, not a name, so a staged copy and an
    untarred package are scanned by the same call the live collection is.
    `compare_dir` is the pre-alias twin, and enables checks 3b and 5 plus the
    document-count comparison; without it those simply do not run.
    """
    root = Path(collection_dir)
    compare_root = Path(compare_dir) if compare_dir else None
    alias_map = json.loads(Path(map_path).read_text(encoding="utf-8"))
    registry = AliasRegistry.load(str(map_path))
    exceptions = {t.lower() for t in exceptions}
    manifest = get_collection_manifest(root.parent, root.name)

    needles = build_needles(alias_map)
    scanner = NeedleScanner(needles)
    sequences = token_sequences(needles)

    given_names = load_public_given_names(given_names_file) | map_given_names(alias_map)
    if len(given_names) < MIN_GAZETTEER_ENTRIES:
        raise ValueError(
            f"the given-name gazetteer has {len(given_names)} entries, below the "
            f"{MIN_GAZETTEER_ENTRIES} floor — check 9 would retain almost nothing and "
            f"report a clean collection.")
    exempt = {label.lower() for label in alias_map.get("non_person_labels", [])}
    allowed = load_allowed_bigrams(allowed_bigrams_path)
    sensitivity = SensitivityScanner(categories=ALL_CATEGORIES, ident_exceptions=exceptions)

    report = ScanReport(collection=root.name, map_version=alias_map.get("version"),
                        policy_version=POLICY_VERSION, map_entries=len(alias_map["entries"]))
    report.checks.append(_check_map_stamp(alias_map, manifest))

    artifact_checks, candidates, before, members = _check_artifacts(
        root, scanner, sequences, sensitivity, given_names, exempt, allowed)
    report.checks.extend(artifact_checks)
    report.candidates = [{"text": text, "occurrences": count}
                         for text, count in candidates.most_common()]
    report.candidates_before_allowlist = before
    report.scanned_members = members

    report.checks.append(_check_bm25(root, needles, compare_root))
    if compare_root is not None:
        report.checks.extend(
            _check_exemption_invariant(compare_root, alias_map, exceptions, registry))
    report.checks.append(
        _check_manifest_and_prefixes(root, manifest, compare_root, scanner, allow_count_drift))
    report.checks.append(_check_document_paths(root, manifest))
    report.checks.sort(key=lambda c: (int(re.match(r"\d+", c.number).group(0)), c.number))
    return report
