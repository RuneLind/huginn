"""The distribution gate: prove a BUILT collection is safe to hand to someone.

``scan_collection(collection_dir, map_path, ...)`` returns a :class:`ScanReport`.
``scripts/audit/scan_index.py`` is the CLI over it, and
``scripts/audit/package_collection.py`` calls it directly — packaging only
writes a tarball when the report passes. That is the point of promoting this out
of a script: a scanner someone has to remember to run is advisory, a scanner the
packager calls is a gate.

The checks, and why each is shaped the way it is:

0. **map/stamp agreement.** The manifest's privacy stamp must name the SAME map
   and policy version this run verifies with, and the map must clear
   MIN_MAP_ENTRIES. A truncated or decoy map certifies everything.
1. **Listed person forms.** Every entry variant, every unmapped-person variant
   and every token permutation of every mapped name: zero hits in the DECODED
   strings of documents/**, both index-mapping files and the manifest. Decoding
   is not a nicety — scanning raw file bytes reports a JSON ``\\n`` escape before
   a redacted birth number as an `n000000`-shaped token, i.e. 285 phantom
   "idents" on one collection. Needles are built with the registry's own
   ``boundaried()``, per shape: a mononym needle must not fire after a dot, a
   slug needle must not fire inside a longer dotted path, and a multi-token
   needle must survive percent-encoding.
2. **BM25 corpus tokens.** Each needle is tokenized the way the indexer
   tokenizes and the corpus is searched for that consecutive token sequence. A
   byte-grep of the pickle is vacuous: `ada example` matches zero bytes while
   both tokens sit side by side in the token list. Sequences without
   discriminative power (a single token, or any one-character token) are
   reported separately — they collide with ordinary prose.
3. **NAV idents** outside the exceptions file: zero. **3b** (with a compare
   twin) asserts aliasing moves no exception token.
4. **Dotted handles** left after the version/package/domain exclusions: zero.
5. **The exemption invariant** (with a compare twin), checked exactly: apply the
   registry to the PRE-ALIAS collection's own decoded text and compare exempt
   label counts before and after. Comparing the two built collections instead
   would measure build drift.
6. **Manifest and contextual prefixes**: the prefix block survives, the privacy
   stamp is present, no cached prefix replayed a real name, and — with a compare
   twin — document/chunk counts match (``allow_count_drift`` downgrades that to
   a warning; a count the twin's manifest does not carry at all is skipped
   rather than compared against ``None``, which used to read as a mismatch and
   fail every rebuild against an older manifest).
7. **Document ids and urls** (never aliased by design) contain no mapped name,
   matched as a token SEQUENCE so ``First-Last.md`` is caught.
8. **Every remaining file** under the collection directory is scanned as text. A
   ``.bak`` or an unrecognised binary is a failure in itself: the scan can only
   certify what it has read.
9. **Capitalised-bigram candidates — people the map does not know.** This is the
   headline category and the one the sweep before it could not see at all: every
   check above works from the map, so a colleague nobody ever added to the map
   is invisible to all of them. Candidates are `Capitalised Capitalised(+)`
   token runs whose FIRST token is a plausible given name — the public
   gazetteer, the given names of the private map, and the map's
   ``bare_given_name_residual``, unioned. **Retention, not subtraction.** An
   earlier draft filtered candidates by dropping anything that looked like a
   name, which removes precisely the category this check exists to find. What is
   then subtracted is only what has been reviewed: the map's
   ``non_person_labels`` and an explicit non-person allow-list.
10. **Distributor fingerprints.** An absolute ``/Users/`` path or a ``.bak``
   reference anywhere in the unit. The tarball goes to another machine; the
   builder's home directory should not travel with it, and on the real corpus
   this check found another person's macOS username inside a pasted stack trace.
11. **Sensitive tokens** (``main/privacy/sensitivity_scanner``): fødselsnummer,
   organisasjonsnummer, bank account numbers, credentials block; email,
   plaintext-password patterns and phone numbers are reported without blocking.
   Measured on the three in-scope collections, the blocking categories fire 6/18/20
   times and everything else fires twice, which is triageable; the *unanchored*
   shapes they replace fired 270/466/66 times, which is not.

SCOPE. Checks 1-8 certify that no *listed* variant, ident or dotted handle
survives. They deliberately do not cover bare given names — 74 of them remain in
the corpus by documented campaign decision, because substituting a bare given
name corrupts far more prose than it protects. Check 9 covers the pairs those
given names appear in, which is where a bare given name becomes identifying.

Real names are read from the gitignored map and NEVER printed: everything this
module reports is a shape (letters -> x) or a count. ``ScanReport.candidates``
is the one exception, and it is returned in memory for a caller to write to a
gitignored file — it is not part of ``to_json()``.
"""

import itertools
import json
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from main.indexes.indexers.bm25_indexer import _tokenize
from main.privacy.alias_registry import (
    POLICY_VERSION, AliasRegistry, boundaried, is_person_handle,
)
from main.privacy.sensitivity_scanner import (
    ADVISORY_CATEGORIES, BLOCKING_CATEGORIES, SensitivityScanner,
)

# The real map has 90 entries. A floor rejects a truncated or decoy map before it
# can certify a collection it knows almost no names from.
MIN_MAP_ENTRIES = 50

GIVEN_NAMES_FILE = Path(__file__).with_name("given_names.txt")

IDENT_RE = re.compile(r"(?i)(?<!\w)[a-z]\d{6}(?!\w)")
HANDLE_RE = re.compile(r"(?<![\w.])@[a-zæøå0-9]+(?:\.[a-zæøå0-9]+)+", re.IGNORECASE)
WORD_SPLIT_RE = re.compile(r"[^\w]+")

# A distributor fingerprint: an absolute macOS home path, or a reference to a
# backup file. `.bak` as a FILENAME is check 8; this is `.bak` inside content.
FINGERPRINT_RE = re.compile(r"/Users/[\w.\-]+|(?<![\w])[\w.\-]*\.bak(?![\w])")

# A capitalised name token: an initial capital and letters only. No digits and no
# underscore, so `MEL-18833` and `H_BUC-er` are not name tokens; an internal
# hyphen or apostrophe is (`Nord-Hansen`, `O'Brien`). At least two characters —
# a single initial after a given name (`Ada E`) is prose as often as it is a
# person, exactly the reason the BM25 check calls such a sequence
# non-discriminative.
_NAME_TOKEN = r"[A-ZÆØÅ][^\W\d_](?:[^\W\d_]|['’\-][^\W\d_])*"
# Horizontal whitespace only: unlike the substituter's variant separator this one
# must NOT span a newline. The last word of one line and the first of the next
# are both capitalised often enough that crossing the break invents candidates,
# and a name that really is line-wrapped is a *listed* form, which check 1 sees.
BIGRAM_RE = re.compile(rf"(?<![\w.])({_NAME_TOKEN}(?:[^\S\n]+{_NAME_TOKEN}){{1,2}})(?![\w])")

JSON_ARTIFACTS = ("indexes/index_document_mapping.json",
                  "indexes/reverse_index_document_mapping.json",
                  "manifest.json")
BM25_INDEX = "indexes/indexer_BM25/indexer"

# Checks 3 and 4 own idents and dotted handles; letting the sensitivity scanner
# report them too would double-count the same finding in two places.
SENSITIVITY_CATEGORIES = (BLOCKING_CATEGORIES | ADVISORY_CATEGORIES) - {
    "nav_ident", "dotted_handle"}


def sanitize(text: str) -> str:
    """Letters -> x. Everything printed by the gate goes through this."""
    return re.sub(r"[^\W\d_]", "x", text)


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


def needle_pattern(needles: list) -> re.Pattern:
    """One alternation, built with the REGISTRY's own per-shape boundaries.

    Sharing them is the point in both directions. A form the substituter removes
    but the scan cannot see (a name fenced by percent-escapes in a query string)
    would let the scan certify a collection that still carries it; a form the
    substituter deliberately leaves (a mononym welded onto a dotted path) but
    the scan flags would block distribution over a non-leak.
    """
    if not needles:
        raise ValueError("No needles built from the alias map — the scan would pass vacuously.")
    return re.compile("|".join(boundaried(n) for n in needles), re.I)


_WORD_RE = re.compile(r"[^\W_]+")
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def needle_prefilter(needles: list) -> set:
    """The lowercased FIRST word of every needle.

    The needle alternation is ~1000 branches, each with its own lookarounds, and
    Python's engine tries every branch at every position: 332 seconds over one
    6 MB collection, which makes a gate the packager calls unusable and a gate
    nobody runs. Every needle begins with an alphanumeric run, so a text
    containing none of these words cannot match any branch — one cheap
    ``[^\\W_]+`` pass rejects almost every string before the alternation sees it.
    The filter is a superset test, so it can only make the scan faster, never
    make it miss: a text that *does* contain a first word still runs the full
    alternation.
    """
    words = set()
    for needle in needles:
        match = _WORD_RE.search(needle)
        if match:
            words.add(match.group(0).lower())
    return words


def may_contain_needle(text: str, prefilter: set) -> bool:
    """Superset test: False only when no needle can possibly match.

    Percent escapes are replaced by a space first, and that is what makes the
    test sound rather than merely fast. `?f=%2CAda%20Example` tokenizes to
    `['f', '2CAda', '20Example']` — the needle's first word is welded to the
    escape's hex digits and the filter rejects a text the needle regex matches.
    Stripping escapes is also exhaustive, not a patch for one case: a percent
    escape is the ONLY reason the registry's boundaries admit a match that
    starts after a word character at all (see `VARIANT_LEFT_BOUNDARY` and the
    slug branch of `boundaried`), so with escapes gone, every needle match
    begins at a token start.
    """
    if "%" in text:
        text = _PERCENT_ESCAPE_RE.sub(" ", text)
    for match in _WORD_RE.finditer(text):
        if match.group(0).lower() in prefilter:
            return True
    return False


def token_sequences(needles: list) -> dict:
    """Needle -> its `[^\\w]+`-split token tuple, for path-shaped text."""
    sequences = {}
    for needle in needles:
        tokens = tuple(t.lower() for t in WORD_SPLIT_RE.split(needle) if t)
        if tokens:
            sequences.setdefault(tokens, needle)
    return sequences


def contains_sequence(tokens, sequences) -> bool:
    for position in range(len(tokens)):
        for length in {len(s) for s in sequences}:
            if tuple(tokens[position:position + length]) in sequences:
                return True
    return False


# --- given names ------------------------------------------------------------

def load_public_given_names(path: Path = GIVEN_NAMES_FILE) -> set:
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def map_given_names(alias_map: dict) -> set:
    """The given names the PRIVATE map knows, loaded at runtime and never shipped.

    Three sources: the first token of every multi-token entry variant, the same
    for the unmapped people, and every key of ``bare_given_name_residual`` — the
    given names the campaign decided not to substitute. That last set is exactly
    the population this check has to be able to retain: a residual given name
    followed by a surname is a full name nobody aliased.
    """
    names = set()
    for entry in alias_map.get("entries", []):
        for variant in entry.get("variants", []):
            if " " in variant:
                names.add(variant.split()[0].lower())
    for variants in alias_map.get("unmapped_people_variants", {}).values():
        for variant in variants:
            if " " in variant:
                names.add(variant.split()[0].lower())
    names |= {name.lower() for name in alias_map.get("bare_given_name_residual", {})}
    return {name for name in names if name}


def load_allowed_bigrams(path) -> set:
    """Explicitly reviewed non-person capitalised pairs, from a gitignored file.

    Seeded from a real corpus run and reviewed one by one: place names
    (`Jan Mayen`), design-system components whose first word is also a given
    name, sentence-initial Norwegian words that happen to be given names
    (`Andre …`, `Endre …`, `Per …`), and the national synthetic test persons.
    Missing file => nothing is allow-listed, which fails towards reporting.
    """
    if not path or not Path(path).exists():
        return set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {b.strip().lower() for b in data.get("bigrams", []) if isinstance(b, str) and b.strip()}


def bigram_candidates(texts, given_names: set, exempt: set, allowed: set):
    """(surviving Counter, count before the allow-list was applied)."""
    retained, before = Counter(), Counter()
    for text in texts:
        for match in BIGRAM_RE.finditer(text):
            candidate = re.sub(r"[^\S\n]+", " ", match.group(1))
            lowered = candidate.lower()
            if candidate.split()[0].lower() not in given_names or lowered in exempt:
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
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path, classify(path.relative_to(root))


def documents_of(root: Path):
    for path in sorted((root / "documents").rglob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


# --- the checks -------------------------------------------------------------

def _check_map_stamp(root: Path, alias_map: dict) -> Check:
    entries = len(alias_map["entries"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    stamp = manifest.get("privacy") or {}
    problems = []
    if entries < MIN_MAP_ENTRIES:
        problems.append(f"map has {entries} entries, below the {MIN_MAP_ENTRIES} floor")
    if stamp.get("map_version") != alias_map.get("version"):
        problems.append(f"manifest map_version {stamp.get('map_version')} "
                        f"!= {alias_map.get('version')}")
    if stamp.get("policy_version") != POLICY_VERSION:
        problems.append(f"manifest policy_version {stamp.get('policy_version')} "
                        f"!= {POLICY_VERSION}")
    return Check("0", "map_stamp", not problems, len(problems),
                 f"({entries} entries, map v{alias_map.get('version')}, "
                 f"policy v{POLICY_VERSION}) {'; '.join(problems) or 'ok'}")


def _check_artifacts(root: Path, needle_re, prefilter, sequences, exceptions, sensitivity,
                     given_names, exempt, allowed) -> list:
    """Checks 1, 3, 4, 7, 8, 9, 10 and 11 — one walk over every file."""
    per_artifact, idents, handles = Counter(), Counter(), Counter()
    fingerprints, sensitive = Counter(), Counter()
    id_url_hits, unreadable, texts = [], [], []
    scanned = 0

    def scan_text(label, text):
        if may_contain_needle(text, prefilter):
            hits = needle_re.findall(text)
            if hits:
                per_artifact[label] += len(hits)
        for token in IDENT_RE.findall(text):
            if token.lower() not in exceptions:
                idents[token] += 1
        for match in HANDLE_RE.finditer(text):
            if is_person_handle(match.group(0)):
                handles[match.group(0)] += 1
        for match in FINGERPRINT_RE.finditer(text):
            fingerprints[sanitize(match.group(0))] += 1
        for finding in sensitivity.detect(text):
            sensitive[finding.category] += 1
        texts.append(text)

    for path, kind in collection_files(root):
        relative = path.relative_to(root).as_posix()
        label = "documents" if relative.startswith("documents/") else relative
        if kind == "bak":
            unreadable.append(relative)
            continue
        if kind == "bm25":
            continue
        if kind == "vectors":
            ok, problem = _scan_vectors(path)
            if not ok:
                unreadable.append(f"{relative}: {problem}")
            scanned += 1
            continue
        scanned += 1
        if kind == "json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            for text in walk_strings(payload):
                scan_text(label, text)
            if relative.startswith("documents/"):
                for field_name in ("id", "url"):
                    value = payload.get(field_name) or ""
                    tokens = [t.lower() for t in WORD_SPLIT_RE.split(value) if t]
                    if contains_sequence(tokens, sequences):
                        id_url_hits.append(sanitize(value))
            continue
        try:
            scan_text(label, path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            unreadable.append(relative)

    retained, before = bigram_candidates(texts, given_names, exempt, allowed)
    blocking = {c: n for c, n in sensitive.items() if c in BLOCKING_CATEGORIES}
    advisory = {c: n for c, n in sensitive.items() if c not in BLOCKING_CATEGORIES}

    checks = [
        Check("1", "person_forms", not per_artifact, sum(per_artifact.values()),
              f"in {scanned} scanned files {dict(per_artifact) or '(clean)'}"),
        Check("3", "nav_idents", not idents, sum(idents.values()),
              f"non-exempt ident tokens {[sanitize(t) for t in list(idents)[:10]]}"),
        Check("4", "dotted_handles", not handles, sum(handles.values()),
              f"after version/package/domain exclusions "
              f"{[sanitize(h) for h in list(handles)[:5]]}"),
        Check("7", "id_url_names", not id_url_hits, len(id_url_hits),
              f"ids/urls containing a mapped name {id_url_hits[:3]}"),
        Check("8", "unreadable_files", not unreadable, len(unreadable),
              f".bak / uncertifiable files {unreadable[:5]}"),
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
    return checks, retained, len(before)


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


def _check_bm25(root: Path, needles, compare_root: Path | None) -> Check:
    path = root / BM25_INDEX
    if not path.exists():
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
        with open(target / BM25_INDEX, "rb") as f:
            corpus = pickle.load(f)["corpus_tokens"]
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
    before_text = []
    for document in documents_of(compare_root):
        before_text.extend(walk_strings(document))
    joined_before = "\n".join(before_text)
    joined_after = registry.apply(joined_before)

    terms = sorted({*alias_map["non_person_labels"], *exceptions}, key=len, reverse=True)
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


def _check_manifest_and_prefixes(root: Path, compare_root: Path | None, needle_re, prefilter,
                                 allow_count_drift: bool) -> Check:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    privacy = manifest.get("privacy")
    prefix_block = manifest.get("contextualPrefix")
    ok = privacy is not None
    notes = [f"privacy stamp: {privacy}"]

    if compare_root is not None:
        before = json.loads((compare_root / "manifest.json").read_text(encoding="utf-8"))
        expected_model = (before.get("contextualPrefix") or {}).get("model")
        if expected_model:
            ok = ok and (prefix_block or {}).get("model") == expected_model
            notes.append(f"contextualPrefix model: {(prefix_block or {}).get('model')} "
                         f"(was {expected_model})")
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
            ok = ok and allow_count_drift

    dirty = 0
    if prefix_block:
        prefixed = 0
        for document in documents_of(root):
            for chunk in document.get("chunks", []):
                prefix = chunk.get("contextualPrefix")
                if not prefix:
                    continue
                prefixed += 1
                if may_contain_needle(prefix, prefilter) and needle_re.search(prefix):
                    dirty += 1
        notes.append(f"chunks carrying a contextual prefix: {prefixed}; "
                     f"prefixes naming a mapped person: {dirty}")
    return Check("6", "manifest_and_prefixes", ok and dirty == 0, dirty, "", notes=notes)


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

    needles = build_needles(alias_map)
    needle_re = needle_pattern(needles)
    prefilter = needle_prefilter(needles)
    sequences = token_sequences(needles)

    given_names = load_public_given_names(given_names_file) | map_given_names(alias_map)
    exempt = {label.lower() for label in alias_map.get("non_person_labels", [])}
    allowed = load_allowed_bigrams(allowed_bigrams_path)
    sensitivity = SensitivityScanner(categories=SENSITIVITY_CATEGORIES,
                                     ident_exceptions=exceptions)

    report = ScanReport(collection=root.name, map_version=alias_map.get("version"),
                        policy_version=POLICY_VERSION, map_entries=len(alias_map["entries"]))
    report.checks.append(_check_map_stamp(root, alias_map))

    artifact_checks, candidates, before = _check_artifacts(
        root, needle_re, prefilter, sequences, exceptions, sensitivity,
        given_names, exempt, allowed)
    report.checks.extend(artifact_checks)
    report.candidates = [{"text": text, "occurrences": count}
                         for text, count in candidates.most_common()]
    report.candidates_before_allowlist = before

    report.checks.append(_check_bm25(root, needles, compare_root))
    if compare_root is not None:
        report.checks.extend(
            _check_exemption_invariant(compare_root, alias_map, exceptions, registry))
    report.checks.append(
        _check_manifest_and_prefixes(root, compare_root, needle_re, prefilter,
                                     allow_count_drift))
    report.checks.sort(key=lambda c: (int(re.match(r"\d+", c.number).group(0)), c.number))
    return report
