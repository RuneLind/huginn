#!/usr/bin/env python3
"""Acceptance sweep: prove a BUILT collection carries no real person name.

Run against `<name>-aliased` before swapping it in, and against the live
collection afterwards.

    .venv/bin/python scripts/audit/verify_aliased_collection.py --collection jira-issues-aliased
    .venv/bin/python scripts/audit/verify_aliased_collection.py --collection jira-issues-aliased \
        --compare jira-issues        # adds the exempt-label / ident-exception invariant

`--collections-dir` verifies a staged copy instead of `data/collections`, and
`--map` pins the alias map (both default to the live ones). Every path is rooted
at the repo, never at the caller's CWD.

The checks and why each is shaped the way it is:

0. The map itself: the manifest's privacy stamp must name the SAME map and
   policy version this run is verifying with, and the map must have at least
   MIN_MAP_ENTRIES entries. A truncated or decoy map certifies everything.
1. Every entry variant, every unmapped-person variant, and every token permutation
   of every mapped name: zero hits in the **decoded** strings of documents/**,
   both index-mapping files and the manifest. Decoding is not a nicety — scanning
   the raw file bytes reports a JSON `\\n` escape before a redacted birth number
   as `n000000`-shaped tokens, i.e. 285 phantom "idents" on jira-issues. Needle
   tokens are joined by the registry's own separator pattern, so a name broken
   across a line, doubled space or `%20` is caught.
2. BM25 corpus tokens: each needle is tokenized the way the indexer tokenizes and
   the corpus is searched for that consecutive token sequence. A byte-grep of the
   pickle is vacuous — `ada example` matches zero bytes while both tokens sit
   side by side in the token list. Sequences without discriminative power (a
   single token, or any one-character token — a 3-letter given name plus an
   initial tokenizes to `['ola', 'k']`) are reported separately instead of
   counted: they collide with ordinary prose and say nothing about names.
3. Ident-shaped tokens outside the exceptions file: zero.
4. Dotted handles left after the version/package/domain exclusions: zero.
5. The exemption invariant, checked EXACTLY: apply the registry to the pre-alias
   collection's own decoded text and compare counts before and after. Comparing
   the two built collections instead would measure build drift — a rebuilt
   collection has newer source documents and freshly generated contextual
   prefixes, and role nouns move for reasons that have nothing to do with
   aliasing.
6. The contextual-prefix block survives the rebuild, the privacy stamp is
   present, no cached prefix replayed a real name into a chunk, and — with
   `--compare` — the document and chunk counts equal the pre-alias twin's.
   `--allow-count-drift` downgrades that last one to a WARN, for a source tree
   that has grown since the live collection was last built.
7. Document ids and urls (never aliased by design) contain no mapped name, matched
   as a token SEQUENCE over `[^\\w]+`-split path text so `First-Last.md` is caught.
8. Every remaining file under the collection directory is scanned as text. A
   `.bak` file or an unrecognised binary is a failure in itself: the sweep can
   only certify what it has read, and a stray `.txt` under `documents/` or a
   forgotten `indexes/index_info.json` used to be invisible to it.

SCOPE. This sweep certifies that no *listed* variant, ident or dotted handle
survives. It deliberately does NOT cover bare given names: 74 of them remain in
the corpus (the map's `bare_given_name_residual`), a documented campaign
decision (9,002 occurrences), because substituting a bare given name corrupts far
more prose than it protects. 9 of the 40 unmapped people are single-token
mononyms, and all 9 tokenize to one BM25 token — no discriminative sequence — so
check 2 cannot see them either. Check 1 covers them in the decoded text, which is
the check that matters for a copied collection.

Real names are read from the gitignored map and never printed.
"""
import argparse
import itertools
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.indexes.indexers.bm25_indexer import _tokenize  # noqa: E402
from main.privacy.alias_registry import (  # noqa: E402
    POLICY_VERSION, VARIANT_LEFT_BOUNDARY, VARIANT_RIGHT_BOUNDARY, VARIANT_SEPARATOR,
    AliasRegistry, is_person_handle,
)

DEFAULT_COLLECTIONS_DIR = REPO_ROOT / "data" / "collections"
MAP_GLOB = "huginn-*/privacy/aliases.json"
EXCEPTIONS_GLOB = "huginn-*/privacy/ident_exceptions.json"

# The real map has 90 entries. A floor rejects a truncated or decoy map before it
# can certify a collection it knows almost no names from.
MIN_MAP_ENTRIES = 50

IDENT_RE = re.compile(r"(?i)(?<!\w)[a-z]\d{6}(?!\w)")
HANDLE_RE = re.compile(r"(?<![\w.])@[a-zæøå0-9]+(?:\.[a-zæøå0-9]+)+", re.IGNORECASE)
WORD_SPLIT_RE = re.compile(r"[^\w]+")

JSON_ARTIFACTS = ("indexes/index_document_mapping.json",
                  "indexes/reverse_index_document_mapping.json",
                  "manifest.json")
BM25_INDEX = "indexes/indexer_BM25/indexer"


def sanitize(text: str) -> str:
    """Letters -> x. Failure output must stay safe to paste into a public PR."""
    return re.sub(r"[A-Za-zÆØÅæøå]", "x", text)


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
    """One alternation, built with the REGISTRY's separator and boundaries.

    Sharing them is the point: a form the substituter removes but the sweep
    cannot see (a name fenced by `%2C`/`%40` in a URL query string, where `\\w`
    lookarounds match the `C` and the `%`) would let the sweep certify a
    collection that still carries it.
    """
    if not needles:
        sys.exit("No needles built from the alias map — the sweep would pass vacuously.")
    alternatives = [VARIANT_LEFT_BOUNDARY
                    + VARIANT_SEPARATOR.join(re.escape(t) for t in n.split())
                    + VARIANT_RIGHT_BOUNDARY
                    for n in needles]
    return re.compile("|".join(alternatives), re.I)


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


def classify(relative: Path) -> str:
    """What kind of artifact a file under the collection root is."""
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


def scan_vectors(path: Path) -> bool:
    """A vector index is allowed to be a bare ndarray and nothing else."""
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception as e:
        print(f"   !! {path.name}: not loadable ({type(e).__name__})")
        return False
    if type(payload).__module__.startswith("numpy") and not list(walk_strings(payload)):
        return True
    print(f"   !! {path.name}: vector index is not a bare ndarray ({type(payload)})")
    return False


def collection_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path, classify(path.relative_to(root))


def check_artifacts(root, needle_re, sequences, exceptions):
    """Checks 1, 3, 4, 7 and 8, over every file in the collection directory."""
    per_artifact, idents, handles = Counter(), Counter(), Counter()
    id_url_hits, unreadable, scanned = [], [], 0

    def scan_text(label, text):
        hits = needle_re.findall(text)
        if hits:
            per_artifact[label] += len(hits)
            if per_artifact[label] <= 3:
                print(f"   !! {label}: {[sanitize(h) for h in hits[:3]]}")
        for token in IDENT_RE.findall(text):
            if token.lower() not in exceptions:
                idents[token] += 1
        for match in HANDLE_RE.finditer(text):
            if is_person_handle(match.group(0)):
                handles[match.group(0)] += 1

    for path, kind in collection_files(root):
        relative = path.relative_to(root).as_posix()
        label = "documents" if relative.startswith("documents/") else relative
        if kind == "bak":
            unreadable.append(relative)
            continue
        if kind == "bm25":
            continue
        if kind == "vectors":
            if not scan_vectors(path):
                unreadable.append(relative)
            scanned += 1
            continue
        scanned += 1
        if kind == "json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            for text in walk_strings(payload):
                scan_text(label, text)
            if relative.startswith("documents/"):
                for field in ("id", "url"):
                    value = payload.get(field) or ""
                    tokens = [t.lower() for t in WORD_SPLIT_RE.split(value) if t]
                    if contains_sequence(tokens, sequences):
                        id_url_hits.append(sanitize(value))
            continue
        try:
            scan_text(label, path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            unreadable.append(relative)

    print(f"1. person forms in {scanned} scanned files: {sum(per_artifact.values())} "
          f"({dict(per_artifact) or 'clean'})")
    print(f"3. non-exempt ident tokens: {sum(idents.values())} "
          f"{dict(list(idents.items())[:10]) if idents else ''}")
    print(f"4. dotted handles after exclusions: {sum(handles.values())} "
          f"{[sanitize(h) for h in handles] or ''}")
    print(f"7. ids/urls containing a mapped name: {len(id_url_hits)} {id_url_hits[:3]}")
    print(f"8. files the sweep could not certify (.bak / unreadable): "
          f"{len(unreadable)} {unreadable[:5]}")
    return not (per_artifact or idents or handles or id_url_hits or unreadable)


def documents_of(root: Path):
    for path in sorted((root / "documents").rglob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


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


def check_bm25(collections_dir, collection, needles, compare=None):
    path = collections_dir / collection / BM25_INDEX
    if not path.exists():
        print("2. BM25: no index, skipped")
        return True

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

    def count(target):
        with open(collections_dir / target / BM25_INDEX, "rb") as f:
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

    chunks, hits, weak_hits = count(collection)
    print(f"2. BM25 token sequences over {chunks} indexed chunks "
          f"({len(sequences)} discriminative sequences): {sum(hits.values())} hits "
          f"{[sanitize(h) for h in list(hits)[:3]]}")
    line = (f"   {len(weak)} non-discriminative sequences excluded "
            f"(single token or a one-character token): {weak_hits} prose collisions")
    if compare:
        _, _, weak_before = count(compare)
        line += f" (pre-alias control: {weak_before})"
    print(line)
    return not hits


def check_exemption_invariant(collections_dir, compare, alias_map, exceptions, registry):
    """Check 5: aliasing must not move a single exempt label or exception token.

    Exact, because both sides are the SAME text: the pre-alias collection's own
    decoded documents, before and after `registry.apply`. One alternation and a
    Counter over `finditer`, i.e. two passes over the corpus rather than ~130.
    """
    before_text = []
    for document in documents_of(collections_dir / compare):
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
    ident_moved = [(term, before[term], after[term])
                   for term in sorted(exception_keys)
                   if before[term] != after[term]]
    print(f"5. non_person_labels whose count aliasing changed (over {len(joined_before)} "
          f"chars of {compare}): {len(moved)} {moved[:5]}")
    print(f"3b. ident exceptions whose count aliasing changed: {len(ident_moved)} {ident_moved}")
    return not (moved or ident_moved)


def check_map_stamp(collections_dir, collection, alias_map):
    """Check 0: this sweep and this build must be talking about the same map."""
    entries = len(alias_map["entries"])
    manifest = json.loads((collections_dir / collection / "manifest.json").read_text(encoding="utf-8"))
    stamp = manifest.get("privacy") or {}
    problems = []
    if entries < MIN_MAP_ENTRIES:
        problems.append(f"map has {entries} entries, below the {MIN_MAP_ENTRIES} floor")
    if stamp.get("map_version") != alias_map.get("version"):
        problems.append(f"manifest map_version {stamp.get('map_version')} != {alias_map.get('version')}")
    if stamp.get("policy_version") != POLICY_VERSION:
        problems.append(f"manifest policy_version {stamp.get('policy_version')} != {POLICY_VERSION}")
    print(f"0. map/stamp agreement ({entries} entries, map v{alias_map.get('version')}, "
          f"policy v{POLICY_VERSION}): {'; '.join(problems) or 'ok'}")
    return not problems


def check_manifest_and_prefixes(collections_dir, collection, compare, needle_re,
                                allow_count_drift=False):
    root = collections_dir / collection
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    privacy = manifest.get("privacy")
    prefix_block = manifest.get("contextualPrefix")
    ok = privacy is not None
    print(f"6. privacy stamp: {privacy}")
    if compare:
        before = json.loads((collections_dir / compare / "manifest.json").read_text(encoding="utf-8"))
        expected_model = (before.get("contextualPrefix") or {}).get("model")
        if expected_model:
            ok = ok and (prefix_block or {}).get("model") == expected_model
            print(f"   contextualPrefix model: {(prefix_block or {}).get('model')} "
                  f"(was {expected_model})")
        # A rebuild that indexed a different set of documents is not a rebuild of
        # the same collection: its extra documents were never in the pre-alias
        # collection this sweep compares against, and a *smaller* count means the
        # rebuild silently dropped documents (a reader pattern that stopped
        # matching). Default is a hard failure. `--allow-count-drift` is for the
        # one knowingly-stale case — a source tree that grew between the last
        # build of the live collection and this rebuild — and downgrades it to a
        # printed WARN rather than removing the line.
        for key in ("numberOfDocuments", "numberOfChunks"):
            new, old = manifest.get(key), before.get(key)
            if new == old:
                print(f"   {key}: {new} (was {old})")
                continue
            print(f"   {'WARN' if allow_count_drift else '!!'} {key}: {new} (was {old}) — "
                  f"the rebuild did not index the same set of documents")
            ok = ok and allow_count_drift

    if not prefix_block:
        return ok
    prefixed, dirty = 0, 0
    for document in documents_of(root):
        for chunk in document.get("chunks", []):
            prefix = chunk.get("contextualPrefix")
            if not prefix:
                continue
            prefixed += 1
            if needle_re.search(prefix):
                dirty += 1
    print(f"   chunks carrying a contextual prefix: {prefixed}; "
          f"prefixes naming a mapped person: {dirty}")
    return ok and dirty == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--compare", default=None,
                    help="Pre-alias collection, for the exemption invariant and the BM25 control")
    ap.add_argument("--collections-dir", default=str(DEFAULT_COLLECTIONS_DIR),
                    help="Verify a staged copy instead of data/collections")
    ap.add_argument("--map", default=None, help="Alias map path (default: the discovered one)")
    ap.add_argument("--allow-count-drift", action="store_true",
                    help="Downgrade a numberOfDocuments/numberOfChunks difference against "
                         "--compare to a WARN. For a source tree that has grown since the "
                         "live collection was last built.")
    args = ap.parse_args()

    collections_dir = Path(args.collections_dir).resolve()
    if args.map:
        map_path = Path(args.map)
    else:
        map_paths = sorted(REPO_ROOT.glob(MAP_GLOB))
        if len(map_paths) != 1:
            sys.exit(f"Expected exactly one alias map ({MAP_GLOB}), found {len(map_paths)}")
        map_path = map_paths[0]
    alias_map = json.loads(map_path.read_text(encoding="utf-8"))
    registry = AliasRegistry.load(str(map_path))

    exceptions = set()
    for path in sorted(REPO_ROOT.glob(EXCEPTIONS_GLOB)):
        exceptions |= {t.lower() for t in json.loads(path.read_text(encoding="utf-8"))["tokens"]}

    needles = build_needles(alias_map)
    needle_re = needle_pattern(needles)
    sequences = token_sequences(needles)
    print(f"collection: {args.collection}  needles: {len(needles)}  "
          f"ident exceptions: {len(exceptions)}\n")

    results = [check_map_stamp(collections_dir, args.collection, alias_map),
               check_artifacts(collections_dir / args.collection, needle_re, sequences, exceptions),
               check_bm25(collections_dir, args.collection, needles, compare=args.compare)]
    if args.compare:
        results.append(check_exemption_invariant(collections_dir, args.compare, alias_map,
                                                 exceptions, registry))
    results.append(check_manifest_and_prefixes(collections_dir, args.collection,
                                               args.compare, needle_re,
                                               allow_count_drift=args.allow_count_drift))

    passed = all(results)
    # The PASS line states what it does and does NOT certify: an unqualified
    # "PASS" would read as "no real name survives", which is not what was checked.
    print("\nRESULT: PASS — no listed variant / ident / handle; bare given names are "
          "out of scope (map `bare_given_name_residual`)" if passed else "\nRESULT: FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
