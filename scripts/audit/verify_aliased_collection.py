#!/usr/bin/env python3
"""Acceptance sweep: prove a BUILT collection carries no real person name.

Run against `<name>-aliased` before swapping it in, and against the live
collection afterwards.

    scripts/audit/verify_aliased_collection.py --collection jira-issues-aliased
    scripts/audit/verify_aliased_collection.py --collection jira-issues-aliased \
        --compare jira-issues        # adds the exempt-label / ident-exception invariant

The checks and why each is shaped the way it is:

1. Every entry variant, every unmapped-person variant, and every token permutation
   of every mapped name: zero hits in the **decoded** strings of documents/**,
   both index-mapping files and the manifest. Decoding is not a nicety — scanning
   the raw file bytes reports a JSON `\\n` escape before a redacted birth number
   as `n000000`-shaped tokens, i.e. 285 phantom "idents" on jira-issues.
2. BM25 corpus tokens: each needle is tokenized the way the indexer tokenizes and
   the corpus is searched for that consecutive token sequence. A byte-grep of the
   pickle is vacuous — `ada example` matches zero bytes while both tokens sit
   side by side in the token list. Sequences without discriminative power (a
   single token, or any one-character token — a 3-letter given name plus an
   initial tokenizes to `['ola', 'k']`) are reported separately instead of
   counted: they collide with ordinary prose and say nothing about names.
3. Ident-shaped tokens outside the exceptions file: zero.
4. Dotted handles left after the TLD/package exclusions: zero.
5. The exemption invariant, checked EXACTLY: apply the registry to the pre-alias
   collection's own decoded text and compare counts before and after. Comparing
   the two built collections instead would measure build drift — a rebuilt
   collection has newer source documents and freshly generated contextual
   prefixes, and role nouns move for reasons that have nothing to do with
   aliasing.
6. The contextual-prefix block survives the rebuild, the privacy stamp is
   present, and no cached prefix replayed a real name into a chunk.
7. Document ids and urls (never aliased by design) contain no mapped name.

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
from main.privacy.alias_registry import AliasRegistry  # noqa: E402

COLLECTIONS_DIR = REPO_ROOT / "data" / "collections"
MAP_GLOB = "huginn-*/privacy/aliases.json"
EXCEPTIONS_GLOB = "huginn-*/privacy/ident_exceptions.json"

IDENT_RE = re.compile(r"(?i)(?<!\w)[a-z]\d{6}(?!\w)")
HANDLE_RE = re.compile(r"@[a-zæøå]+(?:\.[a-zæøå]+)+")
TLDS = {"no", "com", "org", "net", "io", "eu", "co", "uk", "se", "dk", "de", "nl",
        "info", "dev", "ai", "local"}
PACKAGE_ROOTS = {"org", "com", "no", "io", "java"}


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


def collection_artifacts(collection: str):
    """Yield (label, decoded JSON) for every artifact that can leak a name."""
    root = COLLECTIONS_DIR / collection
    for path in sorted((root / "documents").rglob("*.json")):
        yield "documents", json.loads(path.read_text(encoding="utf-8"))
    for name in ("indexes/index_document_mapping.json",
                 "indexes/reverse_index_document_mapping.json",
                 "manifest.json"):
        path = root / name
        if path.exists():
            yield name, json.loads(path.read_text(encoding="utf-8"))


def documents_of(collection: str):
    for path in sorted((COLLECTIONS_DIR / collection / "documents").rglob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def check_decoded_artifacts(collection, needle_re, exceptions):
    """Checks 1, 3, 4 and 7, over decoded strings."""
    per_artifact, idents, handles = Counter(), Counter(), Counter()
    id_url_hits, scanned = [], 0

    for label, payload in collection_artifacts(collection):
        scanned += 1
        for text in walk_strings(payload):
            hits = needle_re.findall(text)
            if hits:
                per_artifact[label] += len(hits)
                if per_artifact[label] <= 3:
                    print(f"   !! {label}: {[sanitize(h) for h in hits[:3]]}")
            for token in IDENT_RE.findall(text):
                if token.lower() not in exceptions:
                    idents[token] += 1
            for match in HANDLE_RE.finditer(text):
                segments = match.group(0)[1:].split(".")
                if segments[-1] in TLDS or (len(segments) >= 3 and segments[0] in PACKAGE_ROOTS):
                    continue
                handles[match.group(0)] += 1
        if label == "documents":
            for field in ("id", "url"):
                if needle_re.search(payload.get(field, "")):
                    id_url_hits.append(sanitize(payload[field]))

    print(f"1. person forms in {scanned} decoded artifacts: {sum(per_artifact.values())} "
          f"({dict(per_artifact) or 'clean'})")
    print(f"3. non-exempt ident tokens: {sum(idents.values())} "
          f"{dict(list(idents.items())[:10]) if idents else ''}")
    print(f"4. dotted handles after exclusions: {sum(handles.values())} "
          f"{[sanitize(h) for h in handles] or ''}")
    print(f"7. ids/urls containing a mapped name: {len(id_url_hits)} {id_url_hits[:3]}")
    return not (per_artifact or idents or handles or id_url_hits)


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


def check_bm25(collection, needles, compare=None):
    path = COLLECTIONS_DIR / collection / "indexes" / "indexer_BM25" / "indexer"
    if not path.exists():
        print("2. BM25: no index, skipped")
        return True

    sequences, weak = {}, {}
    for needle in needles:
        tokens = tuple(_tokenize(needle))
        if not tokens:
            continue
        (sequences if _discriminative(tokens) else weak).setdefault(tokens, needle)

    def count(target):
        with open(COLLECTIONS_DIR / target / "indexes" / "indexer_BM25" / "indexer", "rb") as f:
            corpus = pickle.load(f)["corpus_tokens"]
        by_first, weak_by_first = {}, {}
        for tokens in sequences:
            by_first.setdefault(tokens[0], []).append(tokens)
        for tokens in weak:
            weak_by_first.setdefault(tokens[0], []).append(tokens)
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


def check_exemption_invariant(compare, alias_map, exceptions, registry):
    """Check 5: aliasing must not move a single exempt label or exception token.

    Exact, because both sides are the SAME text: the pre-alias collection's own
    decoded documents, before and after `registry.apply`.
    """
    before_text = []
    for document in documents_of(compare):
        before_text.extend(walk_strings(document))
    joined_before = "\n".join(before_text)
    joined_after = registry.apply(joined_before)

    def count(text, term):
        return len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I))

    moved = [(sanitize(label), count(joined_before, label), count(joined_after, label))
             for label in alias_map["non_person_labels"]
             if count(joined_before, label) != count(joined_after, label)]
    print(f"5. non_person_labels whose count aliasing changed (over {len(joined_before)} "
          f"chars of {compare}): {len(moved)} {moved[:5]}")

    ident_moved = [(token, count(joined_before, token), count(joined_after, token))
                   for token in sorted(exceptions)
                   if count(joined_before, token) != count(joined_after, token)]
    print(f"3b. ident exceptions whose count aliasing changed: {len(ident_moved)} {ident_moved}")
    return not (moved or ident_moved)


def check_manifest_and_prefixes(collection, compare, needle_re):
    manifest = json.loads((COLLECTIONS_DIR / collection / "manifest.json").read_text(encoding="utf-8"))
    privacy = manifest.get("privacy")
    prefix_block = manifest.get("contextualPrefix")
    ok = privacy is not None
    print(f"6. privacy stamp: {privacy}")
    if compare:
        before = json.loads((COLLECTIONS_DIR / compare / "manifest.json").read_text(encoding="utf-8"))
        expected_model = (before.get("contextualPrefix") or {}).get("model")
        if expected_model:
            ok = ok and (prefix_block or {}).get("model") == expected_model
            print(f"   contextualPrefix model: {(prefix_block or {}).get('model')} "
                  f"(was {expected_model})")
        for key in ("numberOfDocuments", "numberOfChunks"):
            print(f"   {key}: {manifest.get(key)} (was {before.get(key)})")

    if not prefix_block:
        return ok
    prefixed, dirty = 0, 0
    for document in documents_of(collection):
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
    args = ap.parse_args()

    map_paths = sorted(REPO_ROOT.glob(MAP_GLOB))
    if not map_paths:
        sys.exit(f"No alias map found ({MAP_GLOB})")
    alias_map = json.loads(map_paths[0].read_text(encoding="utf-8"))
    registry = AliasRegistry.load(str(map_paths[0]))

    exceptions = set()
    for path in sorted(REPO_ROOT.glob(EXCEPTIONS_GLOB)):
        exceptions |= {t.lower() for t in json.loads(path.read_text(encoding="utf-8"))["tokens"]}

    needles = build_needles(alias_map)
    needle_re = re.compile("|".join(r"(?<!\w)" + re.escape(n) + r"(?!\w)" for n in needles), re.I)
    print(f"collection: {args.collection}  needles: {len(needles)}  "
          f"ident exceptions: {len(exceptions)}\n")

    results = [check_decoded_artifacts(args.collection, needle_re, exceptions),
               check_bm25(args.collection, needles, compare=args.compare)]
    if args.compare:
        results.append(check_exemption_invariant(args.compare, alias_map, exceptions, registry))
    results.append(check_manifest_and_prefixes(args.collection, args.compare, needle_re))

    print("\nRESULT:", "PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
