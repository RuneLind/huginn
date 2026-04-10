#!/usr/bin/env python3
"""
Cross-collection gap analysis.

Compares two Huginn collections to find content gaps — documents in one
collection that have no close counterpart in the other.

Two analysis modes:
  1. Embedding cross-search: for each document in A, find nearest neighbor in B
     using FAISS indexes. Documents with high distance are "gaps".
  2. Wiki-anchored topic analysis: use wiki entity/concept pages as topic
     anchors, search both collections via BM25 for each topic, and report
     topics with lopsided coverage.

Usage:
    .venv/bin/python scripts/cross_collection_gap_analysis.py \\
        --collections melosys-confluence-v3 jira-issues

    .venv/bin/python scripts/cross_collection_gap_analysis.py \\
        --collections melosys-confluence-v3 jira-issues \\
        --wiki ./huginn-nav/wiki \\
        --top 30
"""
import argparse
import json
import logging
import os
import sys
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import faiss
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.persisters.disk_persister import DiskPersister
from main.indexes.indexer_factory import detect_faiss_index, load_indexer
from main.utils.logger import setup_root_logger

setup_root_logger()
logger = logging.getLogger(__name__)

COLLECTIONS_BASE = "./data/collections"


def _persister():
    return DiskPersister(base_path=COLLECTIONS_BASE)


def load_faiss_index(collection_name):
    """Load a FAISS index from a collection, auto-detecting the index type."""
    p = _persister()
    faiss_index_name = detect_faiss_index(collection_name, p)
    indexer = load_indexer(faiss_index_name, collection_name, p)
    return indexer.faiss_index


def load_document_mapping(collection_name):
    """Load chunk-to-document mapping."""
    path = os.path.join(COLLECTIONS_BASE, collection_name, "indexes", "index_document_mapping.json")
    with open(path) as f:
        return json.load(f)


def load_bm25(collection_name):
    """Load a BM25 indexer from a collection."""
    p = _persister()
    bm25_path = f"{collection_name}/indexes/indexer_BM25/indexer"
    if not p.is_path_exists(bm25_path):
        return None
    return load_indexer("indexer_BM25", collection_name, p)


def load_manifest(collection_name):
    path = os.path.join(COLLECTIONS_BASE, collection_name, "manifest.json")
    with open(path) as f:
        return json.load(f)


def extract_vectors(idx):
    """Extract all vectors and ID map from a FAISS IndexIDMap."""
    inner = faiss.downcast_index(idx.index)
    n = idx.ntotal
    vectors = np.zeros((n, idx.d), dtype=np.float32)
    for i in range(n):
        vectors[i] = inner.reconstruct(i)
    return vectors, faiss.vector_to_array(idx.id_map)


def cross_search(vectors_a, idx_b, k=1):
    """For each vector in A, find the k nearest neighbors in B."""
    inner_b = faiss.downcast_index(idx_b.index)
    distances, _ = inner_b.search(vectors_a, k)
    return distances


def _build_source_file_index(base_path):
    """Build a filename→path lookup for a source directory (flat + one level nested)."""
    index = {}
    if not os.path.isdir(base_path):
        return index
    for entry in os.scandir(base_path):
        if entry.is_file():
            index[entry.name] = entry.path
        elif entry.is_dir():
            for child in os.scandir(entry.path):
                if child.is_file():
                    index.setdefault(child.name, child.path)
    return index


def _read_doc_created_time(document_path, source_file_index):
    """Read the created time from the source markdown YAML frontmatter."""
    doc_filename = os.path.basename(document_path)
    if doc_filename.endswith(".json"):
        source_filename = doc_filename[:-5]
    else:
        source_filename = doc_filename

    source_path = source_file_index.get(source_filename)
    if source_path:
        try:
            with open(source_path, encoding="utf-8") as f:
                text = f.read(2000)  # frontmatter is always in the first ~2KB
            fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
            if fm_match:
                fm = yaml.safe_load(fm_match.group(1))
                if isinstance(fm, dict):
                    return str(fm.get("created", fm.get("modifiedTime", "")))
        except (FileNotFoundError, UnicodeDecodeError, yaml.YAMLError):
            pass

    # Fallback: document JSON modifiedTime
    json_path = os.path.join(COLLECTIONS_BASE, document_path)
    try:
        with open(json_path) as f:
            doc = json.load(f)
        return doc.get("modifiedTime", "")
    except Exception:
        return ""


def _parse_datetime(dt_str):
    """Parse a datetime string to a timezone-aware datetime, tolerating various formats."""
    if not dt_str:
        return None
    dt_str = str(dt_str).strip('"')
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def aggregate_by_document(distances, id_map, mapping, max_age_cutoff=None, source_file_index=None):
    """Aggregate per-chunk distances to per-document mean distance.

    Returns dict of {doc_id: {"mean_dist": float, "min_dist": float, "n_chunks": int, "url": str}}.
    """
    doc_dists = {}
    doc_urls = {}
    doc_paths = {}
    for pos in range(len(id_map)):
        orig_id = str(int(id_map[pos]))
        entry = mapping.get(orig_id)
        if not entry:
            continue
        doc_id = entry["documentId"]
        d = float(distances[pos][0])
        doc_dists.setdefault(doc_id, []).append(d)
        if doc_id not in doc_urls:
            doc_urls[doc_id] = entry.get("documentUrl", "")
            doc_paths[doc_id] = entry.get("documentPath", "")

    result = {}
    skipped = 0
    for doc_id, dists in doc_dists.items():
        if max_age_cutoff and doc_paths.get(doc_id):
            mod_time = _read_doc_created_time(doc_paths[doc_id], source_file_index or {})
            dt = _parse_datetime(mod_time)
            if dt and dt < max_age_cutoff:
                skipped += 1
                continue

        result[doc_id] = {
            "mean_dist": float(np.mean(dists)),
            "min_dist": float(np.min(dists)),
            "n_chunks": len(dists),
            "url": doc_urls.get(doc_id, ""),
        }

    if skipped:
        print(f"  (filtered out {skipped} documents older than {max_age_cutoff.strftime('%Y-%m-%d')})")

    return result


def run_embedding_cross_search(coll_a, coll_b, top_n, max_age_cutoff=None,
                                manifests=None, mappings=None):
    """Run bidirectional embedding cross-search between two collections."""
    print(f"\n{'='*70}")
    print(f"  EMBEDDING CROSS-SEARCH: {coll_a} ↔ {coll_b}")
    if max_age_cutoff:
        print(f"  Filtering documents older than {max_age_cutoff.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")

    print(f"\nLoading FAISS indexes...")
    idx_a = load_faiss_index(coll_a)
    idx_b = load_faiss_index(coll_b)
    mapping_a = mappings[coll_a]
    mapping_b = mappings[coll_b]

    print(f"  {coll_a}: {idx_a.ntotal} vectors")
    print(f"  {coll_b}: {idx_b.ntotal} vectors")

    # Build source file indexes once for date lookups
    src_index_a, src_index_b = {}, {}
    if max_age_cutoff:
        src_index_a = _build_source_file_index(manifests[coll_a]["reader"]["basePath"])
        src_index_b = _build_source_file_index(manifests[coll_b]["reader"]["basePath"])

    # Extract vectors once per index
    vectors_a, ids_a = extract_vectors(idx_a)
    vectors_b, ids_b = extract_vectors(idx_b)

    # A → B
    print(f"\nSearching {coll_a} → {coll_b} ...")
    dists_a_to_b = cross_search(vectors_a, idx_b)
    docs_a = aggregate_by_document(dists_a_to_b, ids_a, mapping_a, max_age_cutoff, src_index_a)

    # B → A
    print(f"Searching {coll_b} → {coll_a} ...")
    dists_b_to_a = cross_search(vectors_b, idx_a)
    docs_b = aggregate_by_document(dists_b_to_a, ids_b, mapping_b, max_age_cutoff, src_index_b)

    # Report
    sorted_a = sorted(docs_a.items(), key=lambda x: x[1]["mean_dist"], reverse=True)
    sorted_b = sorted(docs_b.items(), key=lambda x: x[1]["mean_dist"], reverse=True)

    all_dists_a = [d["mean_dist"] for d in docs_a.values()]
    all_dists_b = [d["mean_dist"] for d in docs_b.values()]

    print(f"\n── {coll_a} documents least similar to {coll_b} ──")
    print(f"   ({len(docs_a)} docs, mean={np.mean(all_dists_a):.4f}, "
          f"median={np.median(all_dists_a):.4f}, max={np.max(all_dists_a):.4f})\n")
    _print_doc_table(sorted_a[:top_n])

    print(f"\n── {coll_b} documents least similar to {coll_a} ──")
    print(f"   ({len(docs_b)} docs, mean={np.mean(all_dists_b):.4f}, "
          f"median={np.median(all_dists_b):.4f}, max={np.max(all_dists_b):.4f})\n")
    _print_doc_table(sorted_b[:top_n])

    return docs_a, docs_b


def _doc_title(doc_id):
    """Extract a human-readable title from a document ID."""
    name = doc_id.rsplit("/", 1)[-1].replace(".md", "")
    if re.match(r"MELOSYS-\d+", name):
        parts = name.split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]}: {parts[1].replace('_', ' ')}"
    return name.replace("_", " ")


def _print_doc_table(docs):
    for doc_id, info in docs:
        title = _doc_title(doc_id)
        print(f"  {info['mean_dist']:.4f}  ({info['n_chunks']:2d} ch)  {title}")


def load_wiki_topics(wiki_path):
    """Load entity and concept pages from the wiki, returning structured topics."""
    topics = []
    for subdir in ["entities", "concepts"]:
        dirpath = os.path.join(wiki_path, subdir)
        if not os.path.isdir(dirpath):
            continue
        for filename in sorted(os.listdir(dirpath)):
            if not filename.endswith(".md") or filename == "CLAUDE.md":
                continue
            filepath = os.path.join(dirpath, filename)
            topic_type = "entity" if subdir == "entities" else "concept"
            topic = _parse_wiki_page(filepath, topic_type)
            if topic:
                topics.append(topic)
    return topics


def _parse_wiki_page(filepath, topic_type):
    """Parse a wiki page's frontmatter and first paragraph."""
    text = Path(filepath).read_text(encoding="utf-8")

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not fm_match:
        return None

    try:
        fm = yaml.safe_load(fm_match.group(1))
    except yaml.YAMLError:
        return None

    title = fm.get("title", Path(filepath).stem)
    aliases = fm.get("aliases", [])
    tags = fm.get("tags", [])

    body = text[fm_match.end():]
    lines = [l.strip() for l in body.split("\n") if l.strip() and not l.strip().startswith("#")]
    summary = lines[0] if lines else ""

    search_terms = [title] + [a for a in aliases if a != title]

    return {
        "title": title,
        "type": topic_type,
        "aliases": aliases,
        "tags": tags,
        "summary": summary,
        "search_terms": search_terms,
        "filepath": filepath,
    }


def run_wiki_topic_analysis(wiki_path, coll_a, coll_b, top_n, mappings=None):
    """For each wiki topic, search both collections and compare coverage."""
    print(f"\n{'='*70}")
    print(f"  WIKI-ANCHORED TOPIC ANALYSIS")
    print(f"  Wiki: {wiki_path}")
    print(f"  Collections: {coll_a}, {coll_b}")
    print(f"{'='*70}")

    topics = load_wiki_topics(wiki_path)
    if not topics:
        print("  No wiki topics found.")
        return []

    print(f"\n  Found {len(topics)} wiki topics ({sum(1 for t in topics if t['type']=='entity')} entities, "
          f"{sum(1 for t in topics if t['type']=='concept')} concepts)")

    print(f"\nLoading BM25 indexes...")
    bm25_a = load_bm25(coll_a)
    bm25_b = load_bm25(coll_b)
    mapping_a = mappings[coll_a]
    mapping_b = mappings[coll_b]

    if not bm25_a or not bm25_b:
        print("  ERROR: BM25 index not available for one or both collections.")
        return []

    print(f"  {coll_a} BM25: {bm25_a.get_size()} entries")
    print(f"  {coll_b} BM25: {bm25_b.get_size()} entries")

    results = []
    for topic in topics:
        query = " ".join(topic["search_terms"])

        hits_a = _count_bm25_hits(bm25_a, mapping_a, query)
        hits_b = _count_bm25_hits(bm25_b, mapping_b, query)

        ratio = _coverage_ratio(hits_a, hits_b)

        results.append({
            "title": topic["title"],
            "type": topic["type"],
            "query": query,
            "hits_a": hits_a,
            "hits_b": hits_b,
            "ratio": ratio,
        })

    print(f"\n── Topic coverage comparison ──")
    print(f"   {'Topic':<35} {'Type':<9} {coll_a:>12} {coll_b:>12}  {'Balance':>8}")
    print(f"   {'─'*35} {'─'*9} {'─'*12} {'─'*12}  {'─'*8}")

    by_ratio = sorted(results, key=lambda r: abs(r["ratio"]), reverse=True)
    for r in by_ratio:
        balance = _format_ratio(r["ratio"], coll_a, coll_b)
        print(f"   {r['title']:<35} {r['type']:<9} {r['hits_a']['docs']:>5} docs  {r['hits_b']['docs']:>5} docs  {balance}")

    a_only = [r for r in results if r["hits_a"]["docs"] > 0 and r["hits_b"]["docs"] == 0]
    b_only = [r for r in results if r["hits_b"]["docs"] > 0 and r["hits_a"]["docs"] == 0]

    if a_only:
        print(f"\n── Topics found ONLY in {coll_a} ──")
        for r in a_only:
            print(f"   {r['title']} ({r['type']}, {r['hits_a']['docs']} docs)")

    if b_only:
        print(f"\n── Topics found ONLY in {coll_b} ──")
        for r in b_only:
            print(f"   {r['title']} ({r['type']}, {r['hits_b']['docs']} docs)")

    return results


def _count_bm25_hits(bm25, mapping, query, max_results=50):
    """Search BM25 and count unique document hits."""
    scores, ids = bm25.search(query, number_of_results=max_results)

    if ids.size == 0:
        return {"docs": 0, "chunks": 0, "best_score": 0.0}

    seen_docs = set()
    for i in range(ids.shape[1]):
        chunk_id = str(int(ids[0][i]))
        entry = mapping.get(chunk_id)
        if entry:
            seen_docs.add(entry["documentId"])

    best_score = float(-scores[0][0]) if scores.size > 0 else 0.0

    return {"docs": len(seen_docs), "chunks": int(ids.shape[1]), "best_score": best_score}


def _coverage_ratio(hits_a, hits_b):
    """Compute a signed ratio: -1 = only in A, 0 = balanced, +1 = only in B."""
    a, b = hits_a["docs"], hits_b["docs"]
    if a == 0 and b == 0:
        return 0.0
    return (b - a) / (a + b)


def _format_ratio(ratio, name_a, name_b):
    if abs(ratio) < 0.2:
        return "balanced"
    if ratio < -0.6:
        return f"← {name_a}"
    if ratio < -0.2:
        return f"← lean {name_a[:8]}"
    if ratio > 0.6:
        return f"{name_b} →"
    return f"lean {name_b[:8]} →"


def main():
    ap = argparse.ArgumentParser(
        description="Cross-collection gap analysis for Huginn knowledge collections."
    )
    ap.add_argument("--collections", nargs=2, required=True, metavar="NAME",
                    help="Two collection names to compare")
    ap.add_argument("--wiki", default=None, metavar="PATH",
                    help="Path to wiki directory for topic-anchored analysis")
    ap.add_argument("--top", type=int, default=20,
                    help="Number of top gap documents to show per direction (default: 20)")
    ap.add_argument("--max-age-years", type=float, default=None, metavar="N",
                    help="Exclude documents older than N years (based on modifiedTime)")
    ap.add_argument("--json", action="store_true",
                    help="Output results as JSON")
    args = ap.parse_args()

    coll_a, coll_b = args.collections

    for coll in [coll_a, coll_b]:
        if not os.path.isdir(os.path.join(COLLECTIONS_BASE, coll)):
            print(f"ERROR: Collection '{coll}' not found in {COLLECTIONS_BASE}")
            sys.exit(1)

    # Load shared data once
    manifests = {c: load_manifest(c) for c in [coll_a, coll_b]}
    mappings = {c: load_document_mapping(c) for c in [coll_a, coll_b]}

    print(f"\nCollections:")
    for c in [coll_a, coll_b]:
        print(f"  {c}: {manifests[c]['numberOfDocuments']} docs, {manifests[c]['numberOfChunks']} chunks")

    max_age_cutoff = None
    if args.max_age_years:
        max_age_cutoff = datetime.now(timezone.utc) - timedelta(days=args.max_age_years * 365.25)
        print(f"\nAge filter: excluding documents modified before {max_age_cutoff.strftime('%Y-%m-%d')}")

    docs_a, docs_b = run_embedding_cross_search(
        coll_a, coll_b, args.top, max_age_cutoff, manifests=manifests, mappings=mappings)

    wiki_results = []
    if args.wiki:
        wiki_path = os.path.abspath(args.wiki)
        if not os.path.isdir(wiki_path):
            print(f"\nWARNING: Wiki path '{args.wiki}' not found, skipping topic analysis.")
        else:
            wiki_results = run_wiki_topic_analysis(
                wiki_path, coll_a, coll_b, args.top, mappings=mappings)

    if args.json:
        output = {
            "collections": [coll_a, coll_b],
            "embedding_gaps": {
                coll_a: {k: v for k, v in sorted(docs_a.items(), key=lambda x: x[1]["mean_dist"], reverse=True)[:args.top]},
                coll_b: {k: v for k, v in sorted(docs_b.items(), key=lambda x: x[1]["mean_dist"], reverse=True)[:args.top]},
            },
            "wiki_topics": wiki_results,
        }
        print(f"\n{json.dumps(output, indent=2, ensure_ascii=False)}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
