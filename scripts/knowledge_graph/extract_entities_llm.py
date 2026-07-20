#!/usr/bin/env python3
"""
Extract a knowledge graph from documents using a local LLM (Ollama).

Sends each document to the LLM to extract entities (people, technologies,
concepts, organizations) and relationships between them. Produces a graph
JSON file compatible with KnowledgeGraph.

Usage:
    uv run scripts/knowledge_graph/extract_entities_llm.py --collection youtube-summaries
    uv run scripts/knowledge_graph/extract_entities_llm.py --collection youtube-summaries --model qwen3.6:35b-a3b-coding-nvfp4
    uv run scripts/knowledge_graph/extract_entities_llm.py --collection youtube-summaries --limit 10  # test run
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# Ensure the repo root is importable so the shared routing config in
# main.graph.graph_loader can be reused (single source of truth for the
# private-sub-repo discovery convention).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from main.graph.graph_loader import get_collection_manifest, resolve_graph_output_path
from main.utils.ollama_cli import call_ollama as call_ollama_chat


SYSTEM_PROMPT = """You are an entity and relationship extraction system.
Extract the most important entities and relationships from the given text.
Focus on named things: specific technologies, tools, people, companies, and key concepts.
Skip generic terms like "system", "method", "approach" unless they have a specific name.

Return ONLY valid JSON in this exact format:
{"entities": [{"name": "...", "type": "Technology|Person|Concept|Organization|Product"}], "relationships": [{"source": "...", "target": "...", "type": "uses|built_by|part_of|contains|related_to|alternative_to|created_by|improves|responds_to|precedes|succeeds|closes|invalidates|triggered_by"}]}

Rules:
- Entity names should be specific and canonical (e.g. "FAISS" not "faiss library")
- Limit to the 5-15 most important entities per document
- Only include relationships you are confident about
- Merge near-duplicates (e.g. "Claude" and "Claude AI" -> "Claude")

Direction rules for relationships — the order of source and target matters:
- "X part_of Y" means X is contained within Y. The container goes second.
  Correct: "chapter part_of book", "A001 part_of LA_BUC_01"
  Wrong:   "book part_of chapter", "LA_BUC_01 part_of A001"
- "X contains Y" is the inverse of part_of. The container goes first.
- "X uses Y" means X depends on Y. The consumer goes first.
- "X responds_to Y" means X is a reply or response to Y (messages, callbacks, follow-ups).
- "X precedes Y" / "X succeeds Y" — use only when the text states an explicit sequence.
- "X closes Y", "X invalidates Y", "X triggered_by Y" — use only when the text states this directly.

Canonical naming for short identifiers:
- When an entity has a short identifier (e.g. acronym + number) AND a descriptive title in the text, use only the bare identifier as the entity name. Put descriptive context elsewhere; do not concatenate it.
  Correct: name="ISO-8601", name="LA_BUC_01", name="A009"
  Wrong:   name="ISO-8601 Date Format", name="LA_BUC_01 Søknad om unntak", name="A009 — Forespørsel om tilleggsopplysninger"
- Do NOT append descriptive suffixes such as "_Subprocess", "– Close Case", "- Module" to the bare identifier."""

USER_PROMPT_TEMPLATE = """Extract entities and relationships from this document:

Title: {title}

{text}"""


def call_ollama(model: str, text: str, title: str, timeout: int = 300) -> dict | None:
    """Call Ollama to extract entities from a document.

    Wraps the shared ``call_ollama`` transport (imported as ``call_ollama_chat``)
    and restores this caller's swallow-and-continue contract: a transport/JSON
    failure or a response ``error`` field (both of which the shared helper
    raises as ``RuntimeError``) return ``None`` instead of propagating, and the
    raw string content is fence-stripped and ``json.loads``-parsed to a dict.
    ``temperature=0`` is pinned here (the shared default is 0.2) so entity
    extraction stays deterministic; ``num_predict`` rides in via ``options``.
    """
    user_content = USER_PROMPT_TEMPLATE.format(title=title, text=text[:3000])

    try:
        content = call_ollama_chat(
            user_content,
            model=model,
            timeout=timeout,
            temperature=0,
            system=SYSTEM_PROMPT,
            options={"num_predict": 3000},
        )
    except RuntimeError as e:
        print(f"  Error: {e}")
        return None

    if not content:
        return None
    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  Error: {e}")
        return None


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for deduplication."""
    return re.sub(r'\s+', ' ', name.strip())


def build_graph(all_extractions: list[dict], doc_titles: dict[str, str]) -> dict:
    """Merge extractions from all documents into a single graph."""
    # Count entity occurrences across documents for importance
    entity_counts = defaultdict(int)  # normalized_name -> count
    entity_type = {}  # normalized_name -> most common type
    entity_type_counts = defaultdict(lambda: defaultdict(int))
    entity_sources = defaultdict(set)  # normalized_name -> set of doc_ids

    relationship_counts = defaultdict(int)  # (src, tgt, type) -> count

    for doc_id, extraction in all_extractions:
        if not extraction:
            continue

        for ent in extraction.get("entities", []):
            if isinstance(ent, str):
                ent = {"name": ent}
            elif not isinstance(ent, dict):
                continue
            name = normalize_entity_name(ent.get("name", ""))
            etype = ent.get("type", "Concept")
            if not name or len(name) < 2:
                continue
            entity_counts[name] += 1
            entity_type_counts[name][etype] += 1
            entity_sources[name].add(doc_id)

        for rel in extraction.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            src = normalize_entity_name(rel.get("source", ""))
            tgt = normalize_entity_name(rel.get("target", ""))
            rtype = rel.get("type", "related_to")
            if src and tgt and src != tgt:
                relationship_counts[(src, tgt, rtype)] += 1

    # Determine most common type for each entity
    for name, type_counts in entity_type_counts.items():
        entity_type[name] = max(type_counts, key=type_counts.get)

    # Build nodes — include entities that appear in 2+ documents or are important
    nodes = []
    node_names = set()
    for name, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        node_id = f"entity:{name.lower().replace(' ', '_')}"
        nodes.append({
            "id": node_id,
            "type": entity_type[name],
            "label": name,
            "properties": {
                "mention_count": count,
                "source_documents": sorted(entity_sources[name]),
            },
        })
        node_names.add(name)

    # Build edges — only between nodes that exist
    edges = []
    seen_edges = set()
    for (src, tgt, rtype), count in sorted(relationship_counts.items(), key=lambda x: -x[1]):
        if src not in node_names or tgt not in node_names:
            continue
        src_id = f"entity:{src.lower().replace(' ', '_')}"
        tgt_id = f"entity:{tgt.lower().replace(' ', '_')}"
        edge_key = (src_id, tgt_id, rtype)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edges.append({
            "source": src_id,
            "target": tgt_id,
            "type": rtype,
            "properties": {"mention_count": count},
        })

    # Stats
    type_counts = defaultdict(int)
    for n in nodes:
        type_counts[n["type"]] += 1
    edge_type_counts = defaultdict(int)
    for e in edges:
        edge_type_counts[e["type"]] += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "by_type": dict(type_counts),
            "edge_types": dict(edge_type_counts),
        },
    }


def build_source_stamp(collection: str, data_path: str, processed_doc_count: int | None = None) -> dict:
    """Cheap provenance stamp for staleness detection at load time.

    The loader compares the stamp against the collection's current manifest
    and warns on divergence. Fields:

    - ``document_count``: on a full run, the manifest's authoritative
      ``numberOfDocuments`` — omitted when no manifest is readable (a raw file
      count would not be comparable against a later manifest). When ``--limit``
      truncated the run, ``processed_doc_count`` is stamped as-is so the
      partial graph honestly triggers the staleness warning.
    - ``last_modified_document_time``: the manifest's
      ``lastModifiedDocumentTime``, which only moves on real content change
      (``updatedTime`` is rewritten on every reindex run, including no-ops,
      and would warn permanently).
    """
    stamp = {"collection": collection}
    manifest = get_collection_manifest(data_path, collection)
    if processed_doc_count is not None:
        stamp["document_count"] = processed_doc_count
    elif manifest and manifest.get("numberOfDocuments") is not None:
        stamp["document_count"] = manifest["numberOfDocuments"]
    if manifest and manifest.get("lastModifiedDocumentTime"):
        stamp["last_modified_document_time"] = manifest["lastModifiedDocumentTime"]
    return stamp


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge graph using LLM")
    parser.add_argument("--collection", required=True, help="Collection name")
    parser.add_argument("--data-path", default="./data/collections", help="Base data path")
    parser.add_argument("--model", default="qwen3.6:35b-a3b-coding-nvfp4", help="Ollama model name")
    parser.add_argument("--output", default=None, help="Output JSON path (default: auto)")
    parser.add_argument("--limit", type=int, default=0, help="Max documents to process (0=all)")
    parser.add_argument("--min-text-length", type=int, default=100, help="Skip documents shorter than this")
    args = parser.parse_args()

    docs_dir = Path(args.data_path) / args.collection / "documents"
    if not docs_dir.exists():
        print(f"Error: Documents directory not found: {docs_dir}")
        return

    try:
        output_path = resolve_graph_output_path(args.collection, args.output)
    except ValueError as e:
        print(f"Error: {e}")
        return
    cache_path = output_path.with_suffix(".cache.json")

    # Find all document JSON files
    doc_files = sorted(docs_dir.rglob("*.json"))
    limited = 0 < args.limit < len(doc_files)
    if args.limit > 0:
        doc_files = doc_files[:args.limit]

    # Load cache of previously extracted documents
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    print(f"Collection: {args.collection}")
    print(f"Model: {args.model}")
    print(f"Documents: {len(doc_files)} ({len(cache)} in cache)")
    print(f"Output: {output_path}")
    print(f"Cache: {cache_path}")
    print()

    # Check Ollama is running
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("Error: Ollama is not running. Start it with 'ollama serve'")
        return

    # Process each document
    all_extractions = []
    doc_titles = {}
    skipped = 0
    errors = 0
    new_count = 0
    start_time = time.time()

    for i, doc_file in enumerate(doc_files):
        try:
            doc = json.loads(doc_file.read_text(encoding="utf-8"))
        except Exception:
            errors += 1
            continue

        text = doc.get("text", "")
        if len(text) < args.min_text_length:
            skipped += 1
            continue

        doc_id = doc.get("id", doc_file.stem)
        metadata = doc.get("metadata", {})
        title = metadata.get("title", doc_id.rsplit("/", 1)[-1].replace(".md", ""))
        doc_titles[doc_id] = title

        # Use cached extraction if available
        if doc_id in cache:
            all_extractions.append((doc_id, cache[doc_id]))
            continue

        elapsed = time.time() - start_time
        rate = (new_count / elapsed) if elapsed > 0 and new_count > 0 else 0
        remaining = len(doc_files) - i
        eta = (remaining / rate) if rate > 0 else 0
        print(f"  [{i+1}/{len(doc_files)}] {title[:60]}... ", end="", flush=True)

        extraction = call_ollama(args.model, text, title)
        if extraction:
            n_ent = len(extraction.get("entities", []))
            n_rel = len(extraction.get("relationships", []))
            print(f"{n_ent} entities, {n_rel} relationships ({eta:.0f}s remaining)")
            all_extractions.append((doc_id, extraction))
            cache[doc_id] = extraction
            new_count += 1
            if new_count % 20 == 0:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        else:
            print("failed")
            errors += 1

    # Final cache write
    if new_count > 0:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - start_time
    cached_count = len(all_extractions) - new_count
    print(f"\nProcessed {new_count} new + {cached_count} cached documents in {elapsed:.1f}s ({errors} errors, {skipped} skipped)")

    # Build merged graph
    graph = build_graph(all_extractions, doc_titles)
    # Stamp provenance so the loader can warn when the collection has moved on
    # since this graph was extracted (staleness signal, not a rebuild trigger).
    graph["source_stamp"] = build_source_stamp(
        args.collection, args.data_path, len(doc_files) if limited else None
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nGraph written to {output_path}")
    print(f"  Nodes: {graph['stats']['total_nodes']} ({', '.join(f'{t}: {c}' for t, c in graph['stats']['by_type'].items())})")
    print(f"  Edges: {graph['stats']['total_edges']} ({', '.join(f'{t}: {c}' for t, c in graph['stats']['edge_types'].items())})")

    # Show top entities
    top = sorted(graph["nodes"], key=lambda n: -n["properties"]["mention_count"])[:20]
    print(f"\nTop entities:")
    for n in top:
        docs = len(n["properties"]["source_documents"])
        print(f"  {n['label']} ({n['type']}) — {n['properties']['mention_count']} mentions in {docs} docs")


if __name__ == "__main__":
    main()
