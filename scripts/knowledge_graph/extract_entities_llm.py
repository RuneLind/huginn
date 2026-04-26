#!/usr/bin/env python3
"""
Extract a knowledge graph from documents using a local LLM (Ollama).

Sends each document to the LLM to extract entities (people, technologies,
concepts, organizations) and relationships between them. Produces a graph
JSON file compatible with KnowledgeGraph.

Usage:
    uv run scripts/knowledge_graph/extract_entities_llm.py --collection youtube-summaries
    uv run scripts/knowledge_graph/extract_entities_llm.py --collection youtube-summaries --model qwen3.5:35b
    uv run scripts/knowledge_graph/extract_entities_llm.py --collection youtube-summaries --limit 10  # test run
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are an entity and relationship extraction system.
Extract the most important entities and relationships from the given text.
Focus on named things: specific technologies, tools, people, companies, and key concepts.
Skip generic terms like "system", "method", "approach" unless they have a specific name.

Return ONLY valid JSON in this exact format:
{"entities": [{"name": "...", "type": "Technology|Person|Concept|Organization|Product"}], "relationships": [{"source": "...", "target": "...", "type": "uses|built_by|part_of|related_to|alternative_to|created_by|improves"}]}

Rules:
- Entity names should be specific and canonical (e.g. "FAISS" not "faiss library")
- Limit to the 5-15 most important entities per document
- Only include relationships you are confident about
- Merge near-duplicates (e.g. "Claude" and "Claude AI" -> "Claude")"""

USER_PROMPT_TEMPLATE = """Extract entities and relationships from this document:

Title: {title}

{text}"""


def call_ollama(model: str, text: str, title: str, timeout: int = 300) -> dict | None:
    """Call Ollama to extract entities from a document."""
    user_content = USER_PROMPT_TEMPLATE.format(title=title, text=text[:3000])

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 1500},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            content = result.get("message", {}).get("content", "")
            if not content:
                return None
            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            return json.loads(content)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
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

    if args.output:
        output_path = Path(args.output)
    else:
        # Route to the right private repo based on collection name
        nav_collections = {"melosys-confluence-v3", "melosys-jira", "jira-issues", "nav-begreper-eessi"}
        if args.collection in nav_collections:
            preferred = Path(f"./huginn-nav/scripts/knowledge_graph/{args.collection}_llm_graph.json")
        else:
            preferred = Path(f"./huginn-jarvis/scripts/knowledge_graph/{args.collection}_llm_graph.json")

        if preferred.parent.exists():
            output_path = preferred
        else:
            output_path = Path(f"./scripts/knowledge_graph/{args.collection}_llm_graph.json")
    cache_path = output_path.with_suffix(".cache.json")

    # Find all document JSON files
    doc_files = sorted(docs_dir.rglob("*.json"))
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
