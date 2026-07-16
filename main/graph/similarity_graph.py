"""Document similarity graph compute (FAISS embeddings + Louvain communities).

Pure compute — no HTTP, no caching. Callers own caching policy.
"""

import json
import logging

import faiss
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities

from main.utils.frontmatter import parse_tags

logger = logging.getLogger(__name__)


EMPTY_GRAPH = {
    "nodes": [],
    "edges": [],
    "stats": {"node_count": 0, "edge_count": 0, "avg_similarity": 0.0},
    "communities": [],
}


def detect_communities(sim_matrix, doc_ids, nodes, min_similarity=0.5):
    """Run Louvain community detection on the similarity matrix.

    Builds a networkx graph from document pairs above ``min_similarity``,
    then finds communities. Returns a list of community summary dicts and
    mutates each entry in ``nodes`` to add a ``community`` field.
    """
    num_docs = len(doc_ids)
    G = nx.Graph()
    G.add_nodes_from(range(num_docs))

    rows, cols = np.where(np.triu(sim_matrix, k=1) >= min_similarity)
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c), weight=float(sim_matrix[r, c]))

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if G.number_of_nodes() == 0:
        for i, node in enumerate(nodes):
            node["community"] = i
        return []

    communities = louvain_communities(G, weight="weight", resolution=1.0, seed=42)
    communities = sorted(communities, key=len, reverse=True)

    node_to_community = {}
    for comm_id, members in enumerate(communities):
        for member_idx in members:
            node_to_community[member_idx] = comm_id

    next_comm = len(communities)
    for idx in isolates:
        node_to_community[idx] = next_comm
        next_comm += 1

    for i, node in enumerate(nodes):
        node["community"] = node_to_community.get(i, -1)

    community_info = []
    for comm_id, members in enumerate(communities):
        member_nodes = [nodes[idx] for idx in members]
        tag_counts = {}
        for mn in member_nodes:
            for tag in mn.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]

        cat_counts = {}
        for mn in member_nodes:
            cat = mn.get("category", "")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        top_categories = sorted(cat_counts.items(), key=lambda x: -x[1])[:3]

        member_set = set(members)
        internal_degree = {}
        for idx in members:
            deg = sum(1 for neighbor in G.neighbors(idx) if neighbor in member_set)
            internal_degree[idx] = deg
        top_members = sorted(members, key=lambda x: -internal_degree.get(x, 0))[:3]
        representative_titles = [nodes[idx]["title"] for idx in top_members]

        if top_tags:
            name_parts = [t for t, _ in top_tags[:2]]
        elif top_categories:
            name_parts = [c for c, _ in top_categories[:2]]
        else:
            name_parts = [f"Cluster {comm_id}"]
        community_name = " + ".join(name_parts)

        community_info.append({
            "id": comm_id,
            "name": community_name,  # may be deduplicated below
            "size": len(members),
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
            "top_categories": [{"category": c, "count": cnt} for c, cnt in top_categories],
            "representative_docs": representative_titles,
        })

    name_counts = {}
    for c in community_info:
        name_counts[c["name"]] = name_counts.get(c["name"], 0) + 1
    for name_val, count in name_counts.items():
        if count <= 1:
            continue
        for c in community_info:
            if c["name"] == name_val and c["representative_docs"]:
                doc_hint = c["representative_docs"][0]
                if len(doc_hint) > 30:
                    doc_hint = doc_hint[:27] + "..."
                c["name"] = f"{name_val}: {doc_hint}"

    return community_info


def build_similarity_graph(name, searcher, disk_persister):
    """Compute the full similarity graph payload for a collection.

    Returns ``{nodes, sim_matrix, doc_ids, communities}`` ready for caching,
    or ``None`` if the collection has no vectors or its index mapping cannot
    be read.
    """
    indexer = searcher.indexer
    faiss_indexer = indexer.faiss_indexer if hasattr(indexer, "faiss_indexer") else indexer
    idx = faiss_indexer.faiss_index  # IndexIDMap wrapping IndexFlatL2

    n_vectors = idx.ntotal
    if n_vectors == 0:
        return None

    all_vectors = idx.index.reconstruct_n(0, n_vectors)
    id_map = faiss.vector_to_array(idx.id_map)

    try:
        mapping_text = disk_persister.read_text_file(
            f"{name}/indexes/index_document_mapping.json"
        )
        mapping = json.loads(mapping_text)
    except Exception as e:
        logger.warning(f"Could not load index mapping for {name}: {e}")
        return None

    doc_chunks = {}
    doc_meta = {}
    for vec_idx, chunk_id in enumerate(id_map):
        entry = mapping.get(str(int(chunk_id)))
        if not entry:
            continue
        doc_url = entry.get("documentUrl", "")
        doc_id = entry["documentId"]
        doc_chunks.setdefault(doc_id, []).append(vec_idx)
        if doc_id not in doc_meta:
            doc_meta[doc_id] = {"url": doc_url, "path": entry.get("documentPath", "")}

    if not doc_chunks:
        return None

    doc_ids = list(doc_chunks.keys())
    dim = all_vectors.shape[1]
    doc_vectors = np.zeros((len(doc_ids), dim), dtype=np.float32)
    for i, doc_id in enumerate(doc_ids):
        doc_vectors[i] = all_vectors[doc_chunks[doc_id]].mean(axis=0)
    faiss.normalize_L2(doc_vectors)

    sim_matrix = doc_vectors @ doc_vectors.T

    nodes = []
    for doc_id in doc_ids:
        meta = doc_meta[doc_id]
        title = doc_id.rsplit("/", 1)[-1].replace(".md", "")
        category = doc_id.split("/")[0] if "/" in doc_id else "uncategorized"
        doc_date = None
        headings = []
        summary = ""
        tags_list = []
        try:
            doc_json = json.loads(disk_persister.read_text_file(
                f"{name}/documents/{doc_id}.json"
            ))
            stored_meta = doc_json.get("metadata") or {}
            chunk_meta = (doc_json.get("chunks") or [{}])[0].get("metadata", {})
            doc_date = chunk_meta.get("date") or stored_meta.get("date")

            if chunk_meta.get("category"):
                category = chunk_meta["category"]
            elif stored_meta.get("tags"):
                parsed = parse_tags(stored_meta["tags"])
                if parsed:
                    category = parsed[0]
            elif stored_meta.get("epic_summary"):
                category = stored_meta["epic_summary"]

            if stored_meta.get("title"):
                title = stored_meta["title"]
            if stored_meta.get("tags"):
                tags_list = parse_tags(stored_meta["tags"])

            headings = [c["heading"] for c in doc_json.get("chunks", []) if c.get("heading")]
            text = doc_json.get("text", "")
            if text:
                summary = text[:500].rstrip() + ("..." if len(text) > 500 else "")
        except Exception as e:
            logger.warning(f"Could not read metadata for {doc_id} in {name}: {e}")
        if not tags_list:
            tags_list = [t.strip() for t in category.split("/") if t.strip()]
        nodes.append({
            "id": doc_id,
            "title": title,
            "url": meta["url"],
            "category": category,
            "tags": tags_list,
            "date": doc_date,
            "headings": headings,
            "summary": summary,
        })

    # 75th percentile keeps top 25% of connections
    upper_tri = sim_matrix[np.triu_indices(len(doc_ids), k=1)]
    p75 = float(np.percentile(upper_tri, 75)) if len(upper_tri) > 0 else 0.5
    communities = detect_communities(sim_matrix, doc_ids, nodes, min_similarity=p75)

    return {"nodes": nodes, "sim_matrix": sim_matrix, "doc_ids": doc_ids, "communities": communities}


def shape_similarity_response(cached, top_k, min_similarity):
    """Turn a cached similarity-graph payload into the API response shape.

    For each node, picks its top-k neighbors above ``min_similarity`` and
    deduplicates pairs.
    """
    nodes = cached["nodes"]
    sim_matrix = cached["sim_matrix"]
    doc_ids = cached["doc_ids"]
    n = len(doc_ids)
    k = min(top_k, n - 1)

    edges = []
    seen_pairs = set()
    for i in range(n):
        row = sim_matrix[i]
        top_indices = np.argpartition(row, -(k + 1))[-(k + 1):]
        for idx in top_indices:
            j = int(idx)
            if j == i:
                continue
            sim = float(row[j])
            if sim < min_similarity:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append({
                "source": doc_ids[i],
                "target": doc_ids[j],
                "similarity": round(sim, 4),
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "communities": cached.get("communities", []),
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "community_count": len(cached.get("communities", [])),
            "avg_similarity": round(sum(e["similarity"] for e in edges) / max(len(edges), 1), 4),
        },
    }
