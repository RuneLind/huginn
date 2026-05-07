"""Author interaction graph compute.

Pure compute — caching and HTTP error handling stay in the endpoint.

Public API:
- ``build_author_graph(scores, sources_path, min_score, min_tweets, min_interactions)``
  — given pre-loaded author scores and a path to source markdown files,
  return the response dict in the same node/edge/community format as
  ``similarity_graph.shape_response``.
"""

import re
from collections import defaultdict
from pathlib import Path


_RE_HANDLE = re.compile(r"^\d{4}-\d{2}-\d{2}_(.+?)_\d+\.md$")
_RE_QUOTED = re.compile(r"> \*\*Quoted @(\w+):")
_RE_MENTION = re.compile(r"(?<![.\w])@(\w{1,15})(?!\.\w)")


def build_author_graph(scores, sources_path, min_score, min_tweets, min_interactions):
    """Build an author interaction graph from scores + tweet markdown files.

    ``scores`` is the loaded ``{handle: info}`` dict from the author-scores
    JSON. ``sources_path`` is the directory containing ``YYYY-MM-DD_handle_id.md``
    tweet files. Only authors with at least one interaction edge above
    ``min_interactions`` are returned (no isolates).
    """
    candidates = {
        handle for handle, info in scores.items()
        if info.get("author_score", 0) >= min_score
        and info.get("tweet_count", 0) >= min_tweets
    }

    interaction_counts = _count_interactions(Path(sources_path), candidates)

    connected = set()
    for (src, tgt), weight in interaction_counts.items():
        if weight >= min_interactions:
            connected.add(src)
            connected.add(tgt)

    orig_communities = {scores[handle].get("community", -1) for handle in connected}
    comm_remap = {old: new for new, old in enumerate(sorted(orig_communities))}

    nodes = [_make_node(handle, scores[handle], comm_remap) for handle in connected]
    nodes.sort(key=lambda n: -n["score"])

    edges = _make_edges(interaction_counts, connected, min_interactions)
    communities = _summarize_communities(nodes)

    return {
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "community_count": len(communities),
            "avg_similarity": round(sum(e["similarity"] for e in edges) / max(len(edges), 1), 4),
        },
    }


def _count_interactions(tweet_dir, candidates):
    """Walk markdown files and tally (src, tgt) → weighted interaction counts."""
    interaction_counts: dict[tuple[str, str], float] = defaultdict(float)
    if not tweet_dir.exists():
        return interaction_counts

    for f in tweet_dir.glob("*.md"):
        m = _RE_HANDLE.match(f.name)
        if not m:
            continue
        src = m.group(1).lower()
        if src not in candidates:
            continue
        content = f.read_text(encoding="utf-8")
        body = content
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                body = content[end + 3:]

        for qh in _RE_QUOTED.findall(body):
            tgt = qh.lower()
            if tgt in candidates and tgt != src:
                interaction_counts[(src, tgt)] += 3.0
        for line in body.split("\n"):
            if line.startswith("# @") or line.startswith("> **Quoted @") or line.startswith("- **Engagement"):
                continue
            for mh in _RE_MENTION.findall(line):
                tgt = mh.lower()
                if tgt in candidates and tgt != src:
                    interaction_counts[(src, tgt)] += 1.0

    return interaction_counts


def _make_node(handle, info, comm_remap):
    return {
        "id": handle,
        "title": f"@{handle}",
        "url": f"https://x.com/{handle}",
        "category": f"community-{comm_remap.get(info.get('community', -1), 0)}",
        "tags": [f"tweets:{info.get('tweet_count', 0)}"],
        "date": None,
        "headings": [],
        "summary": (
            f"Score: {info.get('author_score', 0):.3f} | "
            f"PageRank: {info.get('pagerank_norm', 0):.3f} | "
            f"Avg engagement: {info.get('avg_engagement', 0):.1f} | "
            f"Tweets: {info.get('tweet_count', 0)}"
        ),
        "community": comm_remap.get(info.get("community", -1), -1),
        "score": info.get("author_score", 0),
    }


def _make_edges(interaction_counts, connected, min_interactions):
    edges = []
    max_weight = max(interaction_counts.values()) if interaction_counts else 1.0
    for (src, tgt), weight in interaction_counts.items():
        if weight < min_interactions or src not in connected or tgt not in connected:
            continue
        edges.append({
            "source": src,
            "target": tgt,
            "similarity": round(weight / max_weight, 4),
        })
    return edges


def _summarize_communities(nodes):
    comm_members: dict[int, list] = defaultdict(list)
    for node in nodes:
        comm_members[node["community"]].append(node)

    communities = []
    for cid, members in sorted(comm_members.items(), key=lambda x: -len(x[1])):
        if len(members) < 2:
            continue
        top_authors = sorted(members, key=lambda n: -n["score"])
        name_parts = [n["title"] for n in top_authors[:2]]
        communities.append({
            "id": cid,
            "name": " + ".join(name_parts),
            "size": len(members),
            "top_tags": [],
            "top_categories": [],
            "representative_docs": [n["title"] for n in top_authors[:3]],
        })
    return communities
