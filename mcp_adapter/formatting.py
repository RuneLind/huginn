"""Pure markdown rendering for the MCP adapter's tool responses.

Everything here is a pure function of the API response payload — no environment
reads, no HTTP, no config gating. The entry file handles fetching, error
handling, and collection gating (all of which touch patchable config), then
hands the decoded JSON here to be rendered. Keeping this half config-free is
what lets the contract test (test_mcp_adapter_render_contract.py) pin the
formatter→adapter field coupling in one place.
"""
from main.graph.graph_search_augmenter import GraphSearchAugmenter

# Metadata keys that are internal and should not be shown in MCP output
_INTERNAL_METADATA_KEYS = {"page_id", "space", "breadcrumb", "title", "wip"}


def _format_relevance(score):
    """Format relevance score as percentage string."""
    if score is None:
        return ""
    return f"{score * 100:.1f}% relevant"


def _format_date(date_str):
    """Extract YYYY-MM-DD from an ISO datetime string."""
    if not date_str:
        return ""
    return date_str[:10]


def _is_wip(result):
    """Check if a search result is flagged as work-in-progress."""
    meta = result.get("metadata") or {}
    return meta.get("wip") == "true"


def _format_relevance_band(r: dict) -> str:
    """Format a result's relevance + confidence band, e.g. ' (82.0% relevant · high)'."""
    rel = r.get("relevance")
    if rel is None:
        return ""
    band = r.get("confidenceBand")
    return f" ({_format_relevance(rel)}{' · ' + band if band else ''})"


def _format_retry_hints(data: dict) -> str:
    """Render retryHints / noConfidentResults into a compact suggestion line, or ''.

    On a successful rescue, emit a one-line marker naming the original + rescue
    query instead of the weak-match footer — gives the model an explicit "these
    came from a fallback" signal without re-listing hints it already tried.
    When rescue fired but the result is still weak, keep the footer so muninn's
    Path C defence-in-depth still has signal."""
    corrective = data.get("corrective") or {}
    if corrective.get("verdict") == "rescued":
        queries_tried = corrective.get("queriesTried") or []
        if len(queries_tried) >= 2:
            original, rescue_q = queries_tried[0], queries_tried[-1]
            strategy = corrective.get("rescueStrategy")
            strategy_suffix = f" [{strategy}]" if strategy else ""
            return (
                f'\n\n*Rescued via broader query "{rescue_q}"{strategy_suffix} — '
                f'original query "{original}" found no confident match.*'
            )
        return ""
    hints = data.get("retryHints") or {}
    bits = []
    related = hints.get("relatedTerms")
    if related:
        bits.append("related terms: " + ", ".join(related))
    if hints.get("narrowerQuery"):
        bits.append(f'narrower query: "{hints["narrowerQuery"]}"')
    if hints.get("broaderQuery"):
        bits.append(f'broader query: "{hints["broaderQuery"]}"')
    if not bits and not data.get("noConfidentResults"):
        return ""
    prefix = "No confident match" if data.get("noConfidentResults") else "Weak match"
    return f"\n\n*{prefix} — try: {' · '.join(bits)}*" if bits else f"\n\n*{prefix}.*"


def render_results(data: dict, brief: bool) -> str:
    """Render the search-results body (graph answer + results), sans retry hints
    and trace marker (the entry file appends those). Mirrors the original inline
    rendering exactly — byte-compatible output."""
    results = data.get("results", [])
    parts = []
    if data.get("graph_answer"):
        parts.append(f"**Graph:**\n{data['graph_answer']}\n")
    if data.get("lowConfidence"):
        parts.append("*Low confidence results — query may not match indexed content well.*\n")

    if brief:
        for i, r in enumerate(results, 1):
            heading = f" > {r['heading']}" if r.get("heading") else ""
            relevance = _format_relevance_band(r)
            date = f" | {_format_date(r['modifiedTime'])}" if r.get("modifiedTime") else ""
            breadcrumb = f"\n   {r['breadcrumb']}" if r.get("breadcrumb") else ""
            wip = " **[UNDER ARBEID]**" if _is_wip(r) else ""
            graph_ctx = f"\n   *{' | '.join(r[GraphSearchAugmenter.GRAPH_CONTEXT_KEY])}*" if r.get(GraphSearchAugmenter.GRAPH_CONTEXT_KEY) else ""
            meta = r.get("metadata") or {}
            visible_meta = {k: v for k, v in meta.items() if k not in _INTERNAL_METADATA_KEYS and v}
            meta_line = f"\n   *{' | '.join(f'{k}: {v}' for k, v in visible_meta.items())}*" if visible_meta else ""
            parts.append(
                f"{i}. **{r['title']}**{heading}{wip}{relevance}{date}\n"
                f"   {r.get('url', '')}{breadcrumb}\n"
                f"   {r.get('snippet', '')}{graph_ctx}{meta_line}\n"
                f"   collection: `{r['collection']}` doc_id: `{r['id']}`"
            )
    else:
        for r in results:
            relevance = _format_relevance_band(r)
            date = f" | updated: {_format_date(r['modifiedTime'])}" if r.get("modifiedTime") else ""
            wip = " **[UNDER ARBEID]**" if _is_wip(r) else ""
            header = f"## {r['title']}{wip}{relevance}{date}"
            if r.get("url"):
                header += f"\n{r['url']}"
            if r.get("breadcrumb"):
                header += f"\n{r['breadcrumb']}"
            header += f"\ncollection: `{r['collection']}` doc_id: `{r['id']}`"
            if r.get(GraphSearchAugmenter.GRAPH_CONTEXT_KEY):
                header += f"\n*{' | '.join(r[GraphSearchAugmenter.GRAPH_CONTEXT_KEY])}*"
            chunks = r.get("matchedChunks", [])
            chunk_lines = []
            for chunk in chunks:
                if chunk.get("heading"):
                    chunk_lines.append(f"**{chunk['heading']}**")
                chunk_lines.append(chunk.get("content", ""))
                if chunk.get("metadata"):
                    visible_meta = {k: v for k, v in chunk["metadata"].items() if k not in _INTERNAL_METADATA_KEYS}
                    if visible_meta:
                        meta_str = " | ".join(f"{k}: {v}" for k, v in visible_meta.items())
                        chunk_lines.append(f"*{meta_str}*")
            parts.append(header + "\n\n" + "\n\n".join(chunk_lines))

    return "\n\n".join(parts)


def render_document(doc: dict, doc_id: str) -> str:
    """Render a full document fetched via get_document."""
    title = doc.get("title", doc_id)
    url = doc.get("url", "")
    text = doc.get("text", "")
    metadata = doc.get("metadata") or {}
    is_wip = metadata.get("wip") == "true"

    header = f"# {title}"
    if is_wip:
        header += "\n**[UNDER ARBEID]** Dette dokumentet er merket som under arbeid i Confluence."
    if url:
        header += f"\n{url}"
    visible_meta = {k: v for k, v in metadata.items() if k not in _INTERNAL_METADATA_KEYS and v}
    if visible_meta:
        header += "\n" + " | ".join(f"**{k}:** {v}" for k, v in visible_meta.items())
    return f"{header}\n\n{text}"


def render_notion_page(page: dict) -> str:
    """Render a Notion page fetched via get_notion_page."""
    title = page.get("title", "Untitled")
    url = page.get("url", "")
    content = page.get("content", "")
    source_label = f" (source: {page.get('source', 'live')})" if page.get("source") else ""

    header = f"# {title}{source_label}"
    if url:
        header += f"\n{url}"
    return f"{header}\n\n{content}"


def render_collections(collections: list) -> str:
    """Render the collection list (already filtered to allowed collections)."""
    lines = ["**Loaded collections:**\n"]
    for c in collections:
        lines.append(
            f"- **{c['name']}**: {c.get('document_count', '?')} documents, "
            f"{c.get('embedding_count', '?')} embeddings"
            f" (updated: {c.get('updatedTime', 'unknown')})"
        )
    return "\n".join(lines)


def render_tags(data: dict) -> str:
    """Render tag counts per collection (already filtered to allowed collections)."""
    parts = []
    for coll_name, info in data.items():
        tags = info.get("tags", {})
        if not tags:
            parts.append(f"**{coll_name}**: no tags")
            continue
        parts.append(f"**{coll_name}** ({info.get('unique_tags', 0)} tags):")
        for tag, count in tags.items():
            parts.append(f"  {tag}: {count} docs")
    return "\n".join(parts)


def render_graph_node(data: dict) -> str:
    """Render a graph node and its edges."""
    parts = [f"**{data['label']}** ({data['type']})"]

    props = data.get("properties", {})
    if props:
        prop_lines = [f"  {k}: {v}" for k, v in props.items()]
        parts.append("Properties:\n" + "\n".join(prop_lines))

    outgoing = data.get("outgoing", [])
    if outgoing:
        edge_lines = [f"  --{e['type']}--> {e['target_label']}" for e in outgoing]
        parts.append(f"Outgoing ({len(outgoing)}):\n" + "\n".join(edge_lines))

    incoming = data.get("incoming", [])
    if incoming:
        edge_lines = [f"  <--{e['type']}-- {e['source_label']}" for e in incoming]
        parts.append(f"Incoming ({len(incoming)}):\n" + "\n".join(edge_lines))

    return "\n\n".join(parts)


def render_graph_subtree(data: dict) -> str:
    """Render a graph subtree (nodes + edges + stats)."""
    stats = data.get("stats", {})
    parts = [
        f"**Subtree from {data['root']}** (depth={stats.get('max_depth')}, direction={stats.get('direction')})",
        f"Nodes: {stats.get('node_count')} ({stats.get('by_node_type', {})})",
        f"Edges: {stats.get('edge_count')} ({stats.get('by_edge_type', {})})",
    ]

    nodes_by_id = {n["id"]: n for n in data.get("nodes", [])}
    node_lines = []
    from_excluded_ids = []
    for n in data.get("nodes", []):
        if n["id"] == data["root"]:
            continue
        marker = " [stub: from_excluded]" if n.get("properties", {}).get("from_excluded") else ""
        node_lines.append(f"  {n['id']}: {n['label']}{marker}")
        if marker:
            from_excluded_ids.append(n["id"])
    if node_lines:
        header = "Nodes"
        if from_excluded_ids:
            header += f" ({len(from_excluded_ids)} stub-subtasks enriched from .excluded/)"
        parts.append(f"{header}:\n" + "\n".join(node_lines))

    edge_lines = []
    for e in data.get("edges", []):
        src_label = nodes_by_id.get(e["source"], {}).get("label", e["source"])
        tgt_label = nodes_by_id.get(e["target"], {}).get("label", e["target"])
        edge_lines.append(f"  {src_label} --{e['type']}--> {tgt_label}")
    if edge_lines:
        parts.append("Edges:\n" + "\n".join(edge_lines))

    return "\n\n".join(parts)
