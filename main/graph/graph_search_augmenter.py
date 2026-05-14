"""Graph-aware query expansion and result enrichment.

Wraps a ``KnowledgeGraph`` and exposes the operations the search endpoint
needs:

- ``augment_query`` — detect entities in the raw query, expand it with
  neighbor terms, and return any direct relational answer.
- ``enrich_results`` — annotate result entries with human-readable graph
  context for entities found in the title.
- ``get_retry_hints`` — when a search comes back weak, suggest concrete
  follow-ups (related graph terms, a narrower/broader query) so the caller
  can drive a corrective re-query.

All methods no-op (or fall back to graph-free heuristics) when the wrapped
graph is ``None``.
"""

import re

from main.core.search_trace import SearchTrace


_CONJUNCTION_SPLITS = (" versus ", " vs. ", " vs ", " and ", " og ", " & ")
_TRAILING_PARENS_RE = re.compile(r'\s*\([^)]*\)\s*$')
_TOKEN_PUNCT = ".,!?;:\"'()"

# Norwegian + English filler — common enough to drop without losing query meaning,
# small enough to maintain by hand. Used both for picking which word to drop and
# for trimming a trailing stopword left behind.
_STOPWORDS = frozenset({
    # Norwegian
    "og", "men", "på", "til", "for", "i", "av", "en", "et", "er", "var", "ikke",
    "som", "det", "den", "de", "vi", "du", "jeg", "han", "hun", "om", "med",
    "fra", "ved", "kan", "skal", "vil", "har", "hadde", "være", "blir",
    "hva", "hvor", "hvilke", "hvilken", "hvordan", "hvorfor",
    # English
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "by", "is", "was", "are", "were", "be", "been", "being", "has", "have", "had",
    "what", "which", "where", "how", "why", "who", "this", "that", "these",
    "those", "from", "into",
})


def _is_stopword(token: str) -> bool:
    return token.lower().strip(_TOKEN_PUNCT) in _STOPWORDS


def _drop_last_content_word(q: str) -> str | None:
    """Drop the last non-stopword token in ``q``, then trim any trailing stopwords.

    Returns ``None`` for queries shorter than 3 tokens or when nothing meaningful
    remains. Used as the final-fallback widening when structural broadening
    (conjunction / parenthetical / quotes) doesn't apply.
    """
    tokens = q.split()
    if len(tokens) < 3:
        return None
    for i in range(len(tokens) - 1, -1, -1):
        clean = tokens[i].lower().strip(_TOKEN_PUNCT)
        if clean and clean not in _STOPWORDS:
            remaining = tokens[:i] + tokens[i + 1:]
            while remaining and _is_stopword(remaining[-1]):
                remaining = remaining[:-1]
            result = " ".join(remaining).strip()
            return result or None
    # All-stopword query — drop the last token regardless.
    result = " ".join(tokens[:-1]).strip()
    return result or None


def _broaden_query(q: str) -> str | None:
    """Heuristically widen a query: drop a trailing clause / parenthetical / quotes,
    or as a final fallback drop the last content word.

    Returns the broadened query, or ``None`` if no safe broadening applies.
    Purely lexical — no graph needed.
    """
    if not q:
        return None
    stripped = q.strip()
    lower = stripped.lower()
    # Conjunctions: keep the first conjunct ("X and Y" → "X").
    for sep in _CONJUNCTION_SPLITS:
        idx = lower.find(sep)
        if idx > 0:
            head = stripped[:idx].strip()
            if head:
                return head
    # Trailing parenthetical.
    m = _TRAILING_PARENS_RE.search(stripped)
    if m and m.start() > 0:
        return stripped[:m.start()].strip()
    # Quoted phrase → unquote.
    if '"' in stripped:
        unquoted = stripped.replace('"', '').strip()
        if unquoted and unquoted != stripped:
            return unquoted
    # No structural cue — drop the last content word.
    return _drop_last_content_word(stripped)


class GraphSearchAugmenter:

    EXPANSION_TERM_LIMIT = 5
    CONTEXT_PER_RESULT_LIMIT = 3
    RETRY_TERM_LIMIT = 8
    GRAPH_CONTEXT_KEY = "graph_context"

    def __init__(self, graph):
        self.graph = graph

    def augment_query(self, q: str, trace: SearchTrace):
        """Detect entities, build expansion, fetch graph answer.

        Returns ``(search_q, graph_answer, detected_entities)``. When the graph
        is unavailable returns ``(q, None, [])``. Trace recording is gated by
        the trace itself — pass a ``NullSearchTrace`` to skip recording.
        """
        if self.graph is None:
            return q, None, []

        entity_pairs = self.graph.detect_entities(q, with_spans=True)
        detected_entities = [eid for eid, _ in entity_pairs]
        for eid, span in entity_pairs:
            node = self.graph.nodes.get(eid, {})
            trace.add_detected_entity(
                entity_id=eid,
                entity_type=node.get("type", ""),
                label=node.get("label", ""),
                matched_span=span,
            )

        if not detected_entities:
            return q, None, []

        graph_answer = self.graph.answer_graph_query(detected_entities, q)
        trace.set_graph_answered(graph_answer is not None)

        expansion_terms = self.graph.get_expansion_terms(detected_entities)[: self.EXPANSION_TERM_LIMIT]
        if expansion_terms:
            search_q = q + " " + " ".join(expansion_terms)
            trace.set_expansion(search_q, expansion_terms)
        else:
            search_q = q

        return search_q, graph_answer, detected_entities

    def enrich_results(self, results: list, detected_entities: list) -> None:
        """Annotate each result with ``graph_context`` for entities found in its title.

        Mutates ``results`` in place. No-op if the graph is unavailable or no
        entities were detected in the query.
        """
        if self.graph is None or not detected_entities:
            return
        for r in results:
            result_entities = self.graph.detect_entities(r.get("title", ""))
            contexts = []
            for eid in result_entities:
                ctx = self.graph.get_entity_context(eid)
                if ctx:
                    contexts.append(ctx)
            if contexts:
                r[self.GRAPH_CONTEXT_KEY] = contexts[: self.CONTEXT_PER_RESULT_LIMIT]

    def get_retry_hints(self, q: str, detected_entities: list) -> dict | None:
        """Suggest concrete follow-ups for a weak/empty search.

        Returns a dict with any of: ``detectedEntities`` (entity labels found
        in the query), ``relatedTerms`` (graph-neighbour terms not already in
        the query), ``narrowerQuery`` (query + the top entity label),
        ``broaderQuery`` (a lexical widening). Returns ``None`` when nothing
        useful can be offered. Cheap — graph lookups only, no search.
        """
        hints: dict = {}
        q_lower = (q or "").lower()

        entity_labels: list[str] = []
        if self.graph is not None:
            for eid in detected_entities:
                node = self.graph.nodes.get(eid)
                label = node.get("label") if node else None
                if label and label not in entity_labels:
                    entity_labels.append(label)
        if entity_labels:
            hints["detectedEntities"] = entity_labels

        if self.graph is not None and detected_entities:
            related: list[str] = []
            for term in self.graph.get_expansion_terms(detected_entities):
                if not term:
                    continue
                if term.lower() in q_lower or term in related:
                    continue
                related.append(term)
                if len(related) >= self.RETRY_TERM_LIMIT:
                    break
            if related:
                hints["relatedTerms"] = related

        if entity_labels:
            top = entity_labels[0]
            if top.lower() not in q_lower:
                hints["narrowerQuery"] = f"{q} {top}".strip()
        else:
            # No entity matched the query — fall back to token-overlap against
            # graph labels and seed a narrower with the top match's neighbour.
            seed = self._fallback_narrower_seed(q)
            if seed and seed.lower() not in q_lower:
                hints["narrowerQuery"] = f"{q} {seed}".strip()

        broader = _broaden_query(q)
        if broader and broader.lower() != q_lower:
            hints["broaderQuery"] = broader

        return hints or None

    def _fallback_narrower_seed(self, q: str) -> str | None:
        """Token-overlap scan of graph labels when no entity was detected.

        Tokenise the query (≥3 chars, stopwords dropped), find the node whose
        label shares the most tokens (ties broken by ``mention_count``), and
        return that node's top neighbour's label as a narrowing seed. Returns
        ``None`` when there's no usable signal — typical for short or
        all-stopword queries. O(N) over the graph; cheap enough on the
        weak-result path.
        """
        if self.graph is None:
            return None
        query_tokens = {
            t.lower().strip(_TOKEN_PUNCT)
            for t in (q or "").split()
        }
        query_tokens = {t for t in query_tokens if t and len(t) >= 3 and t not in _STOPWORDS}
        if not query_tokens:
            return None

        best_node_id = None
        best_score = -1
        for node_id, node in self.graph.nodes.items():
            label = node.get("label", "") or ""
            if not label:
                continue
            label_tokens = {tok.strip(_TOKEN_PUNCT) for tok in label.lower().split()}
            overlap = query_tokens & label_tokens
            if not overlap:
                continue
            mentions = node.get("properties", {}).get("mention_count", 1)
            score = len(overlap) * 1000 + mentions
            if score > best_score:
                best_score = score
                best_node_id = node_id
        if best_node_id is None:
            return None

        # Walk one hop out — prefer outgoing, fall back to incoming.
        for edges_attr, side_key in (("outgoing", "target"), ("incoming", "source")):
            for edge in getattr(self.graph, edges_attr).get(best_node_id, []):
                neighbour = self.graph.nodes.get(edge.get(side_key))
                if neighbour and neighbour.get("label"):
                    return neighbour["label"]
        # No neighbour — fall back to the matched node's own label.
        matched = self.graph.nodes.get(best_node_id)
        return matched.get("label") if matched else None
