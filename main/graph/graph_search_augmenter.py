"""Graph-aware query expansion and result enrichment.

Wraps a ``KnowledgeGraph`` and exposes the two operations the search endpoint
needs:

- ``augment_query`` — detect entities in the raw query, expand it with
  neighbor terms, and return any direct relational answer.
- ``enrich_results`` — annotate result entries with human-readable graph
  context for entities found in the title.

Both methods no-op cleanly when the wrapped graph is ``None``.
"""

from main.core.search_trace import SearchTrace


class GraphSearchAugmenter:

    EXPANSION_TERM_LIMIT = 5
    CONTEXT_PER_RESULT_LIMIT = 3

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
                r["graph_context"] = contexts[: self.CONTEXT_PER_RESULT_LIMIT]
