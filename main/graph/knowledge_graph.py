"""
In-memory knowledge graph for domain entities.

Loaded from JSON files produced by extraction scripts. Supports merging
multiple graph files (e.g. EESSI graph + Jira graph). Provides entity
detection, query expansion, context enrichment, and direct graph answers
for relational queries — all without LLM calls.
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal


ENTITY_PREFIX = "entity:"


class KnowledgeGraph:

    def __init__(self, graph_path):
        """Load graph from one or more JSON files (or pre-parsed graph dicts).

        Args:
            graph_path: Single Path or list of Paths to graph JSON files.
                An entry may also be an already-parsed graph dict, or a
                ``(path, data)`` pair — which is what a caller that has read the
                file for other reasons (e.g. the staleness check in graph_loader)
                should pass, so it neither parses twice nor loses the provenance
                the path carries. A bare dict has no provenance and is treated as
                a corpus of its own.
        """
        if (isinstance(graph_path, tuple) and len(graph_path) == 2
                and isinstance(graph_path[0], (str, Path)) and isinstance(graph_path[1], dict)):
            paths = [graph_path]  # a single (path, data) pair, not a two-entry list
        elif isinstance(graph_path, (list, tuple)):
            paths = graph_path
        else:
            paths = [graph_path]

        self.nodes: dict[str, dict] = {}
        self.outgoing: dict[str, list[dict]] = defaultdict(list)
        self.incoming: dict[str, list[dict]] = defaultdict(list)
        # Which graph *file* each node and edge came from, as a bitmask over
        # `self._origins`. The runtime merges every discovered graph into ONE
        # KnowledgeGraph (knowledge_store._load_knowledge_graph), so an entity like
        # entity:folkeregisteret can exist in two unrelated corpora at once. A
        # multi-hop walk that ignored provenance would route through such a node and
        # emit one corpus's facts into another's results. Per file, not per
        # directory: one `<private-sub-repo>/scripts/knowledge_graph/` holds the graphs of
        # two different collections plus two structural graphs, so a directory says
        # nothing about which corpus a fact belongs to. The cost is that a chain
        # cannot cross from a collection's structural graph into its LLM-extracted
        # one; that is the conservative direction. A mask rather than a set of
        # strings because there are as many entries here as nodes plus edges.
        self._origins: list[str] = []
        self._node_origins: dict[str, int] = {}
        self._edge_origins: dict[tuple[str, str, str], int] = {}
        self._degree_cache: dict[str, int] = {}

        seen_edges: set[tuple[str, str, str]] = set()
        for entry_index, entry in enumerate(paths):
            # (path, data) pairs let a caller that has already parsed the file — as
            # graph_loader does for its staleness check — keep the single parse
            # *and* the provenance the path carries.
            if isinstance(entry, tuple):
                path, data = entry
            elif isinstance(entry, dict):
                path, data = None, entry
            else:
                path, data = entry, json.loads(Path(entry).read_text())
            origin = str(Path(path).resolve()) if path is not None else f"dict:{entry_index}"
            if origin not in self._origins:
                self._origins.append(origin)
            origin_bit = 1 << self._origins.index(origin)
            for node in data["nodes"]:
                self._node_origins[node["id"]] = self._node_origins.get(node["id"], 0) | origin_bit
                existing = self.nodes.get(node["id"])
                if existing is None:
                    # Copy so a later file merging into this node never mutates
                    # the source JSON dict (and tolerate a missing properties key).
                    stored = dict(node)
                    stored["properties"] = dict(node.get("properties", {}))
                    self.nodes[node["id"]] = stored
                else:
                    # Merge properties on duplicate nodes; backfill label/type if
                    # the first-seen copy lacked them.
                    existing.setdefault("properties", {}).update(node.get("properties", {}))
                    for key in ("label", "type"):
                        if not existing.get(key) and node.get(key):
                            existing[key] = node[key]
            for edge in data["edges"]:
                # Dedup edges by (source, target, type) so merging files that both
                # carry the same edge doesn't double it in the adjacency lists.
                edge_key = (edge["source"], edge["target"], edge["type"])
                self._edge_origins[edge_key] = self._edge_origins.get(edge_key, 0) | origin_bit
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                self.outgoing[edge["source"]].append(edge)
                self.incoming[edge["target"]].append(edge)

        # Build fast lookup for LLM-extracted entities (entity:* nodes).
        # Only include labels with 3+ chars, and match on word boundaries so a
        # short label can't match inside an unrelated word ("api" in "rapid",
        # "nav" in "navnet", "sed" in "used"). Patterns are compiled once here.
        self._entity_patterns = []
        for node_id, node in self.nodes.items():
            # A node without a label is malformed rather than impossible — graphs are
            # merged from whatever JSON the extractors wrote — and it used to raise
            # here, taking down graph loading for every collection at once.
            label = node.get("label") or ""
            if node_id.startswith(ENTITY_PREFIX) and len(label) >= 3:
                pattern = re.compile(rf'(?<!\w){re.escape(label)}(?!\w)', re.IGNORECASE)
                self._entity_patterns.append((pattern, node_id))

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.outgoing.values())

    # --- Entity detection ---

    def detect_entities(self, text: str, with_spans: bool = False):
        """Detect known graph entities in text.

        Args:
            text: input string to scan.
            with_spans: if True, return list of (node_id, matched_span_text) tuples.
                Default False returns the bare node IDs (existing behavior).

        Returns:
            Deduplicated list of node IDs, or list of (id, span) tuples if with_spans.
        """
        found = []
        spans: dict[str, str] = {}  # node_id → first matched span

        def _add(node_id, span):
            found.append(node_id)
            if node_id not in spans:
                spans[node_id] = span

        # BUC: LA_BUC_02, LA BUC 02, etc.
        for m in re.finditer(r'LA[_ ]?BUC[_ ]?(\d{1,2})', text, re.IGNORECASE):
            node_id = f"buc:LA_BUC_{m.group(1).zfill(2)}"
            if node_id in self.nodes:
                _add(node_id, m.group(0))
        # A-SED: A003, A001, a003
        for m in re.finditer(r'\b(A\d{3})\b', text, re.IGNORECASE):
            node_id = f"sed:{m.group(1).upper()}"
            if node_id in self.nodes:
                _add(node_id, m.group(0))
        # X-SED: X001, X007, x001
        for m in re.finditer(r'\b(X\d{3})\b', text, re.IGNORECASE):
            node_id = f"sed:{m.group(1).upper()}"
            if node_id in self.nodes:
                _add(node_id, m.group(0))
        # Artikkel: artikkel 13, art. 13 nr. 1, art 13.1
        for m in re.finditer(r'art(?:ikkel)?\.?\s*(\d{1,2})(?:\s*(?:nr\.?\s*)?(\d+))?', text, re.IGNORECASE):
            art_num = m.group(1)
            sub_num = m.group(2)
            if sub_num:
                sub_id = f"artikkel:{art_num}.{sub_num}"
                if sub_id in self.nodes:
                    _add(sub_id, m.group(0))
            art_id = f"artikkel:{art_num}"
            if art_id in self.nodes:
                _add(art_id, m.group(0))
        # Forordning: 883/2004, forordning 987/2009
        for m in re.finditer(r'(?:forordning\s+)?(\d{3}/\d{4})\b', text, re.IGNORECASE):
            node_id = f"forordning:{m.group(1)}"
            if node_id in self.nodes:
                _add(node_id, m.group(0))
        # Jira issue keys: PROJECT-1234, TEAM-567, etc.
        for m in re.finditer(r'\b([A-Z][A-Z0-9]+-\d+)\b', text):
            key = m.group(1)
            issue_id = f"issue:{key}"
            epic_id = f"epic:{key}"
            if issue_id in self.nodes:
                _add(issue_id, m.group(0))
            elif epic_id in self.nodes:
                _add(epic_id, m.group(0))
        # LLM-extracted entities: match each label on word boundaries (case-insensitive)
        for pattern, node_id in self._entity_patterns:
            m = pattern.search(text)
            if m:
                _add(node_id, m.group(0))

        deduped = list(dict.fromkeys(found))  # preserve insertion order
        if with_spans:
            return [(node_id, spans[node_id]) for node_id in deduped]
        return deduped

    # --- Query expansion ---

    _EESSI_EXPAND_EDGES = {"inneholder_sed", "hjemlet_i"}

    def get_expansion_terms(self, node_ids: list[str]) -> list[str]:
        """Return search terms from graph neighbors for query expansion.

        Follows 1-hop edges to collect labels and titles of related nodes.
        For Jira epics, includes the epic summary. For issues, includes the
        parent epic label.
        """
        terms = []
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if not node:
                continue
            node_type = node.get("type", "")

            # Add own label (but truncate long Jira labels)
            label = self._label(node_id)
            if node_type in ("Epic", "Issue") and len(label) > 60:
                label = label[:60]
            terms.append(label)

            if node_type == "Epic":
                # Epic → include summary, skip listing all child issues (too many)
                summary = node.get("properties", {}).get("summary", "")
                if summary:
                    terms.append(summary)
            elif node_type == "Issue":
                # Issue → include parent epic label
                for edge in self.outgoing.get(node_id, []):
                    if edge["type"] == "tilhører_epic":
                        epic = self.nodes.get(edge["target"])
                        if epic:
                            terms.append(epic["label"])
                # Issue → include direct cross-references (limited)
                ref_count = 0
                for edge in self.outgoing.get(node_id, []):
                    if edge["type"] == "refererer_til" and ref_count < 3:
                        target = self.nodes.get(edge["target"])
                        if target:
                            terms.append(target["label"][:60])
                            ref_count += 1
            elif node_id.startswith("entity:"):
                # LLM-extracted entities: include labels of neighbors (limited)
                neighbor_count = 0
                for edge in self.outgoing.get(node_id, []):
                    if neighbor_count >= 5:
                        break
                    if edge["target"] in self.nodes:
                        terms.append(self._label(edge["target"]))
                        neighbor_count += 1
                for edge in self.incoming.get(node_id, []):
                    if neighbor_count >= 5:
                        break
                    if edge["source"] in self.nodes:
                        terms.append(self._label(edge["source"]))
                        neighbor_count += 1
            else:
                # EESSI types: BUC → SED, BUC → Artikkel, etc.
                for edge in self.outgoing.get(node_id, []):
                    if edge["type"] in self._EESSI_EXPAND_EDGES:
                        target = self.nodes.get(edge["target"])
                        if target:
                            title = target.get("properties", {}).get("title")
                            terms.append(f"{target['label']} {title}" if title else target["label"])
                for edge in self.incoming.get(node_id, []):
                    if edge["type"] in self._EESSI_EXPAND_EDGES:
                        source = self.nodes.get(edge["source"])
                        if source:
                            terms.append(source["label"])

        return list(dict.fromkeys(terms))

    # --- Bounded multi-hop walk ---

    # A multi-hop walk over the merged entity graph is only useful when it is
    # bounded, because that graph is hub-dominated: over the 5 graph files the
    # runtime merges (13.2k nodes / 20.5k edges) the distinct-neighbour degree runs
    # p50 1, p90 5, p99 24, max 1100. Unbounded, a walk either floods the context
    # from a hub seed or drags every seed back to the same hub.
    WALK_HOPS = 2
    WALK_BEAM = 4  # neighbours taken per node per depth, strongest edge first
    WALK_LIMIT = 8  # total triples returned
    WALK_HUB_DEGREE = 20  # never expand *through* a node this connected (p99; 164 nodes)

    @staticmethod
    def _edge_weight(edge: dict) -> int:
        """``mention_count`` as an int. Graphs are merged from arbitrary JSON, so a
        missing, null or non-numeric value must not take down a search request."""
        raw = (edge.get("properties") or {}).get("mention_count", 0)
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else 0

    def _label(self, node_id: str) -> str:
        """Display label, falling back to the id for a node that lacks one."""
        return self.nodes.get(node_id, {}).get("label") or node_id

    def degree(self, node_id: str) -> int:
        """Distinct neighbours — parallel edges and self-loops counted once.

        Distinct neighbours rather than edge count because that is what the walk
        actually traverses: 328 node pairs in the merged graph are joined by more
        than one typed edge, and counting those separately would read as
        "well-connected" a node the walk sees as having two neighbours.
        """
        cached = self._degree_cache.get(node_id)
        if cached is not None:
            return cached
        neighbours = {e["target"] for e in self.outgoing.get(node_id, [])}
        neighbours |= {e["source"] for e in self.incoming.get(node_id, [])}
        neighbours.discard(node_id)
        # Memoised: the graph is immutable after construction, and this is called
        # once per frontier node per depth inside a per-result enrichment loop.
        self._degree_cache[node_id] = len(neighbours)
        return len(neighbours)

    def _reverse_edge(self, source_id: str, target_id: str, predicate: str) -> dict | None:
        """The same predicate stated the other way round, if the graph carries it."""
        for edge in self.outgoing.get(target_id, []):
            if edge["target"] == source_id and edge["type"] == predicate:
                return edge
        return None

    def _ranked_neighbours(self, node_id: str, direction: str | None = None):
        """``(neighbour_id, edge, (source_id, target_id), direction)``, strongest first.

        ``direction`` restricts traversal to outgoing or incoming edges only.
        Ranking by ``mention_count`` keeps the relationships the corpus states
        repeatedly rather than whichever one was extracted first.
        """
        pairs = []
        if direction in (None, "out"):
            pairs += [(e["target"], e, (node_id, e["target"]), "out")
                      for e in self.outgoing.get(node_id, [])]
        if direction in (None, "in"):
            pairs += [(e["source"], e, (e["source"], node_id), "in")
                      for e in self.incoming.get(node_id, [])]
        pairs.sort(key=lambda p: -self._edge_weight(p[1]))
        return pairs

    # Predicates the extractors write as directed edges but which read the same
    # both ways. Continuing the arrival direction says nothing about these, so a
    # second hop through one is a sibling dressed as a chain.
    SYMMETRIC_PREDICATES = frozenset({"related_to", "alternative_to"})

    def walk_triples(
        self,
        seed_ids: list[str],
        hops: int | None = None,
        beam: int | None = None,
        limit: int | None = None,
        hub_degree: int | None = None,
        min_depth: int = 1,
    ) -> list[dict]:
        """Walk out from ``seed_ids``; return facts as dicts, nearest-first.

        Each dict carries ``source``/``target`` ids, ``source_label``/``target_label``,
        ``predicate`` and ``depth`` (1 = directly on a seed). Deterministic and
        LLM-free — the point is to resolve a chain of facts *before* a model reads
        the results, instead of hoping it chains them across passages itself.

        Four things keep a deeper hop meaningful rather than merely adjacent:

        - **One corpus per chain.** The runtime merges every graph file into one
          graph, so a node can sit in two unrelated corpora. Each branch of the walk
          carries the origins shared by every edge it has crossed, so a chain never
          starts in one corpus and finishes in another. A *seed* in two corpora can
          still walk into either — which is why ``get_entity_context`` declines to
          show a chain for one.
        - **Direction is continued.** Past depth 1 only edges pointing the way we
          arrived are followed, so a depth-2 fact is a real chain
          (``seed -> mid -> next``) rather than one of ``mid``'s siblings, which
          would read as a fact about the seed.
        - **Symmetric predicates stop at depth 1** (``SYMMETRIC_PREDICATES``): a
          branch that *arrived* over one is not expanded, since "continuing the
          direction" of a symmetric edge is not evidence of anything, and such an
          edge is not emitted as a deeper fact either. The predicate vocabulary is
          whatever the extractor's LLM wrote (193 distinct strings in the current
          merge), so this set names the two that dominate rather than all of them.
        - **Hubs are not tunnelled through.** A node above ``hub_degree`` links
          everything, so a chain routed through one says nothing about the seed.
          Naming a hub as a *seed* still expands it — the caller asked for it.

        Facts are de-duplicated per *edge*, not per node, so an edge joining two of
        the seed's own neighbours — often the most informative fact available — is
        still stated, while each node is expanded only once.

        ``min_depth`` drops nearer facts from the result *and* from the ``limit``
        budget, so a caller wanting only the second hop gets a full budget of it.
        """
        hops = self.WALK_HOPS if hops is None else hops
        beam = self.WALK_BEAM if beam is None else beam
        limit = self.WALK_LIMIT if limit is None else limit
        hub_degree = self.WALK_HUB_DEGREE if hub_degree is None else hub_degree
        if limit <= 0 or beam <= 0 or hops <= 0:
            return []

        seed_set = {nid for nid in seed_ids if nid in self.nodes}
        seen = set(seed_set)
        emitted: set[tuple[str, str, str]] = set()
        # (node, direction arrived by, predicate arrived by, origins the branch shares).
        # Sorted: set iteration order varies with PYTHONHASHSEED, and frontier order
        # decides which facts a tight budget keeps — "deterministic" has to survive a
        # restart, not just a rerun.
        frontier = [(nid, None, None, self._node_origins.get(nid, 0)) for nid in sorted(seed_set)]
        triples: list[dict] = []

        for depth in range(1, hops + 1):
            queues = [
                (allowed, iter(self._ranked_neighbours(node_id, arrived_by)))
                for node_id, arrived_by, arrived_via, allowed in frontier
                if depth == 1
                or (self.degree(node_id) <= hub_degree
                    and arrived_via not in self.SYMMETRIC_PREDICATES)
            ]
            next_frontier: list[tuple[str, str | None, str | None, int]] = []
            # Round-robin over the frontier so a tight limit spreads across several
            # neighbours instead of draining the strongest one's whole beam.
            for _ in range(beam):
                for allowed, neighbours in queues:
                    for neighbour_id, edge, (src, tgt), direction in neighbours:
                        if neighbour_id not in self.nodes:
                            continue
                        if depth > 1 and (
                            edge["type"] in self.SYMMETRIC_PREDICATES
                            # A depth-2 edge whose far end is a seed is the depth-1
                            # relation restated, not a chain.
                            or neighbour_id in seed_set
                        ):
                            continue
                        edge_key = (edge["source"], edge["target"], edge["type"])
                        # Dedup on the *unordered* pair: the extractors write 203
                        # reversed duplicates ("A created_by B" and "B created_by A"),
                        # and stating both spends the budget saying one thing twice.
                        fact_key = (min(src, tgt), max(src, tgt), edge["type"])
                        if fact_key in emitted:
                            continue
                        shared = allowed & self._edge_origins.get(edge_key, 0)
                        if not shared:
                            continue
                        emitted.add(fact_key)
                        # Of a reversed duplicate, state the better-attested
                        # direction: past depth 1 only one of the two is even
                        # visible, and a lone backwards claim reads as fact where
                        # the pair used to let the reader discount it.
                        reverse = self._reverse_edge(src, tgt, edge["type"])
                        if reverse is not None and self._edge_weight(reverse) > self._edge_weight(edge):
                            src, tgt = tgt, src
                        if neighbour_id not in seen:
                            seen.add(neighbour_id)
                            next_frontier.append((neighbour_id, direction, edge["type"], shared))
                        if depth >= min_depth:
                            triples.append({
                                "source": src,
                                "source_label": self._label(src),
                                "predicate": edge["type"],
                                "target": tgt,
                                "target_label": self._label(tgt),
                                "depth": depth,
                            })
                            if len(triples) >= limit:
                                return triples
                        break  # this node has used its turn in the round
            frontier = next_frontier

        return triples

    # --- Context enrichment ---

    CONTEXT_CHAIN_LIMIT = 3  # second-hop facts appended to an entity context
    CONTEXT_CHAIN_MIN_RELATIONS = 5  # only a seed with fewer neighbours gets a second hop

    def get_entity_context(self, node_id: str) -> str | None:
        """Return a human-readable context string for a graph entity."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        parts = []
        node_type = node.get("type", "")

        if node_type == "SED":
            title = node.get("properties", {}).get("title", "")
            parent_bucs = [
                self.nodes[e["source"]]["label"]
                for e in self.incoming.get(node_id, [])
                if e["type"] == "inneholder_sed" and e["source"] in self.nodes
            ]
            if title:
                parts.append(f"{node['label']}: {title}")
            if parent_bucs:
                parts.append(f"Del av {', '.join(parent_bucs)}")

        elif node_type == "BUC":
            articles = [
                self.nodes[e["target"]]["label"]
                for e in self.outgoing.get(node_id, [])
                if e["type"] == "hjemlet_i" and e["target"] in self.nodes
            ]
            seds = sorted(
                self.nodes[e["target"]]["label"]
                for e in self.outgoing.get(node_id, [])
                if e["type"] == "inneholder_sed" and e["target"] in self.nodes
            )
            parts.append(node["label"])
            if articles:
                parts.append(f"Hjemlet i {', '.join(articles)} (Forordning 883/2004)")
            if seds:
                parts.append(f"SEDer: {', '.join(seds)}")

        elif node_type == "Artikkel":
            forordning = node.get("properties", {}).get("forordning", "883/2004")
            parts.append(f"{node['label']} (Forordning {forordning})")
            bucs = [
                self.nodes[e["source"]]["label"]
                for e in self.incoming.get(node_id, [])
                if e["type"] == "hjemlet_i" and e["source"] in self.nodes
            ]
            if bucs:
                parts.append(f"Brukes i {', '.join(bucs)}")

        elif node_type == "Epic":
            summary = node.get("properties", {}).get("summary", "")
            issue_count = node.get("properties", {}).get("issue_count", 0)
            parts.append(node["label"])
            if issue_count:
                parts.append(f"{issue_count} issues")

        elif node_type == "Issue":
            parts.append(node["label"])
            # Show parent epic
            for edge in self.outgoing.get(node_id, []):
                if edge["type"] == "tilhører_epic":
                    epic = self.nodes.get(edge["target"])
                    if epic:
                        epic_summary = epic.get("properties", {}).get("summary", "")
                        parts.append(f"Epic: {epic_summary}" if epic_summary else f"Epic: {epic['label']}")
                    break

        elif node_id.startswith("entity:"):
            # LLM-extracted entity
            mentions = node.get("properties", {}).get("mention_count", 0)
            parts.append(f"{self._label(node_id)} ({node_type})")
            # Show key relationships
            related = []
            for edge in self.outgoing.get(node_id, []):
                if edge["target"] in self.nodes:
                    related.append(f"{edge['type']} {self._label(edge['target'])}")
            for edge in self.incoming.get(node_id, []):
                if edge["source"] in self.nodes:
                    related.append(f"{self._label(edge['source'])} {edge['type']}")
            if related:
                parts.append(", ".join(related[:5]))
            # Second hop, but only for a thinly-connected entity. A passage rarely
            # states a whole chain, so for a leaf the second hop is the only place
            # the context says anything beyond a name. A well-connected entity has
            # already spent its budget on the relations above, and its second hop is
            # where the drift lives. Measured over the merged production graphs
            # (13.2k nodes, five files): 26% of entity contexts carry a chain, worth
            # ~18 tokens each. Sampling 400 five-result responses from the indexed
            # document titles of the three LLM-graph collections, that is ~6 tokens
            # per response — median 0, p90 22, max 69, since a response's cost is
            # whether its titles happen to name thin entities.
            # A seed that exists in more than one corpus can chain into either, and
            # the reader has no way to tell which one a fact came from — so it gets
            # the direct relations only.
            single_corpus = self._node_origins.get(node_id, 0).bit_count() == 1
            if single_corpus and self.degree(node_id) < self.CONTEXT_CHAIN_MIN_RELATIONS:
                chain = [
                    f"{t['source_label']} {t['predicate']} {t['target_label']}"
                    for t in self.walk_triples(
                        [node_id], limit=self.CONTEXT_CHAIN_LIMIT, min_depth=2
                    )
                ]
                if chain:
                    parts.append("via: " + ", ".join(chain))
            if mentions > 1:
                parts.append(f"{mentions} mentions")

        return " | ".join(parts) if parts else None

    # --- Graph query answering ---

    def answer_graph_query(self, node_ids: list[str], query: str) -> str | None:
        """Try to answer a relational query directly from the graph.

        Returns formatted answer string, or None if the query isn't relational.
        """
        if not node_ids:
            return None
        q = query.lower()
        words = set(re.findall(r'\w+', q))
        is_question = bool(words & {"hvilke", "which", "what", "hva", "inneholder", "contains", "inngår", "tilhører"})
        if not is_question:
            return None

        wants_seds = bool(words & {"sed", "seder", "seds", "sedene"})
        wants_bucs = bool(words & {"buc", "bucer", "bucs", "bucene"})
        wants_artikkel = bool(words & {"artikkel", "article", "hjemmel", "hjemmelen"})
        wants_issues = bool(words & {"issues", "issue", "oppgaver", "oppgave", "saker", "sak", "tasks"})
        wants_epic = bool(words & {"epic", "epics"})
        # Containment intent toward the node ("hva inneholder X", "hva inngår i X").
        # Used to gate the Epic→issues dump so a bare "hva er X" doesn't trigger it.
        wants_contains = bool(words & {"inneholder", "contains", "inngår", "tilhører"})

        results = []
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if not node:
                continue

            # EESSI: BUC → SEDs
            if wants_seds and node["type"] == "BUC":
                seds = []
                for e in self.outgoing.get(node_id, []):
                    if e["type"] == "inneholder_sed":
                        sed = self.nodes.get(e["target"])
                        if sed:
                            title = sed.get("properties", {}).get("title", "")
                            seds.append(f"- {sed['label']}: {title}" if title else f"- {sed['label']}")
                if seds:
                    results.append(f"**{node['label']}** inneholder disse SEDene:\n" + "\n".join(seds))

            # EESSI: SED → BUCs
            elif wants_bucs and node["type"] == "SED":
                bucs = [
                    self.nodes[e["source"]]["label"]
                    for e in self.incoming.get(node_id, [])
                    if e["type"] == "inneholder_sed" and e["source"] in self.nodes
                ]
                if bucs:
                    results.append(f"**{node['label']}** inngår i: {', '.join(bucs)}")

            # EESSI: BUC → articles
            elif wants_artikkel and node["type"] == "BUC":
                arts = [
                    self.nodes[e["target"]]["label"]
                    for e in self.outgoing.get(node_id, [])
                    if e["type"] == "hjemlet_i" and e["target"] in self.nodes
                ]
                if arts:
                    results.append(f"**{node['label']}** er hjemlet i: {', '.join(arts)}")

            # Jira: Epic → issues. Fire only on explicit issue or containment
            # intent — a bare "hva er <epic>" should fall through to normal RAG
            # rather than dumping every child issue.
            elif (wants_issues or wants_contains) and node["type"] == "Epic":
                issues = []
                for e in self.incoming.get(node_id, []):
                    if e["type"] == "tilhører_epic":
                        issue = self.nodes.get(e["source"])
                        if issue:
                            status = issue.get("properties", {}).get("status", "")
                            label = issue["label"]
                            issues.append(f"- {label} [{status}]" if status else f"- {label}")
                if issues:
                    results.append(f"**{node['label']}** har {len(issues)} issues:\n" + "\n".join(issues[:20]))
                    if len(issues) > 20:
                        results[-1] += f"\n- ... og {len(issues) - 20} til"

            # Jira: Issue → epic
            elif wants_epic and node["type"] == "Issue":
                for e in self.outgoing.get(node_id, []):
                    if e["type"] == "tilhører_epic":
                        epic = self.nodes.get(e["target"])
                        if epic:
                            results.append(f"**{node['label']}** tilhører epic: {epic['label']}")
                        break

        return "\n\n".join(results) if results else None

    # --- Debug/inspection ---

    def get_node_detail(self, node_id: str, edge_types: set[str] | None = None) -> dict | None:
        """Return full node info with all neighbors. For debug endpoint.

        edge_types: optional set of edge type names; when given, only edges of those types
        are included on the returned node.
        """
        node = self.nodes.get(node_id)
        if not node:
            return None
        outgoing = [
            {"target": e["target"], "type": e["type"],
             "target_label": self.nodes.get(e["target"], {}).get("label", "")}
            for e in self.outgoing.get(node_id, [])
            if not edge_types or e["type"] in edge_types
        ]
        incoming = [
            {"source": e["source"], "type": e["type"],
             "source_label": self.nodes.get(e["source"], {}).get("label", "")}
            for e in self.incoming.get(node_id, [])
            if not edge_types or e["type"] in edge_types
        ]
        return {**node, "outgoing": outgoing, "incoming": incoming}

    def get_subtree(
        self,
        root_id: str,
        direction: Literal["incoming", "outgoing", "both"] = "incoming",
        edge_types: set[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 1000,
    ) -> dict | None:
        """BFS subtree from root.

        For an epic with the default args, walks incoming tilhører_epic + er_subtask_av to
        return stories + subtasks in one response.

        direction: "incoming" follows edges TO each frontier node, "outgoing" follows edges
        FROM each frontier node, "both" follows either.

        max_nodes caps the result so a hub node with depth=5/direction="both" can't
        accidentally return the entire graph; stats.truncated is set when the cap is hit.

        Returns live references to graph nodes/edges — callers must not mutate them.

        Returns {root, nodes, edges, stats} or None if root_id is unknown.
        """
        if root_id not in self.nodes:
            return None

        visited_nodes = {root_id}
        visited_edges: set[tuple[str, str, str]] = set()
        out_nodes = [self.nodes[root_id]]
        out_edges: list[dict] = []
        frontier = [root_id]
        truncated = False

        for _ in range(max_depth):
            if truncated:
                break
            next_frontier: list[str] = []
            for node_id in frontier:
                if direction == "incoming":
                    candidate_edges: list[dict] = self.incoming.get(node_id, [])
                elif direction == "outgoing":
                    candidate_edges = self.outgoing.get(node_id, [])
                else:
                    candidate_edges = self.incoming.get(node_id, []) + self.outgoing.get(node_id, [])
                for e in candidate_edges:
                    if edge_types and e["type"] not in edge_types:
                        continue
                    edge_key = (e["source"], e["target"], e["type"])
                    if edge_key in visited_edges:
                        continue
                    other_id = e["source"] if e["target"] == node_id else e["target"]
                    if other_id not in self.nodes:
                        # Dangling edge: references a node absent from the graph.
                        # Skip it so the subtree never emits an edge to a node
                        # that isn't in the returned node set.
                        continue
                    visited_edges.add(edge_key)
                    out_edges.append(e)
                    if other_id not in visited_nodes:
                        visited_nodes.add(other_id)
                        out_nodes.append(self.nodes[other_id])
                        next_frontier.append(other_id)
                        if len(out_nodes) >= max_nodes:
                            truncated = True
                            break
                if truncated:
                    break
            if not next_frontier:
                break
            frontier = next_frontier

        node_types: dict[str, int] = defaultdict(int)
        for n in out_nodes:
            node_types[n.get("type", "Unknown")] += 1
        edge_type_counts: dict[str, int] = defaultdict(int)
        for e in out_edges:
            edge_type_counts[e["type"]] += 1

        return {
            "root": root_id,
            "nodes": out_nodes,
            "edges": out_edges,
            "stats": {
                "node_count": len(out_nodes),
                "edge_count": len(out_edges),
                "max_depth": max_depth,
                "direction": direction,
                "truncated": truncated,
                "by_node_type": dict(node_types),
                "by_edge_type": dict(edge_type_counts),
            },
        }
