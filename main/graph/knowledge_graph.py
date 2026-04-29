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


ENTITY_PREFIX = "entity:"


class KnowledgeGraph:

    def __init__(self, graph_path):
        """Load graph from one or more JSON files.

        Args:
            graph_path: Single Path or list of Paths to graph JSON files.
        """
        if isinstance(graph_path, (list, tuple)):
            paths = graph_path
        else:
            paths = [graph_path]

        self.nodes: dict[str, dict] = {}
        self.outgoing: dict[str, list[dict]] = defaultdict(list)
        self.incoming: dict[str, list[dict]] = defaultdict(list)

        for path in paths:
            data = json.loads(Path(path).read_text())
            for node in data["nodes"]:
                if node["id"] in self.nodes:
                    # Merge properties on duplicate nodes
                    self.nodes[node["id"]]["properties"].update(node.get("properties", {}))
                else:
                    self.nodes[node["id"]] = node
            for edge in data["edges"]:
                self.outgoing[edge["source"]].append(edge)
                self.incoming[edge["target"]].append(edge)

        # Build fast lookup for LLM-extracted entities (entity:* nodes)
        # Only include labels with 3+ chars to avoid false positives
        self._entity_patterns = []
        for node_id, node in self.nodes.items():
            if node_id.startswith(ENTITY_PREFIX) and len(node["label"]) >= 3:
                self._entity_patterns.append((node["label"].lower(), node_id))

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
        # LLM-extracted entities: match by label (case-insensitive word boundary)
        if self._entity_patterns:
            text_lower = text.lower()
            for label_lower, node_id in self._entity_patterns:
                if label_lower in text_lower:
                    _add(node_id, label_lower)

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
            node_type = node["type"]

            # Add own label (but truncate long Jira labels)
            label = node["label"]
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
                    target = self.nodes.get(edge["target"])
                    if target:
                        terms.append(target["label"])
                        neighbor_count += 1
                for edge in self.incoming.get(node_id, []):
                    if neighbor_count >= 5:
                        break
                    source = self.nodes.get(edge["source"])
                    if source:
                        terms.append(source["label"])
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

    # --- Context enrichment ---

    def get_entity_context(self, node_id: str) -> str | None:
        """Return a human-readable context string for a graph entity."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        parts = []
        node_type = node["type"]

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
            parts.append(f"{node['label']} ({node_type})")
            # Show key relationships
            related = []
            for edge in self.outgoing.get(node_id, []):
                target = self.nodes.get(edge["target"])
                if target:
                    related.append(f"{edge['type']} {target['label']}")
            for edge in self.incoming.get(node_id, []):
                source = self.nodes.get(edge["source"])
                if source:
                    related.append(f"{source['label']} {edge['type']}")
            if related:
                parts.append(", ".join(related[:5]))
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

            # Jira: Epic → issues
            elif (wants_issues or not any([wants_seds, wants_bucs, wants_artikkel, wants_epic])) and node["type"] == "Epic":
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

    def get_node_detail(self, node_id: str) -> dict | None:
        """Return full node info with all neighbors. For debug endpoint."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        outgoing = [
            {"target": e["target"], "type": e["type"],
             "target_label": self.nodes.get(e["target"], {}).get("label", "")}
            for e in self.outgoing.get(node_id, [])
        ]
        incoming = [
            {"source": e["source"], "type": e["type"],
             "source_label": self.nodes.get(e["source"], {}).get("label", "")}
            for e in self.incoming.get(node_id, [])
        ]
        return {**node, "outgoing": outgoing, "incoming": incoming}
