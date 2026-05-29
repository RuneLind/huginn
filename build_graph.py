#!/usr/bin/env python3
"""Build knowledge graph from Jira årsavregning issues - complete pipeline."""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import networkx as nx
import itertools

DIR = Path("/Users/rune/source/private/huginn")
ISSUES_DIR = DIR / "data/sources/jira-issues"
OUT_DIR = DIR / "graphify-out-arsavregning"
FM_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)

def parse_fm(fp):
    m = FM_RE.search(Path(fp).read_text())
    if not m:
        return {}
    fm = {}
    for line in m.group(1).split("\n"):
        if ":" in line and not line.strip().startswith("#"):
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm

def parse_tags(s):
    if not s or not str(s).strip():
        return []
    s = str(s)
    if "[" in s:
        return [m.group(1) for m in re.finditer(r'[\-\s]*"([^"]+)"', s)]
    return [t.strip() for t in s.split(",") if t.strip()]

def safe_id(text):
    raw = re.sub(r"[^a-z0-9]+", "_", str(text).lower().strip("_"))
    return raw or f"id_{re.hashlib.md5(str(text).encode()).hexdigest()[:12]}"

print("Step 1: Parsing frontmatter from årsavregning files...")
files = sorted(ISSUES_DIR.glob("*.md"))
ar_files = [f for f in files if "årsavreg" in f.name.lower()]
print(f"  Found {len(ar_files)} files")

issues = []
for fp in ar_files:
    fm = parse_fm(fp)
    if not fm or not fm.get("issue_key"):
        continue
    tag_s = fm.get("tags", "")
    tags_raw = parse_tags(tag_s)
    
    assignee_raw = fm.get("assignee", "")
    reporter_raw = fm.get("reporter", "")
    
    issues.append({
         "key": fm.get("issue_key", ""),
         "title": (fm.get("title", "") or "")[:100],
         "summary": ((fm.get("summary", "") or "")[:300]).replace("\n", " "),
         "type": fm.get("issue_type", "?"),
         "status": fm.get("status", "?"),
         "priority": fm.get("priority", "") or "",
         "epic_link": fm.get("epic_link", ""),
         "epic_summary": (fm.get("epic_summary", "") or "")[:100],
         "assignee": assignee_raw.replace(" [", "").split("]")[0].strip() if "[" in assignee_raw else (assignee_raw.strip() or None),
         "reporter": reporter_raw.replace(" [", "").split("]")[0].strip() if "[" in reporter_raw else (reporter_raw.strip() or None),
         "tags": tags_raw,
         "file": str(fp.relative_to(ISSUES_DIR)),
    })

print(f"  Parsed {len(issues)} issues")

# Step 2: Build graph
print("Step 2: Building graph...")
G = nx.Graph()

for iss in issues:
    nid = f"iss_{safe_id(iss['key'])}"
    
     # Issue node attributes
    attrs = {
         "type": "issue",
         "key": iss["key"],
         "title": iss["title"],
        "status": iss["status"],
         "issue_type": iss["type"],
         "priority": iss["priority"],
         "source_file": iss["file"],
     }
    if iss["summary"]:
        attrs["summary"] = iss["summary"]
    
    G.add_node(nid, **attrs)
    
     # Tag edges
    for t in iss["tags"]:
        tid = f"tag_{safe_id(t)}"
        G.add_node(tid, type="tag", name=f"#{t}")
        G.add_edge(nid, tid, relation="has_tag")
    
     # Epic edges  
    if iss["epic_link"]:
        eid = f"epic_{safe_id(iss['epic_link'])}"
        ep_name = (iss["epic_summary"] or iss["epic_link"])[:100]
        G.add_node(eid, type="epic", name=ep_name)
        G.add_edge(nid, eid, relation="part_of_epic")
    
     # Person edges
    for role_key, edge_rel in [("assignee", "assigned_to"), ("reporter", "reported_by")]:
        v = iss[role_key]
        if v and v != "Unassigned":
            pid = f"person_{safe_id(v)}"
            G.add_node(pid, type="person", name=v, role=edge_rel)
            G.add_edge(nid, pid, relation=edge_rel)

# Intra-epic edges (connect issues within same epic)
epic_issues_map = defaultdict(list)
for iss in issues:
    if iss["epic_link"]:
        eid = f"epic_{safe_id(iss['epic_link'])}"
        iid = f"iss_{safe_id(iss['key'])}"
        epic_issues_map[eid].append(iid)

for eid, iids in epic_issues_map.items():
    for a, b in itertools.combinations(iids[:100], 2):
        G.add_edge(a, b, relation="same_epic")

print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Step 3: Community detection
print("Step 3: Detecting communities...")
components = list(nx.connected_components(G))
com_map = {}
for cid, comp in enumerate(components):
    for n in comp:
        com_map[n] = cid

print(f"  {len(components)} communities (connected components)")

# Step 4: Profile communities
print("Step 4: Profiling communities...")
com_profile = defaultdict(lambda: {
     "nodes": [],
     "issues": [],
     "tags": Counter(),
     "epics": Counter(),
     "statuses": Counter(),
     "issue_types": Counter(),
     "assignees": Counter(),
 })

for n in G.nodes(data=True):
    node_id = n[0]
    attrs = n[1]
    cid = com_map.get(node_id)
    if cid is None:
        continue
    
    prof = com_profile[cid]
    prof["nodes"].append((node_id, attrs))
    
     # Tag neighbor counts
    for nb in G.neighbors(node_id):
        nb_attrs = G.nodes[nb]
        if nb_attrs.get("type") == "tag":
            prof["tags"][nb_attrs.get("name", "")] += 1
    
     # From issue nodes, read epic_link and person data from attributes
    if attrs.get("type") == "issue":
        prof["issues"].append(attrs.get("key", node_id))
        for fk in ("status", "issue_type"):
            v = attrs.get(fk)
            if v:
                (prof["statuses"] if fk == "status" else prof["issue_types"])[v] += 1
        
         for rkey, pname in [("assignee", "assigned_to"), ("reporter", "reported_by")]:
             v = attrs.get(rkey) or ""
             if v and v != "Unassigned":
                 v2 = v.split(" [")[0].strip()
                 prof["assignees"][v2] += 1

# Better community labels from epic summaries grouped by component
com_epic_labels = defaultdict(Counter)
for iss in issues:
    ep_name = (iss["epic_summary"] or iss["key"])[:80]
     # Find which component this issue belongs to
    iid = f"iss_{safe_id(iss['key'])}"
    cid = com_map.get(iid)
    if cid is not None and iss["epic_link"]:
        com_epic_labels[cid][ep_name] += 1

com_labels = {}
for cid in sorted(com_profile.keys()):
    prof = com_profile[cid]
    
     # Primary label: dominant epic or tag theme
    top_epics = com_epic_labels.get(cid, Counter())
    if top_epics:
        com_labels[cid] = f"Epikkeskrue: {list(top_epics.keys())[0]}"
    else:
        prof_top_tags = prof["tags"]
        if prof_top_tags:
            tag_theme = " ".join([f"#{t.replace('#', '')[:20]}" for t, _ in prof_top_tags.most_common(4)])
            com_labels[cid] = tag_theme or f"Komponent #{cid}"
        else:
            com_labels[cid] = f"Løst komponent #{cid}"

print(f"  Labeled {len(com_labels)} communities")

# Step 5: God nodes and surprising connections
print("Step 5: Analyzing graph structure...")
top_degrees = sorted(G.degree(), key=lambda x: -x[1])[:20]

bridges = []
br_coms = Counter()
for u, v in G.edges():
    cu, cv = com_map.get(u), com_map.get(v)
    if cu is not None and cv is not None and cu != cv:
        bridges.append((u, v, cu, cv))
        br_coms[(cu, cv)] += 1

print(f"  Bridge edges between communities: {len(bridges)}")

# Step 6: Write graph.json
print("Step 6: Writing output...")
OUT_DIR.mkdir(parents=True, exist_ok=True)

nodes_out = [{"node_id": n, **{k: v for k, v in d.items()}} for n, d in G.nodes(data=True)]
edges_out = [{"source": u, "target": v, "relation": d.get("relation", "")} for u, v, d in G.edges(data=True)]

graph_data = {
     "meta": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), 
              "communities": len(components), 
               "generated_from": f"{len(issues)} Jira issues (årsavregning)",
              "source_dir": str(ISSUES_DIR)},
    "nodes": nodes_out,
     "edges": edges_out,
}

Path(OUT_DIR / "graph.json").write_text(json.dumps(graph_data, indent=2, ensure_ascii=False))
print(f"  Written graphify-out-arsavregning/graph.json")
