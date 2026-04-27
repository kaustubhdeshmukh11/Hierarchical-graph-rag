"""
visualize.py -- Full 3-Layer Hierarchical Graph Visualization

Shows Communities -> Concepts -> Entities in one clear hierarchical view.
  - Layer 3 (Communities): Large diamond nodes at top
  - Layer 2 (Concepts): Medium hexagon nodes in middle  
  - Layer 1 (Entities): Small box nodes at bottom
  - Edges: CONTAINS, INSTANTIATED_BY, and entity discourse relationships

Usage:
    python visualize.py
"""

import os
import sys
import textwrap
import webbrowser

try:
    from pyvis.network import Network
except ImportError:
    print("ERROR: pyvis not installed. Run: pip install pyvis")
    sys.exit(1)

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j not installed. Run: pip install neo4j")
    sys.exit(1)

import config


# =============================================================================
#  NEO4J FETCH
# =============================================================================

def fetch_graph_data(driver) -> dict:
    data = {
        "communities": [], "concepts": [], "entities": [],
        "community_concept_edges": [], "concept_entity_edges": [],
        "entity_entity_edges": [],
    }
    with driver.session() as session:
        for r in session.run(
            "MATCH (cm:Community) RETURN cm.id AS id, cm.name AS name, cm.summary AS summary"
        ):
            data["communities"].append({
                "id": r["id"] or "", "name": r["name"] or "Community",
                "summary": r["summary"] or "",
            })
        for r in session.run(
            "MATCH (co:Concept) RETURN co.name AS name, co.description AS description"
        ):
            data["concepts"].append({
                "name": r["name"] or "", "description": r["description"] or "",
            })
        for r in session.run(
            "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.description AS description"
        ):
            data["entities"].append({
                "name": r["name"] or "", "type": r["type"] or "OTHER",
                "description": r["description"] or "",
            })
        for r in session.run(
            "MATCH (cm:Community)-[:CONTAINS]->(co:Concept) "
            "RETURN cm.id AS comm_id, cm.name AS comm_name, co.name AS concept_name"
        ):
            data["community_concept_edges"].append({
                "from": r["comm_id"] or "", "from_name": r["comm_name"] or "",
                "to": r["concept_name"] or "",
            })
        for r in session.run(
            "MATCH (co:Concept)-[:INSTANTIATED_BY]->(e:Entity) "
            "RETURN co.name AS concept_name, e.name AS entity_name"
        ):
            data["concept_entity_edges"].append({
                "from": r["concept_name"] or "", "to": r["entity_name"] or ""
            })
        for r in session.run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "RETURN a.name AS src, b.name AS tgt, type(r) AS rel_type, r.evidence AS evidence"
        ):
            data["entity_entity_edges"].append({
                "from": r["src"] or "", "to": r["tgt"] or "",
                "type": r["rel_type"] or "RELATED_TO",
                "evidence": r["evidence"] or "",
            })
    return data


# =============================================================================
#  COLORS
# =============================================================================

ENTITY_COLORS = {
    "PERSON":       {"bg": "#e8f5e9", "border": "#2e7d32", "text": "#1b5e20"},
    "TECHNOLOGY":   {"bg": "#e0f2f1", "border": "#00796b", "text": "#004d40"},
    "CONCEPT":      {"bg": "#fff3e0", "border": "#ef6c00", "text": "#e65100"},
    "ORGANIZATION": {"bg": "#e3f2fd", "border": "#1565c0", "text": "#0d47a1"},
    "ORG":          {"bg": "#e3f2fd", "border": "#1565c0", "text": "#0d47a1"},
    "PRODUCT":      {"bg": "#fce4ec", "border": "#ad1457", "text": "#880e4f"},
    "DEPARTMENT":   {"bg": "#f1f8e9", "border": "#558b2f", "text": "#33691e"},
    "PLACE":        {"bg": "#fce4ec", "border": "#c62828", "text": "#b71c1c"},
    "EVENT":        {"bg": "#f3e5f5", "border": "#7b1fa2", "text": "#4a148c"},
    "METRIC":       {"bg": "#e8eaf6", "border": "#283593", "text": "#1a237e"},
    "OTHER":        {"bg": "#f5f5f5", "border": "#616161", "text": "#424242"},
}

COMMUNITY_COLOR = {"bg": "#1a237e", "border": "#0d47a1", "text": "#ffffff"}
CONCEPT_COLOR   = {"bg": "#ff6f00", "border": "#e65100", "text": "#ffffff"}


# =============================================================================
#  BUILD FULL HIERARCHICAL NETWORK
# =============================================================================

def fmt_rel(rel: str) -> str:
    return rel.replace("_", " ").lower()


def build_hierarchical_network(data: dict) -> Network:
    """Build a full 3-layer hierarchical visualization."""
    net = Network(
        height="100vh", width="100%", bgcolor="#fafafa",
        font_color="#212529", directed=True, notebook=False,
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "hierarchicalRepulsion": {
          "centralGravity": 0.2,
          "springLength": 180,
          "springConstant": 0.02,
          "nodeDistance": 160,
          "damping": 0.09,
          "avoidOverlap": 0.5
        },
        "solver": "hierarchicalRepulsion",
        "stabilization": { "enabled": true, "iterations": 2000, "fit": true }
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "hubsize",
          "levelSeparation": 220,
          "nodeSpacing": 180,
          "treeSpacing": 250,
          "blockShifting": true,
          "edgeMinimization": true,
          "parentCentralization": true
        }
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "smooth": { "enabled": true, "type": "cubicBezier", "roundness": 0.5 },
        "font": { "size": 10, "align": "middle", "background": "#ffffff",
                  "strokeWidth": 0, "face": "Segoe UI, Arial, sans-serif" }
      },
      "nodes": {
        "font": { "size": 13, "face": "Segoe UI, Arial, sans-serif",
                  "strokeWidth": 2, "strokeColor": "#ffffff" }
      },
      "interaction": {
        "hover": true, "tooltipDelay": 80, "navigationButtons": true,
        "keyboard": true, "zoomView": true
      }
    }
    """)

    added = set()

    # ── Layer 3: Community nodes (top level) ──────────────────────────
    for comm in data["communities"]:
        nid = f"comm_{comm['id']}"
        if nid in added:
            continue
        added.add(nid)
        summary_short = comm["summary"][:200] + "..." if len(comm["summary"]) > 200 else comm["summary"]
        tooltip = (f"<b style='color:#bbdefb'>COMMUNITY (Layer 3)</b><br><br>"
                   f"<b>{comm['name']}</b><br><br>{summary_short}")
        net.add_node(
            nid, label=f"🏛 {comm['name']}", title=tooltip,
            color={"background": COMMUNITY_COLOR["bg"], "border": COMMUNITY_COLOR["border"],
                   "highlight": {"background": "#283593", "border": "#1565c0"},
                   "hover": {"background": "#283593", "border": "#1565c0"}},
            shape="diamond", size=45, level=0,
            font={"size": 16, "color": COMMUNITY_COLOR["text"], "bold": True,
                  "strokeWidth": 0},
            borderWidth=3, borderWidthSelected=5, margin=12,
        )

    # ── Layer 2: Concept nodes (middle level) ─────────────────────────
    for concept in data["concepts"]:
        nid = f"concept_{concept['name']}"
        if nid in added:
            continue
        added.add(nid)
        tooltip = (f"<b style='color:#ffe0b2'>CONCEPT (Layer 2)</b><br><br>"
                   f"<b>{concept['name']}</b><br><br>{concept['description']}")
        net.add_node(
            nid, label=f"◆ {concept['name']}", title=tooltip,
            color={"background": CONCEPT_COLOR["bg"], "border": CONCEPT_COLOR["border"],
                   "highlight": {"background": "#ff8f00", "border": "#e65100"},
                   "hover": {"background": "#ff8f00", "border": "#e65100"}},
            shape="box", size=30, level=1,
            font={"size": 13, "color": CONCEPT_COLOR["text"], "bold": True,
                  "strokeWidth": 0},
            borderWidth=2, borderWidthSelected=4, margin=10,
        )

    # ── Layer 1: Entity nodes (bottom level) ──────────────────────────
    for entity in data["entities"]:
        nid = f"entity_{entity['name']}"
        if nid in added:
            continue
        added.add(nid)
        etype = entity["type"]
        c = ENTITY_COLORS.get(etype, ENTITY_COLORS["OTHER"])
        tooltip = (f"<b style='color:{c['border']}'>ENTITY: {etype} (Layer 1)</b>"
                   f"<br><br><b>{entity['name']}</b><br><br>{entity['description']}")
        net.add_node(
            nid, label=entity["name"], title=tooltip,
            color={"background": c["bg"], "border": c["border"],
                   "highlight": {"background": "#fff9c4", "border": c["border"]},
                   "hover": {"background": "#fff9c4", "border": c["border"]}},
            shape="box", size=18, level=2,
            font={"size": 11, "color": c["text"], "strokeWidth": 2, "strokeColor": c["bg"]},
            borderWidth=2, borderWidthSelected=3, margin=6,
        )

    # ── CONTAINS edges: Community -> Concept ──────────────────────────
    for edge in data["community_concept_edges"]:
        src = f"comm_{edge['from']}"
        tgt = f"concept_{edge['to']}"
        if src in added and tgt in added:
            net.add_edge(
                src, tgt, label="contains",
                color={"color": "#5c6bc0", "highlight": "#3949ab", "hover": "#3949ab"},
                width=3, dashes=False,
                font={"size": 10, "color": "#3949ab", "background": "#ffffff"},
            )

    # ── INSTANTIATED_BY edges: Concept -> Entity ──────────────────────
    for edge in data["concept_entity_edges"]:
        src = f"concept_{edge['from']}"
        tgt = f"entity_{edge['to']}"
        if src in added and tgt in added:
            net.add_edge(
                src, tgt, label="has member",
                color={"color": "#ff8f00", "highlight": "#e65100", "hover": "#e65100"},
                width=2, dashes=[5, 5],
                font={"size": 9, "color": "#e65100", "background": "#ffffff"},
            )

    # ── Discourse edges: Entity -> Entity ─────────────────────────────
    for edge in data["entity_entity_edges"]:
        src = f"entity_{edge['from']}"
        tgt = f"entity_{edge['to']}"
        if src in added and tgt in added:
            rel_label = fmt_rel(edge["type"])
            evidence = edge.get("evidence", "")
            tooltip = f"<b>{rel_label}</b>"
            if evidence:
                tooltip += f"<br><br><i>Evidence: {evidence[:200]}</i>"
            net.add_edge(
                src, tgt, label=rel_label, title=tooltip,
                color={"color": "#90a4ae", "highlight": "#546e7a", "hover": "#546e7a"},
                width=1.5,
                font={"size": 9, "color": "#546e7a", "background": "#ffffff"},
            )

    return net


# =============================================================================
#  LEGEND OVERLAY
# =============================================================================

def build_overlay_html(data: dict) -> str:
    n_comm = len(data["communities"])
    n_conc = len(data["concepts"])
    n_ent = len(data["entities"])
    n_ee = len(data["entity_entity_edges"])

    etypes = sorted(set(e["type"] for e in data["entities"]))
    type_items = ""
    for etype in etypes:
        c = ENTITY_COLORS.get(etype, ENTITY_COLORS["OTHER"])
        count = sum(1 for e in data["entities"] if e["type"] == etype)
        type_items += f"""
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
      <div style="width:12px;height:12px;border-radius:2px;background:{c['bg']};
                  border:2px solid {c['border']};flex-shrink:0;"></div>
      <div style="font-size:10px;color:{c['text']};font-weight:600;">{etype}
        <span style="color:#94a3b8;font-weight:400;">({count})</span></div>
    </div>"""

    return f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<div id="legend" style="
  position:fixed;top:16px;left:16px;background:#ffffff;border:1.5px solid #e2e8f0;
  border-radius:14px;padding:16px 20px;font-family:'Inter','Segoe UI',sans-serif;
  color:#1e293b;z-index:9999;min-width:240px;max-width:300px;
  box-shadow:0 4px 24px rgba(0,0,0,0.10);max-height:90vh;overflow-y:auto;">

  <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:2px;">
    Hierarchical Graph RAG</div>
  <div style="font-size:11px;color:#64748b;margin-bottom:12px;">
    Meridian Health Systems &middot; 3-Layer Architecture</div>

  <div style="font-size:10px;font-weight:700;color:#0f172a;margin-bottom:6px;
              letter-spacing:.5px;text-transform:uppercase;">Graph Layers</div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
    <div style="width:16px;height:16px;background:#1a237e;transform:rotate(45deg);flex-shrink:0;"></div>
    <div style="font-size:11px;color:#1a237e;font-weight:600;">Communities (L3)
      <span style="color:#94a3b8;font-weight:400;">({n_comm})</span></div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
    <div style="width:14px;height:14px;background:#ff6f00;border-radius:2px;flex-shrink:0;"></div>
    <div style="font-size:11px;color:#e65100;font-weight:600;">Concepts (L2)
      <span style="color:#94a3b8;font-weight:400;">({n_conc})</span></div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
    <div style="width:14px;height:14px;background:#e3f2fd;border:2px solid #1565c0;
                border-radius:2px;flex-shrink:0;"></div>
    <div style="font-size:11px;color:#0d47a1;font-weight:600;">Entities (L1)
      <span style="color:#94a3b8;font-weight:400;">({n_ent})</span></div>
  </div>

  <div style="font-size:10px;font-weight:700;color:#0f172a;margin-bottom:6px;
              letter-spacing:.5px;text-transform:uppercase;">Entity Types</div>
  {type_items}

  <div style="font-size:10px;font-weight:700;color:#0f172a;margin-top:10px;margin-bottom:6px;
              letter-spacing:.5px;text-transform:uppercase;">Edge Types</div>
  <div style="font-size:10px;color:#5c6bc0;margin-bottom:2px;">━━ contains (L3→L2)</div>
  <div style="font-size:10px;color:#ff8f00;margin-bottom:2px;">╌╌ has member (L2→L1)</div>
  <div style="font-size:10px;color:#90a4ae;margin-bottom:2px;">── discourse (L1→L1)
    <span style="color:#94a3b8;">({n_ee})</span></div>

  <div style="font-size:9px;color:#94a3b8;border-top:1px solid #e2e8f0;
              padding-top:8px;margin-top:10px;line-height:1.6;">
    Hover for details &middot; Scroll to zoom<br>
    Drag to pan &middot; Click to highlight</div>
</div>

<div style="position:fixed;top:16px;right:16px;background:#ffffff;border:1.5px solid #e2e8f0;
  border-radius:14px;padding:14px 20px;font-family:'Inter','Segoe UI',sans-serif;
  color:#1e293b;z-index:9999;text-align:right;box-shadow:0 4px 24px rgba(0,0,0,0.10);">
  <div style="font-size:16px;font-weight:700;color:#0f172a;">Hierarchical Graph RAG</div>
  <div style="font-size:11px;color:#64748b;margin-top:3px;">
    Community &rarr; Concept &rarr; Entity</div>
  <div style="font-size:10px;color:#94a3b8;margin-top:4px;">
    Top-down hierarchical layout</div>
</div>
"""


def inject_overlay(html_path: str, overlay_html: str):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace('body {', 'body { background: #fafafa !important;')
    html = html.replace("</body>", overlay_html + "\n</body>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# =============================================================================
#  LAYER 2/3 TEXT HIERARCHY (kept for reference)
# =============================================================================

def build_hierarchy_text(data: dict) -> str:
    lines = []
    W = 76
    lines.append("+" + "=" * W + "+")
    lines.append("|  HIERARCHICAL GRAPH STRUCTURE — LAYERS 2 & 3" + " " * (W - 46) + "|")
    lines.append("+" + "=" * W + "+")
    lines.append("")
    lines.append("  Layer 3 (Communities) = Topic clusters with auto-generated summaries")
    lines.append("  Layer 2 (Concepts)    = Abstract themes grouping related entities")
    lines.append("  Layer 1 (Entities)    = See graph_visualization.html for the visual graph")
    lines.append("")
    lines.append("=" * (W + 2))

    comm_concepts = {}
    for edge in data["community_concept_edges"]:
        cid = edge["from"]
        if cid not in comm_concepts:
            comm_concepts[cid] = {"name": edge.get("from_name", ""), "concepts": []}
        comm_concepts[cid]["concepts"].append(edge["to"])

    concept_entities = {}
    for edge in data["concept_entity_edges"]:
        cname = edge["from"]
        if cname not in concept_entities:
            concept_entities[cname] = []
        concept_entities[cname].append(edge["to"])

    concept_desc = {c["name"]: c["description"] for c in data["concepts"]}
    lines.append("")

    for comm in data["communities"]:
        cid = comm["id"]
        lines.append("=" * (W + 2))
        lines.append(f"  COMMUNITY: {comm['name']}")
        lines.append("=" * (W + 2))
        lines.append("")
        summary_lines = textwrap.fill(comm["summary"], width=W - 4,
                                       initial_indent="    ", subsequent_indent="    ")
        lines.append("  Summary:")
        lines.append(summary_lines)
        lines.append("")
        concept_names = comm_concepts.get(cid, {}).get("concepts", [])
        if concept_names:
            lines.append("  Concepts in this community:")
            lines.append("  " + "-" * 40)
            for i, cn in enumerate(concept_names, 1):
                desc = concept_desc.get(cn, "")
                lines.append(f"    {i}. {cn}")
                if desc:
                    lines.append(textwrap.fill(desc, width=W - 10,
                                               initial_indent="       ", subsequent_indent="       "))
                entities = concept_entities.get(cn, [])
                if entities:
                    lines.append(textwrap.fill(f"Entities: {', '.join(entities)}",
                                               width=W - 10,
                                               initial_indent="       ",
                                               subsequent_indent="                "))
                lines.append("")

    lines.append("")
    lines.append("=" * (W + 2))
    lines.append("  SUMMARY STATISTICS")
    lines.append("=" * (W + 2))
    lines.append(f"    Communities:  {len(data['communities'])}")
    lines.append(f"    Concepts:    {len(data['concepts'])}")
    lines.append(f"    Entities:    {len(data['entities'])}")
    lines.append(f"    Discourse edges: {len(data['entity_entity_edges'])}")
    lines.append("")
    lines.append("  For the visual graph, open: graph_visualization.html")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Hierarchical Graph RAG -- Visualization")
    print("=" * 60)

    print("\n[1/4] Connecting to Neo4j...")
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        print("      Connected.")
        print("[2/4] Fetching graph data from Neo4j...")
        data = fetch_graph_data(driver)
    finally:
        driver.close()

    n_comm = len(data["communities"])
    n_conc = len(data["concepts"])
    n_ent  = len(data["entities"])
    n_ee   = len(data["entity_entity_edges"])

    print(f"""
  +-------------------------------------+
  |  Graph Statistics                   |
  +-------------------------------------+
  |  Layer 3 -- Communities : {n_comm:>4}       |
  |  Layer 2 -- Concepts   : {n_conc:>4}       |
  |  Layer 1 -- Entities   : {n_ent:>4}       |
  +-------------------------------------+
  |  Discourse edges       : {n_ee:>4}       |
  +-------------------------------------+""")

    if n_ent == 0:
        print("\n  Graph is empty -- run graph_builder.py first.")
        return

    # ── Output 1: Full hierarchical visualization (HTML) ──────────────
    print("\n[3/4] Building full 3-layer hierarchical visualization...")
    net = build_hierarchical_network(data)

    out_html = os.path.join(os.path.dirname(__file__), "graph_visualization.html")
    net.save_graph(out_html)
    overlay = build_overlay_html(data)
    inject_overlay(out_html, overlay)
    print(f"      Saved: {out_html}")

    # ── Output 2: Layer 2/3 text hierarchy ───────────────────────────
    print("[4/4] Building text hierarchy...")
    hierarchy_text = build_hierarchy_text(data)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_txt = os.path.join(config.RESULTS_DIR, "graph_hierarchy.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(hierarchy_text)
    print(f"      Saved: {out_txt}")

    print("\n  Opening graph in browser...")
    webbrowser.open(f"file:///{out_html.replace(os.sep, '/')}")

    print(f"""
{'=' * 60}
  Outputs:
    1. {out_html}
       -> Full 3-layer hierarchy: Communities -> Concepts -> Entities
       -> Hover nodes for details, edges for evidence

    2. {out_txt}
       -> Text tree: Communities with summaries and member entities
{'=' * 60}""")


if __name__ == "__main__":
    main()
