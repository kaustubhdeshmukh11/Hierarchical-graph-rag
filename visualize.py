"""
visualize.py -- Clean Layer 1 Entity Graph + Layer 2/3 Text Hierarchy

Two outputs:
  1. graph_visualization.html  -- Layer 1 ONLY: entities as nodes, discourse
     relationships as labeled edges. Clean, uncluttered, perfect for presentations.
  2. results/graph_hierarchy.txt  -- Layer 2 (Concepts) and Layer 3 (Communities)
     in a readable text tree format.

Separating layers avoids the clutter of showing all 3 in one visualization.

Usage:
    python visualize.py
"""

import os
import sys
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
        "communities": [],
        "concepts": [],
        "entities": [],
        "community_concept_edges": [],
        "concept_entity_edges": [],
        "entity_entity_edges": [],
    }

    with driver.session() as session:
        for r in session.run(
            "MATCH (cm:Community) RETURN cm.id AS id, cm.name AS name, cm.summary AS summary"
        ):
            data["communities"].append({
                "id": r["id"] or "",
                "name": r["name"] or "Community",
                "summary": r["summary"] or "",
            })

        for r in session.run(
            "MATCH (co:Concept) RETURN co.name AS name, co.description AS description"
        ):
            data["concepts"].append({
                "name": r["name"] or "",
                "description": r["description"] or "",
            })

        for r in session.run(
            "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.description AS description"
        ):
            data["entities"].append({
                "name": r["name"] or "",
                "type": r["type"] or "OTHER",
                "description": r["description"] or "",
            })

        for r in session.run(
            "MATCH (cm:Community)-[:CONTAINS]->(co:Concept) "
            "RETURN cm.id AS comm_id, cm.name AS comm_name, co.name AS concept_name"
        ):
            data["community_concept_edges"].append({
                "from": r["comm_id"] or "",
                "from_name": r["comm_name"] or "",
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
                "from": r["src"] or "",
                "to":   r["tgt"] or "",
                "type": r["rel_type"] or "RELATED_TO",
                "evidence": r["evidence"] or "",
            })

    return data


# =============================================================================
#  BUILD LAYER 1 NETWORK (Entities + Discourse Edges ONLY)
# =============================================================================

def fmt_rel(rel: str) -> str:
    """Pretty-print a snake_case relationship type."""
    return rel.replace("_", " ").lower()


# Entity type colours — lighter, muted palette for clean look
ENTITY_COLORS = {
    "PERSON":       {"bg": "#e8f5e9", "border": "#2e7d32", "text": "#1b5e20"},
    "TECHNOLOGY":   {"bg": "#e0f2f1", "border": "#00796b", "text": "#004d40"},
    "CONCEPT":      {"bg": "#fff3e0", "border": "#ef6c00", "text": "#e65100"},
    "ORGANIZATION": {"bg": "#e3f2fd", "border": "#1565c0", "text": "#0d47a1"},
    "ORG":          {"bg": "#e3f2fd", "border": "#1565c0", "text": "#0d47a1"},
    "PLACE":        {"bg": "#fce4ec", "border": "#c62828", "text": "#b71c1c"},
    "EVENT":        {"bg": "#f3e5f5", "border": "#7b1fa2", "text": "#4a148c"},
    "METRIC":       {"bg": "#e8eaf6", "border": "#283593", "text": "#1a237e"},
    "OTHER":        {"bg": "#f5f5f5", "border": "#616161", "text": "#424242"},
}


def build_layer1_network(data: dict) -> Network:
    """Build a clean visualization showing ONLY entities and their relationships."""
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#fafafa",
        font_color="#212529",
        directed=True,
        notebook=False,
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -250,
          "centralGravity": 0.008,
          "springLength": 200,
          "springConstant": 0.06,
          "damping": 0.85,
          "avoidOverlap": 0.8
        },
        "solver": "forceAtlas2Based",
        "maxVelocity": 25,
        "minVelocity": 0.3,
        "stabilization": {
          "enabled": true,
          "iterations": 1500,
          "fit": true
        }
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
        "smooth": { "enabled": true, "type": "continuous", "roundness": 0.3 },
        "color": { "inherit": false },
        "font": {
          "size": 11,
          "align": "middle",
          "background": "#ffffff",
          "strokeWidth": 0,
          "face": "Segoe UI, Arial, sans-serif"
        }
      },
      "nodes": {
        "font": {
          "size": 13,
          "face": "Segoe UI, Arial, sans-serif",
          "strokeWidth": 2,
          "strokeColor": "#ffffff"
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": true,
        "zoomView": true
      }
    }
    """)

    added_nodes: set = set()

    # ── Entity nodes ─────────────────────────────────────────────────────
    for entity in data["entities"]:
        node_id = entity["name"]
        if not node_id or node_id in added_nodes:
            continue
        added_nodes.add(node_id)

        etype = entity["type"]
        c = ENTITY_COLORS.get(etype, ENTITY_COLORS["OTHER"])

        tooltip = (
            f"<b style='color:{c['border']}'>{etype}</b><br><br>"
            f"<b>{entity['name']}</b><br><br>"
            f"{entity['description']}"
        )

        net.add_node(
            node_id,
            label=entity["name"],
            title=tooltip,
            color={
                "background": c["bg"],
                "border":     c["border"],
                "highlight":  {"background": "#fff9c4", "border": c["border"]},
                "hover":      {"background": "#fff9c4", "border": c["border"]},
            },
            shape="box",
            size=20,
            font={"size": 12, "color": c["text"], "strokeWidth": 2, "strokeColor": c["bg"]},
            borderWidth=2,
            borderWidthSelected=4,
            margin=8,
        )

    # ── Discourse edges (Entity → Entity) ─────────────────────────────────
    for edge in data["entity_entity_edges"]:
        if edge["from"] in added_nodes and edge["to"] in added_nodes:
            rel_label = fmt_rel(edge["type"])
            evidence = edge.get("evidence", "")
            tooltip = f"<b>{rel_label}</b>"
            if evidence:
                ev_short = evidence[:200] + ("..." if len(evidence) > 200 else "")
                tooltip += f"<br><br><i>Evidence: {ev_short}</i>"

            net.add_edge(
                edge["from"], edge["to"],
                color={"color": "#78909c", "highlight": "#37474f", "hover": "#546e7a"},
                width=1.5,
                label=rel_label,
                title=tooltip,
                font={"size": 11, "color": "#37474f",
                      "background": "#ffffff", "strokeWidth": 0},
            )

    return net


# =============================================================================
#  LEGEND OVERLAY (Layer 1 only — clean)
# =============================================================================

def build_overlay_html(data: dict) -> str:
    n_ent = len(data["entities"])
    n_ee = len(data["entity_entity_edges"])

    # Collect unique entity types for the legend
    etypes = sorted(set(e["type"] for e in data["entities"]))
    type_legend_items = ""
    for etype in etypes:
        c = ENTITY_COLORS.get(etype, ENTITY_COLORS["OTHER"])
        count = sum(1 for e in data["entities"] if e["type"] == etype)
        type_legend_items += f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
      <div style="width:14px;height:14px;border-radius:3px;background:{c['bg']};
                  border:2px solid {c['border']};flex-shrink:0;"></div>
      <div style="font-size:11px;color:{c['text']};font-weight:600;">{etype}
        <span style="color:#94a3b8;font-weight:400;">({count})</span></div>
    </div>"""

    # Collect unique relationship types
    rel_types = sorted(set(e["type"] for e in data["entity_entity_edges"]))
    rel_items = ""
    for rt in rel_types[:12]:  # show at most 12
        rel_items += f"""
    <span style="background:#eceff1;color:#455a64;padding:2px 8px;border-radius:10px;
                 font-size:10px;font-weight:600;margin:2px;">{fmt_rel(rt)}</span>"""
    if len(rel_types) > 12:
        rel_items += f"""
    <span style="color:#94a3b8;font-size:10px;margin:2px;">+{len(rel_types)-12} more</span>"""

    return f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">

<!-- Legend -->
<div id="legend" style="
  position: fixed; top: 16px; left: 16px;
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  border-radius: 14px;
  padding: 18px 22px;
  font-family: 'Inter', 'Segoe UI', sans-serif;
  color: #1e293b;
  z-index: 9999;
  min-width: 220px;
  max-width: 280px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.10);
">
  <div style="font-size:15px;font-weight:700;margin-bottom:4px;color:#0f172a;">
    Layer 1: Entity Graph
  </div>
  <div style="font-size:11px;color:#64748b;margin-bottom:14px;">
    Climate Change &middot; {n_ent} entities &middot; {n_ee} relationships
  </div>

  <div style="font-size:11px;font-weight:700;color:#0f172a;margin-bottom:6px;
              letter-spacing:.5px;text-transform:uppercase;">Entity Types</div>
  {type_legend_items}

  <div style="font-size:11px;font-weight:700;color:#0f172a;margin-top:14px;margin-bottom:6px;
              letter-spacing:.5px;text-transform:uppercase;">Relationship Types</div>
  <div style="display:flex;flex-wrap:wrap;gap:2px;">
    {rel_items}
  </div>

  <div style="font-size:10px;color:#94a3b8;border-top:1px solid #e2e8f0;
              padding-top:10px;margin-top:14px;line-height:1.7;">
    Hover for details &middot; Scroll to zoom<br>
    Drag to pan &middot; Click to highlight<br>
    <span style="color:#78909c;">Edge labels = relationship type</span>
  </div>
</div>

<!-- Title card -->
<div style="
  position: fixed; top: 16px; right: 16px;
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  border-radius: 14px;
  padding: 16px 22px;
  font-family: 'Inter', 'Segoe UI', sans-serif;
  color: #1e293b;
  z-index: 9999;
  text-align: right;
  box-shadow: 0 4px 24px rgba(0,0,0,0.10);
">
  <div style="font-size:18px;font-weight:700;color:#0f172a;">Hierarchical Graph RAG</div>
  <div style="font-size:12px;color:#64748b;margin-top:3px;">
    Layer 1 &mdash; Entity Discourse Graph
  </div>
  <div style="font-size:11px;color:#94a3b8;margin-top:6px;">
    Concepts &amp; Communities shown in graph_hierarchy.txt
  </div>
</div>
"""


def inject_overlay(html_path: str, overlay_html: str):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace(
        'body {',
        'body { background: #fafafa !important;'
    )
    html = html.replace("</body>", overlay_html + "\n</body>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# =============================================================================
#  LAYER 2/3 TEXT HIERARCHY
# =============================================================================

def build_hierarchy_text(data: dict) -> str:
    """Build a clean text representation of Layer 3 (Communities) and Layer 2 (Concepts)."""
    lines = []
    W = 76

    lines.append("+" + "=" * W + "+")
    lines.append("|  HIERARCHICAL GRAPH STRUCTURE — LAYERS 2 & 3" + " " * (W - 46) + "|")
    lines.append("+" + "=" * W + "+")
    lines.append("")
    lines.append("  This file shows the higher-level structure of the knowledge graph:")
    lines.append("    Layer 3 (Communities) = Topic clusters with auto-generated summaries")
    lines.append("    Layer 2 (Concepts)    = Abstract themes grouping related entities")
    lines.append("    Layer 1 (Entities)    = See graph_visualization.html for the visual graph")
    lines.append("")
    lines.append("=" * (W + 2))

    # Build community -> concept -> entity mapping
    comm_concepts = {}  # comm_id -> [concept_names]
    for edge in data["community_concept_edges"]:
        cid = edge["from"]
        cname = edge.get("from_name", "")
        if cid not in comm_concepts:
            comm_concepts[cid] = {"name": cname, "concepts": []}
        comm_concepts[cid]["concepts"].append(edge["to"])

    concept_entities = {}  # concept_name -> [entity_names]
    for edge in data["concept_entity_edges"]:
        cname = edge["from"]
        if cname not in concept_entities:
            concept_entities[cname] = []
        concept_entities[cname].append(edge["to"])

    # Build concept descriptions map
    concept_desc = {c["name"]: c["description"] for c in data["concepts"]}

    # Build community summaries map
    comm_summary = {c["id"]: c["summary"] for c in data["communities"]}
    comm_names = {c["id"]: c["name"] for c in data["communities"]}

    lines.append("")

    for comm in data["communities"]:
        cid = comm["id"]
        cname = comm["name"]
        summary = comm["summary"]

        lines.append("=" * (W + 2))
        lines.append(f"  COMMUNITY: {cname}")
        lines.append("=" * (W + 2))
        lines.append("")

        # Wrap summary
        import textwrap
        summary_lines = textwrap.fill(summary, width=W - 4,
                                       initial_indent="    ",
                                       subsequent_indent="    ")
        lines.append("  Summary:")
        lines.append(summary_lines)
        lines.append("")

        # List concepts in this community
        concept_names = comm_concepts.get(cid, {}).get("concepts", [])
        if concept_names:
            lines.append("  Concepts in this community:")
            lines.append("  " + "-" * 40)
            for i, cn in enumerate(concept_names, 1):
                desc = concept_desc.get(cn, "")
                lines.append(f"    {i}. {cn}")
                if desc:
                    desc_wrapped = textwrap.fill(desc, width=W - 10,
                                                  initial_indent="       ",
                                                  subsequent_indent="       ")
                    lines.append(desc_wrapped)

                # List entities under this concept
                entities = concept_entities.get(cn, [])
                if entities:
                    entity_str = ", ".join(entities)
                    ent_wrapped = textwrap.fill(f"Entities: {entity_str}",
                                                width=W - 10,
                                                initial_indent="       ",
                                                subsequent_indent="                ")
                    lines.append(ent_wrapped)
                lines.append("")
        else:
            lines.append("  (No concepts linked to this community)")
            lines.append("")

    # Show any orphan concepts not in any community
    all_community_concepts = set()
    for info in comm_concepts.values():
        all_community_concepts.update(info["concepts"])
    orphan_concepts = [c for c in data["concepts"] if c["name"] not in all_community_concepts]

    if orphan_concepts:
        lines.append("=" * (W + 2))
        lines.append("  UNGROUPED CONCEPTS (not in any community)")
        lines.append("=" * (W + 2))
        lines.append("")
        for c in orphan_concepts:
            lines.append(f"    - {c['name']}: {c['description']}")
            entities = concept_entities.get(c["name"], [])
            if entities:
                lines.append(f"      Entities: {', '.join(entities)}")
            lines.append("")

    # Statistics footer
    lines.append("")
    lines.append("=" * (W + 2))
    lines.append("  SUMMARY STATISTICS")
    lines.append("=" * (W + 2))
    lines.append(f"    Communities:  {len(data['communities'])}")
    lines.append(f"    Concepts:    {len(data['concepts'])}")
    lines.append(f"    Entities:    {len(data['entities'])}")
    lines.append(f"    Discourse edges: {len(data['entity_entity_edges'])}")
    lines.append("")
    lines.append("  For the visual entity graph, open: graph_visualization.html")
    lines.append("  For the query trace, run: python trace_query.py")
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

    # ── Output 1: Layer 1 visualization (HTML) ───────────────────────────
    print("\n[3/4] Building Layer 1 entity graph visualization...")
    net = build_layer1_network(data)

    out_html = os.path.join(os.path.dirname(__file__), "graph_visualization.html")
    net.save_graph(out_html)
    overlay = build_overlay_html(data)
    inject_overlay(out_html, overlay)
    print(f"      Saved: {out_html}")

    # ── Output 2: Layer 2/3 text hierarchy ───────────────────────────────
    print("[4/4] Building Layer 2/3 text hierarchy...")
    hierarchy_text = build_hierarchy_text(data)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_txt = os.path.join(config.RESULTS_DIR, "graph_hierarchy.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(hierarchy_text)
    print(f"      Saved: {out_txt}")

    # Open the HTML in browser
    print("\n  Opening entity graph in browser...")
    webbrowser.open(f"file:///{out_html.replace(os.sep, '/')}")

    print(f"""
{'=' * 60}
  Outputs:
    1. {out_html}
       -> Layer 1: Entity nodes + discourse edges (visual)
       -> Hover nodes for descriptions, edges for evidence

    2. {out_txt}
       -> Layer 3: Communities with summaries
       -> Layer 2: Concepts with member entities
       -> Clean text tree, no clutter
{'=' * 60}""")


if __name__ == "__main__":
    main()
