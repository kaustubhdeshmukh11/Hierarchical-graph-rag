"""
explain_retrieval.py -- Full Retrieval Trace Report Generator

Runs the 3 demo queries and produces a detailed, self-contained HTML report
showing EXACTLY how each answer is generated, step by step:

  Step 1: Community Matching    -- which communities were scored & why
  Step 2: Concept Expansion     -- all concepts under matched communities
  Step 3: Entity Collection     -- all entities traversed via discourse edges
  Step 4: Chunk Assembly        -- full text of every retrieved chunk
  Step 5: LLM Answer            -- the final generated answer

The report is designed for someone with NO prior knowledge of Graph RAG.
Plain English explanations accompany each step.

Output:  retrieval_explanation.html  (open in any browser)

Usage:
    python explain_retrieval.py
"""

import os
import re
import sys
import time
import textwrap
import numpy as np
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    from neo4j import GraphDatabase
    from groq import Groq
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

import config

# ---------------------------------------------------------------------------
#  DEMO QUERIES (same as demo_queries.py)
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "id": 1,
        "label": "Instance-Level Traversal",
        "layer": "Layer 1 -- Entity Discourse",
        "question": "How does burning fossil fuels lead to rising sea levels?",
        "why": (
            "This query tests the Entity layer. It should find Fossil Fuels, CO2, "
            "Global Warming, Ice Sheets, and Sea Level Rise by traversing discourse "
            "edges like 'emits', 'causes', 'melts'. Each entity links to supporting text chunks."
        ),
    },
    {
        "id": 2,
        "label": "Concept-Level Reasoning",
        "layer": "Layer 2 -- Concept Abstraction",
        "question": "What renewable energy solutions are being used to reduce carbon emissions?",
        "why": (
            "This query tests the Concept layer. It should retrieve concepts like "
            "'Renewable Energy Technologies' and 'Clean Energy Sources', then collect "
            "all entities like Solar Panels, Wind Turbines, Battery Storage, and Nuclear Energy."
        ),
    },
    {
        "id": 3,
        "label": "Community-Level Synthesis",
        "layer": "Layer 3 -- Community Summary",
        "question": "How has international climate policy evolved from the 1990s to today?",
        "why": (
            "This query tests the Community layer. The top-down traversal should match "
            "a Policy community, then drill down through policy concepts to reach "
            "entities like UNFCCC, Kyoto Protocol, Paris Agreement, and COP28."
        ),
    },
]

# ---------------------------------------------------------------------------
#  NEO4J HELPERS
# ---------------------------------------------------------------------------

def get_driver():
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def fetch_all_communities(driver) -> list[dict]:
    with driver.session() as s:
        return [
            {
                "id":      r["id"] or "",
                "name":    r["name"] or "",
                "summary": r["summary"] or "",
                "embedding": np.array(r["embedding"], dtype="float32") if r["embedding"] else None,
            }
            for r in s.run(
                "MATCH (cm:Community) "
                "RETURN cm.id AS id, cm.name AS name, cm.summary AS summary, cm.embedding AS embedding"
            )
        ]


def fetch_concepts_for_community(driver, comm_id: str) -> list[dict]:
    with driver.session() as s:
        return [
            {"name": r["name"] or "", "description": r["desc"] or ""}
            for r in s.run(
                "MATCH (cm:Community {id: $id})-[:CONTAINS]->(co:Concept) "
                "RETURN co.name AS name, co.description AS desc",
                id=comm_id,
            )
        ]


def fetch_entities_for_concept(driver, concept_name: str) -> list[dict]:
    with driver.session() as s:
        return [
            {
                "name":        r["name"] or "",
                "type":        r["type"] or "OTHER",
                "description": r["desc"] or "",
            }
            for r in s.run(
                "MATCH (co:Concept {name: $name})-[:INSTANTIATED_BY]->(e:Entity) "
                "RETURN e.name AS name, e.type AS type, e.description AS desc",
                name=concept_name,
            )
        ]


def fetch_discourse_edges_for_entity(driver, entity_name: str) -> list[dict]:
    with driver.session() as s:
        return [
            {
                "from":     r["src"] or "",
                "to":       r["tgt"] or "",
                "rel":      r["rel"] or "",
                "evidence": r["ev"] or "",
            }
            for r in s.run(
                "MATCH (a:Entity {name: $name})-[r]->(b:Entity) "
                "RETURN a.name AS src, b.name AS tgt, type(r) AS rel, r.evidence AS ev",
                name=entity_name,
            )
        ]


def fetch_chunks_for_entities(driver, entity_names: list[str]) -> list[dict]:
    with driver.session() as s:
        rows = s.run(
            "MATCH (ch:Chunk)-[:HAS_ENTITY]->(e:Entity) "
            "WHERE e.name IN $names "
            "RETURN ch.id AS id, ch.text AS text, ch.index AS idx, e.name AS entity "
            "ORDER BY ch.index",
            names=entity_names,
        )
        seen = set()
        chunks = []
        for r in rows:
            cid = r["id"]
            if cid not in seen:
                seen.add(cid)
                chunks.append({
                    "id":     cid or "",
                    "text":   r["text"] or "",
                    "index":  r["idx"] if r["idx"] is not None else 0,
                    "entity": r["entity"] or "",
                })
        return sorted(chunks, key=lambda x: x["index"])


# ---------------------------------------------------------------------------
#  LLM HELPERS
# ---------------------------------------------------------------------------

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_answer(question: str, chunks: list[dict], client: Groq) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks[:6])
    prompt = (
        "You are a knowledgeable expert. Using ONLY the context below, "
        "answer the question clearly in 3-5 sentences. Synthesize information "
        "from multiple passages when relevant.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a knowledgeable expert assistant "
             "that answers questions strictly from provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=600,
    )
    return strip_think(resp.choices[0].message.content.strip())


# ---------------------------------------------------------------------------
#  FULL TRACE PER QUERY
# ---------------------------------------------------------------------------

def run_full_trace(query: dict, embed_model, driver, groq_client) -> dict:
    question = query["question"]
    print(f"  [{query['id']}/3] \"{question[:60]}...\"")

    # Step 1: Embed query
    q_emb = embed_model.encode([question])[0].astype("float32")

    # Step 2: Score ALL communities
    all_communities = fetch_all_communities(driver)
    for c in all_communities:
        if c["embedding"] is not None:
            c["similarity"] = cosine_sim(q_emb, c["embedding"])
        else:
            c["similarity"] = 0.0
    all_communities.sort(key=lambda x: x["similarity"], reverse=True)

    # Step 3: Pick top-k communities
    top_communities = all_communities[: config.TOP_K_COMMUNITIES]
    print(f"         Top communities: {[c['name'] for c in top_communities]}")

    # Step 4: For each matched community, get concepts
    concept_details: list[dict] = []
    seen_concepts: set = set()
    for comm in top_communities:
        concepts = fetch_concepts_for_community(driver, comm["id"])
        for c in concepts:
            if c["name"] not in seen_concepts:
                seen_concepts.add(c["name"])
                concept_details.append({**c, "from_community": comm["name"]})

    print(f"         Concepts found : {len(concept_details)}")

    # Step 5: For each concept, get entities
    entity_details: list[dict] = []
    seen_entities: set = set()
    for concept in concept_details:
        entities = fetch_entities_for_concept(driver, concept["name"])
        for e in entities:
            if e["name"] not in seen_entities:
                seen_entities.add(e["name"])
                entity_details.append({**e, "from_concept": concept["name"]})

    print(f"         Entities found : {len(entity_details)}")

    # Step 6: Sample discourse edges (top 10) for visualisation
    discourse_sample: list[dict] = []
    for entity in entity_details[:15]:
        edges = fetch_discourse_edges_for_entity(driver, entity["name"])
        discourse_sample.extend(edges[:2])

    # Step 7: Fetch chunks
    entity_names = [e["name"] for e in entity_details]
    chunks = fetch_chunks_for_entities(driver, entity_names)
    print(f"         Chunks found   : {len(chunks)}")

    # Step 8: Generate answer
    answer = generate_answer(question, chunks, groq_client)
    print(f"         Answer generated.")
    time.sleep(3)  # Groq rate-limit buffer between queries

    return {
        "query":           query,
        "all_communities": all_communities,
        "top_communities": top_communities,
        "concept_details": concept_details,
        "entity_details":  entity_details,
        "discourse_sample": discourse_sample[:12],
        "chunks":          chunks,
        "answer":          answer,
    }


# ---------------------------------------------------------------------------
#  HTML REPORT GENERATOR
# ---------------------------------------------------------------------------

def escape(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:20px;font-size:11px;font-weight:600;'
        f'letter-spacing:.4px;">{escape(text)}</span>'
    )


def sim_bar(sim: float) -> str:
    pct = int(sim * 100)
    color = "#16a34a" if sim > 0.7 else "#ca8a04" if sim > 0.5 else "#94a3b8"
    return (
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<div style="width:140px;background:#e2e8f0;border-radius:6px;height:10px;">'
        f'<div style="width:{pct}%;background:{color};height:10px;border-radius:6px;"></div>'
        f'</div>'
        f'<span style="font-size:13px;font-weight:600;color:{color};">{sim:.3f}</span>'
        f'</div>'
    )


def type_badge(etype: str) -> str:
    colors = {
        "PERSON": "#7c3aed", "TECHNOLOGY": "#0369a1",
        "CONCEPT": "#0891b2", "ORG": "#b45309",
        "PLACE": "#0f766e", "OTHER": "#475569",
    }
    return badge(etype, colors.get(etype, "#475569"))


ENTITY_TYPE_LABELS = {
    "PERSON": "👤", "TECHNOLOGY": "⚙️",
    "CONCEPT": "💡", "ORG": "🏛️",
    "PLACE": "🌍", "OTHER": "•",
}


def render_query_section(trace: dict, idx: int) -> str:
    q       = trace["query"]
    COLORS  = ["#dc2626", "#2563eb", "#7c3aed"]   # red, blue, purple for 3 queries
    color   = COLORS[idx % len(COLORS)]
    n_comm  = len(trace["top_communities"])
    n_conc  = len(trace["concept_details"])
    n_ent   = len(trace["entity_details"])
    n_chunk = len(trace["chunks"])

    # ── Overview bar
    overview = f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;">
      <div class="stat-pill" style="background:#fee2e2;color:#b91c1c;">
        🔴 {n_comm} Communities matched
      </div>
      <div class="stat-pill" style="background:#dbeafe;color:#1d4ed8;">
        🔵 {n_conc} Concepts expanded
      </div>
      <div class="stat-pill" style="background:#dcfce7;color:#15803d;">
        🟢 {n_ent} Entities collected
      </div>
      <div class="stat-pill" style="background:#f0fdf4;color:#166534;border:1px solid #bbf7d0;">
        📄 {n_chunk} Text chunks assembled
      </div>
    </div>"""

    # ── Step 1: All community scores
    comm_rows = "".join(
        f"""<tr class="{'highlight' if i < n_comm else ''}">
          <td>{'✅' if i < n_comm else ''} {escape(c['name'])}</td>
          <td>{sim_bar(c['similarity'])}</td>
          <td style="font-size:12px;color:#64748b;">{escape(c['summary'][:120])}...</td>
        </tr>"""
        for i, c in enumerate(trace["all_communities"])
    )

    step1 = f"""
    <div class="step">
      <div class="step-header">
        <span class="step-num">STEP 1</span>
        <span class="step-title">Community Matching <span style="color:#94a3b8;font-weight:400">(Layer 3)</span></span>
      </div>
      <p class="explain">
        Every Community node in the graph has an <b>embedding</b> — a numerical fingerprint of its summary.
        We compute the same fingerprint for the query, then rank communities by
        <b>cosine similarity</b> (how "parallel" the two vectors are).
        The top {config.TOP_K_COMMUNITIES} communities are selected for the next step.
      </p>
      <table>
        <thead><tr>
          <th>Community</th><th>Similarity Score</th><th>Summary preview</th>
        </tr></thead>
        <tbody>{comm_rows}</tbody>
      </table>
      <p style="font-size:12px;color:#64748b;margin-top:8px;">
        ✅ = selected (top {config.TOP_K_COMMUNITIES}) &nbsp;|&nbsp;
        score 1.0 = perfect match &nbsp;|&nbsp; score 0.0 = no overlap
      </p>
    </div>"""

    # ── Step 1b: Full community summaries
    comm_detail_blocks = ""
    for c in trace["top_communities"]:
        comm_detail_blocks += f"""
        <div class="card" style="border-left:4px solid #dc2626;">
          <div style="font-weight:700;color:#b91c1c;margin-bottom:6px;">
            🔴 {escape(c['name'])} &nbsp; <span style="font-weight:400;color:#64748b;font-size:12px;">sim={c['similarity']:.4f}</span>
          </div>
          <div style="font-size:13px;line-height:1.8;color:#1e293b;">{escape(c['summary'])}</div>
        </div>"""

    step1b = f"""
    <div class="step">
      <div class="step-header">
        <span class="step-num">STEP 1b</span>
        <span class="step-title">Full Community Summaries (selected)</span>
      </div>
      <p class="explain">
        These are the complete auto-generated summaries of the matched communities.
        They were created by the LLM during graph construction — it read all the concepts
        and entities in each group and wrote a compact description of the theme.
      </p>
      {comm_detail_blocks}
    </div>"""

    # ── Step 2: Concepts
    concept_rows = "".join(
        f"""<tr>
          <td style="font-weight:600;">{escape(c['name'])}</td>
          <td style="font-size:12px;color:#475569;">{escape(c['from_community'])}</td>
          <td style="font-size:12px;color:#1e293b;">{escape(c['description'])}</td>
        </tr>"""
        for c in trace["concept_details"]
    )

    step2 = f"""
    <div class="step">
      <div class="step-header">
        <span class="step-num">STEP 2</span>
        <span class="step-title">Concept Expansion <span style="color:#94a3b8;font-weight:400">(Layer 2)</span></span>
      </div>
      <p class="explain">
        For each matched community, we follow the <b>CONTAINS</b> edges in the graph
        to reach its <b>Concept nodes</b>. Concepts are mid-level abstractions
        (e.g. "Gravitational Theories", "Space Exploration Missions") that group
        related entities together. This step gives us a focused set of themes to
        search within.
      </p>
      <table>
        <thead><tr>
          <th>Concept</th><th>From Community</th><th>Description</th>
        </tr></thead>
        <tbody>{concept_rows}</tbody>
      </table>
    </div>"""

    # ── Step 3: Entities
    entity_rows = "".join(
        f"""<tr>
          <td style="font-weight:600;">{escape(e['name'])}</td>
          <td>{type_badge(e['type'])} {ENTITY_TYPE_LABELS.get(e['type'], '')}</td>
          <td style="font-size:12px;color:#475569;">{escape(e['from_concept'])}</td>
          <td style="font-size:12px;color:#1e293b;">{escape(e['description'])}</td>
        </tr>"""
        for e in trace["entity_details"]
    )

    step3 = f"""
    <div class="step">
      <div class="step-header">
        <span class="step-num">STEP 3</span>
        <span class="step-title">Entity Collection <span style="color:#94a3b8;font-weight:400">(Layer 1)</span></span>
      </div>
      <p class="explain">
        Each Concept connects to ground-level <b>Entity nodes</b> via
        <b>INSTANTIATED_BY</b> edges. Entities are concrete, named things —
        people, spacecraft, planets, laws, organisations. Collecting them tells
        us <em>exactly</em> which facts in the document are relevant.
      </p>
      <table>
        <thead><tr>
          <th>Entity</th><th>Type</th><th>From Concept</th><th>Description</th>
        </tr></thead>
        <tbody>{entity_rows}</tbody>
      </table>
    </div>"""

    # ── Step 4: Discourse edges sample
    disc_rows = "".join(
        f"""<tr>
          <td style="font-weight:600;">{escape(d['from'])}</td>
          <td><span class="rel-badge">{escape(d['rel'].replace('_',' ').lower())}</span></td>
          <td style="font-weight:600;">{escape(d['to'])}</td>
          <td style="font-size:12px;color:#475569;max-width:300px;">{escape(d['evidence'][:120])}</td>
        </tr>"""
        for d in trace["discourse_sample"]
    )

    step4a = f"""
    <div class="step">
      <div class="step-header">
        <span class="step-num">STEP 4</span>
        <span class="step-title">Discourse Relationship Traversal (sample)</span>
      </div>
      <p class="explain">
        Entities are connected to each other by <b>data-driven discourse edges</b>
        (e.g. "orbits", "discovered_by", "detected_in"). These were extracted from
        the source text by the LLM during ingestion. Following these edges lets the
        system find <em>related</em> entities even if they weren't directly mentioned
        in the same community. Below is a sample of these connections.
      </p>
      <table>
        <thead><tr>
          <th>From</th><th>Relation</th><th>To</th><th>Evidence</th>
        </tr></thead>
        <tbody>{disc_rows if disc_rows else '<tr><td colspan="4" style="color:#94a3b8;">No discourse edges found for sampled entities</td></tr>'}</tbody>
      </table>
    </div>"""

    # ── Step 4b: Chunks
    chunk_blocks = ""
    for i, chunk in enumerate(trace["chunks"]):
        chunk_blocks += f"""
        <div class="card" style="background:#f8fafc;border:1px solid #e2e8f0;">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-weight:600;color:#1e293b;">Chunk #{chunk['index'] + 1}</span>
            <span style="font-size:11px;color:#94a3b8;">Via entity: <b>{escape(chunk['entity'])}</b></span>
          </div>
          <div style="font-size:13px;line-height:1.9;color:#334155;white-space:pre-wrap;font-family:'Segoe UI',sans-serif;">{escape(chunk['text'])}</div>
        </div>"""

    step4b = f"""
    <div class="step">
      <div class="step-header">
        <span class="step-num">STEP 5</span>
        <span class="step-title">Context Chunk Assembly</span>
      </div>
      <p class="explain">
        Every Entity links back to the original <b>text chunk</b> it was extracted from
        (via a HAS_ENTITY edge). We collect the unique chunks from all our entities,
        sort them by document order, and pass them to the LLM as <b>grounding context</b>.
        This is the actual source text the LLM reads to form its answer.
      </p>
      <p style="font-size:13px;color:#64748b;margin-bottom:12px;">
        <b>{n_chunk}</b> unique chunks retrieved from {n_ent} entities.
        Chunks are shown in document order.
      </p>
      {chunk_blocks or '<p style="color:#94a3b8;">No chunks found.</p>'}
    </div>"""

    # ── Step 5: Answer
    step5 = f"""
    <div class="step" style="background:linear-gradient(135deg,#f0fdf4,#f8fafc);border:2px solid #86efac;">
      <div class="step-header">
        <span class="step-num" style="background:#16a34a;">ANSWER</span>
        <span class="step-title">LLM-Generated Response</span>
      </div>
      <p class="explain">
        The LLM receives the assembled chunks as context and the original question.
        It generates an answer <b>grounded only in that context</b> — no hallucination.
        Below is the exact response from <code>{config.GROQ_MODEL}</code>.
      </p>
      <div style="background:#fff;border-radius:10px;padding:20px 24px;font-size:15px;
                  line-height:1.9;color:#0f172a;border:1px solid #bbf7d0;
                  font-family:'Segoe UI',Tahoma,sans-serif;">
        {escape(trace['answer'])}
      </div>
    </div>"""

    return f"""
    <section id="query-{q['id']}" class="query-section">
      <div class="query-header" style="border-top:5px solid {color};">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
          <div style="background:{color};color:#fff;width:44px;height:44px;border-radius:50%;
                      display:flex;align-items:center;justify-content:center;
                      font-size:20px;font-weight:700;flex-shrink:0;">Q{q['id']}</div>
          <div>
            <div style="font-size:20px;font-weight:700;color:#0f172a;">{escape(q['label'])}</div>
            <div style="font-size:12px;color:{color};font-weight:600;">{escape(q['layer'])}</div>
          </div>
        </div>
        <div style="background:#f1f5f9;border-radius:10px;padding:14px 18px;
                    font-size:16px;color:#0f172a;font-style:italic;border-left:4px solid {color};">
          "{escape(q['question'])}"
        </div>
        <div style="margin-top:12px;font-size:13px;color:#475569;">
          <b>Why this query?</b> {escape(q['why'])}
        </div>
      </div>
      {overview}
      {step1}
      {step1b}
      {step2}
      {step3}
      {step4a}
      {step4b}
      {step5}
    </section>"""


def build_html(traces: list[dict]) -> str:
    nav_links = "".join(
        f'<a href="#query-{t["query"]["id"]}" style="color:#3b82f6;text-decoration:none;'
        f'padding:6px 14px;border-radius:20px;border:1px solid #bfdbfe;font-size:13px;'
        f'font-weight:600;">Q{t["query"]["id"]}: {t["query"]["label"]}</a>'
        for t in traces
    )

    query_sections = "".join(render_query_section(t, i) for i, t in enumerate(traces))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Graph RAG — Retrieval Explanation</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', 'Segoe UI', sans-serif;
      background: #f1f5f9;
      color: #1e293b;
      line-height: 1.6;
    }}

    /* HEADER */
    .site-header {{
      background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
      color: #fff;
      padding: 40px 40px 32px;
    }}
    .site-header h1 {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .site-header p {{ font-size: 14px; color: #94a3b8; margin-bottom: 14px; }}
    .nav {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }}

    /* MAIN CONTENT */
    .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px; }}

    /* ARCH OVERVIEW */
    .arch-box {{
      background: #fff;
      border: 1.5px solid #e2e8f0;
      border-radius: 16px;
      padding: 24px 28px;
      margin-bottom: 32px;
    }}
    .arch-box h2 {{ font-size: 17px; font-weight: 700; margin-bottom: 16px; color: #0f172a; }}
    .arch-row {{ display: flex; align-items: stretch; gap: 0; margin-bottom: 0; }}
    .arch-layer {{
      flex: 1;
      padding: 16px;
      border-radius: 10px;
      text-align: center;
    }}
    .arch-arrow {{
      display: flex;
      align-items: center;
      padding: 0 8px;
      font-size: 22px;
      color: #94a3b8;
    }}

    /* QUERY SECTIONS */
    .query-section {{
      background: #fff;
      border-radius: 16px;
      overflow: hidden;
      margin-bottom: 40px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }}
    .query-header {{
      padding: 28px 32px 20px;
      background: #fff;
      border-bottom: 1px solid #e2e8f0;
    }}

    /* STEPS */
    .step {{
      padding: 24px 32px;
      border-bottom: 1px solid #f1f5f9;
    }}
    .step-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .step-num {{
      background: #0f172a;
      color: #fff;
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 20px;
      letter-spacing: .6px;
    }}
    .step-title {{
      font-size: 16px;
      font-weight: 700;
      color: #0f172a;
    }}
    .explain {{
      font-size: 13px;
      color: #475569;
      line-height: 1.8;
      margin-bottom: 16px;
      background: #f8fafc;
      padding: 12px 16px;
      border-radius: 8px;
      border-left: 3px solid #94a3b8;
    }}

    /* TABLES */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th {{
      background: #f1f5f9;
      color: #475569;
      text-align: left;
      padding: 8px 12px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .5px;
      text-transform: uppercase;
      border-bottom: 1px solid #e2e8f0;
    }}
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid #f1f5f9;
      vertical-align: top;
    }}
    tr.highlight {{ background: #f0fdf4; }}
    tr:hover {{ background: #f8fafc; }}

    /* CARDS */
    .card {{
      border-radius: 10px;
      padding: 16px 18px;
      margin-bottom: 12px;
    }}

    /* STAT PILLS */
    .stat-pill {{
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
    }}

    /* REL BADGE */
    .rel-badge {{
      background: #e0f2fe;
      color: #0369a1;
      padding: 2px 9px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
      white-space: nowrap;
    }}

    code {{
      background: #f1f5f9;
      color: #7c3aed;
      padding: 1px 6px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
    }}

    /* FOOTER */
    footer {{
      text-align: center;
      padding: 32px;
      font-size: 12px;
      color: #94a3b8;
    }}
  </style>
</head>
<body>

<div class="site-header">
  <h1>🔍 Hierarchical Graph RAG — Retrieval Explanation</h1>
  <p>How does the system go from a question to an answer? This report shows every step.</p>
  <p style="margin-top:4px;color:#64748b;">Generated: {now} &nbsp;|&nbsp; Topic: Climate Change &nbsp;|&nbsp; Model: {config.GROQ_MODEL}</p>
  <div class="nav">{nav_links}</div>
</div>

<div class="container">

  <!-- ARCHITECTURE OVERVIEW -->
  <div class="arch-box">
    <h2>📐 How Hierarchical Graph RAG Works</h2>
    <p style="font-size:13px;color:#475569;margin-bottom:20px;">
      Unlike standard RAG which just searches for similar text chunks, our system
      uses a <b>3-layer knowledge graph</b> to reason from the big picture down to
      specific facts. Here is the complete flow for every query:
    </p>
    <div class="arch-row">
      <div class="arch-layer" style="background:#fee2e2;border:1.5px solid #fca5a5;">
        <div style="font-size:22px;">🔴</div>
        <div style="font-weight:700;color:#b91c1c;margin:4px 0;">Community Layer</div>
        <div style="font-size:12px;color:#991b1b;">High-level topic clusters.<br>Each has a summary embedding.</div>
        <div style="margin-top:10px;font-size:11px;color:#b91c1c;font-weight:600;">📌 STEP 1: Match query here</div>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-layer" style="background:#dbeafe;border:1.5px solid #93c5fd;">
        <div style="font-size:22px;">🔵</div>
        <div style="font-weight:700;color:#1d4ed8;margin:4px 0;">Concept Layer</div>
        <div style="font-size:12px;color:#1e40af;">Abstract themes grouping related entities.<br>Linked by CONTAINS edges.</div>
        <div style="margin-top:10px;font-size:11px;color:#1d4ed8;font-weight:600;">📌 STEP 2: Expand to concepts</div>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-layer" style="background:#dcfce7;border:1.5px solid #86efac;">
        <div style="font-size:22px;">🟢</div>
        <div style="font-weight:700;color:#15803d;margin:4px 0;">Entity Layer</div>
        <div style="font-size:12px;color:#166534;">Concrete named things.<br>Linked by INSTANTIATED_BY + discourse edges.</div>
        <div style="margin-top:10px;font-size:11px;color:#15803d;font-weight:600;">📌 STEP 3: Collect entities</div>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-layer" style="background:#f0fdf4;border:1.5px solid #4ade80;">
        <div style="font-size:22px;">📄</div>
        <div style="font-weight:700;color:#0f172a;margin:4px 0;">Text Chunks</div>
        <div style="font-size:12px;color:#374151;">Original source text segments.<br>Grounding for the LLM.</div>
        <div style="margin-top:10px;font-size:11px;color:#0f172a;font-weight:600;">📌 STEP 4: Assemble context</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-layer" style="background:#faf5ff;border:1.5px solid #c4b5fd;">
        <div style="font-size:22px;">🤖</div>
        <div style="font-weight:700;color:#7c3aed;margin:4px 0;">LLM Answer</div>
        <div style="font-size:12px;color:#5b21b6;">Groq generates a grounded answer from context only.</div>
        <div style="margin-top:10px;font-size:11px;color:#7c3aed;font-weight:600;">📌 STEP 5: Generate answer</div>
      </div>
    </div>
  </div>

  <!-- QUERY SECTIONS -->
  {query_sections}

</div>

<footer>
  Hierarchical Graph RAG Demo &nbsp;&middot;&nbsp; Climate Change &nbsp;&middot;&nbsp; Generated {now}
</footer>

</body>
</html>"""


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Graph RAG — Detailed Retrieval Explanation Report")
    print("=" * 60)

    print("\n[1/3] Loading embedding model (local, no API) ...")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print("      Model loaded.")

    print("[2/3] Connecting to Neo4j Aura ...")
    driver = get_driver()
    driver.verify_connectivity()
    print("      Connected.")

    groq_client = Groq(api_key=config.GROQ_API_KEY)

    print("[3/3] Running all 3 queries with full trace ...")
    traces = []
    for query in QUERIES:
        trace = run_full_trace(query, embed_model, driver, groq_client)
        traces.append(trace)

    driver.close()

    print("\nGenerating HTML report ...")
    html = build_html(traces)

    out_path = os.path.join(os.path.dirname(__file__), "retrieval_explanation.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'=' * 60}")
    print(f"  Report saved: {out_path}")
    print(f"  Open it in any browser.")
    print(f"{'=' * 60}")

    import webbrowser
    webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
