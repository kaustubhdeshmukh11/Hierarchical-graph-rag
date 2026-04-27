"""
trace_query.py -- Plain-Text Retrieval Trace for Presentation

Runs a query through the full hierarchical retrieval pipeline and outputs
a detailed plain-text trace showing EXACTLY how the answer was generated:

  Step 1: Community Matching   (Layer 3) -- threshold filtering + similarity scores
  Step 2: Concept Drill-Down   (Layer 2) -- reranked by query relevance, top-K
  Step 3: Entity Collection    (Layer 1) -- batched BFS + reranking
  Step 4: Chunk Assembly                 -- original text chunks retrieved
  Step 5: Answer Generation              -- final LLM answer

Output:  results/query_trace.txt  (plain text, no HTML)

Usage:
    python trace_query.py                          # runs 3 built-in demo queries
    python trace_query.py --query "What causes global warming?"  # custom query
"""

import argparse
import os
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
from lib.groq_utils import safe_groq_call, strip_think


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO QUERIES — designed to showcase each layer of the architecture
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_QUERIES = [
    {
        "label": "Multi-Hop Reasoning",
        "question": "How did the Incident Nightfall data breach affect Meridian's product development and research direction?",
        "why": (
            "Tests multi-hop traversal: Incident Nightfall --[caused]--> Project Fortress "
            "--[hired]--> Arjun Mehta, and Incident Nightfall --[influenced]--> "
            "Dr. Rahman + Arjun Mehta --[developed]--> federated learning. "
            "Requires connecting entities across data_breach and ai_research documents."
        ),
    },
    {
        "label": "Concept-Level Retrieval",
        "question": "What are all the major products and platforms that Meridian developed?",
        "why": (
            "Tests concept-level grouping: should match concepts like "
            "'Healthcare Product Ecosystem' that group HealthBridge, MediScan, "
            "PharmaTrack, CovidShield, MediPredict, and Meridian Connect together."
        ),
    },
    {
        "label": "Community-Level Synthesis",
        "question": "How did the NovaCare partnership evolve from the founding years through the COVID-19 pandemic?",
        "why": (
            "Tests community matching: should match communities spanning founding, "
            "growth, crisis, and COVID response. Traces NovaCare through HealthBridge "
            "adoption, MediScan deployment, Incident Nightfall strain, and CovidShield."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  NEO4J HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

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
                "id":        r["id"] or "",
                "name":      r["name"] or "",
                "summary":   r["summary"] or "",
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


def fetch_entities_for_concepts_batched(driver, concept_names: list[str]) -> list[dict]:
    """Batched query: get all entities for multiple concepts at once."""
    with driver.session() as s:
        return [
            {
                "name":        r["name"] or "",
                "type":        r["type"] or "OTHER",
                "description": r["desc"] or "",
                "from_concept": r["concept"] or "",
            }
            for r in s.run(
                "MATCH (co:Concept)-[:INSTANTIATED_BY]->(e:Entity) "
                "WHERE co.name IN $names "
                "RETURN e.name AS name, e.type AS type, e.description AS desc, "
                "co.name AS concept",
                names=concept_names,
            )
        ]


def bfs_hop_batched(driver, frontier: list[str], seen: set[str]) -> list[dict]:
    """Single batched query for one BFS hop — both directions."""
    with driver.session() as s:
        return [
            {
                "src": r["src"],
                "tgt": r["tgt"],
                "rel": r["rel"],
                "type": r["type"] or "DISCOVERED",
                "desc": r["desc"] or "",
            }
            for r in s.run(
                "MATCH (a:Entity)-[r]-(b:Entity) "
                "WHERE a.name IN $frontier AND NOT b.name IN $seen "
                "RETURN DISTINCT a.name AS src, b.name AS tgt, "
                "type(r) AS rel, b.type AS type, b.description AS desc",
                frontier=frontier,
                seen=list(seen),
            )
        ]


def fetch_chunks_for_entities(driver, entity_names: list[str]) -> list[dict]:
    """Batched chunk retrieval."""
    with driver.session() as s:
        rows = s.run(
            "MATCH (ch:Chunk)-[:HAS_ENTITY]->(e:Entity) "
            "WHERE e.name IN $names "
            "RETURN ch.id AS id, ch.text AS text, ch.index AS idx, "
            "collect(DISTINCT e.name) AS entities "
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
                    "id":       cid or "",
                    "text":     r["text"] or "",
                    "index":    r["idx"] if r["idx"] is not None else 0,
                    "entities": r["entities"] or [],
                })
        return sorted(chunks, key=lambda x: x["index"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TRACE A SINGLE QUERY
# ═══════════════════════════════════════════════════════════════════════════════

def trace_single_query(
    question: str,
    label: str,
    why: str,
    embed_model,
    driver,
    groq_client: Groq,
) -> str:
    """Run one query through the full pipeline and return the plain-text trace."""

    lines = []
    W = 72  # wrap width

    def line(text=""):
        lines.append(text)

    def hr(char="─"):
        lines.append(char * W)

    def wrap(text, indent=4):
        return textwrap.fill(text, width=W, initial_indent=" " * indent,
                             subsequent_indent=" " * indent)

    # ── Header ──
    line()
    line("=" * W)
    line(f"  QUERY: \"{question}\"")
    line(f"  Type:  {label}")
    line("=" * W)
    line()
    line(wrap(f"Why this query? {why}", indent=2))
    line()

    # ── Step 1: Community Matching (with threshold) ─────────────────────────
    hr("─")
    line("  STEP 1: COMMUNITY MATCHING (Layer 3)")
    hr("─")
    line()
    line(wrap("We embed the query into a vector and compare it against the "
              "embedding of every community summary using cosine similarity. "
              f"Communities scoring below {config.MIN_COMMUNITY_SIMILARITY} "
              "are filtered out, then top-k are selected.", indent=2))
    line()

    q_emb = embed_model.encode([question])[0].astype("float32")
    all_communities = fetch_all_communities(driver)
    for c in all_communities:
        if c["embedding"] is not None:
            c["similarity"] = cosine_sim(q_emb, c["embedding"])
        else:
            c["similarity"] = 0.0
    all_communities.sort(key=lambda x: x["similarity"], reverse=True)

    top_k = config.TOP_K_COMMUNITIES
    threshold = config.MIN_COMMUNITY_SIMILARITY

    for i, c in enumerate(all_communities):
        above_threshold = c["similarity"] >= threshold
        selected = above_threshold and i < top_k
        bar_len = int(c["similarity"] * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)

        status = ""
        if selected:
            status = " <-- SELECTED"
        elif not above_threshold:
            status = " <-- BELOW THRESHOLD"

        line(f"  [{bar}] {c['similarity']:.4f}  {c['name']}{status}")
        if selected:
            line(wrap(f"Summary: {c['summary'][:200]}", indent=6))
            line()

    # Apply threshold + top-k
    filtered = [c for c in all_communities if c["similarity"] >= threshold]
    if not filtered:
        filtered = all_communities[:1] if all_communities else []
    top_communities = filtered[:top_k]

    line()
    line(f"  => {len(top_communities)} communities selected "
         f"(threshold={threshold}, top-k={top_k})")
    line()

    # ── Step 2: Concept Drill-Down (with reranking) ───────────────────────
    hr("─")
    line("  STEP 2: CONCEPT DRILL-DOWN + RERANKING (Layer 2)")
    hr("─")
    line()
    line(wrap("For each selected community, we follow CONTAINS edges to reach "
              "Concept nodes. Then concepts are RERANKED by cosine similarity "
              "between their description embedding and the query embedding. "
              f"Only the top {config.TOP_K_CONCEPTS} most relevant are kept.", indent=2))
    line()

    all_concepts = []
    seen_concepts = set()
    for comm in top_communities:
        concepts = fetch_concepts_for_community(driver, comm["id"])
        line(f"  From \"{comm['name']}\":")
        for c in concepts:
            if c["name"] not in seen_concepts:
                seen_concepts.add(c["name"])
                all_concepts.append({**c, "from_community": comm["name"]})
                line(f"    -> Concept: \"{c['name']}\"")
                line(wrap(f"Description: {c['description']}", indent=8))
        line()

    line(f"  {len(all_concepts)} concepts found, now reranking by query relevance...")
    line()

    # Rerank concepts
    if all_concepts:
        desc_texts = [c["description"] or c["name"] for c in all_concepts]
        desc_embs = embed_model.encode(desc_texts, show_progress_bar=False)
        for i, c in enumerate(all_concepts):
            c["relevance"] = cosine_sim(q_emb, desc_embs[i].astype("float32"))
        all_concepts.sort(key=lambda x: x["relevance"], reverse=True)

    top_concepts = all_concepts[:config.TOP_K_CONCEPTS]

    for i, c in enumerate(all_concepts):
        selected = i < config.TOP_K_CONCEPTS
        marker = " <-- SELECTED" if selected else " <-- DROPPED"
        line(f"  [{c['relevance']:.4f}] {c['name']}{marker}")

    line()
    line(f"  => {len(top_concepts)} concepts selected after reranking")
    line()

    # ── Step 3: Entity Collection (batched BFS + reranking) ─────────────────
    hr("─")
    line("  STEP 3: ENTITY COLLECTION + BATCHED BFS (Layer 1)")
    hr("─")
    line()
    line(wrap("For each concept, we follow INSTANTIATED_BY edges to reach "
              "ground-level Entity nodes (seed entities). Then we perform "
              f"BATCHED BFS up to {config.MAX_HOPS} hops (1 query per hop "
              "instead of 2N), following discourse edges. Finally, entities are "
              f"RERANKED by relevance and capped at {config.MAX_ENTITIES}.", indent=2))
    line()

    concept_names = [c["name"] for c in top_concepts]
    raw_entities = fetch_entities_for_concepts_batched(driver, concept_names)

    all_entities = []
    seen_entities = set()
    seed_entities = []

    for e in raw_entities:
        if e["name"] not in seen_entities:
            seen_entities.add(e["name"])
            all_entities.append({**e, "hop": 0})
            seed_entities.append(e["name"])
            line(f"    -> {e['name']} [{e['type']}] (from: {e['from_concept']})")

    line()
    line(f"  Seed entities: {len(seed_entities)}")
    line()

    # Batched BFS hops
    frontier = list(seed_entities)
    for hop_num in range(1, config.MAX_HOPS + 1):
        if not frontier:
            break
        line(f"  BFS HOP {hop_num} (1 batched query for {len(frontier)} entities):")

        edges = bfs_hop_batched(driver, frontier, seen_entities)
        next_frontier = []
        for edge in edges:
            neighbor = edge["tgt"]
            if neighbor not in seen_entities:
                seen_entities.add(neighbor)
                all_entities.append({
                    "name": neighbor,
                    "type": edge["type"],
                    "description": edge.get("desc", ""),
                    "from_concept": "via BFS",
                    "hop": hop_num,
                })
                next_frontier.append(neighbor)
                line(f"    {edge['src']} --[{edge['rel']}]--> {edge['tgt']}")

        if not next_frontier:
            line("    (no new entities discovered)")
        frontier = next_frontier
        line()

    line(f"  Total entities before reranking: {len(all_entities)}")
    line()

    # Rerank entities by query relevance
    if all_entities:
        desc_texts = [f"{e['name']}: {e.get('description', '')}" for e in all_entities]
        ent_embs = embed_model.encode(desc_texts, show_progress_bar=False)
        for i, e in enumerate(all_entities):
            e["relevance"] = cosine_sim(q_emb, ent_embs[i].astype("float32"))
        all_entities.sort(key=lambda x: x["relevance"], reverse=True)

    capped_entities = all_entities[:config.MAX_ENTITIES]

    line("  Entity reranking (top entities by query relevance):")
    for i, e in enumerate(capped_entities[:10]):  # show top 10
        line(f"    [{e['relevance']:.4f}] {e['name']} [{e.get('type', '?')}] (hop {e.get('hop', 0)})")
    if len(capped_entities) > 10:
        line(f"    ... and {len(capped_entities) - 10} more")
    line()

    line(f"  => {len(capped_entities)} entities after reranking + cap "
         f"(max={config.MAX_ENTITIES})")
    line()

    # ── Step 4: Chunk Assembly ───────────────────────────────────────────────
    hr("─")
    line("  STEP 4: CHUNK ASSEMBLY")
    hr("─")
    line()
    line(wrap("Each entity links back to the original text chunk it was "
              "extracted from (via HAS_ENTITY edges). We collect all unique "
              "chunks in a single batched query, sort by document order, "
              "and these become the grounding context for the LLM.", indent=2))
    line()

    entity_names = [e["name"] for e in capped_entities]
    chunks = fetch_chunks_for_entities(driver, entity_names)

    for i, chunk in enumerate(chunks[:8]):  # show at most 8 chunks
        text_preview = chunk["text"][:200].replace("\n", " ")
        chunk_entities = ", ".join(chunk["entities"][:5]) if chunk.get("entities") else "N/A"
        line(f"  Chunk #{chunk['index'] + 1} (via entities: {chunk_entities}):")
        line(wrap(f'"{text_preview}..."', indent=4))
        line()

    if len(chunks) > 8:
        line(f"  ... and {len(chunks) - 8} more chunks")
        line()

    line(f"  => {len(chunks)} chunks assembled in document order")
    line()

    # ── Step 5: Answer Generation ────────────────────────────────────────────
    hr("─")
    line("  STEP 5: ANSWER GENERATION")
    hr("─")
    line()
    line(wrap(f"The {len(chunks)} assembled chunks are sent to the LLM "
              f"({config.GROQ_MODEL}) as grounding context along with the "
              "original question. The LLM generates an answer strictly from "
              "this context.", indent=2))
    line()

    # Actually generate the answer
    context = "\n\n---\n\n".join(c["text"] for c in chunks[:6])
    prompt = (
        "You are a knowledgeable expert. Using ONLY the context below, "
        "answer the question clearly in 3-5 sentences. Synthesize information "
        "from multiple passages.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    answer = safe_groq_call(
        groq_client, prompt,
        system_message="You are a knowledgeable expert assistant that answers "
                       "questions strictly from provided context.",
        max_tokens=600,
    )

    line("  ANSWER:")
    line(wrap(answer, indent=4))
    line()

    # ── Flow Summary ─────────────────────────────────────────────────────────
    line("=" * W)
    line("  FLOW SUMMARY:")
    n_comm = len(top_communities)
    n_conc = len(top_concepts)
    n_ent = len(capped_entities)
    n_seed = len(seed_entities)
    n_bfs = sum(1 for e in capped_entities if e.get("hop", 0) > 0)
    n_chunk = len(chunks)
    line(f"    Query -> {n_comm} Communities (threshold={threshold}) "
         f"-> {n_conc} Concepts (reranked) "
         f"-> {n_seed} Seed + {n_bfs} BFS -> {n_ent} Entities (reranked, "
         f"capped at {config.MAX_ENTITIES}) -> {n_chunk} Chunks -> Answer")
    line("=" * W)
    line()

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Trace a query through the hierarchical Graph RAG pipeline (plain text output)"
    )
    parser.add_argument("--query", help="Custom query to trace (omit for 3 built-in demos)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Hierarchical Graph RAG — Query Trace Generator")
    print("=" * 60)

    print("\n[1/3] Loading embedding model (local, no API) ...")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print("      Model loaded.")

    print("[2/3] Connecting to Neo4j ...")
    driver = get_driver()
    driver.verify_connectivity()
    print("      Connected.")

    groq_client = Groq(api_key=config.GROQ_API_KEY)

    # Determine queries to run
    if args.query:
        queries = [{
            "label": "Custom Query",
            "question": args.query,
            "why": "User-specified query.",
        }]
    else:
        queries = DEMO_QUERIES

    print(f"[3/3] Tracing {len(queries)} queries ...\n")

    # Build the full trace document
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        "+" + "=" * 70 + "+\n"
        "|  HIERARCHICAL GRAPH RAG — QUERY TRACE REPORT" + " " * 25 + "|\n"
        "+" + "=" * 70 + "+\n"
        f"\n  Generated: {now}\n"
        f"  Model:     {config.GROQ_MODEL}\n"
        f"  Topic:     Meridian Health Systems\n"
        f"  Graph:     3-Layer (Community -> Concept -> Entity)\n"
        f"  Settings:  top_k_communities={config.TOP_K_COMMUNITIES}, "
        f"max_hops={config.MAX_HOPS}, "
        f"min_community_sim={config.MIN_COMMUNITY_SIMILARITY}, "
        f"top_k_concepts={config.TOP_K_CONCEPTS}, "
        f"max_entities={config.MAX_ENTITIES}\n"
        "\n  This trace shows step-by-step how each query is answered\n"
        "  by traversing the hierarchical knowledge graph.\n"
        "  Quality improvements: threshold filtering, concept reranking,\n"
        "  batched BFS (1 query/hop), entity reranking + cap.\n"
    )

    traces = [header]

    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] \"{q['question'][:60]}...\"")
        trace = trace_single_query(
            q["question"], q["label"], q["why"],
            embed_model, driver, groq_client,
        )
        traces.append(trace)
        if i < len(queries) - 1:
            time.sleep(3)  # rate-limit buffer

    driver.close()

    # Save output
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "query_trace.txt")
    full_text = "\n".join(traces)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\n{'=' * 60}")
    print(f"  Trace saved: {out_path}")
    print(f"  Open it in any text editor.")
    print(f"{'=' * 60}")

    # Also print to console
    print("\n" + full_text)


if __name__ == "__main__":
    main()
