"""
Graph Builder -- 3-Layer Hierarchical Discourse Graph Construction.

Builds a novel knowledge graph with:
  Layer 1: Instance nodes (entities) + data-driven relationship edges
  Layer 2: Concept nodes (abstract themes) + INSTANTIATED_BY edges
  Layer 3: Community nodes (topic clusters) + CONTAINS edges + summaries

Rate-limit optimizations for Groq free tier (30 req/min, 6000 req/day):
  - CHUNK BATCHING : 3 chunks per LLM call
  - DISK CACHE     : extraction saved to cache/ dir (skip re-extraction on re-run)
  - RATE LIMITER   : shared sliding-window via lib/groq_utils.py

Usage:
    python graph_builder.py --pdf data/climate_change.pdf
    python graph_builder.py --pdf data/climate_change.pdf --no-cache
"""

import argparse
import json
import os
import re
import time
import hashlib

import fitz
import networkx as nx
import numpy as np
from groq import Groq
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

import config
from lib.groq_utils import safe_groq_call, parse_json_from_llm, TokenLimitError


# =============================================================================
#  UTILITIES
# =============================================================================

def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_txt(path: str) -> str:
    """Load a plain text file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_data_dir(data_dir: str) -> tuple[str, str]:
    """Load all .txt and .pdf files from a directory.

    Returns:
        (combined_text, doc_name) where doc_name is the directory basename.
    """
    texts = []
    files_loaded = []
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext == ".pdf":
            texts.append(load_pdf(fpath))
            files_loaded.append(fname)
        elif ext == ".txt":
            texts.append(load_txt(fpath))
            files_loaded.append(fname)

    if not texts:
        raise FileNotFoundError(f"No .txt or .pdf files found in {data_dir}")

    print(f"       Loaded {len(files_loaded)} files: {', '.join(files_loaded)}")
    combined = "\n\n".join(texts)
    doc_name = os.path.basename(data_dir.rstrip('/\\')) or "multi_doc"
    return combined, doc_name


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + config.CHUNK_SIZE
        chunks.append(text[start:end])
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


# =============================================================================
#  DISK CACHE
# =============================================================================

def _cache_path(pdf_path: str) -> str:
    pdf_hash = hashlib.md5(os.path.abspath(pdf_path).encode()).hexdigest()[:10]
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"extraction_{pdf_hash}.json")


def load_cache(pdf_path: str) -> dict | None:
    """Return cached extraction for this PDF, or None."""
    p = _cache_path(pdf_path)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"       [Cache] Loaded from {p}")
        return data
    return None


def save_cache(pdf_path: str, data: dict):
    """Persist extraction so future runs skip all LLM calls for this PDF."""
    p = _cache_path(pdf_path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"       [Cache] Saved to {p}")


# =============================================================================
#  LAYER 1: Instance Extraction  (BATCHED -- N chunks per LLM call)
# =============================================================================

INSTANCE_SYSTEM_MSG = (
    "You are a selective knowledge-graph extraction engine. You extract only "
    "the most important named entities and meaningful causal relationships from text. "
    "You prioritise quality and selectivity over completeness. "
    "You always output valid JSON and nothing else."
)

BATCHED_INSTANCE_PROMPT = """\
Below are {n} numbered text segments. For EACH segment, extract a SELECTIVE
set of entities and relationships that capture the core narrative.

## ENTITY EXTRACTION RULES — BE SELECTIVE

Extract ONLY entities that satisfy ALL of the following:
1. It is a named, specific thing — not a number, date, percentage, or generic noun
2. It plays an active role in the narrative — it DOES something or something HAPPENS to it  
3. It would appear in a relationship edge — if you cannot think of a meaningful
   relationship for it, do NOT extract it
4. Extract 5-8 entities per segment — choose the most important ones

Entity types allowed: PERSON | TECHNOLOGY | ORGANIZATION | EVENT | CONCEPT | PRODUCT | DEPARTMENT
DO NOT extract: METRIC (percentages, amounts, dates), PLACE (unless central to the story),
                generic nouns ("system", "approach", "method", "policy")

## ENTITY NAMING RULES
- Always use the FULL official name, never abbreviations
  WRITE: "Nexora Industries" not "Nexora"
  WRITE: "Cortex Knowledge Graph" not "Cortex"  
  WRITE: "DataMesh Labs" not "DML"
- Use exactly the same name every time the same entity appears
- Title Case for all names

## RELATIONSHIP EXTRACTION RULES — CAUSAL CHAINS ONLY

Extract ONLY relationships that represent:
- Cause -> Effect: "Incident Zephyr CAUSED customer_losses"
- Action -> Result: "Evelyn Hartwell FOUNDED Nexora Industries"  
- Problem -> Solution: "data_quality_issues SOLVED_BY Sentinel"
- Creation/Destruction: "Marcus Tan DEVELOPED Athena"

DO NOT extract:
- Structural/membership facts: "Marcus Tan IS_PART_OF R&D" (not causal)
- Temporal facts: "Pulse LAUNCHED_IN 2017" (date is not an entity)
- Vague connections: anything you would label "related_to", "associated_with"

Relationship type must be a specific verb: "founded", "developed", "acquired",
"led", "caused", "funded", "competed_with", "hired", "integrated"

## EXAMPLE — Enterprise document segment

Segment: "Nexora Industries acquired DataMesh Labs for fourteen million dollars,
which gave them access to the Sentinel data quality engine that Chen Wei had
built. Chen Wei joined as VP of Data Engineering under Marcus Tan."

GOOD extraction:
{{
  "segment": 1,
  "entities": [
    {{"name": "Nexora Industries", "type": "ORGANIZATION", "description": "Enterprise intelligence platform company"}},
    {{"name": "DataMesh Labs", "type": "ORGANIZATION", "description": "Boston startup with data quality technology"}},
    {{"name": "Sentinel", "type": "PRODUCT", "description": "Data quality engine for detecting anomalies and duplicates"}},
    {{"name": "Chen Wei", "type": "PERSON", "description": "Founder of DataMesh Labs, became VP Data Engineering"}}
  ],
  "relationships": [
    {{"source": "Nexora Industries", "target": "DataMesh Labs", "type": "acquired", "evidence": "acquired DataMesh Labs for fourteen million dollars"}},
    {{"source": "DataMesh Labs", "target": "Sentinel", "type": "developed", "evidence": "had developed the Sentinel data quality engine"}}
  ]
}}

BAD extraction (DO NOT DO THIS):
- Extracting "fourteen million dollars" as a METRIC entity
- Extracting "2019" as an EVENT entity  
- Extracting "data quality" as a CONCEPT entity
- Adding relationship "Chen Wei IS_PART_OF R&D" (structural, not causal)

Return ONLY a valid JSON array — one object per segment, no extra text.

{segments}
"""


def extract_instances_batch(
    chunks: list[str],
    start_idx: int,
    client: Groq,
) -> list[dict]:
    """
    One LLM call extracts entities/relations from a BATCH of chunks.
    Returns a list of {entities, relationships} dicts (one per chunk).
    """
    n = len(chunks)
    segments_text = "\n\n".join(
        f"=== Segment {i + 1} ===\n{chunk}" for i, chunk in enumerate(chunks)
    )
    prompt = BATCHED_INSTANCE_PROMPT.format(n=n, segments=segments_text)
    response = safe_groq_call(client, prompt, system_message=INSTANCE_SYSTEM_MSG)
    parsed = parse_json_from_llm(response)

    # Normalize: build a map segment_number -> item
    if isinstance(parsed, list):
        seg_map = {}
        for i, item in enumerate(parsed):
            seg_num = item.get("segment", i + 1) if isinstance(item, dict) else i + 1
            seg_map[seg_num] = item if isinstance(item, dict) else {}
    elif isinstance(parsed, dict):
        seg_map = {1: parsed}
    else:
        seg_map = {}

    results = []
    for i, chunk in enumerate(chunks):
        chunk_idx = start_idx + i
        item = seg_map.get(i + 1, {})
        entities = item.get("entities", []) or []
        relationships = item.get("relationships", []) or []

        for e in entities:
            if isinstance(e, dict) and "name" in e:
                e["name"] = e["name"].strip().title()
            if isinstance(e, dict):
                e["chunk_idx"] = chunk_idx

        for r in relationships:
            if isinstance(r, dict):
                if "source" in r:
                    r["source"] = r["source"].strip().title()
                if "target" in r:
                    r["target"] = r["target"].strip().title()
                r["chunk_idx"] = chunk_idx

        results.append({"entities": entities, "relationships": relationships})

    return results


def canonical_key(name: str) -> str:
    """Normalize entity name to a canonical dedup key.

    Strips articles, parenthetical abbreviations, punctuation, and collapses
    whitespace so that 'The Paris Agreement', 'Paris Agreement (2015)', and
    'paris agreement' all map to the same key.
    """
    name = name.lower().strip()
    # Remove articles
    for prefix in ["the ", "a ", "an "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Remove parenthetical content: "United Nations (UN)" -> "united nations"
    name = re.sub(r"\s*\(.*?\)\s*", " ", name)
    # Remove punctuation, keep spaces and alphanumeric
    name = re.sub(r"[^a-z0-9 ]", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _merge_descriptions(desc_a: str, desc_b: str) -> str:
    """Merge two entity descriptions, keeping the longer or combining unique info."""
    if not desc_a:
        return desc_b or ""
    if not desc_b:
        return desc_a
    # If one contains the other, keep the longer
    if desc_a.lower() in desc_b.lower():
        return desc_b
    if desc_b.lower() in desc_a.lower():
        return desc_a
    # Combine, taking the longer as base
    if len(desc_a) >= len(desc_b):
        return desc_a
    return desc_b


def merge_instance_extractions(all_extractions: list[dict],
                                embed_model=None) -> dict:
    """Merge + deduplicate entities/relationships from all chunks.

    Two-phase deduplication:
      Phase 1 -- Canonical key: normalise names (strip articles, punctuation,
                 parenthetical abbreviations) and merge exact matches.
      Phase 2 -- Embedding similarity: for remaining entities, compute pairwise
                 cosine similarity and merge pairs above ENTITY_DEDUP_THRESHOLD.
    """
    # ── Phase 1: canonical-key merge ────────────────────────────────────────
    entity_map: dict[str, dict] = {}      # canonical_key -> best entity dict
    name_to_canon: dict[str, str] = {}    # original name.lower() -> canonical_key

    for ext in all_extractions:
        for e in ext.get("entities", []):
            if not isinstance(e, dict):
                continue
            raw_name = e.get("name", "").strip()
            if not raw_name:
                continue
            canon = canonical_key(raw_name)
            if not canon:
                continue
            name_to_canon[raw_name.lower()] = canon

            if canon not in entity_map:
                entity_map[canon] = {**e, "name": raw_name}
            else:
                # Keep the version with the longer description
                existing = entity_map[canon]
                entity_map[canon]["description"] = _merge_descriptions(
                    existing.get("description", ""),
                    e.get("description", ""),
                )
                # Prefer the version with the nicer (longer) display name
                if len(raw_name) > len(existing.get("name", "")):
                    entity_map[canon]["name"] = raw_name

    # ── Phase 2: embedding-similarity merge ─────────────────────────────────
    if embed_model is not None and len(entity_map) > 1:
        threshold = config.ENTITY_DEDUP_THRESHOLD
        canon_keys = list(entity_map.keys())
        display_names = [entity_map[k]["name"] for k in canon_keys]

        embeddings = embed_model.encode(display_names, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms

        merged_into: dict[int, int] = {}   # idx -> idx it merged into
        for i in range(len(canon_keys)):
            if i in merged_into:
                continue
            for j in range(i + 1, len(canon_keys)):
                if j in merged_into:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    # Merge j into i
                    merged_into[j] = i
                    src = entity_map[canon_keys[j]]
                    dst = entity_map[canon_keys[i]]
                    dst["description"] = _merge_descriptions(
                        dst.get("description", ""),
                        src.get("description", ""),
                    )
                    if len(src.get("name", "")) > len(dst.get("name", "")):
                        dst["name"] = src["name"]
                    # Update name mapping so relationships resolve correctly
                    name_to_canon[src["name"].lower()] = canon_keys[i]
                    print(f"       [Dedup] Merged \"{src['name']}\" into "
                          f"\"{dst['name']}\" (sim={sim:.3f})")

        # Remove merged-away entries
        for j in sorted(merged_into.keys(), reverse=True):
            del entity_map[canon_keys[j]]

    # ── Build final entity list ─────────────────────────────────────────────
    final_entities = list(entity_map.values())

    # Build a lookup: any variant name -> canonical display name
    canon_to_display = {k: entity_map[k]["name"] for k in entity_map}
    def resolve_name(raw: str) -> str:
        """Resolve a relationship endpoint to its canonical display name."""
        canon = canonical_key(raw)
        if canon in canon_to_display:
            return canon_to_display[canon]
        return raw.strip().title()

    # ── Deduplicate relationships ───────────────────────────────────────────
    relationships: list[dict] = []
    seen_rels: set[str] = set()

    for ext in all_extractions:
        for r in ext.get("relationships", []):
            if not isinstance(r, dict):
                continue
            src = resolve_name(r.get("source", ""))
            tgt = resolve_name(r.get("target", ""))
            rtype = r.get("type", "").lower()
            if not src or not tgt or src == tgt:
                continue
            rel_key = f"{canonical_key(src)}|{rtype}|{canonical_key(tgt)}"
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                relationships.append({
                    **r,
                    "source": src,
                    "target": tgt,
                })

    return {"entities": final_entities, "relationships": relationships}


# =============================================================================
#  LAYER 2: Concept Generation
# =============================================================================

CONCEPT_SYSTEM_MSG = (
    "You are a knowledge organization expert. You group specific entities into "
    "meaningful abstract concepts that reveal the thematic structure of a document. "
    "You always output valid JSON."
)

CONCEPT_PROMPT = """\
Below is a list of important entities extracted from a document.
Group them into higher-level CONCEPTS that reveal the document's thematic structure.

## RULES

Grouping rules:
- Each concept should have 2-6 member entities
- Only group entities that share a direct semantic relationship
  (same domain, same causal chain, same stakeholder group)
- DO NOT create a concept just to place leftover entities —
  if an entity doesn't fit cleanly, leave it ungrouped (omit it)
- Target 5-8 concepts total to capture the full thematic breadth

Naming rules:
- Concept name must describe WHAT THE ENTITIES DO TOGETHER, not just what they are
  GOOD: "Fossil Fuel Emissions Driving Global Warming" (describes the dynamic)
  BAD: "Greenhouse Gases" (just a category label)

Cross-membership rule:
- An entity MAY appear in 2 concepts maximum if it genuinely bridges them
- Cross-membership creates graph edges — use it intentionally for entities
  that connect two themes (e.g. "CO2" bridges "Fossil Fuel Emissions"
  and "Ocean Acidification")
- DO NOT put every entity in multiple groups — this destroys community structure

## CRITICAL — USE EXACT ENTITY NAMES
- The member names in your output MUST exactly match names from the entity list
- DO NOT rename, abbreviate, or paraphrase entity names
- Copy-paste entity names from the list below

## ENTITY LIST
{entity_list}

Return ONLY valid JSON:
{{
  "concepts": [
    {{
      "name": "Descriptive Concept Name",
      "description": "One sentence: what dynamic or theme unites these entities",
      "members": ["Entity Full Name 1", "Entity Full Name 2", "Entity Full Name 3"]
    }}
  ]
}}
"""


def _resolve_concept_members(concepts: list[dict], entities: list[dict]) -> list[dict]:
    """Resolve member names in concepts to actual entity names."""
    entity_names = set()
    entity_names_lower = {}
    for e in entities:
        if isinstance(e, dict) and "name" in e:
            entity_names.add(e["name"])
            entity_names_lower[e["name"].lower()] = e["name"]

    for c in concepts:
        if not isinstance(c, dict):
            continue
        c["name"] = c.get("name", "Unknown Concept").strip()
        resolved_members = []
        for m in c.get("members", []):
            m = m.strip()
            if m in entity_names:
                resolved_members.append(m)
            elif m.lower() in entity_names_lower:
                resolved_members.append(entity_names_lower[m.lower()])
            elif m.strip().title().lower() in entity_names_lower:
                resolved_members.append(entity_names_lower[m.strip().title().lower()])
            else:
                matched = False
                for en in entity_names:
                    if m.lower() in en.lower() or en.lower() in m.lower():
                        resolved_members.append(en)
                        matched = True
                        break
                if not matched:
                    print(f"       [Concept] Warning: member \"{m}\" not found in entities, skipping")
        c["members"] = resolved_members

    return [c for c in concepts if len(c.get("members", [])) >= 2]


def _build_entity_list_text(entities: list[dict], max_chars: int = 6000) -> str:
    """Build entity list text for the concept prompt, truncating if needed."""
    lines = []
    total = 0
    for e in entities:
        if isinstance(e, dict) and "name" in e:
            etype = e.get("type", "OTHER")
            desc = e.get("description", "")[:80]  # cap description length
            line = f"- {e['name']} [{etype}]: {desc}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
    return "\n".join(lines)


def generate_concepts(entities: list[dict], client: Groq) -> list[dict]:
    """Group instance entities into abstract concept nodes.

    Token-limit aware: if the entity list is too large, it splits entities
    into batches and merges the resulting concepts.
    """
    MAX_ENTITIES_PER_BATCH = 25

    if len(entities) <= MAX_ENTITIES_PER_BATCH:
        batches = [entities]
    else:
        batches = []
        for i in range(0, len(entities), MAX_ENTITIES_PER_BATCH):
            batches.append(entities[i:i + MAX_ENTITIES_PER_BATCH])
        print(f"       [Concept] {len(entities)} entities -> {len(batches)} batches of ~{MAX_ENTITIES_PER_BATCH}")

    all_concepts = []
    for batch_idx, batch_entities in enumerate(batches):
        entity_list = _build_entity_list_text(batch_entities)
        prompt = CONCEPT_PROMPT.format(entity_list=entity_list)

        try:
            response = safe_groq_call(client, prompt, system_message=CONCEPT_SYSTEM_MSG)
        except TokenLimitError:
            # Trim descriptions more aggressively and retry
            print(f"       [Concept] Token limit hit — trimming descriptions...")
            entity_list = _build_entity_list_text(batch_entities, max_chars=3000)
            prompt = CONCEPT_PROMPT.format(entity_list=entity_list)
            response = safe_groq_call(client, prompt, system_message=CONCEPT_SYSTEM_MSG)

        result = parse_json_from_llm(response)
        concepts = result.get("concepts", []) if isinstance(result, dict) else []
        resolved = _resolve_concept_members(concepts, entities)  # resolve against ALL entities
        all_concepts.extend(resolved)

        if len(batches) > 1:
            print(f"       [Concept] Batch {batch_idx+1}/{len(batches)}: {len(resolved)} concepts")
            time.sleep(3)  # cooldown between batches

    return all_concepts


# =============================================================================
#  LAYER 3: Community Detection
# =============================================================================

COMMUNITY_SYSTEM_MSG = (
    "You are a technical summarization expert. You write information-dense "
    "summaries that capture the key themes, relationships, and narrative of a group "
    "of related concepts. Your summaries are designed for semantic search matching."
)

COMMUNITY_SUMMARY_PROMPT = """\
Below is a group of related concepts and their member entities from a document.
Write a KEYWORD-DENSE summary (3-5 sentences) designed for semantic search retrieval.

## RULES — OPTIMIZE FOR RETRIEVAL
1. Name EVERY member entity explicitly — do not summarize them as "various entities"
2. Use SPECIFIC action verbs: "founded", "developed", "acquired", "caused", "led"
3. Include factual details: what happened, who did it, what it affected
4. Write in a way that would match user questions like:
   - "Who founded [entity]?"
   - "What is [entity]?"
   - "What happened during [event]?"
   - "How does [entity] relate to [entity]?"
5. Do NOT write abstract thematic descriptions like "This community covers..."
6. Do NOT use filler phrases — every word should add searchable information

## BAD EXAMPLE (too abstract):
"This community explores the theme of organizational leadership and innovation,
focusing on how key figures shaped the company's direction."

## GOOD EXAMPLE (keyword-dense):
"Evelyn Hartwell founded Nexora Industries and served as CEO. Marcus Tan led
the R&D department and developed the Athena analytics platform. Nexora Industries
acquired DataMesh Labs, which brought Sentinel data quality technology and
Chen Wei as VP of Data Engineering."

Concepts in this community:
{concept_descriptions}

Member entities:
{member_list}

Return ONLY the summary text, no JSON or formatting.
"""


def detect_communities(concepts: list[dict]) -> list[list[int]]:
    """Detect communities among concepts (shared members as edges)."""
    G = nx.Graph()
    for i, c in enumerate(concepts):
        G.add_node(i, name=c.get("name", ""))

    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            mi = set(m.lower() for m in concepts[i].get("members", []))
            mj = set(m.lower() for m in concepts[j].get("members", []))
            overlap = mi & mj
            if overlap:
                G.add_edge(i, j, weight=len(overlap))

    if G.number_of_edges() == 0:
        return [list(range(len(concepts)))]

    try:
        from cdlib.algorithms import leiden
        coms = leiden(G)
        return [list(c) for c in coms.communities]
    except ImportError:
        print("       Warning: cdlib not installed, using greedy modularity")
    except Exception as ex:
        print(f"       Warning: Leiden failed ({ex}), using greedy modularity")

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        return [list(c) for c in greedy_modularity_communities(G)]
    except Exception:
        return [list(range(len(concepts)))]


def generate_community_summaries(
    communities: list[list[int]], concepts: list[dict], client: Groq
) -> list[dict]:
    """Generate a summary for each detected community.

    Token-limit aware: truncates member lists if prompt is too large.
    """
    community_nodes = []
    for idx, member_indices in enumerate(communities):
        concept_descs = []
        all_members: list[str] = []
        for ci in member_indices:
            c = concepts[ci]
            concept_descs.append(f"- {c['name']}: {c.get('description', '')}")
            all_members.extend(c.get("members", []))

        unique_members = list(set(all_members))
        member_list = ", ".join(unique_members)

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            concept_descriptions="\n".join(concept_descs),
            member_list=member_list,
        )

        try:
            summary = safe_groq_call(client, prompt, system_message=COMMUNITY_SYSTEM_MSG)
        except TokenLimitError:
            # Truncate member list and retry
            print(f"       [Community] Token limit — trimming to 15 members...")
            member_list = ", ".join(unique_members[:15])
            prompt = COMMUNITY_SUMMARY_PROMPT.format(
                concept_descriptions="\n".join(concept_descs),
                member_list=member_list,
            )
            summary = safe_groq_call(client, prompt, system_message=COMMUNITY_SYSTEM_MSG)

        if not summary:
            summary = f"Community covering: {member_list[:200]}"

        community_nodes.append({
            "id": f"community_{idx}",
            "name": f"Community {idx + 1}",
            "summary": summary,
            "concept_indices": member_indices,
            "member_entities": unique_members,
        })

        if idx < len(communities) - 1:
            time.sleep(3)  # cooldown between community LLM calls

    return community_nodes


# =============================================================================
#  NEO4J INGESTION
# =============================================================================

def get_driver():
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )


def clear_graph(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("       Cleared existing graph")


def setup_schema(driver):
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (cm:Community) ON (cm.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (ch:Chunk) ON (ch.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.name)")
    print("       Schema indexes created")


def ingest_document(driver, doc_name: str):
    with driver.session() as session:
        session.run("MERGE (d:Document {name: $name})", name=doc_name)


def ingest_chunks(driver, chunks: list[str], doc_name: str, embeddings: np.ndarray):
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_name}_chunk_{i}"
            session.run(
                """
                MERGE (ch:Chunk {id: $id})
                SET ch.text = $text, ch.index = $idx, ch.embedding = $emb
                WITH ch
                MATCH (d:Document {name: $doc})
                MERGE (d)-[:HAS_CHUNK]->(ch)
                """,
                id=chunk_id, text=chunk, idx=i,
                emb=embeddings[i].tolist(), doc=doc_name,
            )


def ingest_instances(driver, instances: dict, doc_name: str):
    with driver.session() as session:
        for e in instances["entities"]:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type, e.description = $desc
                WITH e
                MATCH (ch:Chunk {id: $chunk_id})
                MERGE (ch)-[:HAS_ENTITY]->(e)
                """,
                name=e["name"],
                type=e.get("type", "OTHER"),
                desc=e.get("description", ""),
                chunk_id=f"{doc_name}_chunk_{e.get('chunk_idx', 0)}",
            )

        for r in instances["relationships"]:
            rel_type = re.sub(r"[^A-Z0-9_]", "_",
                              r.get("type", "related_to").upper().replace(" ", "_"))
            if not rel_type:
                rel_type = "RELATED_TO"
            try:
                session.run(
                    f"""
                    MATCH (a:Entity {{name: $source}})
                    MATCH (b:Entity {{name: $target}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.evidence = $evidence
                    """,
                    source=r["source"], target=r["target"],
                    evidence=r.get("evidence", ""),
                )
            except Exception as ex:
                print(f"       Warning: Could not create rel {rel_type}: {ex}")


def ingest_concepts(driver, concepts: list[dict]):
    with driver.session() as session:
        for c in concepts:
            session.run(
                "MERGE (co:Concept {name: $name}) SET co.description = $desc",
                name=c["name"], desc=c.get("description", ""),
            )
            for member in c.get("members", []):
                session.run(
                    """
                    MATCH (co:Concept {name: $concept})
                    MATCH (e:Entity {name: $entity})
                    MERGE (co)-[:INSTANTIATED_BY]->(e)
                    """,
                    concept=c["name"], entity=member,
                )


def ingest_communities(driver, communities: list[dict], concepts: list[dict], embed_model):
    with driver.session() as session:
        for comm in communities:
            summary_emb = embed_model.encode([comm["summary"]])[0].tolist()
            session.run(
                """
                MERGE (cm:Community {id: $id})
                SET cm.name = $name, cm.summary = $summary, cm.embedding = $emb
                """,
                id=comm["id"], name=comm["name"],
                summary=comm["summary"], emb=summary_emb,
            )
            for ci in comm["concept_indices"]:
                session.run(
                    """
                    MATCH (cm:Community {id: $comm_id})
                    MATCH (co:Concept {name: $concept})
                    MERGE (cm)-[:CONTAINS]->(co)
                    """,
                    comm_id=comm["id"], concept=concepts[ci]["name"],
                )


# =============================================================================
#  MAIN BUILD PIPELINE
# =============================================================================

BATCH_SIZE = 3   # chunks per LLM extraction call


def build_graph(input_path: str, use_cache: bool = True) -> dict:
    """Build the full 3-layer hierarchical discourse graph.

    Args:
        input_path: Path to a single .pdf, a single .txt, or a directory
                    containing .txt and/or .pdf files.
        use_cache: Whether to use disk cache for LLM extraction results.
    """
    start = time.time()
    client = Groq(api_key=config.GROQ_API_KEY)

    print(f"[0/6] Loading embedding model: {config.EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    # ── Step 1: Parse + Chunk ────────────────────────────────────────────────
    if os.path.isdir(input_path):
        print(f"[1/6] Loading all documents from directory: {input_path}")
        text, doc_name = load_data_dir(input_path)
    elif input_path.lower().endswith(".txt"):
        print(f"[1/6] Loading TXT: {input_path}")
        text = load_txt(input_path)
        doc_name = os.path.basename(input_path)
    else:
        print(f"[1/6] Loading PDF: {input_path}")
        text = load_pdf(input_path)
        doc_name = os.path.basename(input_path)
    print(f"       {len(text)} characters extracted")

    print("[2/6] Chunking text")
    chunks = chunk_text(text)
    print(f"       {len(chunks)} chunks created")

    # ── Step 2: Embed chunks ─────────────────────────────────────────────────
    print("[3/6] Embedding chunks with bge-base-en-v1.5 (local)")
    chunk_embeddings = embed_model.encode(chunks, show_progress_bar=False)
    chunk_embeddings = np.array(chunk_embeddings, dtype="float32")

    # ── Step 3: Instance Extraction (Layer 1) — BATCHED + CACHED ────────────
    llm_calls_used = 0
    cached_extractions = None
    if use_cache:
        cached_extractions = load_cache(input_path)

    if cached_extractions is not None:
        all_extractions = cached_extractions.get("all_extractions", [])
        print(f"[4/6] Layer 1: Loaded {len(all_extractions)} chunk results from cache "
              f"(0 LLM calls used)")
    else:
        print(f"[4/6] Layer 1: Extracting instances "
              f"(batched {BATCH_SIZE} chunks/call, "
              f"~{-(-len(chunks) // BATCH_SIZE)} LLM calls) ...")
        all_extractions = []

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start: batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, len(chunks))
            print(f"       Batch chunks {batch_start + 1}-{batch_end}/{len(chunks)} "
                  f"({len(batch)} chunks in 1 call) ...", end=" ", flush=True)

            try:
                batch_results = extract_instances_batch(batch, batch_start, client)
            except TokenLimitError:
                # If batch is too large, try one chunk at a time
                print(f"[token limit — splitting]")
                batch_results = []
                for ci, single_chunk in enumerate(batch):
                    try:
                        sr = extract_instances_batch([single_chunk], batch_start + ci, client)
                        batch_results.extend(sr)
                    except TokenLimitError:
                        print(f"       Chunk {batch_start + ci} too large, skipping")
                        batch_results.append({"entities": [], "relationships": []})

            llm_calls_used += 1

            totals_e = sum(len(r.get("entities", [])) for r in batch_results)
            totals_r = sum(len(r.get("relationships", [])) for r in batch_results)
            print(f"=> {totals_e} entities, {totals_r} relationships")

            # Retry once if rate limit caused 0 results for non-empty chunks
            if totals_e == 0 and any(c.strip() for c in batch):
                print(f"       [Retry] 0 entities — waiting 30s and retrying...", end=" ", flush=True)
                time.sleep(30)
                try:
                    batch_results = extract_instances_batch(batch, batch_start, client)
                    totals_e = sum(len(r.get("entities", [])) for r in batch_results)
                    totals_r = sum(len(r.get("relationships", [])) for r in batch_results)
                    print(f"=> {totals_e} entities, {totals_r} relationships")
                except Exception as e:
                    print(f"retry failed: {e}")

            all_extractions.extend(batch_results)

        save_cache(input_path, {"all_extractions": all_extractions})
        print(f"       LLM calls for extraction: {llm_calls_used}")

    instances = merge_instance_extractions(all_extractions, embed_model=embed_model)
    print(f"       Total unique: {len(instances['entities'])} entities, "
          f"{len(instances['relationships'])} relationships")

    # ── Step 4: Concept Generation (Layer 2) ─────────────────────────────────
    # Cooldown before concept generation to avoid rate-limit carry-over
    if not cached_extractions:
        cooldown = 30
        print(f"\n       [Cooldown] Waiting {cooldown}s before concept generation "
              f"to avoid rate limits...")
        time.sleep(cooldown)

    print("[5/6] Layer 2: Generating concepts (1 LLM call) ...")
    concepts = generate_concepts(instances["entities"], client)

    # Retry if rate limit produced 0 concepts
    if not concepts:
        for retry in range(1, 4):
            wait = 60 * retry
            print(f"       [Retry] 0 concepts returned — waiting {wait}s "
                  f"(attempt {retry}/3) ...")
            time.sleep(wait)
            concepts = generate_concepts(instances["entities"], client)
            if concepts:
                break

    print(f"       {len(concepts)} concepts created")
    for c in concepts:
        members_preview = ", ".join(c.get("members", [])[:3])
        print(f"         - {c['name']}: {members_preview}...")
    time.sleep(3)  # buffer before community summaries

    # ── Step 5: Community Detection (Layer 3) ─────────────────────────────────
    print("[6/6] Layer 3: Detecting communities ...")
    community_indices = detect_communities(concepts)
    print(f"       {len(community_indices)} communities detected")
    communities = generate_community_summaries(community_indices, concepts, client)
    for comm in communities:
        print(f"         - {comm['name']}: {comm['summary'][:80]}...")

    # ── Step 6: Ingest into Neo4j ─────────────────────────────────────────────
    print("\n[Neo4j] Ingesting 3-layer graph ...")
    driver = get_driver()
    try:
        clear_graph(driver)
        setup_schema(driver)
        ingest_document(driver, doc_name)
        ingest_chunks(driver, chunks, doc_name, chunk_embeddings)
        ingest_instances(driver, instances, doc_name)
        ingest_concepts(driver, concepts)
        ingest_communities(driver, communities, concepts, embed_model)
        print("[Neo4j] Ingestion complete!")
    finally:
        driver.close()

    elapsed = time.time() - start
    stats = {
        "document": doc_name,
        "characters": len(text),
        "chunks": len(chunks),
        "llm_calls": llm_calls_used + 1 + len(communities),  # extract + concept + communities
        "entities": len(instances["entities"]),
        "relationships": len(instances["relationships"]),
        "concepts": len(concepts),
        "communities": len(communities),
        "build_time_s": round(elapsed, 2),
    }

    print(f"\n{'=' * 55}")
    print("  Graph Build Summary")
    print(f"{'=' * 55}")
    for k, v in stats.items():
        print(f"  {k:<20}: {v}")
    print(f"{'=' * 55}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "graph_build_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build 3-Layer Hierarchical Discourse Graph"
    )
    parser.add_argument("--input", "-i",
                        help="Path to a .pdf, .txt, or a directory of .txt/.pdf files")
    parser.add_argument("--pdf", default=None,
                        help="(Legacy) Path to input PDF — prefer --input")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore existing cache and re-run all LLM extractions"
    )
    args = parser.parse_args()

    # Resolve input path: --input takes priority, then --pdf, then default
    input_path = args.input or args.pdf or os.path.join("data", "meridian")

    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        return

    build_graph(input_path, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
