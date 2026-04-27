"""
Graph Retriever — Hierarchical Traverse-and-Collect Retrieval.

Novel retrieval strategy that works TOP-DOWN through the 3-layer graph:
  1. Match query to Communities (via embedding similarity + threshold filter)
  2. Drill down to Concepts, rerank by query relevance, filter by min score, take top-K
  3. Collect Instance entities via batched BFS on discourse edges
  4. Rerank entities by query relevance, cap at MAX_ENTITIES
  5. Collect extracted relationships between retrieved entities
  6. Assemble context from original chunks in logical order

Quality improvements over naive version:
  - Community similarity threshold (skip low-relevance communities)
  - Concept reranking + min relevance filter (only query-relevant concepts feed into BFS)
  - Batched BFS (1 Cypher query per hop instead of 2N)
  - Entity reranking + cap (focus context on most relevant entities)
  - Extracted relationships sent alongside chunks to LLM

Usage:
    Imported by graph_rag.py — not typically called directly.
"""

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

import config


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_driver():
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )


# ─── Step 1: Community Matching (with similarity threshold) ─────────────────

def match_communities(query_embedding: np.ndarray, driver,
                      top_k: int = None,
                      min_similarity: float = None) -> list[dict]:
    """Find the most relevant communities by cosine similarity on summaries.

    Applies both top-k selection AND minimum similarity threshold so that
    irrelevant communities are never passed to the concept drill step.
    """
    top_k = top_k or config.TOP_K_COMMUNITIES
    min_similarity = min_similarity if min_similarity is not None else config.MIN_COMMUNITY_SIMILARITY

    with driver.session() as session:
        result = session.run(
            "MATCH (cm:Community) RETURN cm.id AS id, cm.name AS name, "
            "cm.summary AS summary, cm.embedding AS embedding"
        )
        communities = []
        for record in result:
            emb = record["embedding"]
            if emb is not None:
                emb = np.array(emb, dtype="float32")
                sim = _cosine_sim(query_embedding, emb)
                communities.append({
                    "id": record["id"],
                    "name": record["name"],
                    "summary": record["summary"],
                    "similarity": sim,
                })

    # Sort by similarity descending
    communities.sort(key=lambda x: x["similarity"], reverse=True)

    # Apply threshold filter, then take top_k
    filtered = [c for c in communities if c["similarity"] >= min_similarity]
    if not filtered:
        # Fallback: if nothing passes threshold, take best 1
        filtered = communities[:1] if communities else []

    return filtered[:top_k]


# ─── Step 2: Concept Drill + Reranking + Min-Relevance Filter ───────────────

def drill_to_concepts(community_ids: list[str], driver,
                      query_embedding: np.ndarray,
                      embed_model: SentenceTransformer,
                      top_k: int = None,
                      min_relevance: float = None) -> list[dict]:
    """Follow CONTAINS edges from communities to find concepts, then rerank
    by cosine similarity between concept description and query embedding.

    Only returns concepts with relevance >= min_relevance, capped at top-K.
    """
    top_k = top_k or config.TOP_K_CONCEPTS
    min_relevance = min_relevance if min_relevance is not None else config.MIN_CONCEPT_RELEVANCE
    concepts = []
    seen = set()

    with driver.session() as session:
        for comm_id in community_ids:
            result = session.run(
                """
                MATCH (cm:Community {id: $id})-[:CONTAINS]->(co:Concept)
                RETURN co.name AS name, co.description AS description
                """,
                id=comm_id,
            )
            for record in result:
                name = record["name"]
                if name not in seen:
                    seen.add(name)
                    concepts.append({
                        "name": name,
                        "description": record["description"] or "",
                    })

    # Rerank concepts by embedding similarity to query
    if concepts and embed_model is not None:
        desc_texts = [c["description"] or c["name"] for c in concepts]
        desc_embs = embed_model.encode(desc_texts, show_progress_bar=False)
        for i, c in enumerate(concepts):
            c["relevance"] = _cosine_sim(query_embedding, desc_embs[i].astype("float32"))

        concepts.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    else:
        for c in concepts:
            c["relevance"] = 0.0

    # Apply minimum relevance filter
    filtered = [c for c in concepts if c.get("relevance", 0) >= min_relevance]
    if not filtered and concepts:
        # Fallback: if nothing passes threshold, take best 1
        filtered = concepts[:1]

    return filtered[:top_k]


# ─── Step 3: Batched BFS Instance Collection ────────────────────────────────

def collect_instances(concept_names: list[str], driver,
                      query_embedding: np.ndarray = None,
                      embed_model: SentenceTransformer = None,
                      max_hops: int = None,
                      max_entities: int = None) -> list[dict]:
    """BFS from concept members through discourse edges to collect entities.

    Fixes over naive version:
      - Single batched Cypher query per hop (instead of 2 per entity per hop)
      - Entity reranking by query similarity at the end
      - Entity cap to avoid context explosion
    """
    max_hops = max_hops or config.MAX_HOPS
    max_entities = max_entities or config.MAX_ENTITIES
    entities = []
    seen = set()

    with driver.session() as session:
        # Get seed entities from all concepts in one batched query
        result = session.run(
            """
            MATCH (co:Concept)-[:INSTANTIATED_BY]->(e:Entity)
            WHERE co.name IN $names
            RETURN e.name AS name, e.type AS type, e.description AS description,
                   co.name AS from_concept
            """,
            names=concept_names,
        )
        seed_names = []
        for record in result:
            ename = record["name"]
            if ename not in seen:
                seen.add(ename)
                entities.append({
                    "name": ename,
                    "type": record["type"],
                    "description": record["description"] or "",
                    "from_concept": record["from_concept"],
                })
                seed_names.append(ename)

        # BFS: batched query per hop
        frontier = list(seed_names)
        for hop in range(max_hops):
            if not frontier:
                break

            # Single query for ALL frontier entities — both directions
            result = session.run(
                """
                MATCH (a:Entity)-[r]-(b:Entity)
                WHERE a.name IN $frontier AND NOT b.name IN $seen
                RETURN DISTINCT
                    a.name AS src, b.name AS tgt,
                    b.type AS type, b.description AS desc,
                    type(r) AS rel_type
                """,
                frontier=frontier,
                seen=list(seen),
            )

            next_frontier = []
            for record in result:
                neighbor = record["tgt"]
                if neighbor not in seen:
                    seen.add(neighbor)
                    entities.append({
                        "name": neighbor,
                        "type": record["type"],
                        "description": record["desc"] or "",
                        "reached_via": f"{record['src']} --[{record['rel_type']}]--> {neighbor}",
                    })
                    next_frontier.append(neighbor)

            frontier = next_frontier

    # ── Entity reranking by query similarity ────────────────────────────────
    if query_embedding is not None and embed_model is not None and entities:
        desc_texts = [
            f"{e['name']}: {e.get('description', '')}" for e in entities
        ]
        ent_embs = embed_model.encode(desc_texts, show_progress_bar=False)
        for i, e in enumerate(entities):
            e["relevance"] = _cosine_sim(query_embedding, ent_embs[i].astype("float32"))

        entities.sort(key=lambda x: x.get("relevance", 0), reverse=True)

    return entities[:max_entities]


# ─── Step 4: Collect Extracted Relationships (expanded for multi-hop) ────────

def collect_relationships(entity_names: list[str], driver) -> list[dict]:
    """Query Neo4j for all relationship edges between the retrieved entities.

    Returns a list of dicts with source, target, type, evidence.

    Multi-hop improvement: Also finds relationships where only ONE endpoint is
    in our entity set — this catches cross-document bridging relationships
    that connect entities from different parts of the graph.
    """
    if not entity_names:
        return []

    relationships = []
    seen_rels = set()

    with driver.session() as session:
        # Primary: relationships between retrieved entities (both endpoints known)
        result = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.name IN $names AND b.name IN $names
            RETURN a.name AS source, b.name AS target,
                   type(r) AS rel_type, r.evidence AS evidence
            """,
            names=entity_names,
        )
        for record in result:
            rel_key = f"{record['source']}|{record['rel_type']}|{record['target']}"
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                relationships.append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["rel_type"],
                    "evidence": record["evidence"] or "",
                })

        # Secondary: relationships extending ONE hop beyond retrieved entities
        # This catches bridging edges that link to entities in other documents
        result = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE (a.name IN $names OR b.name IN $names)
              AND NOT (a.name IN $names AND b.name IN $names)
            RETURN a.name AS source, b.name AS target,
                   type(r) AS rel_type, r.evidence AS evidence
            LIMIT 30
            """,
            names=entity_names,
        )
        for record in result:
            rel_key = f"{record['source']}|{record['rel_type']}|{record['target']}"
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                relationships.append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["rel_type"],
                    "evidence": record["evidence"] or "",
                })

    return relationships


def order_relationship_chain(relationships: list[dict]) -> list[dict]:
    """Topologically sort relationships into narrative chains.

    Instead of random order, relationships are ordered so that the target
    of one relationship becomes the source of the next, forming readable
    causal chains that dramatically improve LLM multi-hop reasoning.
    """
    if len(relationships) <= 1:
        return relationships

    # Build adjacency from source -> list of relationship indices
    source_map = {}  # entity_name -> [rel_indices where it's source]
    target_set = set()
    for i, r in enumerate(relationships):
        source_map.setdefault(r["source"], []).append(i)
        target_set.add(r["target"])

    # Find chain starters: sources that are NOT targets of any relationship
    starters = []
    for src in source_map:
        if src not in target_set:
            starters.append(src)

    # If no clear starters, use all sources as potential starters
    if not starters:
        starters = list(source_map.keys())

    # Greedy chain building: follow source -> target chains
    ordered = []
    used = set()

    for start in starters:
        current = start
        while current in source_map:
            added_any = False
            for idx in source_map[current]:
                if idx not in used:
                    used.add(idx)
                    ordered.append(relationships[idx])
                    current = relationships[idx]["target"]
                    added_any = True
                    break
            if not added_any:
                break

    # Add any remaining relationships not in chains
    for i, r in enumerate(relationships):
        if i not in used:
            ordered.append(r)

    return ordered


# ─── Step 5: Assemble Context from Original Chunks (multi-hop aware) ────────

def assemble_context(entity_names: list[str], driver,
                     query_embedding: np.ndarray = None,
                     embed_model: SentenceTransformer = None,
                     top_k: int = None,
                     relationship_entity_names: list[str] = None) -> list[str]:
    """Find the original text chunks that contain the collected entities,
    rerank by cosine similarity to query, and cap at top_k for focus.

    Multi-hop improvement: also retrieves chunks for entities discovered
    through relationship edges (relationship_entity_names), ensuring
    cross-document bridging chunks are included.

    This prevents context overload — without capping, BFS on small graphs
    can return 80%+ of all chunks, which kills faithfulness scores.
    """
    top_k = top_k or config.GRAPH_TOP_K_CHUNKS

    # Merge entity names from BFS traversal AND relationship endpoints
    all_entity_names = list(set(entity_names))
    if relationship_entity_names:
        for name in relationship_entity_names:
            if name not in all_entity_names:
                all_entity_names.append(name)

    with driver.session() as session:
        result = session.run(
            """
            MATCH (ch:Chunk)-[:HAS_ENTITY]->(e:Entity)
            WHERE e.name IN $names
            RETURN DISTINCT ch.id AS id, ch.index AS idx, ch.text AS text,
                   ch.embedding AS embedding
            ORDER BY ch.index
            """,
            names=all_entity_names,
        )
        seen_ids = set()
        chunks = []
        for record in result:
            cid = record["id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                chunks.append({
                    "text": record["text"],
                    "idx": record["idx"],
                    "embedding": record["embedding"],
                })

    # Rerank chunks by cosine similarity to query embedding
    if query_embedding is not None and chunks:
        for chunk in chunks:
            if chunk["embedding"] is not None:
                chunk_emb = np.array(chunk["embedding"], dtype="float32")
                chunk["similarity"] = _cosine_sim(query_embedding, chunk_emb)
            else:
                chunk["similarity"] = 0.0
        chunks.sort(key=lambda x: x["similarity"], reverse=True)

    # ── Document-diversity-aware selection ────────────────────────────────
    # Multi-hop queries need facts from MULTIPLE source documents.
    # Pure cosine reranking picks chunks from 1-2 dominant docs only.
    # Fix: pick top-2 chunks from each source doc FIRST, then fill by similarity.

    if len(chunks) > top_k:
        # Detect source document from chunk text header
        def _detect_source(text):
            for line in text.split("\n")[:3]:
                line = line.strip()
                if line and len(line) > 10 and not line.startswith("("):
                    return line[:80]
            return "unknown"

        # Group by source document
        doc_groups = {}
        for chunk in chunks:
            src = _detect_source(chunk["text"])
            doc_groups.setdefault(src, []).append(chunk)

        # Diversity pass: top-2 from each document (multi-hop needs more coverage)
        selected = []
        selected_ids = set()
        for src, group in doc_groups.items():
            for chunk in group[:2]:  # take top-2 per document
                selected.append(chunk)
                selected_ids.add(id(chunk))

        # Fill remaining slots by global similarity ranking
        remaining = top_k - len(selected)
        if remaining > 0:
            for chunk in chunks:
                if id(chunk) not in selected_ids:
                    selected.append(chunk)
                    remaining -= 1
                    if remaining <= 0:
                        break

        chunks = selected[:top_k]
        print(f"  [Retrieve] Diversity reranking: {len(doc_groups)} source docs, "
              f"{len(chunks)} chunks selected")
    else:
        chunks = chunks[:top_k]

    # Re-sort by chunk index for logical reading order
    chunks.sort(key=lambda x: x["idx"])

    return [c["text"] for c in chunks]


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL HIERARCHICAL RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, embed_model: SentenceTransformer = None) -> dict:
    """Run the full hierarchical traverse-and-collect retrieval.

    Returns:
        dict with keys: context_chunks, matched_communities, concepts,
                        entities, relationships, retrieval_path
    """
    if embed_model is None:
        embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    # Embed the query
    query_emb = embed_model.encode([query])[0].astype("float32")

    driver = get_driver()
    try:
        # Step 1: Match Communities (with threshold >= 0.6)
        communities = match_communities(query_emb, driver)
        comm_ids = [c["id"] for c in communities]
        print(f"  [Retrieve] Matched {len(communities)} communities (threshold={config.MIN_COMMUNITY_SIMILARITY}):")
        for c in communities:
            print(f"    - {c['name']} (sim={c['similarity']:.3f}): {c['summary'][:60]}...")

        # Step 2: Drill to Concepts (with reranking + min relevance >= 0.6)
        concepts = drill_to_concepts(comm_ids, driver, query_emb, embed_model)
        concept_names = [c["name"] for c in concepts]
        print(f"  [Retrieve] Reranked to {len(concepts)} concepts "
              f"(top-k={config.TOP_K_CONCEPTS}, min_relevance={config.MIN_CONCEPT_RELEVANCE}):")
        for c in concepts:
            print(f"    - {c['name']} (relevance={c.get('relevance', 0):.3f})")

        # Step 3: Collect Instances (batched BFS + reranking)
        entities = collect_instances(
            concept_names, driver,
            query_embedding=query_emb,
            embed_model=embed_model,
        )
        entity_names = [e["name"] for e in entities]
        print(f"  [Retrieve] Collected {len(entities)} entities "
              f"(capped at {config.MAX_ENTITIES}, reranked by relevance)")

        # Step 4: Collect Relationships between retrieved entities
        relationships = collect_relationships(entity_names, driver)
        print(f"  [Retrieve] Collected {len(relationships)} relationships between entities")

        # Step 4b: Order relationships into narrative chains
        relationships = order_relationship_chain(relationships)

        # Step 4c: Extract entity names from relationship endpoints
        # These may include entities NOT in our BFS set — bridging entities
        # from other documents that are connected via relationship edges
        rel_entity_names = set()
        for r in relationships:
            rel_entity_names.add(r["source"])
            rel_entity_names.add(r["target"])
        # Only keep names that are NOT already in our entity set
        extra_rel_entities = [n for n in rel_entity_names if n not in set(entity_names)]
        if extra_rel_entities:
            print(f"  [Retrieve] Found {len(extra_rel_entities)} bridging entities "
                  f"from relationship edges")

        # Step 5: Assemble Context (batched + reranked + capped)
        # Pass relationship-derived entity names for cross-document chunk retrieval
        context_chunks = assemble_context(entity_names, driver,
                                          query_embedding=query_emb,
                                          embed_model=embed_model,
                                          relationship_entity_names=extra_rel_entities)
        print(f"  [Retrieve] Assembled {len(context_chunks)} context chunks "
              f"(capped at {config.GRAPH_TOP_K_CHUNKS})")

        # Build retrieval path for transparency
        retrieval_path = {
            "communities": [
                {"name": c["name"], "similarity": c["similarity"]}
                for c in communities
            ],
            "concepts": [
                {"name": c["name"], "relevance": c.get("relevance", 0)}
                for c in concepts
            ],
            "entities_traversed": len(entities),
            "relationships_collected": len(relationships),
            "chunks_retrieved": len(context_chunks),
            "discourse_paths": [
                e.get("reached_via", "") for e in entities if e.get("reached_via")
            ],
        }

    finally:
        driver.close()

    return {
        "context_chunks": context_chunks,
        "matched_communities": communities,
        "concepts": concepts,
        "entities": entities,
        "relationships": relationships,
        "retrieval_path": retrieval_path,
    }
