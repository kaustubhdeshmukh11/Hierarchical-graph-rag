import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Standard Graph RAG — Flat Entity-Relationship Graph + Vector Retrieval.

A conventional Graph RAG approach that combines:
  1. Vector embedding retrieval (same as baseline)
  2. A flat entity-relationship knowledge graph (no hierarchical layers)

This does NOT use hierarchical community detection or concept abstraction.
It represents the typical "knowledge graph + RAG" approach found in most
Graph RAG literature.

Key Differences from Hierarchical Graph RAG:
  - NO community detection (Layer 1 absent)
  - NO concept hierarchy / abstraction (Layer 2 absent)
  - Single flat graph of entities + relationships
  - Retrieval: vector similarity + 1-hop entity expansion
  - No cross-community synthesis for global queries

Usage:
    python standard_graph_rag.py --input data/meridian --query "Who founded Meridian?"
"""

import argparse
import json
import os
import time

import fitz  # PyMuPDF
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

import config
from lib.groq_utils import strip_think, safe_groq_call, parse_json_from_llm


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT LOADING & CHUNKING (shared with baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_input(input_path: str) -> tuple[str, str]:
    """Load text from a .pdf, .txt, or directory of mixed files."""
    if os.path.isdir(input_path):
        texts, files_loaded = [], []
        for fname in sorted(os.listdir(input_path)):
            fpath = os.path.join(input_path, fname)
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
            raise FileNotFoundError(f"No .txt or .pdf files in {input_path}")
        print(f"       Loaded {len(files_loaded)} files: {', '.join(files_loaded[:10])}")
        combined = "\n\n".join(texts)
        doc_name = os.path.basename(input_path.rstrip('/\\')) or "multi_doc"
        return combined, doc_name
    elif input_path.lower().endswith(".txt"):
        return load_txt(input_path), os.path.basename(input_path)
    else:
        return load_pdf(input_path), os.path.basename(input_path)


def chunk_text(text: str, size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
#  FLAT ENTITY EXTRACTION (no hierarchy — just entities + relationships)
# ═══════════════════════════════════════════════════════════════════════════════

ENTITY_EXTRACT_SYSTEM = (
    "You are an expert knowledge graph builder. You extract named entities "
    "and their relationships from text. Output valid JSON only."
)

ENTITY_EXTRACT_PROMPT = """\
Extract all named entities and relationships from the text below.

TEXT:
{text}

Return ONLY valid JSON:
{{
  "entities": [
    {{"name": "Entity Name", "type": "PERSON|ORG|PRODUCT|EVENT|LOCATION|DATE|METRIC"}}
  ],
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "type": "RELATIONSHIP_TYPE", "evidence": "brief quote"}}
  ]
}}
"""


def extract_entities_from_chunk(chunk: str, client: Groq) -> dict:
    """Extract entities and relationships from a single chunk using LLM."""
    prompt = ENTITY_EXTRACT_PROMPT.format(text=chunk[:2000])
    try:
        response = safe_groq_call(
            client, prompt,
            system_message=ENTITY_EXTRACT_SYSTEM,
            max_tokens=1500,
        )
        result = parse_json_from_llm(response)
        if isinstance(result, dict):
            return {
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", []),
            }
    except Exception as e:
        print(f"  [EntityExtract] Error: {e}")
    return {"entities": [], "relationships": []}


def build_flat_graph(chunks: list[str], client: Groq) -> dict:
    """Build a flat entity-relationship graph from all chunks.

    Returns dict with 'entities', 'relationships', 'entity_to_chunks' mapping.
    """
    all_entities = {}
    all_relationships = []
    entity_to_chunks = {}  # entity_name -> list of chunk indices

    print(f"  [FlatGraph] Extracting entities from {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        extraction = extract_entities_from_chunk(chunk, client)

        for ent in extraction["entities"]:
            name = ent.get("name", "").strip()
            if name:
                name_lower = name.lower()
                if name_lower not in all_entities:
                    all_entities[name_lower] = {
                        "name": name,
                        "type": ent.get("type", "UNKNOWN"),
                        "chunk_indices": [],
                    }
                all_entities[name_lower]["chunk_indices"].append(i)

        for rel in extraction["relationships"]:
            rel["source_chunk"] = i
            all_relationships.append(rel)

        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{len(chunks)} chunks...")
        time.sleep(config.GROQ_INTER_CALL_DELAY)

    print(f"  [FlatGraph] Built graph: {len(all_entities)} entities, "
          f"{len(all_relationships)} relationships")

    return {
        "entities": all_entities,
        "relationships": all_relationships,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL: VECTOR + 1-HOP ENTITY EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════

def build_vector_store(chunks: list[str]) -> tuple:
    """Build FAISS index for vector retrieval."""
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(embeddings)
    return index, embeddings, model


def retrieve_with_graph(query: str, index, chunks: list[str], model,
                        graph: dict, top_k: int = 7) -> dict:
    """Retrieve chunks via vector similarity + 1-hop entity graph expansion.

    Steps:
    1. Vector similarity retrieval (top-k)
    2. Extract entities from query
    3. Find chunks containing those entities (1-hop expansion)
    4. Merge and deduplicate chunks
    5. Collect relevant relationships
    """
    # Step 1: Vector retrieval
    q_emb = model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    vector_chunks = set(int(i) for i in indices[0] if i < len(chunks))

    # Step 2: Entity matching — find entities mentioned in query
    query_lower = query.lower()
    matched_entities = []
    for ent_key, ent_data in graph["entities"].items():
        if ent_key in query_lower or any(
            word in query_lower for word in ent_key.split() if len(word) > 3
        ):
            matched_entities.append(ent_data)

    # Step 3: 1-hop entity expansion — add chunks containing matched entities
    entity_chunks = set()
    for ent in matched_entities:
        for ci in ent.get("chunk_indices", []):
            entity_chunks.add(ci)

    # Step 4: Merge (vector chunks first, then entity-expanded)
    all_chunk_indices = list(vector_chunks) + [
        ci for ci in entity_chunks if ci not in vector_chunks
    ]
    # Cap total chunks
    all_chunk_indices = all_chunk_indices[:top_k + 3]

    # Step 5: Collect relevant relationships
    relevant_rels = []
    matched_names = set(e["name"].lower() for e in matched_entities)
    for rel in graph["relationships"]:
        src = rel.get("source", "").lower()
        tgt = rel.get("target", "").lower()
        if src in matched_names or tgt in matched_names:
            relevant_rels.append(rel)

    context = [chunks[i] for i in all_chunk_indices if i < len(chunks)]

    return {
        "context_chunks": context,
        "relationships": relevant_rels[:10],
        "retrieval_path": {
            "vector_chunks": len(vector_chunks),
            "entities_matched": len(matched_entities),
            "entity_expanded_chunks": len(entity_chunks),
            "total_chunks": len(context),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM ANSWER GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_answer(query: str, context_chunks: list[str],
                    relationships: list[dict] = None) -> str:
    """Generate answer using retrieved context and entity relationships."""
    client = Groq(api_key=config.GROQ_API_KEY)

    # Build relationship section
    rel_section = ""
    if relationships:
        rel_lines = []
        for i, r in enumerate(relationships[:10]):
            evidence = r.get("evidence", "").strip()[:200]
            if evidence:
                rel_lines.append(
                    f"  {i+1}. {r['source']} —[{r['type']}]→ {r['target']}: "
                    f"\"{evidence}\""
                )
            else:
                rel_lines.append(
                    f"  {i+1}. {r['source']} —[{r['type']}]→ {r['target']}"
                )
        rel_section = (
            "ENTITY RELATIONSHIPS:\n"
            + "\n".join(rel_lines)
            + "\n\n"
        )

    max_chars = getattr(config, "MAX_CHUNK_CHARS", 800)
    truncated = [c[:max_chars] + "..." if len(c) > max_chars else c
                 for c in context_chunks]
    context = "\n\n---\n\n".join(truncated)

    system_msg = (
        "You are a knowledgeable expert assistant. Answer questions using ONLY "
        "the provided context and entity relationships. Be thorough and precise."
    )

    prompt = (
        "Answer the question below using ONLY the provided context.\n\n"
        f"{rel_section}"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            return strip_think(response.choices[0].message.content.strip())
        except Exception as e:
            if "rate_limit" in str(e).lower() or "413" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                raise
    return "Could not generate answer due to API rate limits."


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build(input_path: str) -> dict:
    """Build flat entity graph + vector store from input documents."""
    print(f"\n[Standard GraphRAG] Building from {input_path}")
    text, doc_name = load_input(input_path)
    chunks = chunk_text(text)
    print(f"  {len(chunks)} chunks from {doc_name}")

    client = Groq(api_key=config.GROQ_API_KEY)
    graph = build_flat_graph(chunks, client)
    index, embeddings, model = build_vector_store(chunks)

    return {
        "chunks": chunks,
        "graph": graph,
        "index": index,
        "model": model,
        "doc_name": doc_name,
    }


def query(question: str, pipeline: dict) -> dict:
    """Query using standard graph RAG (vector + entity expansion)."""
    start = time.time()

    retrieval = retrieve_with_graph(
        question, pipeline["index"], pipeline["chunks"],
        pipeline["model"], pipeline["graph"],
    )

    answer = generate_answer(
        question, retrieval["context_chunks"],
        retrieval["relationships"],
    )

    elapsed = time.time() - start

    return {
        "query": question,
        "answer": answer,
        "context": retrieval["context_chunks"],
        "relationships": retrieval["relationships"],
        "retrieval_path": retrieval["retrieval_path"],
        "time_s": round(elapsed, 2),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standard Graph RAG — Flat Entity Graph + Vector Retrieval"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to .pdf, .txt, or directory")
    parser.add_argument("--query", "-q", required=True, help="Query string")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}")
        return

    pipeline = build(args.input)
    result = query(args.query, pipeline)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nRetrieval path: {result['retrieval_path']}")
    print(f"Time: {result['time_s']}s")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "standard_graphrag_last_run.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
