"""
Graph RAG — End-to-End Pipeline.

Combines graph_builder (3-layer construction) and graph_retriever
(hierarchical traverse-and-collect) into a unified pipeline.

Usage:
    # Build graph from PDF:
    python graph_rag.py --pdf data/nexora_industries.pdf --build

    # Query the graph:
    python graph_rag.py --query "How did the acquisition of DataMesh Labs help Nexora?"

    # Build + Query in one shot:
    python graph_rag.py --pdf data/nexora_industries.pdf --build --query "What are Nexora's main products?"
"""

import argparse
import json
import os
import time

from groq import Groq
from sentence_transformers import SentenceTransformer

import config
import graph_builder
import graph_retriever
from lib.groq_utils import strip_think


# ─── LLM Answer Generation ──────────────────────────────────────────────────

def generate_answer(query: str, context_chunks: list[str],
                    retrieval_path: dict,
                    relationships: list[dict] = None) -> str:
    """Generate an answer using Groq, with context from hierarchical retrieval.

    Uses a concise, grounded prompt to avoid faithfulness penalties from
    over-synthesis or speculative reasoning.
    """
    client = Groq(api_key=config.GROQ_API_KEY)

    # Build lightweight relationship context (fewer, no discourse paths)
    rel_info = ""
    if relationships:
        rel_lines = []
        for r in relationships[:8]:  # cap at 8 (was 15 — too much noise)
            rel_lines.append(
                f"  - {r['source']} → {r['type']} → {r['target']}"
            )
        rel_info = (
            "\n\nKEY RELATIONSHIPS:\n"
            + "\n".join(rel_lines)
        )

    context = "\n\n---\n\n".join(context_chunks)

    system_msg = (
        "You are a knowledgeable expert assistant. You answer questions "
        "based strictly on the provided context. You are concise and precise. "
        "You never speculate or add information not directly stated in the context."
    )

    prompt = (
        "Answer the question below using ONLY the provided context.\n\n"
        "RULES:\n"
        "- Be CONCISE: answer in 1-3 short paragraphs maximum\n"
        "- Use ONLY facts explicitly stated in the context\n"
        "- Do NOT infer, speculate, or add interpretive commentary\n"
        "- If the context does not contain enough information, say so clearly\n\n"
        f"CONTEXT:\n{context}"
        f"{rel_info}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

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


# ─── Full Pipeline ───────────────────────────────────────────────────────────

def build(input_path: str) -> dict:
    """Build the 3-layer graph from a PDF, TXT, or directory of files."""
    return graph_builder.build_graph(input_path)


def query(question: str, embed_model: SentenceTransformer = None) -> dict:
    """Query the graph using hierarchical traverse-and-collect retrieval.

    Returns:
        dict with: query, answer, context, retrieval_path, time_s
    """
    start = time.time()

    if embed_model is None:
        embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"\nQuery: {question}")
    print("-" * 50)

    # Hierarchical retrieval
    retrieval = graph_retriever.retrieve(question, embed_model)
    context_chunks = retrieval["context_chunks"]
    relationships = retrieval.get("relationships", [])

    if not context_chunks:
        print("  Warning: No context retrieved. The graph may be empty.")
        return {
            "query": question,
            "answer": "Could not retrieve relevant information from the graph.",
            "context": [],
            "retrieval_path": retrieval["retrieval_path"],
            "time_s": round(time.time() - start, 2),
        }

    # Generate answer with chunks + relationships
    print("\n  [Generate] Sending context + relationships to LLM...")
    answer = generate_answer(question, context_chunks,
                             retrieval["retrieval_path"],
                             relationships)

    elapsed = time.time() - start
    print(f"\n  Answer ({elapsed:.1f}s):\n  {answer[:200]}...")

    return {
        "query": question,
        "answer": answer,
        "context": context_chunks,
        "relationships": relationships,
        "retrieval_path": retrieval["retrieval_path"],
        "time_s": round(elapsed, 2),
    }


def run(pdf_path: str, question: str) -> dict:
    """Build graph + query in one shot."""
    build(pdf_path)
    return query(question)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph RAG — Hierarchical Discourse Graph")
    parser.add_argument("--pdf", help="Path to input PDF")
    parser.add_argument("--build", action="store_true", help="Build the graph from PDF")
    parser.add_argument("--query", help="Query to ask")
    args = parser.parse_args()

    if not args.build and not args.query:
        parser.print_help()
        print("\nExamples:")
        print('  python graph_rag.py --pdf data/nexora_industries.pdf --build')
        print('  python graph_rag.py --query "What is the Cortex knowledge graph?"')
        print('  python graph_rag.py --pdf data/nexora_industries.pdf --build --query "What are Nexora\'s main products?"')
        return

    if args.build:
        if not args.pdf:
            print("Error: --pdf is required when --build is used")
            return
        if not os.path.exists(args.pdf):
            print(f"Error: PDF not found: {args.pdf}")
            return
        build(args.pdf)

    if args.query:
        result = query(args.query)
        print(f"\nFull Answer:\n{result['answer']}")

        # Save result
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(config.RESULTS_DIR, "graphrag_last_run.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
