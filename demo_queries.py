"""
demo_queries.py -- Presentation Demo Queries for the Hierarchical Graph RAG

Fires 3 carefully chosen queries designed to showcase the strengths of each
layer in the 3-layer architecture:

  Query 1: Instance-level  -- "How did the acquisition of DataMesh Labs lead to TrustLayer?"
           Shows multi-hop entity traversal across Instance layer
  Query 2: Concept-level   -- "What products has Nexora Industries built?"
           Shows concept-guided retrieval across the Concept layer
  Query 3: Community-level -- "How did Nexora's strategic direction evolve over time?"
           Shows community matching -> full top-down hierarchical retrieval

Usage:
    python demo_queries.py
"""

import sys
import re
import textwrap
import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

from groq import Groq
import config
import graph_retriever
from lib.groq_utils import strip_think


# ANSI colors for terminal output
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    ORANGE  = "\033[33m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"


DEMO_QUERIES = [
    {
        "id": 1,
        "label": "Instance-Level Traversal",
        "layer": "Layer 1 -- Entity Discourse",
        "color": C.GREEN,
        "question": "How did the acquisition of DataMesh Labs lead to the creation of TrustLayer?",
        "why": (
            "Targets Entity nodes: DataMesh Labs, Sentinel, Chen Wei, Raj Malhotra, "
            "TrustLayer. Tests whether BFS discourse traversal connects "
            "DataMesh Labs -> Sentinel -> GDPR conflict -> TrustLayer."
        ),
    },
    {
        "id": 2,
        "label": "Concept-Level Reasoning",
        "layer": "Layer 2 -- Concept Abstraction",
        "color": C.BLUE,
        "question": "What products has Nexora Industries built and how do they connect?",
        "why": (
            "Targets Concept nodes: Product Portfolio, Enterprise Intelligence. "
            "Tests whether the Concept layer groups Pulse, Athena, Vortex, "
            "and Cortex entities together as related products."
        ),
    },
    {
        "id": 3,
        "label": "Community-Level Synthesis",
        "layer": "Layer 3 -- Community Summary",
        "color": C.RED,
        "question": "How did Nexora's strategic direction evolve from founding to Project Aurora?",
        "why": (
            "Targets Community nodes that cluster strategic themes. "
            "Tests top-down traversal: Community -> Concepts -> Entities -> Chunks, "
            "following Pulse CRM -> Athena AI -> Project Horizon -> Vortex -> Project Aurora."
        ),
    },
]


def separator(char="═", width=68, color=C.DIM):
    print(f"{color}{char * width}{C.RESET}")


def wrap(text: str, indent: int = 4, width: int = 72) -> str:
    return textwrap.fill(text, width=width, initial_indent=" " * indent,
                         subsequent_indent=" " * indent)


def run_query(query: str, embed_model: SentenceTransformer) -> dict:
    """Run the full hierarchical retrieval and return results."""
    return graph_retriever.retrieve(query, embed_model)


def generate_answer(question: str, context_chunks: list[str],
                    relationships: list[dict] = None) -> str:
    """Generate a final answer using the Groq LLM."""
    context = "\n\n---\n\n".join(list(context_chunks)[:5])   # use up to 5 chunks

    # Build relationship context
    rel_info = ""
    if relationships:
        rel_lines = []
        for r in relationships[:10]:
            evidence = f" ({r['evidence']})" if r.get("evidence") else ""
            rel_lines.append(
                f"  - {r['source']} --[{r['type']}]--> {r['target']}{evidence}"
            )
        rel_info = (
            "\n\nEXTRACTED RELATIONSHIPS:\n"
            + "\n".join(rel_lines)
        )

    prompt = (
        "You are a knowledgeable expert. Using ONLY the context and relationships below, "
        "answer the question clearly in 3-5 sentences. Synthesize information "
        "from multiple passages when relevant.\n\n"
        f"CONTEXT:\n{context}"
        f"{rel_info}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a knowledgeable expert assistant "
             "that answers questions strictly from provided context and relationships."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return strip_think(resp.choices[0].message.content.strip())


def print_query_header(q: dict):
    separator("═")
    color = q["color"]
    print(f"\n{color}{C.BOLD}  Query {q['id']}: {q['label']}{C.RESET}")
    print(f"  {C.DIM}{q['layer']}{C.RESET}\n")
    print(f"  {C.WHITE}{C.BOLD}❓ Question:{C.RESET}")
    print(f"  {C.YELLOW}  \"{q['question']}\"{C.RESET}\n")
    print(f"  {C.DIM}Why this query? {q['why']}{C.RESET}")
    separator("─", color=C.DIM)


def print_retrieval_path(result: dict, q_color: str):
    path = result["retrieval_path"]

    # Communities matched
    print(f"\n  {C.RED}{C.BOLD}🔴 Layer 3 — Communities Matched:{C.RESET}")
    for c in path.get("communities", []):
        bar_len = int(c["similarity"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    [{bar}] {c['similarity']:.3f}  {c['name']}")

    # Concepts
    print(f"\n  {C.BLUE}{C.BOLD}🔵 Layer 2 — Concepts Retrieved:{C.RESET}")
    for c in path.get("concepts", []):
        if isinstance(c, dict):
            rel = c.get("relevance", 0)
            print(f"    • {c['name']} (relevance={rel:.3f})")
        else:
            print(f"    • {c}")

    # Instances
    entities_count = path.get("entities_traversed", 0)
    chunks_count = path.get("chunks_retrieved", 0)
    rels_count = path.get("relationships_collected", 0)
    print(f"\n  {C.GREEN}{C.BOLD}🟢 Layer 1 — Entities Traversed: {entities_count}{C.RESET}")
    print(f"  {C.CYAN}   Chunks assembled:           {chunks_count}{C.RESET}")
    print(f"  {C.MAGENTA}   Relationships collected:     {rels_count}{C.RESET}")

    # Sample discourse paths
    discourse = path.get("discourse_paths", [])
    if discourse:
        print(f"\n  {C.DIM}   Sample discourse traversal paths:{C.RESET}")
        for dp in discourse[:4]:
            print(f"  {C.DIM}    ↳ {dp}{C.RESET}")


def main():
    # Header banner
    print()
    separator("═")
    print(f"{C.CYAN}{C.BOLD}")
    print("    ███████╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗")
    print("    ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██║  ██║")
    print("    ███████╗██║   ██║██║  ██║███████║██████╔╝███████║")
    print("    ╚════██║██║   ██║██║  ██║██╔══██║██╔══██╗██╔══██║")
    print("    ███████║╚██████╔╝██████╔╝██║  ██║██║  ██║██║  ██║")
    print("    ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝")
    print(f"{C.RESET}")
    print(f"  {C.WHITE}{C.BOLD}Hierarchical Graph RAG -- 3-Layer Demo Queries{C.RESET}")
    print(f"  {C.DIM}Topic: Nexora Industries  |  Architecture: Community->Concept->Entity{C.RESET}")
    separator("═")
    print()

    print(f"  Loading embedding model ({config.EMBEDDING_MODEL})...")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"  {C.GREEN}✓ Model loaded{C.RESET}\n")

    for q in DEMO_QUERIES:
        print_query_header(q)

        question: str = str(q["question"])
        color: str = str(q["color"])
        qid: int = int(str(q["id"]))

        print(f"\n  {C.CYAN}⚙  Running hierarchical retrieval...{C.RESET}")
        result = run_query(question, embed_model)
        print_retrieval_path(result, color)

        if result["context_chunks"]:
            print(f"\n  {C.YELLOW}{C.BOLD}💡 Generating answer using Groq ({config.GROQ_MODEL})...{C.RESET}")
            answer = generate_answer(
                question, result["context_chunks"],
                result.get("relationships", []),
            )
            print(f"\n  {C.WHITE}{C.BOLD}✅ Answer:{C.RESET}")
            print(wrap(answer, indent=4, width=76))
        else:
            print(f"\n  {C.ORANGE}⚠  No context retrieved. Make sure graph_builder.py has run.{C.RESET}")

        print()
        if qid < len(DEMO_QUERIES):
            time.sleep(2)   # rate-limit buffer between queries

    separator("═")
    print(f"\n  {C.GREEN}{C.BOLD}🏁 Demo complete!{C.RESET}")
    print(f"  {C.DIM}Run `python visualize.py` to view the interactive graph.{C.RESET}\n")
    separator("═")


if __name__ == "__main__":
    main()
