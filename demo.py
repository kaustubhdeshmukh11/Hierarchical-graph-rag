"""
Interactive Demo — Hierarchical Graph RAG Live Presentation.

Shows:
  1. Full retrieval trace (Communities → Concepts → Entities → Chunks)
  2. Final answer generation
  3. Optional side-by-side comparison with baseline RAG

Usage:
    python demo.py                                        # Interactive mode
    python demo.py --query "Who founded Meridian?"        # Single query
    python demo.py --case-study                           # Show multihop case study
"""

import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import os
import textwrap
import time

import config

# ═══════════════════════════════════════════════════════════════════════════════
#  TERMINAL FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

# ANSI color codes
class C:
    HEADER    = "\033[95m"
    BLUE      = "\033[94m"
    CYAN      = "\033[96m"
    GREEN     = "\033[92m"
    YELLOW    = "\033[93m"
    RED       = "\033[91m"
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    UNDERLINE = "\033[4m"
    END       = "\033[0m"


def banner(text: str, width: int = 70):
    """Print a bold banner."""
    print()
    print(f"{C.BOLD}{C.CYAN}{'=' * width}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'=' * width}{C.END}")
    print()


def section(text: str, width: int = 70):
    """Print a section header."""
    print(f"\n{C.BOLD}{C.YELLOW}{'─' * width}{C.END}")
    print(f"  {C.BOLD}{C.YELLOW}{text}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}{'─' * width}{C.END}\n")


def info(text: str):
    print(f"  {C.DIM}{text}{C.END}")


def highlight(text: str):
    print(f"  {C.GREEN}{C.BOLD}{text}{C.END}")


def wrap_print(text: str, indent: int = 4, width: int = 70):
    wrapped = textwrap.fill(text, width=width,
                            initial_indent=" " * indent,
                            subsequent_indent=" " * indent)
    print(wrapped)


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE STUDY (pre-built, no API needed)
# ═══════════════════════════════════════════════════════════════════════════════

def show_case_study():
    """Show the multi-hop case study comparison."""
    banner("MULTI-HOP CASE STUDY: Baseline RAG vs Graph RAG")

    query = "How did the Incident Nightfall data breach affect Meridian's product development and research direction?"

    print(f"  {C.BOLD}Query:{C.END} {C.CYAN}{query}{C.END}")
    print()

    ground_truth = (
        "The breach led to Project Fortress ($5 million security overhaul), "
        "the hiring of Arjun Mehta as CISO, and forced layoffs of 35 employees. "
        "It directly influenced the AI research direction: Dr. Rahman and Arjun Mehta "
        "collaborated on federated learning techniques that could train MediScan models "
        "without centralizing patient data, published at NeurIPS 2020. They also filed "
        "a joint patent for the privacy-preserving framework."
    )

    print(f"  {C.BOLD}Ground Truth:{C.END}")
    wrap_print(ground_truth)
    print()

    # ── Baseline RAG ─────────────────────────────────────
    section("BASELINE RAG (Vector Similarity Only)")

    info("Retrieved 5 chunks via FAISS cosine similarity")
    info("Chunks primarily from: 03_data_breach_crisis.txt (3/5)")
    info("Missing: No chunks about federated learning or NeurIPS")
    print()

    baseline_answer = (
        "The Incident Nightfall data breach in September 2019 compromised 4.2 million "
        "patient records. In response, Meridian launched Project Fortress, a $5 million "
        "security overhaul, and promoted Arjun Mehta to CISO. The company also conducted "
        "layoffs of 35 employees due to the financial impact of lost hospital contracts. "
        "The available context does not provide enough detail to fully trace the remaining "
        "connections to product development and research direction."
    )

    print(f"  {C.BOLD}Answer:{C.END}")
    wrap_print(baseline_answer)
    print()

    print(f"  {C.BOLD}Metrics:{C.END}")
    print(f"    Answer Relevancy:    {C.YELLOW}0.7234{C.END}")
    print(f"    Faithfulness:        {C.RED}0.5800{C.END}")
    print(f"    Context Recall:      {C.RED}0.3750{C.END}  {C.DIM}<-- only 1.5/4 ground truth facts{C.END}")
    print(f"    Avg Relevance Score: {C.YELLOW}0.5741{C.END}")
    print()

    print(f"  {C.RED}{C.BOLD}Issue:{C.END} {C.RED}Baseline cannot connect 'data breach' to 'federated learning'{C.END}")
    print(f"  {C.RED}because they have low lexical/semantic overlap.{C.END}")

    # ── Graph RAG ─────────────────────────────────────────
    section("GRAPH RAG (Hierarchical Retrieval)")

    print(f"  {C.BOLD}Step 1: Community Matching (Layer 3){C.END}")
    print(f"    {C.GREEN}[####################..........] 0.7821{C.END}  Community 2 (Security & Crisis) {C.GREEN}<-- SELECTED{C.END}")
    print(f"    {C.GREEN}[################..............] 0.6234{C.END}  Community 3 (AI Research)       {C.GREEN}<-- SELECTED{C.END}")
    print()

    print(f"  {C.BOLD}Step 2: Concept Drill-Down (Layer 2){C.END}")
    print(f"    {C.CYAN}-> \"Data Breach Response and Security Overhaul\"{C.END}")
    print(f"    {C.CYAN}-> \"Privacy-Preserving AI Research\"{C.END}")
    print(f"    {C.CYAN}-> \"Leadership During Crisis\"{C.END}")
    print()

    print(f"  {C.BOLD}Step 3: Entity Collection + BFS (Layer 1){C.END}")
    print(f"    Seed entities: Incident Nightfall, Arjun Mehta, Project Fortress")
    print(f"    {C.BOLD}BFS Hop 1:{C.END}")
    print(f"      {C.BLUE}Incident Nightfall --[caused]--> Project Fortress{C.END}")
    print(f"      {C.BLUE}Incident Nightfall --[led_to]--> Layoffs{C.END}")
    print(f"      {C.BLUE}Arjun Mehta --[promoted_to]--> CISO{C.END}")
    print(f"    {C.BOLD}BFS Hop 2:{C.END}")
    print(f"      {C.GREEN}Arjun Mehta --[collaborated_with]--> Dr. Aisha Rahman{C.END}")
    print(f"      {C.GREEN}Dr. Aisha Rahman --[developed]--> Federated Learning Framework{C.END}")
    print(f"      {C.GREEN}Federated Learning Framework --[published_at]--> NeurIPS 2020{C.END}")
    print()

    print(f"  {C.BOLD}Step 4: Chunk Assembly{C.END}")
    info("8 chunks assembled from 3 source documents")
    print()

    graphrag_answer = (
        "The Incident Nightfall data breach in September 2019 had far-reaching effects "
        "on Meridian's trajectory. Immediately, Meridian launched Project Fortress, a "
        "$5 million security overhaul led by Arjun Mehta, who was promoted to Chief "
        "Information Security Officer (CISO). The financial fallout also forced layoffs "
        "of 35 employees. More significantly, the breach directly reshaped Meridian's "
        "AI research direction: Dr. Aisha Rahman and Arjun Mehta collaborated on "
        "privacy-preserving federated learning techniques that could train MediScan "
        "diagnostic models across hospitals without centralizing sensitive patient data. "
        "This research was published at NeurIPS 2020 and they jointly filed a patent "
        "for the federated learning framework in August 2021."
    )

    print(f"  {C.BOLD}Answer:{C.END}")
    wrap_print(graphrag_answer)
    print()

    print(f"  {C.BOLD}Metrics:{C.END}")
    print(f"    Answer Relevancy:    {C.GREEN}0.8123{C.END}")
    print(f"    Faithfulness:        {C.GREEN}0.8750{C.END}")
    print(f"    Context Recall:      {C.GREEN}0.8750{C.END}  {C.DIM}<-- 3.5/4 ground truth facts{C.END}")
    print(f"    Avg Relevance Score: {C.GREEN}0.7211{C.END}")
    print()

    highlight("Graph RAG connects the dots via BFS entity traversal:")
    print(f"    Incident Nightfall {C.DIM}-->{C.END} Arjun Mehta {C.DIM}-->{C.END} Dr. Rahman {C.DIM}-->{C.END} Federated Learning")
    print()

    # ── Side by Side ──────────────────────────────────────
    section("COMPARISON SUMMARY")

    headers = f"  {'Aspect':<35} {'Baseline RAG':<18} {'Graph RAG':<18}"
    print(f"{C.BOLD}{headers}{C.END}")
    print(f"  {'─' * 65}")
    comparisons = [
        ("Documents accessed",          "1-2",     "3"),
        ("Chunks retrieved",            "5",       "8"),
        ("Mentions Project Fortress",   "Yes",     "Yes"),
        ("Mentions Arjun Mehta as CISO","Yes",     "Yes"),
        ("Mentions layoffs",            "Yes",     "Yes"),
        ("Connects to federated learning","No",    "Yes"),
        ("Mentions NeurIPS publication", "No",      "Yes"),
        ("Mentions patent filing",       "No",      "Yes"),
        ("Context Recall",              "0.375",   "0.875"),
        ("Overall completeness",        "Partial", "Complete"),
    ]
    for aspect, baseline, graphrag in comparisons:
        b_color = C.RED if baseline in ("No", "Partial", "0.50") else C.END
        g_color = C.GREEN if graphrag in ("Yes", "Complete", "0.875", "3", "8") else C.END
        print(f"  {aspect:<35} {b_color}{baseline:<18}{C.END} {g_color}{graphrag:<18}{C.END}")

    print()
    highlight("Conclusion: Knowledge graph traversal enables cross-document reasoning")
    highlight("that pure vector similarity cannot achieve.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE QUERY (requires Neo4j + Groq API)
# ═══════════════════════════════════════════════════════════════════════════════

def run_live_query(question: str):
    """Run a live query through the graph RAG pipeline with trace output."""
    try:
        from sentence_transformers import SentenceTransformer
        from neo4j import GraphDatabase
        from groq import Groq
    except ImportError as e:
        print(f"  {C.RED}Missing dependency: {e}{C.END}")
        return

    banner(f"LIVE QUERY: {question}")

    print(f"  {C.DIM}Loading embedding model...{C.END}")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"  {C.DIM}Connecting to Neo4j...{C.END}")
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        print(f"  {C.GREEN}Connected to Neo4j.{C.END}")
    except Exception as e:
        print(f"  {C.RED}Neo4j connection failed: {e}{C.END}")
        print(f"  {C.YELLOW}Use --case-study to see pre-built demo instead.{C.END}")
        return

    # Import and run the trace
    import trace_query
    groq_client = Groq(api_key=config.GROQ_API_KEY)

    section("RETRIEVAL TRACE")

    trace = trace_query.trace_single_query(
        question=question,
        label="Live Query",
        why="User-specified query for live demonstration.",
        embed_model=embed_model,
        driver=driver,
        groq_client=groq_client,
    )

    print(trace)
    driver.close()

    # Save trace
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "demo_trace.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(trace)
    print(f"\n  {C.DIM}Trace saved: {out_path}{C.END}")


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_mode():
    """Run interactive demo loop."""
    banner("HIERARCHICAL GRAPH RAG — Interactive Demo")

    print(f"  {C.BOLD}Dataset:{C.END} Meridian Health Systems (6 documents)")
    print(f"  {C.BOLD}Graph:{C.END}   3-Layer (Community -> Concept -> Entity)")
    print(f"  {C.BOLD}LLM:{C.END}    {config.GROQ_MODEL}")
    print()

    print(f"  {C.BOLD}Available commands:{C.END}")
    print(f"    {C.CYAN}1{C.END} — Show multi-hop case study (no API needed)")
    print(f"    {C.CYAN}2{C.END} — Ask a custom query (requires Neo4j + Groq)")
    print(f"    {C.CYAN}3{C.END} — Show sample queries")
    print(f"    {C.CYAN}q{C.END} — Quit")
    print()

    sample_queries = [
        "Who founded Meridian Health Systems and when?",
        "What are all the major products Meridian developed?",
        "How did the Incident Nightfall breach affect research direction?",
        "How did the NovaCare partnership evolve through COVID-19?",
        "What role did Priya Sharma play across different crises?",
        "Summarize the key leadership team at Meridian.",
    ]

    while True:
        try:
            choice = input(f"  {C.BOLD}>{C.END} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "q" or choice == "quit":
            break
        elif choice == "1":
            show_case_study()
        elif choice == "2":
            query = input(f"  {C.CYAN}Enter query:{C.END} ").strip()
            if query:
                run_live_query(query)
        elif choice == "3":
            print(f"\n  {C.BOLD}Sample queries:{C.END}")
            for i, q in enumerate(sample_queries, 1):
                print(f"    {C.CYAN}{i}.{C.END} {q}")
            print()
        else:
            # Treat as direct query
            if choice:
                run_live_query(choice)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Graph RAG — Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python demo.py                             # Interactive mode
              python demo.py --case-study                # Show multi-hop comparison
              python demo.py --query "Who founded Meridian?"  # Single query
        """),
    )
    parser.add_argument("--query", help="Run a single query and show trace")
    parser.add_argument("--case-study", action="store_true",
                        help="Show the multi-hop case study comparison")
    args = parser.parse_args()

    if args.case_study:
        show_case_study()
    elif args.query:
        run_live_query(args.query)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
