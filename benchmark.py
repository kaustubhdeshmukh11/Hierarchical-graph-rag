"""
Benchmark — Quantitative RAG vs Graph RAG Comparison.

Runs both pipelines on the same question set and compares using:
  - Faithfulness (LLM-as-judge: are answer claims grounded in context?)
  - Answer Relevancy (embedding: does the answer address the question?)
  - Context Precision (embedding: is retrieved context relevant to query?)
  - Context Recall (LLM-as-judge: does context contain ground truth info?)

Aggregates metrics by question category (factual, multi-hop, global)
and generates comparison charts.

Usage:
    python benchmark.py --input data/ --questions data/questions.csv --skip-build
    python benchmark.py --pdf data/doc.pdf --questions data/questions.json
"""

import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

import config
import baseline_rag
import graph_rag
from lib.groq_utils import safe_groq_call, parse_json_from_llm


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM-AS-JUDGE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

FAITHFULNESS_SYSTEM = (
    "You are an impartial evaluation judge. You evaluate whether an answer "
    "is faithfully grounded in the provided context. You always output valid JSON."
)

FAITHFULNESS_PROMPT = """\
Given the CONTEXT and ANSWER below, evaluate faithfulness.

Step 1: Extract every distinct factual claim from the ANSWER (list them).
Step 2: For each claim, determine if it is SUPPORTED or NOT SUPPORTED by the CONTEXT.
Step 3: Compute the faithfulness score = (number of supported claims) / (total claims).

CONTEXT:
{context}

ANSWER:
{answer}

Return ONLY valid JSON:
{{
  "claims": [
    {{"claim": "...", "supported": true}},
    {{"claim": "...", "supported": false}}
  ],
  "score": 0.75
}}
"""

CONTEXT_RECALL_SYSTEM = (
    "You are an impartial evaluation judge. You evaluate whether retrieved "
    "context contains the information needed to answer a question. "
    "You always output valid JSON."
)

CONTEXT_RECALL_PROMPT = """\
Given the GROUND TRUTH answer and the retrieved CONTEXT, evaluate context recall.

Step 1: Extract every distinct key fact from the GROUND TRUTH (list them).
Step 2: For each fact, determine if it is PRESENT or ABSENT in the CONTEXT.
Step 3: Compute the recall score = (number of facts present) / (total facts).

GROUND TRUTH:
{ground_truth}

CONTEXT:
{context}

Return ONLY valid JSON:
{{
  "facts": [
    {{"fact": "...", "present": true}},
    {{"fact": "...", "present": false}}
  ],
  "score": 0.80
}}
"""


def llm_faithfulness(answer: str, context: list[str], client: Groq) -> float:
    """LLM judges whether each claim in the answer is supported by context.

    Returns a score between 0.0 and 1.0.
    """
    if not context or not answer:
        return 0.0

    ctx_text = "\n\n---\n\n".join(context[:6])  # cap context length
    prompt = FAITHFULNESS_PROMPT.format(context=ctx_text, answer=answer)

    try:
        response = safe_groq_call(
            client, prompt,
            system_message=FAITHFULNESS_SYSTEM,
            max_tokens=1500,
        )
        result = parse_json_from_llm(response)
        if isinstance(result, dict) and "score" in result:
            score = float(result["score"])
            return max(0.0, min(1.0, score))

        # Fallback: count from claims list
        if isinstance(result, dict) and "claims" in result:
            claims = result["claims"]
            if claims:
                supported = sum(1 for c in claims if c.get("supported", False))
                return supported / len(claims)
    except Exception as e:
        print(f"  [Faithfulness] LLM judge error: {e}")

    return 0.0


def llm_context_recall(ground_truth: str, context: list[str], client: Groq) -> float:
    """LLM judges whether the context contains the ground truth information.

    Returns a score between 0.0 and 1.0.
    """
    if not context or not ground_truth:
        return 0.0

    ctx_text = "\n\n---\n\n".join(context[:6])
    prompt = CONTEXT_RECALL_PROMPT.format(
        ground_truth=ground_truth, context=ctx_text
    )

    try:
        response = safe_groq_call(
            client, prompt,
            system_message=CONTEXT_RECALL_SYSTEM,
            max_tokens=1500,
        )
        result = parse_json_from_llm(response)
        if isinstance(result, dict) and "score" in result:
            score = float(result["score"])
            return max(0.0, min(1.0, score))

        # Fallback: count from facts list
        if isinstance(result, dict) and "facts" in result:
            facts = result["facts"]
            if facts:
                present = sum(1 for f in facts if f.get("present", False))
                return present / len(facts)
    except Exception as e:
        print(f"  [ContextRecall] LLM judge error: {e}")

    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING-BASED METRICS (answer_relevancy, context_precision)
# ═══════════════════════════════════════════════════════════════════════════════

def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def embedding_answer_relevancy(query: str, answer: str,
                                embed_model: SentenceTransformer) -> float:
    """Cosine similarity between query and answer embeddings."""
    q_emb = embed_model.encode([query])[0]
    a_emb = embed_model.encode([answer])[0]
    return round(_cosine_sim(q_emb, a_emb), 4)


def embedding_context_precision(query: str, context: list[str],
                                 embed_model: SentenceTransformer) -> float:
    """Average cosine similarity of each context chunk to the query."""
    if not context:
        return 0.0
    q_emb = embed_model.encode([query])[0]
    chunk_embs = embed_model.encode(context)
    sims = [_cosine_sim(q_emb, ce) for ce in chunk_embs]
    return round(float(np.mean(sims)), 4)


# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(query: str, answer: str, context: list[str],
                    ground_truth: str, embed_model: SentenceTransformer,
                    groq_client: Groq) -> dict:
    """Compute all 4 evaluation metrics for a single question.

    Metrics:
      - answer_relevancy:  embedding cosine sim (query vs answer)
      - context_precision:  embedding cosine sim (query vs each chunk, averaged)
      - faithfulness:  LLM-as-judge (are answer claims grounded in context?)
      - context_recall:  LLM-as-judge (does context contain ground truth info?)
    """
    answer_relevancy = embedding_answer_relevancy(query, answer, embed_model)
    context_precision = embedding_context_precision(query, context, embed_model)
    faithfulness = llm_faithfulness(answer, context, groq_client)
    context_recall = llm_context_recall(ground_truth, context, groq_client)

    return {
        "answer_relevancy": round(answer_relevancy, 4),
        "faithfulness": round(faithfulness, 4),
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def load_questions(path: str) -> list[dict]:
    """Load benchmark questions from a CSV or JSON file.

    CSV format: columns must include 'id', 'category', 'query', 'ground_truth'.
    JSON format: list of dicts with same keys.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        questions = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append({
                    "id": row.get("id", f"q_{len(questions)+1}"),
                    "category": row.get("category", "general"),
                    "query": row["query"],
                    "ground_truth": row["ground_truth"],
                })
        return questions
    else:
        with open(path) as f:
            return json.load(f)


def run_baseline(input_path: str, questions: list[dict], embed_model,
                 groq_client: Groq) -> list[dict]:
    """Run baseline RAG on all questions."""
    print("\n" + "=" * 60)
    print("RUNNING BASELINE RAG")
    print("=" * 60)

    # Build vector store once (supports .txt, .pdf, or directories)
    text, doc_name = baseline_rag.load_input(input_path)
    chunks = baseline_rag.chunk_text(text)
    index, embeddings, model = baseline_rag.build_vector_store(chunks)
    print(f"  Loaded {doc_name}: {len(text)} chars, {len(chunks)} chunks")

    results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['query'][:60]}...")
        try:
            retrieved = baseline_rag.retrieve(q["query"], index, chunks, model)
            answer = baseline_rag.generate_answer(q["query"], retrieved)
            metrics = compute_metrics(
                q["query"], answer, retrieved, q["ground_truth"],
                embed_model, groq_client,
            )
            results.append({
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "answer": answer,
                "context": retrieved,
                "ground_truth": q["ground_truth"],
                "metrics": metrics,
            })
            print(f"  Metrics: {metrics}")
            time.sleep(2)  # Rate limit
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "answer": f"Error: {e}",
                "context": [],
                "ground_truth": q["ground_truth"],
                "metrics": {"answer_relevancy": 0, "faithfulness": 0,
                           "context_recall": 0, "context_precision": 0},
            })
            time.sleep(5)

    return results


def run_graphrag(questions: list[dict], embed_model,
                 groq_client: Groq) -> list[dict]:
    """Run Graph RAG on all questions (graph must already be built)."""
    print("\n" + "=" * 60)
    print("RUNNING GRAPH RAG")
    print("=" * 60)

    results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['query'][:60]}...")
        try:
            result = graph_rag.query(q["query"], embed_model)
            metrics = compute_metrics(
                q["query"], result["answer"], result["context"],
                q["ground_truth"], embed_model, groq_client,
            )
            results.append({
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "answer": result["answer"],
                "context": result["context"],
                "ground_truth": q["ground_truth"],
                "metrics": metrics,
                "retrieval_path": result["retrieval_path"],
            })
            print(f"  Metrics: {metrics}")
            time.sleep(2)  # Rate limit
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "answer": f"Error: {e}",
                "context": [],
                "ground_truth": q["ground_truth"],
                "metrics": {"answer_relevancy": 0, "faithfulness": 0,
                           "context_recall": 0, "context_precision": 0},
            })
            time.sleep(5)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate metrics by category."""
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["metrics"])

    aggregated = {}
    for cat, metric_list in categories.items():
        aggregated[cat] = {}
        for metric_name in ["answer_relevancy", "faithfulness", "context_recall", "context_precision"]:
            values = [m[metric_name] for m in metric_list]
            aggregated[cat][metric_name] = round(np.mean(values), 4)

    # Overall average
    all_metrics = [r["metrics"] for r in results]
    aggregated["overall"] = {}
    for metric_name in ["answer_relevancy", "faithfulness", "context_recall", "context_precision"]:
        values = [m[metric_name] for m in all_metrics]
        aggregated["overall"][metric_name] = round(np.mean(values), 4)

    return aggregated


def generate_comparison_chart(baseline_agg: dict, graphrag_agg: dict, output_path: str):
    """Generate bar charts comparing baseline RAG vs Graph RAG."""
    categories = ["factual", "multi-hop", "global", "overall"]
    metrics = ["answer_relevancy", "faithfulness", "context_recall", "context_precision"]
    metric_labels = ["Answer\nRelevancy", "Faithfulness", "Context\nRecall", "Context\nPrecision"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("RAG vs Graph RAG — Quantitative Comparison", fontsize=16, fontweight="bold")

    bar_width = 0.35
    colors_baseline = "#5dade2"    # blue
    colors_graphrag = "#58d68d"    # green

    for ax_idx, category in enumerate(categories):
        ax = axes[ax_idx]
        baseline_vals = [baseline_agg.get(category, {}).get(m, 0) for m in metrics]
        graphrag_vals = [graphrag_agg.get(category, {}).get(m, 0) for m in metrics]

        x = np.arange(len(metrics))
        bars1 = ax.bar(x - bar_width/2, baseline_vals, bar_width,
                       label="Baseline RAG", color=colors_baseline, edgecolor="white")
        bars2 = ax.bar(x + bar_width/2, graphrag_vals, bar_width,
                       label="Graph RAG", color=colors_graphrag, edgecolor="white")

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(category.replace("-", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to {output_path}")


def print_summary_table(baseline_agg: dict, graphrag_agg: dict):
    """Print a formatted comparison table."""
    categories = ["factual", "multi-hop", "global", "overall"]
    metrics = ["answer_relevancy", "faithfulness", "context_recall", "context_precision"]

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Faithfulness:      LLM-as-judge (claim grounding)")
    print(f"  Context Recall:    LLM-as-judge (ground truth coverage)")
    print(f"  Answer Relevancy:  Embedding cosine similarity")
    print(f"  Context Precision: Embedding cosine similarity")
    print("-" * 80)

    header = f"{'Category':<12} {'Metric':<20} {'Baseline RAG':>12} {'Graph RAG':>12} {'Delta':>8}"
    print(header)
    print("-" * 80)

    for cat in categories:
        for m in metrics:
            b_val = baseline_agg.get(cat, {}).get(m, 0)
            g_val = graphrag_agg.get(cat, {}).get(m, 0)
            delta = g_val - b_val
            delta_str = f"{delta:+.4f}"
            marker = " ✓" if delta > 0 else ""
            print(f"{cat:<12} {m:<20} {b_val:>12.4f} {g_val:>12.4f} {delta_str:>8}{marker}")
        print("-" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark: RAG vs Graph RAG")
    parser.add_argument("--input", "-i",
                        help="Path to a .pdf, .txt, or directory of .txt/.pdf files")
    parser.add_argument("--pdf", default=None,
                        help="(Legacy) Path to input PDF — prefer --input")
    parser.add_argument("--questions", required=True,
                        help="Path to questions file (.csv or .json) in data folder")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip graph building (assumes graph already exists)")
    args = parser.parse_args()

    # Resolve input path
    input_path = args.input or args.pdf
    if not input_path:
        print("Error: Provide --input or --pdf")
        return

    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        return

    # Load questions from CSV or JSON
    if not os.path.exists(args.questions):
        print(f"Error: Questions file not found: {args.questions}")
        return
    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions}")

    # Shared embedding model and Groq client
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    groq_client = Groq(api_key=config.GROQ_API_KEY)

    # Build graph if needed
    if not args.skip_build:
        print("\n[Step 0] Building 3-layer graph...")
        graph_rag.build(input_path)

    # Run both pipelines
    baseline_results = run_baseline(input_path, questions, embed_model, groq_client)
    graphrag_results = run_graphrag(questions, embed_model, groq_client)

    # Aggregate
    baseline_agg = aggregate_metrics(baseline_results)
    graphrag_agg = aggregate_metrics(graphrag_results)

    # Print summary
    print_summary_table(baseline_agg, graphrag_agg)

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    with open(os.path.join(config.RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump({"aggregated": baseline_agg, "detailed": baseline_results}, f, indent=2, default=str)

    with open(os.path.join(config.RESULTS_DIR, "graphrag_results.json"), "w") as f:
        json.dump({"aggregated": graphrag_agg, "detailed": graphrag_results}, f, indent=2, default=str)

    # Generate chart
    chart_path = os.path.join(config.RESULTS_DIR, "comparison_chart.png")
    generate_comparison_chart(baseline_agg, graphrag_agg, chart_path)

    print(f"\nAll results saved to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
