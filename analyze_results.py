import json

with open("results/graphrag_results.json") as f:
    gr = json.load(f)
with open("results/baseline_results.json") as f:
    bl = json.load(f)

print("=" * 100)
print("MULTI-HOP QUESTION-BY-QUESTION ANALYSIS")
print("=" * 100)

gr_detail = {r["id"]: r for r in gr["detailed"]}
bl_detail = {r["id"]: r for r in bl["detailed"]}

for qid in ["q6", "q7", "q8", "q9", "q10"]:
    g = gr_detail[qid]
    b = bl_detail[qid]
    print()
    print("-" * 100)
    print("[%s] %s" % (qid, g["query"]))
    print("-" * 100)

    print("\n  METRICS COMPARISON:")
    for m in ["faithfulness", "context_recall", "answer_relevancy", "avg_relevance_score"]:
        gv = g["metrics"][m]
        bv = b["metrics"][m]
        delta = gv - bv
        winner = "GRAPH" if delta > 0 else "BASE" if delta < 0 else "TIE"
        print("    %-25s  Baseline=%.3f  GraphRAG=%.3f  delta=%+.3f  [%s]" % (m, bv, gv, delta, winner))

    if "Error" in str(g.get("answer", "")):
        print("\n  >>> API ERROR: %s" % g["answer"][:150])

    if g["metrics"]["context_recall"] == 0.0:
        print("\n  >>> CONTEXT RECALL = 0 -- Ground truth facts NOT in retrieved chunks")
        print("  Ground truth: %s" % g["ground_truth"][:250])
        print("  Graph answer:  %s" % str(g["answer"])[:250])

    if g["metrics"]["faithfulness"] == 0.0:
        print("\n  >>> FAITHFULNESS = 0 -- Answer has unsupported claims")
        print("  Graph answer: %s" % str(g["answer"])[:300])

    if "retrieval_path" in g:
        rp = g["retrieval_path"]
        print("\n  RETRIEVAL PATH:")
        for c in rp.get("communities", []):
            print("    Community: %s (sim=%.3f)" % (c["name"], c["similarity"]))
        for c in rp.get("concepts", []):
            print("    Concept: %s (rel=%.3f)" % (c["name"], c["relevance"]))
        print("    Entities traversed: %s" % rp.get("entities_traversed", "N/A"))
        print("    Chunks retrieved: %s" % rp.get("chunks_retrieved", "N/A"))

print("\n\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

mh_ids = ["q6", "q7", "q8", "q9", "q10"]
errors = sum(1 for qid in mh_ids if "Error" in str(gr_detail[qid].get("answer", "")))
recall_zero = sum(1 for qid in mh_ids if gr_detail[qid]["metrics"]["context_recall"] == 0.0)
faith_zero = sum(1 for qid in mh_ids if gr_detail[qid]["metrics"]["faithfulness"] == 0.0)

print("  Graph RAG multi-hop: API Errors=%d/5, Recall=0: %d/5, Faith=0: %d/5" % (errors, recall_zero, faith_zero))

bl_errors = sum(1 for qid in mh_ids if "Error" in str(bl_detail[qid].get("answer", "")))
bl_recall_zero = sum(1 for qid in mh_ids if bl_detail[qid]["metrics"]["context_recall"] == 0.0)
bl_faith_zero = sum(1 for qid in mh_ids if bl_detail[qid]["metrics"]["faithfulness"] == 0.0)
print("  Baseline multi-hop:  API Errors=%d/5, Recall=0: %d/5, Faith=0: %d/5" % (bl_errors, bl_recall_zero, bl_faith_zero))
