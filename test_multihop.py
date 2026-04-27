"""Quick test: run a single multi-hop query and inspect the retrieval."""
import graph_rag

query = "How did the NovaCare partnership evolve from the founding years through the COVID-19 pandemic?"
result = graph_rag.query(query)

print("\n" + "=" * 80)
print("FINAL ANSWER:")
print(result["answer"])

print(f"\nRelationships used: {len(result.get('relationships', []))}")
print(f"Chunks used: {len(result.get('context', []))}")

print("\nTop relationships (ordered chain):")
for i, r in enumerate(result.get("relationships", [])[:15]):
    ev = r.get("evidence", "")[:80]
    print(f"  {i+1}. {r['source']} --[{r['type']}]--> {r['target']}: {ev}")

print("\nContext chunk previews:")
for i, c in enumerate(result.get("context", [])):
    print(f"  Chunk {i}: {c[:120]}...")
