"""
Baseline RAG — Standard vector-similarity retrieval + LLM generation.
This serves as the comparison baseline for the novel Graph RAG approach.

Usage:
    python baseline_rag.py --input data/my_folder --query "What is ReAct?"
    python baseline_rag.py --pdf data/file.pdf --query "What is ReAct?"
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
from lib.groq_utils import strip_think


# ── Document Loading ─────────────────────────────────────────────────────────

def load_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
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


def load_input(input_path: str) -> tuple[str, str]:
    """Load text from a .pdf, .txt, or a directory of mixed files.

    Returns:
        (text, doc_name)
    """
    if os.path.isdir(input_path):
        texts = []
        files_loaded = []
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


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks of `size` characters."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ── Embedding + FAISS Index ──────────────────────────────────────────────────

def build_vector_store(chunks: list[str]) -> tuple:
    """Embed chunks with MiniLM and build a FAISS index.
    
    Returns:
        (faiss_index, embeddings_array, embed_model)
    """
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity (use inner product after normalization)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(embeddings)

    return index, embeddings, model


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, index, chunks: list[str], model, top_k: int = None) -> list[str]:
    """Retrieve top-k most similar chunks for a query."""
    top_k = top_k or config.TOP_K_CHUNKS
    q_emb = model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ── LLM Generation ──────────────────────────────────────────────────────────

def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Send query + retrieved context to Groq and return the answer."""
    client = Groq(api_key=config.GROQ_API_KEY)

    context = "\n\n---\n\n".join(context_chunks)

    system_msg = (
        "You are a knowledgeable expert assistant. You answer questions by "
        "synthesizing information from provided context passages. You always "
        "ground your answers in the provided evidence and never make up facts."
    )

    prompt = (
        "Answer the question below using ONLY the provided context. "
        "Synthesize information from multiple passages when relevant. "
        "If the context does not contain enough information, say so clearly.\n\n"
        f"CONTEXT:\n{context}\n\n"
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


# ── Full Pipeline ────────────────────────────────────────────────────────────

def run(input_path: str, query: str) -> dict:
    """Run the full baseline RAG pipeline.

    Args:
        input_path: Path to a .pdf, .txt, or a directory of .txt/.pdf files.
        query: The question to answer.

    Returns:
        dict with keys: query, answer, context, num_chunks, time_s
    """
    start = time.time()

    print(f"[1/4] Loading: {input_path}")
    text, doc_name = load_input(input_path)
    print(f"       Extracted {len(text)} characters from {doc_name}")

    print(f"[2/4] Chunking ({config.CHUNK_SIZE} chars, {config.CHUNK_OVERLAP} overlap)")
    chunks = chunk_text(text)
    print(f"       Created {len(chunks)} chunks")

    print("[3/4] Building vector store (FAISS + bge-base-en-v1.5)")
    index, embeddings, model = build_vector_store(chunks)

    print(f"[4/4] Retrieving top-{config.TOP_K_CHUNKS} chunks and generating answer")
    retrieved = retrieve(query, index, chunks, model)
    answer = generate_answer(query, retrieved)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"\nAnswer:\n{answer}")

    return {
        "query": query,
        "answer": answer,
        "context": retrieved,
        "num_chunks": len(chunks),
        "time_s": round(elapsed, 2),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline RAG Pipeline")
    parser.add_argument("--input", "-i",
                        help="Path to a .pdf, .txt, or a directory of .txt/.pdf files")
    parser.add_argument("--pdf", default=None,
                        help="(Legacy) Path to input PDF — prefer --input")
    parser.add_argument("--query", required=True, help="Query string")
    args = parser.parse_args()

    input_path = args.input or args.pdf
    if not input_path:
        print("Error: Provide --input or --pdf")
        return

    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        return

    result = run(input_path, args.query)

    # Save result
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "baseline_last_run.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
