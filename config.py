"""
Shared configuration for the Hierarchical Discourse Graph RAG project.
Loads credentials from .env and defines constants used across all modules.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# -- LLM (Groq) ---------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "qwen/qwen3-32b"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096

# -- Groq Rate Limits (free tier: 30 req/min, 6000 req/day) -------------------
GROQ_MAX_RPM = 28              # sliding-window cap (keep 2 buffer below 30)
GROQ_RETRY_ATTEMPTS = 3        # max retries on 429 / transient errors
GROQ_INTER_CALL_DELAY = 2      # seconds to sleep between consecutive LLM calls

# -- Neo4j Aura ----------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://xxxxxxxx.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# -- Embeddings (LOCAL -- runs entirely on your machine, no API needed) --------
# BAAI/bge-base-en-v1.5: ~438 MB, 768-dim, strong general-purpose quality
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# -- Chunking ------------------------------------------------------------------
# Larger chunks = fewer LLM calls (saves Groq quota)
CHUNK_SIZE = 1500     # characters per chunk (larger = richer extraction)
CHUNK_OVERLAP = 150   # ~10% overlap

# -- Retrieval -----------------------------------------------------------------
TOP_K_CHUNKS = 5          # baseline RAG: top-k chunks
GRAPH_TOP_K_CHUNKS = 5    # graph RAG: cap retrieved chunks (prevents context overload)
TOP_K_COMMUNITIES = 3     # graph RAG: top-k communities to match
MAX_HOPS = 1              # graph RAG: BFS hops (1 = focused, 2 = too broad on small graphs)
MIN_COMMUNITY_SIMILARITY = 0.45  # lowered: allow broader community matching for global queries
TOP_K_CONCEPTS = 5               # max concepts kept after reranking
MIN_CONCEPT_RELEVANCE = 0.4      # lowered: keep more concepts to improve recall
MAX_ENTITIES = 30                # cap on entities after reranking

# -- Entity Deduplication ------------------------------------------------------
ENTITY_DEDUP_THRESHOLD = 0.95   # embedding cosine sim threshold for fuzzy merge (0.95 = conservative)

# -- Paths ---------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
