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
LLM_MAX_TOKENS = 2048          # allow richer, more detailed answers

# -- Groq Rate Limits (free tier: 30 req/min, 6000 req/day) -------------------
GROQ_MAX_RPM = 28              # sliding-window cap (keep 2 buffer below 30)
GROQ_RETRY_ATTEMPTS = 5        # max retries on 429 / transient errors
GROQ_INTER_CALL_DELAY = 4      # seconds to sleep between consecutive LLM calls

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
GRAPH_TOP_K_CHUNKS = 10   # graph RAG: cross-doc coverage via relationship-aware retrieval
MAX_CHUNK_CHARS = 1200    # preserve more context per chunk for better faithfulness
TOP_K_COMMUNITIES = 3     # graph RAG: top-k communities to match
MAX_HOPS = 2              # graph RAG: BFS hops (2 = reach bridging entities for multi-hop)
MIN_COMMUNITY_SIMILARITY = 0.35  # allow broader community matching for global queries
TOP_K_CONCEPTS = 10              # max concepts kept after reranking (multi-hop spans more concepts)
MIN_CONCEPT_RELEVANCE = 0.3      # keep more concepts to improve recall
MAX_ENTITIES = 25                # tighter cap avoids context dilution
MAX_RELATIONSHIPS_FOR_LLM = 20   # more relationships sent to LLM for multi-hop chains

# -- Entity Deduplication ------------------------------------------------------
ENTITY_DEDUP_THRESHOLD = 0.95   # embedding cosine sim threshold for fuzzy merge (0.95 = conservative)

# -- Paths ---------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
