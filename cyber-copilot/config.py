"""
Configuration module for Cybersecurity Copilot.

Loads environment variables from .env and provides project-wide defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Model Defaults ---
DEFAULT_LLM = "gpt-4o-mini"
DEFAULT_EMBEDDING = "openai"  # "openai" or "local"

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "automotive_incidents"

# --- Retrieval ---
TOP_K_RETRIEVAL = 7  # candidates from each retrieval source
TOP_K_RERANK = 2     # final results after cross-encoder reranking
RRF_K = 60           # RRF fusion constant

# --- RAG Context Guardrails ---
RERANK_SCORE_THRESHOLD = 30.0  # min normalized score (10-95 scale) to include in LLM context
MAX_CONTEXT_CHARS = 4000       # max characters for similar-incidents context

# --- Paths ---
INCIDENTS_PATH = os.path.join(os.path.dirname(__file__), "data", "incidents.json")

# --- Structured Output ---
STRUCTURED_OUTPUT = True       # Enable structured JSON output (False = legacy markdown)
CONFIDENCE_THRESHOLD = 0.4     # Below this, auto-trigger clarification

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Cost Estimates (USD per 1K tokens, approximate) ---
MODEL_COSTS = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
}
