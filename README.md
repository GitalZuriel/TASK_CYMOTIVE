# GenAI Automotive Security Copilot

AI-powered incident analysis copilot for automotive cybersecurity reports — built with a hybrid RAG pipeline, cross-encoder reranking, and multi-model LLM support.

> **PoC** developed at CYMOTIVE Technologies

## Demo

![Dashboard Screenshot](web/static/img/demo-screenshot.png)

<!-- Replace the path above with an actual screenshot or GIF of the dashboard -->

## Architecture Overview

```
Incident Report (free text)
        │
        ▼
┌───────────────┐
│  Preprocessor │──→ Validate · Extract entities · Detect edge cases
└───────┬───────┘
        │
        ▼
┌────────────────────────────────┐
│       Hybrid Retrieval         │
│  ┌───────────┐  ┌───────────┐ │
│  │ ChromaDB  │  │   BM25    │ │   ← Parallel (ThreadPoolExecutor)
│  │ Semantic  │  │  Keyword  │ │
│  └─────┬─────┘  └─────┬─────┘ │
│        └──────┬────────┘       │
│               ▼                │
│      RRF Fusion (k=60)        │
└───────────┬────────────────────┘
            │
            ▼
┌────────────────────┐
│  Cross-Encoder     │──→ Rerank + Domain Scoring → Top-2 results
│  Reranker          │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Context Guardrails│──→ Relevance threshold · Dedup · Token budget
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│     LLM Chain      │──→ Structured Summary + Mitigation Plan
│ (GPT-4o / Claude)  │
└────────────────────┘
```

## Features

| Feature | Details |
|---------|---------|
| Hybrid RAG | Semantic search (ChromaDB) + BM25 keyword search, fused via Reciprocal Rank Fusion |
| Cross-Encoder Reranking | `ms-marco-MiniLM-L-6-v2` with domain-aware scoring (attack type, ECU, MITRE tactics) |
| Entity Extraction | CVE IDs, IP addresses, ECU names, CAN arb IDs, timestamps, attack indicators |
| Incident Grounding | Every claim tied to evidence; inferences explicitly marked |
| Mitigation Generation | Phased response plan (immediate / short-term / long-term) with ISO 21434 & UNECE WP.29 references |
| Multi-Model Support | GPT-4o, GPT-4o-mini, Claude Sonnet 4 — switchable at runtime |
| Structured Output | Pydantic v2 schemas with per-section confidence scores (0.0–1.0) |
| Confidence & Clarification | Auto-triggers follow-up questions when confidence < 0.4 |
| Bilingual UI | English + Hebrew with RTL support |
| Cost & Token Tracking | Real-time per-request cost estimates and token counts |

## Pipeline

```
1. Preprocessing      → Validate input, extract CVEs / IPs / ECUs / timestamps / attack indicators
2. Entity Extraction  → Structured entities feed into domain scoring and context building
3. Hybrid Retrieval   → ChromaDB cosine similarity ‖ BM25 custom tokenizer → RRF fusion (k=60)
4. Reranking          → Cross-encoder scoring + domain scoring (α=0.8 · CE + β=0.2 · domain)
5. LLM Reasoning      → Two-stage: incident summary → mitigation plan (with few-shot prompts)
6. Structured Output  → JSON with confidence scores, or markdown fallback
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangChain 0.3+ |
| LLMs | GPT-4o · GPT-4o-mini · Claude Sonnet 4 |
| Vector DB | ChromaDB |
| Keyword Search | BM25 (`rank-bm25`) |
| Reranker | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-d) · `all-MiniLM-L6-v2` (384-d) |
| Structured Output | Pydantic v2 |
| Web UI | Flask + Bootstrap 4 (dark theme) |
| CLI | Rich |

## Example Input

```
A fleet of connected SUVs exhibited anomalous DNS query patterns originating from the
head-unit infotainment ECU (Harman Gen4). Investigation revealed that a compromised
third-party navigation app was exfiltrating vehicle telemetry data through encoded DNS
TXT records to a C2 server at 185.220.101.42. The DNS tunnel was transmitting GPS
coordinates, VIN numbers, and CAN bus diagnostic snapshots at 30-second intervals.
Over 2,400 vehicles were affected before the rogue app was identified. CVE-2024-2851.
```

## Example Output

**Incident Summary**

| Field | Value | Confidence |
|-------|-------|------------|
| Severity | HIGH | 0.95 |
| Attack Vector | Network Communication — DNS Tunneling | 0.90 |
| Affected Systems | Infotainment ECU (Harman Gen4) | 0.95 |
| Key Indicators | C2: `185.220.101.42`, Protocol: DNS TXT, CVE-2024-2851 | 0.90 |
| Scope | 2,400 vehicles | 0.85 |

**Mitigation Plan**

| Phase | Actions |
|-------|---------|
| Immediate (0–24 h) | Block outbound DNS TXT queries from vehicle ECUs; revoke compromised nav app certificate |
| Short-Term (1–7 d) | Deploy DNS anomaly detection rules in VSOC; implement application whitelisting on infotainment ECU |
| Long-Term | Enforce OEM app store vetting per ISO/SAE 21434; add egress traffic monitoring to UNECE WP.29 audit scope |

**Similar Past Incidents**

| Rank | Incident | Score |
|------|----------|-------|
| 1 | INC-001 — DNS Tunneling via Infotainment ECU | 92.3 |
| 2 | INC-009 — Telematics Gateway Data Breach | 61.7 |

## Evaluation

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of top-K results that are relevant |
| Recall@K | Fraction of relevant documents found in top-K |
| MRR | Mean Reciprocal Rank — 1/rank of first relevant result |
| nDCG@K | Normalized Discounted Cumulative Gain |
| Paraphrase Stability | Top-1 agreement & Jaccard similarity across query rewrites |

Run the evaluation suite:

```bash
python evaluation/compare_models.py       # LLM comparison (latency, tokens, cost)
python evaluation/compare_embeddings.py    # Embedding model recall & precision
python evaluation/dev_set.py               # Development set ground-truth evaluation
python evaluation/holdout_set.py           # Holdout test set evaluation
```

## Cost & Latency

| Model | Input ($/1K tokens) | Output ($/1K tokens) | Typical Request Cost* |
|-------|---------------------|----------------------|-----------------------|
| GPT-4o-mini | $0.00015 | $0.0006 | ~$0.001 |
| GPT-4o | $0.0025 | $0.01 | ~$0.02 |
| Claude Sonnet 4 | $0.003 | $0.015 | ~$0.03 |

*\*Estimated for a ~1,500 input + ~500 output token request (summary + mitigation).*

## Project Structure

```
cyber-copilot/
├── app.py                         # Flask web server + REST API
├── main.py                        # CLI interface (Rich terminal)
├── config.py                      # Configuration & environment variables
├── requirements.txt
│
├── src/
│   ├── copilot.py                 # Main orchestrator
│   ├── preprocessor.py            # Input validation & entity extraction
│   ├── embeddings.py              # Embedding models (OpenAI + MiniLM)
│   ├── vector_store.py            # ChromaDB vector store
│   ├── bm25_search.py             # BM25 with automotive-specific tokenizer
│   ├── hybrid_retrieval.py        # Parallel retrieval + RRF fusion
│   ├── reranker.py                # Cross-Encoder + domain scoring
│   ├── llm_chain.py               # Multi-model LLM processing
│   ├── prompts.py                 # System & task prompts (few-shot)
│   └── schemas.py                 # Pydantic v2 output schemas
│
├── data/
│   └── incidents.json             # 15 automotive cybersecurity incidents
│
├── web/
│   ├── templates/dashboard.html   # Bootstrap 4 dark-themed dashboard
│   └── static/
│       ├── css/copilot.css
│       └── js/copilot.js          # Frontend logic + i18n (EN/HE)
│
└── evaluation/
    ├── compare_models.py          # LLM model comparison
    ├── compare_embeddings.py      # Embedding model evaluation
    ├── dev_set.py                 # Dev set ground-truth tests
    ├── holdout_set.py             # Holdout set evaluation
    └── metrics.py                 # Shared retrieval metrics (P@K, MRR, nDCG)
```

## Running Locally

```bash
# 1. Clone & enter directory
cd cyber-copilot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux / Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...

# 5a. Run Web UI
python app.py
# → http://localhost:5000

# 5b. Run CLI
python main.py
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Full pipeline analysis |
| `POST` | `/api/precheck` | Lightweight input validation |
| `GET` | `/api/incidents` | List all incidents in DB |
| `POST` | `/api/search` | Retrieval-only (no LLM) |

## Incident Database

15 realistic automotive cybersecurity incidents covering:

| Category | Examples |
|----------|----------|
| Network | DNS tunneling, telematics gateway breach, cellular modem exploits |
| CAN Bus | CAN injection, OBD-II exploitation, ECU firmware rollback |
| Supply Chain | OTA update compromise, firmware tampering |
| Wireless | Bluetooth pairing attacks, key fob relay, RKE attacks |
| Sensors | ADAS sensor manipulation, GPS spoofing, V2X spoofing |
| Charging | EV charging station MITM |

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_LLM` | `gpt-4o-mini` | LLM model for analysis |
| `DEFAULT_EMBEDDING` | `openai` | Embedding provider (`openai` / `local`) |
| `TOP_K_RETRIEVAL` | 7 | Candidates per retriever |
| `TOP_K_RERANK` | 2 | Final results after reranking |
| `RRF_K` | 60 | RRF fusion constant |
| `RERANK_SCORE_THRESHOLD` | 30.0 | Min score to include in LLM context |
| `MAX_CONTEXT_CHARS` | 4000 | Max context length for LLM |
| `CONFIDENCE_THRESHOLD` | 0.4 | Below this, auto-trigger clarification |