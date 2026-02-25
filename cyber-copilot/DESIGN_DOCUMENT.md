# Cybersecurity Copilot – Full Solution Document
### GenAI Assignment – CYMOTIVE Technologies

---

## 1. Solution Design & Architecture

### 1.1 Overview

The solution is a **Cybersecurity Copilot** – an internal AI assistant for automotive cybersecurity analysts. The system receives a free-text security incident report and performs three tasks:

1. **Incident Summary** – a structured synopsis of the report including severity, attack vector, affected systems, and indicators
2. **Response Plan** – phased mitigation recommendations (immediate / short-term / long-term)
3. **Similar Incident Retrieval** – hybrid search (semantic + keyword) over a knowledge base of past incidents

### 1.2 End-to-End System Flow

```
Input: Free-text incident report
         │
         ▼
┌─────────────────────────┐
│   Preprocessing         │  ← preprocessor.py
│   • Cleaning & validation│
│   • Entity extraction    │
│     (CVE, IP, ECU,       │
│     attack indicators)   │
│   • Auto field extraction│
└────────────┬────────────┘
             ▼
┌─────────────────────────────────────────┐
│   Hybrid Retrieval (parallel)           │  ← hybrid_retrieval.py
│                                         │
│   ┌──────────────┐  ┌────────────────┐  │
│   │ Vector Search │  │ BM25 Keyword   │  │
│   │ (ChromaDB)   │  │ Search         │  │
│   │  top-7       │  │  top-7         │  │
│   └──────┬───────┘  └──────┬─────────┘  │
│          └──────┬───────────┘           │
│                 ▼                       │
│        RRF Fusion (k=60)               │
│        Deduplication                   │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│   Reranking                             │  ← reranker.py
│   • Cross-Encoder (ms-marco-MiniLM)     │
│   • Domain scoring (attack type, ECU,   │
│     MITRE tactics, severity)            │
│   • Blending: 80% CE + 20% Domain      │
│   → Top-2 final results                │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│   Context Guardrails                    │
│   • Relevance threshold filtering (30.0)│
│   • Metadata-based deduplication        │
│   • Token budget limit (4000 chars)     │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│   LLM – Summarization (Step 5a)        │  ← llm_chain.py
│   System Prompt + Summary Prompt        │
│   + retrieved similar incidents context │
│   → Structured incident summary        │
│   → + Entity Validation: verify, fix,  │
│     and complete regex-extracted entities│
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│   LLM – Mitigation (Step 5b)           │
│   System Prompt + Mitigation Prompt     │
│   + summary + similar incidents         │
│   → Phased response plan               │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│   Confidence Check                      │
│   If average confidence < 0.4:          │
│   → Generate clarification questions    │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│   Final Output Assembly                 │
│   • Summary + Mitigation               │
│   • Similar incidents (ranked)          │
│   • Detected entities                   │
│   • Warnings + clarifications           │
│   • Stats (cost, tokens, latency)       │
└─────────────────────────────────────────┘
```

### 1.3 Tools, APIs & Frameworks

| Component | Technology | Role |
|-----------|------------|------|
| **LLM Orchestration** | LangChain ≥0.3.0 | Manages prompt chains and model invocations |
| **Primary LLM** | GPT-4o-mini (default) | Summarization & mitigation – excellent cost-performance balance |
| **Additional Models** | GPT-4o, Claude Sonnet 4 | Upgradable on demand |
| **Embeddings** | OpenAI text-embedding-3-small (1536-d) | Semantic representation of incidents |
| **Local Embeddings** | all-MiniLM-L6-v2 (384-d) | Zero API-cost alternative |
| **Vector Database** | ChromaDB | Embedding storage and retrieval |
| **Keyword Search** | rank-bm25 (BM25Okapi) | Complementary keyword-based search |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint query-document scoring |
| **Structured Output** | Pydantic v2 | JSON output validation and schemas |
| **API Server** | Flask | REST API endpoints |
| **CLI Interface** | Rich | Interactive command-line interface |
| **Environment** | python-dotenv | API key loading from .env |

### 1.4 Design Decisions & Tradeoffs

#### Model Choice: GPT-4o-mini as Default

| Criterion | GPT-4o-mini | GPT-4o | Claude Sonnet |
|-----------|-------------|--------|---------------|
| **Cost (input/output per 1K tokens)** | $0.00015 / $0.0006 | $0.0025 / $0.01 | $0.003 / $0.015 |
| **Quality** | Very good for summarization | Excellent | Excellent |
| **Speed** | Very fast | Medium | Medium |
| **Cost-effectiveness** | Best | ~17x more expensive | ~25x more expensive |

**Rationale**: For a PoC, GPT-4o-mini delivers excellent quality at negligible cost. The system supports hot-swapping to more powerful models (GPT-4o or Claude Sonnet) without code changes – just a parameter switch.

#### Hybrid Search Instead of Vector-Only

**Why not just Vector Search?**
- Semantic search alone misses cases with exact technical terms (e.g., CVE-2024-2851 or a specific CAN arbitration ID)
- BM25 excels at exact term matching
- Combining both via RRF (Reciprocal Rank Fusion) gives the best of both worlds

**Why RRF instead of linear score combination?**
- RRF works with ranks, not scores, so no cross-method normalization is needed
- Simple to implement, efficient, and research-proven
- Formula: `score(d) = Σ 1/(k + rank(d))` where k=60

#### Cross-Encoder Reranking

**Why an additional reranking stage?**
- Bi-encoder (embedding search) is fast but less accurate – each document is encoded independently
- Cross-encoder processes query and document together (joint encoding), producing far more accurate scores
- Reranking operates on only ~7-14 candidates (after RRF), so overhead is minimal

**Domain-aware blending:**
- 80% Cross-Encoder (general relevance)
- 20% Domain score (attack type match, ECU match, MITRE tactics, severity)
- This ensures that even if the Cross-Encoder doesn't "understand" specific automotive terms, domain scoring compensates

#### Entity Validation – Regex + LLM at No Extra Cost

**The problem**: The Preprocessor (step 1) extracts entities using regex – fast and free, but misses typos and terms not in the dictionary.

**The solution**: Instead of adding a separate LLM call, we extended the summarization step (5a) that already exists. The LLM receives the regex-extracted entities and is asked to also verify, correct, and complete them – **in the same call, at the same cost**.

```
Step 1: Preprocessing (regex)          Step 5a: Summarization (LLM)
───────────────────────                ─────────────────────────────
✓ CVE-2024-2851 (exact match)    →    ✓ CVE-2024-2851 [confirmed]
✗ "infotanment" (typo, missed)   →    + Infotainment ECU [corrected]
✗ DNS protocol (missed)          →    + DNS [added]

Cost: $0                              Cost: $0 additional (already running!)
```

**Three source types per entity**:
- **confirmed** – regex extracted correctly, LLM confirms
- **corrected** – regex extracted partially or missed due to typo, LLM corrected (includes original text)
- **added** – regex missed entirely, LLM identified in the report

**Why not use only LLM?** Because regex provides 100% certainty on exact matches. `CVE-2024-2851` found by regex is ground truth – the LLM cannot get confused about it. The combination yields high precision (from regex) + high recall (from LLM).

#### Grounding & Hallucination Prevention

**Core principle**: Every claim in the output must be grounded in:
1. An explicit fact from the incident report
2. A pattern from a retrieved past incident
3. A direct countermeasure for the identified attack vector

Anything not grounded in one of these must be marked as `[General best practice]` or `[Inferred]`.

#### Scalability & Production Readiness

**Current PoC implementation:**

| Component | Current Implementation | Details |
|-----------|----------------------|---------|
| **Knowledge Base** | Local ChromaDB, 15 incidents | Local disk storage, persist directory |
| **Embeddings** | OpenAI text-embedding-3-small (API) | Additional local option: all-MiniLM-L6-v2 |
| **LLM** | Synchronous calls, GPT-4o-mini | Hot-swap support for GPT-4o / Claude Sonnet |
| **Server** | Flask dev server (port 5000) | REST API + web dashboard |
| **Monitoring** | Python `logging` module | Console logs at every pipeline stage |
| **Cost Tracking** | Built into `LLMChain` | Token count, latency, cost per run – returned in API response |

**Production upgrade options (not implemented – future development ideas):**

| Component | Proposed Upgrade | Explanation |
|-----------|-----------------|-------------|
| **Knowledge Base** | Managed Pinecone / Weaviate | Cloud vector DB, thousands of incidents, automatic backups |
| **Embeddings** | Batch API or local GPU model | Reduce cost and external API dependency |
| **LLM** | Async + Queue + Response Caching | Handle concurrent load, save on repeated calls |
| **Server** | Gunicorn + nginx, Docker | Stability, load balancing, containerization |
| **Monitoring** | Prometheus + Grafana | Real-time metrics: latency, error rates, token usage |
| **LLM Observability** | LangSmith / LangFuse | Prompt tracing, chain debugging, A/B testing |
| **Logging** | ELK Stack / CloudWatch | Log search, audit trail, retention |
| **Alerting** | PagerDuty / OpsGenie | Failure alerts, cost anomalies, drift detection |

---

## 2. Prompt Engineering

### 2.1 Prompting Strategy

The system uses a layered approach to prompts:

1. **System Prompt** – defines the model's identity, domain, and behavioral rules
2. **Task Prompts** – specific instructions per task (summarization / mitigation)
3. **Few-Shot Examples** – complete input/output examples of the desired format
4. **Dynamic Context** – retrieved similar incidents + extracted entities

### 2.2 System Prompt

```
You are a specialized automotive cybersecurity incident analyst AI assistant
for CYMOTIVE Technologies. Your role is to analyze vehicle security incidents
with precision and provide actionable intelligence.

CRITICAL GROUNDING RULES:
- Every claim must be traceable to: the incident report, a retrieved past
  incident, or a direct countermeasure for the identified attack vector.
- Mark any inference not directly from the report as [Inferred] or
  [General best practice].
- If information is missing, explicitly state what is unknown rather than
  fabricating details.
- When referencing past incidents, cite them by ID (e.g., INC-001).
- Automotive-specific context: vehicles are safety-critical systems.
  Always consider passenger safety implications.
```

**Breakdown**:
- **Defined role** – the model knows it's an automotive cyber analyst, not a general assistant
- **Grounding rules** – prevent hallucinations. The model must base every claim on evidence
- **Explicit marking** – `[Inferred]` and `[General best practice]` let the analyst know what's grounded and what's not
- **Safety reminder** – emphasizes that these are safety-critical systems

### 2.3 Summarization Prompt

```
Analyze the following automotive cybersecurity incident report and provide
a structured summary.

## EXTRACTED INFORMATION (auto-detected):
{extracted_fields}

## SIMILAR PAST INCIDENTS:
{similar_incidents}

## INCIDENT REPORT:
{incident_report}

---

Provide a structured summary with exactly these sections:

### 1. Incident Overview
2–3 sentences summarizing what happened. Use ONLY facts from the report.

### 2. Severity Assessment
Rate as Critical / High / Medium / Low with justification.
Consider: safety impact, scope, exploitability, data sensitivity.

### 3. Attack Vector
Classify: Network / Physical / Wireless / Supply Chain / Insider.
Describe the attack method based on report evidence.

### 4. Affected Systems
List all ECUs, modules, and subsystems impacted.

### 5. Key Indicators (IOCs)
List ONLY explicitly mentioned: IPs, CVEs, CAN IDs, domains, hashes.
Do NOT invent indicators.

### 6. Timeline
Reconstruct the attack sequence. Mark uncertain steps as [Estimated].
```

**Techniques used**:

| Technique | Implementation | Why |
|-----------|---------------|-----|
| **Entity Validation** | LLM verifies/corrects/completes entities from regex | Precision (regex) + recall (LLM) at no extra cost |
| **Structured Output** | 6 defined sections with headers + validated entities | Ensures consistency and completeness |
| **Grounding Instructions** | "Use ONLY facts from the report" | Prevents hallucinations |
| **Dynamic Context Injection** | `{extracted_fields}`, `{similar_incidents}` | RAG – model receives relevant information |
| **Explicit Constraints** | "Do NOT invent indicators" | Negative prompting to prevent unwanted behavior |
| **Confidence Markers** | "Mark uncertain steps as [Estimated]" | Transparency about certainty level |
| **Few-Shot Example** | Complete input/output example (see code) | Shows the model exactly what format is expected |

### 2.4 Mitigation Prompt

```
Based on the incident summary and similar past incidents, provide a
mitigation and response plan.

## INCIDENT SUMMARY:
{summary}

## SIMILAR PAST INCIDENTS:
{similar_incidents}

---

Provide a phased response plan:

### 1. Immediate Actions (0-24 hours)
Containment and isolation steps. Each action must include:
- What to do
- Why (grounding: which evidence supports this)

### 2. Short-term Actions (1-7 days)
Investigation, patching, and hardening. Include:
- Root cause investigation steps
- Specific patches or configuration changes
- Monitoring enhancements

### 3. Long-term Recommendations
Architecture and process improvements. ONLY include if supported
by patterns from retrieved past incidents.

### 4. Related Standards & Frameworks
Reference ISO 21434, UNECE WP.29, AUTOSAR, etc.
Mark general practices as [General best practice].
```

**Techniques**:

| Technique | Implementation | Why |
|-----------|---------------|-----|
| **Chain-of-Context** | Summary is injected into the mitigation prompt | Mitigation is based on the summary, not raw report |
| **Phased Structure** | Immediate / short-term / long-term | Matches real-world Incident Response workflows |
| **Action + Grounding** | "What to do" + "Why" | Every recommendation must be justified |
| **Conditional Generation** | "ONLY include if supported by..." | Prevents irrelevant generic recommendations |
| **Standard References** | ISO 21434, UNECE WP.29 | Links to recognized industry standards |

### 2.5 Edge Case Handling

#### The Problem
Real-world incident reports are often:
- Incomplete (missing critical fields)
- Noisy (unclear text, spelling errors)
- Ambiguous (unclear what happened)

#### The Solution: Three-Stage Pipeline

**Stage 1 – Preprocessing (Preprocessor)**
- Minimum length validation: 20 characters
- Garbled text detection: >30% non-ASCII characters
- Automatic entity extraction (CVE, IP, ECU names)
- Missing field detection

**Stage 2 – Edge Case Prompt (for incomplete reports)**
```
The following incident report is incomplete or unclear.
Your task is NOT to analyze it, but to help the analyst
provide better information.

Report: {incomplete_report}
Detected entities: {entities}
Missing fields: {missing_fields}

Generate 3-5 specific clarification questions:
- Each question should target a missing critical field
- Explain WHY this information is important for analysis
- Suggest example answers to guide the analyst
```

**Stage 3 – Progressive Disclosure**
- If average confidence < 0.4, the system switches to `needs_input` mode
- Returns a partial summary + clarification questions
- The analyst can provide missing information and receive a full analysis

#### Example: Noisy Report and System Response

**Input (partial and noisy report):**
```
car hacked yesterday. something with the CAN bus.
brakes felt weird. maybe CVE-2024-something??
```

**System response:**
```json
{
  "status": "clarification_needed",
  "analysis_mode": "needs_input",
  "entities_detected": {
    "ecu_names": [],
    "attack_indicators": ["hacked"],
    "cve_ids": []
  },
  "warnings": [
    "No specific CVE ID detected (partial reference found)",
    "No ECU explicitly identified",
    "Missing: timestamp, severity, affected vehicle model"
  ],
  "clarification_questions": [
    "1. Which specific vehicle make/model/year was affected?
        (e.g., VW Golf 2024) - needed to identify relevant ECU architecture",
    "2. Can you provide the exact CVE ID? You mentioned 'CVE-2024-something'
        - the full ID helps cross-reference known vulnerabilities",
    "3. What exactly happened with the brakes? Did they fail to respond,
        activate unexpectedly, or show warning lights?
        - this determines severity classification",
    "4. How was the CAN bus compromise detected? Through diagnostic tools,
        physical symptoms, or monitoring alerts?
        - helps identify the attack vector",
    "5. Is the vehicle still in the compromised state or has it been isolated?
        - critical for immediate response prioritization"
  ]
}
```

### 2.6 Entity Validation – LLM-Based Entity Verification

During the summarization step (5a), the LLM doesn't just summarize – it also **verifies, corrects, and completes** the entities extracted by the Preprocessor. This happens **within the same LLM call, at no additional cost**.

Prompt instruction:
```
## Entity Validation Task
Review the auto-extracted entities listed under "Preprocessor Extraction" above.
For each:
1. CONFIRM if correct (source: "confirmed")
2. CORRECT if there's a typo (source: "corrected", include original text)
3. ADD entities you find in the report that were missed (source: "added")

⚠️ ONLY add entities EXPLICITLY mentioned in the report.
   Do NOT invent CVEs, IPs, or identifiers.
```

Example output (part of the structured JSON):
```json
{
  "validated_entities": {
    "cve_ids": [
      {"value": "CVE-2024-2851", "source": "confirmed", "original": null}
    ],
    "ecu_names": [
      {"value": "Infotainment ECU (Harman Gen4)", "source": "corrected",
       "original": "infotanment"}
    ],
    "protocols": [
      {"value": "DNS", "source": "added", "original": null}
    ]
  }
}
```

**The advantage**: The analyst can see exactly what was auto-detected (regex) and what the LLM added, and can trust each type at a different level.

### 2.7 Structured JSON Output

In addition to the markdown versions, there are JSON variants of every prompt with:
- Pre-defined Pydantic v2 schemas (including `ValidatedEntities`)
- Per-section confidence scoring (0.0–1.0)
- Automatic structure validation
- Retry logic with up to 3 attempts on invalid JSON

---

## 3. Sample Output

### 3.1 Example 1 – DNS Tunneling Incident

**Input:**
```
On January 15, 2024, our VSOC detected anomalous DNS traffic patterns
originating from the infotainment ECU (Harman Gen4) in multiple vehicles
across the EU fleet. The ECU was observed making unusual DNS TXT queries
to the domain c2.darknet-auto.xyz at regular 30-second intervals.

Analysis revealed that the queries contained encoded payloads carrying
GPS coordinates, VIN numbers, and CAN bus diagnostic data. The attack
exploited CVE-2024-2851, a buffer overflow in the DNS resolver library
used by the Harman Gen4 platform.

Source IP of C2 communication: 185.220.101.42. Approximately 340 vehicles
were affected before detection. No safety-critical systems were directly
compromised, but the data exfiltration poses significant privacy and
fleet security risks.
```

**Output – Summary:**
```
### 1. Incident Overview
On January 15, 2024, a DNS tunneling attack was detected targeting the
Harman Gen4 infotainment ECU across an EU vehicle fleet. The attacker
exploited CVE-2024-2851 (buffer overflow in DNS resolver) to establish
covert C2 communication via DNS TXT queries, exfiltrating GPS coordinates,
VIN numbers, and CAN bus diagnostics from approximately 340 vehicles.

### 2. Severity Assessment
**HIGH** – While no safety-critical systems were directly compromised,
the scale (340 vehicles), data sensitivity (GPS + VIN + CAN diagnostics),
and the presence of an active C2 channel represent significant fleet-wide
risk. The attack could serve as a foothold for deeper network penetration.

### 3. Attack Vector
**Network Communication** – The attacker exploited a buffer overflow
(CVE-2024-2851) in the DNS resolver library to inject a persistent C2
implant. Communication was tunneled through DNS TXT queries to
c2.darknet-auto.xyz at 30-second intervals.

### 4. Affected Systems
- Infotainment ECU (Harman Gen4 platform)
- DNS resolver library (vulnerable component)
- Vehicle telemetry data (GPS, VIN, CAN diagnostics)

### 5. Key Indicators
- **CVE**: CVE-2024-2851
- **C2 Domain**: c2.darknet-auto.xyz
- **C2 IP**: 185.220.101.42
- **Protocol**: DNS TXT queries
- **Interval**: 30-second beacon
- **Scale**: ~340 vehicles affected

### 6. Timeline
1. [Estimated] Attacker identifies CVE-2024-2851 in Harman Gen4 DNS resolver
2. [Estimated] Exploit deployed to target vehicles (method unknown)
3. C2 implant begins DNS TXT beaconing to c2.darknet-auto.xyz every 30 seconds
4. Data exfiltration of GPS coordinates, VIN numbers, CAN bus diagnostics
5. January 15, 2024 – VSOC detects anomalous DNS traffic patterns
```

**Output – Mitigation:**
```
### 1. Immediate Actions (0-24 hours)
- **Block C2 communication**: Add c2.darknet-auto.xyz and 185.220.101.42
  to DNS/firewall blocklists across all fleet vehicles.
  *Grounding: Direct C2 indicators from incident report.*
- **Isolate affected ECUs**: Disable outbound DNS for Harman Gen4 units
  in the 340 identified vehicles pending investigation.
  *Grounding: 340 vehicles confirmed affected.*
- **Preserve forensic evidence**: Capture DNS logs and ECU memory dumps
  before any remediation.
  *Grounding: Standard IR procedure for confirmed C2 activity.*

### 2. Short-term Actions (1-7 days)
- **Patch CVE-2024-2851**: Deploy patched DNS resolver library to all
  Harman Gen4 units via OTA update.
  *Grounding: Identified vulnerability in report.*
- **Assess data exposure**: Determine exact scope of exfiltrated data
  (GPS/VIN/CAN) and affected vehicle owners for notification.
- **Hunt for lateral movement**: Analyze CAN bus logs from affected
  vehicles for evidence of pivot from infotainment to other ECUs.
  *Grounding: CAN diagnostics data was accessed, suggesting CAN bus visibility.*
- **Implement DNS monitoring**: Deploy DNS anomaly detection focusing on
  TXT query patterns and beacon intervals.

### 3. Long-term Recommendations
- **Network segmentation**: Isolate infotainment ECU from vehicle-internal
  networks (CAN bus access should be read-only at most).
  *Based on pattern from INC-001: DNS tunneling succeeded partly because
  infotainment had CAN bus access.*
- **DNS security**: Implement DNS-over-HTTPS with allowlisting for
  automotive ECU communications.

### 4. Related Standards
- **ISO 21434** – Cybersecurity risk management for road vehicles
  [General best practice]
- **UNECE WP.29 R155** – Mandatory CSMS for type approval
  [General best practice]
- **AUTOSAR SecOC** – Secure on-board communication for CAN bus hardening
  [General best practice]
```

**Model:** GPT-4o-mini | **Latency:** ~3.2s | **Tokens:** ~1,850 | **Cost:** ~$0.0015

### 3.2 Example 2 – CAN Bus Injection Incident

**Input:**
```
Security alert: Our testing facility detected unauthorized CAN frames
on the high-speed CAN bus (500kbps) of a prototype vehicle. The frames
were using arbitration ID 0x0012 (brake ECU) and 0x00A5 (steering control).

The injection appears to originate from a compromised OBD-II diagnostic
port. Physical inspection revealed an unauthorized device connected to
the OBD-II connector under the dashboard. The device was transmitting
spoofed brake and steering commands at high frequency, causing momentary
brake activation and slight steering drift during low-speed testing.

No injuries occurred. The vehicle was immediately isolated.
Severity: CRITICAL due to safety system involvement.
```

**Output – Summary:**
```
### 1. Incident Overview
An unauthorized CAN bus injection attack was detected at a testing
facility targeting a prototype vehicle's safety-critical systems. A rogue
device connected to the OBD-II port injected spoofed frames on arbitration
IDs 0x0012 (brake) and 0x00A5 (steering), causing momentary brake
activation and steering drift during low-speed testing.

### 2. Severity Assessment
**CRITICAL** – Direct manipulation of brake and steering systems
constitutes the highest severity level. Although no injuries occurred
(low-speed testing environment), the same attack in normal driving
conditions could result in loss of vehicle control and endanger lives.

### 3. Attack Vector
**Physical** – The attacker required physical access to the vehicle's
OBD-II diagnostic port to connect the injection device. This is a
direct hardware attack on the CAN bus.

### 4. Affected Systems
- High-speed CAN bus (500kbps)
- Brake ECU (arbitration ID 0x0012)
- Steering control (arbitration ID 0x00A5)
- OBD-II diagnostic interface

### 5. Key Indicators
- **CAN IDs**: 0x0012 (brake), 0x00A5 (steering)
- **Bus Speed**: 500kbps high-speed CAN
- **Physical Evidence**: Unauthorized device on OBD-II port
- **Symptoms**: Momentary brake activation, steering drift

### 6. Timeline
1. [Estimated] Attacker gains physical access to prototype vehicle
2. Unauthorized device connected to OBD-II port under dashboard
3. Device begins transmitting spoofed CAN frames on IDs 0x0012 and 0x00A5
4. Momentary brake activation and steering drift observed during testing
5. Security team detects anomaly and isolates vehicle
6. Physical inspection reveals unauthorized OBD-II device
```

**Model:** GPT-4o-mini | **Latency:** ~2.8s | **Tokens:** ~1,620 | **Cost:** ~$0.0012

---

## 4. Working Prototype

### 4.1 Interfaces

**REST API (Flask):**
```
POST /api/analyze     → Full analysis (summary + mitigation + similar incidents)
POST /api/precheck    → Quick validation without LLM
POST /api/search      → Similar incident search only (no LLM)
GET  /api/incidents   → List all incidents in the knowledge base
GET  /                → Web dashboard
```

**CLI (Rich):**
```
$ python main.py
[Cyber-Copilot Banner]
Enter incident report (blank line to submit):
> ...
→ Displays: Summary | Mitigation | Similar incidents table | Stats
```

### 4.2 Dashboard

Web interface built with Bootstrap 4 and a dark theme, featuring:
- Text field for incident report input
- LLM model selection
- Summary and mitigation display
- Similar incidents list with relevance scores
- Detected entities
- Cost and performance statistics

---

## 5. Retrieval Pipeline (RAG)

### 5.1 Knowledge Base

**15 automotive cybersecurity incidents** in `data/incidents.json`, covering:

| Category | Examples |
|----------|---------|
| **Attack Vectors** | Network, Physical, Wireless, Supply Chain |
| **ECUs** | Infotainment, TCU, BCM, ADAS, Gateway, GNSS |
| **Protocols** | CAN, DNS, Bluetooth, V2X, OTA, LIN, JTAG |
| **Severity** | Critical, High, Medium, Low |
| **MITRE Tactics** | TA0001–TA0011 (Initial Access through Exfiltration) |

Each incident includes: title, detailed description, severity, CVE, indicators, attack stages, and mitigation recommendations.

### 5.2 Embedding Pipeline

```
Incident → Text document (title + description + attack_vector + affected_system)
         → OpenAI text-embedding-3-small (1536 dimensions)
         → ChromaDB (cosine similarity)
```

**Metadata storage**: For each incident: title, date, severity, severity_score, attack_vector, affected_system, cve, mitre_tactics, protocols, components.

### 5.3 Hybrid Search – The Flow

```
Query: "CAN bus injection on brake system"
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌──────────┐     ┌──────────┐
│ Semantic │     │  BM25    │
│ (ChromaDB)│    │ (Keyword)│
│           │     │          │
│ top-7:   │     │ top-7:   │
│ INC-002  │     │ INC-002  │
│ INC-010  │     │ INC-014  │
│ INC-004  │     │ INC-010  │
│ INC-014  │     │ INC-006  │
│ ...      │     │ ...      │
└────┬─────┘     └────┬─────┘
     └────────┬───────┘
              ▼
     ┌────────────────┐
     │  RRF Fusion    │
     │  k=60          │
     │                │
     │  INC-002: both │ ← appeared in both methods → high score
     │  INC-010: both │
     │  INC-014: both │
     │  INC-004: sem  │
     │  INC-006: bm25 │
     └───────┬────────┘
             ▼
     ┌────────────────────────┐
     │  Cross-Encoder Rerank  │
     │                        │
     │  INC-002: 87.3 (brake  │ ← CAN injection + brake = perfect match
     │    + CAN = high domain)│
     │  INC-014: 72.1         │
     │  INC-010: 58.4         │
     └───────┬────────────────┘
             ▼
     Top-2 Results:
     1. INC-002 "CAN Bus Injection Attack on Brake System" (87.3)
     2. INC-014 "CAN Frame Masquerading via Wired Backdoor" (72.1)
```

### 5.4 Why Hybrid Outperforms Each Method Alone

| Query | Semantic Only | BM25 Only | Hybrid |
|-------|---------------|-----------|--------|
| "DNS tunneling attack" | Finds semantic similarity (great) | Matches "DNS" + "tunneling" (great) | Both → reinforced score |
| "CVE-2024-2851" | Understands general security context | Exact CVE match (excellent) | BM25 adds what semantic misses |
| "car security problem" | Understands intent (good) | Too generic (weak) | Semantic compensates for BM25 |
| "0x0012 brake CAN" | Understands context (medium) | Exact CAN ID match (excellent) | Both complement each other |

### 5.5 Custom BM25 Tokenizer

The tokenizer preserves technical terms as single units:
- `CVE-2024-1234` → single token (not split into "CVE", "2024", "1234")
- `192.168.1.1` → single token
- Filters words shorter than 2 characters (except abbreviations like `v2x`, `can`)

---

## 6. Evaluation & Cost Awareness

### 6.1 Evaluation Metrics

#### Retrieval Metrics

| Metric | Description | Usage |
|--------|-------------|-------|
| **Precision@K** | How many of the top-K results are relevant | Primary metric – aiming for 100% at K=2 |
| **Recall@K** | How many relevant results were found | Important for cases with multiple relevant incidents |
| **MRR** | Mean Reciprocal Rank – position of first relevant result | Measures "how fast" we reach the answer |
| **nDCG@K** | Normalized Discounted Cumulative Gain | Accounts for result ordering |

#### LLM Output Quality Metrics

| Metric | Measurement Method | Target |
|--------|-------------------|--------|
| **Grounding Accuracy** | % of claims traceable to a source | > 90% |
| **Completeness** | Coverage of all 6 summary sections | 100% |
| **Hallucination Rate** | % of incorrect or fabricated claims | < 5% |
| **Actionability** | % of mitigation recommendations that are implementable | > 85% |
| **Confidence Calibration** | Alignment between confidence scores and actual accuracy | Pearson r > 0.7 |

#### User Experience Metrics

| Metric | Measurement Method |
|--------|-------------------|
| **Analyst Adoption Rate** | % of analysts using the tool regularly |
| **Time-to-Insight** | Time from report receipt to decision (with vs. without the tool) |
| **Edit Distance** | How much the analyst modifies the output before use |
| **Feedback Score** | 1–5 rating from the analyst per response |

### 6.2 Cost Estimates

#### Cost Per Single Run (GPT-4o-mini)

| Stage | Tokens (est.) | Cost |
|-------|---------------|------|
| **Summarization** | ~1,200 input + ~500 output | ~$0.00048 |
| **Mitigation** | ~1,500 input + ~600 output | ~$0.00059 |
| **Embeddings** | ~200 tokens query | ~$0.000004 |
| **Total per run** | **~4,000 tokens** | **~$0.001 – $0.002** |

#### Monthly Cost Comparison (100 queries/day)

| Model | Cost per run | Daily (100) | Monthly (~3,000) |
|-------|-------------|-------------|-----------------|
| **GPT-4o-mini** | ~$0.0015 | ~$0.15 | ~$4.50 |
| **GPT-4o** | ~$0.025 | ~$2.50 | ~$75 |
| **Claude Sonnet** | ~$0.04 | ~$4.00 | ~$120 |

#### Additional Production Costs

| Component | Estimated Monthly Cost |
|-----------|----------------------|
| ChromaDB (managed) / Pinecone | $0 (self-hosted) – $70 (managed) |
| Cross-encoder (GPU) | $0 (CPU) – $50 (GPU instance) |
| Flask hosting | $5–20 (cloud) |
| **Total production** | **~$10–$200/month** (depending on volume) |

### 6.3 Monitoring – Current State & Upgrade Plan

#### What Exists in the PoC Today

The system already includes basic tracking **built into the code**:

| What's Measured | How | Where in Code |
|-----------------|-----|---------------|
| **Latency per LLM call** | `time.perf_counter()` before and after each call | `llm_chain.py` |
| **Token count** | Returned from OpenAI/Anthropic API response | `llm_chain.py` |
| **Estimated cost per run** | Calculated from token count × model price | `llm_chain.py` |
| **Pipeline total latency** | End-to-end timing of the full pipeline | `copilot.py` |
| **Pipeline logging** | Log at every stage (preprocess → retrieval → rerank → LLM) | `copilot.py` |
| **Confidence scores** | Per-section confidence in Pydantic output | `schemas.py` |

All this data is returned in the API response under `llm_stats`:
```json
{
  "llm_stats": {
    "summary": {"model": "gpt-4o-mini", "latency": 3.2, "tokens": 1850, "cost": 0.0015},
    "mitigation": {"model": "gpt-4o-mini", "latency": 2.1, "tokens": 1620, "cost": 0.0012}
  }
}
```

#### What We Would Add in Production (Not Implemented – Future Plan)

**Recommended tools:**

| Need | Tool | What It Does |
|------|------|-------------|
| **LLM Observability** | LangSmith / LangFuse | Visual tracing of prompts, chains, latency, and debugging – dashboard for viewing every LLM call |
| **Real-time Metrics** | Prometheus + Grafana | Collect metrics (latency, error rates, token usage) and display in graphical dashboards with alerts |
| **Centralized Logging** | ELK Stack / CloudWatch | Aggregate logs from all servers, search, audit trail |
| **Alerting** | PagerDuty / OpsGenie | Automatic alerts on failures, cost anomalies, or drift |

**Additional metrics we would track:**

```
┌─────────────────────────────────────────────────────┐
│         Additional Production Metrics                │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Reliability │  │  Drift       │  │  Budget   │ │
│  │              │  │  Detection   │  │  Alerts   │ │
│  │ • Error rate │  │ • Embedding  │  │ • $/day   │ │
│  │ • Timeout %  │  │   drift      │  │ • $/month │ │
│  │ • API avail. │  │ • Topic      │  │ • Budget  │ │
│  │ • Fallback   │  │   shift      │  │   ceiling │ │
│  │   triggers   │  │ • Score      │  │           │ │
│  │              │  │   degradation│  │           │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────┘
```

**Drift Detection (performance degradation over time):**

- **Embedding Drift**: Track average cosine similarity of queries to results. Sustained decline = knowledge base doesn't cover new incidents
- **Topic Shift**: Cluster incoming queries. New clusters not previously seen = need to update the knowledge base
- **Score Degradation**: Track average confidence scores. Decline = prompts or model no longer fit

> **Transparency**: The above is an **architectural plan** for production migration. In the current PoC, monitoring is based on Python logging + built-in cost/performance metrics returned in every response.

### 6.4 Summary – What Was Done and Why

| Decision | Reason |
|----------|--------|
| **GPT-4o-mini** as default | Optimal balance of cost, speed, and quality for a PoC |
| **Hybrid Search** (Vector + BM25) | Combining semantic + keyword search covers more cases |
| **Cross-Encoder Reranking** | Far higher accuracy than bi-encoder alone |
| **Grounding rules in prompts** | Hallucination prevention – critical in cybersecurity |
| **Structured output (Pydantic)** | Consistent, machine-parseable output |
| **Entity Validation (Regex + LLM)** | Regex = 100% precision, LLM = high recall. Combined at no extra cost |
| **Confidence scores** | Transparency – the analyst knows when to trust and when to verify |
| **Progressive disclosure** | Incomplete reports get clarification questions instead of guesses |
| **Multi-model support** | Flexibility – switch models without code changes |

---
