"""
Cybersecurity Copilot — Flask Web UI.

Serves the dashboard and provides API endpoints for incident analysis.
The pipeline logic lives in src/copilot.py — this module is a thin web layer
that converts CopilotResponse to JSON.
"""

import sys
import os
import json
import logging
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify

import config
from src.preprocessor import preprocess
from src.copilot import CybersecurityCopilot, CopilotResponse

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval metrics (used only when the caller provides expected_ids)
# ---------------------------------------------------------------------------

def _precision_at_k(retrieved_ids: list, expected_ids: list, k: int = 2) -> float:
    top_k = retrieved_ids[:k]
    relevant = sum(1 for rid in top_k if rid in expected_ids)
    return relevant / k if k > 0 else 0.0


def _recall_at_k(retrieved_ids: list, expected_ids: list, k: int = 2) -> float:
    if not expected_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for eid in expected_ids if eid in top_k)
    return found / len(expected_ids)


def _mrr(retrieved_ids: list, expected_ids: list) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_ids: list, expected_ids: list, k: int = 2) -> float:
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, rid in enumerate(top_k):
        rel = 1.0 if rid in expected_ids else 0.0
        dcg += rel / math.log2(i + 2)
    ideal_relevant = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(retrieved_ids: list, expected_ids: list, k: int = 2) -> dict:
    """Compute retrieval metrics against explicit expected_ids."""
    return {
        "precision_at_k": round(_precision_at_k(retrieved_ids, expected_ids, k), 2),
        "recall_at_k": round(_recall_at_k(retrieved_ids, expected_ids, k), 2),
        "mrr": round(_mrr(retrieved_ids, expected_ids), 2),
        "ndcg_at_k": round(_ndcg_at_k(retrieved_ids, expected_ids, k), 2),
        "k": k,
        "retrieved_ids": retrieved_ids[:k],
        "expected_ids": expected_ids,
    }


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

_base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(_base_dir, "web", "templates"),
    static_folder=os.path.join(_base_dir, "web", "static"),
)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


@app.after_request
def add_no_cache(response):
    """Disable browser caching during development."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# Global copilot singleton (initialized once on first request)
# ---------------------------------------------------------------------------

_copilot: CybersecurityCopilot | None = None


def get_copilot() -> CybersecurityCopilot:
    """Lazy-initialize the Copilot (loads embedding + reranker models once)."""
    global _copilot
    if _copilot is None:
        logger.info("Initializing CybersecurityCopilot...")
        _copilot = CybersecurityCopilot(
            llm_model=config.DEFAULT_LLM,
            embedding_model="local",  # local for fast startup, no embedding API needed
        )
        logger.info("CybersecurityCopilot ready")
    return _copilot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_clarification(preprocessed) -> str:
    """Format extracted fields + missing questions into a clear clarification message."""
    if not preprocessed.needs_clarification and not preprocessed.extracted_fields:
        return ""

    parts: list[str] = []

    if preprocessed.extracted_fields:
        parts.append("**Extracted from report:**")
        field_labels = {
            "affected_subsystem": "Affected Subsystem",
            "attack_type": "Attack Type",
            "severity": "Severity",
            "timestamp": "Timestamp",
            "cve": "CVE",
        }
        for name, f in preprocessed.extracted_fields.items():
            label = field_labels.get(name, name.replace("_", " ").title())
            confidence_note = "" if f.confidence == "high" else f" *(needs confirmation)*"
            parts.append(f"- {label}: {f.value}{confidence_note}")

    if preprocessed.clarification_questions:
        parts.append("\n**Please confirm or provide:**")
        for q in preprocessed.clarification_questions:
            parts.append(f"- {q}")

    return "\n".join(parts)


def _response_to_json(result: CopilotResponse, expected_ids: list | None = None) -> dict:
    """Convert a CopilotResponse to the JSON format expected by the frontend."""

    # --- Clarification-only response (invalid input) ---
    if result.status == "clarification_needed":
        stats: dict = {
            "total_latency": result.total_latency_seconds,
            "model": result.model_name,
        }
        if result.summary_response:
            stats["total_tokens"] = result.summary_response.total_tokens
            stats["total_cost"] = result.summary_response.estimated_cost_usd
        return {
            "status": "clarification_needed",
            "clarification": result.clarification,
            "warnings": result.preprocessed.warnings,
            "entities": {},
            "similar_incidents": [],
            "summary": "",
            "mitigation": "",
            "stats": stats,
        }

    # --- Full analysis response ---
    preprocessed = result.preprocessed
    entities = preprocessed.entities

    # Retrieval metrics (only when expected_ids provided by caller)
    retrieval_metrics = None
    if expected_ids:
        retrieved_ids = [r.incident_id for r in result.similar_incidents]
        retrieval_metrics = compute_retrieval_metrics(
            retrieved_ids, expected_ids, k=config.TOP_K_RERANK,
        )

    # Build stats
    stats = {
        "total_latency": result.total_latency_seconds,
        "model": result.model_name,
    }
    if result.summary_response and result.mitigation_response:
        stats.update({
            "summary_tokens": result.summary_response.total_tokens,
            "summary_latency": result.summary_response.latency_seconds,
            "mitigation_tokens": result.mitigation_response.total_tokens,
            "mitigation_latency": result.mitigation_response.latency_seconds,
            "total_tokens": (
                result.summary_response.total_tokens
                + result.mitigation_response.total_tokens
            ),
            "total_cost": round(
                result.summary_response.estimated_cost_usd
                + result.mitigation_response.estimated_cost_usd,
                6,
            ),
        })

    return {
        "status": "success",
        "analysis_mode": result.analysis_mode,
        # Backward-compatible raw text
        "summary": result.summary,
        "mitigation": result.mitigation,
        # Structured data (None if parsing failed or disabled)
        "structured_summary": (
            result.structured_summary.model_dump()
            if result.structured_summary else None
        ),
        "structured_mitigation": (
            result.structured_mitigation.model_dump()
            if result.structured_mitigation else None
        ),
        "partial_mitigation": result.partial_mitigation,
        "is_structured": result.is_structured,
        "low_confidence_sections": result.low_confidence_sections,
        "auto_clarification": result.auto_clarification,
        # Similar incidents
        "similar_incidents": [
            {
                "id": r.incident_id,
                "title": r.title,
                "severity": r.metadata.get("severity", "N/A"),
                "attack_vector": r.metadata.get("attack_vector", "N/A"),
                "affected_system": r.metadata.get("affected_system", "N/A"),
                "rerank_score": r.normalized_score,
                "raw_score": r.raw_score,
                "source": r.source,
                "in_context": r.incident_id in result.context_ids,
            }
            for r in result.similar_incidents
        ],
        # Entities
        "entities": {
            "cve_ids": entities.cve_ids,
            "ip_addresses": entities.ip_addresses,
            "ecu_names": entities.ecu_names,
            "attack_indicators": entities.attack_indicators,
        },
        "validated_entities": (
            result.structured_summary.validated_entities.model_dump()
            if result.structured_summary and result.structured_summary.validated_entities
            else None
        ),
        "clarification": _format_clarification(preprocessed),
        "extracted_fields": {
            name: {"value": f.value, "confidence": f.confidence, "source": f.source}
            for name, f in preprocessed.extracted_fields.items()
        },
        "warnings": preprocessed.warnings + result.retrieval_warnings,
        "retrieval_metrics": retrieval_metrics,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("dashboard.html")


@app.route("/architecture")
def architecture():
    """Serve the Architecture & Design Decisions page."""
    return render_template("architecture.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Full analysis endpoint.

    Expects JSON: {"report": "...", "model": "gpt-4o-mini"}
    Returns JSON with summary, mitigation, similar incidents, and stats.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    report_text = data.get("report", "").strip()
    model_name = data.get("model", config.DEFAULT_LLM)
    expected_ids = data.get("expected_ids")

    if not report_text:
        return jsonify({"error": "No report text provided"}), 400

    try:
        copilot = get_copilot()
        result = copilot.analyze(report_text, model_name=model_name)
        return jsonify(_response_to_json(result, expected_ids))

    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({
            "error": str(e),
            "stats": {"total_latency": 0},
        }), 500


@app.route("/api/precheck", methods=["POST"])
def precheck():
    """
    Lightweight validation endpoint.

    Runs preprocessing only — no LLM call, no retrieval.
    Returns whether the input has enough information for a full analysis.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    report_text = data.get("report", "").strip()

    if not report_text:
        return jsonify({"error": "No report text provided"}), 400

    preprocessed = preprocess(report_text)

    if not preprocessed.is_valid or preprocessed.needs_clarification:
        return jsonify({
            "status": "needs_input",
            "warnings": preprocessed.warnings,
            "clarification_questions": preprocessed.clarification_questions,
            "clarification": _format_clarification(preprocessed),
            "extracted_fields": {
                name: {"value": f.value, "confidence": f.confidence, "source": f.source}
                for name, f in preprocessed.extracted_fields.items()
            },
        })

    return jsonify({
        "status": "ok",
        "extracted_fields": {
            name: {"value": f.value, "confidence": f.confidence, "source": f.source}
            for name, f in preprocessed.extracted_fields.items()
        },
    })


@app.route("/api/incidents", methods=["GET"])
def list_incidents():
    """Return all incidents from the database."""
    with open(config.INCIDENTS_PATH, "r", encoding="utf-8") as f:
        incidents = json.load(f)
    return jsonify(incidents)


@app.route("/api/search", methods=["POST"])
def search():
    """Retrieval-only endpoint (no LLM call)."""
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    copilot = get_copilot()
    reranked = copilot.search(query, top_k=5)

    return jsonify({
        "results": [
            {
                "id": r.incident_id,
                "title": r.title,
                "severity": r.metadata.get("severity", "N/A"),
                "attack_vector": r.metadata.get("attack_vector", "N/A"),
                "rerank_score": r.normalized_score,
                "raw_score": r.raw_score,
                "rrf_score": round(r.rrf_score, 4),
                "source": r.source,
            }
            for r in reranked
        ]
    })


if __name__ == "__main__":
    # Pre-warm pipeline on startup
    print("Pre-loading pipeline...")
    get_copilot()
    print("Ready! Opening http://localhost:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
