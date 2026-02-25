"""
Cross-Encoder reranker with lightweight domain scoring.

Takes candidate documents from RRF fusion, jointly scores each
(query, document) pair with a cross-encoder, then blends in a
lightweight domain signal derived from the preprocessor's structured
entity extraction (attack indicators, ECU names, severity).

Design note (PoC):
    The cross-encoder provides the primary semantic ranking signal.
    The domain score is a small, general-purpose boost that leverages
    the structured metadata already extracted by the preprocessor —
    no hand-tuned keyword lists or vocabulary bridges.
"""

import logging
import math
import re
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

import config
from src.hybrid_retrieval import FusedResult
from src.preprocessor import PreprocessResult

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Blending weights ---
# Cross-encoder is the primary signal; domain score is a light boost.
ALPHA = 0.80  # cross-encoder weight
BETA = 0.20   # domain score weight

# Severity string → numeric score mapping
_SEVERITY_TO_SCORE: dict[str, int] = {
    "critical": 5, "high": 4, "medium": 3, "low": 2,
}


def _tokenize_lower(text: str) -> set[str]:
    """Split text into lowercase alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _compute_domain_score(
    query_attack_indicators: list[str],
    query_ecu_names: list[str],
    query_severity: str | None,
    query_text: str,
    candidate_metadata: dict,
) -> float:
    """
    Lightweight domain similarity between query and candidate.

    Uses only the structured fields already extracted by the preprocessor
    (attack indicators, ECU names, severity) and the candidate's metadata.
    No hand-tuned keyword lists or vocabulary bridges.

    Returns a score in [0, 1].
    """
    score = 0.0

    attack_vector = candidate_metadata.get("attack_vector", "").lower()
    title = candidate_metadata.get("title", "").lower()
    affected_system = candidate_metadata.get("affected_system", "").lower()

    # --- 1. Attack type match (weight: 0.35) ---
    # Check if any preprocessor-extracted attack indicator appears in
    # the candidate's title or attack_vector field.
    searchable = f"{title} {attack_vector}"
    for indicator in query_attack_indicators:
        if indicator.lower() in searchable:
            score += 0.35
            break

    # --- 2. System / ECU match (weight: 0.35) ---
    # Check if any preprocessor-extracted ECU name appears in the
    # candidate's affected_system or attack_vector.
    system_searchable = f"{affected_system} {attack_vector}"
    for ecu in query_ecu_names:
        if ecu.lower() in system_searchable:
            score += 0.35
            break

    # --- 3. MITRE tactics overlap (weight: 0.15) ---
    incident_mitre_str = candidate_metadata.get("mitre_tactics", "")
    if incident_mitre_str:
        incident_mitre = set(incident_mitre_str.split(","))
        query_mitre = set(re.findall(r"TA\d{4}", query_text, re.IGNORECASE))
        if query_mitre and incident_mitre:
            intersection = query_mitre & incident_mitre
            union = query_mitre | incident_mitre
            score += (len(intersection) / len(union)) * 0.15

    # --- 4. Severity similarity (weight: 0.15) ---
    incident_severity_score = candidate_metadata.get("severity_score", 0)
    if query_severity and incident_severity_score:
        q_score = _SEVERITY_TO_SCORE.get(
            query_severity.lower().split()[0], 0
        )
        if q_score > 0:
            diff = abs(q_score - incident_severity_score)
            score += (1 - diff / 4) * 0.15

    return score


@dataclass
class RerankedResult:
    """A reranked result with raw and display-normalized scores."""
    incident_id: str
    title: str
    description: str
    raw_score: float             # blended sigmoid(CE)*ALPHA + domain*BETA — used for ranking
    normalized_score: float      # min-max normalized (10-95) — for display & guardrails
    cross_encoder_score: float   # raw CE logit (for debug)
    domain_score: float          # domain similarity (0-1)
    rrf_score: float             # original RRF score for comparison
    metadata: dict
    source: str


class Reranker:
    """Cross-Encoder reranker with lightweight domain scoring."""

    def __init__(self) -> None:
        """Load the cross-encoder model."""
        logger.info("Loading cross-encoder model: %s", CROSS_ENCODER_MODEL)
        self._model = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder model loaded successfully")

    def rerank(
        self,
        query: str,
        candidates: list[FusedResult],
        top_k: int | None = None,
        preprocessed: PreprocessResult | None = None,
    ) -> list[RerankedResult]:
        """
        Rerank candidate documents using cross-encoder + domain scoring.

        Args:
            query: The original search query.
            candidates: Candidate documents from RRF fusion.
            top_k: Number of top results to return. Defaults to config.TOP_K_RERANK.
            preprocessed: Preprocessor output for domain scoring. When None,
                          domain scores are 0 and ranking falls back to CE-only.

        Returns:
            List of RerankedResult sorted by blended score (highest first).
        """
        top_k = top_k or config.TOP_K_RERANK

        if not candidates:
            logger.warning("No candidates to rerank")
            return []

        # --- STEP 1: Cross-encoder scores ---
        pairs = [(query, c.description) for c in candidates]
        scores = self._model.predict(pairs)

        # --- STEP 2: Domain scores ---
        query_indicators: list[str] = []
        query_ecu_names: list[str] = []
        query_severity: str | None = None

        if preprocessed and preprocessed.entities:
            query_indicators = preprocessed.entities.attack_indicators
            query_ecu_names = preprocessed.entities.ecu_names
            severity_field = preprocessed.extracted_fields.get("severity")
            if severity_field:
                query_severity = severity_field.value

        domain_scores = [
            _compute_domain_score(
                query_indicators, query_ecu_names, query_severity,
                query, c.metadata,
            )
            for c in candidates
        ]

        # --- STEP 3: Blend raw scores and sort ---
        results: list[RerankedResult] = []
        for candidate, ce_logit, d_score in zip(candidates, scores, domain_scores):
            # Sigmoid normalization keeps the raw CE logit on a 0-100 scale
            # for blending, but preserves the original ranking order.
            sigmoid_ce = 1.0 / (1.0 + math.exp(-float(ce_logit))) * 100.0
            raw = ALPHA * sigmoid_ce + BETA * (d_score * 100.0)

            results.append(RerankedResult(
                incident_id=candidate.incident_id,
                title=candidate.title,
                description=candidate.description,
                raw_score=round(raw, 4),
                normalized_score=0.0,  # filled after top-k
                cross_encoder_score=round(float(ce_logit), 4),
                domain_score=round(d_score, 4),
                rrf_score=candidate.rrf_score,
                metadata=candidate.metadata,
                source=candidate.source,
            ))

        # Sort by raw_score (determines final ranking — never changes)
        results.sort(key=lambda x: x.raw_score, reverse=True)
        results = results[:top_k]

        # --- STEP 4: Post-rerank min-max normalization (display only) ---
        if results:
            raw_vals = [r.raw_score for r in results]
            min_raw = min(raw_vals)
            max_raw = max(raw_vals)
            spread = max_raw - min_raw

            if spread < 1e-9:
                # All scores identical — assign midpoint to avoid div-by-zero
                for r in results:
                    r.normalized_score = 50.0
            else:
                for r in results:
                    r.normalized_score = round(
                        10.0 + ((r.raw_score - min_raw) / spread) * 85.0, 2
                    )

        logger.info(
            "Reranked %d candidates → top %d | best raw=%.4f norm=%.1f CE_logit=%.2f domain=%.4f",
            len(candidates),
            len(results),
            results[0].raw_score if results else 0.0,
            results[0].normalized_score if results else 0.0,
            results[0].cross_encoder_score if results else 0.0,
            results[0].domain_score if results else 0.0,
        )
        return results
