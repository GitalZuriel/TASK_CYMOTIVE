"""
Main Cybersecurity Copilot orchestrator.

Wires together all pipeline stages: preprocessing → hybrid retrieval → RRF fusion →
cross-encoder reranking → LLM summarization & mitigation.

This is the single source of truth for pipeline logic. Both the web layer (app.py)
and the CLI (main.py) delegate to this module.
"""

import time
import logging
from dataclasses import dataclass, field

import config
from src.preprocessor import preprocess, PreprocessResult
from src.embeddings import get_embedding_model
from src.vector_store import VectorStore
from src.bm25_search import BM25Index
from src.hybrid_retrieval import HybridRetriever, FusedResult
from src.reranker import Reranker, RerankedResult
from src.prompts import (
    SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    MITIGATION_PROMPT,
    SUMMARY_PROMPT_JSON,
    MITIGATION_PROMPT_JSON,
    EDGE_CASE_PROMPT,
    format_similar_incidents,
    format_extracted_fields,
)
from src.llm_chain import LLMChain, LLMResponse
from src.schemas import StructuredSummary, StructuredMitigation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM-based query rewriting: symptoms → technical cybersecurity language
# ---------------------------------------------------------------------------
_REWRITE_PROMPT = """\
You are an automotive cybersecurity expert. Rewrite the following incident report
into a short, keyword-rich technical search query (2-3 sentences max).
Use precise automotive cybersecurity terms: ECU names, attack types, protocols,
standards (UDS, DoIP, OTA, CAN, etc.).
Do NOT explain — just output the rewritten query."""


def _rewrite_query_for_retrieval(llm: "LLMChain", original_query: str) -> str:
    """Use a cheap LLM call to translate symptom language into technical terms."""
    try:
        resp = llm.invoke(_REWRITE_PROMPT, original_query)
        rewritten = resp.content.strip()
        logger.info("Query rewritten for retrieval: %s", rewritten[:120])
        return rewritten
    except Exception:
        logger.warning("Query rewrite failed — falling back to original", exc_info=True)
        return original_query


@dataclass
class CopilotResponse:
    """Full structured response from the Copilot pipeline."""

    # Preprocessing
    preprocessed: PreprocessResult

    # Status
    status: str = "success"          # "success" | "clarification_needed"
    analysis_mode: str = "full"      # "full" | "needs_input"

    # Retrieval
    similar_incidents: list[RerankedResult] = field(default_factory=list)
    context_ids: set[str] = field(default_factory=set)
    retrieval_warnings: list[str] = field(default_factory=list)

    # LLM outputs (raw text — always available)
    summary: str = ""
    mitigation: str = ""
    clarification: str = ""

    # Structured output (None if parsing failed or structured output disabled)
    structured_summary: StructuredSummary | None = None
    structured_mitigation: StructuredMitigation | None = None
    is_structured: bool = False

    # Confidence check / progressive disclosure
    low_confidence_sections: list[str] = field(default_factory=list)
    auto_clarification: str | None = None
    partial_mitigation: dict | None = None

    # Stats
    total_latency_seconds: float = 0.0
    model_name: str = ""
    summary_response: LLMResponse | None = None
    mitigation_response: LLMResponse | None = None

    @property
    def llm_stats(self) -> dict:
        """Backward-compatible stats dict (used by main.py CLI)."""
        stats: dict = {}
        if self.status == "clarification_needed" and self.summary_response:
            stats["clarification"] = {
                "model": self.summary_response.model,
                "latency": self.summary_response.latency_seconds,
                "tokens": self.summary_response.total_tokens,
                "cost": self.summary_response.estimated_cost_usd,
            }
        else:
            if self.summary_response:
                stats["summary"] = {
                    "model": self.summary_response.model,
                    "latency": self.summary_response.latency_seconds,
                    "tokens": self.summary_response.total_tokens,
                    "cost": self.summary_response.estimated_cost_usd,
                }
            if self.mitigation_response:
                stats["mitigation"] = {
                    "model": self.mitigation_response.model,
                    "latency": self.mitigation_response.latency_seconds,
                    "tokens": self.mitigation_response.total_tokens,
                    "cost": self.mitigation_response.estimated_cost_usd,
                }
        return stats


class CybersecurityCopilot:
    """
    End-to-end automotive cybersecurity incident analysis copilot.

    Pipeline:
        1. Preprocess & validate input
        2. Hybrid retrieval (semantic + BM25 in parallel)
        3. RRF fusion
        4. Cross-Encoder reranking → top-K similar incidents
        5. Context guardrails (threshold, dedup, token budget)
        6. LLM summarization & mitigation (structured or markdown)
        7. Confidence check → progressive disclosure
        8. Return structured CopilotResponse
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "openai",
    ) -> None:
        """
        Initialize all copilot components.

        Args:
            llm_model: Default LLM ("gpt-4o", "gpt-4o-mini", or "claude-sonnet").
            embedding_model: Embedding provider ("openai" or "local").
        """
        logger.info(
            "Initializing CybersecurityCopilot (llm=%s, embeddings=%s)",
            llm_model, embedding_model,
        )

        self._default_model = llm_model

        # Embeddings
        self._emb_model = get_embedding_model(embedding_model)

        # Vector store
        self._vector_store = VectorStore(self._emb_model)
        if self._vector_store.count == 0:
            logger.info("Vector store empty — ingesting incidents")
            self._vector_store.ingest_incidents()

        # BM25 index
        self._bm25_index = BM25Index()
        self._bm25_index.build_index()

        # Hybrid retriever
        self._retriever = HybridRetriever(self._vector_store, self._bm25_index)

        # Reranker
        self._reranker = Reranker()

        logger.info("CybersecurityCopilot initialization complete")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        incident_report: str,
        model_name: str | None = None,
    ) -> CopilotResponse:
        """
        Full analysis pipeline for an incident report.

        Args:
            incident_report: Raw incident report text from the analyst.
            model_name: LLM model override. Defaults to the model set at init.

        Returns:
            CopilotResponse with summary, mitigation plan, similar incidents, and stats.
        """
        model = model_name or self._default_model
        pipeline_start = time.perf_counter()
        response = CopilotResponse(
            preprocessed=PreprocessResult(
                cleaned_text="", entities=None, is_valid=False,
            ),
            model_name=model,
        )

        # --- Step 1: Preprocess ---
        logger.info("Step 1: Preprocessing input")
        preprocessed = preprocess(incident_report)
        response.preprocessed = preprocessed

        # Handle invalid / too-short input
        if not preprocessed.is_valid:
            logger.info("Input invalid — generating clarification via LLM")
            llm = LLMChain(model)
            issues = "; ".join(preprocessed.warnings + preprocessed.clarification_questions)
            edge_prompt = EDGE_CASE_PROMPT.format(
                incident_report=incident_report,
                issues=issues,
            )
            llm_resp = llm.invoke(SYSTEM_PROMPT, edge_prompt)
            response.status = "clarification_needed"
            response.clarification = llm_resp.content
            response.summary_response = llm_resp
            response.total_latency_seconds = round(
                time.perf_counter() - pipeline_start, 2,
            )
            return response

        query = preprocessed.cleaned_text
        llm = LLMChain(model)

        # --- Step 1.5: LLM query rewrite (symptoms → technical terms) ---
        retrieval_query = _rewrite_query_for_retrieval(llm, query)

        # --- Step 2 & 3: Hybrid retrieval + RRF ---
        logger.info("Step 2-3: Hybrid retrieval + RRF fusion")
        retrieval = self._retriever.retrieve(retrieval_query)
        fused_results = retrieval.results
        if retrieval.is_degraded:
            logger.warning(
                "Degraded retrieval: %s unavailable",
                ", ".join(retrieval.degraded_sources),
            )
            response.retrieval_warnings.append(
                f"Partial retrieval ({', '.join(retrieval.degraded_sources)} unavailable)"
            )

        # --- Step 4: Cross-Encoder reranking + domain scoring ---
        logger.info("Step 4: Cross-encoder reranking + domain scoring")
        reranked = self._reranker.rerank(
            query, fused_results, preprocessed=preprocessed,
        )
        response.similar_incidents = reranked
        for r in reranked:
            logger.info(
                "SCORE_DEBUG %s: raw=%.4f norm=%.2f ce_logit=%.4f domain=%.4f",
                r.incident_id, r.raw_score, r.normalized_score,
                r.cross_encoder_score, r.domain_score,
            )

        # --- Step 5: RAG Context Guardrails ---
        similar_for_prompt = self._apply_context_guardrails(reranked, response)
        similar_text = format_similar_incidents(similar_for_prompt)

        # --- Step 6: LLM summarization & mitigation ---
        extracted_text = format_extracted_fields(preprocessed.extracted_fields)
        use_structured = config.STRUCTURED_OUTPUT

        # 6a: Summary
        logger.info("Step 6a: LLM summarization")
        summary_parsed, summary_resp, summary_ok = self._run_summary(
            llm, query, similar_text, extracted_text, use_structured,
        )
        response.summary = summary_resp.content
        response.summary_response = summary_resp
        response.structured_summary = summary_parsed if summary_ok else None

        # 6b: Mitigation
        logger.info("Step 6b: LLM mitigation planning")
        mitigation_parsed, mitigation_resp, mitigation_ok = self._run_mitigation(
            llm, summary_resp.content, similar_text, extracted_text, use_structured,
        )
        response.mitigation = mitigation_resp.content
        response.mitigation_response = mitigation_resp
        response.structured_mitigation = mitigation_parsed if mitigation_ok else None
        response.is_structured = summary_ok and mitigation_ok

        # --- Step 7: Confidence check → progressive disclosure ---
        self._check_confidence(
            llm, response, summary_parsed, summary_ok,
            mitigation_parsed, mitigation_ok, incident_report,
        )

        # Handle clarification for valid-but-incomplete input
        if preprocessed.needs_clarification and not response.clarification:
            response.clarification = (
                "Note: The report was analyzed but some information may be missing. "
                "Consider providing:\n"
                + "\n".join(f"- {q}" for q in preprocessed.clarification_questions)
            )

        response.total_latency_seconds = round(
            time.perf_counter() - pipeline_start, 2,
        )
        logger.info("Pipeline complete in %.2fs", response.total_latency_seconds)
        return response

    def search(self, query: str, top_k: int = 5) -> list[RerankedResult]:
        """
        Retrieval-only pipeline (no LLM call).

        Runs hybrid retrieval + RRF + reranking and returns similar incidents.
        """
        preprocessed = preprocess(query)
        retrieval = self._retriever.retrieve(query)
        return self._reranker.rerank(
            query, retrieval.results, top_k=top_k, preprocessed=preprocessed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_context_guardrails(
        reranked: list[RerankedResult],
        response: CopilotResponse,
    ) -> list[dict]:
        """Apply threshold, dedup, and token-budget guardrails."""
        # 1. Relevance threshold
        filtered = [
            r for r in reranked
            if r.normalized_score >= config.RERANK_SCORE_THRESHOLD
        ]

        # 2. Near-duplicate removal
        seen_keys: set[tuple[str, str]] = set()
        unique: list[RerankedResult] = []
        for r in filtered:
            key = (r.title, r.metadata.get("attack_vector", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(r)

        # 3. Token budget
        similar_for_prompt: list[dict] = []
        char_count = 0
        for r in unique:
            entry = {
                "title": r.title,
                "description": r.description,
                "metadata": r.metadata,
            }
            chunk_len = len(r.title) + len(r.description) + 200
            if char_count + chunk_len > config.MAX_CONTEXT_CHARS:
                break
            similar_for_prompt.append(entry)
            response.context_ids.add(r.incident_id)
            char_count += chunk_len

        logger.info(
            "Context guardrails: %d reranked → %d threshold → %d dedup → %d budget",
            len(reranked), len(filtered), len(unique), len(similar_for_prompt),
        )
        return similar_for_prompt

    @staticmethod
    def _run_summary(
        llm: LLMChain,
        query: str,
        similar_text: str,
        extracted_text: str,
        use_structured: bool,
    ) -> tuple:
        """Run the summarization step (structured or markdown)."""
        if use_structured:
            prompt = SUMMARY_PROMPT_JSON.format(
                similar_incidents=similar_text,
                incident_report=query,
                extracted_fields=extracted_text,
            )
            parsed, resp, ok = llm.invoke_structured(
                SYSTEM_PROMPT, prompt, StructuredSummary,
            )
            return parsed, resp, ok
        else:
            prompt = SUMMARY_PROMPT.format(
                similar_incidents=similar_text,
                incident_report=query,
                extracted_fields=extracted_text,
            )
            resp = llm.invoke(SYSTEM_PROMPT, prompt)
            return None, resp, False

    @staticmethod
    def _run_mitigation(
        llm: LLMChain,
        summary_content: str,
        similar_text: str,
        extracted_text: str,
        use_structured: bool,
    ) -> tuple:
        """Run the mitigation step (structured or markdown)."""
        if use_structured:
            prompt = MITIGATION_PROMPT_JSON.format(
                incident_summary=summary_content,
                similar_incidents=similar_text,
                extracted_fields=extracted_text,
            )
            parsed, resp, ok = llm.invoke_structured(
                SYSTEM_PROMPT, prompt, StructuredMitigation,
            )
            return parsed, resp, ok
        else:
            prompt = MITIGATION_PROMPT.format(
                incident_summary=summary_content,
                similar_incidents=similar_text,
                extracted_fields=extracted_text,
            )
            resp = llm.invoke(SYSTEM_PROMPT, prompt)
            return None, resp, False

    @staticmethod
    def _check_confidence(
        llm: LLMChain,
        response: CopilotResponse,
        summary_parsed: StructuredSummary | None,
        summary_ok: bool,
        mitigation_parsed: StructuredMitigation | None,
        mitigation_ok: bool,
        incident_report: str,
    ) -> None:
        """Run confidence check and set progressive-disclosure fields."""
        if summary_ok and summary_parsed:
            section_names = [
                "incident_overview", "severity", "attack_vector",
                "affected_systems", "key_indicators", "timeline",
            ]
            for name in section_names:
                section = getattr(summary_parsed, name)
                if section.confidence < config.CONFIDENCE_THRESHOLD:
                    response.low_confidence_sections.append(name)

        if response.low_confidence_sections:
            issues = f"Low confidence in: {', '.join(response.low_confidence_sections)}"
            edge_prompt = EDGE_CASE_PROMPT.format(
                incident_report=incident_report, issues=issues,
            )
            clar_resp = llm.invoke(SYSTEM_PROMPT, edge_prompt)
            response.auto_clarification = clar_resp.content

        # Determine analysis_mode
        if response.low_confidence_sections or response.auto_clarification:
            response.analysis_mode = "needs_input"

        # Build partial mitigation for needs_input mode
        if (
            response.analysis_mode == "needs_input"
            and mitigation_ok
            and mitigation_parsed
        ):
            partial_dump = mitigation_parsed.model_dump()
            partial_dump["short_term_actions"]["actions"] = []
            partial_dump["long_term_recommendations"]["actions"] = []
            partial_dump["related_standards"]["standards"] = []
            response.partial_mitigation = partial_dump
