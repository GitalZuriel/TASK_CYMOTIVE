"""
Unit tests for core pipeline components.

Covers: preprocessor entity extraction, RRF fusion, reranker normalization,
and Pydantic schema parsing.
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ── Test 1: Preprocessor — entity extraction ──────────────────────────

class TestPreprocessor:
    """Test that the preprocessor correctly extracts entities and handles edge cases."""

    def test_extracts_cve_ids(self):
        from src.preprocessor import preprocess

        result = preprocess(
            "Detected exploit of CVE-2024-2851 on the infotainment ECU. "
            "Also related to CVE-2023-1234."
        )
        assert result.is_valid
        assert "CVE-2024-2851" in result.entities.cve_ids
        assert "CVE-2023-1234" in result.entities.cve_ids

    def test_extracts_ip_addresses(self):
        from src.preprocessor import preprocess

        result = preprocess(
            "C2 traffic detected from 10.0.0.45 to external host 185.220.101.42 "
            "via the telematics control unit."
        )
        assert "10.0.0.45" in result.entities.ip_addresses
        assert "185.220.101.42" in result.entities.ip_addresses

    def test_extracts_ecu_names(self):
        from src.preprocessor import preprocess

        result = preprocess(
            "The gateway ECU forwarded unauthorized CAN frames from the "
            "infotainment head unit to the ADAS domain controller."
        )
        ecus = result.entities.ecu_names
        assert "gateway" in ecus
        assert "infotainment" in ecus
        assert "ADAS" in ecus

    def test_extracts_attack_indicators(self):
        from src.preprocessor import preprocess

        result = preprocess(
            "CAN bus injection attack detected with spoofed brake frames. "
            "Data exfiltration via DNS tunneling confirmed."
        )
        indicators = result.entities.attack_indicators
        assert "injection" in indicators
        assert "spoofing" in indicators
        assert "exfiltration" in indicators
        assert "tunneling" in indicators

    def test_short_input_is_invalid(self):
        from src.preprocessor import preprocess

        result = preprocess("ecu hacked")
        assert not result.is_valid
        assert result.needs_clarification
        assert len(result.clarification_questions) > 0

    def test_empty_input_is_invalid(self):
        from src.preprocessor import preprocess

        result = preprocess("")
        assert not result.is_valid

    def test_valid_report_extracts_severity(self):
        from src.preprocessor import preprocess

        result = preprocess(
            "Critical severity alert: unauthorized firmware modification "
            "detected on the brake control ECU during routine diagnostics."
        )
        assert "severity" in result.extracted_fields
        assert result.extracted_fields["severity"].value == "Critical"


# ── Test 2: RRF Fusion — documents appearing in both retrievers ───────

class TestRRFFusion:
    """Test that RRF fusion correctly merges and ranks results."""

    def test_document_in_both_retrievers_ranks_higher(self):
        from src.hybrid_retrieval import rrf_fusion
        from src.vector_store import VectorSearchResult
        from src.bm25_search import BM25SearchResult

        # INC-002 appears in both; INC-004 only in semantic; INC-006 only in BM25
        semantic = [
            VectorSearchResult(
                incident_id="INC-002", title="CAN Injection", description="...",
                score=0.9, metadata={},
            ),
            VectorSearchResult(
                incident_id="INC-004", title="Firmware Tamper", description="...",
                score=0.7, metadata={},
            ),
        ]
        bm25 = [
            BM25SearchResult(
                incident_id="INC-002", title="CAN Injection", description="...",
                score=5.0, metadata={},
            ),
            BM25SearchResult(
                incident_id="INC-006", title="BLE Attack", description="...",
                score=3.0, metadata={},
            ),
        ]

        fused = rrf_fusion(semantic, bm25, k=60)

        # INC-002 should be first (appeared in both)
        assert fused[0].incident_id == "INC-002"
        assert fused[0].source == "both"

        # Its score should be higher than either single-source document
        single_source_scores = [r.rrf_score for r in fused if r.source != "both"]
        assert fused[0].rrf_score > max(single_source_scores)

    def test_empty_inputs_return_empty(self):
        from src.hybrid_retrieval import rrf_fusion

        assert rrf_fusion([], []) == []

    def test_single_source_still_works(self):
        from src.hybrid_retrieval import rrf_fusion
        from src.vector_store import VectorSearchResult

        semantic = [
            VectorSearchResult(
                incident_id="INC-001", title="DNS Tunneling", description="...",
                score=0.8, metadata={},
            ),
        ]
        fused = rrf_fusion(semantic, [], k=60)
        assert len(fused) == 1
        assert fused[0].incident_id == "INC-001"
        assert fused[0].source == "semantic"


# ── Test 3: Reranker — normalization range ─────────────────────────────

class TestReranker:
    """Test that reranker normalization produces scores in the expected range."""

    def test_normalized_scores_in_range(self):
        from src.reranker import Reranker
        from src.hybrid_retrieval import FusedResult

        reranker = Reranker()
        candidates = [
            FusedResult(
                incident_id="INC-001", title="DNS Tunneling via Infotainment",
                description="Anomalous DNS traffic from infotainment ECU.",
                rrf_score=0.03, metadata={}, source="both",
            ),
            FusedResult(
                incident_id="INC-002", title="CAN Bus Injection on Brake",
                description="Unauthorized CAN frames targeting brake ECU.",
                rrf_score=0.02, metadata={}, source="semantic",
            ),
            FusedResult(
                incident_id="INC-003", title="OTA Server Compromise",
                description="Compromised OTA update server.",
                rrf_score=0.01, metadata={}, source="bm25",
            ),
        ]

        results = reranker.rerank("DNS tunneling attack", candidates, top_k=3)

        assert len(results) > 0
        for r in results:
            # Normalized scores should be in the 10-95 display range
            assert 10.0 <= r.normalized_score <= 95.0, (
                f"{r.incident_id} score {r.normalized_score} outside [10, 95]"
            )

    def test_single_candidate_gets_midpoint(self):
        from src.reranker import Reranker
        from src.hybrid_retrieval import FusedResult

        reranker = Reranker()
        candidates = [
            FusedResult(
                incident_id="INC-001", title="DNS Tunneling",
                description="DNS tunneling attack on infotainment.",
                rrf_score=0.03, metadata={}, source="both",
            ),
        ]

        results = reranker.rerank("DNS attack", candidates, top_k=1)
        # Single result → spread is 0 → midpoint (50.0)
        assert results[0].normalized_score == 50.0

    def test_empty_candidates(self):
        from src.reranker import Reranker

        reranker = Reranker()
        assert reranker.rerank("test", [], top_k=2) == []


# ── Test 4: Pydantic schemas — structured output parsing ──────────────

class TestSchemas:
    """Test that Pydantic schemas correctly validate structured LLM output."""

    def test_structured_summary_parses(self):
        from src.schemas import StructuredSummary

        data = {
            "incident_overview": {
                "text": "DNS tunneling attack detected on infotainment ECU.",
                "confidence": 0.95,
            },
            "severity": {
                "level": "High",
                "justification": "Fleet-wide data exfiltration via C2 channel.",
                "confidence": 0.9,
            },
            "attack_vector": {
                "method": "Network",
                "details": "DNS TXT query tunneling",
                "confidence": 0.85,
            },
            "affected_systems": {
                "systems": ["Infotainment ECU", "TCU"],
                "confidence": 0.9,
            },
            "key_indicators": {
                "indicators": ["CVE-2024-2851", "185.220.101.42"],
                "confidence": 0.95,
            },
            "timeline": {
                "events": [
                    {"step": "OTA update deployed", "is_estimated": False},
                    {"step": "C2 beaconing begins", "is_estimated": True},
                ],
                "confidence": 0.7,
            },
        }

        summary = StructuredSummary.model_validate(data)
        assert summary.severity.level.value == "High"
        assert summary.avg_confidence > 0.5
        assert summary.min_confidence == 0.7
        assert len(summary.affected_systems.systems) == 2

    def test_structured_summary_rejects_invalid_severity(self):
        from src.schemas import StructuredSummary
        from pydantic import ValidationError

        data = {
            "incident_overview": {"text": "Something happened.", "confidence": 0.5},
            "severity": {
                "level": "EXTREME",  # invalid enum value
                "justification": "Bad.",
                "confidence": 0.5,
            },
            "attack_vector": {"method": "Network", "confidence": 0.5},
            "affected_systems": {"systems": ["ECU"], "confidence": 0.5},
            "key_indicators": {"confidence": 0.5},
            "timeline": {"confidence": 0.5},
        }

        with pytest.raises(ValidationError):
            StructuredSummary.model_validate(data)

    def test_structured_mitigation_parses(self):
        from src.schemas import StructuredMitigation

        data = {
            "immediate_actions": {
                "actions": [{"action": "Block C2 IP", "grounding": "From report"}],
                "confidence": 0.9,
            },
            "short_term_actions": {
                "actions": [{"action": "Patch CVE", "grounding": "Known vuln"}],
                "confidence": 0.8,
            },
            "long_term_recommendations": {
                "actions": [],
                "confidence": 0.6,
            },
            "related_standards": {
                "standards": [
                    {"standard": "ISO 21434", "relevance": "Risk assessment", "is_general_practice": False},
                ],
                "confidence": 0.7,
            },
        }

        mitigation = StructuredMitigation.model_validate(data)
        assert mitigation.avg_confidence > 0.5
        assert len(mitigation.immediate_actions.actions) == 1


# ── Test 5: CopilotResponse — llm_stats backward compatibility ────────

class TestCopilotResponse:
    """Test that CopilotResponse.llm_stats works for main.py CLI."""

    def test_llm_stats_full_analysis(self):
        from src.copilot import CopilotResponse
        from src.preprocessor import PreprocessResult, ExtractedEntities
        from src.llm_chain import LLMResponse

        resp = CopilotResponse(
            preprocessed=PreprocessResult(
                cleaned_text="test", entities=ExtractedEntities(),
            ),
            summary_response=LLMResponse(
                content="s", model="gpt-4o-mini", latency_seconds=1.0,
                input_tokens=100, output_tokens=50, total_tokens=150,
                estimated_cost_usd=0.001,
            ),
            mitigation_response=LLMResponse(
                content="m", model="gpt-4o-mini", latency_seconds=0.8,
                input_tokens=80, output_tokens=40, total_tokens=120,
                estimated_cost_usd=0.0008,
            ),
        )

        stats = resp.llm_stats
        assert "summary" in stats
        assert "mitigation" in stats
        assert stats["summary"]["tokens"] == 150
        assert stats["mitigation"]["tokens"] == 120

    def test_llm_stats_clarification_mode(self):
        from src.copilot import CopilotResponse
        from src.preprocessor import PreprocessResult, ExtractedEntities
        from src.llm_chain import LLMResponse

        resp = CopilotResponse(
            preprocessed=PreprocessResult(
                cleaned_text="", entities=ExtractedEntities(), is_valid=False,
            ),
            status="clarification_needed",
            summary_response=LLMResponse(
                content="q", model="gpt-4o-mini", latency_seconds=0.5,
                input_tokens=50, output_tokens=30, total_tokens=80,
                estimated_cost_usd=0.0003,
            ),
        )

        stats = resp.llm_stats
        assert "clarification" in stats
        assert "summary" not in stats
