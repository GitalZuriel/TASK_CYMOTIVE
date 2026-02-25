"""
Hybrid retrieval combining ChromaDB semantic search and BM25 keyword search.

Runs both retrievers in parallel using ThreadPoolExecutor and merges results
with Reciprocal Rank Fusion (RRF).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import config
from src.vector_store import VectorStore, VectorSearchResult
from src.bm25_search import BM25Index, BM25SearchResult

logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """A single result after RRF fusion."""
    incident_id: str
    title: str
    description: str
    rrf_score: float
    metadata: dict
    source: str  # "semantic", "bm25", or "both"


@dataclass
class RetrievalResult:
    """Result of hybrid retrieval, including degradation info."""
    results: list[FusedResult]
    degraded_sources: list[str]  # e.g. ["semantic"] if semantic search failed

    @property
    def is_degraded(self) -> bool:
        return len(self.degraded_sources) > 0


def rrf_fusion(
    semantic_results: list[VectorSearchResult],
    bm25_results: list[BM25SearchResult],
    k: int = 60,
) -> list[FusedResult]:
    """
    Merge results using Reciprocal Rank Fusion.

    RRF score for each document = sum(1 / (k + rank)) across all ranked lists
    where the document appears. This produces a unified ranking that balances
    semantic and keyword relevance without needing score normalization.

    Args:
        semantic_results: Results from vector similarity search (best first).
        bm25_results: Results from BM25 keyword search (best first).
        k: RRF constant (default 60). Higher values give more weight to
           documents that appear in multiple lists vs. high-ranked in one.

    Returns:
        Deduplicated list of FusedResult sorted by RRF score (highest first).
    """
    rrf_scores: dict[str, float] = {}
    doc_info: dict[str, dict] = {}
    doc_sources: dict[str, set] = {}

    # Score semantic results
    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result.incident_id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_sources.setdefault(doc_id, set()).add("semantic")
        doc_info[doc_id] = {
            "title": result.title,
            "description": result.description,
            "metadata": result.metadata,
        }

    # Score BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result.incident_id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_sources.setdefault(doc_id, set()).add("bm25")
        # Keep info from BM25 if not already present from semantic
        if doc_id not in doc_info:
            doc_info[doc_id] = {
                "title": result.title,
                "description": result.description,
                "metadata": result.metadata,
            }

    # Build fused results sorted by RRF score
    fused: list[FusedResult] = []
    for doc_id in sorted(rrf_scores, key=rrf_scores.get, reverse=True):
        sources = doc_sources[doc_id]
        source_label = "both" if len(sources) > 1 else sources.pop()
        info = doc_info[doc_id]
        fused.append(FusedResult(
            incident_id=doc_id,
            title=info["title"],
            description=info["description"],
            rrf_score=rrf_scores[doc_id],
            metadata=info["metadata"],
            source=source_label,
        ))

    logger.info(
        "RRF fusion: %d semantic + %d BM25 → %d fused results",
        len(semantic_results),
        len(bm25_results),
        len(fused),
    )
    return fused


class HybridRetriever:
    """Orchestrates parallel semantic + BM25 retrieval with RRF fusion."""

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> RetrievalResult:
        """
        Run hybrid retrieval: semantic and BM25 in parallel, then RRF fusion.

        If one retriever fails, results from the other are used alone.
        If both fail, an empty result is returned.

        Args:
            query: The search query.
            top_k: Number of candidates per retriever. Defaults to config value.
            rrf_k: RRF constant. Defaults to config value.

        Returns:
            RetrievalResult with fused results and degradation info.
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        rrf_k = rrf_k or config.RRF_K

        semantic_results: list[VectorSearchResult] = []
        bm25_results: list[BM25SearchResult] = []
        degraded_sources: list[str] = []

        # Run both searches in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_semantic = executor.submit(self._vector_store.query, query, top_k)
            future_bm25 = executor.submit(self._bm25_index.search, query, top_k)

            for future in as_completed([future_semantic, future_bm25]):
                if future is future_semantic:
                    try:
                        semantic_results = future.result()
                    except Exception:
                        logger.warning(
                            "Semantic search failed — continuing with BM25 only",
                            exc_info=True,
                        )
                        degraded_sources.append("semantic")
                else:
                    try:
                        bm25_results = future.result()
                    except Exception:
                        logger.warning(
                            "BM25 search failed — continuing with semantic only",
                            exc_info=True,
                        )
                        degraded_sources.append("bm25")

        if len(degraded_sources) == 2:
            logger.error("Both semantic and BM25 search failed — returning empty results")
            return RetrievalResult(results=[], degraded_sources=degraded_sources)

        fused = rrf_fusion(semantic_results, bm25_results, k=rrf_k)
        return RetrievalResult(results=fused, degraded_sources=degraded_sources)
