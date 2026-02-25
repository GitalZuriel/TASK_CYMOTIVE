"""
Shared retrieval evaluation metrics.

Used by dev set, holdout set, and embedding comparison evaluations.
Centralises metric computation to avoid duplication across scripts.
"""

import math
from dataclasses import dataclass, field
from itertools import combinations


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result of evaluating a single query."""

    query_id: str
    description: str
    retrieved_ids: list[str]
    expected_top1: list[str]
    acceptable_top2: list[str]

    # Metrics
    p_at_1: float = 0.0
    p_at_2: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_2: float = 0.0

    # Paraphrase tracking (optional)
    group: str | None = None
    variant: str | None = None
    noise_level: str = "clean"


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in expected) / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of relevant documents found in top-k."""
    if not expected:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for e in expected if e in top_k) / len(expected)


def mrr_score(retrieved: list[str], expected: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant result."""
    for rank, r in enumerate(retrieved, 1):
        if r in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ k (binary relevance)."""
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 if r in expected else 0.0) / math.log2(i + 2)
        for i, r in enumerate(top_k)
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Flexible P@2 for holdout (uses acceptable_top2 set)
# ---------------------------------------------------------------------------

def flexible_precision_at_2(retrieved: list[str],
                            expected_top1: list[str],
                            acceptable_top2: list[str]) -> float:
    """
    P@2 that accepts *any* of ``acceptable_top2`` in slot 2.

    Slot 1 is scored against ``expected_top1``.
    Slot 2 is scored against ``acceptable_top2`` (superset of expected).
    """
    hits = 0
    if len(retrieved) >= 1 and retrieved[0] in expected_top1:
        hits += 1
    if len(retrieved) >= 2 and retrieved[1] in acceptable_top2:
        hits += 1
    return hits / 2


# ---------------------------------------------------------------------------
# Paraphrase stability
# ---------------------------------------------------------------------------

def paraphrase_stability(group_results: list[QueryResult]) -> dict:
    """
    Compute agreement between paraphrase variants of the same base query.

    Parameters
    ----------
    group_results : list[QueryResult]
        Typically 3 results (base, para1, para2) for one paraphrase group.

    Returns
    -------
    dict with keys: top1_agreement, top2_jaccard, top5_jaccard
    """
    if len(group_results) < 2:
        return {"top1_agreement": 1.0, "top2_jaccard": 1.0, "top5_jaccard": 1.0}

    pairs = list(combinations(group_results, 2))

    def _jaccard(set_a: set, set_b: set) -> float:
        union = set_a | set_b
        return len(set_a & set_b) / len(union) if union else 1.0

    top1_agree = sum(
        1 for a, b in pairs
        if a.retrieved_ids[:1] == b.retrieved_ids[:1]
    ) / len(pairs)

    top2_jacc = sum(
        _jaccard(set(a.retrieved_ids[:2]), set(b.retrieved_ids[:2]))
        for a, b in pairs
    ) / len(pairs)

    top5_jacc = sum(
        _jaccard(set(a.retrieved_ids[:5]), set(b.retrieved_ids[:5]))
        for a, b in pairs
    ) / len(pairs)

    return {
        "top1_agreement": round(top1_agree, 3),
        "top2_jaccard": round(top2_jacc, 3),
        "top5_jaccard": round(top5_jacc, 3),
    }
