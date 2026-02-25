"""
BM25 keyword search for automotive cybersecurity incidents.

Uses rank-bm25 to index incident documents and perform keyword-based retrieval.
Custom tokenization handles technical terms, CVE IDs, and acronyms.
"""

import json
import re
import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)


@dataclass
class BM25SearchResult:
    """A single BM25 search result."""
    incident_id: str
    title: str
    description: str
    score: float
    metadata: dict


def _tokenize(text: str) -> list[str]:
    """
    Custom tokenizer for automotive cybersecurity text.

    Preserves technical terms, CVE IDs, IP addresses, and common acronyms
    while normalizing for BM25 matching.
    """
    text = text.lower()

    # Preserve CVE IDs as single tokens (e.g., "cve-2024-1234")
    cve_ids = re.findall(r"cve-\d{4}-\d{4,7}", text)

    # Preserve IP addresses as single tokens
    ip_addrs = re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text)

    # Replace preserved tokens with placeholders to avoid splitting
    preserved: dict[str, str] = {}
    for i, token in enumerate(cve_ids + ip_addrs):
        placeholder = f"__PRESERVED_{i}__"
        preserved[placeholder] = token
        text = text.replace(token, placeholder, 1)

    # Split on non-alphanumeric characters (keep underscores for placeholders)
    raw_tokens = re.findall(r"[a-z0-9_]+", text)

    # Restore preserved tokens and filter very short tokens
    tokens: list[str] = []
    for t in raw_tokens:
        if t in preserved:
            tokens.append(preserved[t])
        elif len(t) >= 2:  # drop single chars but keep abbreviations like "v2x"
            tokens.append(t)

    return tokens


class BM25Index:
    """BM25 index over automotive cybersecurity incidents."""

    def __init__(self) -> None:
        self._incidents: list[dict] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build_index(self, incidents_path: str | None = None) -> int:
        """
        Load incidents from JSON and build the BM25 index.

        Args:
            incidents_path: Path to incidents JSON file. Defaults to config value.

        Returns:
            Number of documents indexed.
        """
        path = incidents_path or config.INCIDENTS_PATH
        with open(path, "r", encoding="utf-8") as f:
            self._incidents = json.load(f)

        self._corpus = []
        for inc in self._incidents:
            # Combine all searchable fields for richer keyword matching
            doc_text = (
                f"{inc['title']} {inc['description']} "
                f"{inc['attack_vector']} {inc['affected_system']} "
                f"{inc.get('cve', '')}"
            )
            self._corpus.append(_tokenize(doc_text))

        self._bm25 = BM25Okapi(self._corpus)
        logger.info("BM25 index built with %d documents", len(self._incidents))
        return len(self._incidents)

    def search(self, query: str, top_k: int | None = None) -> list[BM25SearchResult]:
        """
        Search the BM25 index.

        Args:
            query: Search query text.
            top_k: Number of results to return. Defaults to config.TOP_K_RETRIEVAL.

        Returns:
            List of BM25SearchResult sorted by BM25 score (highest first).
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        top_k = top_k or config.TOP_K_RETRIEVAL
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score descending
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results: list[BM25SearchResult] = []
        for idx, score in scored_indices:
            if score <= 0:
                continue  # skip zero-score results
            inc = self._incidents[idx]
            results.append(BM25SearchResult(
                incident_id=inc["id"],
                title=inc["title"],
                description=f"{inc['title']}. {inc['description']}",
                score=float(score),
                metadata={
                    "title": inc["title"],
                    "date": inc["date"],
                    "severity": inc["severity"],
                    "severity_score": inc.get("severity_score", 0),
                    "attack_vector": inc["attack_vector"],
                    "affected_system": inc["affected_system"],
                    "cve": inc.get("cve") or "",
                    "mitre_tactics": ",".join(inc.get("mitre_tactics", [])),
                    "protocols": ",".join(
                        inc.get("indicators", {}).get("protocols", [])
                    ),
                },
            ))

        logger.info("BM25 search returned %d results for query", len(results))
        return results
