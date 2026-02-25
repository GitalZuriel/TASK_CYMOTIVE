"""
Dev-set-only evaluation (20 queries).

Uses shared metrics and dev_set data.  This is the refactored version
of the original run_eval_20.py.
"""

import sys
import os
import math

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

logging.basicConfig(level=logging.WARNING)

from src.embeddings import get_embedding_model
from src.vector_store import VectorStore
from src.bm25_search import BM25Index
from src.hybrid_retrieval import HybridRetriever
from src.reranker import Reranker
from src.preprocessor import preprocess

from evaluation.dev_set import DEV_QUERIES
from evaluation.metrics import precision_at_k, recall_at_k, mrr_score, ndcg_at_k


def main():
    print("Initializing pipeline...")
    emb = get_embedding_model("local")
    vs = VectorStore(emb)
    vs.ingest_incidents()
    bm25 = BM25Index()
    bm25.build_index()
    retriever = HybridRetriever(vs, bm25)
    reranker = Reranker()
    print("Pipeline ready.\n")

    total_p1, total_p, total_r, total_m, total_n = 0, 0, 0, 0, 0
    passes = 0
    fails = []

    for i, tq in enumerate(DEV_QUERIES, 1):
        preprocessed = preprocess(tq["query"])
        fused = retriever.retrieve(tq["query"])
        reranked = reranker.rerank(
            tq["query"], fused, top_k=5, preprocessed=preprocessed
        )

        retrieved_ids = [r.incident_id for r in reranked]
        expected = tq["expected_ids"]

        p1 = precision_at_k(retrieved_ids, expected, 1)
        p = precision_at_k(retrieved_ids, expected, 2)
        r = recall_at_k(retrieved_ids, expected, 5)
        m = mrr_score(retrieved_ids, expected)
        n = ndcg_at_k(retrieved_ids, expected, 2)

        total_p1 += p1
        total_p += p
        total_r += r
        total_m += m
        total_n += n

        match = set(retrieved_ids[:2]) == set(expected)
        status = "PASS" if match else "FAIL"
        if match:
            passes += 1
        else:
            fails.append(i)

        print(
            f"{i:2d}. [{status}] {tq['description']:30s} | "
            f"P@1={p1:.2f} P@2={p:.2f} R@5={r:.2f} MRR={m:.2f} nDCG={n:.2f} | "
            f"Got={retrieved_ids[:5]} Expected={expected}"
        )

        if not match:
            for r_item in reranked[:3]:
                print(
                    f"      {r_item.incident_id}: raw={r_item.raw_score:.4f} norm={r_item.normalized_score:.1f} "
                    f"CE_logit={r_item.cross_encoder_score:.2f} domain={r_item.domain_score:.4f}"
                )

    n_queries = len(DEV_QUERIES)
    print(f"\n{'=' * 100}")
    print(
        f"RESULTS: {passes}/{n_queries} queries passed "
        f"({passes / n_queries * 100:.0f}%)"
    )
    print(
        f"Avg P@1={total_p1 / n_queries:.2f}  "
        f"Avg P@2={total_p / n_queries:.2f}  "
        f"Avg R@5={total_r / n_queries:.2f}  "
        f"Avg MRR={total_m / n_queries:.2f}  "
        f"Avg nDCG@2={total_n / n_queries:.2f}"
    )
    if fails:
        print(f"Failed queries: {fails}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
