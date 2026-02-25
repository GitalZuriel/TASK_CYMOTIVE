"""
Full evaluation: Dev set vs. Holdout set comparison.

Runs both query sets through the retrieval pipeline and reports:
  - P@1, P@2, Recall@5, MRR, nDCG@2  (dev & holdout)
  - Flexible P@2 for holdout (using acceptable_top2)
  - Paraphrase stability per group
  - Hard-negative performance
  - Generalisation gap

Usage:
    python evaluation/run_eval_full.py
"""

import sys
import os
import math
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

logging.basicConfig(level=logging.WARNING)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from src.embeddings import get_embedding_model
from src.vector_store import VectorStore
from src.bm25_search import BM25Index
from src.hybrid_retrieval import HybridRetriever
from src.reranker import Reranker
from src.preprocessor import preprocess

from evaluation.dev_set import DEV_QUERIES
from evaluation.holdout_set import (
    HOLDOUT_QUERIES,
    HARD_NEGATIVE_QUERIES,
    ALL_HOLDOUT_QUERIES,
    get_all_groups,
    get_group,
)
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mrr_score,
    ndcg_at_k,
    flexible_precision_at_2,
    paraphrase_stability,
    QueryResult,
)

console = Console()

# Retrieve top-5 for Recall@5 (pipeline gets 7+ from RRF, this is safe)
EVAL_TOP_K = 5


# ───────────────────────────────────────────────────────────────────
# Pipeline
# ───────────────────────────────────────────────────────────────────

def init_pipeline():
    """Initialise retrieval + reranking pipeline once."""
    console.print("[bold]Initializing pipeline...[/bold]")
    emb = get_embedding_model("local")
    vs = VectorStore(emb)
    vs.ingest_incidents()
    bm25 = BM25Index()
    bm25.build_index()
    retriever = HybridRetriever(vs, bm25)
    reranker = Reranker()
    console.print("[green]Pipeline ready.[/green]\n")
    return retriever, reranker


def run_query(query_text: str, retriever, reranker) -> list[str]:
    """Run a single query, return top-EVAL_TOP_K incident IDs."""
    preprocessed = preprocess(query_text)
    fused = retriever.retrieve(query_text)
    reranked = reranker.rerank(
        query_text, fused, top_k=EVAL_TOP_K, preprocessed=preprocessed
    )
    return [r.incident_id for r in reranked]


# ───────────────────────────────────────────────────────────────────
# Evaluate dev set
# ───────────────────────────────────────────────────────────────────

def evaluate_dev(retriever, reranker) -> list[QueryResult]:
    """Run the 20 dev queries and compute metrics."""
    results: list[QueryResult] = []
    for q in DEV_QUERIES:
        ids = run_query(q["query"], retriever, reranker)
        expected = q["expected_ids"]
        r = QueryResult(
            query_id=q["id"],
            description=q["description"],
            retrieved_ids=ids,
            expected_top1=expected,       # dev set: both expected are "top1"
            acceptable_top2=expected,      # dev set: same strict set
            p_at_1=precision_at_k(ids, expected, 1),
            p_at_2=precision_at_k(ids, expected, 2),
            recall_at_5=recall_at_k(ids, expected, 5),
            mrr=mrr_score(ids, expected),
            ndcg_at_2=ndcg_at_k(ids, expected, 2),
        )
        results.append(r)
    return results


# ───────────────────────────────────────────────────────────────────
# Evaluate holdout set
# ───────────────────────────────────────────────────────────────────

def evaluate_holdout(retriever, reranker) -> list[QueryResult]:
    """Run all holdout queries (paraphrases + hard negatives)."""
    results: list[QueryResult] = []
    for q in ALL_HOLDOUT_QUERIES:
        ids = run_query(q.query, retriever, reranker)

        # For P@1 — strict: first result must be in expected_top1
        p1 = precision_at_k(ids, q.expected_top1, 1)

        # For P@2 — flexible: slot 2 scored against acceptable_top2
        p2 = flexible_precision_at_2(ids, q.expected_top1, q.acceptable_top2)

        # All expected for recall/MRR/nDCG = expected_top1 + acceptable_top2
        all_relevant = list(set(q.expected_top1 + q.acceptable_top2))

        r = QueryResult(
            query_id=q.id,
            description=q.description,
            retrieved_ids=ids,
            expected_top1=q.expected_top1,
            acceptable_top2=q.acceptable_top2,
            p_at_1=p1,
            p_at_2=p2,
            recall_at_5=recall_at_k(ids, all_relevant, 5),
            mrr=mrr_score(ids, all_relevant),
            ndcg_at_2=ndcg_at_k(ids, all_relevant, 2),
            group=q.group,
            variant=q.variant,
            noise_level=q.noise_level,
        )
        results.append(r)
    return results


# ───────────────────────────────────────────────────────────────────
# Aggregation helpers
# ───────────────────────────────────────────────────────────────────

def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(results: list[QueryResult]) -> dict[str, float]:
    return {
        "P@1": avg([r.p_at_1 for r in results]),
        "P@2": avg([r.p_at_2 for r in results]),
        "R@5": avg([r.recall_at_5 for r in results]),
        "MRR": avg([r.mrr for r in results]),
        "nDCG@2": avg([r.ndcg_at_2 for r in results]),
    }


# ───────────────────────────────────────────────────────────────────
# Printing
# ───────────────────────────────────────────────────────────────────

def print_aggregate_table(dev_agg: dict, hold_agg: dict,
                          hold_para_agg: dict, hold_hn_agg: dict):
    """Main comparison table: dev vs holdout + gap."""
    table = Table(
        title="Dev Set vs. Holdout Set — Aggregate Metrics",
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Metric", style="bold", min_width=14)
    table.add_column("Dev (20)", justify="center", min_width=9)
    table.add_column("Holdout (33)", justify="center", min_width=11)
    table.add_column("Para (30)", justify="center", min_width=9)
    table.add_column("Hard Neg (3)", justify="center", min_width=10)
    table.add_column("Gap", justify="center", min_width=8)

    for metric in ["P@1", "P@2", "R@5", "MRR", "nDCG@2"]:
        d = dev_agg[metric]
        h = hold_agg[metric]
        gap = h - d
        gap_style = "red" if gap < -0.05 else "green" if gap > 0.05 else "white"
        table.add_row(
            f"Avg {metric}",
            f"{d:.3f}",
            f"{h:.3f}",
            f"{hold_para_agg[metric]:.3f}",
            f"{hold_hn_agg[metric]:.3f}",
            Text(f"{gap:+.3f}", style=gap_style),
        )
    console.print(table)


def print_per_query_table(results: list[QueryResult], title: str):
    """Per-query detail table."""
    table = Table(title=title, border_style="green", show_lines=False)
    table.add_column("ID", style="bold", min_width=10)
    table.add_column("Description", min_width=28)
    table.add_column("P@1", justify="center", min_width=5)
    table.add_column("P@2", justify="center", min_width=5)
    table.add_column("R@5", justify="center", min_width=5)
    table.add_column("MRR", justify="center", min_width=5)
    table.add_column("nDCG", justify="center", min_width=5)
    table.add_column("Retrieved", min_width=22)
    table.add_column("Expected", min_width=18)

    for r in results:
        p1_style = "green" if r.p_at_1 >= 1.0 else "red"
        table.add_row(
            r.query_id,
            r.description[:35],
            Text(f"{r.p_at_1:.2f}", style=p1_style),
            f"{r.p_at_2:.2f}",
            f"{r.recall_at_5:.2f}",
            f"{r.mrr:.2f}",
            f"{r.ndcg_at_2:.2f}",
            ", ".join(r.retrieved_ids[:5]),
            ", ".join(r.expected_top1),
        )
    console.print(table)


def print_paraphrase_table(holdout_results: list[QueryResult]):
    """Paraphrase stability per group."""
    table = Table(
        title="Paraphrase Stability Report",
        border_style="magenta",
        show_lines=True,
    )
    table.add_column("Group", style="bold", min_width=6)
    table.add_column("Description", min_width=28)
    table.add_column("Top1 Agree", justify="center", min_width=10)
    table.add_column("Top2 Jaccard", justify="center", min_width=11)
    table.add_column("Top5 Jaccard", justify="center", min_width=11)

    all_stabilities = []
    for group_id in get_all_groups():
        group_qrs = [r for r in holdout_results if r.group == group_id]
        if len(group_qrs) < 2:
            continue
        stab = paraphrase_stability(group_qrs)
        all_stabilities.append(stab)

        desc = group_qrs[0].description.split("(")[0].strip()
        t1_style = "green" if stab["top1_agreement"] >= 0.67 else "red"
        table.add_row(
            group_id,
            desc[:35],
            Text(f"{stab['top1_agreement']:.2f}", style=t1_style),
            f"{stab['top2_jaccard']:.2f}",
            f"{stab['top5_jaccard']:.2f}",
        )

    # Average row
    if all_stabilities:
        avg_t1 = avg([s["top1_agreement"] for s in all_stabilities])
        avg_t2 = avg([s["top2_jaccard"] for s in all_stabilities])
        avg_t5 = avg([s["top5_jaccard"] for s in all_stabilities])
        t1_style = "green bold" if avg_t1 >= 0.70 else "red bold"
        table.add_row(
            "AVG", "Overall",
            Text(f"{avg_t1:.2f}", style=t1_style),
            f"{avg_t2:.2f}",
            f"{avg_t5:.2f}",
        )
    console.print(table)


def print_noise_breakdown(holdout_results: list[QueryResult]):
    """Breakdown by noise level (clean / moderate / noisy)."""
    table = Table(
        title="Performance by Noise Level",
        border_style="yellow",
        show_lines=True,
    )
    table.add_column("Noise Level", style="bold", min_width=12)
    table.add_column("Count", justify="center", min_width=6)
    table.add_column("Avg P@1", justify="center", min_width=8)
    table.add_column("Avg P@2", justify="center", min_width=8)
    table.add_column("Avg R@5", justify="center", min_width=8)

    for level in ["clean", "moderate", "noisy"]:
        subset = [r for r in holdout_results if r.noise_level == level]
        if not subset:
            continue
        table.add_row(
            level,
            str(len(subset)),
            f"{avg([r.p_at_1 for r in subset]):.3f}",
            f"{avg([r.p_at_2 for r in subset]):.3f}",
            f"{avg([r.recall_at_5 for r in subset]):.3f}",
        )
    console.print(table)


def print_hard_negative_detail(holdout_results: list[QueryResult]):
    """Detail for hard-negative queries."""
    hn_results = [r for r in holdout_results if r.group == "HN"]
    if not hn_results:
        return

    table = Table(
        title="Hard Negative Queries — Detail",
        border_style="red",
        show_lines=True,
    )
    table.add_column("ID", style="bold", min_width=5)
    table.add_column("Description", min_width=38)
    table.add_column("P@1", justify="center", min_width=5)
    table.add_column("P@2", justify="center", min_width=5)
    table.add_column("Retrieved Top-5", min_width=28)
    table.add_column("Expected #1", min_width=10)

    for r in hn_results:
        p1_style = "green" if r.p_at_1 >= 1.0 else "red bold"
        table.add_row(
            r.query_id,
            r.description[:45],
            Text(f"{r.p_at_1:.2f}", style=p1_style),
            f"{r.p_at_2:.2f}",
            ", ".join(r.retrieved_ids[:5]),
            ", ".join(r.expected_top1),
        )
    console.print(table)


def print_verdict(dev_agg: dict, hold_agg: dict):
    """Final verdict: does the pipeline generalise?"""
    gaps = {m: hold_agg[m] - dev_agg[m] for m in dev_agg}
    bad_gaps = [m for m, g in gaps.items() if g < -0.10]

    if bad_gaps:
        console.print(Panel(
            f"[red bold]OVERFITTING WARNING[/red bold]\n"
            f"Metrics with >10% gap: {', '.join(bad_gaps)}\n"
            f"The pipeline may be overfit to the dev set.",
            border_style="red",
        ))
    elif any(g < -0.05 for g in gaps.values()):
        console.print(Panel(
            f"[yellow bold]MODERATE GAP[/yellow bold]\n"
            f"Some metrics show 5-10% gap between dev and holdout.\n"
            f"Monitor carefully — may indicate mild overfitting.",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            f"[green bold]GENERALISATION CONFIRMED[/green bold]\n"
            f"Holdout performance is within 5% of dev set.\n"
            f"Pipeline improvements genuinely generalise.",
            border_style="green",
        ))


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold]Full Evaluation: Dev Set vs. Holdout Set[/bold]\n"
        "P@1 · P@2 · Recall@5 · MRR · nDCG@2 · Paraphrase Stability",
        border_style="cyan",
    ))

    retriever, reranker = init_pipeline()

    # ── Evaluate ──
    t0 = time.perf_counter()
    dev_results = evaluate_dev(retriever, reranker)
    t_dev = time.perf_counter() - t0

    t0 = time.perf_counter()
    holdout_results = evaluate_holdout(retriever, reranker)
    t_hold = time.perf_counter() - t0

    # ── Aggregate ──
    dev_agg = aggregate(dev_results)

    hold_all_agg = aggregate(holdout_results)
    hold_para = [r for r in holdout_results if r.group != "HN"]
    hold_hn = [r for r in holdout_results if r.group == "HN"]
    hold_para_agg = aggregate(hold_para)
    hold_hn_agg = aggregate(hold_hn) if hold_hn else {m: 0.0 for m in dev_agg}

    # ── Print ──
    console.print()
    print_aggregate_table(dev_agg, hold_all_agg, hold_para_agg, hold_hn_agg)

    console.print()
    print_per_query_table(dev_results, "Dev Set — Per-Query Results (20)")

    console.print()
    print_per_query_table(holdout_results, "Holdout Set — Per-Query Results (33)")

    console.print()
    print_paraphrase_table(holdout_results)

    console.print()
    print_noise_breakdown(holdout_results)

    console.print()
    print_hard_negative_detail(holdout_results)

    # ── Timing ──
    console.print(f"\n[dim]Dev eval: {t_dev:.1f}s | Holdout eval: {t_hold:.1f}s[/dim]")

    # ── Verdict ──
    console.print()
    print_verdict(dev_agg, hold_all_agg)


if __name__ == "__main__":
    main()
