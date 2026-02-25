"""
Compare GPT-4o vs GPT-4o-mini vs Claude Sonnet on automotive incident analysis.

Runs the same test incidents through all three LLM models and outputs a
comparison table of response quality, latency, token usage, and cost.
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

import config
from src.embeddings import get_embedding_model
from src.vector_store import VectorStore
from src.bm25_search import BM25Index
from src.hybrid_retrieval import HybridRetriever
from src.reranker import Reranker
from src.prompts import SYSTEM_PROMPT, SUMMARY_PROMPT, format_similar_incidents
from src.llm_chain import LLMChain

logging.basicConfig(level=logging.WARNING)
console = Console()

# Test incidents for comparison
TEST_INCIDENTS = [
    {
        "name": "CAN Bus Attack",
        "text": (
            "Our vehicle IDPS detected anomalous CAN frames on the powertrain bus. "
            "Frames with arbitration ID 0x1A0 are being injected at 10ms intervals, "
            "targeting the transmission control module. The source appears to be the "
            "OBD-II diagnostic port. No authorized diagnostic session is active. "
            "Affected vehicle: 2024 Model X sedan, VIN: WBA12345678901234."
        ),
    },
    {
        "name": "OTA Compromise",
        "text": (
            "VSOC alert: Suspicious firmware update package detected in the OTA pipeline. "
            "The package targets TCU firmware version 3.2.1 and has a valid signature but "
            "the signing timestamp is 48 hours in the future. Hash mismatch detected between "
            "the manifest and the actual binary. Approximately 8,000 vehicles in the EU fleet "
            "are scheduled to receive this update in the next maintenance window."
        ),
    },
    {
        "name": "Telematics Breach",
        "text": (
            "Anomalous outbound traffic detected from the telematics gateway of a connected "
            "SUV fleet. The TCU is establishing TLS connections to IP 45.33.32.156 on port 8443 "
            "every 5 minutes. Data exfiltration suspected — payload analysis shows encoded "
            "vehicle telemetry including GPS, speed, and diagnostic PIDs. The cellular modem "
            "firmware was last updated 2 weeks ago. CVE-2024-8901 may be related."
        ),
    },
]

MODELS = ["gpt-4o-mini", "gpt-4o", "claude-sonnet"]


def main() -> None:
    console.print(Panel.fit(
        "[bold]LLM Model Comparison — Automotive Cybersecurity Analysis[/bold]",
        border_style="cyan",
    ))

    # Setup shared retrieval pipeline (run once)
    console.print("[dim]Setting up retrieval pipeline...[/dim]")
    emb = get_embedding_model(config.DEFAULT_EMBEDDING)
    vs = VectorStore(emb)
    if vs.count == 0:
        vs.ingest_incidents()
    bm25 = BM25Index()
    bm25.build_index()
    retriever = HybridRetriever(vs, bm25)
    reranker = Reranker()
    console.print("[green]Retrieval pipeline ready.[/green]\n")

    for test in TEST_INCIDENTS:
        console.print(Panel(test["text"], title=f"Test: {test['name']}", border_style="yellow"))

        # Retrieve similar incidents (shared across models)
        fused = retriever.retrieve(test["text"])
        reranked = reranker.rerank(test["text"], fused)
        similar_text = format_similar_incidents([
            {"title": r.title, "description": r.description, "metadata": r.metadata}
            for r in reranked
        ])

        prompt = SUMMARY_PROMPT.format(
            similar_incidents=similar_text,
            incident_report=test["text"],
        )

        # Compare table
        table = Table(title=f"Results: {test['name']}", border_style="green")
        table.add_column("Model", style="bold")
        table.add_column("Latency (s)", justify="right")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Total Tokens", justify="right")
        table.add_column("Est. Cost ($)", justify="right")

        for model in MODELS:
            try:
                llm = LLMChain(model)
                resp = llm.invoke(SYSTEM_PROMPT, prompt)
                table.add_row(
                    model,
                    f"{resp.latency_seconds:.2f}",
                    str(resp.input_tokens),
                    str(resp.output_tokens),
                    str(resp.total_tokens),
                    f"${resp.estimated_cost_usd:.4f}",
                )
                console.print(Panel(
                    Markdown(resp.content[:500] + "..." if len(resp.content) > 500 else resp.content),
                    title=f"{model} response (truncated)",
                    border_style="dim",
                ))
            except Exception as e:
                table.add_row(model, "ERR", "-", "-", "-", "-")
                console.print(f"  [red]{model} error: {e}[/red]")

        console.print(table)
        console.print()


if __name__ == "__main__":
    main()
