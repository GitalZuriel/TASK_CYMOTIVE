"""
Cybersecurity Copilot CLI — interactive interface for automotive incident analysis.
"""

import sys
import os
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

import config
from src.copilot import CybersecurityCopilot

# --- Logging setup ---
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

console = Console()


def print_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]Cybersecurity Copilot — Automotive Incident Analysis PoC[/bold cyan]\n"
        "Hybrid RAG | Cross-Encoder Reranking | Multi-Model LLM",
        border_style="cyan",
    ))


def print_response(response) -> None:
    """Pretty-print a CopilotResponse."""

    # Clarification (for invalid/short input)
    if response.clarification and not response.summary:
        console.print(Panel(
            Markdown(response.clarification),
            title="Clarification Needed",
            border_style="yellow",
        ))
        return

    # Summary
    if response.summary:
        console.print(Panel(
            Markdown(response.summary),
            title="Incident Summary",
            border_style="green",
        ))

    # Mitigation
    if response.mitigation:
        console.print(Panel(
            Markdown(response.mitigation),
            title="Mitigation Plan",
            border_style="blue",
        ))

    # Similar Incidents
    if response.similar_incidents:
        table = Table(title="Similar Past Incidents", border_style="magenta")
        table.add_column("ID", style="bold")
        table.add_column("Title")
        table.add_column("Severity")
        table.add_column("Relevance", justify="right")
        table.add_column("Source")

        for inc in response.similar_incidents:
            table.add_row(
                inc.incident_id,
                inc.title,
                inc.metadata.get("severity", "N/A"),
                f"{inc.normalized_score:.1f}%",
                inc.source,
            )
        console.print(table)

    # Clarification notes (for valid but incomplete input)
    if response.clarification and response.summary:
        console.print(Panel(
            Markdown(response.clarification),
            title="Additional Information Suggested",
            border_style="yellow",
        ))

    # Stats
    total_tokens = sum(
        s.get("tokens", 0) for s in response.llm_stats.values()
    )
    total_cost = sum(
        s.get("cost", 0) for s in response.llm_stats.values()
    )
    console.print(
        f"\n[dim]Stats: latency={response.total_latency_seconds}s | "
        f"tokens={total_tokens} | estimated_cost=${total_cost:.4f}[/dim]"
    )


def main() -> None:
    print_banner()

    llm_model = config.DEFAULT_LLM
    embedding_model = config.DEFAULT_EMBEDDING

    console.print(f"[dim]Initializing with LLM={llm_model}, Embeddings={embedding_model}...[/dim]")

    try:
        copilot = CybersecurityCopilot(
            llm_model=llm_model,
            embedding_model=embedding_model,
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize copilot: {e}[/red]")
        sys.exit(1)

    console.print("[green]Ready.[/green]\n")

    while True:
        try:
            console.print("[bold]Enter incident report[/bold] (or 'quit' to exit):")
            lines: list[str] = []
            console.print("[dim]  (Enter a blank line to submit)[/dim]")
            while True:
                line = input("> ")
                if line == "":
                    break
                lines.append(line)

            text = "\n".join(lines).strip()

            if text.lower() in ("quit", "exit", "q"):
                console.print("[cyan]Goodbye.[/cyan]")
                break

            if not text:
                continue

            console.print("\n[dim]Analyzing...[/dim]\n")
            response = copilot.analyze(text)
            print_response(response)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye.[/cyan]")
            break
        except Exception as e:
            logger.exception("Error during analysis")
            console.print(f"[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()
