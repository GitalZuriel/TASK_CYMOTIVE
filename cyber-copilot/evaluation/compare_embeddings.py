"""
Compare OpenAI text-embedding-3-small vs all-MiniLM-L6-v2 embeddings.

Evaluates retrieval quality (precision@2) and latency for both models
on the same set of test queries.
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config
from src.embeddings import get_embedding_model
from src.vector_store import VectorStore
from src.bm25_search import BM25Index
from src.hybrid_retrieval import HybridRetriever
from src.reranker import Reranker
from src.preprocessor import preprocess

logging.basicConfig(level=logging.WARNING)
console = Console()

# Test queries with expected relevant incident IDs (ground truth)
# Written as messy, realistic analyst input — not polished structured text
TEST_QUERIES = [
    {
        "query": "fleet ops flagged weird can traffic on the powertrain bus.. someone plugged something into the obd port and we're seeing injected frames hitting the brake ecu, arb ID 0x130 every 10ms. no diagnostic session active. multiple vehicles affected drivers reporting brakes acting up at highway speed",
        "expected_ids": ["INC-002", "INC-006"],
        "description": "CAN bus injection",
    },
    {
        "query": "found during supplier audit - central gateway ECU has unsigned firmware running. secure boot looks bypassed somehow.. internal firewall rules are gone. supplier says they didnt push any update, someone tampered with the gateway firmware. possible physical access",
        "expected_ids": ["INC-004", "INC-015"],
        "description": "Gateway firmware tampering",
    },
    {
        "query": "VSOC alert - weird outbound connections from ~200 EVs right after last OTA push went out. turns out the firmware signing server was compromised and malicious code got pushed to fleet. CI/CD pipeline breach confirmed, signatures looked valid. need immediate rollback",
        "expected_ids": ["INC-003", "INC-015"],
        "description": "OTA supply chain compromise",
    },
    {
        "query": "seeing tons of abnormal DNS TXT queries from infotainment units. encoded payloads carrying VIN numbers and GPS coords being tunneled out to external server. classic DNS exfiltration pattern, TCU involved too",
        "expected_ids": ["INC-001", "INC-010"],
        "description": "DNS data exfiltration",
    },
    {
        "query": "3 vehicles stolen overnight same parking garage. security cameras show two guys with devices - one near the house one near the car. classic relay attack extending key fob signal to unlock and start wirelessly",
        "expected_ids": ["INC-009", "INC-013"],
        "description": "Keyless relay theft",
    },
    {
        "query": "security researcher demo'd unlocking our cars by relaying BLE pairing between owners phone and BCM. bluetooth low energy digital key has a relay vuln, demonstrated at 50m range. affects all models with phone-as-key feature",
        "expected_ids": ["INC-007", "INC-009"],
        "description": "BLE digital key attack",
    },
    {
        "query": "autonomous shuttle went completely off route today!! GNSS module receiving spoofed satellite signals from portable SDR nearby. navigation system has no integrity check at all, shuttle nearly drove into oncoming traffic",
        "expected_ids": ["INC-008", "INC-005"],
        "description": "GPS / GNSS spoofing",
    },
    {
        "query": "got hit via crafted SMS on TCU. text message triggered buffer overflow in qualcomm baseband modem firmware.. attacker got remote shell on telematics unit and started pivoting into vehicle network. 12 vehicles confirmed compromised",
        "expected_ids": ["INC-010", "INC-014"],
        "description": "Cellular modem exploit",
    },
    {
        "query": "caught a rogue cell tower IMSI catcher style near test facility. exploiting vulns in LTE baseband chipset to intercept vehicle telemetry. also tracking individual vehicles via cellular modem fingerprinting",
        "expected_ids": ["INC-014", "INC-010"],
        "description": "Baseband / rogue cell tower",
    },
    {
        "query": "found rogue device in charging station doing MITM on ISO 15118 PLC communication. intercepting Plug&Charge certs and cloning auth tokens. V2G protocol completely compromised at this station, multiple EVs affected",
        "expected_ids": ["INC-012", "INC-003"],
        "description": "EV charging MITM",
    },
    {
        "query": "cameras on 3 test vehicles all misclassify same stop sign as speed limit 80.. someone stuck adversarial stickers on it. ADAS traffic sign recognition completely fooled, affects forward camera ML pipeline",
        "expected_ids": ["INC-011", "INC-008"],
        "description": "ADAS adversarial patch",
    },
    {
        "query": "reverse engineered key fob RF protocol - the rolling code PRNG is weak. can predict next 100 unlock codes after capturing just 2 transmissions. affects all vehicles with this RKE module, keyless entry completely broken",
        "expected_ids": ["INC-013", "INC-009"],
        "description": "Rolling code weakness",
    },
    {
        "query": "the fleet management OBD dongles have open API with zero authentication. anyone on network can send raw CAN frames through them remotely.. tested it ourselves and injected messages on powertrain bus. 500 vehicles have these installed",
        "expected_ids": ["INC-006", "INC-002"],
        "description": "OBD fleet dongle exploit",
    },
    {
        "query": "attacker used UDS diagnostic services RequestDownload 0x34 to rollback ECM firmware to version with known vulns. bypassed firmware version check via doip, downgraded security patches. ECU running year-old software now",
        "expected_ids": ["INC-015", "INC-004"],
        "description": "Diagnostic rollback attack",
    },
    {
        "query": "smart intersection V2X broadcasting fake emergency vehicle warnings and road hazard alerts.. triggered automatic braking on 5 vehicles. V2I messages had valid format but came from unauthorized transmitter, no PKI cert validation",
        "expected_ids": ["INC-005", "INC-008"],
        "description": "V2X spoofing",
    },
    {
        "query": "third party nav app on IVI is leaking data. caught it doing covert DNS tunneling to exfiltrate location history and CAN diagnostic PIDs. app was sideloaded, no sandboxing on infotainment linux platform",
        "expected_ids": ["INC-001", "INC-006"],
        "description": "Infotainment app compromise",
    },
    {
        "query": "bus off condition on powertrain CAN!! something flooding high-priority frames at max bus speed. ABS and transmission modules went safe mode. looks like DoS on the CAN bus, possibly through OBD port or compromised ECU",
        "expected_ids": ["INC-002", "INC-006"],
        "description": "CAN bus flooding / DoS",
    },
    {
        "query": "central gateway lost network segmentation.. infotainment CAN traffic leaking into safety-critical powertrain domain. no filtering between bus segments. either firmware bug or gateway config tampered. TCU also showing anomalous outbound traffic",
        "expected_ids": ["INC-004", "INC-010"],
        "description": "Gateway network bridging",
    },
    {
        "query": "unauthorized RF beacon planted on vehicles transmitting location on LTE. traced to cellular modem vulnerability that lets external parties query TCU location without auth. vehicle tracking at scale, fleet affected",
        "expected_ids": ["INC-014", "INC-010"],
        "description": "Wireless vehicle tracking",
    },
    {
        "query": "during fast charging session EVSE pushed malicious firmware update to onboard charger via ISO 15118 PLC. charger behavior changed - drawing more current than rated. communication was intercepted and modified at charging station",
        "expected_ids": ["INC-012", "INC-003"],
        "description": "Charging firmware injection",
    },
]

EMBEDDING_PROVIDERS = ["openai", "local"]


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 2) -> float:
    """
    Precision@k: fraction of top-k results that are relevant.

    "Out of the k documents I retrieved, how many are actually relevant?"
    High precision = few false positives (don't recommend junk).
    """
    top_k = retrieved_ids[:k]
    relevant = sum(1 for rid in top_k if rid in expected_ids)
    return relevant / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 2) -> float:
    """
    Recall@k: fraction of all relevant documents found in top-k.

    "Out of all the relevant documents that exist, how many did I find?"
    High recall = we didn't miss relevant incidents.
    """
    if not expected_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for eid in expected_ids if eid in top_k)
    return found / len(expected_ids)


def mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """
    Mean Reciprocal Rank: 1 / position of the first relevant result.

    "How quickly did the first relevant document appear?"
    MRR=1.0 means the top result is relevant. MRR=0.5 means it was 2nd.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 2) -> float:
    """
    Normalized Discounted Cumulative Gain @ k.

    Measures ranking quality: relevant documents ranked higher score better.
    Uses binary relevance (1 if relevant, 0 if not).

    DCG@k  = sum( rel_i / log2(i+1) )  for i in 1..k
    IDCG@k = best possible DCG@k (all relevant docs at the top)
    nDCG@k = DCG@k / IDCG@k
    """
    import math

    top_k = retrieved_ids[:k]

    # DCG: actual score based on where relevant docs appear
    dcg = 0.0
    for i, rid in enumerate(top_k):
        rel = 1.0 if rid in expected_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # IDCG: ideal score if all relevant docs were at the top
    ideal_relevant = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    console.print(Panel.fit(
        "[bold]Embedding Model Comparison — Retrieval Quality & Latency[/bold]",
        border_style="cyan",
    ))

    # Shared BM25 index
    bm25 = BM25Index()
    bm25.build_index()
    reranker = Reranker()

    # Results accumulator
    results: dict[str, dict] = {
        p: {"p_at_k": [], "r_at_k": [], "mrr": [], "ndcg_at_k": [], "latencies": []}
        for p in EMBEDDING_PROVIDERS
    }

    for provider in EMBEDDING_PROVIDERS:
        console.print(f"\n[bold]Testing: {provider}[/bold]")

        emb = get_embedding_model(provider)
        vs = VectorStore(emb, persist_dir=f"./chroma_db_{provider}_eval")
        vs.ingest_incidents()
        retriever = HybridRetriever(vs, bm25)

        for tq in TEST_QUERIES:
            start = time.perf_counter()
            preprocessed = preprocess(tq["query"])
            fused = retriever.retrieve(tq["query"])
            reranked = reranker.rerank(
                tq["query"], fused, top_k=2, preprocessed=preprocessed,
            )
            latency = time.perf_counter() - start

            retrieved_ids = [r.incident_id for r in reranked]
            k = 2
            p = precision_at_k(retrieved_ids, tq["expected_ids"], k=k)
            r = recall_at_k(retrieved_ids, tq["expected_ids"], k=k)
            m = mrr(retrieved_ids, tq["expected_ids"])
            n = ndcg_at_k(retrieved_ids, tq["expected_ids"], k=k)

            results[provider]["p_at_k"].append(p)
            results[provider]["r_at_k"].append(r)
            results[provider]["mrr"].append(m)
            results[provider]["ndcg_at_k"].append(n)
            results[provider]["latencies"].append(latency)

            console.print(
                f"  {tq['description']}: P@{k}={p:.2f} R@{k}={r:.2f} "
                f"MRR={m:.2f} nDCG@{k}={n:.2f} | "
                f"Retrieved={retrieved_ids} | Expected={tq['expected_ids']} | "
                f"Latency={latency:.3f}s"
            )

    # Summary table
    console.print()
    table = Table(title="Embedding Comparison Summary", border_style="green")
    table.add_column("Metric", style="bold")
    for p in EMBEDDING_PROVIDERS:
        table.add_column(p, justify="center")

    def avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    # Average metrics
    table.add_row(
        "Avg Precision@2",
        *[f"{avg(results[p]['p_at_k']):.2f}" for p in EMBEDDING_PROVIDERS],
    )
    table.add_row(
        "Avg Recall@2",
        *[f"{avg(results[p]['r_at_k']):.2f}" for p in EMBEDDING_PROVIDERS],
    )
    table.add_row(
        "Avg MRR",
        *[f"{avg(results[p]['mrr']):.2f}" for p in EMBEDDING_PROVIDERS],
    )
    table.add_row(
        "Avg nDCG@2",
        *[f"{avg(results[p]['ndcg_at_k']):.2f}" for p in EMBEDDING_PROVIDERS],
    )
    table.add_row(
        "Avg Latency (s)",
        *[f"{avg(results[p]['latencies']):.3f}" for p in EMBEDDING_PROVIDERS],
    )

    # Per-query breakdown
    for i, tq in enumerate(TEST_QUERIES):
        table.add_row(
            f"  {tq['description']}",
            *[
                f"P={results[p]['p_at_k'][i]:.2f} R={results[p]['r_at_k'][i]:.2f} "
                f"MRR={results[p]['mrr'][i]:.2f} nDCG={results[p]['ndcg_at_k'][i]:.2f}"
                for p in EMBEDDING_PROVIDERS
            ],
        )

    console.print(table)


if __name__ == "__main__":
    main()
