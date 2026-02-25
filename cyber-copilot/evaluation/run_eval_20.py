"""Quick evaluation of all 20 test queries with domain-aware reranking."""

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

TEST_QUERIES = [
    {"query": "fleet ops flagged weird can traffic on the powertrain bus.. someone plugged something into the obd port and we're seeing injected frames hitting the brake ecu, arb ID 0x130 every 10ms. no diagnostic session active. multiple vehicles affected drivers reporting brakes acting up at highway speed", "expected_ids": ["INC-002", "INC-006"], "description": "CAN bus injection"},
    {"query": "found during supplier audit - central gateway ECU has unsigned firmware running. secure boot looks bypassed somehow.. internal firewall rules are gone. supplier says they didnt push any update, someone tampered with the gateway firmware. possible physical access", "expected_ids": ["INC-004", "INC-015"], "description": "Gateway firmware tampering"},
    {"query": "VSOC alert - weird outbound connections from ~200 EVs right after last OTA push went out. turns out the firmware signing server was compromised and malicious code got pushed to fleet. CI/CD pipeline breach confirmed, signatures looked valid. need immediate rollback", "expected_ids": ["INC-003", "INC-015"], "description": "OTA supply chain compromise"},
    {"query": "seeing tons of abnormal DNS TXT queries from infotainment units. encoded payloads carrying VIN numbers and GPS coords being tunneled out to external server. classic DNS exfiltration pattern, TCU involved too", "expected_ids": ["INC-001", "INC-010"], "description": "DNS data exfiltration"},
    {"query": "3 vehicles stolen overnight same parking garage. security cameras show two guys with devices - one near the house one near the car. classic relay attack extending key fob signal to unlock and start wirelessly", "expected_ids": ["INC-009", "INC-013"], "description": "Keyless relay theft"},
    {"query": "security researcher demo'd unlocking our cars by relaying BLE pairing between owners phone and BCM. bluetooth low energy digital key has a relay vuln, demonstrated at 50m range. affects all models with phone-as-key feature", "expected_ids": ["INC-007", "INC-009"], "description": "BLE digital key attack"},
    {"query": "autonomous shuttle went completely off route today!! GNSS module receiving spoofed satellite signals from portable SDR nearby. navigation system has no integrity check at all, shuttle nearly drove into oncoming traffic", "expected_ids": ["INC-008", "INC-005"], "description": "GPS / GNSS spoofing"},
    {"query": "got hit via crafted SMS on TCU. text message triggered buffer overflow in qualcomm baseband modem firmware.. attacker got remote shell on telematics unit and started pivoting into vehicle network. 12 vehicles confirmed compromised", "expected_ids": ["INC-010", "INC-014"], "description": "Cellular modem exploit"},
    {"query": "caught a rogue cell tower IMSI catcher style near test facility. exploiting vulns in LTE baseband chipset to intercept vehicle telemetry. also tracking individual vehicles via cellular modem fingerprinting", "expected_ids": ["INC-014", "INC-010"], "description": "Baseband / rogue cell tower"},
    {"query": "found rogue device in charging station doing MITM on ISO 15118 PLC communication. intercepting Plug and Charge certs and cloning auth tokens. V2G protocol completely compromised at this station, multiple EVs affected", "expected_ids": ["INC-012", "INC-003"], "description": "EV charging MITM"},
    {"query": "cameras on 3 test vehicles all misclassify same stop sign as speed limit 80.. someone stuck adversarial stickers on it. ADAS traffic sign recognition completely fooled, affects forward camera ML pipeline", "expected_ids": ["INC-011", "INC-008"], "description": "ADAS adversarial patch"},
    {"query": "reverse engineered key fob RF protocol - the rolling code PRNG is weak. can predict next 100 unlock codes after capturing just 2 transmissions. affects all vehicles with this RKE module, keyless entry completely broken", "expected_ids": ["INC-013", "INC-009"], "description": "Rolling code weakness"},
    {"query": "the fleet management OBD dongles have open API with zero authentication. anyone on network can send raw CAN frames through them remotely.. tested it ourselves and injected messages on powertrain bus. 500 vehicles have these installed", "expected_ids": ["INC-006", "INC-002"], "description": "OBD fleet dongle exploit"},
    {"query": "attacker used UDS diagnostic services RequestDownload 0x34 to rollback ECM firmware to version with known vulns. bypassed firmware version check via doip, downgraded security patches. ECU running year-old software now", "expected_ids": ["INC-015", "INC-004"], "description": "Diagnostic rollback attack"},
    {"query": "smart intersection V2X broadcasting fake emergency vehicle warnings and road hazard alerts.. triggered automatic braking on 5 vehicles. V2I messages had valid format but came from unauthorized transmitter, no PKI cert validation", "expected_ids": ["INC-005", "INC-008"], "description": "V2X spoofing"},
    {"query": "third party nav app on IVI is leaking data. caught it doing covert DNS tunneling to exfiltrate location history and CAN diagnostic PIDs. app was sideloaded, no sandboxing on infotainment linux platform", "expected_ids": ["INC-001", "INC-006"], "description": "Infotainment app compromise"},
    {"query": "bus off condition on powertrain CAN!! something flooding high-priority frames at max bus speed. ABS and transmission modules went safe mode. looks like DoS on the CAN bus, possibly through OBD port or compromised ECU", "expected_ids": ["INC-002", "INC-006"], "description": "CAN bus flooding / DoS"},
    {"query": "central gateway lost network segmentation.. infotainment CAN traffic leaking into safety-critical powertrain domain. no filtering between bus segments. either firmware bug or gateway config tampered. TCU also showing anomalous outbound traffic", "expected_ids": ["INC-004", "INC-010"], "description": "Gateway network bridging"},
    {"query": "unauthorized RF beacon planted on vehicles transmitting location on LTE. traced to cellular modem vulnerability that lets external parties query TCU location without auth. vehicle tracking at scale, fleet affected", "expected_ids": ["INC-014", "INC-010"], "description": "Wireless vehicle tracking"},
    {"query": "during fast charging session EVSE pushed malicious firmware update to onboard charger via ISO 15118 PLC. charger behavior changed - drawing more current than rated. communication was intercepted and modified at charging station", "expected_ids": ["INC-012", "INC-003"], "description": "Charging firmware injection"},
]


def precision_at_k(retrieved, expected, k=2):
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in expected) / k if k > 0 else 0.0


def recall_at_k(retrieved, expected, k=2):
    if not expected:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for e in expected if e in top_k) / len(expected)


def mrr_score(retrieved, expected):
    for rank, r in enumerate(retrieved, 1):
        if r in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, expected, k=2):
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 if r in expected else 0.0) / math.log2(i + 2)
        for i, r in enumerate(top_k)
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected), k)))
    return dcg / idcg if idcg > 0 else 0.0


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

    total_p, total_r, total_m, total_n = 0, 0, 0, 0
    passes = 0
    fails = []

    for i, tq in enumerate(TEST_QUERIES, 1):
        preprocessed = preprocess(tq["query"])
        fused = retriever.retrieve(tq["query"])
        reranked = reranker.rerank(
            tq["query"], fused, top_k=2, preprocessed=preprocessed
        )

        retrieved_ids = [r.incident_id for r in reranked]
        k = 2
        p = precision_at_k(retrieved_ids, tq["expected_ids"], k)
        r = recall_at_k(retrieved_ids, tq["expected_ids"], k)
        m = mrr_score(retrieved_ids, tq["expected_ids"])
        n = ndcg_at_k(retrieved_ids, tq["expected_ids"], k)

        total_p += p
        total_r += r
        total_m += m
        total_n += n

        match = set(retrieved_ids) == set(tq["expected_ids"])
        status = "PASS" if match else "FAIL"
        if match:
            passes += 1
        else:
            fails.append(i)

        print(
            f"{i:2d}. [{status}] {tq['description']:30s} | "
            f"P@2={p:.2f} R@2={r:.2f} MRR={m:.2f} nDCG={n:.2f} | "
            f"Got={retrieved_ids} Expected={tq['expected_ids']}"
        )

        # Show detailed scores for failed queries
        if not match:
            for r_item in reranked:
                print(
                    f"      {r_item.incident_id}: raw={r_item.raw_score:.4f} norm={r_item.normalized_score:.1f} "
                    f"CE_logit={r_item.cross_encoder_score:.2f} domain={r_item.domain_score:.4f}"
                )

    n_queries = len(TEST_QUERIES)
    print(f"\n{'=' * 90}")
    print(
        f"RESULTS: {passes}/{n_queries} queries passed "
        f"({passes / n_queries * 100:.0f}%)"
    )
    print(
        f"Avg P@2={total_p / n_queries:.2f}  "
        f"Avg R@2={total_r / n_queries:.2f}  "
        f"Avg MRR={total_m / n_queries:.2f}  "
        f"Avg nDCG@2={total_n / n_queries:.2f}"
    )
    if fails:
        print(f"Failed queries: {fails}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
