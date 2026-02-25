"""
Development set: 20 queries used during pipeline tuning.

WARNING: These queries were visible during development of the reranker
domain scoring, concept groups, and alpha/beta weight tuning.  Results
on this set are inherently optimistic.

For unbiased evaluation, use holdout_set.py.

Frozen: 2026-02-24
"""

DEV_QUERIES = [
    {
        "id": "D01",
        "query": "fleet ops flagged weird can traffic on the powertrain bus.. someone plugged something into the obd port and we're seeing injected frames hitting the brake ecu, arb ID 0x130 every 10ms. no diagnostic session active. multiple vehicles affected drivers reporting brakes acting up at highway speed",
        "expected_ids": ["INC-002", "INC-006"],
        "description": "CAN bus injection",
    },
    {
        "id": "D02",
        "query": "found during supplier audit - central gateway ECU has unsigned firmware running. secure boot looks bypassed somehow.. internal firewall rules are gone. supplier says they didnt push any update, someone tampered with the gateway firmware. possible physical access",
        "expected_ids": ["INC-004", "INC-015"],
        "description": "Gateway firmware tampering",
    },
    {
        "id": "D03",
        "query": "VSOC alert - weird outbound connections from ~200 EVs right after last OTA push went out. turns out the firmware signing server was compromised and malicious code got pushed to fleet. CI/CD pipeline breach confirmed, signatures looked valid. need immediate rollback",
        "expected_ids": ["INC-003", "INC-015"],
        "description": "OTA supply chain compromise",
    },
    {
        "id": "D04",
        "query": "seeing tons of abnormal DNS TXT queries from infotainment units. encoded payloads carrying VIN numbers and GPS coords being tunneled out to external server. classic DNS exfiltration pattern, TCU involved too",
        "expected_ids": ["INC-001", "INC-010"],
        "description": "DNS data exfiltration",
    },
    {
        "id": "D05",
        "query": "3 vehicles stolen overnight same parking garage. security cameras show two guys with devices - one near the house one near the car. classic relay attack extending key fob signal to unlock and start wirelessly",
        "expected_ids": ["INC-009", "INC-013"],
        "description": "Keyless relay theft",
    },
    {
        "id": "D06",
        "query": "security researcher demo'd unlocking our cars by relaying BLE pairing between owners phone and BCM. bluetooth low energy digital key has a relay vuln, demonstrated at 50m range. affects all models with phone-as-key feature",
        "expected_ids": ["INC-007", "INC-009"],
        "description": "BLE digital key attack",
    },
    {
        "id": "D07",
        "query": "autonomous shuttle went completely off route today!! GNSS module receiving spoofed satellite signals from portable SDR nearby. navigation system has no integrity check at all, shuttle nearly drove into oncoming traffic",
        "expected_ids": ["INC-008", "INC-005"],
        "description": "GPS / GNSS spoofing",
    },
    {
        "id": "D08",
        "query": "got hit via crafted SMS on TCU. text message triggered buffer overflow in qualcomm baseband modem firmware.. attacker got remote shell on telematics unit and started pivoting into vehicle network. 12 vehicles confirmed compromised",
        "expected_ids": ["INC-010", "INC-014"],
        "description": "Cellular modem exploit",
    },
    {
        "id": "D09",
        "query": "caught a rogue cell tower IMSI catcher style near test facility. exploiting vulns in LTE baseband chipset to intercept vehicle telemetry. also tracking individual vehicles via cellular modem fingerprinting",
        "expected_ids": ["INC-014", "INC-010"],
        "description": "Baseband / rogue cell tower",
    },
    {
        "id": "D10",
        "query": "found rogue device in charging station doing MITM on ISO 15118 PLC communication. intercepting Plug&Charge certs and cloning auth tokens. V2G protocol completely compromised at this station, multiple EVs affected",
        "expected_ids": ["INC-012", "INC-003"],
        "description": "EV charging MITM",
    },
    {
        "id": "D11",
        "query": "cameras on 3 test vehicles all misclassify same stop sign as speed limit 80.. someone stuck adversarial stickers on it. ADAS traffic sign recognition completely fooled, affects forward camera ML pipeline",
        "expected_ids": ["INC-011", "INC-008"],
        "description": "ADAS adversarial patch",
    },
    {
        "id": "D12",
        "query": "reverse engineered key fob RF protocol - the rolling code PRNG is weak. can predict next 100 unlock codes after capturing just 2 transmissions. affects all vehicles with this RKE module, keyless entry completely broken",
        "expected_ids": ["INC-013", "INC-009"],
        "description": "Rolling code weakness",
    },
    {
        "id": "D13",
        "query": "the fleet management OBD dongles have open API with zero authentication. anyone on network can send raw CAN frames through them remotely.. tested it ourselves and injected messages on powertrain bus. 500 vehicles have these installed",
        "expected_ids": ["INC-006", "INC-002"],
        "description": "OBD fleet dongle exploit",
    },
    {
        "id": "D14",
        "query": "attacker used UDS diagnostic services RequestDownload 0x34 to rollback ECM firmware to version with known vulns. bypassed firmware version check via doip, downgraded security patches. ECU running year-old software now",
        "expected_ids": ["INC-015", "INC-004"],
        "description": "Diagnostic rollback attack",
    },
    {
        "id": "D15",
        "query": "smart intersection V2X broadcasting fake emergency vehicle warnings and road hazard alerts.. triggered automatic braking on 5 vehicles. V2I messages had valid format but came from unauthorized transmitter, no PKI cert validation",
        "expected_ids": ["INC-005", "INC-008"],
        "description": "V2X spoofing",
    },
    {
        "id": "D16",
        "query": "third party nav app on IVI is leaking data. caught it doing covert DNS tunneling to exfiltrate location history and CAN diagnostic PIDs. app was sideloaded, no sandboxing on infotainment linux platform",
        "expected_ids": ["INC-001", "INC-006"],
        "description": "Infotainment app compromise",
    },
    {
        "id": "D17",
        "query": "bus off condition on powertrain CAN!! something flooding high-priority frames at max bus speed. ABS and transmission modules went safe mode. looks like DoS on the CAN bus, possibly through OBD port or compromised ECU",
        "expected_ids": ["INC-002", "INC-006"],
        "description": "CAN bus flooding / DoS",
    },
    {
        "id": "D18",
        "query": "central gateway lost network segmentation.. infotainment CAN traffic leaking into safety-critical powertrain domain. no filtering between bus segments. either firmware bug or gateway config tampered. TCU also showing anomalous outbound traffic",
        "expected_ids": ["INC-004", "INC-010"],
        "description": "Gateway network bridging",
    },
    {
        "id": "D19",
        "query": "unauthorized RF beacon planted on vehicles transmitting location on LTE. traced to cellular modem vulnerability that lets external parties query TCU location without auth. vehicle tracking at scale, fleet affected",
        "expected_ids": ["INC-014", "INC-010"],
        "description": "Wireless vehicle tracking",
    },
    {
        "id": "D20",
        "query": "during fast charging session EVSE pushed malicious firmware update to onboard charger via ISO 15118 PLC. charger behavior changed - drawing more current than rated. communication was intercepted and modified at charging station",
        "expected_ids": ["INC-012", "INC-003"],
        "description": "Charging firmware injection",
    },
]
