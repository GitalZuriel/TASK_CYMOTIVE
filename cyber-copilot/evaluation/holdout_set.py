"""
Holdout evaluation set: 33 queries (10 groups x 3 variants + 3 hard negatives).

STRICT HOLDOUT — Do NOT use these queries for tuning retrieval parameters,
reranker weights, domain scoring rules, or any pipeline component.
Results on this set measure true generalisation.

Each query has:
  expected_top1    — the single best incident (P@1 scored against this)
  acceptable_top2  — 2-3 incidents acceptable in slot #2 (flexible P@2)

Created: 2026-02-24
Dev set freeze date: 2026-02-24
"""

from dataclasses import dataclass


@dataclass
class HoldoutQuery:
    id: str                     # e.g. "H01-base"
    group: str                  # e.g. "H01"
    variant: str                # "base", "para1", "para2"
    query: str
    expected_top1: list[str]    # strict: best incident(s) for P@1
    acceptable_top2: list[str]  # relaxed: any of these OK in slot #2
    description: str
    noise_level: str            # "clean", "moderate", "noisy"


# ───────────────────────────────────────────────────────────────────
# 10 paraphrase groups  (base + para1 + para2 = 30 queries)
# ───────────────────────────────────────────────────────────────────

HOLDOUT_QUERIES: list[HoldoutQuery] = [

    # ── H01  AV sensor/perception + V2X  (INC-011, INC-005) ──────
    HoldoutQuery(
        id="H01-base", group="H01", variant="base",
        noise_level="clean",
        description="AV sensor/perception + V2X",
        query=(
            "our lidar and camera fusion on the autonomous test fleet keeps "
            "getting confused at the same intersection on rte 7. forward camera "
            "sees phantom obstacles that dont exist, vehicles slam brakes for "
            "nothing. checked the spot - looks like someone modified the road "
            "markings and placed weird patterned stickers on nearby "
            "infrastructure. also getting bogus safety alerts from the C-V2X "
            "radio about nonexistent road closures."
        ),
        expected_top1=["INC-011"],
        acceptable_top2=["INC-005", "INC-008"],
    ),
    HoldoutQuery(
        id="H01-para1", group="H01", variant="para1",
        noise_level="clean",
        description="AV sensor/perception + V2X (paraphrase)",
        query=(
            "perception pipeline freaking out on our AV test route. the DNN "
            "keeps hallucinating objects in the forward camera feed at one "
            "specific location - three separate shuttles had emergency stops "
            "from ghost obstacles. we sent someone to investigate and found "
            "modified signage with printed overlays. separately the V2X unit "
            "is receiving spoofed infrastructure warnings."
        ),
        expected_top1=["INC-011"],
        acceptable_top2=["INC-005", "INC-008"],
    ),
    HoldoutQuery(
        id="H01-para2", group="H01", variant="para2",
        noise_level="noisy",
        description="AV sensor/perception + V2X (noisy)",
        query=(
            "test fleet keeps braking randomly at same spot on route.. camera "
            "thinks theres something blocking the road but nothing is there. "
            "also C-V2X alerts about fake road hazards. looks intentional"
        ),
        expected_top1=["INC-011"],
        acceptable_top2=["INC-005", "INC-008"],
    ),

    # ── H02  BLE digital key + rolling code  (INC-007, INC-013) ──
    HoldoutQuery(
        id="H02-base", group="H02", variant="base",
        noise_level="clean",
        description="BLE digital key + rolling code (insurance report)",
        query=(
            "insurance claims spike for our connected SUV line - 47 thefts in "
            "3 weeks across 6 cities. all vehicles had phone-as-key enabled. "
            "forensic analysis shows the BLE authentication between the mobile "
            "device and body control module was intercepted and replayed. "
            "separately, some older models in the same fleet that use "
            "traditional fobs were also compromised - the random number "
            "generator for the keyless codes appears predictable after sniffing "
            "a few button presses."
        ),
        expected_top1=["INC-007"],
        acceptable_top2=["INC-013", "INC-009"],
    ),
    HoldoutQuery(
        id="H02-para1", group="H02", variant="para1",
        noise_level="clean",
        description="BLE digital key + rolling code (paraphrase)",
        query=(
            "seeing a wave of thefts targeting vehicles with smartphone digital "
            "key feature. the wireless handshake between phone app and car BCM "
            "is being relayed by attackers using portable hardware. additionally "
            "vehicles with conventional remote key fobs are vulnerable too - "
            "the rolling code sequence can be predicted from a small number of "
            "captured RF transmissions."
        ),
        expected_top1=["INC-007"],
        acceptable_top2=["INC-013", "INC-009"],
    ),
    HoldoutQuery(
        id="H02-para2", group="H02", variant="para2",
        noise_level="noisy",
        description="BLE digital key + rolling code (noisy)",
        query=(
            "lots of cars stolen, all had the phone key feature.. something "
            "wrong with the blutetooth low energy pairing i think. also the "
            "regular key fobs might have weak encryption on the rolling codes"
        ),
        expected_top1=["INC-007"],
        acceptable_top2=["INC-013", "INC-009"],
    ),

    # ── H03  DNS tunneling + baseband  (INC-001, INC-014) ────────
    HoldoutQuery(
        id="H03-base", group="H03", variant="base",
        noise_level="clean",
        description="DNS tunneling + baseband (VSOC night shift)",
        query=(
            "night shift VSOC report: head unit telemetry from approximately "
            "800 sedans is generating unusual DNS resolution patterns. the "
            "domain name queries contain base64 encoded blocks that decode to "
            "structured vehicle data including chassis numbers and trip logs. "
            "traced the origin to the HMI application layer. separately, three "
            "vehicles near our Munich facility have modems connecting to an "
            "unrecognized base station - suspect rogue cellular infrastructure "
            "intercepting telematics data."
        ),
        expected_top1=["INC-001"],
        acceptable_top2=["INC-014", "INC-010"],
    ),
    HoldoutQuery(
        id="H03-para1", group="H03", variant="para1",
        noise_level="clean",
        description="DNS tunneling + baseband (paraphrase)",
        query=(
            "found two simultaneous data leaks in our fleet monitoring. first: "
            "infotainment units across 800 vehicles are encoding sensitive "
            "telemetry into DNS queries and sending them to external resolvers "
            "- classic covert channel exfiltration via the name resolution "
            "protocol. second: handful of vehicles near the test track have "
            "their cellular modems associating with a fake cell tower "
            "performing IMSI catching and traffic interception."
        ),
        expected_top1=["INC-001"],
        acceptable_top2=["INC-014", "INC-010"],
    ),
    HoldoutQuery(
        id="H03-para2", group="H03", variant="para2",
        noise_level="noisy",
        description="DNS tunneling + baseband (noisy)",
        query=(
            "weird dns traffic from head units, looks like data is being snuck "
            "out through domain lookups.. also some cars near munchen "
            "connecting to suspicous cell tower that shouldnt be there"
        ),
        expected_top1=["INC-001"],
        acceptable_top2=["INC-014", "INC-010"],
    ),

    # ── H04  OTA + gateway firmware  (INC-003, INC-004) ──────────
    HoldoutQuery(
        id="H04-base", group="H04", variant="base",
        noise_level="clean",
        description="OTA + gateway firmware (QA audit)",
        query=(
            "QA team found two critical supply chain issues during production "
            "audit. first: the continuous integration system that builds and "
            "signs firmware packages for wireless updates was accessed by an "
            "unauthorized party last month. code review shows a hidden function "
            "was inserted into the TCU update package that phones home on port "
            "443. second: batch of central gateway ECUs from our tier-1 "
            "supplier arrived with modified bootloader code - hardware root of "
            "trust chain was defeated and the CAN-Ethernet bridge filtering is "
            "disabled in these units."
        ),
        expected_top1=["INC-003"],
        acceptable_top2=["INC-004", "INC-015"],
    ),
    HoldoutQuery(
        id="H04-para1", group="H04", variant="para1",
        noise_level="clean",
        description="OTA + gateway firmware (paraphrase)",
        query=(
            "supply chain double breach identified. the OTA build "
            "infrastructure was infiltrated - someone injected malicious "
            "payload into the firmware update pipeline and the cryptographic "
            "signatures still pass validation because the signing keys were "
            "compromised. also the central gateway controller from supplier X "
            "has tampered secure boot - unsigned code running on the ARM "
            "processor, firewall between vehicle network domains not functional."
        ),
        expected_top1=["INC-003"],
        acceptable_top2=["INC-004", "INC-015"],
    ),
    HoldoutQuery(
        id="H04-para2", group="H04", variant="para2",
        noise_level="noisy",
        description="OTA + gateway firmware (noisy)",
        query=(
            "bad firmware in OTA pipeline, signing server hacked and malicous "
            "code got thru. also gateway ECUs from suplier have broken "
            "secureboot, someone messed with the bootloader. both seem like "
            "supply chain issues"
        ),
        expected_top1=["INC-003"],
        acceptable_top2=["INC-004", "INC-015"],
    ),

    # ── H05  EV charging + diagnostic rollback  (INC-012, INC-015)
    HoldoutQuery(
        id="H05-base", group="H05", variant="base",
        noise_level="clean",
        description="EV charging + diagnostic rollback (field service)",
        query=(
            "service ticket from charging network partner: customers at three "
            "DC fast charge stations reported unexpected vehicle behavior after "
            "sessions. investigation found hardware implants on the charging "
            "cables performing interception of the power-line communication "
            "handshake between EVSE and vehicle OBC. the devices are cloning "
            "Plug and Charge credentials. additionally, one service "
            "technician's diagnostic laptop was stolen and used to connect to "
            "vehicles via the IP diagnostics port - the attacker initiated UDS "
            "download service to flash old ECM firmware versions that have "
            "known security holes."
        ),
        expected_top1=["INC-012"],
        acceptable_top2=["INC-015", "INC-003", "INC-004"],
    ),
    HoldoutQuery(
        id="H05-para1", group="H05", variant="para1",
        noise_level="clean",
        description="EV charging + diagnostic rollback (paraphrase)",
        query=(
            "two separate physical attacks reported. at charging stations, "
            "rogue interceptor devices on the ISO 15118 communication link are "
            "stealing vehicle payment certificates during V2G sessions. and at "
            "a dealership, someone used a stolen diagnostic tool to downgrade "
            "engine control unit firmware through the diagnostic protocol, "
            "bypassing version checks to reintroduce patched vulnerabilities."
        ),
        expected_top1=["INC-012"],
        acceptable_top2=["INC-015", "INC-003", "INC-004"],
    ),
    HoldoutQuery(
        id="H05-para2", group="H05", variant="para2",
        noise_level="noisy",
        description="EV charging + diagnostic rollback (noisy)",
        query=(
            "charging stations compromised, something stealing plug&charge "
            "certs during charging sessions. also someone with a diag tool "
            "rolled back ecu firmware using UDS to an old version with bugs"
        ),
        expected_top1=["INC-012"],
        acceptable_top2=["INC-015", "INC-003", "INC-004"],
    ),

    # ── H06  CAN injection + key fob relay  (INC-002, INC-009) ───
    HoldoutQuery(
        id="H06-base", group="H06", variant="base",
        noise_level="clean",
        description="CAN injection + key fob relay (law enforcement)",
        query=(
            "forensics report from auto theft task force: suspects are using "
            "two-stage approach. phase one gains entry by extending the "
            "wireless signal from the owner's key fob using a pair of radio "
            "repeater devices positioned near the home and vehicle. phase two: "
            "once inside, a custom microcontroller plugged into the under-dash "
            "diagnostic connector injects forged messages onto the high-speed "
            "vehicle bus targeting the electronic braking module, allowing the "
            "immobilizer to be bypassed."
        ),
        expected_top1=["INC-009"],
        acceptable_top2=["INC-002", "INC-013", "INC-006"],
    ),
    HoldoutQuery(
        id="H06-para1", group="H06", variant="para1",
        noise_level="clean",
        description="CAN injection + key fob relay (paraphrase)",
        query=(
            "vehicle theft ring combining RF relay technology for PKES bypass "
            "to gain physical access, followed by CAN bus message injection "
            "through the OBD port to override the immobilizer. the relay "
            "extends passive keyless range to 50+ meters. the injected frames "
            "target safety-critical arbitration IDs on the powertrain network."
        ),
        expected_top1=["INC-009"],
        acceptable_top2=["INC-002", "INC-013", "INC-006"],
    ),
    HoldoutQuery(
        id="H06-para2", group="H06", variant="para2",
        noise_level="noisy",
        description="CAN injection + key fob relay (noisy)",
        query=(
            "theives using relay boxes to unlock cars from far away then "
            "plugging something into obd2 port to inject fake can messages on "
            "brake bus. key signal gets boosted wirelessly"
        ),
        expected_top1=["INC-009"],
        acceptable_top2=["INC-002", "INC-013", "INC-006"],
    ),

    # ── H07  Telematics + OBD dongle  (INC-010, INC-006) ─────────
    HoldoutQuery(
        id="H07-base", group="H07", variant="base",
        noise_level="clean",
        description="Telematics + OBD dongle (pentest report)",
        query=(
            "pentest findings summary: we identified two remote entry points "
            "into the vehicle network. first, the telematics control unit has "
            "a parsable AT command interface reachable through the cellular "
            "data channel - we achieved code execution by sending a malformed "
            "message to the modem's baseband processor and pivoted to the "
            "vehicle ethernet backbone. second, the aftermarket fleet tracking "
            "device connected to the diagnostic port exposes an "
            "unauthenticated web API over its mobile data connection - we were "
            "able to read all vehicle parameters and write arbitrary messages "
            "to the internal bus."
        ),
        expected_top1=["INC-010"],
        acceptable_top2=["INC-006", "INC-014", "INC-002"],
    ),
    HoldoutQuery(
        id="H07-para1", group="H07", variant="para1",
        noise_level="clean",
        description="Telematics + OBD dongle (paraphrase)",
        query=(
            "two remote attack surfaces found during security assessment. the "
            "embedded cellular module in the TCU can be exploited via crafted "
            "SMS triggering a memory corruption bug in the Qualcomm chipset "
            "firmware. separately, the fleet management OBD-II dongle has wide "
            "open REST endpoints allowing unauthenticated CAN frame "
            "transmission."
        ),
        expected_top1=["INC-010"],
        acceptable_top2=["INC-006", "INC-014", "INC-002"],
    ),
    HoldoutQuery(
        id="H07-para2", group="H07", variant="para2",
        noise_level="noisy",
        description="Telematics + OBD dongle (noisy)",
        query=(
            "found we can get into the car remotely two ways - through the "
            "cell modem on the TCU (buffer overlow in baseband) and thru the "
            "fleet obd dongle which has no auth on its API"
        ),
        expected_top1=["INC-010"],
        acceptable_top2=["INC-006", "INC-014", "INC-002"],
    ),

    # ── H08  GPS spoofing + ADAS camera  (INC-008, INC-011) ──────
    HoldoutQuery(
        id="H08-base", group="H08", variant="base",
        noise_level="clean",
        description="GPS spoofing + ADAS camera (safety investigation)",
        query=(
            "safety investigation into two AV incidents this week. the first "
            "involved our level 4 shuttle receiving false positioning data "
            "from what appears to be a software-defined radio transmitting "
            "counterfeit satellite navigation signals. the shuttle's position "
            "estimate drifted 4 meters and it crossed a lane boundary. the "
            "GNSS unit has no multi-constellation verification. second "
            "incident: traffic sign classification on the forward-facing "
            "camera pipeline was defeated by physical perturbation patches - "
            "the neural network confidently misidentified regulatory signage "
            "causing inappropriate speed behavior."
        ),
        expected_top1=["INC-008"],
        acceptable_top2=["INC-011", "INC-005"],
    ),
    HoldoutQuery(
        id="H08-para1", group="H08", variant="para1",
        noise_level="clean",
        description="GPS spoofing + ADAS camera (paraphrase)",
        query=(
            "dual sensor attacks on autonomous fleet. GPS receiver on shuttle "
            "compromised by spoofing attack using portable SDR - no inertial "
            "navigation cross-check to catch the drift. also the ADAS "
            "perception system's MobileNet classifier was tricked by "
            "adversarial stickers applied to road signs near the test zone."
        ),
        expected_top1=["INC-008"],
        acceptable_top2=["INC-011", "INC-005"],
    ),
    HoldoutQuery(
        id="H08-para2", group="H08", variant="para2",
        noise_level="noisy",
        description="GPS spoofing + ADAS camera (noisy)",
        query=(
            "gps on the shuttle went haywire, someone spoofin the satellite "
            "signal with an SDR. also cameras on test cars misreading stop "
            "signs because of weird stickers someone put on them"
        ),
        expected_top1=["INC-008"],
        acceptable_top2=["INC-011", "INC-005"],
    ),

    # ── H09  Gateway + telematics — VAGUE  (INC-004, INC-010) ────
    HoldoutQuery(
        id="H09-base", group="H09", variant="base",
        noise_level="moderate",
        description="Gateway + telematics (vague manager email)",
        query=(
            "forwarding from field ops: we think there might be a security "
            "issue with the networking inside some of our newer vehicles. the "
            "team says data that should stay on one side of the car's internal "
            "network is showing up on the other side where the safety systems "
            "are. the firewall component in the middle doesn't seem to be "
            "filtering properly, possibly firmware was changed. also the "
            "communication module that connects to the mobile network is "
            "sending data somewhere unexpected."
        ),
        expected_top1=["INC-004"],
        acceptable_top2=["INC-010", "INC-001"],
    ),
    HoldoutQuery(
        id="H09-para1", group="H09", variant="para1",
        noise_level="clean",
        description="Gateway + telematics (paraphrase)",
        query=(
            "internal escalation: gateway ECU appears to have lost its network "
            "isolation function. traffic from the infotainment domain is "
            "bleeding into the safety-critical powertrain domain with no "
            "firewall enforcement. firmware integrity suspect. concurrently, "
            "the telematics unit cellular interface is establishing connections "
            "to unrecognized external endpoints."
        ),
        expected_top1=["INC-004"],
        acceptable_top2=["INC-010", "INC-001"],
    ),
    HoldoutQuery(
        id="H09-para2", group="H09", variant="para2",
        noise_level="noisy",
        description="Gateway + telematics (noisy)",
        query=(
            "something wrong with car network, safety stuff not separated "
            "from entertainment stuff anymore. also the cell connection thingy "
            "sending weird data outside. gateway firmware maybe hacked?"
        ),
        expected_top1=["INC-004"],
        acceptable_top2=["INC-010", "INC-001"],
    ),

    # ── H10  V2X + EV charging  (INC-005, INC-012) ───────────────
    HoldoutQuery(
        id="H10-base", group="H10", variant="base",
        noise_level="clean",
        description="V2X + EV charging (smart city audit)",
        query=(
            "smart city security audit findings: at two signalized "
            "intersections the ITS-G5 roadside units are being impersonated. "
            "an unauthorized radio is broadcasting cooperative awareness and "
            "hazard notification messages that trigger emergency responses in "
            "passing connected vehicles. no certificate chain validation at "
            "the receiving end. at the municipal EV charging depot, the "
            "communication between supply equipment and vehicles over the "
            "power line has been tampered - a device installed in the charging "
            "post is intercepting session credentials and injecting modified "
            "parameters to the on-board charging controller."
        ),
        expected_top1=["INC-005"],
        acceptable_top2=["INC-012", "INC-008"],
    ),
    HoldoutQuery(
        id="H10-para1", group="H10", variant="para1",
        noise_level="clean",
        description="V2X + EV charging (paraphrase)",
        query=(
            "infrastructure attack report: V2X communication at intersections "
            "compromised by rogue transmitter sending fake DENM messages via "
            "802.11p, causing unnecessary emergency braking. PKI validation "
            "missing on vehicle side. at EV charging facility, MITM device on "
            "ISO 15118 PLC link stealing Plug and Charge certificates and "
            "manipulating the SECC communication with vehicle OBC."
        ),
        expected_top1=["INC-005"],
        acceptable_top2=["INC-012", "INC-008"],
    ),
    HoldoutQuery(
        id="H10-para2", group="H10", variant="para2",
        noise_level="noisy",
        description="V2X + EV charging (noisy)",
        query=(
            "v2x radio at intersection sending fake warnings making cars brake "
            "for nothing, no cert checking. also charging station hacked, "
            "someone stealing payment credentials during plug and charge"
        ),
        expected_top1=["INC-005"],
        acceptable_top2=["INC-012", "INC-008"],
    ),
]

# ───────────────────────────────────────────────────────────────────
# 3 hard-negative queries
#
# Designed to confuse the cross-encoder by using highly similar
# language to a *wrong* incident.  Only the domain scoring should
# save the ranking.
# ───────────────────────────────────────────────────────────────────

HARD_NEGATIVE_QUERIES: list[HoldoutQuery] = [

    # HN1: Talks about "firmware update" and "cellular" — CE will love
    #       INC-003 (OTA) or INC-014 (cellular), but the real answer
    #       is INC-015 (diagnostic rollback) + INC-004 (gateway tamper).
    #       Domain scoring should catch UDS/DoIP keywords.
    HoldoutQuery(
        id="HN1", group="HN", variant="hard_neg",
        noise_level="clean",
        description="Hard neg: firmware language → should be UDS rollback",
        query=(
            "firmware update irregularity detected on engine control module "
            "during scheduled service. version check shows regression to "
            "v2.1.3 which was deprecated six months ago due to CVE-2025-4412. "
            "the downgrade was performed over the DoIP diagnostic channel using "
            "UDS service 0x34 RequestDownload. no OTA involvement - this was "
            "a wired diagnostic session. gateway logs show the same tool also "
            "modified the gateway boot sequence."
        ),
        expected_top1=["INC-015"],
        acceptable_top2=["INC-004", "INC-003"],
    ),

    # HN2: Mentions "spoofing" and "radio" heavily — CE will gravitate
    #       to INC-008 (GPS) or INC-005 (V2X), but the real scenario is
    #       a keyless relay (INC-009) using radio amplification.
    HoldoutQuery(
        id="HN2", group="HN", variant="hard_neg",
        noise_level="clean",
        description="Hard neg: spoofing/radio language → should be key relay",
        query=(
            "radio frequency analysis confirms signal anomaly in the 125 kHz "
            "and 433 MHz bands around the parking facility. the low-frequency "
            "challenge from the vehicle PKES module is being captured by one "
            "device and re-transmitted to a second device near the owner's "
            "residence where the key fob responds. the UHF reply is then "
            "relayed back. no satellite or V2X signals involved - purely a "
            "passive keyless entry relay exploitation using wideband SDR "
            "equipment."
        ),
        expected_top1=["INC-009"],
        acceptable_top2=["INC-013", "INC-007"],
    ),

    # HN3: Heavy "network", "data", "exfiltration" language — CE will
    #       pull toward INC-001 (DNS tunneling), but the actual attack
    #       is through the charging station (INC-012) stealing Plug&Charge
    #       certificates.  Domain scoring should prefer EV/PLC keywords.
    HoldoutQuery(
        id="HN3", group="HN", variant="hard_neg",
        noise_level="clean",
        description="Hard neg: exfiltration language → should be EV charging",
        query=(
            "data exfiltration confirmed at fleet charging depot. the stolen "
            "artifacts are not vehicle telemetry but rather the X.509 "
            "certificates used for automated payment during Plug and Charge "
            "sessions. the interception occurs on the power-line communication "
            "layer between the supply equipment and on-board charger during "
            "the ISO 15118 TLS handshake. no DNS or network-layer tunneling "
            "involved - extraction is via physical hardware implant on the "
            "charging cable."
        ),
        expected_top1=["INC-012"],
        acceptable_top2=["INC-003", "INC-005"],
    ),
]

# Combine all queries for unified evaluation
ALL_HOLDOUT_QUERIES: list[HoldoutQuery] = HOLDOUT_QUERIES + HARD_NEGATIVE_QUERIES


# ───────────────────────────────────────────────────────────────────
# Convenience accessors
# ───────────────────────────────────────────────────────────────────

def get_base_queries() -> list[HoldoutQuery]:
    """Return only the 10 base variants (no paraphrases or hard negatives)."""
    return [q for q in HOLDOUT_QUERIES if q.variant == "base"]


def get_group(group_id: str) -> list[HoldoutQuery]:
    """Return all variants for a paraphrase group."""
    return [q for q in HOLDOUT_QUERIES if q.group == group_id]


def get_all_groups() -> list[str]:
    """Return sorted list of unique paraphrase group IDs."""
    return sorted(set(q.group for q in HOLDOUT_QUERIES))
