"""
Input validation, entity extraction, and edge case handling for incident reports.

Handles noisy, incomplete, or malformed input by extracting structured entities
and generating warnings or clarification prompts.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Minimum character threshold for a meaningful incident report
MIN_INPUT_LENGTH = 20

# Patterns for entity extraction
CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
ECU_KEYWORDS = [
    "ECU", "TCU", "TCM", "BCM", "ECM", "EBCM", "ADAS",
    "infotainment", "head-unit", "head unit", "IVI", "HMI",
    "gateway", "telematics", "OBC", "OBD-II", "OBD",
    "domain controller", "body control", "engine control",
    "brake control", "transmission control", "powertrain",
    "GNSS", "V2X", "modem", "PKES", "CAN bus", "CAN",
]
# Keywords that need word-boundary matching to avoid false positives
# (e.g., "IVI" inside "activity", "CAN" inside "scan", "OBC" inside "tobacco")
_SHORT_ECU_KEYWORDS = {"IVI", "HMI", "CAN", "OBC", "ECU", "TCU", "TCM",
                       "BCM", "ECM", "EBCM", "OBD", "PKES"}
# Map: human-readable label -> search stem (prefix matching catches verb forms)
ATTACK_INDICATORS_MAP = {
    "injection": "inject",
    "spoofing": "spoof",
    "buffer overflow": "overflow",
    "exploit": "exploit",
    "backdoor": "backdoor",
    "relay attack": "relay attack",
    "tunneling": "tunnel",
    "tampering": "tamper",
    "MITM": "mitm",
    "man-in-the-middle": "man-in-the-middle",
    "brute force": "brute force",
    "rollback": "rollback",
    "replay": "replay",
    "fuzzing": "fuzz",
    "reverse engineering": "reverse engineer",
    "exfiltration": "exfiltrat",
    "lateral movement": "lateral movement",
    "privilege escalation": "privilege escalation",
    "denial of service": "denial of service",
    "ransomware": "ransomware",
    "malware": "malware",
    "rootkit": "rootkit",
    "command and control": "command and control",
    "C2": " c2 ",
    "unauthorized access": "unauthori",
    "manipulation": "manipulat",
}
TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b"
    r"|\b\d{2}:\d{2}(?::\d{2})?\s*(?:UTC|GMT|[A-Z]{2,4})?\b"
)
# Simple heuristic for garbled / non-English text: high ratio of non-ASCII chars
NON_ASCII_THRESHOLD = 0.3


@dataclass
class ExtractedEntities:
    """Structured entities pulled from an incident report."""
    cve_ids: list[str] = field(default_factory=list)
    ip_addresses: list[str] = field(default_factory=list)
    ecu_names: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    attack_indicators: list[str] = field(default_factory=list)


@dataclass
class IncidentField:
    """A single extracted incident field with confidence."""
    value: str
    confidence: str  # "high" or "medium"
    source: str      # explanation of extraction basis


@dataclass
class PreprocessResult:
    """Result of the preprocessing pipeline."""
    cleaned_text: str
    entities: ExtractedEntities
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True
    needs_clarification: bool = False
    clarification_questions: list[str] = field(default_factory=list)
    extracted_fields: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    """Normalize whitespace, strip control characters, basic cleaning."""
    # Remove control characters except newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_garbled_text(text: str) -> bool:
    """Flag text with a high ratio of non-ASCII characters as potentially garbled."""
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) > NON_ASCII_THRESHOLD


def _extract_entities(text: str) -> ExtractedEntities:
    """Extract cybersecurity-relevant entities from the report text."""
    entities = ExtractedEntities()

    entities.cve_ids = CVE_PATTERN.findall(text)
    entities.ip_addresses = IP_PATTERN.findall(text)
    entities.timestamps = TIMESTAMP_PATTERN.findall(text)

    text_lower = text.lower()
    entities.ecu_names = []
    for kw in ECU_KEYWORDS:
        kw_lower = kw.lower()
        if kw in _SHORT_ECU_KEYWORDS:
            # Word-boundary match to avoid false positives (e.g., "IVI" in "activity")
            if re.search(rf"\b{re.escape(kw_lower)}\b", text_lower):
                entities.ecu_names.append(kw)
        else:
            if kw_lower in text_lower:
                entities.ecu_names.append(kw)
    entities.attack_indicators = [
        label for label, stem in ATTACK_INDICATORS_MAP.items()
        if stem.lower() in text_lower
    ]

    return entities


def _extract_incident_fields(
    text: str, entities: ExtractedEntities,
) -> tuple[dict[str, IncidentField], list[str]]:
    """
    Extract incident fields with confidence, return only truly missing questions.

    Returns:
        (extracted_fields, missing_field_questions)
    """
    extracted: dict[str, IncidentField] = {}
    missing: list[str] = []
    text_lower = text.lower()

    # --- Affected Subsystem ---
    if entities.ecu_names:
        extracted["affected_subsystem"] = IncidentField(
            value=", ".join(entities.ecu_names),
            confidence="high",
            source="Directly mentioned in report",
        )
    else:
        missing.append(
            "Which ECU or vehicle subsystem is affected? "
            "(e.g., infotainment, telematics, gateway, ADAS, BCM)"
        )

    # --- Attack Type ---
    if entities.attack_indicators:
        extracted["attack_type"] = IncidentField(
            value=", ".join(entities.attack_indicators),
            confidence="high",
            source="Directly mentioned in report",
        )
    else:
        missing.append(
            "What type of attack or anomaly was observed? "
            "(e.g., injection, spoofing, exploit, unauthorized access)"
        )

    # --- Severity ---
    severity_map = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}
    found_severity = None
    for term, label in severity_map.items():
        if term in text_lower:
            found_severity = label
            break

    if found_severity:
        extracted["severity"] = IncidentField(
            value=found_severity,
            confidence="high",
            source="Explicitly stated in report",
        )
    else:
        # Infer severity from contextual clues
        clues = []
        if any(w in text_lower for w in [
            "safety", "brake", "steering", "powertrain", "transmission", "adas",
        ]):
            clues.append("safety-related system affected")
        if any(w in text_lower for w in ["fleet", "multiple vehicle", "widespread"]):
            clues.append("multiple vehicles impacted")
        if entities.cve_ids:
            clues.append("known CVE referenced")

        if clues:
            extracted["severity"] = IncidentField(
                value="High (inferred)",
                confidence="medium",
                source="; ".join(clues),
            )
            # Don't ask for confirmation — inferred value is good enough
            # for PoC.  The UI shows confidence="medium" so the analyst
            # knows it's inferred and can override if needed.
        else:
            missing.append(
                "What is the assessed severity of this incident? "
                "(Critical / High / Medium / Low)"
            )

    # --- Timestamp ---
    if entities.timestamps:
        extracted["timestamp"] = IncidentField(
            value=", ".join(entities.timestamps),
            confidence="high",
            source="Directly mentioned in report",
        )
    elif any(w in text_lower for w in [
        "today", "yesterday", "last week", "recently", "hours ago",
    ]):
        extracted["timestamp"] = IncidentField(
            value="Relative time reference found",
            confidence="medium",
            source="Relative time reference in report",
        )
    else:
        missing.append(
            "When did this incident occur or when was it first detected?"
        )

    # --- CVE (bonus — always extract if present) ---
    if entities.cve_ids:
        extracted["cve"] = IncidentField(
            value=", ".join(entities.cve_ids),
            confidence="high",
            source="Directly mentioned in report",
        )

    return extracted, missing


# ---------------------------------------------------------------------------
# Rule-based query expansion for automotive cybersecurity
# ---------------------------------------------------------------------------
_EXPANSION_RULES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"\bupdate[sd]?\b", re.I),       ["firmware", "OTA", "rollback", "patch"]),
    (re.compile(r"\bconnectivity\b", re.I),       ["telematics", "network communication"]),
    (re.compile(r"\bfleet\b", re.I),              ["OTA", "multiple vehicles"]),
    (re.compile(r"\bmisconfiguration\b", re.I),   ["firmware", "rollback", "software bug"]),
    (re.compile(r"\btelemetry\b", re.I),          ["telematics", "data exfiltration"]),
]


def expand_query(text: str) -> str:
    """Append domain synonyms to improve retrieval recall (retrieval only)."""
    extras: set[str] = set()
    for pattern, terms in _EXPANSION_RULES:
        if pattern.search(text):
            extras.update(terms)
    return f"{text} {' '.join(sorted(extras))}" if extras else text


def preprocess(text: str) -> PreprocessResult:
    """
    Full preprocessing pipeline for an incident report.

    Steps:
        1. Clean and normalize text
        2. Check length and detect garbled content
        3. Extract structured entities
        4. Identify missing critical information
        5. Return structured result with warnings

    Args:
        text: Raw incident report text.

    Returns:
        PreprocessResult with cleaned text, entities, warnings, and validity flags.
    """
    if not text or not text.strip():
        logger.warning("Empty input received")
        return PreprocessResult(
            cleaned_text="",
            entities=ExtractedEntities(),
            warnings=["Input is empty."],
            is_valid=False,
            needs_clarification=True,
            clarification_questions=[
                "Please provide an incident report to analyze."
            ],
        )

    cleaned = _clean_text(text)
    warnings: list[str] = []

    # Very short input
    if len(cleaned) < MIN_INPUT_LENGTH:
        logger.warning("Input too short (%d chars): '%s'", len(cleaned), cleaned)
        return PreprocessResult(
            cleaned_text=cleaned,
            entities=_extract_entities(cleaned),
            warnings=["Input is too short for meaningful analysis."],
            is_valid=False,
            needs_clarification=True,
            clarification_questions=[
                "The report is very brief. Could you provide more details?",
                "What vehicle system or ECU is affected?",
                "What symptoms or anomalies were observed?",
                "When did this occur and how was it detected?",
            ],
        )

    # Garbled / mixed-language text
    if _detect_garbled_text(cleaned):
        warnings.append(
            "Input contains a high proportion of non-ASCII characters and may be "
            "garbled or in a non-English language. Results may be less reliable."
        )
        logger.warning("Potentially garbled text detected")

    entities = _extract_entities(cleaned)
    extracted_fields, clarification_questions = _extract_incident_fields(cleaned, entities)

    result = PreprocessResult(
        cleaned_text=cleaned,
        entities=entities,
        warnings=warnings,
        is_valid=True,
        needs_clarification=len(clarification_questions) > 0,
        clarification_questions=clarification_questions,
        extracted_fields=extracted_fields,
    )

    logger.info(
        "Preprocessed input: %d chars, %d CVEs, %d IPs, %d ECUs, %d attack indicators",
        len(cleaned),
        len(entities.cve_ids),
        len(entities.ip_addresses),
        len(entities.ecu_names),
        len(entities.attack_indicators),
    )
    return result
