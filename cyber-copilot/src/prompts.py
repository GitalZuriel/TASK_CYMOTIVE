"""
Prompt templates for the Cybersecurity Copilot.

Contains system prompt, summarization, mitigation, and edge-case prompts
with few-shot examples for consistent, structured LLM output.
"""

# --- System Prompt ---
SYSTEM_PROMPT = """\
You are a senior automotive cybersecurity analyst assistant. You help security analysts \
investigate and respond to vehicle cybersecurity incidents. You have deep expertise in:
- Automotive network protocols (CAN, LIN, FlexRay, Automotive Ethernet)
- ECU security and firmware analysis
- V2X communication security
- OTA update security
- UNECE WP.29 and ISO/SAE 21434 standards
- AUTOSAR security architecture
- Automotive IDPS (Intrusion Detection and Prevention Systems)

Always provide structured, actionable responses. Be precise with technical details.
When referencing similar past incidents, explain how they relate to the current case.
If information is insufficient, clearly state what additional data would improve your analysis.

## Grounding Rules (CRITICAL)
Your responses MUST be grounded in the provided evidence:
1. **Retrieved incidents**: Reference specific past incidents from the RAG context when making \
comparisons or recommendations. Cite the incident title or ID.
2. **Report data only**: Only state facts explicitly present in the incident report or \
retrieved context. If you infer something, clearly mark it as "[Inferred]".
3. **No ungrounded knowledge**: Do NOT introduce standards, CVEs, tools, or frameworks \
that are not mentioned in the report or retrieved incidents, unless directly relevant to \
the specific attack type described. If you reference external knowledge, mark it as \
"[General best practice]".
4. **Severity consistency**: If the preprocessor provides an inferred severity level, \
your assessment must either agree with it (with justification) or explicitly explain \
why you diverge. Never silently escalate or downgrade."""


# --- Summarization Prompt ---
SUMMARY_PROMPT = """\
Analyze the following automotive cybersecurity incident report and any similar past incidents \
retrieved from our database. Produce a structured incident summary.

## Similar Past Incidents (from RAG retrieval)
{similar_incidents}

## Current Incident Report
{incident_report}

## Preprocessor Extraction (automated)
{extracted_fields}

## Instructions
Produce a structured summary with EXACTLY the following sections:

**Incident Overview**: 2-3 sentences describing what happened. Use ONLY facts from the report.
**Severity**: Critical / High / Medium / Low — with brief justification. \
If the preprocessor already inferred a severity, you must either confirm it with evidence \
or explicitly explain why you disagree. Do NOT silently change the severity.
**Attack Vector**: The method of attack (e.g., network, physical, wireless, supply chain).
**Affected Systems**: Which ECUs, modules, or vehicle subsystems are impacted. \
Use the preprocessor extraction as a starting point.
**Key Indicators**: List ONLY IPs, CVEs, anomalous behaviors, CAN IDs, or other IOCs \
that are explicitly present in the report or retrieved incidents.
**Timeline**: Known or estimated sequence of events. Mark uncertain steps as "[Estimated]".

## Entity Validation
Review the auto-extracted entities from the preprocessor above. At the end of your summary, \
add a section called **Validated Entities** where you:
1. **Confirm** correctly extracted entities
2. **Correct** any entities with typos (show original → corrected)
3. **Add** entities explicitly mentioned in the report that the preprocessor missed
⚠️ Do NOT invent entities. Only include what is explicitly in the report text.

## Few-Shot Example

Input: "Detected unusual CAN traffic from infotainment ECU. Frames with arbitration ID 0x7DF \
being sent at 500ms intervals to the OBD diagnostic bus. The infotainment system was recently \
updated via OTA. Source IP 10.0.0.45 found in TCU logs connecting to external host 192.168.1.100."

Output:
**Incident Overview**: The infotainment ECU is generating unauthorized diagnostic CAN frames \
(arb ID 0x7DF) targeting the OBD bus at regular intervals, suggesting a compromised OTA update \
may have introduced diagnostic scanning capability. Suspicious network activity from the TCU \
to an external host was also observed.
**Severity**: High — Unauthorized cross-domain CAN communication from the infotainment to \
the diagnostic bus indicates a potential domain isolation breach.
**Attack Vector**: Supply Chain — compromised OTA update with network-based C2 communication.
**Affected Systems**: Infotainment ECU, OBD diagnostic bus, Telematics Control Unit (TCU).
**Key Indicators**: CAN arb ID 0x7DF (OBD broadcast), 500ms transmission interval, \
source IP 10.0.0.45, external host 192.168.1.100, recent OTA update.
**Timeline**: OTA update deployed → infotainment ECU begins anomalous CAN transmissions → \
TCU establishes outbound connection to external host → anomaly detected in CAN IDS logs.

---
Now analyze the current incident:"""


# --- Mitigation Prompt ---
MITIGATION_PROMPT = """\
Based on the following incident summary and similar past incidents, provide a structured \
mitigation and response plan for this automotive cybersecurity incident.

## Incident Summary
{incident_summary}

## Similar Past Incidents
{similar_incidents}

## Preprocessor Extraction (automated)
{extracted_fields}

## Instructions
Provide recommendations in EXACTLY the following sections:

**Immediate Actions (First 24 Hours)**: Critical containment and triage steps that a SOC \
analyst can execute NOW. Focus on isolation, monitoring, and evidence preservation.
**Short-term Actions (1-7 Days)**: Investigation, patching, and hardening steps. \
Must be scoped to the specific attack — no general hardening wish-lists.
**Long-term Recommendations**: Process improvements and targeted hardening. \
Only recommend architectural changes if a similar past incident from the retrieved context \
shows that incremental fixes were insufficient. Mark any architectural suggestion as \
"[Architectural — requires engineering review]".
**Related Standards & Regulations**: ONLY reference standards that are directly relevant \
to the specific attack type and affected systems. If a standard was mentioned in a retrieved \
past incident, cite that incident. For standards from general knowledge, mark as \
"[General best practice]". Do NOT list standards just to appear comprehensive.

## Grounding Constraint
Every recommendation must trace back to either:
- A fact in the incident report (e.g., "since OBD-II port access was used...")
- A pattern from a retrieved past incident (e.g., "similar to Past Incident X where...")
- A direct countermeasure for the identified attack vector

If you cannot ground a recommendation, do not include it.

## Few-Shot Example

Input Summary: "CAN bus injection attack via OBD-II port on brake ECU, no MAC on safety frames."

Output:
**Immediate Actions (First 24 Hours)**:
1. Issue fleet-wide advisory to disable physical OBD-II port access where possible \
(direct countermeasure for the identified physical access vector)
2. Deploy CAN IDS rule to detect anomalous 0x130 frames on powertrain bus \
(targets the specific arbitration ID from the report)
3. Activate enhanced monitoring on all EBCM-related CAN arbitration IDs
4. Collect CAN bus logs from affected vehicles for forensic analysis

**Short-term Actions (1-7 Days)**:
1. Develop and test OTA patch to add CMAC authentication on safety-critical CAN frames \
(direct fix for "no MAC on safety frames" noted in report)
2. Implement CAN frame rate limiting on gateway ECU for diagnostic arbitration IDs
3. Add OBD-II port tamper detection alert to vehicle IDPS
4. Conduct impact assessment across all vehicle models using the same EBCM

**Long-term Recommendations**:
1. Deploy vehicle-level IDPS with CAN anomaly detection baseline
2. Add physical OBD-II port authentication (e.g., challenge-response before diagnostic session)
3. [Architectural — requires engineering review] Evaluate migration to CAN-FD with SecOC \
for safety-critical communications, based on similar pattern in Past Incident X

**Related Standards & Regulations**:
- ISO/SAE 21434: Threat analysis and risk assessment — directly applicable to CAN bus \
injection risk identified in this incident
- UNECE WP.29 R155: CSMS requirement for CAN bus security monitoring — mandates detection \
capability for the attack type observed

---
Now provide the mitigation plan for the current incident:"""


# --- Edge Case / Clarification Prompt ---
EDGE_CASE_PROMPT = """\
The following incident report appears to be incomplete or unclear. Generate a structured \
set of clarification questions to help the analyst provide a more complete report.

## Incomplete Report
{incident_report}

## Detected Issues
{issues}

## Instructions
Generate 3-5 specific, actionable clarification questions. Each question should:
- Target a specific missing piece of information
- Explain why this information is important for analysis
- Suggest examples of what the answer might look like

## Few-Shot Example

Input: "ecu hacked help"
Issues: Input too short, no ECU specified, no attack type, no severity, no timestamp.

Output:
1. **Which ECU or vehicle subsystem was compromised?**
   This is critical for assessing the blast radius. Examples: infotainment head unit, \
telematics control unit (TCU), body control module (BCM), ADAS domain controller.

2. **What symptoms or anomalies were observed?**
   Helps classify the attack type and severity. Examples: unexpected CAN frames, \
unauthorized network connections, firmware checksum mismatch, unusual diagnostic sessions.

3. **When was the incident first detected, and how?**
   Establishes the timeline and detection method. Examples: "CAN IDS alert at 14:30 UTC", \
"anomalous OTA download flagged by VSOC", "customer reported unexpected vehicle behavior".

4. **What is the vehicle make, model, and year?**
   Determines the specific ECU hardware/software stack and known vulnerability surface.

5. **Is this a single vehicle or fleet-wide incident?**
   Impacts the urgency and scope of the response.

---
Now generate clarification questions for the current report:"""


# ── JSON Structured Output Variants ──────────────────────────────────

SUMMARY_PROMPT_JSON = """\
Analyze the following automotive cybersecurity incident report and any similar past incidents \
retrieved from our database. Produce a structured incident summary as a JSON object.

## Similar Past Incidents (from RAG retrieval)
{similar_incidents}

## Current Incident Report
{incident_report}

## Preprocessor Extraction (automated)
{extracted_fields}

## JSON Output Schema
You MUST respond with ONLY a valid JSON object (no markdown fences, no commentary before or \
after the JSON) matching this exact structure:

{{
  "incident_overview": {{
    "text": "2-3 sentences describing what happened, using ONLY facts from the report",
    "confidence": 0.0
  }},
  "severity": {{
    "level": "Critical | High | Medium | Low",
    "justification": "Brief justification with evidence",
    "confidence": 0.0
  }},
  "attack_vector": {{
    "method": "e.g. network, physical, wireless, supply chain",
    "details": "Additional details about the attack vector",
    "confidence": 0.0
  }},
  "affected_systems": {{
    "systems": ["ECU1", "Module2"],
    "confidence": 0.0
  }},
  "key_indicators": {{
    "indicators": ["IP: x.x.x.x", "CAN ID: 0xNNN", "CVE-YYYY-NNNN"],
    "confidence": 0.0
  }},
  "timeline": {{
    "events": [
      {{"step": "description of event", "is_estimated": false}}
    ],
    "confidence": 0.0
  }},
  "validated_entities": {{
    "cve_ids": [
      {{"value": "CVE-YYYY-NNNN", "source": "confirmed | corrected | added", "original": null}}
    ],
    "ip_addresses": [
      {{"value": "x.x.x.x", "source": "confirmed | corrected | added", "original": null}}
    ],
    "ecu_names": [
      {{"value": "ECU Name", "source": "confirmed | corrected | added", "original": null}}
    ],
    "protocols": [
      {{"value": "Protocol Name", "source": "confirmed | corrected | added", "original": null}}
    ],
    "attack_indicators": [
      {{"value": "Attack Type", "source": "confirmed | corrected | added", "original": null}}
    ]
  }}
}}

## Entity Validation Task
The preprocessor auto-extracted entities listed under "Preprocessor Extraction" above. \
As part of your analysis, you MUST also validate and enrich these entities:
1. **CONFIRM** each preprocessor-extracted entity if it is correct (source: "confirmed")
2. **CORRECT** if the report has a typo or partial match — fix it and set source: "corrected", \
put the original text in "original" (e.g., "infotanment" → value: "Infotainment ECU", original: "infotanment")
3. **ADD** entities you find in the report text that the preprocessor missed (source: "added")

⚠️ ONLY add entities that are EXPLICITLY mentioned in the report text. \
Do NOT invent CVEs, IPs, or identifiers that are not in the report.

## Confidence Scoring Rules
- 1.0: Directly stated in the report with clear evidence
- 0.7-0.9: Strongly supported by retrieved similar incidents
- 0.4-0.6: Partially supported, some inference required
- 0.1-0.3: Weak evidence, mostly inferred
- Set confidence below 0.4 if you lack sufficient information for that section

## Few-Shot Example

Input: "Detected unusual CAN traffic from infotainment ECU. Frames with arbitration ID 0x7DF \
being sent at 500ms intervals to the OBD diagnostic bus. The infotainment system was recently \
updated via OTA. Source IP 10.0.0.45 found in TCU logs connecting to external host 192.168.1.100."

Output:
{{
  "incident_overview": {{
    "text": "The infotainment ECU is generating unauthorized diagnostic CAN frames (arb ID 0x7DF) targeting the OBD bus at regular intervals, suggesting a compromised OTA update may have introduced diagnostic scanning capability. Suspicious network activity from the TCU to an external host was also observed.",
    "confidence": 0.95
  }},
  "severity": {{
    "level": "High",
    "justification": "Unauthorized cross-domain CAN communication from the infotainment to the diagnostic bus indicates a potential domain isolation breach.",
    "confidence": 0.9
  }},
  "attack_vector": {{
    "method": "Supply Chain",
    "details": "Compromised OTA update with network-based C2 communication",
    "confidence": 0.85
  }},
  "affected_systems": {{
    "systems": ["Infotainment ECU", "OBD diagnostic bus", "Telematics Control Unit (TCU)"],
    "confidence": 0.95
  }},
  "key_indicators": {{
    "indicators": ["CAN arb ID 0x7DF (OBD broadcast)", "500ms transmission interval", "Source IP 10.0.0.45", "External host 192.168.1.100", "Recent OTA update"],
    "confidence": 0.95
  }},
  "timeline": {{
    "events": [
      {{"step": "OTA update deployed", "is_estimated": false}},
      {{"step": "Infotainment ECU begins anomalous CAN transmissions", "is_estimated": true}},
      {{"step": "TCU establishes outbound connection to external host", "is_estimated": true}},
      {{"step": "Anomaly detected in CAN IDS logs", "is_estimated": false}}
    ],
    "confidence": 0.7
  }}
}}

---
Now analyze the current incident. Respond with ONLY the JSON object:"""


MITIGATION_PROMPT_JSON = """\
Based on the following incident summary and similar past incidents, provide a structured \
mitigation and response plan as a JSON object.

## Incident Summary
{incident_summary}

## Similar Past Incidents
{similar_incidents}

## Preprocessor Extraction (automated)
{extracted_fields}

## JSON Output Schema
You MUST respond with ONLY a valid JSON object (no markdown fences, no commentary before or \
after the JSON) matching this exact structure:

{{
  "immediate_actions": {{
    "actions": [
      {{"action": "What to do", "grounding": "Why — cite report fact or past incident"}}
    ],
    "confidence": 0.0
  }},
  "short_term_actions": {{
    "actions": [
      {{"action": "What to do (1-7 days)", "grounding": "Why"}}
    ],
    "confidence": 0.0
  }},
  "long_term_recommendations": {{
    "actions": [
      {{"action": "What to do long-term", "grounding": "Why"}}
    ],
    "confidence": 0.0
  }},
  "related_standards": {{
    "standards": [
      {{"standard": "ISO/SAE 21434", "relevance": "Why it applies", "is_general_practice": false}}
    ],
    "confidence": 0.0
  }}
}}

## Grounding Constraint
Every recommendation must trace back to either:
- A fact in the incident report (e.g., "since OBD-II port access was used...")
- A pattern from a retrieved past incident (e.g., "similar to Past Incident X where...")
- A direct countermeasure for the identified attack vector

If you cannot ground a recommendation, do not include it.

## Confidence Scoring Rules
- 1.0: Action directly addresses a stated fact in the report
- 0.7-0.9: Action based on strong pattern from retrieved past incidents
- 0.4-0.6: Action partially supported, some inference
- 0.1-0.3: Weak grounding, mostly general knowledge

## Few-Shot Example

Input Summary: "CAN bus injection attack via OBD-II port on brake ECU, no MAC on safety frames."

Output:
{{
  "immediate_actions": {{
    "actions": [
      {{"action": "Issue fleet-wide advisory to disable physical OBD-II port access where possible", "grounding": "Direct countermeasure for the identified physical access vector"}},
      {{"action": "Deploy CAN IDS rule to detect anomalous 0x130 frames on powertrain bus", "grounding": "Targets the specific arbitration ID from the report"}},
      {{"action": "Activate enhanced monitoring on all EBCM-related CAN arbitration IDs", "grounding": "Brake ECU identified as target in report"}},
      {{"action": "Collect CAN bus logs from affected vehicles for forensic analysis", "grounding": "Evidence preservation for identified CAN injection"}}
    ],
    "confidence": 0.95
  }},
  "short_term_actions": {{
    "actions": [
      {{"action": "Develop and test OTA patch to add CMAC authentication on safety-critical CAN frames", "grounding": "Direct fix for 'no MAC on safety frames' noted in report"}},
      {{"action": "Implement CAN frame rate limiting on gateway ECU for diagnostic arbitration IDs", "grounding": "Prevents injection flooding pattern observed"}},
      {{"action": "Add OBD-II port tamper detection alert to vehicle IDPS", "grounding": "Physical access vector identified in report"}}
    ],
    "confidence": 0.9
  }},
  "long_term_recommendations": {{
    "actions": [
      {{"action": "Deploy vehicle-level IDPS with CAN anomaly detection baseline", "grounding": "Detection capability for the class of CAN injection attacks"}},
      {{"action": "[Architectural] Evaluate migration to CAN-FD with SecOC for safety-critical communications", "grounding": "Similar pattern in Past Incident X showed incremental fixes insufficient"}}
    ],
    "confidence": 0.7
  }},
  "related_standards": {{
    "standards": [
      {{"standard": "ISO/SAE 21434", "relevance": "Threat analysis and risk assessment — directly applicable to CAN bus injection risk", "is_general_practice": false}},
      {{"standard": "UNECE WP.29 R155", "relevance": "CSMS requirement for CAN bus security monitoring", "is_general_practice": false}}
    ],
    "confidence": 0.85
  }}
}}

---
Now provide the mitigation plan for the current incident. Respond with ONLY the JSON object:"""


def format_extracted_fields(extracted_fields: dict) -> str:
    """
    Format extracted fields for inclusion in LLM prompts as grounding context.

    Args:
        extracted_fields: Dict mapping field names to IncidentField objects
                          (with .value, .confidence, .source attributes).

    Returns:
        Formatted string for prompt insertion.
    """
    if not extracted_fields:
        return "No fields were automatically extracted."

    field_labels = {
        "affected_subsystem": "Affected Subsystem",
        "attack_type": "Attack Type",
        "severity": "Severity",
        "timestamp": "Timestamp",
        "cve": "CVE",
    }
    lines = []
    for name, f in extracted_fields.items():
        label = field_labels.get(name, name.replace("_", " ").title())
        confidence_tag = f" [confidence: {f.confidence}]"
        lines.append(f"- {label}: {f.value}{confidence_tag} — {f.source}")
    return "\n".join(lines)


def format_similar_incidents(incidents: list[dict]) -> str:
    """
    Format a list of similar incidents for inclusion in prompts.

    Args:
        incidents: List of incident dicts with at minimum 'title', 'description',
                   and optionally 'metadata'.

    Returns:
        Formatted string for prompt insertion.
    """
    if not incidents:
        return "No similar past incidents found in the database."

    parts: list[str] = []
    for i, inc in enumerate(incidents, 1):
        meta = inc.get("metadata", {})
        parts.append(
            f"### Past Incident {i}: {inc.get('title', 'Unknown')}\n"
            f"- **Severity**: {meta.get('severity', 'N/A')}\n"
            f"- **Attack Vector**: {meta.get('attack_vector', 'N/A')}\n"
            f"- **Affected System**: {meta.get('affected_system', 'N/A')}\n"
            f"- **CVE**: {meta.get('cve', 'N/A') or 'None'}\n"
            f"- **Details**: {inc.get('description', 'No details available.')}\n"
        )
    return "\n".join(parts)
