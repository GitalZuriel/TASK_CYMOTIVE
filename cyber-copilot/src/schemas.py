"""
Pydantic v2 models for structured LLM output.

Defines the schema for Summary and Mitigation JSON responses,
including per-section confidence scores and validation logic.
"""

from pydantic import BaseModel, Field
from enum import Enum


class SeverityLevel(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ConfidenceMixin(BaseModel):
    """Mixin providing a confidence score to any section."""
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Model confidence in this section (0.0–1.0)",
    )


# ── Summary sub-models ──────────────────────────────────────────────

class IncidentOverview(ConfidenceMixin):
    text: str = Field(..., min_length=10)


class SeverityAssessment(ConfidenceMixin):
    level: SeverityLevel
    justification: str = Field(..., min_length=5)


class AttackVector(ConfidenceMixin):
    method: str
    details: str = ""


class AffectedSystems(ConfidenceMixin):
    systems: list[str] = Field(..., min_length=1)


class KeyIndicators(ConfidenceMixin):
    indicators: list[str] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    step: str
    is_estimated: bool = False


class Timeline(ConfidenceMixin):
    events: list[TimelineEntry] = Field(default_factory=list)


class ValidatedEntity(BaseModel):
    """A single entity validated by the LLM against the report text."""
    value: str
    source: str = Field(
        default="confirmed",
        description="confirmed | corrected | added",
    )
    original: str | None = Field(
        default=None,
        description="Original text if corrected (e.g., typo → fixed form)",
    )


class ValidatedEntities(BaseModel):
    """LLM-validated entities extracted from the incident report."""
    cve_ids: list[ValidatedEntity] = Field(default_factory=list)
    ip_addresses: list[ValidatedEntity] = Field(default_factory=list)
    ecu_names: list[ValidatedEntity] = Field(default_factory=list)
    protocols: list[ValidatedEntity] = Field(default_factory=list)
    attack_indicators: list[ValidatedEntity] = Field(default_factory=list)


class StructuredSummary(BaseModel):
    """Complete structured incident summary."""
    incident_overview: IncidentOverview
    severity: SeverityAssessment
    attack_vector: AttackVector
    affected_systems: AffectedSystems
    key_indicators: KeyIndicators
    timeline: Timeline
    validated_entities: ValidatedEntities | None = None

    @property
    def min_confidence(self) -> float:
        return min(
            self.incident_overview.confidence,
            self.severity.confidence,
            self.attack_vector.confidence,
            self.affected_systems.confidence,
            self.key_indicators.confidence,
            self.timeline.confidence,
        )

    @property
    def avg_confidence(self) -> float:
        scores = [
            self.incident_overview.confidence,
            self.severity.confidence,
            self.attack_vector.confidence,
            self.affected_systems.confidence,
            self.key_indicators.confidence,
            self.timeline.confidence,
        ]
        return sum(scores) / len(scores)


# ── Mitigation sub-models ───────────────────────────────────────────

class ActionItem(BaseModel):
    action: str
    grounding: str = ""


class ImmediateActions(ConfidenceMixin):
    actions: list[ActionItem] = Field(..., min_length=1)


class ShortTermActions(ConfidenceMixin):
    actions: list[ActionItem] = Field(..., min_length=1)


class LongTermRecommendations(ConfidenceMixin):
    actions: list[ActionItem] = Field(default_factory=list)


class StandardReference(BaseModel):
    standard: str
    relevance: str
    is_general_practice: bool = False


class RelatedStandards(ConfidenceMixin):
    standards: list[StandardReference] = Field(default_factory=list)


class StructuredMitigation(BaseModel):
    """Complete structured mitigation plan."""
    immediate_actions: ImmediateActions
    short_term_actions: ShortTermActions
    long_term_recommendations: LongTermRecommendations
    related_standards: RelatedStandards

    @property
    def min_confidence(self) -> float:
        return min(
            self.immediate_actions.confidence,
            self.short_term_actions.confidence,
            self.long_term_recommendations.confidence,
            self.related_standards.confidence,
        )

    @property
    def avg_confidence(self) -> float:
        scores = [
            self.immediate_actions.confidence,
            self.short_term_actions.confidence,
            self.long_term_recommendations.confidence,
            self.related_standards.confidence,
        ]
        return sum(scores) / len(scores)
