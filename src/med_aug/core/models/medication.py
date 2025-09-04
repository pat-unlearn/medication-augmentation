"""Core medication and drug class models."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum
from datetime import datetime
from ..mixins import DictMixin


class MedicationType(str, Enum):
    """Types of medication names."""

    BRAND = "brand"
    GENERIC = "generic"
    CLINICAL_TRIAL = "clinical_trial"
    ABBREVIATION = "abbreviation"
    COMBINATION = "combination"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence levels for classifications."""

    HIGH = "high"  # >90%
    MEDIUM = "medium"  # 70-90%
    LOW = "low"  # <70%


@dataclass
class Medication(DictMixin):
    """Represents a single medication with metadata."""

    name: str
    type: MedicationType
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


@dataclass
class DrugClass(DictMixin):
    """Represents a drug class with associated medications."""

    name: str
    taking_variable: str  # e.g., "taking_pembrolizumab"
    current_medications: List[Medication]
    category: str  # "chemotherapy", "immunotherapy", etc.
    disease: str

    def add_medication(self, medication: Medication) -> "DrugClass":
        """Add medication (immutable update)."""
        return DrugClass(
            name=self.name,
            taking_variable=self.taking_variable,
            current_medications=[*self.current_medications, medication],
            category=self.category,
            disease=self.disease,
        )

    def get_medication_names(self) -> List[str]:
        """Get list of medication names."""
        return [med.name for med in self.current_medications]

    def get_high_confidence_medications(self) -> List[Medication]:
        """Get medications with high confidence scores."""
        return [
            med
            for med in self.current_medications
            if med.confidence_level == ConfidenceLevel.HIGH
        ]
