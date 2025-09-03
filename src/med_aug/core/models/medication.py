"""Core data models for medication augmentation system."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
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

    # to_dict() method provided by DictMixin


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


@dataclass
class ColumnAnalysisResult(DictMixin):
    """Result of analyzing a data column for medication content."""

    column: str
    confidence: float
    total_count: int
    unique_count: int
    sample_medications: List[str]
    reasoning: str

    @property
    def is_likely_medication_column(self) -> bool:
        """Check if column likely contains medications."""
        return self.confidence >= 0.7

    # to_dict() method provided by DictMixin


@dataclass
class AugmentationResult(DictMixin):
    """Result of medication augmentation process."""

    original_count: int
    augmented_count: int
    new_medications: List[Medication]
    improvement_percentage: float
    processing_time: float
    quality_score: float
    disease: str

    @property
    def medications_added(self) -> int:
        """Number of new medications added."""
        return self.augmented_count - self.original_count

    @property
    def was_successful(self) -> bool:
        """Check if augmentation was successful."""
        return self.improvement_percentage > 0 and self.quality_score >= 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with special handling for nested objects."""
        data = super().to_dict()
        # Handle nested Medication objects manually
        data["new_medications"] = [med.to_dict() for med in self.new_medications]
        return data


class MedicationClassification(BaseModel):
    """Classification result from LLM."""

    medication_name: str = Field(..., description="Original medication name")
    drug_class: str = Field(..., description="Assigned drug class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Classification reasoning")
    alternative_classes: List[str] = Field(
        default_factory=list, description="Alternative possible classes"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    @property
    def needs_review(self) -> bool:
        """Check if classification needs human review."""
        return self.confidence < 0.8 or len(self.alternative_classes) > 0


class WebResearchResult(BaseModel):
    """Result from web research on a medication."""

    medication_name: str = Field(..., description="Medication being researched")
    generic_names: List[str] = Field(
        default_factory=list, description="Generic names found"
    )
    brand_names: List[str] = Field(
        default_factory=list, description="Brand names found"
    )
    drug_class_hints: List[str] = Field(
        default_factory=list, description="Drug class hints from sources"
    )
    mechanism_of_action: Optional[str] = Field(None, description="Mechanism of action")
    fda_approval_date: Optional[str] = Field(
        None, description="FDA approval date if available"
    )
    clinical_trials: List[str] = Field(
        default_factory=list, description="Related clinical trials"
    )
    sources: List[str] = Field(default_factory=list, description="Data sources used")

    @property
    def has_sufficient_data(self) -> bool:
        """Check if research found sufficient data."""
        return (
            len(self.generic_names) > 0
            or len(self.brand_names) > 0
            or len(self.drug_class_hints) > 0
        )

    def get_all_names(self) -> List[str]:
        """Get all medication names found."""
        return list(set(self.generic_names + self.brand_names))


class PipelineState(BaseModel):
    """State tracking for the augmentation pipeline."""

    disease: str = Field(..., description="Disease module being used")
    input_file: str = Field(..., description="Input data file path")
    output_file: str = Field(..., description="Output file path")
    current_phase: str = Field(
        default="initialization", description="Current pipeline phase"
    )
    phases_completed: List[str] = Field(
        default_factory=list, description="Completed phases"
    )
    medications_processed: int = Field(
        default=0, description="Number of medications processed"
    )
    medications_classified: int = Field(
        default=0, description="Number of medications classified"
    )
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    start_time: datetime = Field(
        default_factory=datetime.now, description="Pipeline start time"
    )

    def update_phase(self, phase: str) -> None:
        """Update current phase and mark previous as completed."""
        if self.current_phase != "initialization":
            self.phases_completed.append(self.current_phase)
        self.current_phase = phase

    def add_error(self, error: str) -> None:
        """Add an error to the list."""
        self.errors.append(f"[{datetime.now().isoformat()}] {error}")

    @property
    def has_errors(self) -> bool:
        """Check if pipeline has encountered errors."""
        return len(self.errors) > 0

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
