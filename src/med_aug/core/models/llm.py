"""LLM-related models for classification and processing."""

from typing import List
from pydantic import BaseModel, Field, field_validator


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
