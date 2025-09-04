"""Result models for pipeline operations."""

from dataclasses import dataclass
from typing import List, Dict, Any
from .medication import Medication
from ..mixins import DictMixin


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
