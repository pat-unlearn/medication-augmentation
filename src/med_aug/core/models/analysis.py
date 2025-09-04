"""Data analysis result models."""

from dataclasses import dataclass
from typing import List, Dict, Any
from ..mixins import DictMixin


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including computed properties."""
        data = super().to_dict()
        data["is_likely"] = self.is_likely_medication_column
        return data
