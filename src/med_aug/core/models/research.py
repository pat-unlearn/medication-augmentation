"""Web research and external data models."""

from typing import List, Optional
from pydantic import BaseModel, Field


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
