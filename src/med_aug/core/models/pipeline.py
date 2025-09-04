"""Pipeline state and execution tracking models."""

from typing import List
from datetime import datetime
from pydantic import BaseModel, Field


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
