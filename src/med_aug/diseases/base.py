"""Abstract base class for disease modules."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class DrugClassConfig:
    """Configuration for a drug class within a disease module."""

    name: str
    keywords: List[str]
    confidence_threshold: float
    web_sources: List[str]

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"Confidence threshold must be between 0 and 1, got {self.confidence_threshold}"
            )
        if not self.keywords:
            raise ValueError(f"Drug class {self.name} must have at least one keyword")


class DiseaseModule(ABC):
    """Abstract base class for all disease modules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Disease identifier (e.g., 'nsclc', 'prostate')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable disease name."""
        pass

    @property
    @abstractmethod
    def drug_classes(self) -> List[DrugClassConfig]:
        """Disease-specific drug class configurations."""
        pass

    @abstractmethod
    def get_web_sources(self) -> List[str]:
        """Disease-specific data sources for web scraping."""
        pass

    @abstractmethod
    def get_llm_context(self) -> str:
        """Disease-specific context for LLM classification."""
        pass

    @abstractmethod
    def validate_medication(self, medication: str, drug_class: str) -> bool:
        """
        Disease-specific medication validation.

        Args:
            medication: Medication name to validate
            drug_class: Drug class to validate against

        Returns:
            True if medication is valid for the drug class, False otherwise
        """
        pass

    def get_drug_class_by_name(self, name: str) -> Optional[DrugClassConfig]:
        """Get drug class configuration by name."""
        for drug_class in self.drug_classes:
            if drug_class.name == name:
                return drug_class
        return None

    def get_all_keywords(self) -> List[str]:
        """Get all keywords from all drug classes."""
        keywords = []
        for drug_class in self.drug_classes:
            keywords.extend(drug_class.keywords)
        return list(set(keywords))

    def get_confidence_threshold(self, drug_class: str) -> float:
        """Get confidence threshold for a specific drug class."""
        config = self.get_drug_class_by_name(drug_class)
        return config.confidence_threshold if config else 0.7

    def get_taking_variables(self) -> List[str]:
        """Get list of taking_{drug} variable names."""
        variables = []
        for drug_class in self.drug_classes:
            # Convert drug class name to taking variable format
            # e.g., "immunotherapy" -> "taking_immunotherapy"
            for keyword in drug_class.keywords[:1]:  # Use first keyword as primary
                variable = (
                    f"taking_{keyword.lower().replace(' ', '_').replace('-', '_')}"
                )
                variables.append(variable)
        return variables

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert module to configuration dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "drug_classes": [
                {
                    "name": dc.name,
                    "keywords": dc.keywords,
                    "confidence_threshold": dc.confidence_threshold,
                    "web_sources": dc.web_sources,
                }
                for dc in self.drug_classes
            ],
            "web_sources": self.get_web_sources(),
            "taking_variables": self.get_taking_variables(),
        }


class DiseaseModuleConfig(BaseModel):
    """Configuration model for disease modules."""

    name: str = Field(..., description="Disease identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Disease description")
    drug_classes: List[Dict[str, Any]] = Field(
        ..., description="Drug class configurations"
    )
    web_sources: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Web source configurations"
    )
    llm_settings: Dict[str, Any] = Field(
        default_factory=dict, description="LLM settings"
    )
    validation: Dict[str, Any] = Field(
        default_factory=dict, description="Validation settings"
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DiseaseModuleConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data.get("disease", {}))

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        data = {"disease": self.model_dump()}
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
