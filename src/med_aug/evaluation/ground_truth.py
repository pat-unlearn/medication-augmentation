"""Ground truth data management for evaluation."""

from typing import Dict, Set, List, Optional, Any
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GroundTruthEntry:
    """Single ground truth medication entry."""

    medication: str
    drug_class: str
    confidence: float = 1.0
    source: str = "manual"
    notes: Optional[str] = None
    validated_by: Optional[str] = None
    validated_at: Optional[str] = None


@dataclass
class GroundTruthDataset:
    """Complete ground truth dataset."""

    entries: List[GroundTruthEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_entry(self, entry: GroundTruthEntry) -> None:
        """Add a ground truth entry."""
        self.entries.append(entry)

    def get_mappings(self) -> Dict[str, str]:
        """Get medication -> drug_class mappings."""
        return {entry.medication: entry.drug_class for entry in self.entries}

    def get_by_class(self) -> Dict[str, List[str]]:
        """Get drug_class -> [medications] mappings."""
        by_class = defaultdict(list)
        for entry in self.entries:
            by_class[entry.drug_class].append(entry.medication)
        return dict(by_class)

    def filter_by_confidence(self, min_confidence: float = 0.8) -> "GroundTruthDataset":
        """Get only high-confidence entries."""
        high_conf_entries = [
            entry for entry in self.entries if entry.confidence >= min_confidence
        ]
        return GroundTruthDataset(
            entries=high_conf_entries,
            metadata={**self.metadata, "filtered_min_confidence": min_confidence},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metadata": self.metadata,
            "entries": [
                {
                    "medication": entry.medication,
                    "drug_class": entry.drug_class,
                    "confidence": entry.confidence,
                    "source": entry.source,
                    "notes": entry.notes,
                    "validated_by": entry.validated_by,
                    "validated_at": entry.validated_at,
                }
                for entry in self.entries
            ],
        }


class GroundTruthManager:
    """Manages ground truth datasets for evaluation."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize with optional data directory."""
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Current dataset
        self._dataset: Optional[GroundTruthDataset] = None

    def create_from_existing_conmeds(self, conmeds_file: Path) -> GroundTruthDataset:
        """
        Create ground truth dataset from existing conmeds.yml file.

        This gives us a baseline ground truth from the current NSCLC pipeline.
        """
        logger.info("creating_ground_truth_from_conmeds", file=str(conmeds_file))

        if not conmeds_file.exists():
            logger.error("conmeds_file_not_found", file=str(conmeds_file))
            return GroundTruthDataset()

        try:
            with open(conmeds_file, "r") as f:
                conmeds_data = yaml.safe_load(f)

            dataset = GroundTruthDataset(
                metadata={
                    "source": "existing_conmeds",
                    "source_file": str(conmeds_file),
                    "created_from": "nsclc_pipeline_conmeds",
                }
            )

            for drug_class, medications in conmeds_data.items():
                if isinstance(medications, list):
                    for medication in medications:
                        entry = GroundTruthEntry(
                            medication=medication.lower().strip(),
                            drug_class=drug_class,
                            confidence=1.0,  # Existing conmeds are high confidence
                            source="existing_conmeds",
                        )
                        dataset.add_entry(entry)

            logger.info(
                "ground_truth_created",
                total_entries=len(dataset.entries),
                drug_classes=len(set(e.drug_class for e in dataset.entries)),
            )

            return dataset

        except Exception as e:
            logger.error("ground_truth_creation_failed", error=str(e))
            return GroundTruthDataset()

    def load_from_msk_chord_sample(self) -> GroundTruthDataset:
        """
        Create ground truth from a curated sample of MSK CHORD data.

        This would need to be manually curated by clinical experts.
        """
        # This is where we'd load a manually validated sample
        # For now, return empty dataset
        logger.info("msk_chord_sample_not_available")
        return GroundTruthDataset()

    def create_from_expert_validation(
        self,
        medications: List[str],
        expert_classifications: Dict[str, str],
        expert_name: str = "clinical_expert",
    ) -> GroundTruthDataset:
        """Create ground truth from expert-validated classifications."""
        dataset = GroundTruthDataset(
            metadata={
                "source": "expert_validation",
                "expert": expert_name,
                "validation_type": "manual_review",
            }
        )

        for medication in medications:
            if medication in expert_classifications:
                entry = GroundTruthEntry(
                    medication=medication.lower().strip(),
                    drug_class=expert_classifications[medication],
                    confidence=1.0,
                    source="expert_validation",
                    validated_by=expert_name,
                )
                dataset.add_entry(entry)

        return dataset

    def load_dataset(self, name: str = "default") -> Optional[GroundTruthDataset]:
        """Load a named ground truth dataset."""
        file_path = self.data_dir / f"{name}_ground_truth.json"

        if not file_path.exists():
            logger.warning("ground_truth_file_not_found", file=str(file_path))
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            entries = [GroundTruthEntry(**entry_data) for entry_data in data["entries"]]

            dataset = GroundTruthDataset(
                entries=entries, metadata=data.get("metadata", {})
            )

            logger.info("ground_truth_loaded", name=name, entries=len(entries))

            return dataset

        except Exception as e:
            logger.error("ground_truth_load_failed", error=str(e))
            return None

    def save_dataset(self, dataset: GroundTruthDataset, name: str = "default") -> None:
        """Save ground truth dataset."""
        file_path = self.data_dir / f"{name}_ground_truth.json"

        try:
            with open(file_path, "w") as f:
                json.dump(dataset.to_dict(), f, indent=2)

            logger.info(
                "ground_truth_saved",
                name=name,
                file=str(file_path),
                entries=len(dataset.entries),
            )

        except Exception as e:
            logger.error("ground_truth_save_failed", error=str(e))

    def get_ground_truth_mappings(self, name: str = "default") -> Dict[str, str]:
        """Get simple medication -> drug_class mappings."""
        if self._dataset is None:
            self._dataset = self.load_dataset(name)

        if self._dataset is None:
            # Try to create from existing conmeds as fallback
            logger.info("creating_fallback_ground_truth")
            # You would specify the actual conmeds file path here
            conmeds_file = Path("data/conmeds_defaults.yml")  # Placeholder
            if conmeds_file.exists():
                self._dataset = self.create_from_existing_conmeds(conmeds_file)
            else:
                logger.warning("no_ground_truth_available")
                return {}

        return self._dataset.get_mappings()

    def get_statistics(self, name: str = "default") -> Dict[str, Any]:
        """Get statistics about the ground truth dataset."""
        dataset = self.load_dataset(name)
        if not dataset:
            return {}

        by_class = dataset.get_by_class()

        stats = {
            "total_medications": len(dataset.entries),
            "total_drug_classes": len(by_class),
            "medications_per_class": {cls: len(meds) for cls, meds in by_class.items()},
            "avg_medications_per_class": sum(len(meds) for meds in by_class.values())
            / len(by_class),
            "source_distribution": {},
            "confidence_distribution": {},
            "metadata": dataset.metadata,
        }

        # Source distribution
        source_counts = defaultdict(int)
        confidence_counts = defaultdict(int)

        for entry in dataset.entries:
            source_counts[entry.source] += 1
            conf_bucket = f"{entry.confidence:.1f}"
            confidence_counts[conf_bucket] += 1

        stats["source_distribution"] = dict(source_counts)
        stats["confidence_distribution"] = dict(confidence_counts)

        return stats

    def merge_datasets(
        self, *dataset_names: str, output_name: str = "merged"
    ) -> GroundTruthDataset:
        """Merge multiple ground truth datasets."""
        all_entries = []
        merged_metadata = {"source": "merged", "source_datasets": list(dataset_names)}

        for name in dataset_names:
            dataset = self.load_dataset(name)
            if dataset:
                all_entries.extend(dataset.entries)

        # Remove duplicates (keep highest confidence)
        medication_entries = {}
        for entry in all_entries:
            key = entry.medication.lower().strip()
            if (
                key not in medication_entries
                or entry.confidence > medication_entries[key].confidence
            ):
                medication_entries[key] = entry

        merged_dataset = GroundTruthDataset(
            entries=list(medication_entries.values()), metadata=merged_metadata
        )

        self.save_dataset(merged_dataset, output_name)

        logger.info(
            "datasets_merged",
            input_datasets=dataset_names,
            output_dataset=output_name,
            total_entries=len(merged_dataset.entries),
        )

        return merged_dataset
