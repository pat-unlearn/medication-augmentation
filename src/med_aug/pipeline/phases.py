"""Pipeline phases and phase management."""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio

from ..core.logging import get_logger, PerformanceLogger

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)


class PhaseStatus(Enum):
    """Status of a pipeline phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class PhaseResult:
    """Result from a pipeline phase execution."""

    phase_name: str
    status: PhaseStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate duration if end time is set."""
        if self.end_time and self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase_name": self.phase_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "output_data": self.output_data,
            "error": self.error,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }


class PipelinePhase(ABC):
    """Abstract base class for pipeline phases."""

    def __init__(self, name: str, required: bool = True):
        """
        Initialize pipeline phase.

        Args:
            name: Phase name
            required: Whether phase is required for pipeline success
        """
        self.name = name
        self.required = required
        self.status = PhaseStatus.PENDING
        self.result: Optional[PhaseResult] = None
        self.dependencies: List[str] = []

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """
        Execute the phase.

        Args:
            context: Shared context between phases

        Returns:
            Phase execution result
        """
        pass

    @abstractmethod
    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """
        Validate phase inputs before execution.

        Args:
            context: Shared context

        Returns:
            True if inputs are valid
        """
        pass

    async def cleanup(self, context: Dict[str, Any]) -> None:
        """
        Cleanup after phase execution.

        Args:
            context: Shared context
        """
        pass

    def can_run(self, completed_phases: List[str]) -> bool:
        """
        Check if phase can run based on dependencies.

        Args:
            completed_phases: List of completed phase names

        Returns:
            True if all dependencies are met
        """
        return all(dep in completed_phases for dep in self.dependencies)


class DataIngestionPhase(PipelinePhase):
    """Phase for data ingestion and initial loading."""

    def __init__(self):
        """Initialize data ingestion phase."""
        super().__init__("data_ingestion", required=True)

    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate data file exists and is readable."""
        import os

        data_file = context.get("input_file")
        if not data_file:
            logger.error("no_input_file_specified")
            return False

        if not os.path.exists(data_file):
            logger.error("input_file_not_found", file=data_file)
            return False

        return True

    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """Load and prepare input data."""
        import pandas as pd
        from pathlib import Path

        logger.info("data_ingestion_started", file=context["input_file"])
        perf_logger.start_operation("data_ingestion")

        start_time = datetime.now()

        try:
            # Load data file
            file_path = Path(context["input_file"])

            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Store in context
            context["dataframe"] = df
            context["original_shape"] = df.shape

            duration = perf_logger.end_operation(
                "data_ingestion", rows=len(df), columns=len(df.columns)
            )

            logger.info(
                "data_ingestion_completed",
                rows=len(df),
                columns=len(df.columns),
                duration=duration,
            )

            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data={
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                },
                metrics={
                    "file_size_bytes": file_path.stat().st_size,
                    "load_time_seconds": duration,
                },
            )

        except Exception as e:
            logger.error("data_ingestion_failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e),
            )


class ColumnAnalysisPhase(PipelinePhase):
    """Phase for analyzing columns to identify medication columns."""

    def __init__(self):
        """Initialize column analysis phase."""
        super().__init__("column_analysis", required=True)
        self.dependencies = ["data_ingestion"]

    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate dataframe exists in context."""
        if "dataframe" not in context:
            logger.error("no_dataframe_in_context")
            return False
        return True

    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """Analyze columns to find medication columns."""
        from ..core.analyzer import DataAnalyzer

        logger.info("column_analysis_started")
        perf_logger.start_operation("column_analysis")

        start_time = datetime.now()

        try:
            df = context["dataframe"]
            analyzer = DataAnalyzer()

            # Analyze with configurable threshold
            threshold = context.get("confidence_threshold", 0.5)
            results = analyzer.analyze_dataframe(df, threshold)

            # Store results in context
            context["column_analysis_results"] = results
            context["medication_columns"] = [r.column for r in results]

            duration = perf_logger.end_operation(
                "column_analysis",
                columns_analyzed=len(df.columns),
                medication_columns_found=len(results),
            )

            logger.info(
                "column_analysis_completed",
                medication_columns=len(results),
                best_column=results[0].column if results else None,
            )

            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data={
                    "medication_columns": [r.column for r in results],
                    "confidence_scores": {r.column: r.confidence for r in results},
                },
                metrics={
                    "columns_analyzed": len(df.columns),
                    "medication_columns_found": len(results),
                    "analysis_time_seconds": duration,
                },
            )

        except Exception as e:
            logger.error("column_analysis_failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e),
            )


class MedicationExtractionPhase(PipelinePhase):
    """Phase for extracting and normalizing medications."""

    def __init__(self):
        """Initialize medication extraction phase."""
        super().__init__("medication_extraction", required=True)
        self.dependencies = ["column_analysis"]

    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate medication columns exist."""
        if not context.get("medication_columns"):
            logger.warning("no_medication_columns_found")
            return False
        return True

    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """Extract medications from identified columns."""
        from ..core.extractor import MedicationExtractor

        logger.info("medication_extraction_started")
        perf_logger.start_operation("medication_extraction")

        start_time = datetime.now()

        try:
            df = context["dataframe"]
            columns = context["medication_columns"]
            extractor = MedicationExtractor()

            all_medications = []
            extraction_results = {}

            for column in columns:
                if column in df.columns:
                    result = extractor.extract_from_series(df[column], column)
                    extraction_results[column] = result
                    all_medications.extend(result.normalized_medications)

            # Store in context
            context["extraction_results"] = extraction_results
            context["all_medications"] = list(set(all_medications))

            duration = perf_logger.end_operation(
                "medication_extraction",
                columns_processed=len(columns),
                unique_medications=len(context["all_medications"]),
            )

            logger.info(
                "medication_extraction_completed",
                unique_medications=len(context["all_medications"]),
                duration=duration,
            )

            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data={
                    "unique_medications": len(context["all_medications"]),
                    "columns_processed": len(columns),
                },
                metrics={
                    "extraction_time_seconds": duration,
                    "total_medications_found": len(all_medications),
                },
            )

        except Exception as e:
            logger.error("medication_extraction_failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e),
            )


class WebResearchPhase(PipelinePhase):
    """Phase for web research and data augmentation."""

    def __init__(self):
        """Initialize web research phase."""
        super().__init__("web_research", required=False)
        self.dependencies = ["medication_extraction"]

    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate medications exist for research."""
        if not context.get("all_medications"):
            logger.warning("no_medications_for_research")
            return False
        return True

    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """Research medications using web scrapers."""
        from ..infrastructure.scrapers.fda import FDAScraper
        from ..infrastructure.scrapers.clinicaltrials import ClinicalTrialsScraper

        logger.info("web_research_started")
        perf_logger.start_operation("web_research")

        start_time = datetime.now()

        try:
            medications = context["all_medications"][:10]  # Limit for demo

            # Initialize scrapers
            fda_scraper = FDAScraper()
            ct_scraper = ClinicalTrialsScraper()

            research_results = {}

            # Research each medication
            async with fda_scraper, ct_scraper:
                for medication in medications:
                    research_results[medication] = {
                        "fda": await fda_scraper.scrape_medication_info(medication),
                        "clinical_trials": await ct_scraper.scrape_medication_info(
                            medication
                        ),
                    }

            # Store in context
            context["research_results"] = research_results

            duration = perf_logger.end_operation(
                "web_research", medications_researched=len(medications)
            )

            logger.info(
                "web_research_completed",
                medications_researched=len(medications),
                duration=duration,
            )

            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data={
                    "medications_researched": len(medications),
                    "sources_used": ["FDA", "ClinicalTrials.gov"],
                },
                metrics={"research_time_seconds": duration},
            )

        except Exception as e:
            logger.error("web_research_failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e),
            )


class ValidationPhase(PipelinePhase):
    """Phase for validating extracted medications."""

    def __init__(self):
        """Initialize validation phase."""
        super().__init__("validation", required=False)
        self.dependencies = ["medication_extraction"]

    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Check if disease module is specified."""
        return context.get("disease_module") is not None

    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """Validate medications against disease module."""
        from ..diseases import disease_registry

        logger.info("validation_started")
        perf_logger.start_operation("validation")

        start_time = datetime.now()

        try:
            disease_name = context.get("disease_module", "nsclc")
            module = disease_registry.get_module(disease_name)

            if not module:
                raise ValueError(f"Disease module not found: {disease_name}")

            medications = context["all_medications"]
            validation_results = {}

            for medication in medications:
                # Check each drug class
                found_valid = False
                found_drug_class = None

                for drug_class_config in module.drug_classes:
                    is_valid = module.validate_medication(
                        medication, drug_class_config.name
                    )
                    if is_valid:
                        found_valid = True
                        found_drug_class = drug_class_config.name
                        break

                validation_results[medication] = {
                    "valid": found_valid,
                    "drug_class": found_drug_class,
                    "confidence": 1.0 if found_valid else 0.0,
                }

            # Store in context
            context["validation_results"] = validation_results

            valid_count = sum(1 for v in validation_results.values() if v["valid"])

            duration = perf_logger.end_operation(
                "validation",
                medications_validated=len(medications),
                valid_medications=valid_count,
            )

            logger.info(
                "validation_completed",
                valid_medications=valid_count,
                total_medications=len(medications),
            )

            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data={
                    "valid_medications": valid_count,
                    "invalid_medications": len(medications) - valid_count,
                    "disease_module": disease_name,
                },
                metrics={"validation_time_seconds": duration},
            )

        except Exception as e:
            logger.error("validation_failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e),
            )


class LLMClassificationPhase(PipelinePhase):
    """Phase for LLM-based medication classification."""

    def __init__(self):
        """Initialize LLM classification phase."""
        super().__init__("llm_classification", required=False)
        self.dependencies = ["medication_extraction"]

    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Check if medications are available for classification."""
        return bool(context.get("all_medications"))

    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """Classify medications using LLM."""
        from ..llm.classifier import MedicationClassifier
        from ..llm.service import LLMService
        from ..llm.providers import LLMConfig, LLMModel, ProviderFactory

        logger.info("llm_classification_started")
        perf_logger.start_operation("llm_classification")

        start_time = datetime.now()

        try:
            # Get configuration
            disease_module = context.get("disease_module", "nsclc")
            medications = context.get("all_medications", [])

            # Check if we should use LLM
            enable_llm = context.get("enable_llm_classification", True)
            if not enable_llm:
                logger.info("llm_classification_disabled")
                return PhaseResult(
                    phase_name=self.name,
                    status=PhaseStatus.SKIPPED,
                    start_time=start_time,
                    end_time=datetime.now(),
                )

            # Initialize LLM service with appropriate provider
            provider_type = context.get("llm_provider", "claude_cli")
            config = LLMConfig(
                model=LLMModel.CLAUDE_3_SONNET,  # Explicitly use sonnet
                temperature=0.0,  # Use deterministic outputs
                max_tokens=2048,
                timeout=120,
                retry_attempts=2,
            )

            # Check if provider is available
            provider = ProviderFactory.create(provider_type, config)
            if not await provider.is_available():
                logger.warning(f"llm_provider_not_available", provider=provider_type)
                # Fall back to mock provider for testing
                provider = ProviderFactory.create("mock", config)

            llm_service = LLMService(provider=provider, config=config)

            # Initialize classifier
            classifier = MedicationClassifier(
                llm_service=llm_service,
                disease_module=disease_module,
                min_confidence=context.get("confidence_threshold", 0.5),
            )

            # Classify medications
            # Limit to first 20 for performance in initial implementation
            meds_to_classify = (
                medications[:20] if len(medications) > 20 else medications
            )

            logger.info(f"classifying_medications", count=len(meds_to_classify))

            # Create progress callback for live updates
            def llm_progress_callback(current: int, total: int, message: str):
                logger.info("llm_classification_progress", 
                          current=current, total=total, message=message)
                # Update progress tracker if available
                if hasattr(context, 'progress_tracker') and context['progress_tracker']:
                    progress_percent = (current / total) * 100 if total > 0 else 0
                    context['progress_tracker'].update_phase_progress(
                        "llm_classification", progress_percent, f"{message} ({current}/{total})"
                    )
            
            # Use batch classification with progress tracking
            batch_result = await classifier.classify_batch(
                meds_to_classify, batch_size=5, progress_callback=llm_progress_callback
            )

            # Store results in context
            context["llm_classifications"] = batch_result.to_dict()
            context["classification_results"] = batch_result.individual_results

            duration = perf_logger.end_operation("llm_classification")

            logger.info(
                "llm_classification_completed",
                classified=batch_result.classified_count,
                unclassified=len(batch_result.unclassified),
                confidence=batch_result.overall_confidence,
                duration=duration,
            )

            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data={
                    "medications_processed": len(meds_to_classify),
                    "classified": batch_result.classified_count,
                    "unclassified": len(batch_result.unclassified),
                    "confidence": batch_result.overall_confidence,
                    "classifications_by_class": {
                        k: len(v) for k, v in batch_result.classifications.items()
                    },
                },
                metrics={
                    "classification_time_seconds": duration,
                    "avg_confidence": batch_result.overall_confidence,
                },
            )

        except Exception as e:
            logger.error("llm_classification_failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e),
            )


class OutputGenerationPhase(PipelinePhase):
    """Phase for generating output files and reports."""

    def __init__(self):
        """Initialize output generation phase."""
        super().__init__("output_generation", required=True)
        self.dependencies = ["medication_extraction"]

    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Validate output path is specified."""
        return context.get("output_path") is not None

    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """Generate output files."""
        from pathlib import Path
        import json
        import pandas as pd

        logger.info("output_generation_started")
        perf_logger.start_operation("output_generation")

        start_time = datetime.now()

        try:
            output_path = Path(context["output_path"])
            output_path.mkdir(parents=True, exist_ok=True)

            artifacts = []

            # Export medications
            if context.get("all_medications"):
                medications_file = output_path / "extracted_medications.json"
                with open(medications_file, "w") as f:
                    json.dump(
                        {
                            "medications": context["all_medications"],
                            "count": len(context["all_medications"]),
                            "extraction_date": datetime.now().isoformat(),
                        },
                        f,
                        indent=2,
                    )
                artifacts.append(str(medications_file))

            # Export validation results
            if context.get("validation_results"):
                validation_file = output_path / "validation_results.csv"
                validation_df = pd.DataFrame.from_dict(
                    context["validation_results"], orient="index"
                )
                validation_df.to_csv(validation_file)
                artifacts.append(str(validation_file))

            # Export LLM classification results
            if context.get("llm_classifications"):
                llm_file = output_path / "llm_classifications.json"
                with open(llm_file, "w") as f:
                    json.dump(context["llm_classifications"], f, indent=2)
                artifacts.append(str(llm_file))

                # Also export as CSV for easy viewing
                if context.get("classification_results"):
                    class_csv = output_path / "medication_classifications.csv"
                    class_data = []
                    for med, result in context["classification_results"].items():
                        class_data.append(
                            {
                                "medication": med,
                                "drug_class": result.primary_class,
                                "confidence": result.confidence,
                                "confidence_level": result.confidence_level.value,
                                "alternatives": ", ".join(result.alternative_classes),
                                "mechanism": result.mechanism_of_action,
                            }
                        )
                    if class_data:
                        pd.DataFrame(class_data).to_csv(class_csv, index=False)
                        artifacts.append(str(class_csv))

            # Export conmeds.yml - THE KEY OUTPUT FOR PRD GOAL
            if context.get("llm_classifications") or context.get(
                "classification_results"
            ):
                from ..output.exporters import ConmedsYAMLExporter

                conmeds_file = output_path / "conmeds_augmented.yml"
                exporter = ConmedsYAMLExporter()

                # Prepare data for export
                export_data = {}
                if context.get("llm_classifications"):
                    export_data.update(context["llm_classifications"])
                if context.get("classification_results"):
                    export_data["classification_results"] = context[
                        "classification_results"
                    ]

                # Export the conmeds.yml file
                result_path = exporter.export(
                    export_data,
                    conmeds_file,
                    title=f"NSCLC Medication Augmentation - {datetime.now().strftime('%Y-%m-%d')}",
                    conmeds_file=context.get("conmeds_file"),  # Pass base conmeds for augmentation
                )
                artifacts.append(str(result_path))

                logger.info(
                    "conmeds_yaml_exported",
                    path=str(result_path),
                    size_bytes=result_path.stat().st_size,
                )

            # Export column analysis
            if context.get("column_analysis_results"):
                analysis_file = output_path / "column_analysis.json"
                with open(analysis_file, "w") as f:
                    json.dump(
                        {
                            "columns": [
                                r.to_dict() for r in context["column_analysis_results"]
                            ]
                        },
                        f,
                        indent=2,
                    )
                artifacts.append(str(analysis_file))

            duration = perf_logger.end_operation(
                "output_generation", files_created=len(artifacts)
            )

            logger.info(
                "output_generation_completed",
                artifacts_created=len(artifacts),
                output_path=str(output_path),
            )

            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data={
                    "output_path": str(output_path),
                    "files_created": len(artifacts),
                },
                metrics={"generation_time_seconds": duration},
                artifacts=artifacts,
            )

        except Exception as e:
            logger.error("output_generation_failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e),
            )
