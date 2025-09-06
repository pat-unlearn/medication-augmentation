"""Pipeline phases and phase management."""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from ..core.logging import get_logger, PerformanceLogger
from ..core.mixins import DictMixin

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
class PhaseResult(DictMixin):
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

    # to_dict() method provided by DictMixin


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

        file_path = Path(context["input_file"])
        file_size = file_path.stat().st_size
        logger.info(
            "data_ingestion_started",
            file=context["input_file"],
            file_size_bytes=file_size,
            file_format=file_path.suffix.lower(),
        )
        perf_logger.start_operation("data_ingestion")

        start_time = datetime.now()

        try:
            # Load data file
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Log detailed dataset information
            logger.info(
                "dataset_loaded",
                rows=len(df),
                columns=len(df.columns),
                column_names=list(df.columns),
                memory_usage_mb=round(
                    df.memory_usage(deep=True).sum() / 1024 / 1024, 2
                ),
            )

            # Log sample data for debugging
            if len(df) > 0:
                # Show first few rows of important columns (limit to avoid log spam)
                sample_data = df.head(3).to_dict("records")
                logger.info("dataset_sample", sample_rows=sample_data)

                # Log data types
                dtypes_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
                logger.info("dataset_dtypes", column_types=dtypes_info)

                # Log null/missing data info
                null_info = df.isnull().sum().to_dict()
                if any(null_info.values()):
                    logger.info("dataset_null_values", null_counts=null_info)

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

        df = context["dataframe"]
        threshold = context.get("confidence_threshold", 0.5)
        logger.info(
            "column_analysis_started",
            total_columns=len(df.columns),
            columns=list(df.columns),
            confidence_threshold=threshold,
        )
        perf_logger.start_operation("column_analysis")

        start_time = datetime.now()

        try:
            analyzer = DataAnalyzer()

            # Analyze each column and log details
            results = analyzer.analyze_dataframe(df, threshold)

            # Log detailed analysis results for each column
            for result in results:
                logger.info(
                    "column_identified_as_medication",
                    column=result.column,
                    confidence=round(result.confidence, 3),
                    unique_values=result.unique_count,
                    sample_medications=result.sample_medications[:5],
                )  # Limit samples

            # Log columns that didn't meet threshold
            identified_columns = {r.column for r in results}
            rejected_columns = [
                col for col in df.columns if col not in identified_columns
            ]
            if rejected_columns:
                logger.info(
                    "columns_rejected_as_medication",
                    columns=rejected_columns,
                    reason=f"confidence below {threshold}",
                )

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

        df = context["dataframe"]
        columns = context["medication_columns"]
        logger.info(
            "medication_extraction_started",
            medication_columns=columns,
            total_rows=len(df),
        )
        perf_logger.start_operation("medication_extraction")

        start_time = datetime.now()

        try:
            extractor = MedicationExtractor()

            all_medications = []
            extraction_results = {}

            # Get column analysis results to prioritize high-confidence columns
            column_results = context.get("column_analysis_results", [])

            # Only process the top confidence column(s) to avoid noise from low-quality columns
            if column_results:
                # Sort by confidence and only take columns with confidence > 0.5
                high_confidence_columns = [
                    r.column for r in column_results if r.confidence > 0.5
                ]
                if high_confidence_columns:
                    logger.info(
                        "filtering_to_high_confidence_columns",
                        columns=high_confidence_columns,
                        total_available=len(columns),
                    )
                    columns = [col for col in columns if col in high_confidence_columns]
                else:
                    # Fallback to just the top column if none meet threshold
                    logger.info("using_top_column_only", column=columns[0])
                    columns = [columns[0]]

            for column in columns:
                if column in df.columns:
                    result = extractor.extract_from_series(df[column], column)
                    extraction_results[column] = result
                    all_medications.extend(result.normalized_medications)

                    # Show top medications from this column for context
                    top_meds = result.get_top_medications(10)
                    logger.info(
                        "extracted_from_column",
                        column=column,
                        medications_found=len(result.normalized_medications),
                        total_rows_in_column=result.total_rows,
                        top_medications=dict(top_meds) if top_meds else {},
                    )

            # Store in context
            unique_medications = list(set(all_medications))
            context["extraction_results"] = extraction_results
            context["all_medications"] = unique_medications

            # Log summary of all unique medications found across all columns
            logger.info(
                "all_unique_medications_extracted",
                unique_count=len(unique_medications),
                medications=unique_medications[:20],
            )  # Show first 20 for context

            duration = perf_logger.end_operation(
                "medication_extraction",
                columns_processed=len(columns),
                unique_medications=len(unique_medications),
            )

            logger.info(
                "medication_extraction_completed",
                unique_medications=len(unique_medications),
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


class MedicationNormalizationPhase(PipelinePhase):
    """Phase for LLM-based medication normalization and variant discovery."""

    def __init__(self):
        """Initialize medication normalization phase."""
        super().__init__("medication_normalization", required=False)
        self.dependencies = ["medication_extraction"]

    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Check if medications are available for normalization."""
        return bool(context.get("all_medications"))

    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """Normalize medications and find variants using LLM with batch processing."""
        from ..llm.service import LLMService
        from ..llm.providers import LLMConfig, LLMModel, ProviderFactory

        logger.info("medication_normalization_started")
        perf_logger.start_operation("medication_normalization")

        start_time = datetime.now()

        try:
            # Get configuration and progress tracker
            disease_module = context.get("disease_module", "nsclc")
            raw_medications = context.get("all_medications", [])
            progress_tracker = context.get("progress_tracker")
            phase_name = context.get("current_phase_name", self.name)

            # Check if we should use LLM
            enable_llm = context.get("enable_llm_classification", True)
            if not enable_llm:
                logger.info("medication_normalization_disabled")
                return PhaseResult(
                    phase_name=self.name,
                    status=PhaseStatus.SKIPPED,
                    start_time=start_time,
                    end_time=datetime.now(),
                )

            # Phase 1: Pre-filtering and deduplication (0-20%)
            if progress_tracker:
                progress_tracker.update_phase_progress(
                    phase_name, 5.0, "Pre-filtering medications"
                )

            logger.info("medication_normalization_phase1_prefiltering")
            filtered_medications = self._prefilter_medications(raw_medications)
            logger.info(
                "prefiltering_completed",
                original_count=len(raw_medications),
                filtered_count=len(filtered_medications),
            )

            if progress_tracker:
                progress_tracker.update_phase_progress(
                    phase_name,
                    20.0,
                    f"Pre-filtered to {len(filtered_medications)} medications",
                )

            if not filtered_medications:
                logger.info("no_medications_after_filtering")
                return PhaseResult(
                    phase_name=self.name,
                    status=PhaseStatus.COMPLETED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    output_data={"normalized_medications": {}, "total_processed": 0},
                )

            # Initialize LLM service
            provider_type = context.get("llm_provider", "claude_cli")
            config = LLMConfig(
                model=LLMModel.CLAUDE_3_SONNET,
                temperature=0.0,
                max_tokens=2048,
                timeout=120,
                retry_attempts=2,
            )

            provider = ProviderFactory.create(provider_type, config)
            if not await provider.is_available():
                logger.warning("llm_provider_not_available", provider=provider_type)
                return PhaseResult(
                    phase_name=self.name,
                    status=PhaseStatus.SKIPPED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    error="LLM provider not available",
                )

            llm_service = LLMService(provider)

            # Phase 2: Batch normalization with parallel processing
            # Phase 2: Batch processing (20-90%)
            if progress_tracker:
                progress_tracker.update_phase_progress(
                    phase_name,
                    25.0,
                    f"Starting LLM normalization of {len(filtered_medications)} medications",
                )

            logger.info("medication_normalization_phase2_batch_processing")

            normalized_drugs = await self._batch_normalize_medications(
                llm_service, filtered_medications, context, progress_tracker, phase_name
            )

            if progress_tracker:
                progress_tracker.update_phase_progress(
                    phase_name,
                    90.0,
                    f"Completed LLM processing, found {len(normalized_drugs)} valid drugs",
                )

            # Phase 3: Post-processing and validation (90-100%)
            if progress_tracker:
                progress_tracker.update_phase_progress(
                    phase_name, 95.0, "Post-processing and validating results"
                )

            logger.info("medication_normalization_phase3_postprocessing")
            final_results = self._postprocess_normalized_medications(normalized_drugs)

            # Complete the phase
            if progress_tracker:
                progress_tracker.update_phase_progress(
                    phase_name,
                    100.0,
                    f"Normalization complete - {len(final_results)} valid drugs found",
                )

            # Store results
            context["normalized_medications"] = final_results
            context["normalization_stats"] = {
                "original_count": len(raw_medications),
                "filtered_count": len(filtered_medications),
                "batch_processed": len(normalized_drugs),
                "final_valid_drugs": len(final_results),
            }

            duration = perf_logger.end_operation("medication_normalization")

            logger.info(
                "medication_normalization_completed",
                original_count=len(raw_medications),
                filtered_count=len(filtered_medications),
                valid_drugs=len(final_results),
                duration=duration,
            )

            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output_data={
                    "normalized_medications": final_results,
                    "original_count": len(raw_medications),
                    "filtered_count": len(filtered_medications),
                    "valid_oncology_drugs": len(final_results),
                },
                metrics={
                    "normalization_time_seconds": duration,
                    "medications_original": len(raw_medications),
                    "medications_filtered": len(filtered_medications),
                    "valid_drugs_found": len(final_results),
                    "filtering_efficiency": len(filtered_medications)
                    / max(1, len(raw_medications)),
                },
            )

        except Exception as e:
            logger.error("medication_normalization_failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                status=PhaseStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e),
            )

    def _prefilter_medications(self, medications: List[str]) -> List[str]:
        """Phase 1: Pre-filter medications to reduce LLM calls."""
        import re

        filtered = []
        seen = set()

        # Common non-medication patterns
        non_med_patterns = [
            r"^(test|blood|lab|procedure|surgery)\b",
            r"\b(error|unknown|n/?a|nil|none|missing)\b",
            r"^[0-9]+$",  # Pure numbers
            r"^[^a-zA-Z]*$",  # No letters
        ]

        for med in medications:
            if not med or len(med.strip()) < 3:
                continue

            med_clean = med.strip().lower()

            # Skip duplicates
            if med_clean in seen:
                continue

            # Skip obvious non-medications
            is_non_med = any(
                re.search(pattern, med_clean, re.IGNORECASE)
                for pattern in non_med_patterns
            )
            if is_non_med:
                continue

            filtered.append(med.strip())
            seen.add(med_clean)

        return filtered[:100]  # Limit to first 100 unique meds for performance

    async def _batch_normalize_medications(
        self,
        llm_service: "LLMService",
        medications: List[str],
        context: Dict[str, Any],
        progress_tracker=None,
        phase_name: str = "medication_normalization",
    ) -> Dict[str, Any]:
        """Phase 2: Batch process medications with parallel execution."""
        import asyncio
        import json

        # Get disease module from context
        disease_module = context.get("disease_module", "nsclc")

        batch_size = (
            8  # Process 8 medications per batch (smaller batches = faster completion)
        )
        max_concurrent = 6  # Run 6 batches concurrently (more parallelism)
        normalized_results = {}
        total_processed = 0

        # Create batches
        batches = [
            medications[i : i + batch_size]
            for i in range(0, len(medications), batch_size)
        ]
        logger.info(
            "batch_normalization_started",
            total_batches=len(batches),
            batch_size=batch_size,
        )

        # Process batches with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(
            batch_medications: List[str], batch_num: int, disease_module_name: str
        ) -> Dict[str, Any]:
            async with semaphore:
                # Update progress tracker
                progress_tracker = context.get("progress_tracker")
                if progress_tracker:
                    progress_percent = (batch_num / len(batches)) * 100
                    progress_tracker.update_phase_progress(
                        "medication_normalization",
                        progress_percent,
                        f"Processing batch {batch_num + 1}/{len(batches)} ({len(batch_medications)} medications)",
                    )

                logger.info(
                    "processing_batch",
                    batch_num=batch_num,
                    medications=len(batch_medications),
                    medication_list=batch_medications[:3] + ["..."]
                    if len(batch_medications) > 3
                    else batch_medications,
                )
                batch_results = {}

                # Prepare batch prompts
                batch_prompts = []
                for med in batch_medications:
                    # Get disease display name from the disease module
                    from ..diseases import disease_registry

                    disease_module_instance = disease_registry.get_module(
                        disease_module_name
                    )
                    disease_display_name = (
                        disease_module_instance.display_name
                        if disease_module_instance
                        else disease_module_name
                    )

                    system, prompt = llm_service.prompt_manager.format_prompt(
                        "normalization",
                        medication=med,
                        disease=disease_display_name,
                    )
                    batch_prompts.append((prompt, system))

                try:
                    # Log which medications are being sent to LLM
                    logger.info(
                        "llm_batch_request_started",
                        batch_num=batch_num,
                        medications=batch_medications,
                        concurrent_calls=min(3, len(batch_medications)),
                    )

                    # Process batch concurrently (3 concurrent calls per batch)
                    batch_context = {
                        "batch_num": batch_num,
                        "medications": batch_medications,
                    }
                    responses = await llm_service.batch_generate(
                        batch_prompts,
                        use_cache=True,
                        max_concurrent=3,
                        context=batch_context,
                    )

                    logger.info(
                        "llm_batch_request_completed",
                        batch_num=batch_num,
                        responses_received=len(responses),
                    )

                    # Process responses
                    for med, response in zip(batch_medications, responses):
                        try:
                            # Try to parse as JSON first
                            response_text = response.content.strip()
                            result = None

                            try:
                                result = json.loads(response_text)
                            except json.JSONDecodeError:
                                # If JSON parsing fails, LLM likely returned explanatory text for non-drug
                                # Try to extract JSON from the response if it contains it
                                import re

                                json_match = re.search(
                                    r'\{[^{}]*"input_medication"[^{}]*\}',
                                    response_text,
                                    re.DOTALL,
                                )
                                if json_match:
                                    try:
                                        result = json.loads(json_match.group())
                                    except:
                                        pass

                                if not result:
                                    # Create a default "invalid" result for non-JSON responses
                                    result = {
                                        "input_medication": med,
                                        "generic_name": "",
                                        "brand_names": [],
                                        "is_oncology_drug": False,
                                        "is_valid_medication": False,
                                        "confidence": 0.0,
                                        "reasoning": "LLM returned non-JSON response, likely not a medication",
                                    }

                            # Debug logging with full LLM response details
                            logger.info(
                                "llm_normalization_response",
                                medication=med,
                                batch_num=batch_num,
                                raw_llm_response=response_text[:500] + "..."
                                if len(response_text) > 500
                                else response_text,
                                is_valid=result.get("is_valid_medication"),
                                is_disease_specific=result.get(
                                    "is_disease_specific_drug",
                                    result.get("is_oncology_drug"),
                                ),  # Backward compatibility
                                confidence=result.get("confidence"),
                                response_type="json"
                                if "json" not in response_text.lower()
                                or "{" in response_text
                                else "text",
                            )

                            # Relaxed criteria: lower confidence threshold and fallback for known drugs
                            is_valid_med = result.get("is_valid_medication", False)
                            is_disease_specific = result.get(
                                "is_disease_specific_drug",
                                result.get("is_oncology_drug", False),
                            )  # Backward compatibility
                            confidence = result.get("confidence", 0)

                            # Accept if:
                            # 1. Valid medication AND disease-specific drug with confidence > 0.5, OR
                            # 2. Valid medication with very high confidence (> 0.8), OR
                            # 3. Known drug names from disease module (modular approach)
                            med_lower = med.lower()

                            # Get known drugs from current disease module (modular)
                            disease_name = context.get("disease_module", "nsclc")
                            from ..diseases import disease_registry

                            disease_module = disease_registry.get_module(disease_name)

                            is_known_disease_drug = False
                            if disease_module:
                                # Check if medication matches any keywords from any drug class
                                for drug_class_config in disease_module.drug_classes:
                                    for keyword in drug_class_config.keywords:
                                        if (
                                            keyword.lower() in med_lower
                                            or med_lower in keyword.lower()
                                        ):
                                            is_known_disease_drug = True
                                            break
                                    if is_known_disease_drug:
                                        break

                            should_include = (
                                (
                                    is_valid_med
                                    and is_disease_specific
                                    and confidence > 0.5
                                )
                                or (is_valid_med and confidence > 0.8)
                                or (
                                    is_known_disease_drug
                                    and is_valid_med
                                    and confidence > 0.3
                                )
                                or (
                                    is_known_disease_drug and confidence > 0.5
                                )  # Fallback for known drugs
                            )

                            if should_include:
                                generic_name = result.get("generic_name", "").lower()
                                brand_names = result.get("brand_names", [])
                                medication_type = result.get(
                                    "medication_type", "single"
                                )
                                components = result.get("components", [])
                                alternative_names = result.get("alternative_names", [])

                                # If no generic name provided but we have a known disease drug, use the input
                                if not generic_name and is_known_disease_drug:
                                    generic_name = med_lower

                                # Handle combination drugs and protocols
                                if (
                                    medication_type in ["combination", "protocol"]
                                    and components
                                ):
                                    # Process the combination/protocol itself
                                    if generic_name:
                                        key = f"taking_{generic_name.replace(' + ', '_').replace('+', '_').replace(' ', '_')}"

                                        # Include all variations: brand names + alternative names
                                        all_variants = list(
                                            set(
                                                [generic_name]
                                                + brand_names
                                                + alternative_names
                                            )
                                        )
                                        batch_results[key] = all_variants

                                        logger.info(
                                            "combination_drug_accepted",
                                            medication=med,
                                            medication_type=medication_type,
                                            generic_name=generic_name,
                                            components=components,
                                            variants=len(all_variants),
                                        )

                                    # Also create entries for individual components
                                    for component in components:
                                        component_lower = component.lower().strip()
                                        component_key = f"taking_{component_lower.replace(' ', '_')}"

                                        # Check if we already have this component
                                        if component_key not in batch_results:
                                            batch_results[component_key] = [
                                                component_lower
                                            ]
                                            logger.info(
                                                "component_drug_added",
                                                component=component_lower,
                                                from_combination=med,
                                            )
                                        else:
                                            # Add to existing entry if not already there
                                            if (
                                                component_lower
                                                not in batch_results[component_key]
                                            ):
                                                batch_results[component_key].append(
                                                    component_lower
                                                )

                                elif generic_name:  # Single drug
                                    key = f"taking_{generic_name.replace(' ', '_')}"

                                    # Log detailed acceptance
                                    logger.info(
                                        "medication_accepted_for_normalization",
                                        input_medication=med,
                                        generic_name=generic_name,
                                        brand_names=brand_names,
                                        alternative_names=alternative_names,
                                        is_valid=is_valid_med,
                                        is_disease_specific=is_disease_specific,
                                        is_known_disease_drug=is_known_disease_drug,
                                        confidence=confidence,
                                        reasoning=result.get("reasoning", ""),
                                        batch_num=batch_num,
                                    )

                                    # Include brand names and alternative names
                                    all_variants = list(
                                        set(
                                            [generic_name]
                                            + brand_names
                                            + alternative_names
                                        )
                                    )
                                    batch_results[key] = all_variants
                                    logger.info(
                                        "oncology_drug_accepted",
                                        medication=med,
                                        generic_name=generic_name,
                                        variants=len(all_variants),
                                        reason="json_parsed"
                                        if is_valid_med
                                        else "known_drug_fallback",
                                    )
                            else:
                                # Log detailed rejection with full context
                                logger.info(
                                    "medication_rejected_from_normalization",
                                    input_medication=med,
                                    is_valid=is_valid_med,
                                    is_disease_specific=is_disease_specific,
                                    is_known_disease_drug=is_known_disease_drug,
                                    confidence=confidence,
                                    generic_name=result.get("generic_name", ""),
                                    reasoning=result.get("reasoning", ""),
                                    rejection_reason="Failed criteria: valid AND disease_specific AND confidence>0.5, OR valid AND confidence>0.8, OR known_drug",
                                    batch_num=batch_num,
                                )

                        except Exception as e:
                            logger.warning(
                                "batch_medication_processing_failed",
                                medication=med,
                                error=str(e),
                            )
                            continue

                    # Update progress when batch completes
                    if progress_tracker:
                        completed_percent = ((batch_num + 1) / len(batches)) * 100
                        progress_tracker.update_phase_progress(
                            "medication_normalization",
                            completed_percent,
                            f"Completed batch {batch_num + 1}/{len(batches)} - {len(batch_results)} valid drugs found",
                        )

                    logger.info(
                        "batch_completed",
                        batch_num=batch_num,
                        valid_drugs=len(batch_results),
                    )

                    # Update progress for this batch (progress between 25% and 90%)
                    if progress_tracker:
                        batch_progress = 25.0 + (batch_num + 1) / len(batches) * 65.0
                        progress_tracker.update_phase_progress(
                            phase_name,
                            batch_progress,
                            f"Batch {batch_num + 1}/{len(batches)} complete - {len(batch_results)} valid drugs",
                        )

                    return batch_results

                except Exception as e:
                    logger.error(
                        "batch_processing_failed", batch_num=batch_num, error=str(e)
                    )
                    return {}

        # Execute all batches concurrently
        batch_tasks = [
            process_batch(batch, i, disease_module) for i, batch in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Combine results
        for result in batch_results:
            if isinstance(result, dict):
                normalized_results.update(result)
            else:
                logger.warning("batch_result_error", error=str(result))

        logger.info(
            "batch_normalization_completed", total_drugs=len(normalized_results)
        )
        return normalized_results

    def _postprocess_normalized_medications(
        self, normalized_drugs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 3: Post-process and validate normalized medications."""
        final_results = {}

        for key, variants in normalized_drugs.items():
            if not key or not variants:
                continue

            # Clean variants
            clean_variants = []
            for variant in variants:
                if variant and isinstance(variant, str) and len(variant.strip()) > 2:
                    clean_variants.append(variant.strip())

            if clean_variants:
                final_results[key] = list(set(clean_variants))  # Remove duplicates

        logger.info("postprocessing_completed", final_count=len(final_results))
        return final_results


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
                logger.warning("llm_provider_not_available", provider=provider_type)
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

            logger.info("classifying_medications", count=len(meds_to_classify))

            # Create progress callback for live updates
            def llm_progress_callback(current: int, total: int, message: str):
                logger.info(
                    "llm_classification_progress",
                    current=current,
                    total=total,
                    message=message,
                )
                # Update progress tracker if available
                if hasattr(context, "progress_tracker") and context["progress_tracker"]:
                    progress_percent = (current / total) * 100 if total > 0 else 0
                    context["progress_tracker"].update_phase_progress(
                        "llm_classification",
                        progress_percent,
                        f"{message} ({current}/{total})",
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
            if (
                context.get("normalized_medications")
                or context.get("llm_classifications")
                or context.get("classification_results")
            ):
                from ..output.exporters import ConmedsYAMLExporter

                conmeds_file = output_path / "conmeds_augmented.yml"
                exporter = ConmedsYAMLExporter()

                # Prepare data for export - prioritize normalized medications
                export_data = {}
                if context.get("normalized_medications"):
                    export_data["normalized_medications"] = context[
                        "normalized_medications"
                    ]
                    logger.info(
                        "exporting_normalized_medications",
                        count=len(context["normalized_medications"]),
                    )
                elif context.get("llm_classifications"):
                    # Fallback to old classification format
                    export_data.update(context["llm_classifications"])
                    logger.info("exporting_legacy_classifications")
                if context.get("classification_results"):
                    export_data["classification_results"] = context[
                        "classification_results"
                    ]

                # Export the conmeds.yml file
                result_path = exporter.export(
                    export_data,
                    conmeds_file,
                    title=f"NSCLC Medication Augmentation - {datetime.now().strftime('%Y-%m-%d')}",
                    conmeds_file=context.get(
                        "conmeds_file"
                    ),  # Pass base conmeds for augmentation
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
