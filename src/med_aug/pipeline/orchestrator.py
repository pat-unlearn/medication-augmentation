"""Main pipeline orchestrator for medication augmentation."""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .phases import (
    PipelinePhase,
    PhaseStatus,
    PhaseResult,
    DataIngestionPhase,
    ColumnAnalysisPhase,
    MedicationExtractionPhase,
    WebResearchPhase,
    ValidationPhase,
    LLMClassificationPhase,
    OutputGenerationPhase,
)
from .checkpoint import CheckpointManager, PipelineCheckpoint
from .progress import ProgressTracker, ProgressReport
from ..core.logging import get_logger, PerformanceLogger, ErrorLogger

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)
error_logger = ErrorLogger(logger)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    input_file: str
    output_path: str = "./output"
    conmeds_file: Optional[str] = None  # Existing conmeds.yml to augment
    disease_module: str = "nsclc"
    confidence_threshold: float = 0.5
    enable_web_research: bool = True
    enable_validation: bool = True
    enable_llm_classification: bool = True  # Default to True - core feature
    llm_provider: str = "claude_cli"  # Provider type for LLM
    enable_evaluation: bool = (
        False  # Enable comprehensive evaluation with LLM assistance
    )
    enable_checkpoints: bool = True
    checkpoint_interval: int = 1  # Checkpoint after every N phases
    max_retries: int = 3
    retry_delay: float = 1.0
    parallel_phases: bool = False
    display_progress: bool = True
    progress_mode: str = "rich"  # rich, simple, none

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_file": self.input_file,
            "output_path": self.output_path,
            "disease_module": self.disease_module,
            "confidence_threshold": self.confidence_threshold,
            "enable_web_research": self.enable_web_research,
            "enable_validation": self.enable_validation,
            "enable_llm_classification": self.enable_llm_classification,
            "llm_provider": self.llm_provider,
            "enable_evaluation": self.enable_evaluation,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_interval": self.checkpoint_interval,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "parallel_phases": self.parallel_phases,
            "display_progress": self.display_progress,
            "progress_mode": self.progress_mode,
        }


class PipelineOrchestrator:
    """Orchestrates the medication augmentation pipeline."""

    def __init__(self, config: PipelineConfig, pipeline_id: Optional[str] = None):
        """
        Initialize pipeline orchestrator.

        Args:
            config: Pipeline configuration
            pipeline_id: Optional pipeline ID (will generate if not provided)
        """
        self.config = config
        self.pipeline_id = pipeline_id or str(uuid.uuid4())[:8]
        self.context: Dict[str, Any] = {
            "pipeline_id": self.pipeline_id,
            "config": config.to_dict(),
            "start_time": datetime.now(),
        }

        # Initialize phases
        self.phases: List[PipelinePhase] = self._initialize_phases()
        self.phase_results: Dict[str, PhaseResult] = {}
        self.completed_phases: List[str] = []

        # Initialize managers
        self.checkpoint_manager = (
            CheckpointManager() if config.enable_checkpoints else None
        )
        self.progress_tracker = ProgressTracker(
            self.pipeline_id,
            [p.name for p in self.phases],
            config.progress_mode if config.display_progress else "none",
        )

        # Add context from config
        self.context.update(
            {
                "input_file": config.input_file,
                "output_path": config.output_path,
                "disease_module": config.disease_module,
                "confidence_threshold": config.confidence_threshold,
                "enable_llm_classification": config.enable_llm_classification,
                "llm_provider": config.llm_provider,
                "conmeds_file": config.conmeds_file,  # Pass base conmeds file for augmentation
                "progress_tracker": self.progress_tracker,  # Pass progress tracker to phases
            }
        )

        logger.info(
            "pipeline_initialized",
            pipeline_id=self.pipeline_id,
            phases=len(self.phases),
            config=config.to_dict(),
        )

    def _initialize_phases(self) -> List[PipelinePhase]:
        """Initialize pipeline phases based on configuration."""
        phases = [
            DataIngestionPhase(),
            ColumnAnalysisPhase(),
            MedicationExtractionPhase(),
        ]

        if self.config.enable_web_research:
            phases.append(WebResearchPhase())

        if self.config.enable_validation:
            phases.append(ValidationPhase())

        # Add LLM classification phase if enabled
        if self.config.enable_llm_classification:
            phases.append(LLMClassificationPhase())

        # Add evaluation phase if enabled (after LLM classification)
        if self.config.enable_evaluation:
            from .evaluation_phase import EvaluationPhase

            phases.append(EvaluationPhase())

        phases.append(OutputGenerationPhase())

        logger.debug(
            "phases_initialized", count=len(phases), names=[p.name for p in phases]
        )
        return phases

    async def run(self, resume_from: Optional[str] = None) -> ProgressReport:
        """
        Run the pipeline.

        Args:
            resume_from: Optional phase name to resume from

        Returns:
            Final progress report
        """
        logger.info(
            "pipeline_starting", pipeline_id=self.pipeline_id, resume_from=resume_from
        )
        perf_logger.start_operation(f"pipeline_{self.pipeline_id}")

        # Start progress display
        if self.config.display_progress:
            self.progress_tracker.start_display()

        try:
            # Resume from checkpoint if specified
            if resume_from:
                await self._resume_from_checkpoint(resume_from)

            # Execute phases
            for phase in self.phases:
                if phase.name in self.completed_phases:
                    logger.debug("phase_already_completed", phase=phase.name)
                    continue

                # Check dependencies
                if not phase.can_run(self.completed_phases):
                    logger.warning(
                        "phase_dependencies_not_met",
                        phase=phase.name,
                        dependencies=phase.dependencies,
                    )
                    self.progress_tracker.phase_skipped(
                        phase.name, "Dependencies not met"
                    )
                    continue

                # Execute phase
                result = await self._execute_phase(phase)
                self.phase_results[phase.name] = result

                if result.status == PhaseStatus.COMPLETED:
                    self.completed_phases.append(phase.name)

                    # Create checkpoint if enabled
                    if self.config.enable_checkpoints:
                        if (
                            len(self.completed_phases) % self.config.checkpoint_interval
                            == 0
                        ):
                            await self._create_checkpoint(phase.name)

                elif result.status == PhaseStatus.FAILED and phase.required:
                    logger.error(
                        "required_phase_failed", phase=phase.name, error=result.error
                    )
                    break

            # Mark pipeline as completed
            self.progress_tracker.pipeline_completed()

            duration = perf_logger.end_operation(
                f"pipeline_{self.pipeline_id}",
                phases_completed=len(self.completed_phases),
                phases_total=len(self.phases),
            )

            logger.info(
                "pipeline_completed",
                pipeline_id=self.pipeline_id,
                duration=duration,
                phases_completed=len(self.completed_phases),
            )

        except Exception as e:
            error_logger.log_error(e, f"pipeline_{self.pipeline_id}")
            self.progress_tracker.pipeline_completed()
            raise

        finally:
            # Stop progress display
            if self.config.display_progress:
                self.progress_tracker.stop_display()
                self.progress_tracker.print_summary()

        return self.progress_tracker.get_report()

    async def _execute_phase(self, phase: PipelinePhase) -> PhaseResult:
        """
        Execute a single phase with retry logic.

        Args:
            phase: Phase to execute

        Returns:
            Phase result
        """
        logger.info("executing_phase", phase=phase.name)
        self.progress_tracker.phase_started(phase.name)

        for attempt in range(self.config.max_retries):
            try:
                # Validate inputs
                if not await phase.validate_inputs(self.context):
                    logger.warning("phase_validation_failed", phase=phase.name)
                    self.progress_tracker.phase_skipped(phase.name, "Validation failed")
                    return PhaseResult(
                        phase_name=phase.name,
                        status=PhaseStatus.SKIPPED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error="Input validation failed",
                    )

                # Execute phase
                result = await phase.execute(self.context)

                if result.status == PhaseStatus.COMPLETED:
                    self.progress_tracker.phase_completed(phase.name, result)

                    # Cleanup
                    await phase.cleanup(self.context)

                    return result
                else:
                    raise Exception(result.error or "Phase execution failed")

            except Exception as e:
                logger.warning(
                    "phase_execution_attempt_failed",
                    phase=phase.name,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.progress_tracker.phase_failed(phase.name, str(e))
                    return PhaseResult(
                        phase_name=phase.name,
                        status=PhaseStatus.FAILED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error=str(e),
                    )

        # Should not reach here
        return PhaseResult(
            phase_name=phase.name,
            status=PhaseStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error="Max retries exceeded",
        )

    async def _create_checkpoint(self, current_phase: str):
        """Create a checkpoint."""
        if not self.checkpoint_manager:
            return

        checkpoint = PipelineCheckpoint(
            pipeline_id=self.pipeline_id,
            timestamp=datetime.now(),
            current_phase=current_phase,
            completed_phases=self.completed_phases.copy(),
            context=self.context,
            phase_results={
                name: result.to_dict() for name, result in self.phase_results.items()
            },
        )

        try:
            path = self.checkpoint_manager.save_checkpoint(checkpoint)
            logger.info(
                "checkpoint_created",
                pipeline_id=self.pipeline_id,
                phase=current_phase,
                file=str(path),
            )
        except Exception as e:
            logger.error(
                "checkpoint_creation_failed", pipeline_id=self.pipeline_id, error=str(e)
            )

    async def _resume_from_checkpoint(self, phase_name: str):
        """Resume from a checkpoint."""
        if not self.checkpoint_manager:
            logger.warning("checkpoints_not_enabled")
            return

        checkpoint = self.checkpoint_manager.load_checkpoint(self.pipeline_id)
        if checkpoint:
            # Find the phase index to resume from
            phase_names = [p.name for p in self.phases]
            if phase_name in phase_names:
                resume_index = phase_names.index(phase_name)
                self.completed_phases = phase_names[:resume_index]

                # Restore context
                self.context.update(checkpoint.context)

                # Restore phase results
                for name, result_dict in checkpoint.phase_results.items():
                    if name in self.completed_phases:
                        # Reconstruct PhaseResult from dict
                        self.phase_results[name] = PhaseResult(
                            phase_name=result_dict["phase_name"],
                            status=PhaseStatus(result_dict["status"]),
                            start_time=datetime.fromisoformat(
                                result_dict["start_time"]
                            ),
                            end_time=(
                                datetime.fromisoformat(result_dict["end_time"])
                                if result_dict.get("end_time")
                                else None
                            ),
                            output_data=result_dict.get("output_data", {}),
                            error=result_dict.get("error"),
                            metrics=result_dict.get("metrics", {}),
                        )

                logger.info(
                    "resumed_from_checkpoint",
                    pipeline_id=self.pipeline_id,
                    resume_phase=phase_name,
                    completed_phases=len(self.completed_phases),
                )
            else:
                logger.error("invalid_resume_phase", phase=phase_name)

    def get_phase_result(self, phase_name: str) -> Optional[PhaseResult]:
        """Get result for a specific phase."""
        return self.phase_results.get(phase_name)

    def get_context(self) -> Dict[str, Any]:
        """Get current pipeline context."""
        return self.context.copy()

    def get_artifacts(self) -> List[str]:
        """Get all artifacts produced by the pipeline."""
        artifacts = []
        for result in self.phase_results.values():
            artifacts.extend(result.artifacts)
        return artifacts

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all phases."""
        metrics = {}
        for result in self.phase_results.values():
            metrics.update(result.metrics)
        return metrics
