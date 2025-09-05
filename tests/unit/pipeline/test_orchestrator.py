"""Unit tests for pipeline orchestrator."""

import pytest
from pathlib import Path
import tempfile
import pandas as pd
from datetime import datetime

from med_aug.pipeline import PipelineOrchestrator, PipelineConfig
from med_aug.pipeline.phases import PhaseStatus


class TestPipelineOrchestrator:
    """Test pipeline orchestrator functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_data_file(self, temp_dir):
        """Create sample data file for testing."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003", "P004", "P005"],
                "medication": [
                    "pembrolizumab",
                    "osimertinib",
                    "carboplatin",
                    "nivolumab",
                    "paclitaxel",
                ],
                "dose": ["200mg", "80mg", "300mg", "240mg", "175mg"],
                "diagnosis": ["NSCLC", "NSCLC", "NSCLC", "NSCLC", "NSCLC"],
            }
        )

        file_path = temp_dir / "test_data.csv"
        df.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def pipeline_config(self, sample_data_file, temp_dir):
        """Create pipeline configuration."""
        return PipelineConfig(
            input_file=str(sample_data_file),
            output_path=str(temp_dir / "output"),
            disease_module="nsclc",
            confidence_threshold=0.5,
            enable_web_research=False,  # Disable for testing
            enable_validation=True,
            enable_checkpoints=True,
            display_progress=False,  # Disable for testing
        )

    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline orchestrator initialization."""
        orchestrator = PipelineOrchestrator(pipeline_config)

        assert orchestrator.pipeline_id is not None
        assert len(orchestrator.phases) > 0
        assert orchestrator.config == pipeline_config
        assert len(orchestrator.completed_phases) == 0

    def test_pipeline_config_to_dict(self, pipeline_config):
        """Test pipeline configuration serialization."""
        config_dict = pipeline_config.to_dict()

        assert "input_file" in config_dict
        assert "output_path" in config_dict
        assert "disease_module" in config_dict
        assert config_dict["confidence_threshold"] == 0.5
        assert config_dict["enable_web_research"] is False

    @pytest.mark.asyncio
    async def test_pipeline_execution(self, pipeline_config):
        """Test basic pipeline execution."""
        orchestrator = PipelineOrchestrator(pipeline_config)

        # Run pipeline
        report = await orchestrator.run()

        assert report is not None
        assert report.pipeline_id == orchestrator.pipeline_id
        assert report.total_phases == len(orchestrator.phases)
        assert report.completed_phases > 0

    @pytest.mark.asyncio
    async def test_data_ingestion_phase(self, pipeline_config):
        """Test data ingestion phase execution."""
        from med_aug.pipeline.phases import DataIngestionPhase

        orchestrator = PipelineOrchestrator(pipeline_config)
        phase = DataIngestionPhase()

        # Execute phase
        result = await phase.execute(orchestrator.context)

        assert result.status == PhaseStatus.COMPLETED
        assert "dataframe" in orchestrator.context
        assert orchestrator.context["original_shape"] == (5, 4)

    @pytest.mark.asyncio
    async def test_column_analysis_phase(self, pipeline_config):
        """Test column analysis phase."""
        from med_aug.pipeline.phases import DataIngestionPhase, ColumnAnalysisPhase

        orchestrator = PipelineOrchestrator(pipeline_config)

        # First run data ingestion
        ingestion = DataIngestionPhase()
        await ingestion.execute(orchestrator.context)

        # Then run column analysis
        analysis = ColumnAnalysisPhase()
        result = await analysis.execute(orchestrator.context)

        assert result.status == PhaseStatus.COMPLETED
        assert "medication_columns" in orchestrator.context
        assert len(orchestrator.context["medication_columns"]) > 0

    @pytest.mark.asyncio
    async def test_medication_extraction_phase(self, pipeline_config):
        """Test medication extraction phase."""
        from med_aug.pipeline.phases import (
            DataIngestionPhase,
            ColumnAnalysisPhase,
            MedicationExtractionPhase,
        )

        orchestrator = PipelineOrchestrator(pipeline_config)

        # Run prerequisite phases
        await DataIngestionPhase().execute(orchestrator.context)
        await ColumnAnalysisPhase().execute(orchestrator.context)

        # Run extraction
        extraction = MedicationExtractionPhase()
        result = await extraction.execute(orchestrator.context)

        assert result.status == PhaseStatus.COMPLETED
        assert "all_medications" in orchestrator.context
        assert len(orchestrator.context["all_medications"]) > 0

    @pytest.mark.asyncio
    async def test_validation_phase(self, pipeline_config):
        """Test validation phase."""
        from med_aug.pipeline.phases import ValidationPhase

        orchestrator = PipelineOrchestrator(pipeline_config)
        orchestrator.context["all_medications"] = [
            "pembrolizumab",
            "osimertinib",
            "unknown_drug",
        ]

        validation = ValidationPhase()
        result = await validation.execute(orchestrator.context)

        assert result.status == PhaseStatus.COMPLETED
        assert "validation_results" in orchestrator.context

        # Check validation results
        results = orchestrator.context["validation_results"]
        assert "pembrolizumab" in results
        assert results["pembrolizumab"]["valid"] is True

    @pytest.mark.asyncio
    async def test_output_generation_phase(self, pipeline_config, temp_dir):
        """Test output generation phase."""
        from med_aug.pipeline.phases import OutputGenerationPhase

        orchestrator = PipelineOrchestrator(pipeline_config)
        orchestrator.context["all_medications"] = ["pembrolizumab", "osimertinib"]
        orchestrator.context["output_path"] = str(temp_dir / "test_output")

        output = OutputGenerationPhase()
        result = await output.execute(orchestrator.context)

        assert result.status == PhaseStatus.COMPLETED
        assert len(result.artifacts) > 0

        # Check files were created
        output_path = Path(orchestrator.context["output_path"])
        assert output_path.exists()
        assert (output_path / "extracted_medications.json").exists()

    def test_phase_dependencies(self, pipeline_config):
        """Test phase dependency checking."""
        from med_aug.pipeline.phases import ColumnAnalysisPhase

        orchestrator = PipelineOrchestrator(pipeline_config)
        phase = ColumnAnalysisPhase()

        # Should not be able to run without dependencies
        assert not phase.can_run([])

        # Should be able to run with dependencies met
        assert phase.can_run(["data_ingestion"])

    @pytest.mark.asyncio
    async def test_phase_retry_logic(self, pipeline_config):
        """Test phase retry on failure."""
        from med_aug.pipeline.phases import PipelinePhase, PhaseResult

        class FailingPhase(PipelinePhase):
            def __init__(self):
                super().__init__("failing_phase")
                self.attempt_count = 0

            async def validate_inputs(self, context):
                return True

            async def execute(self, context):
                self.attempt_count += 1
                if self.attempt_count < 2:
                    raise Exception("Simulated failure")
                return PhaseResult(
                    phase_name=self.name,
                    status=PhaseStatus.COMPLETED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                )

        orchestrator = PipelineOrchestrator(pipeline_config)
        phase = FailingPhase()

        result = await orchestrator._execute_phase(phase)

        assert result.status == PhaseStatus.COMPLETED
        assert phase.attempt_count == 2

    def test_get_phase_result(self, pipeline_config):
        """Test getting phase results."""
        from med_aug.pipeline.phases import PhaseResult

        orchestrator = PipelineOrchestrator(pipeline_config)

        # Add a mock result
        result = PhaseResult(
            phase_name="test_phase",
            status=PhaseStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        orchestrator.phase_results["test_phase"] = result

        retrieved = orchestrator.get_phase_result("test_phase")
        assert retrieved == result

        # Non-existent phase
        assert orchestrator.get_phase_result("non_existent") is None

    def test_get_artifacts(self, pipeline_config):
        """Test getting pipeline artifacts."""
        from med_aug.pipeline.phases import PhaseResult

        orchestrator = PipelineOrchestrator(pipeline_config)

        # Add results with artifacts
        result1 = PhaseResult(
            phase_name="phase1",
            status=PhaseStatus.COMPLETED,
            start_time=datetime.now(),
            artifacts=["file1.json", "file2.csv"],
        )
        result2 = PhaseResult(
            phase_name="phase2",
            status=PhaseStatus.COMPLETED,
            start_time=datetime.now(),
            artifacts=["file3.txt"],
        )

        orchestrator.phase_results["phase1"] = result1
        orchestrator.phase_results["phase2"] = result2

        artifacts = orchestrator.get_artifacts()
        assert len(artifacts) == 3
        assert "file1.json" in artifacts
        assert "file3.txt" in artifacts

    def test_get_metrics(self, pipeline_config):
        """Test getting aggregated metrics."""
        from med_aug.pipeline.phases import PhaseResult

        orchestrator = PipelineOrchestrator(pipeline_config)

        # Add results with metrics
        result1 = PhaseResult(
            phase_name="phase1",
            status=PhaseStatus.COMPLETED,
            start_time=datetime.now(),
            metrics={"metric1": 100, "metric2": 200},
        )
        result2 = PhaseResult(
            phase_name="phase2",
            status=PhaseStatus.COMPLETED,
            start_time=datetime.now(),
            metrics={"metric3": 300},
        )

        orchestrator.phase_results["phase1"] = result1
        orchestrator.phase_results["phase2"] = result2

        metrics = orchestrator.get_metrics()
        assert metrics["metric1"] == 100
        assert metrics["metric2"] == 200
        assert metrics["metric3"] == 300
