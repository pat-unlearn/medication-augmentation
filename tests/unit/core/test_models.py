"""Unit tests for core data models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from med_aug.core.models import (
    MedicationType,
    ConfidenceLevel,
    Medication,
    DrugClass,
    ColumnAnalysisResult,
    AugmentationResult,
    MedicationClassification,
    WebResearchResult,
    PipelineState,
)


class TestMedicationType:
    """Test MedicationType enum."""

    def test_medication_types_exist(self):
        """Test that all medication types are defined."""
        assert MedicationType.BRAND == "brand"
        assert MedicationType.GENERIC == "generic"
        assert MedicationType.CLINICAL_TRIAL == "clinical_trial"
        assert MedicationType.ABBREVIATION == "abbreviation"
        assert MedicationType.COMBINATION == "combination"
        assert MedicationType.UNKNOWN == "unknown"

    def test_medication_type_values(self):
        """Test medication type string values."""
        assert MedicationType.BRAND.value == "brand"
        assert MedicationType.GENERIC.value == "generic"


class TestConfidenceLevel:
    """Test ConfidenceLevel enum."""

    def test_confidence_levels_exist(self):
        """Test that all confidence levels are defined."""
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"


class TestMedication:
    """Test Medication dataclass."""

    def test_medication_creation(self):
        """Test creating a medication instance."""
        med = Medication(
            name="Pembrolizumab",
            type=MedicationType.GENERIC,
            confidence=0.95,
            source="FDA",
            metadata={"approval_year": 2014},
        )

        assert med.name == "Pembrolizumab"
        assert med.type == MedicationType.GENERIC
        assert med.confidence == 0.95
        assert med.source == "FDA"
        assert med.metadata["approval_year"] == 2014
        assert isinstance(med.discovered_at, datetime)

    def test_confidence_level_property(self):
        """Test confidence level categorization."""
        high_conf = Medication("Drug1", MedicationType.BRAND, 0.95, "FDA")
        assert high_conf.confidence_level == ConfidenceLevel.HIGH

        med_conf = Medication("Drug2", MedicationType.BRAND, 0.75, "FDA")
        assert med_conf.confidence_level == ConfidenceLevel.MEDIUM

        low_conf = Medication("Drug3", MedicationType.BRAND, 0.5, "FDA")
        assert low_conf.confidence_level == ConfidenceLevel.LOW

    def test_confidence_level_boundaries(self):
        """Test confidence level boundary conditions."""
        # Test boundary at 0.9
        boundary_high = Medication("Drug", MedicationType.BRAND, 0.9, "FDA")
        assert boundary_high.confidence_level == ConfidenceLevel.HIGH

        just_below_high = Medication("Drug", MedicationType.BRAND, 0.89, "FDA")
        assert just_below_high.confidence_level == ConfidenceLevel.MEDIUM

        # Test boundary at 0.7
        boundary_med = Medication("Drug", MedicationType.BRAND, 0.7, "FDA")
        assert boundary_med.confidence_level == ConfidenceLevel.MEDIUM

        just_below_med = Medication("Drug", MedicationType.BRAND, 0.69, "FDA")
        assert just_below_med.confidence_level == ConfidenceLevel.LOW

    def test_medication_to_dict(self):
        """Test converting medication to dictionary."""
        med = Medication(
            name="Keytruda",
            type=MedicationType.BRAND,
            confidence=0.99,
            source="Clinical",
            metadata={"generic": "pembrolizumab"},
        )

        med_dict = med.to_dict()

        assert med_dict["name"] == "Keytruda"
        assert med_dict["type"] == "brand"
        assert med_dict["confidence"] == 0.99
        assert med_dict["source"] == "Clinical"
        assert med_dict["metadata"]["generic"] == "pembrolizumab"
        assert "discovered_at" in med_dict


class TestDrugClass:
    """Test DrugClass dataclass."""

    @pytest.fixture
    def sample_medications(self):
        """Create sample medications for testing."""
        return [
            Medication("pembrolizumab", MedicationType.GENERIC, 0.95, "FDA"),
            Medication("Keytruda", MedicationType.BRAND, 0.98, "FDA"),
            Medication("nivolumab", MedicationType.GENERIC, 0.92, "Clinical"),
        ]

    def test_drug_class_creation(self, sample_medications):
        """Test creating a drug class instance."""
        drug_class = DrugClass(
            name="immunotherapy",
            taking_variable="taking_pembrolizumab",
            current_medications=sample_medications,
            category="immunotherapy",
            disease="nsclc",
        )

        assert drug_class.name == "immunotherapy"
        assert drug_class.taking_variable == "taking_pembrolizumab"
        assert len(drug_class.current_medications) == 3
        assert drug_class.category == "immunotherapy"
        assert drug_class.disease == "nsclc"

    def test_add_medication(self, sample_medications):
        """Test adding medication to drug class."""
        drug_class = DrugClass(
            name="immunotherapy",
            taking_variable="taking_immunotherapy",
            current_medications=sample_medications[:2],
            category="immunotherapy",
            disease="nsclc",
        )

        new_med = Medication("atezolizumab", MedicationType.GENERIC, 0.9, "FDA")
        updated_class = drug_class.add_medication(new_med)

        # Original should be unchanged (immutable)
        assert len(drug_class.current_medications) == 2

        # New instance should have the added medication
        assert len(updated_class.current_medications) == 3
        assert updated_class.current_medications[-1].name == "atezolizumab"

    def test_get_medication_names(self, sample_medications):
        """Test getting medication names from drug class."""
        drug_class = DrugClass(
            name="immunotherapy",
            taking_variable="taking_immunotherapy",
            current_medications=sample_medications,
            category="immunotherapy",
            disease="nsclc",
        )

        names = drug_class.get_medication_names()
        assert names == ["pembrolizumab", "Keytruda", "nivolumab"]

    def test_get_high_confidence_medications(self, sample_medications):
        """Test filtering high confidence medications."""
        # Add a low confidence medication
        meds = sample_medications + [
            Medication("experimental", MedicationType.CLINICAL_TRIAL, 0.5, "Trial")
        ]

        drug_class = DrugClass(
            name="immunotherapy",
            taking_variable="taking_immunotherapy",
            current_medications=meds,
            category="immunotherapy",
            disease="nsclc",
        )

        high_conf = drug_class.get_high_confidence_medications()
        assert len(high_conf) == 3  # Only the high confidence ones
        assert all(med.confidence >= 0.9 for med in high_conf)


class TestColumnAnalysisResult:
    """Test ColumnAnalysisResult dataclass."""

    def test_column_analysis_creation(self):
        """Test creating column analysis result."""
        result = ColumnAnalysisResult(
            column="AGENT",
            confidence=0.85,
            total_count=1000,
            unique_count=150,
            sample_medications=["pembrolizumab", "nivolumab", "carboplatin"],
            reasoning="High medication pattern match",
        )

        assert result.column == "AGENT"
        assert result.confidence == 0.85
        assert result.total_count == 1000
        assert result.unique_count == 150
        assert len(result.sample_medications) == 3

    def test_is_likely_medication_column(self):
        """Test medication column likelihood check."""
        likely = ColumnAnalysisResult(
            column="DRUG",
            confidence=0.75,
            total_count=100,
            unique_count=50,
            sample_medications=[],
            reasoning="Test",
        )
        assert likely.is_likely_medication_column is True

        unlikely = ColumnAnalysisResult(
            column="PATIENT_ID",
            confidence=0.3,
            total_count=100,
            unique_count=100,
            sample_medications=[],
            reasoning="Test",
        )
        assert unlikely.is_likely_medication_column is False

    def test_column_analysis_to_dict(self):
        """Test converting column analysis to dictionary."""
        result = ColumnAnalysisResult(
            column="MEDICATION",
            confidence=0.9,
            total_count=500,
            unique_count=75,
            sample_medications=["drug1", "drug2"],
            reasoning="Strong indicators",
        )

        result_dict = result.to_dict()

        assert result_dict["column"] == "MEDICATION"
        assert result_dict["confidence"] == 0.9
        assert result_dict["is_likely"] is True
        assert len(result_dict["sample_medications"]) == 2


class TestAugmentationResult:
    """Test AugmentationResult dataclass."""

    @pytest.fixture
    def sample_new_medications(self):
        """Create sample new medications."""
        return [
            Medication("new_drug1", MedicationType.GENERIC, 0.9, "Web"),
            Medication("new_drug2", MedicationType.BRAND, 0.85, "Web"),
        ]

    def test_augmentation_result_creation(self, sample_new_medications):
        """Test creating augmentation result."""
        result = AugmentationResult(
            original_count=50,
            augmented_count=75,
            new_medications=sample_new_medications,
            improvement_percentage=50.0,
            processing_time=120.5,
            quality_score=0.85,
            disease="nsclc",
        )

        assert result.original_count == 50
        assert result.augmented_count == 75
        assert result.medications_added == 25
        assert result.improvement_percentage == 50.0
        assert result.disease == "nsclc"

    def test_was_successful(self, sample_new_medications):
        """Test success determination."""
        successful = AugmentationResult(
            original_count=50,
            augmented_count=75,
            new_medications=sample_new_medications,
            improvement_percentage=50.0,
            processing_time=120.5,
            quality_score=0.85,
            disease="nsclc",
        )
        assert successful.was_successful is True

        # No improvement
        no_improvement = AugmentationResult(
            original_count=50,
            augmented_count=50,
            new_medications=[],
            improvement_percentage=0.0,
            processing_time=120.5,
            quality_score=0.85,
            disease="nsclc",
        )
        assert no_improvement.was_successful is False

        # Low quality
        low_quality = AugmentationResult(
            original_count=50,
            augmented_count=75,
            new_medications=sample_new_medications,
            improvement_percentage=50.0,
            processing_time=120.5,
            quality_score=0.5,
            disease="nsclc",
        )
        assert low_quality.was_successful is False


class TestMedicationClassification:
    """Test MedicationClassification Pydantic model."""

    def test_valid_classification(self):
        """Test creating valid classification."""
        classification = MedicationClassification(
            medication_name="pembrolizumab",
            drug_class="immunotherapy",
            confidence=0.95,
            reasoning="PD-1 inhibitor commonly used in NSCLC",
            alternative_classes=["checkpoint_inhibitor"],
        )

        assert classification.medication_name == "pembrolizumab"
        assert classification.drug_class == "immunotherapy"
        assert classification.confidence == 0.95
        assert len(classification.alternative_classes) == 1

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        valid = MedicationClassification(
            medication_name="drug", drug_class="class", confidence=0.5, reasoning="test"
        )
        assert valid.confidence == 0.5

        # Invalid confidence - too high
        with pytest.raises(ValidationError) as exc_info:
            MedicationClassification(
                medication_name="drug",
                drug_class="class",
                confidence=1.5,
                reasoning="test",
            )
        assert "confidence" in str(exc_info.value).lower()

        # Invalid confidence - negative
        with pytest.raises(ValidationError) as exc_info:
            MedicationClassification(
                medication_name="drug",
                drug_class="class",
                confidence=-0.1,
                reasoning="test",
            )
        assert "confidence" in str(exc_info.value).lower()

    def test_needs_review_property(self):
        """Test needs_review property logic."""
        # High confidence, no alternatives - no review needed
        no_review = MedicationClassification(
            medication_name="drug", drug_class="class", confidence=0.9, reasoning="test"
        )
        assert no_review.needs_review is False

        # Low confidence - needs review
        low_conf = MedicationClassification(
            medication_name="drug", drug_class="class", confidence=0.7, reasoning="test"
        )
        assert low_conf.needs_review is True

        # Has alternatives - needs review
        has_alternatives = MedicationClassification(
            medication_name="drug",
            drug_class="class",
            confidence=0.9,
            reasoning="test",
            alternative_classes=["other_class"],
        )
        assert has_alternatives.needs_review is True


class TestWebResearchResult:
    """Test WebResearchResult Pydantic model."""

    def test_web_research_creation(self):
        """Test creating web research result."""
        result = WebResearchResult(
            medication_name="osimertinib",
            generic_names=["osimertinib"],
            brand_names=["Tagrisso"],
            drug_class_hints=["EGFR inhibitor", "targeted therapy"],
            mechanism_of_action="3rd generation EGFR-TKI",
            fda_approval_date="2015-11-13",
            clinical_trials=["NCT02296125", "NCT02511106"],
            sources=["FDA", "ClinicalTrials.gov"],
        )

        assert result.medication_name == "osimertinib"
        assert "Tagrisso" in result.brand_names
        assert len(result.drug_class_hints) == 2
        assert result.fda_approval_date == "2015-11-13"

    def test_has_sufficient_data(self):
        """Test sufficient data check."""
        # Has data
        with_data = WebResearchResult(
            medication_name="drug", generic_names=["generic"], sources=["FDA"]
        )
        assert with_data.has_sufficient_data is True

        # No useful data
        no_data = WebResearchResult(medication_name="unknown", sources=["Web"])
        assert no_data.has_sufficient_data is False

    def test_get_all_names(self):
        """Test getting all medication names."""
        result = WebResearchResult(
            medication_name="drug",
            generic_names=["generic1", "generic2"],
            brand_names=["brand1", "brand2", "generic1"],  # Duplicate
            sources=["FDA"],
        )

        all_names = result.get_all_names()
        assert len(all_names) == 4  # Duplicates removed
        assert "generic1" in all_names
        assert "brand1" in all_names


class TestPipelineState:
    """Test PipelineState Pydantic model."""

    def test_pipeline_state_creation(self):
        """Test creating pipeline state."""
        state = PipelineState(
            disease="nsclc",
            input_file="/path/to/input.csv",
            output_file="/path/to/output.yml",
        )

        assert state.disease == "nsclc"
        assert state.input_file == "/path/to/input.csv"
        assert state.current_phase == "initialization"
        assert len(state.phases_completed) == 0
        assert state.medications_processed == 0

    def test_update_phase(self):
        """Test phase update logic."""
        state = PipelineState(
            disease="nsclc", input_file="input.csv", output_file="output.yml"
        )

        # Update from initialization
        state.update_phase("analysis")
        assert state.current_phase == "analysis"
        assert (
            "initialization" not in state.phases_completed
        )  # initialization not added

        # Update to next phase
        state.update_phase("extraction")
        assert state.current_phase == "extraction"
        assert "analysis" in state.phases_completed

        # Update again
        state.update_phase("classification")
        assert state.current_phase == "classification"
        assert len(state.phases_completed) == 2
        assert "extraction" in state.phases_completed

    def test_add_error(self):
        """Test error tracking."""
        state = PipelineState(
            disease="nsclc", input_file="input.csv", output_file="output.yml"
        )

        assert state.has_errors is False

        state.add_error("Failed to read file")
        assert state.has_errors is True
        assert len(state.errors) == 1
        assert "Failed to read file" in state.errors[0]
        assert "2025" in state.errors[0]  # Contains timestamp

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        import time

        state = PipelineState(
            disease="nsclc", input_file="input.csv", output_file="output.yml"
        )

        # Wait a bit
        time.sleep(0.1)

        elapsed = state.elapsed_time
        assert elapsed > 0.1
        assert elapsed < 1.0  # Should be less than 1 second
