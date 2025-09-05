"""Integration tests for full workflow."""

import pytest

from med_aug.diseases import disease_registry
from med_aug.diseases.nsclc.module import NSCLCModule


@pytest.mark.integration
class TestFullWorkflow:
    """Test complete workflows across multiple components."""

    def test_disease_registry_discovers_nsclc(self):
        """Test that disease registry can discover NSCLC module."""
        # Re-initialize registry to ensure discovery
        disease_registry.reload_modules()

        # Check NSCLC is discovered
        available = disease_registry.list_available()
        assert "nsclc" in available

        # Get NSCLC module
        nsclc = disease_registry.get_module("nsclc")
        assert nsclc is not None
        assert isinstance(nsclc, NSCLCModule)
        assert nsclc.display_name == "Non-Small Cell Lung Cancer"

    def test_nsclc_module_complete_configuration(self):
        """Test that NSCLC module is fully configured."""
        nsclc = disease_registry.get_module("nsclc")

        # Verify all components are present
        assert len(nsclc.drug_classes) == 10
        assert len(nsclc.get_web_sources()) == 7
        assert len(nsclc.get_llm_context()) > 1000

        # Verify drug classes have proper configuration
        for drug_class in nsclc.drug_classes:
            assert drug_class.name
            assert len(drug_class.keywords) > 0
            assert 0 <= drug_class.confidence_threshold <= 1
            assert len(drug_class.web_sources) > 0

    def test_medication_validation_workflow(self):
        """Test medication validation through the system."""
        nsclc = disease_registry.get_module("nsclc")

        # Test known medications
        test_cases = [
            ("pembrolizumab", "immunotherapy", True),
            ("Keytruda", "immunotherapy", True),
            ("osimertinib", "targeted_therapy", True),
            ("carboplatin", "chemotherapy", True),
            ("unknown_drug", "unknown_class", True),  # Permissive for unknowns
        ]

        for medication, drug_class, expected in test_cases:
            result = nsclc.validate_medication(medication, drug_class)
            assert result == expected, f"Failed for {medication} in {drug_class}"

    def test_drug_class_retrieval(self):
        """Test retrieving drug classes and their configurations."""
        nsclc = disease_registry.get_module("nsclc")

        # Get specific drug class
        chemo = nsclc.get_drug_class_by_name("chemotherapy")
        assert chemo is not None
        assert "carboplatin" in chemo.keywords
        assert "paclitaxel" in chemo.keywords

        # Get confidence threshold
        threshold = nsclc.get_confidence_threshold("immunotherapy")
        assert threshold == 0.85

        # Get all keywords
        all_keywords = nsclc.get_all_keywords()
        assert len(all_keywords) > 100
        assert "pembrolizumab" in all_keywords
        assert "osimertinib" in all_keywords

    def test_multiple_disease_registration(self):
        """Test that registry can handle multiple disease modules."""
        from med_aug.diseases.base import DiseaseModule, DrugClassConfig

        # Create a test disease module
        class TestDiseaseModule(DiseaseModule):
            @property
            def name(self):
                return "test_disease"

            @property
            def display_name(self):
                return "Test Disease"

            @property
            def drug_classes(self):
                return [
                    DrugClassConfig(
                        name="test_class",
                        keywords=["test_drug"],
                        confidence_threshold=0.8,
                        web_sources=["test"],
                    )
                ]

            def get_web_sources(self):
                return ["http://test.com"]

            def get_llm_context(self):
                return "Test context"

            def validate_medication(self, medication, drug_class):
                return True

        # Register the test module
        disease_registry.register_module(TestDiseaseModule)

        # Verify both modules are available
        available = disease_registry.list_available()
        assert "nsclc" in available
        assert "test_disease" in available

        # Clean up
        disease_registry.unregister_module("test_disease")
        assert "test_disease" not in disease_registry.list_available()

    @pytest.mark.slow
    def test_comprehensive_nsclc_coverage(self):
        """Test comprehensive NSCLC medication coverage."""
        nsclc = disease_registry.get_module("nsclc")

        # Check major drug categories are covered
        drug_categories = {
            "Immunotherapy": ["pembrolizumab", "nivolumab", "atezolizumab"],
            "EGFR Inhibitors": ["osimertinib", "erlotinib", "afatinib"],
            "ALK Inhibitors": ["alectinib", "crizotinib", "lorlatinib"],
            "KRAS Inhibitors": ["sotorasib", "adagrasib"],
            "Chemotherapy": ["carboplatin", "paclitaxel", "pemetrexed"],
        }

        all_keywords = nsclc.get_all_keywords()

        for category, drugs in drug_categories.items():
            for drug in drugs:
                assert drug in all_keywords, f"{drug} from {category} not found"

    def test_medication_notes_functionality(self):
        """Test that medication notes provide useful information."""
        nsclc = disease_registry.get_module("nsclc")

        # Test multi-target drugs
        criz_notes = nsclc.get_medication_notes("crizotinib")
        assert "ALK and ROS1" in criz_notes

        # Test resistance information
        osim_notes = nsclc.get_medication_notes("osimertinib")
        assert "3rd generation" in osim_notes
        assert "T790M" in osim_notes

        # Test histology-specific drugs
        pem_notes = nsclc.get_medication_notes("pemetrexed")
        assert "non-squamous" in pem_notes
