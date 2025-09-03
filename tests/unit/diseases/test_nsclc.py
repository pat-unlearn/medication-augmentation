"""Unit tests for NSCLC disease module."""

import pytest
from med_aug.diseases.nsclc.module import NSCLCModule


class TestNSCLCModule:
    """Test NSCLCModule implementation."""
    
    @pytest.fixture
    def nsclc_module(self):
        """Create NSCLC module instance."""
        return NSCLCModule()
    
    def test_module_basic_properties(self, nsclc_module):
        """Test basic properties of NSCLC module."""
        assert nsclc_module.name == "nsclc"
        assert nsclc_module.display_name == "Non-Small Cell Lung Cancer"
    
    def test_drug_classes_count(self, nsclc_module):
        """Test that NSCLC has expected number of drug classes."""
        drug_classes = nsclc_module.drug_classes
        assert len(drug_classes) == 10
        
        # Check specific classes exist
        class_names = [dc.name for dc in drug_classes]
        assert "chemotherapy" in class_names
        assert "immunotherapy" in class_names
        assert "targeted_therapy" in class_names
        assert "anti_angiogenic" in class_names
        assert "antibody_drug_conjugates" in class_names
        assert "kras_inhibitors" in class_names
        assert "egfr_inhibitors" in class_names
        assert "alk_inhibitors" in class_names
        assert "ros1_inhibitors" in class_names
        assert "met_inhibitors" in class_names
    
    def test_chemotherapy_keywords(self, nsclc_module):
        """Test chemotherapy drug class keywords."""
        chemo_class = next(dc for dc in nsclc_module.drug_classes if dc.name == "chemotherapy")
        
        assert "carboplatin" in chemo_class.keywords
        assert "paclitaxel" in chemo_class.keywords
        assert "pemetrexed" in chemo_class.keywords
        assert "docetaxel" in chemo_class.keywords
        assert "gemcitabine" in chemo_class.keywords
        assert "cisplatin" in chemo_class.keywords
        
        # Check brand names
        assert "abraxane" in chemo_class.keywords
        assert "taxol" in chemo_class.keywords
        assert "alimta" in chemo_class.keywords
    
    def test_immunotherapy_keywords(self, nsclc_module):
        """Test immunotherapy drug class keywords."""
        immuno_class = next(dc for dc in nsclc_module.drug_classes if dc.name == "immunotherapy")
        
        # Generic names
        assert "pembrolizumab" in immuno_class.keywords
        assert "nivolumab" in immuno_class.keywords
        assert "atezolizumab" in immuno_class.keywords
        assert "durvalumab" in immuno_class.keywords
        
        # Brand names
        assert "keytruda" in immuno_class.keywords
        assert "opdivo" in immuno_class.keywords
        assert "tecentriq" in immuno_class.keywords
        assert "imfinzi" in immuno_class.keywords
    
    def test_targeted_therapy_keywords(self, nsclc_module):
        """Test targeted therapy has comprehensive coverage."""
        targeted_class = next(dc for dc in nsclc_module.drug_classes if dc.name == "targeted_therapy")
        
        # Should have the most keywords
        assert len(targeted_class.keywords) > 30
        
        # Key EGFR inhibitors
        assert "osimertinib" in targeted_class.keywords
        assert "tagrisso" in targeted_class.keywords
        
        # ALK inhibitors
        assert "alectinib" in targeted_class.keywords
        assert "lorlatinib" in targeted_class.keywords
        
        # KRAS inhibitors
        assert "sotorasib" in targeted_class.keywords
        assert "adagrasib" in targeted_class.keywords
    
    def test_confidence_thresholds(self, nsclc_module):
        """Test confidence thresholds for different drug classes."""
        for dc in nsclc_module.drug_classes:
            assert 0.8 <= dc.confidence_threshold <= 0.9
            
            if dc.name == "chemotherapy":
                assert dc.confidence_threshold == 0.8
            elif dc.name in ["targeted_therapy", "egfr_inhibitors", "alk_inhibitors"]:
                assert dc.confidence_threshold == 0.9
    
    def test_web_sources(self, nsclc_module):
        """Test web sources are properly defined."""
        sources = nsclc_module.get_web_sources()
        
        assert len(sources) == 7
        
        # Check key sources
        assert any("fda.gov" in s for s in sources)
        assert any("clinicaltrials.gov" in s for s in sources)
        assert any("nccn.org" in s for s in sources)
        assert any("oncokb.org" in s for s in sources)
        assert any("asco.org" in s for s in sources)
    
    def test_llm_context(self, nsclc_module):
        """Test LLM context contains key information."""
        context = nsclc_module.get_llm_context()
        
        assert len(context) > 1000  # Should be comprehensive
        
        # Key terms that should be present
        assert "NSCLC" in context
        assert "Non-Small Cell Lung Cancer" in context
        assert "2024-2025" in context
        assert "EGFR" in context
        assert "ALK" in context
        assert "PD-L1" in context
        assert "KRAS G12C" in context
        assert "First-line" in context or "FIRST-LINE" in context
    
    def test_validate_medication_chemotherapy(self, nsclc_module):
        """Test medication validation for chemotherapy."""
        # Valid chemotherapy drugs
        assert nsclc_module.validate_medication("carboplatin", "chemotherapy") is True
        assert nsclc_module.validate_medication("Carboplatin", "chemotherapy") is True  # Case insensitive
        assert nsclc_module.validate_medication("paclitaxel", "chemotherapy") is True
        assert nsclc_module.validate_medication("abraxane", "chemotherapy") is True
        
        # Invalid for chemotherapy
        assert nsclc_module.validate_medication("pembrolizumab", "chemotherapy") is False
        assert nsclc_module.validate_medication("osimertinib", "chemotherapy") is False
    
    def test_validate_medication_immunotherapy(self, nsclc_module):
        """Test medication validation for immunotherapy."""
        assert nsclc_module.validate_medication("pembrolizumab", "immunotherapy") is True
        assert nsclc_module.validate_medication("Keytruda", "immunotherapy") is True
        assert nsclc_module.validate_medication("nivolumab", "immunotherapy") is True
        assert nsclc_module.validate_medication("opdivo", "immunotherapy") is True
        
        # Not immunotherapy
        assert nsclc_module.validate_medication("carboplatin", "immunotherapy") is False
    
    def test_validate_medication_targeted_therapy(self, nsclc_module):
        """Test medication validation for targeted therapy."""
        assert nsclc_module.validate_medication("osimertinib", "targeted_therapy") is True
        assert nsclc_module.validate_medication("tagrisso", "targeted_therapy") is True
        assert nsclc_module.validate_medication("alectinib", "targeted_therapy") is True
        assert nsclc_module.validate_medication("sotorasib", "targeted_therapy") is True
    
    def test_validate_medication_unknown_class(self, nsclc_module):
        """Test validation with unknown drug class."""
        # Should be permissive for unknown classes
        result = nsclc_module.validate_medication("any_drug", "unknown_class")
        assert result is True
    
    def test_validate_medication_unknown_drug(self, nsclc_module):
        """Test validation with unknown drug."""
        # For known drug classes, unknown drugs should return False
        result = nsclc_module.validate_medication("experimental_drug_xyz", "chemotherapy")
        assert result is False  # Not permissive for known classes
        
        # For unknown drug classes, should be permissive
        result_unknown = nsclc_module.validate_medication("experimental_drug_xyz", "unknown_class")
        assert result_unknown is True  # Permissive for unknown classes
    
    def test_get_medication_notes_crizotinib(self, nsclc_module):
        """Test medication notes for multi-target drugs."""
        notes = nsclc_module.get_medication_notes("crizotinib")
        assert "ALK and ROS1" in notes
        
        notes_branded = nsclc_module.get_medication_notes("Xalkori")
        assert "ALK and ROS1" in notes_branded
    
    def test_get_medication_notes_osimertinib(self, nsclc_module):
        """Test medication notes for osimertinib."""
        notes = nsclc_module.get_medication_notes("osimertinib")
        assert "3rd generation" in notes
        assert "T790M" in notes
        
        notes_branded = nsclc_module.get_medication_notes("Tagrisso")
        assert "3rd generation" in notes
    
    def test_get_medication_notes_pemetrexed(self, nsclc_module):
        """Test medication notes for histology-specific drugs."""
        notes = nsclc_module.get_medication_notes("pemetrexed")
        assert "non-squamous" in notes
        
        notes = nsclc_module.get_medication_notes("gemcitabine")
        assert "squamous" in notes
    
    def test_get_medication_notes_pembrolizumab(self, nsclc_module):
        """Test medication notes for pembrolizumab."""
        notes = nsclc_module.get_medication_notes("pembrolizumab")
        assert "PD-L1" in notes
        assert "monotherapy" in notes or "chemotherapy" in notes
    
    def test_get_medication_notes_unknown_drug(self, nsclc_module):
        """Test medication notes for unknown drug."""
        notes = nsclc_module.get_medication_notes("unknown_drug")
        assert notes == ""
    
    def test_get_all_keywords_unique(self, nsclc_module):
        """Test that all keywords are unique across module."""
        all_keywords = nsclc_module.get_all_keywords()
        
        # Should have many keywords
        assert len(all_keywords) > 100
        
        # Should be unique
        assert len(all_keywords) == len(set(all_keywords))
    
    def test_overlapping_drugs_in_classes(self, nsclc_module):
        """Test that some drugs appear in multiple classes (e.g., crizotinib)."""
        alk_class = next(dc for dc in nsclc_module.drug_classes if dc.name == "alk_inhibitors")
        ros1_class = next(dc for dc in nsclc_module.drug_classes if dc.name == "ros1_inhibitors")
        
        # Crizotinib should be in both
        assert "crizotinib" in alk_class.keywords
        assert "crizotinib" in ros1_class.keywords
        
        # Lorlatinib too
        assert "lorlatinib" in alk_class.keywords
        assert "lorlatinib" in ros1_class.keywords
    
    def test_module_class_export(self):
        """Test that MODULE_CLASS is properly exported."""
        from med_aug.diseases.nsclc import module
        assert hasattr(module, 'MODULE_CLASS')
        assert module.MODULE_CLASS == NSCLCModule
    
    def test_to_config_dict(self, nsclc_module):
        """Test conversion to configuration dictionary."""
        config = nsclc_module.to_config_dict()
        
        assert config["name"] == "nsclc"
        assert config["display_name"] == "Non-Small Cell Lung Cancer"
        assert len(config["drug_classes"]) == 10
        assert len(config["web_sources"]) == 7
        assert len(config["taking_variables"]) > 0