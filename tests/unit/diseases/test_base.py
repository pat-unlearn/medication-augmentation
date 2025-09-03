"""Unit tests for disease base module."""

import pytest
from typing import List
import tempfile
import yaml
from pathlib import Path

from med_aug.diseases.base import (
    DrugClassConfig,
    DiseaseModule,
    DiseaseModuleConfig,
)


class TestDrugClassConfig:
    """Test DrugClassConfig dataclass."""
    
    def test_valid_drug_class_config(self):
        """Test creating valid drug class configuration."""
        config = DrugClassConfig(
            name="chemotherapy",
            keywords=["carboplatin", "paclitaxel", "docetaxel"],
            confidence_threshold=0.8,
            web_sources=["fda", "nccn"]
        )
        
        assert config.name == "chemotherapy"
        assert len(config.keywords) == 3
        assert config.confidence_threshold == 0.8
        assert len(config.web_sources) == 2
    
    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence thresholds raise errors."""
        # Too high
        with pytest.raises(ValueError, match="Confidence threshold must be between 0 and 1"):
            DrugClassConfig(
                name="test",
                keywords=["drug"],
                confidence_threshold=1.5,
                web_sources=["fda"]
            )
        
        # Too low
        with pytest.raises(ValueError, match="Confidence threshold must be between 0 and 1"):
            DrugClassConfig(
                name="test",
                keywords=["drug"],
                confidence_threshold=-0.1,
                web_sources=["fda"]
            )
    
    def test_empty_keywords_validation(self):
        """Test that empty keywords list raises error."""
        with pytest.raises(ValueError, match="must have at least one keyword"):
            DrugClassConfig(
                name="test",
                keywords=[],
                confidence_threshold=0.8,
                web_sources=["fda"]
            )
    
    def test_boundary_confidence_values(self):
        """Test boundary values for confidence threshold."""
        # Test 0.0
        config_zero = DrugClassConfig(
            name="test",
            keywords=["drug"],
            confidence_threshold=0.0,
            web_sources=["fda"]
        )
        assert config_zero.confidence_threshold == 0.0
        
        # Test 1.0
        config_one = DrugClassConfig(
            name="test",
            keywords=["drug"],
            confidence_threshold=1.0,
            web_sources=["fda"]
        )
        assert config_one.confidence_threshold == 1.0


class TestDiseaseModule:
    """Test DiseaseModule abstract base class."""
    
    @pytest.fixture
    def concrete_disease_module(self):
        """Create a concrete implementation of DiseaseModule for testing."""
        
        class TestDiseaseModule(DiseaseModule):
            @property
            def name(self) -> str:
                return "test_disease"
            
            @property
            def display_name(self) -> str:
                return "Test Disease"
            
            @property
            def drug_classes(self) -> List[DrugClassConfig]:
                return [
                    DrugClassConfig(
                        name="class1",
                        keywords=["drug1", "drug2", "drug3"],
                        confidence_threshold=0.8,
                        web_sources=["source1"]
                    ),
                    DrugClassConfig(
                        name="class2",
                        keywords=["drug4", "drug5"],
                        confidence_threshold=0.9,
                        web_sources=["source2"]
                    ),
                ]
            
            def get_web_sources(self) -> List[str]:
                return ["http://example.com", "http://test.com"]
            
            def get_llm_context(self) -> str:
                return "Test disease context for LLM"
            
            def validate_medication(self, medication: str, drug_class: str) -> bool:
                # Simple validation for testing
                if drug_class == "class1":
                    return medication.lower() in ["drug1", "drug2", "drug3"]
                elif drug_class == "class2":
                    return medication.lower() in ["drug4", "drug5"]
                return False
        
        return TestDiseaseModule()
    
    def test_disease_module_properties(self, concrete_disease_module):
        """Test basic properties of disease module."""
        assert concrete_disease_module.name == "test_disease"
        assert concrete_disease_module.display_name == "Test Disease"
        assert len(concrete_disease_module.drug_classes) == 2
        assert len(concrete_disease_module.get_web_sources()) == 2
        assert "LLM" in concrete_disease_module.get_llm_context()
    
    def test_get_drug_class_by_name(self, concrete_disease_module):
        """Test retrieving drug class by name."""
        # Existing class
        class1 = concrete_disease_module.get_drug_class_by_name("class1")
        assert class1 is not None
        assert class1.name == "class1"
        assert len(class1.keywords) == 3
        
        # Non-existent class
        none_class = concrete_disease_module.get_drug_class_by_name("nonexistent")
        assert none_class is None
    
    def test_get_all_keywords(self, concrete_disease_module):
        """Test getting all keywords from all drug classes."""
        all_keywords = concrete_disease_module.get_all_keywords()
        
        assert len(all_keywords) == 5  # drug1, drug2, drug3, drug4, drug5
        assert "drug1" in all_keywords
        assert "drug5" in all_keywords
        
        # Check uniqueness
        assert len(all_keywords) == len(set(all_keywords))
    
    def test_get_confidence_threshold(self, concrete_disease_module):
        """Test getting confidence threshold for drug classes."""
        # Existing class
        threshold1 = concrete_disease_module.get_confidence_threshold("class1")
        assert threshold1 == 0.8
        
        threshold2 = concrete_disease_module.get_confidence_threshold("class2")
        assert threshold2 == 0.9
        
        # Non-existent class - should return default
        default_threshold = concrete_disease_module.get_confidence_threshold("nonexistent")
        assert default_threshold == 0.7
    
    def test_get_taking_variables(self, concrete_disease_module):
        """Test generation of taking_{drug} variable names."""
        variables = concrete_disease_module.get_taking_variables()
        
        # Should use first keyword from each class
        assert "taking_drug1" in variables
        assert "taking_drug4" in variables
        assert len(variables) == 2
    
    def test_validate_medication(self, concrete_disease_module):
        """Test medication validation logic."""
        # Valid medications
        assert concrete_disease_module.validate_medication("drug1", "class1") is True
        assert concrete_disease_module.validate_medication("Drug2", "class1") is True  # Case insensitive
        assert concrete_disease_module.validate_medication("drug4", "class2") is True
        
        # Invalid medications
        assert concrete_disease_module.validate_medication("drug4", "class1") is False
        assert concrete_disease_module.validate_medication("drug1", "class2") is False
        assert concrete_disease_module.validate_medication("unknown", "class1") is False
    
    def test_to_config_dict(self, concrete_disease_module):
        """Test conversion to configuration dictionary."""
        config_dict = concrete_disease_module.to_config_dict()
        
        assert config_dict["name"] == "test_disease"
        assert config_dict["display_name"] == "Test Disease"
        assert len(config_dict["drug_classes"]) == 2
        assert len(config_dict["web_sources"]) == 2
        assert len(config_dict["taking_variables"]) == 2
        
        # Check drug class structure
        first_class = config_dict["drug_classes"][0]
        assert first_class["name"] == "class1"
        assert len(first_class["keywords"]) == 3
        assert first_class["confidence_threshold"] == 0.8
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""
        # Try to create incomplete implementation
        with pytest.raises(TypeError):
            class IncompleteDiseaseModule(DiseaseModule):
                @property
                def name(self) -> str:
                    return "incomplete"
                # Missing other required methods
            
            IncompleteDiseaseModule()


class TestDiseaseModuleConfig:
    """Test DiseaseModuleConfig Pydantic model."""
    
    def test_disease_module_config_creation(self):
        """Test creating disease module configuration."""
        config = DiseaseModuleConfig(
            name="Test Disease",
            display_name="Test Disease Display",
            description="A test disease module",
            drug_classes=[
                {
                    "name": "class1",
                    "keywords": ["drug1", "drug2"],
                    "confidence_threshold": 0.8
                }
            ],
            web_sources={
                "fda": {
                    "base_url": "http://fda.gov",
                    "rate_limit": 1.0
                }
            },
            llm_settings={
                "model": "gpt-4",
                "temperature": 0.1
            },
            validation={
                "require_human_review": True
            }
        )
        
        assert config.name == "Test Disease"
        assert config.description == "A test disease module"
        assert len(config.drug_classes) == 1
        assert "fda" in config.web_sources
    
    def test_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        # Create a temporary YAML file
        yaml_content = {
            "disease": {
                "name": "YAML Disease",
                "display_name": "YAML Disease Display",
                "description": "Loaded from YAML",
                "drug_classes": [
                    {
                        "name": "yaml_class",
                        "keywords": ["yaml_drug1", "yaml_drug2"],
                        "confidence_threshold": 0.85
                    }
                ]
            }
        }
        
        yaml_file = tmp_path / "test_config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        # Load configuration
        config = DiseaseModuleConfig.from_yaml(str(yaml_file))
        
        assert config.name == "YAML Disease"
        assert config.description == "Loaded from YAML"
        assert len(config.drug_classes) == 1
        assert config.drug_classes[0]["name"] == "yaml_class"
    
    def test_to_yaml(self, tmp_path):
        """Test saving configuration to YAML file."""
        config = DiseaseModuleConfig(
            name="Save Test",
            display_name="Save Test Display",
            description="To be saved",
            drug_classes=[
                {
                    "name": "save_class",
                    "keywords": ["save_drug"],
                    "confidence_threshold": 0.7
                }
            ]
        )
        
        yaml_file = tmp_path / "save_test.yaml"
        config.to_yaml(str(yaml_file))
        
        # Read the saved file
        with open(yaml_file, 'r') as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["disease"]["name"] == "Save Test"
        assert loaded_data["disease"]["description"] == "To be saved"
        assert len(loaded_data["disease"]["drug_classes"]) == 1
    
    def test_default_values(self):
        """Test that default values are properly set."""
        config = DiseaseModuleConfig(
            name="Minimal",
            display_name="Minimal Display",
            description="Minimal config",
            drug_classes=[]
        )
        
        assert config.web_sources == {}
        assert config.llm_settings == {}
        assert config.validation == {}
    
    def test_complex_drug_classes(self):
        """Test handling complex drug class configurations."""
        config = DiseaseModuleConfig(
            name="Complex",
            display_name="Complex Display",
            description="Complex drug classes",
            drug_classes=[
                {
                    "name": f"class_{i}",
                    "keywords": [f"drug_{i}_{j}" for j in range(3)],
                    "confidence_threshold": 0.7 + i * 0.05,
                    "web_sources": [f"source_{i}"]
                }
                for i in range(5)
            ]
        )
        
        assert len(config.drug_classes) == 5
        assert config.drug_classes[0]["keywords"][0] == "drug_0_0"
        assert config.drug_classes[4]["confidence_threshold"] == pytest.approx(0.9)