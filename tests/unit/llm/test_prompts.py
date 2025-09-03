"""Unit tests for prompt templates."""

import pytest
import json

from med_aug.llm.prompts import (
    PromptTemplate,
    MedicationPrompts,
    PromptManager
)


class TestPromptTemplate:
    """Test prompt template."""
    
    def test_template_creation(self):
        """Test creating prompt template."""
        template = PromptTemplate(
            name="test_template",
            system="System message",
            user_template="User prompt with $variable",
            output_format="JSON format",
            examples=[
                {"input": "example input", "output": "example output"}
            ]
        )
        
        assert template.name == "test_template"
        assert template.system == "System message"
        assert "$variable" in template.user_template
        assert template.output_format == "JSON format"
        assert len(template.examples) == 1
    
    def test_template_format_basic(self):
        """Test basic template formatting."""
        template = PromptTemplate(
            name="test",
            system="System",
            user_template="Hello $name, you are $age years old"
        )
        
        system, prompt = template.format(name="Alice", age=30)
        
        assert system == "System"
        assert prompt == "Hello Alice, you are 30 years old"
    
    def test_template_format_with_output(self):
        """Test template formatting with output format."""
        template = PromptTemplate(
            name="test",
            system="System",
            user_template="Classify: $item",
            output_format="Return JSON with 'class' field"
        )
        
        system, prompt = template.format(item="test")
        
        assert "Classify: test" in prompt
        assert "Output Format:" in prompt
        assert "Return JSON with 'class' field" in prompt
    
    def test_template_format_with_examples(self):
        """Test template formatting with examples."""
        template = PromptTemplate(
            name="test",
            system="System",
            user_template="Task: $task",
            examples=[
                {"input": "A", "output": "1"},
                {"input": "B", "output": "2"}
            ]
        )
        
        system, prompt = template.format(task="classify")
        
        assert "Task: classify" in prompt
        assert "Examples:" in prompt
        assert "Input: A" in prompt
        assert "Output: 1" in prompt
        assert "Input: B" in prompt
        assert "Output: 2" in prompt
    
    def test_template_safe_substitute(self):
        """Test safe substitution (missing variables don't cause errors)."""
        template = PromptTemplate(
            name="test",
            system="System",
            user_template="$provided and $missing"
        )
        
        system, prompt = template.format(provided="value")
        
        assert "value and $missing" in prompt
    
    def test_template_to_dict(self):
        """Test template serialization."""
        template = PromptTemplate(
            name="test",
            system="System message",
            user_template="Template",
            output_format="Format",
            examples=[{"input": "in", "output": "out"}]
        )
        
        template_dict = template.to_dict()
        
        assert template_dict['name'] == "test"
        assert template_dict['system'] == "System message"
        assert template_dict['user_template'] == "Template"
        assert template_dict['output_format'] == "Format"
        assert len(template_dict['examples']) == 1


class TestMedicationPrompts:
    """Test medication-specific prompts."""
    
    def test_classification_prompt(self):
        """Test classification prompt template."""
        template = MedicationPrompts.classification_prompt()
        
        assert template.name == "medication_classification"
        assert "medical expert" in template.system
        assert "$medication" in template.user_template
        assert "$disease" in template.user_template
        assert "$drug_classes" in template.user_template
        assert "JSON" in template.output_format
        
        # Test formatting
        system, prompt = template.format(
            medication="pembrolizumab",
            disease="NSCLC",
            drug_classes="immunotherapy, chemotherapy"
        )
        
        assert "pembrolizumab" in prompt
        assert "NSCLC" in prompt
        assert "immunotherapy" in prompt
    
    def test_validation_prompt(self):
        """Test validation prompt template."""
        template = MedicationPrompts.validation_prompt()
        
        assert template.name == "medication_validation"
        assert "pharmaceutical expert" in template.system
        assert "$medication" in template.user_template
        
        system, prompt = template.format(medication="pembro")
        
        assert "pembro" in prompt
        assert "Validate" in prompt
    
    def test_augmentation_prompt(self):
        """Test augmentation prompt template."""
        template = MedicationPrompts.augmentation_prompt()
        
        assert template.name == "medication_augmentation"
        assert "clinical pharmacology" in template.system
        assert "$medication" in template.user_template
        assert "$disease" in template.user_template
        assert "$drug_class" in template.user_template
        
        system, prompt = template.format(
            medication="osimertinib",
            disease="NSCLC",
            drug_class="targeted_therapy"
        )
        
        assert "osimertinib" in prompt
        assert "NSCLC" in prompt
        assert "targeted_therapy" in prompt
    
    def test_extraction_enhancement_prompt(self):
        """Test extraction enhancement prompt."""
        template = MedicationPrompts.extraction_enhancement_prompt()
        
        assert template.name == "extraction_enhancement"
        assert "medical text processing" in template.system
        assert "$text" in template.user_template
        
        system, prompt = template.format(
            text="Patient on pembro 200mg and carbo/taxol"
        )
        
        assert "pembro 200mg and carbo/taxol" in prompt
        assert "abbreviations" in prompt.lower()
    
    def test_batch_classification_prompt(self):
        """Test batch classification prompt."""
        template = MedicationPrompts.batch_classification_prompt()
        
        assert template.name == "batch_classification"
        assert "$medications" in template.user_template
        assert "$disease" in template.user_template
        
        system, prompt = template.format(
            medications="drug1\ndrug2\ndrug3",
            disease="cancer",
            drug_classes="class1, class2"
        )
        
        assert "drug1" in prompt
        assert "cancer" in prompt
        assert "class1" in prompt
    
    def test_context_aware_prompt(self):
        """Test context-aware classification prompt."""
        template = MedicationPrompts.context_aware_prompt()
        
        assert template.name == "context_aware_classification"
        assert "clinical oncology" in template.system
        assert "$treatment_line" in template.user_template
        assert "$concomitant_meds" in template.user_template
        
        system, prompt = template.format(
            medication="pembrolizumab",
            disease="NSCLC",
            treatment_line="first-line",
            concomitant_meds="carboplatin",
            patient_factors="EGFR wild-type"
        )
        
        assert "pembrolizumab" in prompt
        assert "first-line" in prompt
        assert "carboplatin" in prompt
        assert "EGFR wild-type" in prompt


class TestPromptManager:
    """Test prompt manager."""
    
    def test_manager_initialization(self):
        """Test prompt manager initialization with default templates."""
        manager = PromptManager()
        
        # Should have default templates loaded
        templates = manager.list_templates()
        
        assert 'classification' in templates
        assert 'validation' in templates
        assert 'augmentation' in templates
        assert 'extraction' in templates
        assert 'batch_classification' in templates
        assert 'context_aware' in templates
    
    def test_get_template(self):
        """Test getting template by name."""
        manager = PromptManager()
        
        template = manager.get_template('classification')
        
        assert template is not None
        assert template.name == "medication_classification"
        
        # Non-existent template
        missing = manager.get_template('non_existent')
        assert missing is None
    
    def test_add_custom_template(self):
        """Test adding custom template."""
        manager = PromptManager()
        
        custom_template = PromptTemplate(
            name="custom",
            system="Custom system",
            user_template="Custom prompt: $param"
        )
        
        manager.add_template(custom_template)
        
        # Should be able to retrieve it
        retrieved = manager.get_template("custom")
        assert retrieved is not None
        assert retrieved.system == "Custom system"
        
        # Should appear in list
        templates = manager.list_templates()
        assert "custom" in templates
    
    def test_format_prompt_with_manager(self):
        """Test formatting prompt through manager."""
        manager = PromptManager()
        
        system, prompt = manager.format_prompt(
            'classification',
            medication='test_drug',
            disease='test_disease',
            drug_classes='test_classes'
        )
        
        assert system is not None
        assert "test_drug" in prompt
        assert "test_disease" in prompt
        assert "test_classes" in prompt
    
    def test_format_prompt_missing_template(self):
        """Test formatting with missing template."""
        manager = PromptManager()
        
        with pytest.raises(ValueError, match="Template not found"):
            manager.format_prompt('non_existent', param='value')
    
    def test_override_default_template(self):
        """Test overriding default template."""
        manager = PromptManager()
        
        # Get original
        original = manager.get_template('classification')
        original_system = original.system
        
        # Override with custom
        custom_classification = PromptTemplate(
            name="classification",
            system="Custom classification system",
            user_template="Custom classification: $medication"
        )
        
        manager.add_template(custom_classification)
        
        # Should get the custom one
        retrieved = manager.get_template('classification')
        assert retrieved.system == "Custom classification system"
        assert retrieved.system != original_system