"""Unit tests for medication classifier."""

import pytest
import json
from typing import Dict, Any

from med_aug.llm.classifier import (
    MedicationClassifier,
    ClassificationResult,
    BatchClassificationResult,
    ClassificationConfidence
)
from med_aug.llm.service import LLMService
from med_aug.llm.providers import MockProvider, LLMConfig


class TestClassificationResult:
    """Test classification result."""
    
    def test_result_creation(self):
        """Test creating classification result."""
        result = ClassificationResult(
            medication="pembrolizumab",
            primary_class="immunotherapy",
            confidence=0.95,
            alternative_classes=["PD-1 inhibitor"],
            reasoning="Monoclonal antibody targeting PD-1",
            mechanism_of_action="PD-1 blockade"
        )
        
        assert result.medication == "pembrolizumab"
        assert result.primary_class == "immunotherapy"
        assert result.confidence == 0.95
        assert "PD-1 inhibitor" in result.alternative_classes
    
    def test_confidence_level(self):
        """Test confidence level categorization."""
        high_conf = ClassificationResult(
            medication="test", primary_class="class", confidence=0.9
        )
        assert high_conf.confidence_level == ClassificationConfidence.HIGH
        
        med_conf = ClassificationResult(
            medication="test", primary_class="class", confidence=0.6
        )
        assert med_conf.confidence_level == ClassificationConfidence.MEDIUM
        
        low_conf = ClassificationResult(
            medication="test", primary_class="class", confidence=0.3
        )
        assert low_conf.confidence_level == ClassificationConfidence.LOW
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result = ClassificationResult(
            medication="test_drug",
            primary_class="test_class",
            confidence=0.75,
            metadata={'source': 'test'}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['medication'] == "test_drug"
        assert result_dict['primary_class'] == "test_class"
        assert result_dict['confidence'] == 0.75
        assert result_dict['confidence_level'] == "medium"
        assert result_dict['metadata']['source'] == 'test'
    
    def test_from_llm_response_json(self):
        """Test parsing from JSON LLM response."""
        json_response = json.dumps({
            "medication": "osimertinib",
            "classification": "targeted_therapy",
            "confidence": 0.98,
            "alternative_classifications": ["EGFR inhibitor"],
            "reasoning": "Third-generation EGFR TKI",
            "mechanism_of_action": "EGFR T790M inhibition",
            "therapeutic_use": "NSCLC with EGFR mutations"
        })
        
        result = ClassificationResult.from_llm_response(
            json_response,
            "osimertinib"
        )
        
        assert result.medication == "osimertinib"
        assert result.primary_class == "targeted_therapy"
        assert result.confidence == 0.98
        assert "EGFR inhibitor" in result.alternative_classes
        assert result.mechanism_of_action == "EGFR T790M inhibition"
    
    def test_from_llm_response_text(self):
        """Test parsing from text LLM response."""
        text_response = """
        Classification: immunotherapy
        Confidence: 0.85
        This is a checkpoint inhibitor used in cancer treatment.
        """
        
        result = ClassificationResult.from_llm_response(
            text_response,
            "test_drug"
        )
        
        assert result.medication == "test_drug"
        assert result.primary_class == "immunotherapy"
        assert result.confidence == 0.85
        assert "checkpoint inhibitor" in result.reasoning
    
    def test_from_llm_response_fallback(self):
        """Test fallback when parsing fails."""
        bad_response = "Invalid response format"
        
        result = ClassificationResult.from_llm_response(
            bad_response,
            "test_drug"
        )
        
        assert result.medication == "test_drug"
        assert result.primary_class == "unknown"
        assert result.confidence == 0.0


class TestBatchClassificationResult:
    """Test batch classification result."""
    
    def test_batch_result_creation(self):
        """Test creating batch result."""
        individual_results = {
            "drug1": ClassificationResult("drug1", "class_a", 0.9),
            "drug2": ClassificationResult("drug2", "class_a", 0.8),
            "drug3": ClassificationResult("drug3", "class_b", 0.7),
            "drug4": ClassificationResult("drug4", "unknown", 0.2)
        }
        
        batch_result = BatchClassificationResult(
            classifications={
                "class_a": ["drug1", "drug2"],
                "class_b": ["drug3"]
            },
            unclassified=["drug4"],
            total=4,
            classified_count=3,
            overall_confidence=0.8,
            individual_results=individual_results
        )
        
        assert batch_result.total == 4
        assert batch_result.classified_count == 3
        assert len(batch_result.unclassified) == 1
        assert "drug4" in batch_result.unclassified
        assert len(batch_result.classifications["class_a"]) == 2
    
    def test_batch_result_to_dict(self):
        """Test batch result serialization."""
        batch_result = BatchClassificationResult(
            classifications={"immunotherapy": ["drug1", "drug2"]},
            unclassified=["drug3"],
            total=3,
            classified_count=2,
            overall_confidence=0.75,
            individual_results={}
        )
        
        result_dict = batch_result.to_dict()
        
        assert result_dict['summary']['total'] == 3
        assert result_dict['summary']['classified'] == 2
        assert result_dict['summary']['unclassified'] == 1
        assert result_dict['summary']['confidence'] == 0.75
        assert "immunotherapy" in result_dict['classifications']


class TestMedicationClassifier:
    """Test medication classifier."""
    
    @pytest.fixture
    def mock_service(self):
        """Create mock LLM service."""
        provider = MockProvider()
        service = LLMService(provider=provider, enable_cache=False)
        return service
    
    @pytest.fixture
    def classifier(self, mock_service):
        """Create medication classifier."""
        return MedicationClassifier(
            llm_service=mock_service,
            disease_module="nsclc",
            min_confidence=0.5
        )
    
    @pytest.mark.asyncio
    async def test_classify_single_medication(self, classifier):
        """Test classifying a single medication."""
        # Set mock response
        classifier.llm_service.provider.set_responses([
            json.dumps({
                "medication": "pembrolizumab",
                "classification": "immunotherapy",
                "confidence": 0.95,
                "alternative_classifications": ["PD-1 inhibitor"],
                "reasoning": "Anti-PD-1 monoclonal antibody",
                "mechanism_of_action": "PD-1 blockade",
                "therapeutic_use": "NSCLC treatment"
            })
        ])
        
        result = await classifier.classify("pembrolizumab")
        
        assert result.medication == "pembrolizumab"
        assert result.primary_class == "immunotherapy"
        assert result.confidence == 0.95
        assert result.confidence_level == ClassificationConfidence.HIGH
    
    @pytest.mark.asyncio
    async def test_classify_with_context(self, classifier):
        """Test classification with additional context."""
        classifier.llm_service.provider.set_responses([
            json.dumps({
                "medication": "carboplatin",
                "classification": "chemotherapy",
                "confidence": 0.9,
                "therapeutic_role": "first-line",
                "combination_potential": ["paclitaxel", "pemetrexed"]
            })
        ])
        
        result = await classifier.classify(
            "carboplatin",
            use_context=True,
            additional_context={
                'treatment_line': 'first-line',
                'concomitant_meds': ['paclitaxel']
            }
        )
        
        assert result.medication == "carboplatin"
        assert result.primary_class == "chemotherapy"
        assert result.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_classify_batch(self, classifier):
        """Test batch classification."""
        medications = ["pembrolizumab", "osimertinib", "carboplatin"]
        
        # Set mock batch response
        classifier.llm_service.provider.set_responses([
            json.dumps({
                "classifications": {
                    "immunotherapy": ["pembrolizumab"],
                    "targeted_therapy": ["osimertinib"],
                    "chemotherapy": ["carboplatin"]
                },
                "unclassified": [],
                "summary": {
                    "total": 3,
                    "classified": 3,
                    "confidence": 0.9
                }
            })
        ])
        
        result = await classifier.classify_batch(medications, batch_size=5)
        
        assert result.total == 3
        assert result.classified_count == 3
        assert len(result.unclassified) == 0
        assert "pembrolizumab" in result.classifications.get("immunotherapy", [])
    
    @pytest.mark.asyncio
    async def test_classify_batch_with_unclassified(self, classifier):
        """Test batch classification with unclassified medications."""
        medications = ["drug1", "drug2", "unknown_drug"]
        
        classifier.llm_service.provider.set_responses([
            json.dumps({
                "classifications": {
                    "class_a": ["drug1", "drug2"]
                },
                "unclassified": ["unknown_drug"],
                "summary": {
                    "total": 3,
                    "classified": 2,
                    "confidence": 0.7
                }
            })
        ])
        
        result = await classifier.classify_batch(medications)
        
        assert result.total == 3
        assert result.classified_count == 2
        assert "unknown_drug" in result.unclassified
    
    @pytest.mark.asyncio
    async def test_validate_medication(self, classifier):
        """Test medication validation."""
        classifier.llm_service.provider.set_responses([
            json.dumps({
                "input": "pembro",
                "is_valid": True,
                "standard_name": "pembrolizumab",
                "confidence": 0.95
            })
        ])
        
        is_valid, standard_name, confidence = await classifier.validate_medication("pembro")
        
        assert is_valid is True
        assert standard_name == "pembrolizumab"
        assert confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_validate_invalid_medication(self, classifier):
        """Test validating invalid medication."""
        classifier.llm_service.provider.set_responses([
            json.dumps({
                "input": "notadrug",
                "is_valid": False,
                "standard_name": "notadrug",
                "confidence": 0.0
            })
        ])
        
        is_valid, standard_name, confidence = await classifier.validate_medication("notadrug")
        
        assert is_valid is False
        assert standard_name == "notadrug"
        assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_augment_medication_info(self, classifier):
        """Test augmenting medication information."""
        classifier.llm_service.provider.set_responses([
            json.dumps({
                "medication": "pembrolizumab",
                "dosing": {
                    "standard": "200mg IV q3w or 400mg IV q6w",
                    "adjustments": "No dose adjustments for renal/hepatic impairment"
                },
                "side_effects": {
                    "common": ["fatigue", "rash", "diarrhea"],
                    "serious": ["pneumonitis", "colitis", "hepatitis"]
                },
                "monitoring": {
                    "labs": ["LFTs", "TSH", "cortisol"],
                    "frequency": "Before each cycle"
                }
            })
        ])
        
        info = await classifier.augment_medication_info(
            "pembrolizumab",
            drug_class="immunotherapy"
        )
        
        assert info['medication'] == "pembrolizumab"
        assert 'dosing' in info
        assert 'side_effects' in info
        assert 'monitoring' in info
    
    @pytest.mark.asyncio
    async def test_classification_error_handling(self, classifier):
        """Test error handling during classification."""
        # Create a provider that always fails
        class FailingProvider:
            async def generate(self, prompt, system=None, **kwargs):
                raise Exception("Provider failed")
            
            async def is_available(self):
                return True
        
        classifier.llm_service.provider = FailingProvider()
        
        # This should handle the error gracefully
        result = await classifier.classify("test_drug")
        
        # Should return low-confidence unknown classification
        assert result.medication == "test_drug"
        assert result.primary_class == "unknown"
        assert result.confidence == 0.0
        assert "Classification failed" in result.reasoning
    
    def test_classifier_stats(self, classifier):
        """Test getting classifier statistics."""
        stats = classifier.get_stats()
        
        assert stats['disease_module'] == "nsclc"
        assert stats['min_confidence'] == 0.5
        assert 'llm_stats' in stats