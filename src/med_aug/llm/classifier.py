"""Medication classifier using LLM."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import re
from enum import Enum

from ..core.logging import get_logger, PerformanceLogger
from .service import LLMService
from .providers import LLMConfig, LLMModel
from ..diseases import disease_registry

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)


class ClassificationConfidence(Enum):
    """Confidence levels for classification."""
    HIGH = "high"  # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"  # < 0.5


@dataclass
class ClassificationResult:
    """Result of medication classification."""
    
    medication: str
    primary_class: str
    confidence: float
    alternative_classes: List[str] = field(default_factory=list)
    reasoning: str = ""
    mechanism_of_action: str = ""
    therapeutic_use: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence_level(self) -> ClassificationConfidence:
        """Get confidence level category."""
        if self.confidence > 0.8:
            return ClassificationConfidence.HIGH
        elif self.confidence >= 0.5:
            return ClassificationConfidence.MEDIUM
        else:
            return ClassificationConfidence.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'medication': self.medication,
            'primary_class': self.primary_class,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'alternative_classes': self.alternative_classes,
            'reasoning': self.reasoning,
            'mechanism_of_action': self.mechanism_of_action,
            'therapeutic_use': self.therapeutic_use,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_llm_response(cls, response_text: str, medication: str) -> 'ClassificationResult':
        """
        Parse classification result from LLM response.
        
        Args:
            response_text: Raw LLM response
            medication: Input medication name
            
        Returns:
            Classification result
        """
        try:
            # Try to parse as JSON
            data = json.loads(response_text)
            
            return cls(
                medication=data.get('medication', medication),
                primary_class=data.get('classification', 'unknown'),
                confidence=float(data.get('confidence', 0.0)),
                alternative_classes=data.get('alternative_classifications', []),
                reasoning=data.get('reasoning', ''),
                mechanism_of_action=data.get('mechanism_of_action', ''),
                therapeutic_use=data.get('therapeutic_use', '')
            )
            
        except json.JSONDecodeError:
            # Try to extract from text
            logger.warning("classification_json_parse_failed", medication=medication)
            
            # Look for patterns in text
            confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', response_text, re.IGNORECASE)
            class_match = re.search(r'class(?:ification)?[:\s]+(\w+)', response_text, re.IGNORECASE)
            
            return cls(
                medication=medication,
                primary_class=class_match.group(1) if class_match else 'unknown',
                confidence=float(confidence_match.group(1)) if confidence_match else 0.0,
                reasoning=response_text[:200]  # Use first 200 chars as reasoning
            )


@dataclass
class BatchClassificationResult:
    """Result of batch medication classification."""
    
    classifications: Dict[str, List[str]]  # drug_class -> medications
    unclassified: List[str]
    total: int
    classified_count: int
    overall_confidence: float
    individual_results: Dict[str, ClassificationResult] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'classifications': self.classifications,
            'unclassified': self.unclassified,
            'summary': {
                'total': self.total,
                'classified': self.classified_count,
                'unclassified': len(self.unclassified),
                'confidence': self.overall_confidence
            },
            'individual_results': {
                med: result.to_dict()
                for med, result in self.individual_results.items()
            }
        }


class MedicationClassifier:
    """Classifier for medications using LLM."""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        disease_module: str = "nsclc",
        min_confidence: float = 0.5
    ):
        """
        Initialize medication classifier.
        
        Args:
            llm_service: LLM service instance
            disease_module: Disease module for context
            min_confidence: Minimum confidence threshold
        """
        self.llm_service = llm_service or LLMService()
        self.disease_module = disease_module
        self.min_confidence = min_confidence
        
        # Load disease module
        self.disease = disease_registry.get_module(disease_module)
        if not self.disease:
            logger.warning("disease_module_not_found", module=disease_module)
            self.disease = disease_registry.get_module("nsclc")  # Fallback
    
    async def classify(
        self,
        medication: str,
        use_context: bool = True,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        """
        Classify a single medication.
        
        Args:
            medication: Medication name to classify
            use_context: Whether to use disease context
            additional_context: Additional context for classification
            
        Returns:
            Classification result
        """
        logger.info("medication_classification_started", medication=medication)
        perf_logger.start_operation(f"classify_{medication}")
        
        # Prepare drug classes
        drug_classes = "\n".join([
            f"- {dc.name}: {', '.join(dc.keywords[:3])}"
            for dc in self.disease.drug_classes
        ])
        
        # Build template parameters
        template_params = {
            'medication': medication,
            'disease': self.disease.display_name if use_context else "general",
            'drug_classes': drug_classes
        }
        
        # Add additional context if provided
        if additional_context:
            if 'treatment_line' in additional_context:
                template_params['treatment_line'] = additional_context['treatment_line']
            if 'concomitant_meds' in additional_context:
                template_params['concomitant_meds'] = ', '.join(additional_context['concomitant_meds'])
        
        try:
            # Generate classification
            template = 'context_aware' if additional_context else 'classification'
            response = await self.llm_service.generate_with_template(
                template,
                **template_params
            )
            
            # Parse result
            result = ClassificationResult.from_llm_response(response.content, medication)
            
            # Add metadata
            result.metadata = {
                'disease_module': self.disease_module,
                'llm_model': response.model,
                'template_used': template
            }
            
            duration = perf_logger.end_operation(f"classify_{medication}")
            
            logger.info(
                "medication_classification_completed",
                medication=medication,
                classification=result.primary_class,
                confidence=result.confidence,
                duration=duration
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "medication_classification_failed",
                medication=medication,
                error=str(e)
            )
            
            # Return low-confidence result on error
            return ClassificationResult(
                medication=medication,
                primary_class="unknown",
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}"
            )
    
    async def classify_batch(
        self,
        medications: List[str],
        batch_size: int = 10
    ) -> BatchClassificationResult:
        """
        Classify multiple medications.
        
        Args:
            medications: List of medication names
            batch_size: Number of medications per batch
            
        Returns:
            Batch classification result
        """
        logger.info("batch_classification_started", count=len(medications))
        perf_logger.start_operation("batch_classification")
        
        all_results: Dict[str, ClassificationResult] = {}
        
        # Process in batches
        for i in range(0, len(medications), batch_size):
            batch = medications[i:i + batch_size]
            
            # Prepare batch prompt
            drug_classes = "\n".join([
                f"- {dc.name}"
                for dc in self.disease.drug_classes
            ])
            
            response = await self.llm_service.generate_with_template(
                'batch_classification',
                medications="\n".join(f"- {med}" for med in batch),
                disease=self.disease.display_name,
                drug_classes=drug_classes
            )
            
            # Parse batch response
            try:
                batch_data = json.loads(response.content)
                classifications = batch_data.get('classifications', {})
                
                # Create individual results
                for drug_class, meds in classifications.items():
                    for med in meds:
                        all_results[med] = ClassificationResult(
                            medication=med,
                            primary_class=drug_class,
                            confidence=batch_data.get('summary', {}).get('confidence', 0.7),
                            reasoning="Batch classification"
                        )
                
                # Handle unclassified
                for med in batch_data.get('unclassified', []):
                    all_results[med] = ClassificationResult(
                        medication=med,
                        primary_class="unknown",
                        confidence=0.0,
                        reasoning="Could not classify"
                    )
                    
            except json.JSONDecodeError:
                logger.warning("batch_response_parse_failed", batch_index=i)
                
                # Fallback to individual classification
                for med in batch:
                    result = await self.classify(med, use_context=True)
                    all_results[med] = result
        
        # Aggregate results
        classifications_by_class: Dict[str, List[str]] = {}
        unclassified: List[str] = []
        total_confidence = 0.0
        
        for med, result in all_results.items():
            if result.confidence >= self.min_confidence:
                if result.primary_class not in classifications_by_class:
                    classifications_by_class[result.primary_class] = []
                classifications_by_class[result.primary_class].append(med)
                total_confidence += result.confidence
            else:
                unclassified.append(med)
        
        classified_count = len(medications) - len(unclassified)
        avg_confidence = total_confidence / max(1, classified_count)
        
        duration = perf_logger.end_operation("batch_classification")
        
        logger.info(
            "batch_classification_completed",
            total=len(medications),
            classified=classified_count,
            unclassified=len(unclassified),
            confidence=avg_confidence,
            duration=duration
        )
        
        return BatchClassificationResult(
            classifications=classifications_by_class,
            unclassified=unclassified,
            total=len(medications),
            classified_count=classified_count,
            overall_confidence=avg_confidence,
            individual_results=all_results
        )
    
    async def validate_medication(
        self,
        medication: str
    ) -> Tuple[bool, str, float]:
        """
        Validate if a medication name is correct.
        
        Args:
            medication: Medication name to validate
            
        Returns:
            Tuple of (is_valid, standard_name, confidence)
        """
        response = await self.llm_service.generate_with_template(
            'validation',
            medication=medication
        )
        
        try:
            data = json.loads(response.content)
            return (
                data.get('is_valid', False),
                data.get('standard_name', medication),
                data.get('confidence', 0.0)
            )
        except json.JSONDecodeError:
            return False, medication, 0.0
    
    async def augment_medication_info(
        self,
        medication: str,
        drug_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get augmented information about a medication.
        
        Args:
            medication: Medication name
            drug_class: Known drug class
            
        Returns:
            Augmented medication information
        """
        # Classify first if no drug class provided
        if not drug_class:
            classification = await self.classify(medication)
            drug_class = classification.primary_class
        
        response = await self.llm_service.generate_with_template(
            'augmentation',
            medication=medication,
            disease=self.disease.display_name,
            drug_class=drug_class
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                'medication': medication,
                'drug_class': drug_class,
                'raw_info': response.content
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            'disease_module': self.disease_module,
            'min_confidence': self.min_confidence,
            'llm_stats': self.llm_service.get_stats()
        }