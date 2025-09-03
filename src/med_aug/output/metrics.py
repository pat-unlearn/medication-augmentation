"""Quality metrics calculation and analysis."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for medication augmentation."""
    
    # Data quality metrics
    total_rows: int = 0
    total_columns: int = 0
    medication_columns_found: int = 0
    
    # Extraction metrics
    total_medications_extracted: int = 0
    unique_medications: int = 0
    normalized_medications: int = 0
    variants_identified: int = 0
    
    # Classification metrics  
    medications_classified: int = 0
    high_confidence_classifications: int = 0
    medium_confidence_classifications: int = 0
    low_confidence_classifications: int = 0
    unclassified_medications: int = 0
    average_confidence: float = 0.0
    
    # Validation metrics
    valid_medications: int = 0
    invalid_medications: int = 0
    validation_coverage: float = 0.0
    
    # Web research metrics
    medications_researched: int = 0
    fda_data_found: int = 0
    clinical_trials_found: int = 0
    
    # Performance metrics
    processing_time_seconds: float = 0.0
    extraction_time_seconds: float = 0.0
    classification_time_seconds: float = 0.0
    web_research_time_seconds: float = 0.0
    
    # Data distribution
    drug_class_distribution: Dict[str, int] = field(default_factory=dict)
    confidence_distribution: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data_quality': {
                'total_rows': self.total_rows,
                'total_columns': self.total_columns,
                'medication_columns_found': self.medication_columns_found
            },
            'extraction': {
                'total_extracted': self.total_medications_extracted,
                'unique_medications': self.unique_medications,
                'normalized': self.normalized_medications,
                'variants_identified': self.variants_identified,
                'extraction_rate': self._calculate_extraction_rate()
            },
            'classification': {
                'classified': self.medications_classified,
                'high_confidence': self.high_confidence_classifications,
                'medium_confidence': self.medium_confidence_classifications,
                'low_confidence': self.low_confidence_classifications,
                'unclassified': self.unclassified_medications,
                'average_confidence': round(self.average_confidence, 3),
                'classification_rate': self._calculate_classification_rate()
            },
            'validation': {
                'valid': self.valid_medications,
                'invalid': self.invalid_medications,
                'coverage': round(self.validation_coverage, 2)
            },
            'web_research': {
                'researched': self.medications_researched,
                'fda_data': self.fda_data_found,
                'clinical_trials': self.clinical_trials_found,
                'research_coverage': self._calculate_research_coverage()
            },
            'performance': {
                'total_time': round(self.processing_time_seconds, 2),
                'extraction_time': round(self.extraction_time_seconds, 2),
                'classification_time': round(self.classification_time_seconds, 2),
                'web_research_time': round(self.web_research_time_seconds, 2)
            },
            'distribution': {
                'drug_classes': self.drug_class_distribution,
                'confidence_levels': self._get_confidence_level_distribution()
            }
        }
    
    def _calculate_extraction_rate(self) -> float:
        """Calculate extraction success rate."""
        if self.total_rows == 0:
            return 0.0
        return round(self.total_medications_extracted / self.total_rows, 3)
    
    def _calculate_classification_rate(self) -> float:
        """Calculate classification success rate."""
        if self.unique_medications == 0:
            return 0.0
        return round(self.medications_classified / self.unique_medications, 3)
    
    def _calculate_research_coverage(self) -> float:
        """Calculate web research coverage."""
        if self.unique_medications == 0:
            return 0.0
        return round(self.medications_researched / self.unique_medications, 3)
    
    def _get_confidence_level_distribution(self) -> Dict[str, float]:
        """Get distribution by confidence level."""
        total = (self.high_confidence_classifications + 
                self.medium_confidence_classifications + 
                self.low_confidence_classifications)
        
        if total == 0:
            return {'high': 0.0, 'medium': 0.0, 'low': 0.0}
        
        return {
            'high': round(self.high_confidence_classifications / total, 3),
            'medium': round(self.medium_confidence_classifications / total, 3),
            'low': round(self.low_confidence_classifications / total, 3)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get executive summary of metrics."""
        return {
            'Total Medications': self.unique_medications,
            'Extraction Rate': f"{self._calculate_extraction_rate() * 100:.1f}%",
            'Classification Rate': f"{self._calculate_classification_rate() * 100:.1f}%",
            'Average Confidence': f"{self.average_confidence:.2f}",
            'Processing Time': f"{self.processing_time_seconds:.1f}s",
            'Data Quality Score': self._calculate_quality_score()
        }
    
    def _calculate_quality_score(self) -> str:
        """Calculate overall quality score."""
        scores = []
        
        # Extraction score
        if self.total_rows > 0:
            scores.append(self._calculate_extraction_rate())
        
        # Classification score
        if self.unique_medications > 0:
            scores.append(self._calculate_classification_rate())
            scores.append(self.average_confidence)
        
        # Validation score
        if self.validation_coverage > 0:
            scores.append(self.validation_coverage)
        
        if not scores:
            return "N/A"
        
        avg_score = statistics.mean(scores)
        
        if avg_score >= 0.8:
            return f"Excellent ({avg_score:.1%})"
        elif avg_score >= 0.6:
            return f"Good ({avg_score:.1%})"
        elif avg_score >= 0.4:
            return f"Fair ({avg_score:.1%})"
        else:
            return f"Poor ({avg_score:.1%})"


class MetricsCalculator:
    """Calculator for quality metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics = QualityMetrics()
        self.start_time = datetime.now()
        
        logger.info("metrics_calculator_initialized")
    
    def calculate_from_pipeline_results(
        self,
        phase_results: Dict[str, Any]
    ) -> QualityMetrics:
        """
        Calculate metrics from pipeline results.
        
        Args:
            phase_results: Results from pipeline phases
            
        Returns:
            Calculated quality metrics
        """
        logger.info("calculating_metrics_from_pipeline")
        
        # Data ingestion metrics
        if 'data_ingestion' in phase_results:
            ingestion = phase_results['data_ingestion']
            self.metrics.total_rows = ingestion.output_data.get('rows', 0)
            self.metrics.total_columns = ingestion.output_data.get('columns', 0)
        
        # Column analysis metrics
        if 'column_analysis' in phase_results:
            analysis = phase_results['column_analysis']
            self.metrics.medication_columns_found = len(
                analysis.output_data.get('medication_columns', [])
            )
        
        # Medication extraction metrics
        if 'medication_extraction' in phase_results:
            extraction = phase_results['medication_extraction']
            self.metrics.unique_medications = extraction.output_data.get('unique_medications', 0)
            self.metrics.total_medications_extracted = extraction.metrics.get('total_medications_found', 0)
            self.metrics.extraction_time_seconds = extraction.metrics.get('extraction_time_seconds', 0)
        
        # LLM classification metrics
        if 'llm_classification' in phase_results:
            classification = phase_results['llm_classification']
            self.metrics.medications_classified = classification.output_data.get('classified', 0)
            self.metrics.unclassified_medications = classification.output_data.get('unclassified', 0)
            self.metrics.average_confidence = classification.output_data.get('confidence', 0.0)
            self.metrics.classification_time_seconds = classification.metrics.get('classification_time_seconds', 0)
            
            # Parse drug class distribution
            if 'classifications_by_class' in classification.output_data:
                self.metrics.drug_class_distribution = classification.output_data['classifications_by_class']
        
        # Validation metrics
        if 'validation' in phase_results:
            validation = phase_results['validation']
            self.metrics.valid_medications = validation.output_data.get('valid_medications', 0)
            self.metrics.invalid_medications = validation.output_data.get('invalid_medications', 0)
            
            total_validated = self.metrics.valid_medications + self.metrics.invalid_medications
            if self.metrics.unique_medications > 0:
                self.metrics.validation_coverage = total_validated / self.metrics.unique_medications
        
        # Web research metrics
        if 'web_research' in phase_results:
            research = phase_results['web_research']
            self.metrics.medications_researched = research.output_data.get('medications_researched', 0)
            self.metrics.web_research_time_seconds = research.metrics.get('research_time_seconds', 0)
        
        # Calculate total processing time
        self.metrics.processing_time_seconds = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("metrics_calculated", summary=self.metrics.get_summary())
        
        return self.metrics
    
    def calculate_from_data(
        self,
        dataframe: Any,
        extraction_results: Optional[Dict[str, Any]] = None,
        classification_results: Optional[Dict[str, Any]] = None,
        validation_results: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """
        Calculate metrics from raw data.
        
        Args:
            dataframe: Input dataframe
            extraction_results: Medication extraction results
            classification_results: Classification results
            validation_results: Validation results
            
        Returns:
            Calculated quality metrics
        """
        logger.info("calculating_metrics_from_data")
        
        # Basic data metrics
        if dataframe is not None:
            self.metrics.total_rows = len(dataframe)
            self.metrics.total_columns = len(dataframe.columns)
        
        # Extraction metrics
        if extraction_results:
            for column, result in extraction_results.items():
                self.metrics.total_medications_extracted += result.get('total_extracted', 0)
                self.metrics.unique_medications = max(
                    self.metrics.unique_medications,
                    result.get('unique_count', 0)
                )
                self.metrics.normalized_medications += result.get('normalized_count', 0)
                self.metrics.variants_identified += len(result.get('variants', {}))
        
        # Classification metrics
        if classification_results:
            self._process_classification_results(classification_results)
        
        # Validation metrics
        if validation_results:
            self._process_validation_results(validation_results)
        
        # Calculate total processing time
        self.metrics.processing_time_seconds = (datetime.now() - self.start_time).total_seconds()
        
        return self.metrics
    
    def _process_classification_results(self, results: Dict[str, Any]):
        """Process classification results for metrics."""
        confidence_scores = []
        
        for med, result in results.items():
            if isinstance(result, dict):
                confidence = result.get('confidence', 0.0)
                confidence_scores.append(confidence)
                
                if confidence > 0:
                    self.metrics.medications_classified += 1
                    
                    if confidence > 0.8:
                        self.metrics.high_confidence_classifications += 1
                    elif confidence > 0.5:
                        self.metrics.medium_confidence_classifications += 1
                    else:
                        self.metrics.low_confidence_classifications += 1
                    
                    # Track drug class distribution
                    drug_class = result.get('primary_class', 'unknown')
                    self.metrics.drug_class_distribution[drug_class] = \
                        self.metrics.drug_class_distribution.get(drug_class, 0) + 1
                else:
                    self.metrics.unclassified_medications += 1
        
        # Calculate average confidence
        if confidence_scores:
            self.metrics.average_confidence = statistics.mean(confidence_scores)
    
    def _process_validation_results(self, results: Dict[str, Any]):
        """Process validation results for metrics."""
        for med, result in results.items():
            if isinstance(result, dict):
                if result.get('valid', False):
                    self.metrics.valid_medications += 1
                else:
                    self.metrics.invalid_medications += 1
    
    def add_custom_metric(self, name: str, value: Any):
        """Add a custom metric."""
        if not hasattr(self.metrics, 'custom_metrics'):
            self.metrics.custom_metrics = {}
        self.metrics.custom_metrics[name] = value
        
        logger.debug("custom_metric_added", name=name, value=value)
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on metrics."""
        recommendations = []
        
        # Check extraction rate
        if self._get_extraction_rate() < 0.5:
            recommendations.append(
                "Low extraction rate detected. Consider reviewing column detection settings."
            )
        
        # Check classification confidence
        if self.metrics.average_confidence < 0.6:
            recommendations.append(
                "Low average classification confidence. Consider using additional context or validation."
            )
        
        # Check unclassified medications
        if self.metrics.unclassified_medications > self.metrics.medications_classified:
            recommendations.append(
                "High number of unclassified medications. Review classification criteria."
            )
        
        # Check validation coverage
        if self.metrics.validation_coverage < 0.5:
            recommendations.append(
                "Low validation coverage. Consider expanding validation rules."
            )
        
        # Check performance
        if self.metrics.processing_time_seconds > 300:  # 5 minutes
            recommendations.append(
                "Long processing time. Consider optimizing pipeline or using batch processing."
            )
        
        return recommendations
    
    def _get_extraction_rate(self) -> float:
        """Get extraction rate."""
        if self.metrics.total_rows == 0:
            return 0.0
        return self.metrics.total_medications_extracted / self.metrics.total_rows