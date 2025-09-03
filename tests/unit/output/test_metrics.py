"""Unit tests for metrics calculation."""

import pytest
from datetime import datetime

from med_aug.output.metrics import (
    QualityMetrics,
    MetricsCalculator
)
from med_aug.pipeline.phases import PhaseResult, PhaseStatus


class TestQualityMetrics:
    """Test quality metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = QualityMetrics()
        
        assert metrics.total_rows == 0
        assert metrics.unique_medications == 0
        assert metrics.average_confidence == 0.0
        assert metrics.drug_class_distribution == {}
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = QualityMetrics(
            total_rows=100,
            unique_medications=25,
            medications_classified=20,
            average_confidence=0.85
        )
        
        metrics_dict = metrics.to_dict()
        
        assert 'data_quality' in metrics_dict
        assert metrics_dict['data_quality']['total_rows'] == 100
        assert 'extraction' in metrics_dict
        assert metrics_dict['extraction']['unique_medications'] == 25
        assert 'classification' in metrics_dict
        assert metrics_dict['classification']['average_confidence'] == 0.85
    
    def test_extraction_rate_calculation(self):
        """Test extraction rate calculation."""
        metrics = QualityMetrics(
            total_rows=100,
            total_medications_extracted=75
        )
        
        rate = metrics._calculate_extraction_rate()
        assert rate == 0.75
        
        # Test with zero rows
        metrics.total_rows = 0
        rate = metrics._calculate_extraction_rate()
        assert rate == 0.0
    
    def test_classification_rate_calculation(self):
        """Test classification rate calculation."""
        metrics = QualityMetrics(
            unique_medications=50,
            medications_classified=40
        )
        
        rate = metrics._calculate_classification_rate()
        assert rate == 0.8
        
        # Test with zero medications
        metrics.unique_medications = 0
        rate = metrics._calculate_classification_rate()
        assert rate == 0.0
    
    def test_confidence_level_distribution(self):
        """Test confidence level distribution calculation."""
        metrics = QualityMetrics(
            high_confidence_classifications=30,
            medium_confidence_classifications=15,
            low_confidence_classifications=5
        )
        
        distribution = metrics._get_confidence_level_distribution()
        
        assert distribution['high'] == 0.6
        assert distribution['medium'] == 0.3
        assert distribution['low'] == 0.1
    
    def test_get_summary(self):
        """Test getting metrics summary."""
        metrics = QualityMetrics(
            unique_medications=100,
            medications_classified=85,
            average_confidence=0.78,
            processing_time_seconds=25.5
        )
        
        summary = metrics.get_summary()
        
        assert summary['Total Medications'] == 100
        assert 'Extraction Rate' in summary
        assert 'Classification Rate' in summary
        assert summary['Average Confidence'] == "0.78"
        assert summary['Processing Time'] == "25.5s"
    
    def test_quality_score_calculation(self):
        """Test overall quality score calculation."""
        # Excellent score
        metrics = QualityMetrics(
            total_rows=100,
            total_medications_extracted=90,
            unique_medications=50,
            medications_classified=48,
            average_confidence=0.92
        )
        
        score = metrics._calculate_quality_score()
        assert "Excellent" in score
        
        # Good score
        metrics.average_confidence = 0.65
        metrics.medications_classified = 35
        score = metrics._calculate_quality_score()
        assert "Good" in score or "Fair" in score


class TestMetricsCalculator:
    """Test metrics calculator."""
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calculator = MetricsCalculator()
        
        assert calculator.metrics is not None
        assert calculator.start_time is not None
        assert isinstance(calculator.metrics, QualityMetrics)
    
    def test_calculate_from_pipeline_results(self):
        """Test calculating metrics from pipeline results."""
        calculator = MetricsCalculator()
        
        # Create mock phase results
        phase_results = {
            'data_ingestion': PhaseResult(
                phase_name='data_ingestion',
                status=PhaseStatus.COMPLETED,
                start_time=datetime.now(),
                output_data={'rows': 1000, 'columns': 10}
            ),
            'column_analysis': PhaseResult(
                phase_name='column_analysis',
                status=PhaseStatus.COMPLETED,
                start_time=datetime.now(),
                output_data={'medication_columns': ['med1', 'med2']}
            ),
            'medication_extraction': PhaseResult(
                phase_name='medication_extraction',
                status=PhaseStatus.COMPLETED,
                start_time=datetime.now(),
                output_data={'unique_medications': 150},
                metrics={'total_medications_found': 500, 'extraction_time_seconds': 5.2}
            ),
            'llm_classification': PhaseResult(
                phase_name='llm_classification',
                status=PhaseStatus.COMPLETED,
                start_time=datetime.now(),
                output_data={
                    'classified': 140,
                    'unclassified': 10,
                    'confidence': 0.88,
                    'classifications_by_class': {'immunotherapy': 50, 'chemotherapy': 90}
                },
                metrics={'classification_time_seconds': 12.3}
            )
        }
        
        metrics = calculator.calculate_from_pipeline_results(phase_results)
        
        assert metrics.total_rows == 1000
        assert metrics.total_columns == 10
        assert metrics.medication_columns_found == 2
        assert metrics.unique_medications == 150
        assert metrics.medications_classified == 140
        assert metrics.unclassified_medications == 10
        assert metrics.average_confidence == 0.88
        assert metrics.drug_class_distribution['immunotherapy'] == 50
        assert metrics.drug_class_distribution['chemotherapy'] == 90
    
    def test_calculate_from_data(self):
        """Test calculating metrics from raw data."""
        import pandas as pd
        
        calculator = MetricsCalculator()
        
        # Create mock data
        df = pd.DataFrame({
            'col1': range(100),
            'medications': ['drug' + str(i) for i in range(100)]
        })
        
        extraction_results = {
            'medications': {
                'total_extracted': 95,
                'unique_count': 80,
                'normalized_count': 75,
                'variants': {'drug1': ['Drug1', 'DRUG1']}
            }
        }
        
        classification_results = {
            'drug1': {'confidence': 0.95, 'primary_class': 'class_a'},
            'drug2': {'confidence': 0.75, 'primary_class': 'class_b'},
            'drug3': {'confidence': 0.4, 'primary_class': 'class_a'},
            'drug4': {'confidence': 0.0, 'primary_class': 'unknown'}
        }
        
        metrics = calculator.calculate_from_data(
            df,
            extraction_results,
            classification_results
        )
        
        assert metrics.total_rows == 100
        assert metrics.total_columns == 2
        assert metrics.unique_medications == 80
        assert metrics.medications_classified == 3  # confidence > 0
        assert metrics.unclassified_medications == 1  # confidence == 0
        assert metrics.high_confidence_classifications == 1
        assert metrics.medium_confidence_classifications == 1
        assert metrics.low_confidence_classifications == 1
    
    def test_add_custom_metric(self):
        """Test adding custom metrics."""
        calculator = MetricsCalculator()
        
        calculator.add_custom_metric('custom_score', 0.95)
        calculator.add_custom_metric('custom_count', 42)
        
        assert hasattr(calculator.metrics, 'custom_metrics')
        assert calculator.metrics.custom_metrics['custom_score'] == 0.95
        assert calculator.metrics.custom_metrics['custom_count'] == 42
    
    def test_get_recommendations(self):
        """Test getting recommendations based on metrics."""
        calculator = MetricsCalculator()
        
        # Set up metrics that trigger recommendations
        calculator.metrics.total_rows = 100
        calculator.metrics.total_medications_extracted = 30  # Low extraction rate
        calculator.metrics.average_confidence = 0.45  # Low confidence
        calculator.metrics.unclassified_medications = 60
        calculator.metrics.medications_classified = 40
        calculator.metrics.validation_coverage = 0.3
        calculator.metrics.processing_time_seconds = 400  # Long processing
        
        recommendations = calculator.get_recommendations()
        
        assert len(recommendations) > 0
        assert any('extraction rate' in r.lower() for r in recommendations)
        assert any('confidence' in r.lower() for r in recommendations)
        assert any('unclassified' in r.lower() for r in recommendations)
        assert any('validation' in r.lower() for r in recommendations)
        assert any('processing time' in r.lower() for r in recommendations)
    
    def test_process_classification_results(self):
        """Test processing classification results."""
        calculator = MetricsCalculator()
        
        results = {
            'drug1': {'confidence': 0.95, 'primary_class': 'immunotherapy'},
            'drug2': {'confidence': 0.65, 'primary_class': 'chemotherapy'},
            'drug3': {'confidence': 0.3, 'primary_class': 'immunotherapy'},
            'drug4': {'confidence': 0.0}
        }
        
        calculator._process_classification_results(results)
        
        metrics = calculator.metrics
        
        assert metrics.medications_classified == 3
        assert metrics.unclassified_medications == 1
        assert metrics.high_confidence_classifications == 1
        assert metrics.medium_confidence_classifications == 1
        assert metrics.low_confidence_classifications == 1
        assert metrics.drug_class_distribution['immunotherapy'] == 2
        assert metrics.drug_class_distribution['chemotherapy'] == 1
        assert metrics.average_confidence == pytest.approx(0.475, 0.01)
    
    def test_process_validation_results(self):
        """Test processing validation results."""
        calculator = MetricsCalculator()
        
        results = {
            'drug1': {'valid': True},
            'drug2': {'valid': True},
            'drug3': {'valid': False},
            'drug4': {'valid': True}
        }
        
        calculator._process_validation_results(results)
        
        assert calculator.metrics.valid_medications == 3
        assert calculator.metrics.invalid_medications == 1