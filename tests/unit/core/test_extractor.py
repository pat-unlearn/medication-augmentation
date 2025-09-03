"""Unit tests for MedicationExtractor."""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import json
from med_aug.core.extractor import MedicationExtractor, ExtractionResult
from med_aug.core.models import Medication, MedicationType


class TestMedicationExtractor:
    """Test MedicationExtractor functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return MedicationExtractor()
    
    @pytest.fixture
    def sample_medication_series(self):
        """Create sample medication Series."""
        return pd.Series([
            'pembrolizumab',
            'Keytruda',
            'osimertinib 80mg',
            'Tagrisso',
            'carboplatin/paclitaxel',
            'nivolumab + ipilimumab',
            'Opdivo',
            'bevacizumab (Avastin)',
            'docetaxel 75mg/m2',
            'pemetrexed',
            None,  # Test null handling
            'n/a',  # Test n/a handling
            'pembrolizumab',  # Duplicate
            'PEMBROLIZUMAB',  # Case variation
        ])
    
    @pytest.fixture
    def combination_drug_series(self):
        """Create series with combination drugs."""
        return pd.Series([
            'carboplatin/paclitaxel',
            'nivolumab + ipilimumab',
            'pembrolizumab and chemotherapy',
            'bevacizumab with carboplatin',
            'cisplatin-etoposide',
        ])
    
    @pytest.fixture
    def dosage_form_series(self):
        """Create series with dosages and forms."""
        return pd.Series([
            'pembrolizumab 200mg',
            'nivolumab 3mg/kg',
            'osimertinib 80mg tablets',
            'docetaxel 75mg/m2 iv',
            'paclitaxel 175mg/m2 infusion',
            'metformin 1000mg extended release',
            'lisinopril 10mg oral',
            'atorvastatin 40mg ER',
        ])
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.extracted_medications == []
        assert extractor.normalization_cache == {}
        assert len(extractor.REMOVE_PATTERNS) > 0
        assert len(extractor.COMBINATION_SEPARATORS) > 0
        assert len(extractor.BRAND_GENERIC_MAP) > 0
    
    def test_extract_from_series(self, extractor, sample_medication_series):
        """Test extracting medications from a Series."""
        result = extractor.extract_from_series(sample_medication_series, 'test_column')
        
        assert isinstance(result, ExtractionResult)
        assert result.total_rows == len(sample_medication_series)
        assert result.unique_medications > 0
        assert result.column_name == 'test_column'
        assert len(result.normalized_medications) > 0
        assert len(result.frequency_map) > 0
        assert len(result.variants_map) > 0
    
    def test_normalization(self, extractor):
        """Test medication name normalization."""
        # Brand to generic mapping
        assert extractor._normalize_medication('Keytruda') == 'Pembrolizumab'
        assert extractor._normalize_medication('keytruda') == 'Pembrolizumab'
        assert extractor._normalize_medication('Tagrisso') == 'Osimertinib'
        assert extractor._normalize_medication('Opdivo') == 'Nivolumab'
        
        # Dosage removal
        assert extractor._normalize_medication('pembrolizumab 200mg') == 'Pembrolizumab'
        assert extractor._normalize_medication('nivolumab 3mg/kg') == 'Nivolumab'
        assert extractor._normalize_medication('osimertinib 80mg tablets') == 'Osimertinib'
        
        # Form removal
        assert extractor._normalize_medication('docetaxel injection') == 'Docetaxel'
        assert extractor._normalize_medication('paclitaxel solution') == 'Paclitaxel'
        assert extractor._normalize_medication('metformin tablets') == 'Metformin'
        
        # Case normalization
        assert extractor._normalize_medication('PEMBROLIZUMAB') == 'Pembrolizumab'
        assert extractor._normalize_medication('pembrolizumab') == 'Pembrolizumab'
        assert extractor._normalize_medication('PeMbRoLiZuMaB') == 'Pembrolizumab'
        
        # Salt removal
        assert extractor._normalize_medication('lisinopril sodium') == 'Lisinopril'
        assert extractor._normalize_medication('metformin hcl') == 'Metformin'
        assert extractor._normalize_medication('morphine sulfate') == 'Morphine'
    
    def test_combination_drug_extraction(self, extractor, combination_drug_series):
        """Test extraction of combination drugs."""
        result = extractor.extract_from_series(combination_drug_series, 'combinations')
        
        # Should extract individual components
        assert 'Carboplatin' in result.frequency_map
        assert 'Paclitaxel' in result.frequency_map
        assert 'Nivolumab' in result.frequency_map
        assert 'Ipilimumab' in result.frequency_map
        assert 'Pembrolizumab' in result.frequency_map
        assert 'Bevacizumab' in result.frequency_map
        assert 'Cisplatin' in result.frequency_map
        assert 'Etoposide' in result.frequency_map
    
    def test_dosage_form_removal(self, extractor, dosage_form_series):
        """Test removal of dosage and form information."""
        result = extractor.extract_from_series(dosage_form_series, 'dosages')
        
        # All should be normalized without dosages/forms
        for med in result.frequency_map.keys():
            assert 'mg' not in med.lower()
            assert 'tablets' not in med.lower()
            assert 'oral' not in med.lower()
            assert 'infusion' not in med.lower()
            assert '/kg' not in med.lower()
            assert '/m2' not in med.lower()
    
    def test_frequency_counting(self, extractor):
        """Test frequency counting of medications."""
        series = pd.Series([
            'pembrolizumab',
            'Pembrolizumab',
            'PEMBROLIZUMAB',
            'Keytruda',  # Maps to pembrolizumab
            'nivolumab',
            'Opdivo',  # Maps to nivolumab
            'osimertinib',
        ])
        
        result = extractor.extract_from_series(series, 'freq_test')
        
        # Pembrolizumab should have count of 4 (3 variations + Keytruda)
        assert result.frequency_map.get('Pembrolizumab', 0) == 4
        # Nivolumab should have count of 2 (nivolumab + Opdivo)
        assert result.frequency_map.get('Nivolumab', 0) == 2
        # Osimertinib should have count of 1
        assert result.frequency_map.get('Osimertinib', 0) == 1
    
    def test_variant_tracking(self, extractor):
        """Test tracking of medication variants."""
        series = pd.Series([
            'pembrolizumab',
            'Pembrolizumab 200mg',
            'PEMBROLIZUMAB',
            'Keytruda',
            'keytruda 100mg',
        ])
        
        result = extractor.extract_from_series(series, 'variant_test')
        
        # Should track all unique variants for pembrolizumab
        variants = result.get_variants_for('Pembrolizumab')
        assert len(variants) > 0
        assert 'pembrolizumab' in variants or 'Pembrolizumab 200mg' in variants
        assert 'Keytruda' in variants or 'keytruda 100mg' in variants
    
    def test_null_and_invalid_handling(self, extractor):
        """Test handling of null and invalid values."""
        series = pd.Series([
            None,
            'n/a',
            'N/A',
            'none',
            'unknown',
            'null',
            '',
            '   ',
            'pembrolizumab',  # One valid medication
        ])
        
        result = extractor.extract_from_series(series, 'null_test')
        
        # Should only extract the valid medication
        assert result.unique_medications == 1
        assert 'Pembrolizumab' in result.frequency_map
        assert result.frequency_map['Pembrolizumab'] == 1
    
    def test_deduplication(self, extractor):
        """Test medication deduplication."""
        medications = [
            'pembrolizumab',
            'Pembrolizumab',
            'PEMBROLIZUMAB',
            'Keytruda',
            'nivolumab',
            'Opdivo',
            'nivolumab 3mg/kg',
        ]
        
        deduplicated = extractor.deduplicate_medications(medications)
        
        # Should keep first occurrence of each unique medication
        assert len(deduplicated) <= 3  # pembrolizumab, nivolumab, and possibly one variant
    
    def test_medication_statistics(self, extractor, sample_medication_series):
        """Test statistical analysis of medications."""
        result = extractor.extract_from_series(sample_medication_series, 'stats_test')
        stats = extractor.get_medication_statistics(result)
        
        assert 'total_rows' in stats
        assert 'unique_medications' in stats
        assert 'total_occurrences' in stats
        assert 'coverage_rate' in stats
        assert 'top_10_medications' in stats
        assert 'frequency_stats' in stats
        assert 'variant_stats' in stats
        
        # Check frequency stats
        assert 'min' in stats['frequency_stats']
        assert 'max' in stats['frequency_stats']
        assert 'mean' in stats['frequency_stats']
        assert 'median' in stats['frequency_stats']
        
        # Check variant stats
        assert 'medications_with_variants' in stats['variant_stats']
        assert 'max_variants' in stats['variant_stats']
        assert 'avg_variants' in stats['variant_stats']
    
    def test_get_top_medications(self, extractor):
        """Test getting top medications by frequency."""
        series = pd.Series([
            'drug1', 'drug1', 'drug1', 'drug1',  # 4 times
            'drug2', 'drug2', 'drug2',  # 3 times
            'drug3', 'drug3',  # 2 times
            'drug4',  # 1 time
        ])
        
        result = extractor.extract_from_series(series, 'top_test')
        top_3 = result.get_top_medications(3)
        
        assert len(top_3) == 3
        assert top_3[0][0] == 'Drug1'
        assert top_3[0][1] == 4
        assert top_3[1][0] == 'Drug2'
        assert top_3[1][1] == 3
        assert top_3[2][0] == 'Drug3'
        assert top_3[2][1] == 2
    
    def test_medication_type_determination(self, extractor):
        """Test determination of medication types."""
        # Brand names
        assert extractor._determine_medication_type('keytruda', ['Keytruda']) == MedicationType.BRAND
        assert extractor._determine_medication_type('opdivo', ['Opdivo']) == MedicationType.BRAND
        
        # Combinations
        assert extractor._determine_medication_type('Carboplatin/Paclitaxel', []) == MedicationType.COMBINATION
        assert extractor._determine_medication_type('Drug1+Drug2', []) == MedicationType.COMBINATION
        
        # Clinical trial
        assert extractor._determine_medication_type('ABC-123', []) == MedicationType.CLINICAL_TRIAL
        assert extractor._determine_medication_type('XYZ-4567', []) == MedicationType.CLINICAL_TRIAL
        
        # Abbreviations
        assert extractor._determine_medication_type('EGFR', []) == MedicationType.ABBREVIATION
        assert extractor._determine_medication_type('PD-L1', []) == MedicationType.ABBREVIATION
        
        # Generic (default)
        assert extractor._determine_medication_type('pembrolizumab', []) == MedicationType.GENERIC
        assert extractor._determine_medication_type('osimertinib', []) == MedicationType.GENERIC
    
    def test_create_medication_objects(self, extractor, sample_medication_series):
        """Test creation of Medication objects."""
        result = extractor.extract_from_series(sample_medication_series, 'med_obj_test')
        medications = extractor.create_medication_objects(result, 'test_source')
        
        assert len(medications) > 0
        assert all(isinstance(med, Medication) for med in medications)
        
        # Check medication properties
        for med in medications:
            assert med.name is not None
            assert med.type is not None
            assert 0 <= med.confidence <= 1
            assert med.source == 'test_source'
            assert 'frequency' in med.metadata
            assert 'variants' in med.metadata
            assert 'column' in med.metadata
            assert isinstance(med.discovered_at, datetime)
    
    def test_export_medications_csv(self, extractor, sample_medication_series):
        """Test exporting medications to CSV."""
        result = extractor.extract_from_series(sample_medication_series, 'export_test')
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            extractor.export_medications(result, tmp.name, format='csv')
            
            # Read back and verify
            df = pd.read_csv(tmp.name)
            assert 'normalized_name' in df.columns
            assert 'frequency' in df.columns
            assert 'variants' in df.columns
            assert 'variant_count' in df.columns
            assert len(df) == result.unique_medications
            
            Path(tmp.name).unlink()
    
    def test_export_medications_json(self, extractor, sample_medication_series):
        """Test exporting medications to JSON."""
        result = extractor.extract_from_series(sample_medication_series, 'export_test')
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            extractor.export_medications(result, tmp.name, format='json')
            
            # Read back and verify
            with open(tmp.name, 'r') as f:
                data = json.load(f)
            
            assert isinstance(data, list)
            assert len(data) == result.unique_medications
            assert all('normalized_name' in item for item in data)
            assert all('frequency' in item for item in data)
            
            Path(tmp.name).unlink()
    
    def test_normalization_cache(self, extractor):
        """Test normalization caching."""
        # First call - not cached
        result1 = extractor._normalize_medication('pembrolizumab 200mg')
        assert 'pembrolizumab 200mg' in extractor.normalization_cache
        
        # Second call - should use cache
        result2 = extractor._normalize_medication('pembrolizumab 200mg')
        assert result1 == result2
        assert len(extractor.normalization_cache) == 1
        
        # Different medication - adds to cache
        result3 = extractor._normalize_medication('nivolumab 3mg/kg')
        assert 'nivolumab 3mg/kg' in extractor.normalization_cache
        assert len(extractor.normalization_cache) == 2
    
    @pytest.mark.parametrize("text,expected", [
        ('pembrolizumab', True),
        ('yes', False),
        ('no', False),
        ('unknown', False),
        ('n/a', False),
        ('none', False),
        ('test', False),
        ('', False),
        ('ab', False),  # Too short
        ('drug-mab', True),  # Has mab suffix
        ('drug-nib', True),  # Has nib suffix
        ('keytruda', True),  # Known brand name
    ])
    def test_is_likely_medication(self, extractor, text, expected):
        """Test medication likelihood detection."""
        assert extractor._is_likely_medication(text) == expected
    
    def test_extended_release_handling(self, extractor):
        """Test handling of extended release formulations."""
        series = pd.Series([
            'metformin extended release',
            'metformin ER',
            'metformin XR',
            'metformin SR',
            'metformin LA',
        ])
        
        result = extractor.extract_from_series(series, 'er_test')
        
        # All should normalize to same medication
        assert result.unique_medications == 1
        assert 'Metformin' in result.frequency_map
        assert result.frequency_map['Metformin'] == 5
    
    def test_parenthetical_content_removal(self, extractor):
        """Test removal of parenthetical content."""
        assert extractor._normalize_medication('pembrolizumab (Keytruda)') == 'Pembrolizumab'
        assert extractor._normalize_medication('bevacizumab (Avastin)') == 'Bevacizumab'
        assert extractor._normalize_medication('drug (test info)') == 'Drug'
        assert extractor._normalize_medication('medication [extra data]') == 'Medication'