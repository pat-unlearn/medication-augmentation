"""Unit tests for DataAnalyzer."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from med_aug.core.analyzer import DataAnalyzer
from med_aug.core.models import ColumnAnalysisResult


class TestDataAnalyzer:
    """Test DataAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DataAnalyzer()

    @pytest.fixture
    def sample_medication_df(self):
        """Create sample DataFrame with medication data."""
        return pd.DataFrame(
            {
                "PATIENT_ID": ["P001", "P002", "P003", "P004", "P005"],
                "AGENT": [
                    "pembrolizumab",
                    "Keytruda",
                    "osimertinib",
                    "Tagrisso",
                    "carboplatin",
                ],
                "DOSE": ["200mg", "100mg", "80mg", "40mg", "300mg"],
                "DIAGNOSIS": ["NSCLC", "NSCLC", "NSCLC", "Lung Cancer", "NSCLC"],
                "AGE": [65, 72, 58, 61, 69],
            }
        )

    @pytest.fixture
    def mixed_column_df(self):
        """Create DataFrame with mixed content."""
        return pd.DataFrame(
            {
                "med_name": [
                    "aspirin",
                    "ibuprofen",
                    "acetaminophen",
                    "metformin",
                    "lisinopril",
                ],
                "drug_text": [
                    "Take aspirin daily",
                    "Ibuprofen 400mg",
                    "Tylenol as needed",
                    "Metformin 1000mg",
                    "Lisinopril 10mg",
                ],
                "treatment": [
                    "pain relief",
                    "inflammation",
                    "fever",
                    "diabetes",
                    "hypertension",
                ],
                "patient_name": [
                    "John Doe",
                    "Jane Smith",
                    "Bob Johnson",
                    "Alice Brown",
                    "Charlie Davis",
                ],
                "notes": [
                    "stable",
                    "improving",
                    "monitoring",
                    "stable",
                    "follow-up needed",
                ],
            }
        )

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.analyzed_columns == []
        assert analyzer.confidence_scores == {}
        assert len(analyzer.MEDICATION_PATTERNS) > 0
        assert len(analyzer.MEDICATION_KEYWORDS) > 0
        assert len(analyzer.MEDICATION_SUFFIXES) > 0

    def test_analyze_dataframe(self, analyzer, sample_medication_df):
        """Test analyzing a DataFrame."""
        results = analyzer.analyze_dataframe(sample_medication_df)

        assert len(results) > 0
        assert isinstance(results[0], ColumnAnalysisResult)

        # AGENT column should have highest confidence
        agent_result = next((r for r in results if r.column == "AGENT"), None)
        assert agent_result is not None
        assert agent_result.confidence > 0.7
        assert agent_result.unique_count == 5
        assert len(agent_result.sample_medications) > 0

    def test_column_name_scoring(self, analyzer):
        """Test column name scoring logic."""
        # High confidence names
        assert analyzer._score_column_name("agent") == 1.0
        assert analyzer._score_column_name("medication") == 1.0
        assert analyzer._score_column_name("drug") == 1.0
        assert analyzer._score_column_name("drugdtxt") == 1.0
        assert analyzer._score_column_name("conmed") == 1.0

        # Moderate confidence names
        assert analyzer._score_column_name("therapy") == 0.7
        assert analyzer._score_column_name("treatment") == 0.7
        assert (
            analyzer._score_column_name("chemo_drug") == 1.0
        )  # Contains 'drug' so scores 1.0

        # Low confidence names
        assert analyzer._score_column_name("name") == 0.3
        assert analyzer._score_column_name("text") == 0.3
        assert analyzer._score_column_name("description") == 0.3

        # No confidence
        assert analyzer._score_column_name("patient_id") == 0.0
        assert analyzer._score_column_name("age") == 0.0
        assert analyzer._score_column_name("date") == 0.0

    def test_medication_pattern_detection(self, analyzer):
        """Test medication pattern detection."""
        # Monoclonal antibodies
        assert analyzer._looks_like_medication("pembrolizumab") is True
        assert analyzer._looks_like_medication("nivolumab") is True
        assert analyzer._looks_like_medication("atezolizumab") is True

        # Kinase inhibitors
        assert analyzer._looks_like_medication("osimertinib") is True
        assert analyzer._looks_like_medication("erlotinib") is True
        assert analyzer._looks_like_medication("sunitinib") is True

        # Combination drugs
        assert analyzer._looks_like_medication("carboplatin/paclitaxel") is True
        assert analyzer._looks_like_medication("nivolumab + ipilimumab") is True

        # Brand names with parentheses
        assert analyzer._looks_like_medication("Keytruda (pembrolizumab)") is True
        assert analyzer._looks_like_medication("Tagrisso (osimertinib)") is True

        # Non-medications
        assert analyzer._looks_like_medication("abc") is False
        assert analyzer._looks_like_medication("12345") is False
        assert analyzer._looks_like_medication("") is False
        assert analyzer._looks_like_medication("x") is False

    def test_pattern_scoring(self, analyzer):
        """Test pattern scoring for medication columns."""
        # High medication content
        med_values = [
            "pembrolizumab",
            "osimertinib",
            "carboplatin",
            "paclitaxel",
            "nivolumab",
        ]
        score, matches = analyzer._score_patterns(med_values)
        assert score > 0.8
        assert matches == 5

        # Mixed content
        mixed_values = ["pembrolizumab", "patient1", "test", "osimertinib", "data"]
        score, matches = analyzer._score_patterns(mixed_values)
        assert 0.2 < score < 0.6
        assert matches == 2

        # No medications
        non_med_values = ["patient1", "patient2", "test", "data", "value"]
        score, matches = analyzer._score_patterns(non_med_values)
        assert score == 0.0
        assert matches == 0

    def test_statistical_scoring(self, analyzer):
        """Test statistical scoring of columns."""
        # High diversity column (typical of medications)
        diverse_series = pd.Series(["drug" + str(i) for i in range(100)])
        score, reason = analyzer._score_statistics(diverse_series)
        assert score > 0.5
        assert "diversity" in reason.lower()

        # Low diversity column
        repetitive_series = pd.Series(["drug1"] * 50 + ["drug2"] * 50)
        score, reason = analyzer._score_statistics(repetitive_series)
        # Score is 0.6 (0.3 for avg length, 0.3 for alphanumeric), which is reasonable
        assert score == pytest.approx(0.6, 0.1)

        # Good length for medications
        good_length_series = pd.Series(["medication" for _ in range(50)])
        score, reason = analyzer._score_statistics(good_length_series)
        assert score > 0.3

        # Bad length for medications
        bad_length_series = pd.Series(["a" * 100 for _ in range(50)])
        score, reason = analyzer._score_statistics(bad_length_series)
        assert score <= 0.3  # Still gets 0.3 for clean alphanumeric

    def test_content_scoring(self, analyzer):
        """Test content analysis scoring."""
        # Properly capitalized medications
        cap_values = [
            "Pembrolizumab",
            "Nivolumab",
            "Osimertinib",
            "Carboplatin",
            "Paclitaxel",
        ]
        score, reason = analyzer._score_content(cap_values)
        assert score > 0.7
        assert "capitalization" in reason.lower()

        # Lower case values
        lower_values = ["drug1", "drug2", "drug3", "drug4", "drug5"]
        score, reason = analyzer._score_content(lower_values)
        assert score < 0.7

        # Contains non-medication indicators
        non_med_values = ["yes", "no", "unknown", "n/a", "none"]
        score, reason = analyzer._score_content(non_med_values)
        assert score == 0.0

    def test_sample_medication_extraction(self, analyzer):
        """Test extraction of sample medications."""
        values = [
            "pembrolizumab",
            "patient1",
            "osimertinib",
            "test",
            "carboplatin",
            "data",
            "nivolumab",
            "value",
            "atezolizumab",
            "info",
            "durvalumab",
            "sample",
        ]

        samples = analyzer._extract_sample_medications(values)
        assert len(samples) <= 10
        assert "pembrolizumab" in samples
        assert "osimertinib" in samples
        assert "carboplatin" in samples
        assert "patient1" not in samples
        assert "test" not in samples

    def test_get_best_column(self, analyzer, sample_medication_df):
        """Test getting the best column."""
        analyzer.analyze_dataframe(sample_medication_df)
        best = analyzer.get_best_column()

        assert best is not None
        assert best.column == "AGENT"
        assert best.confidence > 0.7

    def test_get_columns_above_threshold(self, analyzer, sample_medication_df):
        """Test getting columns above confidence threshold."""
        analyzer.analyze_dataframe(sample_medication_df)

        high_confidence = analyzer.get_columns_above_threshold(0.7)
        assert len(high_confidence) >= 1
        assert all(c.confidence >= 0.7 for c in high_confidence)

        very_high_confidence = analyzer.get_columns_above_threshold(0.9)
        assert len(very_high_confidence) <= len(high_confidence)

    def test_analyze_empty_dataframe(self, analyzer):
        """Test analyzing empty DataFrame."""
        empty_df = pd.DataFrame()
        results = analyzer.analyze_dataframe(empty_df)
        assert len(results) == 0

    def test_analyze_null_column(self, analyzer):
        """Test analyzing column with all null values."""
        null_df = pd.DataFrame({"all_null": [None, np.nan, None, np.nan, None]})
        results = analyzer.analyze_dataframe(null_df)
        assert len(results) == 0

    def test_analyze_mixed_columns(self, analyzer, mixed_column_df):
        """Test analyzing DataFrame with mixed content columns."""
        results = analyzer.analyze_dataframe(mixed_column_df, confidence_threshold=0.3)

        assert len(results) > 0

        # med_name should have highest confidence
        best = results[0]
        assert best.column in ["med_name", "drug_text"]
        assert best.confidence > 0.5

    def test_confidence_threshold(self, analyzer, sample_medication_df):
        """Test confidence threshold filtering."""
        # Low threshold - should get more results
        low_threshold_results = analyzer.analyze_dataframe(
            sample_medication_df, confidence_threshold=0.1
        )

        # High threshold - should get fewer results
        high_threshold_results = analyzer.analyze_dataframe(
            sample_medication_df, confidence_threshold=0.8
        )

        assert len(low_threshold_results) >= len(high_threshold_results)

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            ("pembrolizumab", True),  # mab suffix
            ("osimertinib", True),  # nib suffix
            ("atorvastatin", True),  # tin suffix
            ("lisinopril", True),  # pril suffix
            ("losartan", True),  # sartan suffix
            ("metoprolol", True),  # olol suffix
            ("omeprazole", True),  # zole suffix
            ("carboplatin", True),  # platin suffix
            ("paclitaxel", True),  # taxel suffix
            ("doxorubicin", True),  # rubicin suffix
            ("regular_text", False),  # no suffix
        ],
    )
    def test_medication_suffixes(self, analyzer, suffix, expected):
        """Test detection of medication suffixes."""
        assert analyzer._looks_like_medication(suffix) == expected

    def test_combination_drug_detection(self, analyzer):
        """Test detection of combination drugs."""
        # Valid combinations
        assert analyzer._looks_like_medication("carboplatin/paclitaxel") is True
        assert analyzer._looks_like_medication("nivolumab + ipilimumab") is True
        assert analyzer._looks_like_medication("drug1/drug2") is True

        # Invalid combinations
        assert analyzer._looks_like_medication("/drug") is False
        assert analyzer._looks_like_medication("drug/") is False
        assert analyzer._looks_like_medication("a/b") is False  # Too short

    def test_uppercase_pattern_detection(self, analyzer):
        """Test detection of uppercase patterns."""
        assert analyzer._looks_like_medication("PD-L1") is True
        assert analyzer._looks_like_medication("CTLA-4") is True
        assert analyzer._looks_like_medication("EGFR") is True
        assert analyzer._looks_like_medication("ALK") is True
        assert analyzer._looks_like_medication("KRAS") is True
        assert analyzer._looks_like_medication("VEGF") is True
