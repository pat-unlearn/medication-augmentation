"""Data analysis module for medication column detection and analysis."""

import re
from typing import List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
from .models import ColumnAnalysisResult
from .logging import get_logger, PerformanceLogger

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)


class DataAnalyzer:
    """Analyze datasets to identify medication columns."""

    # Common medication name patterns
    MEDICATION_PATTERNS = [
        r"\b[A-Z][a-z]+mab\b",  # Monoclonal antibodies
        r"\b[A-Z][a-z]*nib\b",  # Kinase inhibitors
        r"\b[A-Z][a-z]*tin\b",  # Statins and related
        r"\b[A-Z][a-z]*pril\b",  # ACE inhibitors
        r"\b[A-Z][a-z]*sartan\b",  # ARBs
        r"\b[A-Z][a-z]*olol\b",  # Beta blockers
        r"\b[A-Z][a-z]*zole\b",  # PPIs and antifungals
        r"\b[A-Z][a-z]*cillin\b",  # Penicillins
        r"\b[A-Z][a-z]*mycin\b",  # Antibiotics
        r"\b[A-Z][a-z]*platin\b",  # Platinum compounds
        r"\b[A-Z][a-z]*taxel\b",  # Taxanes
        r"\b[A-Z][a-z]*rubicin\b",  # Anthracyclines
        r"\bPLATINUM|PLATIN\b",  # Platinum compounds
        r"\bTAXEL|TAXANE\b",  # Taxanes
        r"\bPD-?[1L]-?\d*\b",  # PD-1/PD-L1 inhibitors
        r"\bCTLA-?\d*\b",  # CTLA inhibitors
        r"\bEGFR\b",  # EGFR inhibitors
        r"\bALK\b",  # ALK inhibitors
        r"\bKRAS\b",  # KRAS inhibitors
        r"\bVEGF\b",  # VEGF inhibitors
    ]

    # Keywords that indicate medication columns
    MEDICATION_KEYWORDS = [
        "drug",
        "medication",
        "agent",
        "therapy",
        "treatment",
        "chemo",
        "immuno",
        "targeted",
        "hormone",
        "biological",
        "pharmaceutical",
        "medicine",
        "therapeutic",
        "compound",
        "regimen",
        "protocol",
        "substance",
        "conmed",
        "concomitant",
    ]

    # Common medication suffixes
    MEDICATION_SUFFIXES = [
        "mab",
        "nib",
        "tin",
        "pril",
        "sartan",
        "olol",
        "zole",
        "platin",
        "taxel",
        "rubicin",
        "cillin",
        "mycin",
        "tide",
        "vir",
        "dine",
        "pine",
        "zine",
        "stat",
        "parin",
        "xaban",
    ]

    def __init__(self):
        """Initialize the analyzer."""
        self.analyzed_columns = []
        self.confidence_scores = {}
        logger.debug("analyzer_initialized")

    def analyze_file(
        self,
        file_path: Union[str, Path],
        sample_size: int = 1000,
        confidence_threshold: float = 0.5,
    ) -> List[ColumnAnalysisResult]:
        """
        Analyze dataset columns to identify medication columns.

        Args:
            file_path: Path to data file
            sample_size: Number of rows to sample for analysis
            confidence_threshold: Minimum confidence to consider a column

        Returns:
            List of column analysis results sorted by confidence
        """
        file_path = Path(file_path)

        # Load data based on file type
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, nrows=sample_size)
        elif file_path.suffix.lower() in [".txt", ".tsv"]:
            df = pd.read_csv(file_path, sep="\t", nrows=sample_size)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, nrows=sample_size)
        elif file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path).head(sample_size)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return self.analyze_dataframe(df, confidence_threshold)

    def analyze_dataframe(
        self, df: pd.DataFrame, confidence_threshold: float = 0.5
    ) -> List[ColumnAnalysisResult]:
        """
        Analyze DataFrame columns to identify medication columns.

        Args:
            df: DataFrame to analyze
            confidence_threshold: Minimum confidence to consider a column

        Returns:
            List of column analysis results sorted by confidence
        """
        results = []

        for column in df.columns:
            if df[column].dtype == "object":  # String columns only
                result = self._analyze_column(df[column], column)
                if result.confidence >= confidence_threshold:
                    results.append(result)
                    self.confidence_scores[column] = result.confidence

        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        self.analyzed_columns = results
        return results

    def _analyze_column(
        self, series: pd.Series, column_name: str
    ) -> ColumnAnalysisResult:
        """
        Analyze a single column for medication content.

        Args:
            series: Column data
            column_name: Name of the column

        Returns:
            Analysis result for the column
        """
        # Remove null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return ColumnAnalysisResult(
                column=column_name,
                confidence=0.0,
                total_count=0,
                unique_count=0,
                sample_medications=[],
                reasoning="Column contains no non-null values",
            )

        # Convert to strings and get unique values
        str_values = non_null.astype(str).str.strip()
        unique_values = str_values.unique()

        # Calculate confidence score components
        confidence_score = 0.0
        reasoning_parts = []

        # 1. Column name scoring (30% weight)
        name_score = self._score_column_name(column_name)
        confidence_score += name_score * 0.3
        if name_score > 0:
            reasoning_parts.append(f"Column name relevance: {name_score:.2f}")

        # 2. Pattern matching scoring (35% weight)
        pattern_score, pattern_matches = self._score_patterns(unique_values)
        confidence_score += pattern_score * 0.35
        if pattern_score > 0:
            reasoning_parts.append(
                f"Pattern matches: {pattern_matches} medications detected"
            )

        # 3. Statistical scoring (20% weight)
        stats_score, stats_reason = self._score_statistics(str_values)
        confidence_score += stats_score * 0.2
        if stats_reason:
            reasoning_parts.append(stats_reason)

        # 4. Content analysis scoring (15% weight)
        content_score, content_reason = self._score_content(unique_values)
        confidence_score += content_score * 0.15
        if content_reason:
            reasoning_parts.append(content_reason)

        # Get sample medications for preview
        sample_meds = self._extract_sample_medications(unique_values)

        return ColumnAnalysisResult(
            column=column_name,
            confidence=min(confidence_score, 1.0),
            total_count=len(non_null),
            unique_count=len(unique_values),
            sample_medications=sample_meds[:10],  # Limit to 10 samples
            reasoning=(
                "; ".join(reasoning_parts)
                if reasoning_parts
                else "No medication indicators found"
            ),
        )

    def _score_column_name(self, column_name: str) -> float:
        """
        Score column name for medication-related keywords.

        Args:
            column_name: Name of the column

        Returns:
            Score between 0 and 1
        """
        name_lower = column_name.lower().replace("_", " ").replace("-", " ")
        score = 0.0

        # Exact matches for strong indicators
        strong_indicators = [
            "agent",
            "medication",
            "drug",
            "drugdtxt",
            "conmed",
            "concomitant",
        ]
        if any(indicator in name_lower.split() for indicator in strong_indicators):
            score = 1.0
        # Check if 'drug' is part of a compound word (like chemo_drug)
        elif "drug" in name_lower:
            score = 1.0
        # Partial matches for medication keywords
        elif any(keyword in name_lower for keyword in self.MEDICATION_KEYWORDS):
            score = 0.7
        # Generic terms that might contain medications
        elif any(
            word in name_lower
            for word in ["name", "text", "desc", "treatment", "therapy"]
        ):
            score = 0.3

        return score

    def _score_patterns(self, values: List[str]) -> Tuple[float, int]:
        """
        Score based on medication naming patterns.

        Args:
            values: Unique values from column

        Returns:
            Tuple of (score, number of matches)
        """
        if len(values) == 0:
            return 0.0, 0

        pattern_matches = 0
        sample_size = min(len(values), 500)  # Sample first 500 for efficiency

        for value in values[:sample_size]:
            if self._looks_like_medication(value):
                pattern_matches += 1

        score = pattern_matches / sample_size
        return score, pattern_matches

    def _looks_like_medication(self, value: str) -> bool:
        """
        Check if a value looks like a medication name.

        Args:
            value: String value to check

        Returns:
            True if value appears to be a medication
        """
        if not value or len(value.strip()) < 3:
            return False

        value = value.strip()

        # Check against regex patterns
        for pattern in self.MEDICATION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        # Check for common medication suffixes
        value_lower = value.lower()
        if any(value_lower.endswith(suffix) for suffix in self.MEDICATION_SUFFIXES):
            return True

        # Check for combination drugs (contains '/')
        if "/" in value and not value.startswith("/") and not value.endswith("/"):
            parts = value.split("/")
            if all(len(p.strip()) >= 3 for p in parts):
                return True

        # Check for parenthetical brand/generic names
        if "(" in value and ")" in value:
            if any(suffix in value_lower for suffix in self.MEDICATION_SUFFIXES):
                return True

        return False

    def _score_statistics(self, series: pd.Series) -> Tuple[float, str]:
        """
        Score based on statistical properties.

        Args:
            series: Column data

        Returns:
            Tuple of (score, reasoning)
        """
        score = 0.0
        reasons = []

        # Diversity score (medications should have many unique values)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio > 0.5:
            score += 0.4
            reasons.append(f"High diversity ({unique_ratio:.1%} unique)")
        elif unique_ratio > 0.2:
            score += 0.2
            reasons.append(f"Moderate diversity ({unique_ratio:.1%} unique)")

        # Average length score (medications typically 5-30 characters)
        avg_length = series.str.len().mean()
        if 5 <= avg_length <= 30:
            score += 0.3
            reasons.append(f"Typical medication length (avg: {avg_length:.0f} chars)")
        elif 4 <= avg_length <= 40:
            score += 0.1

        # Alphanumeric content (medications are mostly letters)
        alphanumeric_ratio = series.str.contains(
            r"^[A-Za-z0-9\-\s/\(\)]+$", na=False
        ).mean()
        if alphanumeric_ratio > 0.8:
            score += 0.3
            reasons.append(f"Clean alphanumeric content ({alphanumeric_ratio:.1%})")
        elif alphanumeric_ratio > 0.5:
            score += 0.1

        return min(score, 1.0), "; ".join(reasons) if reasons else ""

    def _score_content(self, values: List[str]) -> Tuple[float, str]:
        """
        Score based on content analysis.

        Args:
            values: Unique values from column

        Returns:
            Tuple of (score, reasoning)
        """
        if len(values) == 0:
            return 0.0, ""

        score = 0.0
        reasons = []

        # Check for capitalization patterns (medications often start with capital)
        capitalized = sum(1 for v in values[:100] if v and v[0].isupper())
        cap_ratio = capitalized / min(len(values), 100)
        if cap_ratio > 0.7:
            score += 0.5
            reasons.append(f"Proper capitalization ({cap_ratio:.1%})")

        # Check for no obvious non-medication content
        non_med_indicators = [
            "yes",
            "no",
            "true",
            "false",
            "male",
            "female",
            "unknown",
            "n/a",
            "none",
            "null",
        ]
        non_med_count = sum(1 for v in values[:100] if v.lower() in non_med_indicators)
        if non_med_count == 0:
            score += 0.5
        elif non_med_count < 5:
            score += 0.2

        return score, "; ".join(reasons) if reasons else ""

    def _extract_sample_medications(self, values: List[str]) -> List[str]:
        """
        Extract sample medications from values.

        Args:
            values: Unique values from column

        Returns:
            List of sample medication names
        """
        samples = []
        for value in values[:100]:  # Check first 100
            if self._looks_like_medication(value):
                samples.append(value)
                if len(samples) >= 10:
                    break
        return samples

    def get_best_column(self) -> Optional[ColumnAnalysisResult]:
        """
        Get the column with highest confidence score.

        Returns:
            Best column analysis result or None
        """
        return self.analyzed_columns[0] if self.analyzed_columns else None

    def get_columns_above_threshold(
        self, threshold: float = 0.7
    ) -> List[ColumnAnalysisResult]:
        """
        Get columns with confidence above threshold.

        Args:
            threshold: Minimum confidence threshold

        Returns:
            List of high-confidence columns
        """
        return [col for col in self.analyzed_columns if col.confidence >= threshold]
