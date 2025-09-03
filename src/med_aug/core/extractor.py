"""Medication extraction and normalization module."""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
from pathlib import Path
import pandas as pd
import polars as pl
from dataclasses import dataclass, field
from datetime import datetime
from .models import Medication, MedicationType
from .logging import get_logger, PerformanceLogger

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)


@dataclass
class ExtractionResult:
    """Result from medication extraction."""

    total_rows: int
    unique_medications: int
    normalized_medications: List[str]
    frequency_map: Dict[str, int]
    variants_map: Dict[str, List[str]]  # Normalized -> original variants
    extraction_time: float
    column_name: str

    def get_top_medications(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get top N medications by frequency."""
        return Counter(self.frequency_map).most_common(n)

    def get_variants_for(self, normalized_name: str) -> List[str]:
        """Get all variants for a normalized medication name."""
        return self.variants_map.get(normalized_name, [])


class MedicationExtractor:
    """Extract and normalize medication names from datasets."""

    # Common prefixes/suffixes to remove for normalization
    REMOVE_PATTERNS = [
        r"\s+\d+\s*mg(?:/ml)?",  # Dosage: 100mg, 50mg/ml
        r"\s+\d+\s*mcg",  # Microgram dosages
        r"\s+\d+\s*mg/kg",  # Per kg dosing
        r"\s+\d+\s*mg/m2",  # Per m2 dosing
        r"\s+\d+\s*ml",  # Volume
        r"\s+\d+\s*%",  # Percentage
        r"/m2",  # Surface area dosing
        r"/kg",  # Weight-based dosing
        r"\s+tablets?",  # Tablet form
        r"\s+capsules?",  # Capsule form
        r"\s+injection",  # Injection form
        r"\s+infusion",  # Infusion form
        r"\s+solution",  # Solution form
        r"\s+cream",  # Topical form
        r"\s+gel",  # Gel form
        r"\s+oral",  # Route
        r"\s+iv\b",  # IV route
        r"\s+extended[\s-]release",  # ER formulation
        r"\s+er\b",  # ER abbreviation
        r"\s+xr\b",  # XR abbreviation
        r"\s+sr\b",  # SR abbreviation
        r"\s+la\b",  # LA abbreviation
        r"\s+hcl\b",  # HCl salt
        r"\s+sulfate",  # Sulfate salt
        r"\s+sodium",  # Sodium salt
        r"\s+calcium",  # Calcium salt
        r"\s+potassium",  # Potassium salt
        r"^\d+\s*[-\.]\s*",  # Leading numbers
        r"\s*[-\.]\s*\d+$",  # Trailing numbers
        r"\s+\(.*?\)",  # Parenthetical content
        r"\s+\[.*?\]",  # Bracketed content
    ]

    # Common separators for combination drugs
    COMBINATION_SEPARATORS = ["/", "+", "and", "with", "-"]

    # Known brand-generic mappings (expandable)
    BRAND_GENERIC_MAP = {
        "keytruda": "pembrolizumab",
        "opdivo": "nivolumab",
        "tecentriq": "atezolizumab",
        "imfinzi": "durvalumab",
        "tagrisso": "osimertinib",
        "tarceva": "erlotinib",
        "xalkori": "crizotinib",
        "alecensa": "alectinib",
        "lorbrena": "lorlatinib",
        "avastin": "bevacizumab",
        "cyramza": "ramucirumab",
        "abraxane": "nab-paclitaxel",
        "taxol": "paclitaxel",
        "alimta": "pemetrexed",
        "taxotere": "docetaxel",
        "gemzar": "gemcitabine",
        "paraplatin": "carboplatin",
        "platinol": "cisplatin",
        "gilotrif": "afatinib",
        "iressa": "gefitinib",
        "vizimpro": "dacomitinib",
        "alunbrig": "brigatinib",
        "zykadia": "ceritinib",
        "rozlytrek": "entrectinib",
        "vitrakvi": "larotrectinib",
        "lumakras": "sotorasib",
        "krazati": "adagrasib",
        "rybrevant": "amivantamab",
        "exkivity": "mobocertinib",
        "tabrecta": "capmatinib",
        "tepmetko": "tepotinib",
        "retevmo": "selpercatinib",
        "gavreto": "pralsetinib",
        "enhertu": "trastuzumab deruxtecan",
        "trodelvy": "sacituzumab govitecan",
    }

    def __init__(self):
        """Initialize the extractor."""
        self.extracted_medications = []
        self.normalization_cache = {}

    def extract_from_file(
        self, file_path: str, column_name: str, sample_size: Optional[int] = None
    ) -> ExtractionResult:
        """
        Extract medications from a specific column in a file.

        Args:
            file_path: Path to data file
            column_name: Name of column containing medications
            sample_size: Optional limit on rows to process

        Returns:
            Extraction result with normalized medications
        """
        import time

        start_time = time.time()

        file_path = Path(file_path)

        # Load data based on file type
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, nrows=sample_size)
        elif file_path.suffix.lower() in [".txt", ".tsv"]:
            df = pd.read_csv(file_path, sep="\t", nrows=sample_size)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, nrows=sample_size)
        elif file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
            if sample_size:
                df = df.head(sample_size)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in file")

        result = self.extract_from_series(df[column_name], column_name)
        result.extraction_time = time.time() - start_time
        return result

    def extract_from_series(
        self, series: pd.Series, column_name: str = "medications"
    ) -> ExtractionResult:
        """
        Extract medications from a pandas Series.

        Args:
            series: Series containing medication names
            column_name: Name of the column

        Returns:
            Extraction result with normalized medications
        """
        import time

        start_time = time.time()

        # Remove nulls and convert to strings
        non_null = series.dropna()
        str_values = non_null.astype(str).str.strip()

        # Track original and normalized forms
        normalized_medications = []
        frequency_map = {}
        variants_map = {}

        for original_value in str_values:
            if not original_value or original_value.lower() in [
                "nan",
                "none",
                "n/a",
                "null",
                "unknown",
                "unk",
            ]:
                continue

            # Extract individual medications from the value (handle combinations)
            medications = self._extract_medications_from_text(original_value)

            for med in medications:
                # Normalize the medication name
                normalized = self._normalize_medication(med)

                if normalized:
                    normalized_medications.append(normalized)

                    # Track frequency
                    frequency_map[normalized] = frequency_map.get(normalized, 0) + 1

                    # Track variants
                    if normalized not in variants_map:
                        variants_map[normalized] = set()
                    variants_map[normalized].add(med)

        # Convert variant sets to lists
        for key in variants_map:
            variants_map[key] = list(variants_map[key])

        return ExtractionResult(
            total_rows=len(series),
            unique_medications=len(set(normalized_medications)),
            normalized_medications=normalized_medications,
            frequency_map=frequency_map,
            variants_map=variants_map,
            extraction_time=time.time() - start_time,
            column_name=column_name,
        )

    def _extract_medications_from_text(self, text: str) -> List[str]:
        """
        Extract individual medication names from text.

        Handles combination drugs and multiple medications in one field.

        Args:
            text: Raw text possibly containing medications

        Returns:
            List of extracted medication names
        """
        medications = []

        # First, check if it's a combination drug
        is_combination = False
        for sep in self.COMBINATION_SEPARATORS:
            if sep in text:
                # Special handling for dates (e.g., "2024-01-01")
                if sep == "-" and re.match(r"\d{4}-\d{2}-\d{2}", text):
                    continue

                # Split by separator
                parts = text.split(sep)
                if len(parts) == 2 and all(len(p.strip()) > 2 for p in parts):
                    is_combination = True
                    for part in parts:
                        cleaned = part.strip()
                        if cleaned and self._is_likely_medication(cleaned):
                            medications.append(cleaned)
                    break

        # If not a combination or no medications found, treat as single medication
        if not medications:
            medications.append(text)

        return medications

    def _normalize_medication(self, medication: str) -> Optional[str]:
        """
        Normalize a medication name.

        Args:
            medication: Raw medication name

        Returns:
            Normalized medication name or None if invalid
        """
        if not medication or len(medication) < 2:
            return None

        # Check cache first
        if medication in self.normalization_cache:
            return self.normalization_cache[medication]

        normalized = medication.strip()

        # Convert to lowercase for processing
        normalized_lower = normalized.lower()

        # Remove dosage, formulation, and other modifiers
        for pattern in self.REMOVE_PATTERNS:
            normalized_lower = re.sub(
                pattern, "", normalized_lower, flags=re.IGNORECASE
            )

        # Clean up extra whitespace
        normalized_lower = " ".join(normalized_lower.split())

        # Map brand names to generic names if known
        if normalized_lower in self.BRAND_GENERIC_MAP:
            normalized_lower = self.BRAND_GENERIC_MAP[normalized_lower]

        # Capitalize properly (first letter of each word for multi-word drugs)
        if normalized_lower:
            parts = normalized_lower.split()
            # Keep certain acronyms uppercase
            acronyms = ["ii", "iii", "iv", "er", "xr", "sr", "la", "hcl"]
            normalized = " ".join(
                part.upper() if part in acronyms else part.capitalize()
                for part in parts
            )
        else:
            normalized = None

        # Cache the result
        self.normalization_cache[medication] = normalized

        return normalized

    def _is_likely_medication(self, text: str) -> bool:
        """
        Check if text is likely a medication name.

        Args:
            text: Text to check

        Returns:
            True if likely a medication
        """
        if not text or len(text) < 3:
            return False

        # Check for common non-medication values
        non_meds = [
            "yes",
            "no",
            "true",
            "false",
            "unknown",
            "n/a",
            "none",
            "na",
            "null",
            "not applicable",
            "test",
            "sample",
            "unk",
        ]
        if text.lower().strip() in non_meds:
            return False

        # Check for medication-like patterns
        med_patterns = [
            r"[a-z]+mab$",  # Monoclonal antibodies
            r"[a-z]+nib$",  # Kinase inhibitors
            r"[a-z]+tin$",  # Various drug classes
            r"[a-z]+pril$",  # ACE inhibitors
            r"[a-z]+sartan$",  # ARBs
            r"[a-z]+olol$",  # Beta blockers
            r"[a-z]+zole$",  # PPIs and antifungals
            r"[a-z]+cillin$",  # Penicillins
            r"[a-z]+mycin$",  # Antibiotics
            r"[a-z]+platin$",  # Platinum compounds
        ]

        text_lower = text.lower()
        for pattern in med_patterns:
            if re.search(pattern, text_lower):
                return True

        # Check if it's in our known brand/generic names
        if (
            text_lower in self.BRAND_GENERIC_MAP
            or text_lower in self.BRAND_GENERIC_MAP.values()
        ):
            return True

        # Default to True if it passes basic checks (length, no obvious non-med indicators)
        return True

    def deduplicate_medications(self, medications: List[str]) -> List[str]:
        """
        Remove duplicate medications considering variations.

        Args:
            medications: List of medication names

        Returns:
            Deduplicated list of medications
        """
        seen_normalized = set()
        unique_medications = []

        for med in medications:
            normalized = self._normalize_medication(med)
            if normalized and normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_medications.append(med)

        return unique_medications

    def get_medication_statistics(self, result: ExtractionResult) -> Dict[str, Any]:
        """
        Get statistical analysis of extracted medications.

        Args:
            result: Extraction result

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_rows": result.total_rows,
            "unique_medications": result.unique_medications,
            "total_occurrences": sum(result.frequency_map.values()),
            "coverage_rate": (
                result.unique_medications / result.total_rows
                if result.total_rows > 0
                else 0
            ),
        }

        # Most common medications
        top_meds = result.get_top_medications(10)
        stats["top_10_medications"] = top_meds

        # Frequency distribution
        frequencies = list(result.frequency_map.values())
        if frequencies:
            stats["frequency_stats"] = {
                "min": min(frequencies),
                "max": max(frequencies),
                "mean": sum(frequencies) / len(frequencies),
                "median": sorted(frequencies)[len(frequencies) // 2],
            }

        # Variant statistics
        variant_counts = [len(variants) for variants in result.variants_map.values()]
        if variant_counts:
            stats["variant_stats"] = {
                "medications_with_variants": sum(1 for v in variant_counts if v > 1),
                "max_variants": max(variant_counts),
                "avg_variants": sum(variant_counts) / len(variant_counts),
            }

        return stats

    def export_medications(
        self, result: ExtractionResult, output_path: str, format: str = "csv"
    ) -> None:
        """
        Export extracted medications to file.

        Args:
            result: Extraction result
            output_path: Path for output file
            format: Output format (csv, json, xlsx)
        """
        # Create DataFrame with medication data
        data = []
        for normalized_name, frequency in result.frequency_map.items():
            variants = result.variants_map.get(normalized_name, [])
            data.append(
                {
                    "normalized_name": normalized_name,
                    "frequency": frequency,
                    "variants": "|".join(variants),
                    "variant_count": len(variants),
                }
            )

        df = pd.DataFrame(data)
        df = df.sort_values("frequency", ascending=False)

        # Export based on format
        output_path = Path(output_path)
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif format == "xlsx":
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def create_medication_objects(
        self, result: ExtractionResult, source: str = "data_extraction"
    ) -> List[Medication]:
        """
        Convert extraction results to Medication objects.

        Args:
            result: Extraction result
            source: Source identifier

        Returns:
            List of Medication objects
        """
        medications = []

        for normalized_name, frequency in result.frequency_map.items():
            variants = result.variants_map.get(normalized_name, [])

            # Determine medication type based on name patterns
            med_type = self._determine_medication_type(normalized_name, variants)

            # Create medication object
            med = Medication(
                name=normalized_name,
                type=med_type,
                confidence=min(
                    0.9, 0.5 + (frequency / result.total_rows) * 2
                ),  # Confidence based on frequency
                source=source,
                metadata={
                    "frequency": frequency,
                    "variants": variants,
                    "column": result.column_name,
                },
                discovered_at=datetime.now(),
            )
            medications.append(med)

        return medications

    def _determine_medication_type(
        self, normalized_name: str, variants: List[str]
    ) -> MedicationType:
        """
        Determine the type of medication.

        Args:
            normalized_name: Normalized medication name
            variants: List of variants

        Returns:
            Medication type
        """
        name_lower = normalized_name.lower()

        # Check if it's a known brand name
        if name_lower in self.BRAND_GENERIC_MAP:
            return MedicationType.BRAND

        # Check if it's a combination
        if any(sep in normalized_name for sep in ["/", "+"]):
            return MedicationType.COMBINATION

        # Check for clinical trial patterns
        if re.match(r"^[A-Z]{2,}-\d+", normalized_name):
            return MedicationType.CLINICAL_TRIAL

        # Check for abbreviations
        if len(normalized_name) <= 5 and normalized_name.isupper():
            return MedicationType.ABBREVIATION

        # Default to generic
        return MedicationType.GENERIC
