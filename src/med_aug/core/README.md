# Core Module

## Overview

The core module contains the fundamental components for medication data analysis and extraction. It provides the essential algorithms for identifying, extracting, and normalizing medication information from various data sources.

## Structure

```
core/
├── __init__.py
├── analyzer.py       # Column detection and analysis
├── extractor.py      # Medication extraction and normalization
├── models.py         # Core data models
├── logging.py        # Logging configuration and utilities
└── exceptions.py     # Custom exceptions
```

## Key Components

### Data Analyzer (`analyzer.py`)

Sophisticated column detection for identifying medication data:

```python
from med_aug.core import DataAnalyzer

analyzer = DataAnalyzer()
results = analyzer.analyze_dataframe(df, confidence_threshold=0.5)

for result in results:
    print(f"Column: {result.column}")
    print(f"Confidence: {result.confidence}")
    print(f"Sample medications: {result.sample_medications}")
```

**Features:**
- Pattern-based medication detection
- Statistical scoring algorithms
- Content type analysis
- Confidence scoring
- Multi-column support

### Medication Extractor (`extractor.py`)

Advanced medication extraction and normalization:

```python
from med_aug.core import MedicationExtractor

extractor = MedicationExtractor()
result = extractor.extract_from_series(df['medications'], 'medications')

print(f"Unique medications: {result.unique_medications}")
print(f"Normalized medications: {result.normalized_medications}")
print(f"Variants: {result.variants_map}")
```

**Features:**
- Text normalization (lowercase, whitespace)
- Dosage and form removal
- Brand-generic mapping
- Combination drug parsing
- Variant tracking
- Frequency analysis

### Data Models (`models.py`)

Core data structures used throughout the system:

#### ColumnAnalysisResult
```python
@dataclass
class ColumnAnalysisResult:
    column: str
    confidence: float
    medication_count: int
    unique_count: int
    sample_medications: List[str]
    metadata: Dict[str, Any]
```

#### ExtractionResult
```python
@dataclass
class ExtractionResult:
    column_name: str
    total_rows: int
    unique_medications: int
    normalized_medications: List[str]
    frequency_map: Dict[str, int]
    variants_map: Dict[str, List[str]]
    metadata: Dict[str, Any]
```

### Logging System (`logging.py`)

Comprehensive structured logging with multiple features:

```python
from med_aug.core.logging import setup_logging, get_logger

# Setup logging
setup_logging(
    level="INFO",
    log_file="medaug.log",
    json_logs=True,
    include_timestamp=True
)

# Use logger
logger = get_logger(__name__)
logger.info("processing_started", file="data.csv", rows=1000)
```

**Components:**
- **StructuredLogger**: Key-value pair logging
- **PerformanceLogger**: Operation timing and metrics
- **AuditLogger**: Compliance and audit trails
- **ErrorLogger**: Detailed error tracking
- **SecurityLogger**: Security event logging

## Core Algorithms

### Column Detection Algorithm

1. **Pattern Matching**
   - Regex patterns for medication names
   - Common medication suffixes (-mab, -ib, -in, etc.)
   - Dosage pattern detection

2. **Statistical Analysis**
   - Unique value ratio
   - Character length distribution
   - Special character frequency

3. **Content Analysis**
   - Medical terminology detection
   - Brand name recognition
   - Combination drug patterns

4. **Scoring**
   - Weighted scoring system
   - Confidence calculation
   - Threshold-based selection

### Normalization Pipeline

1. **Text Cleaning**
   ```python
   text = text.lower().strip()
   text = re.sub(r'\s+', ' ', text)
   ```

2. **Dosage Removal**
   ```python
   # Remove patterns like "100mg", "5ml", etc.
   text = re.sub(r'\d+\s*(?:mg|ml|mcg|iu|units?)', '', text)
   ```

3. **Form Removal**
   ```python
   # Remove "tablet", "capsule", "injection", etc.
   forms = ['tablet', 'capsule', 'injection', 'solution']
   for form in forms:
       text = text.replace(form, '')
   ```

4. **Variant Mapping**
   ```python
   # Track different representations
   variants['pembrolizumab'] = [
       'Pembrolizumab',
       'PEMBROLIZUMAB',
       'pembrolizumab 200mg',
       'Keytruda'
   ]
   ```

## Usage Patterns

### Basic Analysis
```python
from med_aug.core import DataAnalyzer
import pandas as pd

# Load data
df = pd.read_csv('clinical_data.csv')

# Analyze columns
analyzer = DataAnalyzer()
results = analyzer.analyze_dataframe(df)

# Get best medication column
best_column = results[0].column if results else None
```

### Extraction with Options
```python
from med_aug.core import MedicationExtractor

extractor = MedicationExtractor(
    normalize_case=True,
    remove_dosages=True,
    remove_forms=True,
    track_variants=True
)

result = extractor.extract_from_text(
    "Patient on Pembrolizumab 200mg IV and carboplatin"
)
```

### Custom Pattern Matching
```python
analyzer = DataAnalyzer()

# Add custom patterns
analyzer.add_pattern(r'custom-drug-\d+')
analyzer.add_keyword('experimental_med')

# Analyze with custom patterns
results = analyzer.analyze_dataframe(df)
```

## Performance Optimization

### Caching
- Column analysis results cached
- Normalization mappings cached
- Pattern compilation cached

### Parallel Processing
```python
# Parallel column analysis
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(
        analyzer.analyze_column,
        df.columns
    )
```

### Memory Management
- Streaming processing for large files
- Chunked reading for DataFrames
- Efficient string operations

## Error Handling

```python
from med_aug.core.exceptions import (
    AnalysisError,
    ExtractionError,
    ValidationError
)

try:
    results = analyzer.analyze_dataframe(df)
except AnalysisError as e:
    logger.error("analysis_failed", error=str(e))
except ValidationError as e:
    logger.error("invalid_input", error=str(e))
```

## Testing

Comprehensive test coverage for:
- Column detection accuracy
- Normalization correctness
- Edge cases (empty data, special characters)
- Performance benchmarks
- Memory usage

## Best Practices

1. **Set appropriate thresholds** based on data quality
2. **Review sample medications** before processing
3. **Monitor confidence scores** for quality assurance
4. **Use logging** for debugging and audit trails
5. **Handle errors gracefully** with fallbacks
6. **Cache results** for repeated operations
7. **Validate inputs** before processing

## 🧹 Dirty Data Handling & Normalization

### **Problem Statement**
Clinical datasets contain messy, inconsistent medication data that requires sophisticated cleaning:

- **Typos & Misspellings**: "pemebroxed" → "pemetrexed", "keytruda" → "pembrolizumab"
- **Inconsistent Casing**: "PEMBROLIZUMAB" → "pembrolizumab" 
- **Dosage Contamination**: "pembrolizumab 200mg IV q3w" → "pembrolizumab"
- **Formulation Noise**: "pemetrexed injection solution" → "pemetrexed"
- **Abbreviations**: "pembro", "carbo/taxol" → standardized names
- **Research Codes**: "AZD9291" → "osimertinib"
- **Combination Drugs**: "carboplatin + pemetrexed" → separate medications
- **Non-Standard Characters**: Unicode, special symbols, extra whitespace

### **Current Data Cleaning Capabilities**

#### **1. Text Normalization (`extractor.py`)**
```python
def _normalize_medication(self, medication: str) -> Optional[str]:
    """
    Comprehensive normalization pipeline handles:
    - Case standardization
    - Whitespace cleanup  
    - Dosage removal
    - Formulation stripping
    - Brand-generic mapping
    """
    
    # Example transformations:
    # "PEMBROLIZUMAB 200MG IV" → "Pembrolizumab"
    # "pemetrexed disodium injection" → "Pemetrexed"
    # "keytruda" → "Pembrolizumab" (via brand mapping)
```

#### **2. Dosage & Form Removal Patterns**
```python
REMOVE_PATTERNS = [
    r"\s+\d+\s*mg(?:/ml)?",      # 100mg, 50mg/ml  
    r"\s+\d+\s*mcg",             # Microgram dosages
    r"\s+\d+\s*mg/kg",           # Per kg dosing
    r"\s+\d+\s*mg/m2",           # Per m2 dosing
    r"\s+tablets?",              # Tablet form
    r"\s+capsules?",             # Capsule form
    r"\s+injection",             # Injection form
    r"\s+solution",              # Solution form
    r"\s+extended[\s-]release",  # ER formulation
    r"\s+hcl\b",                # HCl salt
    r"\s+sulfate",              # Sulfate salt
    r"\s*\(.*?\)",              # Parenthetical content
]
```

#### **3. Brand-Generic Mapping (117+ entries)**
```python
BRAND_GENERIC_MAP = {
    "keytruda": "pembrolizumab",
    "opdivo": "nivolumab", 
    "tecentriq": "atezolizumab",
    "tagrisso": "osimertinib",
    "abraxane": "nab-paclitaxel",
    "retevmo": "selpercatinib",    # Recent FDA approvals
    "gavreto": "pralsetinib",
    "lumakras": "sotorasib",
    "exkivity": "mobocertinib",
    # ... 100+ more mappings
}
```

#### **4. Combination Drug Parsing**
```python
def _extract_medications_from_text(self, text: str) -> List[str]:
    """
    Handles combination therapies:
    - "carboplatin + pemetrexed" → ["carboplatin", "pemetrexed"]
    - "carbo/taxol" → ["carboplatin", "paclitaxel"]
    - Separators: ["/", "+", "and", "with", "-"]
    """
```

#### **5. Medication Likelihood Detection**
```python
def _is_likely_medication(self, text: str) -> bool:
    """
    Pattern-based validation to filter non-medications:
    
    Medication Patterns:
    - [drug]mab$ (monoclonal antibodies)
    - [drug]nib$ (kinase inhibitors)  
    - [drug]platin$ (platinum compounds)
    - Known brand/generic names
    
    Non-Medication Filters:
    - "yes", "no", "unknown", "n/a"
    - Obvious non-drug values
    """
```

### **Advanced Dirty Data Scenarios**

#### **Real-World Examples Handled**
```python
# Input → Output examples from our normalization pipeline

# Dosage contamination
"pembrolizumab 200mg IV q3w" → "Pembrolizumab"
"pemetrexed 500mg/m2 injection" → "Pemetrexed"

# Case inconsistencies  
"CARBOPLATIN" → "Carboplatin"
"osimertinib hcl" → "Osimertinib"

# Brand name conversion
"keytruda 200mg" → "Pembrolizumab"  
"abraxane weekly" → "Nab-Paclitaxel"

# Combination parsing
"carbo + taxol" → ["Carboplatin", "Paclitaxel"]
"bevacizumab/carboplatin/paclitaxel" → ["Bevacizumab", "Carboplatin", "Paclitaxel"]

# Research code mapping (via web research + LLM)
"AZD9291" → "Osimertinib" (discovered through FDA lookups)
"CO-1686" → "Rociletinib" (clinical trial compound)

# Salt/formulation removal
"pemetrexed disodium" → "Pemetrexed"
"carboplatin injection solution" → "Carboplatin"
```

#### **Variant Tracking**
```python
# System preserves all original forms for audit/validation
variants_map = {
    "pembrolizumab": [
        "PEMBROLIZUMAB",
        "pembrolizumab 200mg", 
        "Pembrolizumab IV",
        "keytruda",
        "Keytruda 200mg IV q3w"
    ],
    "pemetrexed": [
        "pemetrexed",
        "pemetrexed disodium",
        "Pemetrexed 500mg/m2",
        "alimta",
        "Alimta injection"
    ]
}
```

### **LLM-Assisted Typo Correction**

#### **Current Approach: LLM Enhancement**
Our system uses LLM prompts to handle complex cases that rule-based normalization can't catch:

```python
# From llm/prompts.py - extraction_enhancement_prompt
system_prompt = """
You are an expert in medical text processing and medication identification.
Your task is to identify medications from various text formats including
abbreviations, misspellings, and non-standard notations.
"""

user_template = """
Extract and standardize medication names from: $text

Consider:
- Common abbreviations (e.g., 'pembro' for pembrolizumab)
- Misspellings and typos
- Combination therapies (e.g., 'carbo/taxol') 
- Dosage information that may be attached
- Both generic and brand names
"""
```

#### **LLM Handles Complex Cases**
```json
{
  "original_text": "pembro + carbo/taxol q3w",
  "extracted_medications": [
    {
      "original": "pembro",
      "standardized": "pembrolizumab",
      "type": "abbreviation",
      "confidence": 0.95
    },
    {
      "original": "carbo", 
      "standardized": "carboplatin",
      "type": "abbreviation",
      "confidence": 0.98
    },
    {
      "original": "taxol",
      "standardized": "paclitaxel", 
      "type": "brand_name",
      "confidence": 0.97
    }
  ]
}
```

### **Data Quality Assessment**

#### **Quality Metrics Tracked**
```python
def get_medication_statistics(self, result: ExtractionResult) -> Dict[str, Any]:
    """
    Comprehensive quality assessment:
    - Coverage rate (% rows with medications found)
    - Variant diversity (multiple forms per medication)  
    - Frequency distribution (catch rare/suspicious entries)
    - Normalization success rate
    """
    return {
        "coverage_rate": result.unique_medications / result.total_rows,
        "variant_stats": {
            "medications_with_variants": count_with_multiple_variants,
            "max_variants": max_variants_per_med,
            "avg_variants": average_variants
        }
    }
```

#### **Quality Gates**
- **Low Coverage (<50%)**: Indicates poor column detection or very dirty data
- **High Variants (>10 per med)**: Suggests excessive noise or abbreviation issues
- **Suspicious Patterns**: Very short names, all caps, special characters

### **Current Limitations & Gaps**

#### **❌ Not Currently Implemented**
1. **Fuzzy String Matching**: No Levenshtein distance or similarity scoring
2. **Phonetic Matching**: No Soundex/Metaphone for sound-alike drugs  
3. **Edit Distance Correction**: No automated typo correction algorithms
4. **Machine Learning Normalization**: No trained models for drug name standardization
5. **Context-Aware Correction**: No disambiguation using surrounding clinical context

#### **⚠️ Relies on Fallback Strategies**
- **LLM Processing**: Complex typos handled by Claude's medical knowledge
- **Manual Review**: Evaluation framework flags suspicious normalizations
- **Web Research**: FDA/clinical databases help resolve unknown compounds
- **Expert Validation**: Clinical review for ambiguous cases

### **Best Practices for Dirty Data**

#### **1. Multi-Stage Cleaning Pipeline**
```python
# Recommended processing order:
1. Basic normalization (case, whitespace)
2. Pattern-based cleaning (dosages, forms)  
3. Brand-generic mapping
4. Combination drug parsing
5. LLM enhancement for complex cases
6. Manual review of low-confidence results
```

#### **2. Validation & Quality Checks**
```python
# Before processing large datasets:
1. Sample analysis to assess data quality
2. Column confidence scoring
3. Manual review of top variants
4. Set appropriate confidence thresholds
5. Track normalization success rates
```

#### **3. Audit Trail Maintenance**
```python
# Always preserve original data:
- Original text values stored in variants_map
- Normalization decisions logged
- Low-confidence cases flagged for review
- Expert validations tracked
```

### **Future Enhancement Roadmap**

#### **Phase 1: Fuzzy Matching (Planned)**
```python
# Add fuzzy string matching capabilities
from fuzzywuzzy import fuzz, process

def find_closest_medication(query: str, 
                          known_meds: List[str], 
                          threshold: int = 85) -> Optional[str]:
    """Find closest medication using edit distance."""
    match, score = process.extractOne(query, known_meds)
    return match if score >= threshold else None

# Usage examples:
# "pemebroxed" → "pemetrexed" (score: 89)
# "keytruda" → "pembrolizumab" (via existing mapping)
# "carboplton" → "carboplatin" (score: 91)
```

#### **Phase 2: ML-Based Normalization**
- Train medication name embeddings model
- Context-aware medication disambiguation  
- Automated abbreviation detection
- Clinical note parsing enhancement

#### **Phase 3: Advanced NLP**
- Named entity recognition for medications
- Relationship extraction (drug-dose-route)
- Multi-language medication support
- Real-time streaming normalization

### **Integration with Pipeline**

#### **Quality Assurance Flow**
```python
# During extraction phase:
1. Raw extraction with variants tracking
2. Normalization confidence scoring
3. Quality metrics calculation
4. LLM enhancement for low-confidence cases
5. Expert review queue for ambiguous normalizations
6. Final validation before classification
```

#### **Error Recovery Strategies**
```python
# When normalization fails:
1. Fall back to LLM processing
2. Flag for manual expert review  
3. Web research for unknown compounds
4. Context-based disambiguation
5. Conservative approach: preserve original if uncertain
```

This comprehensive dirty data handling ensures that even the messiest clinical datasets can be successfully processed while maintaining high accuracy and providing full audit trails for validation.

## Integration Points

The core module integrates with:
- **Pipeline**: Provides analysis and extraction phases
- **LLM**: Supplies candidates for classification
- **Output**: Exports classifications to conmeds.yml
- **Web**: Provides medications for research

## Future Enhancements Ideas

- Machine learning-based column detection
- Multi-language medication support  
- **Fuzzy matching for misspellings** (high priority)
- Real-time streaming analysis
- Custom medication dictionaries
- Advanced NLP techniques
