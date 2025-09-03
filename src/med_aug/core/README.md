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

## Integration Points

The core module integrates with:
- **Pipeline**: Provides analysis and extraction phases
- **LLM**: Supplies candidates for classification
- **Output**: Exports classifications to conmeds.yml
- **Web**: Provides medications for research

## Future Enhancements Ideas

- Machine learning-based column detection
- Multi-language medication support
- Fuzzy matching for misspellings
- Real-time streaming analysis
- Custom medication dictionaries
- Advanced NLP techniques
