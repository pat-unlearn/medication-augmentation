# Output Module

## Overview

The output module provides essential export capabilities for the medication augmentation system, focusing on generating production-ready conmeds.yml files for any disease indication. The primary goal is to convert classification results into the YAML format required by clinical pipelines across therapeutic areas.

## Structure

```
output/
├── __init__.py
└── exporters.py        # Format-specific exporters focused on conmeds.yml
```

## Key Components

### ConmedsYAMLExporter (Primary Focus)

The core exporter for generating conmeds.yml files from medication classifications:

```python
from med_aug.output import ConmedsYAMLExporter

exporter = ConmedsYAMLExporter()

# Classifications from LLM or validation phases
classifications = {
    "pembrolizumab": ["taking_pembrolizumab"],
    "osimertinib": ["taking_osimertinib"],
    "paclitaxel": ["taking_paclitaxel"],
    # ... more medications
}

# Export to conmeds.yml format
output_file = exporter.export(
    classifications,
    output_path="./output/conmeds_augmented.yml"
)
```

#### Output Format

The exporter generates YAML files compatible with clinical pipelines for any disease:

```yaml
# conmeds_augmented.yml - NSCLC example
taking_pembrolizumab:
  - pembrolizumab
  - Keytruda
  - pembrolizumab injection

taking_osimertinib:
  - osimertinib
  - Tagrisso
  - AZD9291

# conmeds_augmented.yml - Breast cancer example
taking_trastuzumab:
  - trastuzumab
  - Herceptin
  - trastuzumab-dkst

taking_palbociclib:
  - palbociclib
  - Ibrance
  - PD-0332991
```

#### Drug Class Name Conversion

The exporter automatically converts drug class names to the conmeds format:

```python
# Input: drug class names (works for any disease)
nsclc_classes = ["PD-1 Inhibitors", "EGFR Inhibitors", "Taxane Chemotherapy"]
breast_classes = ["HER2-Targeted Therapy", "CDK4/6 Inhibitors", "Hormone Therapy"]

# Output: conmeds keys
nsclc_keys = ["taking_pd1_inhibitors", "taking_egfr_inhibitors", "taking_taxane_chemotherapy"]
breast_keys = ["taking_her2_targeted_therapy", "taking_cdk46_inhibitors", "taking_hormone_therapy"]
```

### Additional Exporters

#### JSONExporter

For structured data export and pipeline integration:

```python
from med_aug.output import JSONExporter

exporter = JSONExporter(pretty_print=True)

data = {
    "classifications": classifications,
    "metadata": {
        "generated": "2025-09-03T10:00:00Z",
        "total_medications": 150,
        "drug_classes": 54
    }
}

exporter.export(data, "classification_results.json")
```

#### CSVExporter

For tabular data export and analysis:

```python
from med_aug.output import CSVExporter
import pandas as pd

# Convert classifications to DataFrame
medications_df = pd.DataFrame([
    {"medication": "pembrolizumab", "drug_class": "taking_pembrolizumab", "confidence": 0.95},
    {"medication": "osimertinib", "drug_class": "taking_osimertinib", "confidence": 0.92},
    # ... more rows
])

exporter = CSVExporter()
exporter.export(medications_df, "medications.csv")
```

#### ExcelExporter

For comprehensive reporting with multiple sheets:

```python
from med_aug.output import ExcelExporter

data = {
    "Summary": summary_df,
    "Classifications": classifications_df,
    "Validation": validation_results_df
}

exporter = ExcelExporter()
exporter.export(data, "medication_analysis.xlsx")
```

#### HTMLExporter

For web-friendly reporting:

```python
from med_aug.output import HTMLExporter

# Convert data to HTML table
html_data = {
    "title": "NSCLC Medication Classifications",
    "tables": {
        "Classifications": classifications_df,
        "Summary": summary_stats
    }
}

exporter = HTMLExporter()
exporter.export(html_data, "report.html")
```

#### PDFExporter

For formal documentation:

```python
from med_aug.output import PDFExporter

exporter = PDFExporter(
    page_size="A4",
    include_header=True,
    title="NSCLC Medication Augmentation Report"
)

exporter.export(report_data, "report.pdf")
```

#### MarkdownExporter

For documentation and sharing:

```python
from med_aug.output import MarkdownExporter

exporter = MarkdownExporter(
    include_toc=True,
    github_flavored=True
)

markdown_content = exporter.export(documentation_data, "report.md")
```

## Pipeline Integration

The output module integrates with the pipeline's OutputGenerationPhase:

```python
# In OutputGenerationPhase
from ..output.exporters import ConmedsYAMLExporter

# Primary deliverable - conmeds.yml export
if context.get('llm_classifications') or context.get('classification_results'):
    conmeds_file = output_path / "conmeds_augmented.yml"
    exporter = ConmedsYAMLExporter()

    # Get classifications from LLM or validation phases
    classifications = context.get('llm_classifications', {})

    try:
        result_file = exporter.export(classifications, conmeds_file)
        logger.info("conmeds_yaml_generated", file=str(result_file))

        # Track this as the primary deliverable
        results['conmeds_yaml'] = {
            'file': str(result_file),
            'drug_classes': len(classifications),
            'total_medications': sum(len(meds) for meds in classifications.values())
        }
    except Exception as e:
        logger.error("conmeds_export_failed", error=str(e))
```

## Usage Examples

### Basic Conmeds Export

```python
from med_aug.output import ConmedsYAMLExporter

# Simple classification results
results = {
    "pembrolizumab": "taking_pembrolizumab",
    "keytruda": "taking_pembrolizumab",
    "osimertinib": "taking_osimertinib",
    "tagrisso": "taking_osimertinib"
}

# Group by drug class
grouped = {}
for medication, drug_class in results.items():
    if drug_class not in grouped:
        grouped[drug_class] = []
    grouped[drug_class].append(medication)

# Export to YAML
exporter = ConmedsYAMLExporter()
output_file = exporter.export(grouped, "conmeds_expanded.yml")
print(f"Generated: {output_file}")
```

### Pipeline Results Export

```python
from med_aug.output import JSONExporter, ConmedsYAMLExporter

# Complete pipeline results
pipeline_results = {
    'pipeline_id': 'nsclc_augmentation_001',
    'start_time': '2025-09-03T09:00:00Z',
    'end_time': '2025-09-03T10:30:00Z',
    'phases': {
        'data_analysis': {'status': 'completed', 'duration': 120},
        'medication_extraction': {'status': 'completed', 'duration': 300},
        'llm_classification': {'status': 'completed', 'duration': 1800},
        'output_generation': {'status': 'completed', 'duration': 60}
    },
    'results': {
        'total_medications': 1250,
        'unique_medications': 850,
        'classified_medications': 780,
        'drug_classes': 57
    },
    'classifications': {
        'taking_pembrolizumab': ['pembrolizumab', 'keytruda', 'pembrolizumab injection'],
        'taking_osimertinib': ['osimertinib', 'tagrisso', 'azd9291'],
        # ... more classifications
    }
}

# Export primary deliverable
conmeds_exporter = ConmedsYAMLExporter()
conmeds_file = conmeds_exporter.export(
    pipeline_results['classifications'],
    "conmeds_augmented.yml"
)

# Export detailed results
json_exporter = JSONExporter(pretty_print=True)
json_exporter.export(pipeline_results, "pipeline_results.json")
```

## CLI Integration

Export functionality is integrated into the pipeline CLI:

```bash
# Pipeline run automatically generates exports
python -m src.med_aug.cli.app pipeline run \
  --input data.csv \
  --disease nsclc \
  --output ./output

# Outputs generated:
# - conmeds_augmented.yml (primary)
# - classification_results.json
# - pipeline_summary.json
```

## Configuration

Export behavior can be configured through the pipeline:

```python
from med_aug.pipeline import PipelineConfig

config = PipelineConfig(
    input_file="data.csv",
    output_path="./output",
    disease_module="nsclc",  # or any disease module
    # Export settings handled automatically
    # Focus on conmeds.yml generation
)
```

## Success Metrics

The output module supports multi-disease success criteria:

- **Primary Deliverable**: conmeds_augmented.yml with expanded drug name coverage for any disease
- **Coverage Expansion**: Significant increase in drug classes with 20-50+ names each per disease
- **Quality Tracking**: Classification confidence and validation results across therapeutic areas
- **Format Compatibility**: Direct integration with existing clinical pipelines for any disease
- **Scalability**: Consistent format enables rapid deployment to new therapeutic areas

## File Structure

Generated output directory structure:

```
output/
├── conmeds_augmented.yml      # PRIMARY: Enhanced conmeds file
├── classification_results.json # Detailed classifications with confidence
├── pipeline_summary.json      # Execution summary and metrics
├── medications.csv            # Tabular medication data
└── report.html               # Human-readable summary
```

## Error Handling

Robust error handling for export operations:

```python
from med_aug.output import ConmedsYAMLExporter
from med_aug.output.exceptions import ExportError

try:
    exporter = ConmedsYAMLExporter()
    result = exporter.export(classifications, output_path)
except ExportError as e:
    logger.error(f"Export failed: {e}")
    # Fallback to JSON format
    json_exporter = JSONExporter()
    json_exporter.export(classifications, "classifications_backup.json")
```

## Testing

Test coverage focuses on core export functionality:

```python
import pytest
from med_aug.output import ConmedsYAMLExporter

def test_conmeds_yaml_export():
    """Test conmeds YAML export functionality."""
    classifications = {
        "taking_pembrolizumab": ["pembrolizumab", "keytruda"],
        "taking_osimertinib": ["osimertinib", "tagrisso"]
    }

    exporter = ConmedsYAMLExporter()
    output_file = exporter.export(classifications, "test_output.yml")

    assert output_file.exists()

    # Verify YAML structure
    import yaml
    with open(output_file) as f:
        data = yaml.safe_load(f)

    assert "taking_pembrolizumab" in data
    assert "pembrolizumab" in data["taking_pembrolizumab"]
    assert "keytruda" in data["taking_pembrolizumab"]
```

## Best Practices

1. **Focus on conmeds.yml** - This is the primary deliverable for the PRD
2. **Preserve drug class naming** - Use proper "taking_" prefix format
3. **Handle duplicates** - Remove duplicate medication names within classes
4. **Sort alphabetically** - Consistent ordering for version control
5. **Validate output** - Ensure YAML is valid and parseable
6. **Include metadata** - Track generation timestamp and source information
7. **Handle errors gracefully** - Provide fallback export options

## Integration Points

The output module connects with:

- **Pipeline Phases**: Receives classification results from LLM and validation phases
- **Disease Modules**: Uses disease-specific drug class definitions for proper naming
- **CLI Commands**: Automatically generates exports during pipeline runs
- **External Systems**: Produces files compatible with existing clinical pipelines for any disease

## Future Enhancements

- **Cross-Disease Validation**: Automatic conmeds.yml validation across therapeutic areas
- **Disease-Specific Diff Reports**: Compare before/after medication coverage per disease
- **Multi-Disease Merge**: Combine conmeds.yml files from multiple disease modules
- **Format Validation**: Ensure output meets pipeline requirements for any disease
- **Cross-Therapeutic Insights**: Identify medications used across multiple diseases
- **Version Control**: Track changes and augmentation history per disease module
