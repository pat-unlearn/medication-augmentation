# Medication Augmentation System

**Version:** 1.0
**Focus:** Multi-Disease Clinical Pipeline Framework

## Overview

The Medication Augmentation System is a flexible, modular framework that automatically expands `conmeds_defaults.yml` configuration files for any disease indication. It captures comprehensive generic and brand names for each drug class, improving medication matching accuracy in clinical data processing pipelines across multiple therapeutic areas.

**Currently Supported Diseases:**
- **NSCLC** (Non-Small Cell Lung Cancer) - Primary implementation
- **Extensible Architecture** for additional diseases (breast cancer, prostate cancer, etc.)

## Problem Statement

Clinical research across therapeutic areas faces common challenges with medication enrichment processes that rely on manually curated `conmeds_defaults.yml` files with limited medication name coverage, resulting in:

- **Low medication matching rates** - Missing many medication variations found in clinical datasets
- **Manual bottlenecks** - Data scientists spend significant time manually reviewing medication lists
- **High false negative rates** - LLM classification without disease-specific context misses therapeutic area medications
- **Scalability issues** - Each new disease requires separate manual curation efforts

## Solution

A modular, disease-agnostic automated pipeline that intelligently expands `conmeds_defaults.yml` files for any therapeutic area:

1. **Clinical data analysis** - Identify medication columns and extract unique drug names
2. **Disease-specific context** - Load configurable disease modules with drug class definitions
3. **Web research** - Gather comprehensive drug information from medical databases (FDA, clinical guidelines, etc.)
4. **LLM classification** - Categorize medications using disease-specific therapeutic context via Claude CLI
5. **YAML export** - Generate expanded conmeds.yml with comprehensive drug name coverage

**Key Innovation:** Disease modules are completely interchangeable - the same pipeline works for NSCLC, breast cancer, prostate cancer, or any therapeutic area by simply switching the disease module.

## Architecture

```
Raw Data  ->  Column Detection  ->  Name Extraction  ->  Web Research  ->  LLM Classification  ->  Validation  ->  conmeds.yml
```

### Core Components

- **Pipeline Orchestrator** - Manages multi-phase execution with checkpoint recovery
- **Data Analyzer** - Identifies medication columns in clinical datasets
- **Name Extractor** - Cleans and normalizes medication names
- **Disease Module Registry** - Pluggable disease-specific configurations (NSCLC, breast cancer, etc.)
- **Web Scrapers** - Gather drug information from medical databases (configurable per disease)
- **LLM Classifier** - Categorizes medications using disease-specific therapeutic context
- **Conmeds Exporter** - Generates production-ready conmeds.yml files for any disease

## Success Metrics

- **Coverage expansion**: Expand drug classes with 20-50+ names each (NSCLC: 54 → 70+ classes)
- **Matching improvement**: 30%+ increase in medication matching rates across diseases
- **Processing efficiency**: 80% reduction in manual review time
- **Scalability**: New diseases can be added in days, not months
- **False negative reduction**: 50%+ decrease in missed therapeutic area medications

## Installation

```bash
# Clone repository
git clone <repository-url>
cd medication-augmentation

# Install dependencies
uv install

# Install development tools
uv add --dev black pytest ruff
```

## Quick Start

### 1. Basic Pipeline Execution

```bash
# Run for NSCLC (lung cancer)
python -m src.med_aug.cli.app pipeline run \
  --input data/clinical_data.csv \
  --disease nsclc \
  --output ./output

# Run for breast cancer (when module is available)
python -m src.med_aug.cli.app pipeline run \
  --input data/breast_cancer_data.csv \
  --disease breast_cancer \
  --output ./output

# Enable LLM classification for any disease
python -m src.med_aug.cli.app pipeline run \
  --input data/clinical_data.csv \
  --disease nsclc \
  --enable-llm \
  --output ./output
```

### 2. Disease Module Management

```bash
# List all available disease modules
python -m src.med_aug.cli.app diseases list

# Show specific disease module details
python -m src.med_aug.cli.app diseases show nsclc
python -m src.med_aug.cli.app diseases show breast_cancer

# Validate a disease module configuration
python -m src.med_aug.cli.app diseases validate nsclc
```

### 3. Output Files

The pipeline generates:
- **`conmeds_augmented.yml`** - Enhanced medication database (primary deliverable)
- **`classification_results.json`** - Detailed classification results with confidence scores
- **`pipeline_summary.json`** - Execution summary and statistics

## Configuration

### Pipeline Configuration

```python
from src.med_aug.pipeline import PipelineConfig

config = PipelineConfig(
    input_file="data/clinical_data.csv",
    output_path="./output",
    disease_module="nsclc",  # Or "breast_cancer", "prostate_cancer", etc.
    enable_llm_classification=True,  # Requires Claude CLI
    llm_provider="claude_cli"
)
```

### Disease Module Examples

**NSCLC (Non-Small Cell Lung Cancer)** - 54+ drug classes:
- **Chemotherapy**: paclitaxel, carboplatin, pemetrexed, docetaxel, gemcitabine
- **Immunotherapy**: pembrolizumab, nivolumab, atezolizumab, durvalumab
- **Targeted Therapy**: osimertinib, erlotinib, crizotinib, alectinib, lorlatinib

**Future Disease Modules** (architecture ready):
- **Breast Cancer**: CDK4/6 inhibitors, HER2-targeted therapy, hormone therapy, etc.
- **Prostate Cancer**: Androgen receptor inhibitors, chemotherapy, bone-targeting agents, etc.

See `src/med_aug/diseases/` for complete disease module definitions and examples of creating new modules.

## Data Sources

### Input Requirements
- Clinical datasets with medication name columns (CSV, Excel, Parquet formats)
- Examples: MSK CHORD dataset, clinical trial data, EHR medication records

### Web Sources (Disease-Configurable)
- **FDA.gov** - Drug approvals and labeling information (all diseases)
- **ClinicalTrials.gov** - Clinical trial medication data (disease-specific searches)
- **Clinical Guidelines** - NCCN, ASCO, disease-specific treatment guidelines
- **Disease Databases** - OncoKB (oncology), specialty databases per therapeutic area

## Development

### Project Structure

```
src/med_aug/
├── cli/                    # Command-line interface
│   ├── app.py             # Main CLI application
│   └── commands/          # CLI command implementations
├── pipeline/               # Core pipeline components
│   ├── orchestrator.py    # Pipeline execution management
│   ├── phases.py          # Individual processing phases
│   ├── checkpoint.py      # Recovery and resumption
│   └── progress.py        # Execution tracking
├── diseases/               # Disease-specific modules
│   ├── nsclc/             # NSCLC drug definitions
│   ├── base.py            # Base disease module interface
│   └── registry.py        # Disease module registry
├── infrastructure/         # External service integrations
│   └── scrapers/          # Web scraping implementations
├── llm/                   # LLM integration (Claude CLI)
├── core/                  # Data processing utilities
└── output/                # Export functionality
    └── exporters.py       # ConmedsYAMLExporter
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/pipeline/
pytest tests/unit/output/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type check
mypy src/
```

## Usage Examples

### Process MSK CHORD Dataset

```bash
# Complete NSCLC augmentation with LLM classification
python -m src.med_aug.cli.app pipeline run \
  --input s3://dataset/nsclc_medication_data.csv \
  --disease nsclc \
  --enable-llm \
  --output ./nsclc_output

# Process breast cancer dataset
python -m src.med_aug.cli.app pipeline run \
  --input data/breast_cancer_treatments.csv \
  --disease breast_cancer \
  --enable-llm \
  --output ./breast_cancer_output
```

### Resume Failed Pipeline

```bash
# Resume from specific phase (works for any disease)
python -m src.med_aug.cli.app pipeline run \
  --input data/clinical_data.csv \
  --disease nsclc \
  --resume-from llm_classification
```

## Complete Workflow Example

📋 **See [WORKFLOW_EXAMPLE.md](WORKFLOW_EXAMPLE.md) for a detailed end-to-end example** showing how the system processes real clinical data to augment conmeds.yml files with comprehensive medication coverage.

The example demonstrates:
- Processing clinical trial data with mixed medication nomenclature
- LLM-assisted classification and evaluation
- Quality assurance preventing false positives
- Generation of augmented conmeds.yml with new medication names

## Integration

The generated `conmeds_augmented.yml` file can be directly integrated into existing clinical pipelines for any disease:

```python
# Load augmented medication database
import yaml

with open('output/conmeds_augmented.yml', 'r') as f:
    conmeds = yaml.safe_load(f)

# Use in medication enrichment
for drug_class, medications in conmeds.items():
    # Apply regex matching: \b({medications_pattern})\b
    pattern = '|'.join(re.escape(med) for med in medications)
    # Generate taking_{drug_class} boolean indicators
```

## Troubleshooting

### Claude CLI Setup
```bash
# Install Claude CLI
npm install -g @anthropic-ai/claude-cli

# Verify installation
claude --version
```

### Common Issues

**ModuleNotFoundError**: Install dependencies with `uv install`

**LLM Classification Disabled**: Ensure Claude CLI is installed and accessible

**Web Scraping Errors**: Check network connectivity and rate limiting

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Make changes and add tests
4. Ensure code quality (`black`, `ruff`, `mypy`)
5. Submit a pull request

## License

[License information]

## Support

For issues and questions:
- Create GitHub issues for bugs and feature requests
- Review documentation in `docs/` directory
- Check troubleshooting section above

---

**Status**: ✅ Production-ready multi-disease medication augmentation framework focused on expanding conmeds.yml with comprehensive drug name coverage across therapeutic areas.

## Adding New Disease Modules

The system is designed for easy extension to new diseases:

1. **Create Disease Module**: Define drug classes and validation rules in `src/med_aug/diseases/your_disease/`
2. **Configure Web Sources**: Specify disease-specific medical databases and guidelines
3. **Test with Pipeline**: Run the existing pipeline with `--disease your_disease`
4. **No Code Changes**: The core pipeline automatically adapts to new disease modules

See the [Disease Module Development Guide](src/med_aug/diseases/README.md) for detailed instructions.
