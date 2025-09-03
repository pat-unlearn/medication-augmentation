# Diseases Module

## Overview

The diseases module provides a flexible, plugin-based architecture for disease-specific medication validation and classification. It supports multiple disease types with configurable drug classes, validation rules, and metadata management.

## Structure

```
diseases/
├── __init__.py
├── base.py            # Base disease module interface
├── registry.py        # Disease module registry and discovery
├── nsclc.py           # NSCLC (Non-Small Cell Lung Cancer) module
└── drug_classes.yaml  # Drug class configurations
```

## Key Components

### Base Disease Module (`base.py`)

Abstract base class defining the disease module interface:

```python
from med_aug.diseases import DiseaseModule, DrugClass

class CustomDiseaseModule(DiseaseModule):
    def __init__(self):
        super().__init__(
            name="custom_disease",
            full_name="Custom Disease Type",
            description="Disease-specific module"
        )
    
    def validate_medication(self, medication: str, drug_class: str) -> bool:
        # Custom validation logic
        pass
    
    def get_drug_classes(self) -> List[DrugClass]:
        # Return supported drug classes
        pass
```

### Disease Registry (`registry.py`)

Central registry for disease module management:

```python
from med_aug.diseases import DiseaseRegistry

registry = DiseaseRegistry.get_instance()

# Register a module
registry.register(NSCLCModule())

# Get a module
module = registry.get("nsclc")

# List all modules
modules = registry.list_modules()

# Auto-discover modules
registry.auto_discover()
```

### NSCLC Module (`nsclc.py`)

Complete implementation for Non-Small Cell Lung Cancer:

```python
from med_aug.diseases import NSCLCModule

module = NSCLCModule()

# Validate a medication
is_valid = module.validate_medication("pembrolizumab", "pd1_inhibitors")

# Get all drug classes
drug_classes = module.drug_classes

# Get keywords for a drug class
keywords = module.get_keywords("egfr_inhibitors")
```

## NSCLC Drug Classes

The NSCLC module includes 10 comprehensive drug classes:

### 1. PD-1/PD-L1 Inhibitors
- **Examples**: Pembrolizumab, Nivolumab, Atezolizumab
- **Keywords**: mab, pd1, pdl1, checkpoint
- **Use Case**: Immunotherapy for advanced NSCLC

### 2. EGFR Inhibitors
- **Examples**: Osimertinib, Erlotinib, Gefitinib
- **Keywords**: tinib, egfr, tkis
- **Use Case**: EGFR mutation-positive NSCLC

### 3. ALK Inhibitors
- **Examples**: Alectinib, Crizotinib, Brigatinib
- **Keywords**: tinib, alk
- **Use Case**: ALK-positive NSCLC

### 4. Platinum Chemotherapy
- **Examples**: Cisplatin, Carboplatin, Oxaliplatin
- **Keywords**: platin, platinum
- **Use Case**: First-line combination therapy

### 5. Taxanes
- **Examples**: Paclitaxel, Docetaxel, Nab-paclitaxel
- **Keywords**: taxel, taxane
- **Use Case**: Combination chemotherapy

### 6. Antimetabolites
- **Examples**: Pemetrexed, Gemcitabine, Methotrexate
- **Keywords**: trexed, cytidine
- **Use Case**: Non-squamous NSCLC

### 7. Vinca Alkaloids
- **Examples**: Vinorelbine, Vincristine, Vinblastine
- **Keywords**: vin, alkaloid
- **Use Case**: Combination therapy

### 8. ROS1 Inhibitors
- **Examples**: Crizotinib, Entrectinib, Lorlatinib
- **Keywords**: tinib, ros1
- **Use Case**: ROS1-positive NSCLC

### 9. MET Inhibitors
- **Examples**: Capmatinib, Tepotinib, Savolitinib
- **Keywords**: tinib, met, c-met
- **Use Case**: MET exon 14 skipping mutations

### 10. VEGF Inhibitors
- **Examples**: Bevacizumab, Ramucirumab, Aflibercept
- **Keywords**: mab, vegf, angio
- **Use Case**: Anti-angiogenic therapy

## Drug Class Configuration

### YAML Configuration Format

```yaml
drug_classes:
  - name: pd1_inhibitors
    display_name: "PD-1/PD-L1 Inhibitors"
    keywords:
      - mab
      - pd1
      - pdl1
    medications:
      - pembrolizumab
      - nivolumab
      - atezolizumab
    description: "Checkpoint inhibitors for immunotherapy"
    metadata:
      approval_year: 2014
      mechanism: "PD-1/PD-L1 pathway blockade"
```

### Programmatic Configuration

```python
from med_aug.diseases import DrugClass

drug_class = DrugClass(
    name="targeted_therapy",
    display_name="Targeted Therapy Agents",
    keywords=["mab", "nib"],
    medications=["drug1", "drug2"],
    description="Targeted cancer therapies"
)
```

## Creating Custom Disease Modules

### Step 1: Define the Module

```python
from med_aug.diseases import DiseaseModule, DrugClass
from typing import List, Dict

class BreastCancerModule(DiseaseModule):
    def __init__(self):
        super().__init__(
            name="breast_cancer",
            full_name="Breast Cancer",
            description="Medications for breast cancer treatment"
        )
        self._load_drug_classes()
    
    def _load_drug_classes(self):
        self.drug_classes = [
            DrugClass(
                name="her2_inhibitors",
                display_name="HER2 Inhibitors",
                keywords=["mab", "her2"],
                medications=["trastuzumab", "pertuzumab"],
                description="HER2-targeted therapies"
            ),
            # Add more drug classes
        ]
    
    def validate_medication(self, medication: str, drug_class: str) -> bool:
        # Implement validation logic
        for dc in self.drug_classes:
            if dc.name == drug_class:
                return self._check_medication(medication, dc)
        return False
```

### Step 2: Register the Module

```python
from med_aug.diseases import DiseaseRegistry

registry = DiseaseRegistry.get_instance()
registry.register(BreastCancerModule())
```

### Step 3: Use the Module

```python
# In pipeline configuration
config = PipelineConfig(
    disease_module="breast_cancer",
    # Other settings
)

# Direct usage
module = registry.get("breast_cancer")
is_valid = module.validate_medication("trastuzumab", "her2_inhibitors")
```

## Validation Rules

### Pattern-Based Validation

```python
def validate_by_pattern(medication: str, patterns: List[str]) -> bool:
    med_lower = medication.lower()
    for pattern in patterns:
        if pattern in med_lower:
            return True
    return False
```

### Exact Match Validation

```python
def validate_by_exact_match(medication: str, approved_list: List[str]) -> bool:
    normalized = medication.lower().strip()
    return normalized in [m.lower() for m in approved_list]
```

### Fuzzy Matching

```python
from difflib import get_close_matches

def validate_fuzzy(medication: str, approved_list: List[str], cutoff=0.8) -> bool:
    matches = get_close_matches(
        medication.lower(),
        [m.lower() for m in approved_list],
        cutoff=cutoff
    )
    return len(matches) > 0
```

## Integration with Pipeline

The disease modules integrate seamlessly with the pipeline:

```python
# In validation phase
validation_phase = ValidationPhase(config)
results = validation_phase.execute(medications)

# Automatic module loading
pipeline = PipelineOrchestrator(config)
# Module is loaded based on config.disease_module
```

## CLI Commands

```bash
# List available disease modules
med-aug diseases list

# Get module information
med-aug diseases info nsclc

# Show drug keywords
med-aug diseases keywords nsclc

# Validate medications
med-aug diseases validate nsclc pembrolizumab osimertinib

# Export configuration
med-aug diseases export nsclc --output nsclc_config.yaml
```

## Testing

Comprehensive test coverage for:
- Module registration and discovery
- Medication validation accuracy
- Drug class configuration
- Pattern matching logic
- CLI command functionality

## Best Practices

1. **Modular Design**: Keep disease modules self-contained
2. **Configuration-Driven**: Use YAML for drug class definitions
3. **Keyword Optimization**: Choose specific, unambiguous keywords
4. **Regular Updates**: Keep medication lists current
5. **Documentation**: Document approval status and usage guidelines
6. **Validation Layers**: Combine multiple validation strategies
7. **Testing**: Maintain high test coverage for validation logic

## Metadata Management

Each disease module can track extensive metadata:

```python
metadata = {
    "version": "1.0.0",
    "last_updated": "2025-01-01",
    "data_sources": ["FDA", "NCCN", "OncoKB"],
    "approval_status": {
        "pembrolizumab": {
            "fda_approved": True,
            "approval_date": "2014-09-04",
            "indications": ["Advanced NSCLC", "PD-L1 positive"]
        }
    },
    "clinical_trials": {
        "active": 125,
        "completed": 450
    }
}
```

## Future Enhancements Ideas

- Machine learning-based validation
- Real-time drug database integration
- Multi-disease combination therapy support
- Automatic keyword extraction from literature
- Version control for drug class configurations
- Integration with clinical decision support systems
