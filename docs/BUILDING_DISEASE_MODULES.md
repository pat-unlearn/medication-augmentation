# 🏗️ Building Disease Modules

## 📋 Overview

The medication augmentation system uses a modular architecture where each disease is defined as a separate module with its own drug classes, validation rules, and clinical context. This guide walks you through creating a new disease module from scratch.

## 🎯 Architecture Overview

Disease modules are self-contained units that provide:
- **Drug class definitions** with keywords and confidence thresholds
- **Clinical validation logic** for medication-drug class matching
- **LLM context** for intelligent classification
- **Web source configuration** for research and validation
- **Disease-specific metadata** for enhanced accuracy

## 🚀 Step-by-Step Build Process

### **Step 1: Create the Module Directory Structure**

```bash
# Create the new disease module directory
mkdir -p src/med_aug/diseases/breast_cancer
touch src/med_aug/diseases/breast_cancer/__init__.py
touch src/med_aug/diseases/breast_cancer/module.py
```

### **Step 2: Define the Disease Module Class**

Create `src/med_aug/diseases/breast_cancer/module.py`:

```python
"""Breast Cancer disease module."""

from typing import List
from ..base import DiseaseModule, DrugClassConfig


class BreastCancerModule(DiseaseModule):
    """Breast Cancer disease module implementation."""

    @property
    def name(self) -> str:
        """Disease identifier."""
        return "breast_cancer"

    @property
    def display_name(self) -> str:
        """Human-readable disease name."""
        return "Breast Cancer"

    @property
    def drug_classes(self) -> List[DrugClassConfig]:
        """Breast cancer-specific drug class configurations."""
        return [
            # Hormone therapy
            DrugClassConfig(
                name="hormone_therapy",
                keywords=[
                    "tamoxifen",
                    "letrozole",
                    "anastrozole",
                    "exemestane",
                    "fulvestrant",
                    "abemaciclib",
                    "palbociclib",
                    "ribociclib",
                    "nolvadex",
                    "femara",
                    "arimidex",
                    "aromasin",
                    "faslodx",
                    "verzenio",
                    "ibrance",
                    "kisqali",
                ],
                confidence_threshold=0.85,
                web_sources=["fda", "nccn", "oncokb"],
            ),
            # HER2-targeted therapy
            DrugClassConfig(
                name="her2_targeted",
                keywords=[
                    "trastuzumab",
                    "pertuzumab",
                    "ado-trastuzumab emtansine",
                    "trastuzumab deruxtecan",
                    "lapatinib",
                    "neratinib",
                    "tucatinib",
                    "herceptin",
                    "perjeta",
                    "kadcyla",
                    "enhertu",
                    "tykerb",
                    "nerlynx",
                    "tukysa",
                    "margetuximab",
                    "margenza",
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb", "nccn"],
            ),
            # CDK4/6 inhibitors
            DrugClassConfig(
                name="cdk46_inhibitors",
                keywords=[
                    "palbociclib",
                    "ribociclib",
                    "abemaciclib",
                    "ibrance",
                    "kisqali",
                    "verzenio",
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb"],
            ),
            # Chemotherapy (breast cancer specific)
            DrugClassConfig(
                name="chemotherapy",
                keywords=[
                    "doxorubicin",
                    "cyclophosphamide",
                    "paclitaxel",
                    "docetaxel",
                    "carboplatin",
                    "epirubicin",
                    "capecitabine",
                    "gemcitabine",
                    "vinorelbine",
                    "eribulin",
                    "adriamycin",
                    "cytoxan",
                    "taxol",
                    "taxotere",
                    "ellence",
                    "xeloda",
                    "gemzar",
                    "navelbine",
                    "halaven",
                ],
                confidence_threshold=0.8,
                web_sources=["fda", "nccn"],
            ),
            # mTOR inhibitors
            DrugClassConfig(
                name="mtor_inhibitors",
                keywords=[
                    "everolimus",
                    "afinitor",
                    "temsirolimus",
                    "torisel",
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb"],
            ),
            # PI3K inhibitors
            DrugClassConfig(
                name="pi3k_inhibitors",
                keywords=[
                    "alpelisib",
                    "piqray",
                    "inavolisib",
                    "copanlisib",
                    "aliqopa",
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb", "clinicaltrials"],
            ),
            # Immunotherapy
            DrugClassConfig(
                name="immunotherapy",
                keywords=[
                    "pembrolizumab",
                    "keytruda",
                    "atezolizumab",
                    "tecentriq",
                    "sacituzumab govitecan",
                    "trodelvy",
                    "dostarlimab",
                    "jemperli",
                ],
                confidence_threshold=0.85,
                web_sources=["fda", "nccn", "oncokb"],
            ),
        ]

    def get_web_sources(self) -> List[str]:
        """Breast cancer-specific data sources."""
        return [
            "https://www.fda.gov/drugs/resources-information-approved-drugs/oncology-cancer-hematologic-malignancies-approval-notifications",
            "https://clinicaltrials.gov/search?cond=Breast%20Cancer",
            "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1419",
            "https://www.oncokb.org/cancer-genes/breast",
            "https://www.breastcancer.org/treatment",
            "https://www.cancer.gov/types/breast/treatment",
        ]

    def get_llm_context(self) -> str:
        """Breast cancer-specific context for LLM classification."""
        return """You are a clinical oncologist specializing in breast cancer treatment.
Your expertise includes hormone receptor status, HER2 status, and molecular subtypes.

Current breast cancer treatment landscape (2024-2025):

HORMONE RECEPTOR POSITIVE (HR+/HER2-):
- CDK4/6 inhibitors: Palbociclib (Ibrance), Ribociclib (Kisqali), Abemaciclib (Verzenio)
- Aromatase inhibitors: Letrozole (Femara), Anastrozole (Arimidex), Exemestane (Aromasin)
- SERMs: Tamoxifen (Nolvadex)
- SERDs: Fulvestrant (Faslodx), Elacestrant (Orserdu)
- PI3K inhibitors: Alpelisib (Piqray) for PIK3CA mutations
- mTOR inhibitors: Everolimus (Afinitor)

HER2-POSITIVE BREAST CANCER:
- Anti-HER2 antibodies: Trastuzumab (Herceptin), Pertuzumab (Perjeta)
- ADCs: Ado-trastuzumab emtansine (Kadcyla), Trastuzumab deruxtecan (Enhertu)
- TKIs: Lapatinib (Tykerb), Neratinib (Nerlynx), Tucatinib (Tukysa)
- Margetuximab (Margenza)

TRIPLE-NEGATIVE BREAST CANCER (TNBC):
- Immunotherapy: Pembrolizumab (Keytruda), Atezolizumab (Tecentriq)
- ADCs: Sacituzumab govitecan (Trodelvy)
- PARP inhibitors: Olaparib (Lynparza), Talazoparib (Talzenna)

CHEMOTHERAPY BACKBONE:
- Anthracyclines: Doxorubicin (Adriamycin), Epirubicin (Ellence)
- Alkylating agents: Cyclophosphamide (Cytoxan)
- Taxanes: Paclitaxel (Taxol), Docetaxel (Taxotere)
- Platinums: Carboplatin
- Others: Capecitabine (Xeloda), Eribulin (Halaven), Gemcitabine (Gemzar)

Classify medications considering receptor status and treatment setting."""

    def validate_medication(self, medication: str, drug_class: str) -> bool:
        """Breast cancer-specific medication validation."""
        medication_lower = medication.lower().strip()

        # Known breast cancer medications by class
        known_bc_meds = {
            "hormone_therapy": [
                "tamoxifen", "nolvadx", "letrozole", "femara",
                "anastrozole", "arimidex", "exemestane", "aromasin",
                "fulvestrant", "faslodx", "elacestrant", "orserdu"
            ],
            "cdk46_inhibitors": [
                "palbociclib", "ibrance", "ribociclib", "kisqali",
                "abemaciclib", "verzenio"
            ],
            "her2_targeted": [
                "trastuzumab", "herceptin", "pertuzumab", "perjeta",
                "ado-trastuzumab emtansine", "kadcyla",
                "trastuzumab deruxtecan", "enhertu",
                "lapatinib", "tykerb", "neratinib", "nerlynx",
                "tucatinib", "tukysa", "margetuximab", "margenza"
            ],
        }

        # Check against known medications
        if drug_class in known_bc_meds:
            for known_med in known_bc_meds[drug_class]:
                if known_med in medication_lower or medication_lower in known_med:
                    return True

        # Check against keywords
        drug_class_config = self.get_drug_class_by_name(drug_class)
        if drug_class_config:
            for keyword in drug_class_config.keywords:
                if keyword.lower() in medication_lower:
                    return True

        return drug_class not in known_bc_meds


# Register the module for auto-discovery
MODULE_CLASS = BreastCancerModule
```

### **Step 3: Update Module Registry**

Add to `src/med_aug/diseases/breast_cancer/__init__.py`:

```python
"""Breast Cancer disease module."""

from .module import BreastCancerModule

__all__ = ["BreastCancerModule"]
```

### **Step 4: Register in Main Diseases Module**

Update `src/med_aug/diseases/__init__.py`:

```python
"""Disease modules for medication augmentation."""

from .base import DiseaseModule, DrugClassConfig, DiseaseModuleConfig
from .nsclc import NSCLCModule
from .breast_cancer import BreastCancerModule  # Add this line

# Auto-registration
AVAILABLE_MODULES = {
    "nsclc": NSCLCModule,
    "breast_cancer": BreastCancerModule,  # Add this line
}

def get_disease_module(name: str) -> DiseaseModule:
    """Get a disease module by name."""
    if name not in AVAILABLE_MODULES:
        raise ValueError(f"Unknown disease module: {name}")
    return AVAILABLE_MODULES[name]()

def list_available_diseases() -> list[str]:
    """List all available disease modules."""
    return list(AVAILABLE_MODULES.keys())

__all__ = [
    "DiseaseModule",
    "DrugClassConfig",
    "DiseaseModuleConfig",
    "NSCLCModule",
    "BreastCancerModule",  # Add this line
    "get_disease_module",
    "list_available_diseases",
]
```

### **Step 5: Test Your New Module**

```bash
# Activate environment
source .venv/bin/activate

# List available modules (should include breast_cancer)
med-aug diseases list

# Show details about your new module
med-aug diseases show breast_cancer

# Validate the module configuration
med-aug diseases validate breast_cancer

# Test with sample data
med-aug pipeline run breast_cancer_data.csv \
  --conmeds data/conmeds_defaults.yml \
  --disease breast_cancer \
  --output ./breast_cancer_results
```

### **Step 6: Create Tests**

Create `tests/unit/diseases/test_breast_cancer_module.py`:

```python
"""Tests for breast cancer disease module."""

import pytest
from med_aug.diseases.breast_cancer import BreastCancerModule


class TestBreastCancerModule:
    """Test breast cancer disease module."""

    def setup_method(self):
        """Set up test fixtures."""
        self.module = BreastCancerModule()

    def test_module_properties(self):
        """Test module basic properties."""
        assert self.module.name == "breast_cancer"
        assert self.module.display_name == "Breast Cancer"
        assert len(self.module.drug_classes) > 0

    def test_hormone_therapy_validation(self):
        """Test hormone therapy medication validation."""
        assert self.module.validate_medication("tamoxifen", "hormone_therapy")
        assert self.module.validate_medication("Nolvadex", "hormone_therapy")
        assert self.module.validate_medication("letrozole", "hormone_therapy")
        assert not self.module.validate_medication("pembrolizumab", "hormone_therapy")

    def test_her2_targeted_validation(self):
        """Test HER2-targeted therapy validation."""
        assert self.module.validate_medication("trastuzumab", "her2_targeted")
        assert self.module.validate_medication("Herceptin", "her2_targeted")
        assert self.module.validate_medication("pertuzumab", "her2_targeted")

    def test_drug_class_config(self):
        """Test drug class configurations."""
        configs = {dc.name: dc for dc in self.module.drug_classes}

        assert "hormone_therapy" in configs
        assert "her2_targeted" in configs
        assert "cdk46_inhibitors" in configs

        # Check confidence thresholds are reasonable
        for config in configs.values():
            assert 0.7 <= config.confidence_threshold <= 1.0

    def test_web_sources(self):
        """Test web source configuration."""
        sources = self.module.get_web_sources()
        assert len(sources) > 0
        assert any("fda.gov" in source for source in sources)
        assert any("nccn.org" in source for source in sources)

    def test_llm_context(self):
        """Test LLM context generation."""
        context = self.module.get_llm_context()
        assert "breast cancer" in context.lower()
        assert "her2" in context.lower()
        assert "hormone receptor" in context.lower()
```

## 📚 Information Sources for Drug Class Configuration

Creating accurate drug class configurations requires gathering information from multiple authoritative sources. Here's a comprehensive guide on where and how to obtain this critical information.

### **1. Primary Clinical Guidelines (Most Important)**

#### **Cancer/Oncology Sources:**
- **NCCN Guidelines** (nccn.org) - Gold standard for cancer drug classifications
  ```bash
  # Example NCCN sections to review:
  # - "Systemic Therapy for Advanced Disease"
  # - "Targeted Therapy by Biomarker"
  # - "Adjuvant Treatment Recommendations"
  ```
- **ASCO Guidelines** (asco.org) - Clinical practice recommendations
- **ESMO Guidelines** (esmo.org) - European clinical guidelines
- **FDA Hematology/Oncology Approvals** - Official drug approvals and indications

#### **Cardiovascular Sources:**
- **ACC/AHA Guidelines** - American College of Cardiology/American Heart Association
- **ESC Guidelines** - European Society of Cardiology
- **AHA/ACC Clinical Practice Guidelines** - Treatment recommendations

#### **Other Disease Areas:**
- **American Diabetes Association** - Diabetes medications
- **American Thoracic Society** - Respiratory medications
- **American Gastroenterological Association** - GI medications

### **2. Regulatory and Official Sources**

#### **FDA Resources:**
```python
regulatory_sources = {
    "fda_orange_book": "https://www.fda.gov/drugs/drug-approvals-and-databases/approved-drug-products-therapeutic-equivalence-evaluations-orange-book",
    "fda_drug_approvals": "https://www.fda.gov/drugs/drug-approvals-and-databases",
    "fda_oncology_approvals": "https://www.fda.gov/drugs/resources-information-approved-drugs/oncology-cancer-hematologic-malignancies-approval-notifications"
}
```

#### **Clinical Trials Database:**
```python
# Get emerging therapies and development pipeline
clinical_trials_queries = {
    "breast_cancer": "https://clinicaltrials.gov/search?cond=Breast%20Cancer&aggFilters=status:rec,ages:adult",
    "nsclc": "https://clinicaltrials.gov/search?cond=Non-Small%20Cell%20Lung%20Cancer&aggFilters=phase:2,phase:3"
}
```

### **3. Pharmaceutical Knowledge Bases**

#### **OncoKB (Cancer-Specific):**
```python
oncokb_sources = {
    "breast_cancer": "https://www.oncokb.org/cancer-genes/breast",
    "lung_cancer": "https://www.oncokb.org/cancer-genes/lung",
    "prostate_cancer": "https://www.oncokb.org/cancer-genes/prostate"
}
```

#### **General Drug Information:**
- **Drugs.com** - Generic/brand name mappings
- **Lexicomp** - Clinical drug information
- **UpToDate** - Evidence-based drug classifications
- **Micromedex** - Drug interaction and classification data

### **4. Information Gathering Workflow**

#### **Step 1: Start with Current Guidelines**
```python
# Example research workflow for breast cancer
research_plan = {
    "primary_sources": [
        "NCCN Breast Cancer Guidelines v2.2024",
        "ASCO Breast Cancer Treatment Guidelines 2024",
        "ESMO Advanced Breast Cancer Consensus 2023"
    ],
    "regulatory_sources": [
        "FDA Breast Cancer Drug Approvals 2020-2024",
        "EMA Breast Cancer Approvals"
    ],
    "knowledge_bases": [
        "OncoKB Breast Cancer Genes and Drugs",
        "ClinicalTrials.gov Breast Cancer Studies Phase 2/3"
    ]
}
```

#### **Step 2: Extract Drug Classes from Guidelines**

**Example from NCCN Breast Cancer Guidelines:**
```yaml
# From guideline section: "CDK4/6 inhibitor + endocrine therapy for HR+/HER2-"
extracted_drug_class:
  name: "cdk46_inhibitors"
  clinical_context: "Used with aromatase inhibitors for HR+/HER2- metastatic breast cancer"
  guideline_source: "NCCN Breast Cancer v2.2024, page 85"
  fda_approvals:
    - "Palbociclib: Feb 2015"
    - "Ribociclib: Mar 2017"
    - "Abemaciclib: Sep 2017"
```

#### **Step 3: Cross-Reference Multiple Sources**
```python
def cross_reference_drug_sources(disease: str):
    """Validate drug information across multiple authoritative sources."""

    # Get drugs from each source
    nccn_drugs = extract_from_nccn_guidelines(disease)
    fda_drugs = query_fda_approvals(disease)
    oncokb_drugs = get_oncokb_drugs(disease)
    trials_drugs = get_clinical_trials_drugs(disease)

    # Find consensus drugs (appear in multiple sources)
    consensus_drugs = find_consensus(nccn_drugs, fda_drugs, oncokb_drugs)

    # Flag emerging drugs (in trials but not yet approved)
    emerging_drugs = trials_drugs - consensus_drugs

    return {
        "established": consensus_drugs,
        "emerging": emerging_drugs,
        "validation_sources": len([nccn_drugs, fda_drugs, oncokb_drugs])
    }
```

#### **Step 4: Clinical Expert Validation**
```python
# Expert validation process
expert_validation = {
    "clinical_sme": "Disease specialist (oncologist, cardiologist, etc.)",
    "clinical_pharmacist": "Drug classification and interaction expert",
    "validation_checklist": [
        "Are drug classes clinically meaningful?",
        "Are confidence thresholds appropriate?",
        "Are any important drugs missing?",
        "Are any inappropriate drugs included?",
        "Do classifications match current practice?"
    ]
}
```

### **5. Practical Research Examples**

#### **NSCLC Drug Class Research Process:**
```python
nsclc_research_workflow = {
    "step_1": {
        "source": "NCCN NSCLC Guidelines Section 2",
        "extract": "Systemic Therapy for Advanced Disease",
        "drug_classes": ["EGFR inhibitors", "ALK inhibitors", "Immunotherapy"]
    },
    "step_2": {
        "source": "FDA Lung Cancer Approvals",
        "validate": "Official drug names and approval dates",
        "cross_reference": "Match NCCN recommendations with FDA approvals"
    },
    "step_3": {
        "source": "OncoKB Lung Cancer",
        "enhance": "Biomarker-driven therapy associations",
        "examples": "EGFR mutations → osimertinib, ALK+ → alectinib"
    }
}
```

#### **Breast Cancer Information Sources:**
```python
breast_cancer_sources = {
    "guidelines": {
        "nccn_breast_2024": {
            "sections": ["Systemic Therapy Options", "Targeted Therapy by Subtype"],
            "drug_classes": ["Hormone therapy", "HER2-targeted", "CDK4/6 inhibitors"]
        },
        "asco_breast_2024": {
            "focus": "Treatment recommendations and sequencing",
            "validation": "Cross-reference with NCCN classifications"
        }
    },
    "regulatory": {
        "fda_approvals": "Official drug names, indications, approval dates",
        "recent_approvals": "2023-2024 new drug approvals for breast cancer"
    },
    "knowledge_bases": {
        "oncokb": "Biomarker associations (HR+/HER2-, HER2+, TNBC)",
        "clinical_trials": "Emerging therapies in development"
    }
}
```

### **6. Automated Information Gathering**

#### **Web Scraping (With Permission):**
```python
def automated_drug_research(disease: str):
    """Automated information gathering from public sources."""

    # Query FDA API for approved drugs
    fda_drugs = query_fda_drug_api(disease)

    # Search ClinicalTrials.gov for emerging therapies
    trial_drugs = search_clinical_trials(disease, phases=["Phase 2", "Phase 3"])

    # Get drug-disease associations from public databases
    drug_associations = query_public_databases(disease)

    return {
        "approved_drugs": fda_drugs,
        "emerging_drugs": trial_drugs,
        "associations": drug_associations
    }
```

#### **API Integration Examples:**
```python
# FDA API example
def get_fda_drug_approvals(disease_area: str):
    """Get FDA-approved drugs for specific disease area."""
    api_url = "https://api.fda.gov/drug/drugsfda.json"
    params = {
        "search": f"products_and_ingredients.indication_and_usage:{disease_area}",
        "limit": 100
    }
    response = requests.get(api_url, params=params)
    return parse_fda_response(response.json())

# ClinicalTrials.gov API example
def get_clinical_trials_drugs(condition: str):
    """Get drugs in clinical trials for a condition."""
    api_url = "https://clinicaltrials.gov/api/query/study_fields"
    params = {
        "expr": condition,
        "fields": "InterventionName,Phase,OverallStatus",
        "min_rnk": 1,
        "max_rnk": 100,
        "fmt": "json"
    }
    response = requests.get(api_url, params=params)
    return parse_trials_response(response.json())
```

### **7. Quality Assurance Checklist**

Before implementing drug class configurations, ensure:

#### **✅ Clinical Validation:**
- [ ] Current treatment guidelines reviewed (published within 2 years)
- [ ] FDA-approved drugs included with correct names
- [ ] Standard-of-care drug classes represented
- [ ] Biomarker-driven treatment patterns included
- [ ] Common combination therapies accounted for

#### **✅ Source Validation:**
- [ ] Multiple authoritative sources cross-referenced
- [ ] Clinical expert review completed
- [ ] Recent approval updates included (last 2 years)
- [ ] Emerging therapy pipeline considered

#### **✅ Technical Validation:**
- [ ] Generic and brand names mapped correctly
- [ ] Confidence thresholds clinically justified
- [ ] Keywords comprehensive but specific
- [ ] Web sources accessible and current

### **8. Maintenance and Updates**

#### **Regular Update Schedule:**
```python
maintenance_schedule = {
    "quarterly": [
        "Review FDA new drug approvals",
        "Check for guideline updates",
        "Validate confidence thresholds with clinical data"
    ],
    "annually": [
        "Major guideline review (NCCN, ASCO, ESMO)",
        "Comprehensive clinical expert review",
        "Pipeline and emerging therapy assessment"
    ],
    "as_needed": [
        "Breakthrough therapy designations",
        "Major clinical trial results",
        "Regulatory safety updates"
    ]
}
```

#### **Automated Monitoring:**
```python
def setup_monitoring():
    """Set up automated monitoring for drug classification updates."""

    # Monitor FDA RSS feeds for new approvals
    monitor_fda_approvals()

    # Track guideline update notifications
    monitor_guideline_updates(['nccn', 'asco', 'esmo'])

    # Alert on major clinical trial results
    monitor_clinical_trial_results()

    # Generate monthly update reports
    schedule_update_reports()
```

This comprehensive information gathering approach ensures that disease modules are built on solid clinical foundations with multiple validation sources.

## 🎯 Key Design Principles

### **1. Clinical Accuracy**
- Base drug classes on established treatment guidelines (NCCN, ASCO, etc.)
- Include both generic and brand names
- Consider mechanism of action and clinical usage

### **2. Comprehensive Coverage**
- Include all major drug classes for the disease
- Add appropriate keywords and synonyms
- Set realistic confidence thresholds

### **3. Web Source Integration**
- Specify disease-specific databases and guidelines
- Include regulatory (FDA) and clinical (NCCN, ASCO) sources
- Add specialty databases (OncoKB for cancer)

### **4. LLM Context Optimization**
- Provide current treatment landscape
- Include recent approvals and emerging therapies
- Specify biomarker-driven treatment decisions
- Mention combination therapies and sequencing

### **5. Flexible Validation**
- Allow both exact matches and keyword-based matching
- Support fuzzy matching for variations
- Be permissive for unknown medications to enable discovery

## 🔧 Advanced Customization

### **Custom Validation Logic**
```python
def validate_medication(self, medication: str, drug_class: str) -> bool:
    """Custom validation with disease-specific rules."""

    # Check for contraindicated combinations
    if drug_class == "cdk46_inhibitors" and "pregnancy" in medication.lower():
        return False

    # Handle multi-target drugs
    if medication.lower() in ["lapatinib", "tykerb"]:
        return drug_class in ["her2_targeted", "targeted_therapy"]

    # Standard validation
    return super().validate_medication(medication, drug_class)
```

### **Metadata Integration**
```python
@property
def drug_classes(self) -> List[DrugClassConfig]:
    """Drug classes with rich metadata."""
    return [
        DrugClassConfig(
            name="cdk46_inhibitors",
            keywords=["palbociclib", "ribociclib", "abemaciclib"],
            confidence_threshold=0.9,
            web_sources=["fda", "oncokb"],
            metadata={
                "approval_dates": {
                    "palbociclib": "2015-02-03",
                    "ribociclib": "2017-03-13",
                    "abemaciclib": "2017-09-28"
                },
                "biomarkers": ["HR+", "HER2-"],
                "resistance_patterns": ["CDK6 amplification", "RB1 loss"]
            }
        )
    ]
```

### **Advanced Web Source Configuration**
```python
def get_web_sources(self) -> List[str]:
    """Disease-specific web sources with priority ordering."""
    return [
        # Regulatory sources (highest priority)
        "https://www.fda.gov/drugs/resources-information-approved-drugs/oncology-cancer-hematologic-malignancies-approval-notifications",

        # Clinical guidelines
        "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1419",
        "https://www.asco.org/research-guidelines/quality-guidelines/guidelines/breast-cancer",

        # Knowledge bases
        "https://www.oncokb.org/cancer-genes/breast",
        "https://www.my.breastcancercare.org.uk/treatments",

        # Clinical trials
        "https://clinicaltrials.gov/search?cond=Breast%20Cancer",

        # Patient resources
        "https://www.breastcancer.org/treatment",
        "https://www.cancer.gov/types/breast/treatment",
    ]
```

## 🧪 Testing & Validation

### **Unit Tests**
```bash
# Run unit tests for your module
pytest tests/unit/diseases/test_breast_cancer_module.py -v

# Test coverage
pytest tests/unit/diseases/test_breast_cancer_module.py --cov=src/med_aug/diseases/breast_cancer
```

### **Integration Tests**
```bash
# Test with pipeline
med-aug pipeline run test_breast_cancer.csv \
  --disease breast_cancer \
  --output test_results

# Validate against known datasets
med-aug diseases validate breast_cancer \
  --test-file breast_cancer_medications.txt
```

### **Manual Validation**
```bash
# Test specific medications
med-aug diseases test breast_cancer \
  --medications "tamoxifen,herceptin,ibrance,keytruda"

# Compare with existing modules
med-aug diseases compare breast_cancer nsclc \
  --metric coverage
```

## 📊 Common Disease Module Examples

### **Prostate Cancer Module**
```python
class ProstateCancerModule(DiseaseModule):
    @property
    def name(self) -> str:
        return "prostate_cancer"

    @property
    def drug_classes(self) -> List[DrugClassConfig]:
        return [
            DrugClassConfig(
                name="hormone_therapy",
                keywords=[
                    "leuprolide", "lupron", "goserelin", "zoladex",
                    "triptorelin", "trelstar", "degarelix", "firmagon",
                    "abiraterone", "zytiga", "enzalutamide", "xtandi",
                    "apalutamide", "erleada", "darolutamide", "nubeqa"
                ],
                confidence_threshold=0.85,
                web_sources=["fda", "nccn", "asco"],
            ),
            DrugClassConfig(
                name="chemotherapy",
                keywords=[
                    "docetaxel", "taxotere", "cabazitaxel", "jevtana",
                    "mitoxantrone", "novantrone"
                ],
                confidence_threshold=0.8,
                web_sources=["fda", "nccn"],
            ),
            DrugClassConfig(
                name="parp_inhibitors",
                keywords=[
                    "olaparib", "lynparza", "rucaparib", "rubraca",
                    "niraparib", "zejula"
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb"],
            ),
        ]
```

### **Cardiovascular Disease Module**
```python
class CardiovascularModule(DiseaseModule):
    @property
    def name(self) -> str:
        return "cardiovascular"

    @property
    def drug_classes(self) -> List[DrugClassConfig]:
        return [
            DrugClassConfig(
                name="ace_inhibitors",
                keywords=[
                    "lisinopril", "prinivil", "zestril", "enalapril", "vasotec",
                    "ramipril", "altace", "captopril", "capoten"
                ],
                confidence_threshold=0.85,
                web_sources=["fda", "acc", "aha"],
            ),
            DrugClassConfig(
                name="beta_blockers",
                keywords=[
                    "metoprolol", "lopressor", "toprol", "atenolol", "tenormin",
                    "propranolol", "inderal", "carvedilol", "coreg"
                ],
                confidence_threshold=0.85,
                web_sources=["fda", "acc", "aha"],
            ),
            DrugClassConfig(
                name="statins",
                keywords=[
                    "atorvastatin", "lipitor", "simvastatin", "zocor",
                    "rosuvastatin", "crestor", "pravastatin", "pravachol"
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "acc", "aha"],
            ),
        ]
```

## 📈 Success Metrics

A well-built disease module should achieve:

### **Technical Metrics**
- **>95% accuracy** on known medication validation
- **>90% coverage** of standard-of-care treatments
- **<5% false positive rate** on unknown medications
- **Complete integration** with existing pipeline
- **Comprehensive test coverage** (>90%)

### **Clinical Metrics**
- **Current treatment landscape** reflected in drug classes
- **Recent FDA approvals** included in keyword lists
- **Biomarker-driven** treatment decisions supported
- **Combination therapies** properly classified
- **Emerging therapies** accommodation

### **Usability Metrics**
- **Clear documentation** for clinical team review
- **Intuitive drug class names** matching clinical terminology
- **Appropriate confidence thresholds** for each drug class
- **Comprehensive web source coverage** for validation
- **Easy extensibility** for future updates

## 🔄 Maintenance & Updates

### **Regular Updates**
- **Quarterly FDA approval review** for new medications
- **Annual guideline review** (NCCN, ASCO, ESMO updates)
- **Biannual keyword optimization** based on pipeline feedback
- **Continuous clinical validation** with domain experts

### **Version Control**
```python
class BreastCancerModule(DiseaseModule):
    @property
    def version(self) -> str:
        return "1.2.0"  # Semantic versioning

    @property
    def last_updated(self) -> str:
        return "2025-01-15"  # ISO date format

    @property
    def changelog(self) -> List[str]:
        return [
            "v1.2.0: Added sacituzumab govitecan (Trodelvy) for TNBC",
            "v1.1.0: Updated CDK4/6 inhibitor keywords and confidence",
            "v1.0.0: Initial breast cancer module implementation"
        ]
```

### **Automated Validation**
```bash
# Set up automated testing
crontab -e
# Add: 0 0 * * 0 cd /path/to/project && med-aug diseases validate-all --report

# Continuous integration testing
# .github/workflows/disease-modules.yml
name: Disease Module Validation
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly validation
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Validate all disease modules
        run: |
          source .venv/bin/activate
          med-aug diseases validate-all --strict
```

## 🎓 Best Practices Summary

1. **Start with established guidelines** - Use NCCN, ASCO, FDA as primary sources
2. **Include comprehensive keywords** - Generic names, brand names, common abbreviations
3. **Set appropriate confidence thresholds** - Higher for targeted therapies, lower for broad categories
4. **Write comprehensive tests** - Unit tests, integration tests, clinical validation
5. **Document thoroughly** - Clinical context, validation logic, maintenance notes
6. **Plan for updates** - Version control, changelog, automated validation
7. **Collaborate with clinicians** - Regular review and feedback from domain experts

This creates a robust, maintainable, and clinically accurate disease module that enhances the medication augmentation system's capabilities!
