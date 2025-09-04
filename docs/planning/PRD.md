# Medication Augmentation PRD - NSCLC Pipeline

## Overview

**Product Name:** NSCLC Medication Augmentation System
**Version:** 1.0
**Date:** September 3, 2025
**Owner:** Pat Sheehan + Data Science team
**Focus:** Non-Small Cell Lung Cancer (NSCLC) Pipeline Only

## Executive Summary

The Medication Augmentation System (Phase 1: NSCLC) aims to comprehensively expand the `conmeds_defaults.yml` configuration file for Non-Small Cell Lung Cancer (NSCLC) to capture all generic and brand names for each drug class, improving medication matching accuracy in the NSCLC clinical data processing pipeline.

## Problem Statement

Currently, the NSCLC medication enrichment process relies on a manually curated `conmeds_defaults.yml` file that contains limited examples of medication names for each drug class. The NSCLC pipeline specifically has:

- **Low medication matching rates**: The current NSCLC conmeds_defaults.yml contains only 54 drug classes with 2-10 medication names each, missing many variations found in clinical datasets
- **Manual bottlenecks**: Data scientists spend significant time manually reviewing and augmenting the NSCLC medication list when processing lung cancer trial data
- **Incomplete cancer drug coverage**: Despite NSCLC being a cancer indication, many newer immunotherapies, targeted therapies, and combination treatments are missing
- **High false negative rates**: LLM-based classification without proper context leads to missed NSCLC-specific medications, especially for newer drugs and clinical trial nomenclature

## Solution Overview

An automated system that intelligently expands the NSCLC `conmeds_defaults.yml` file by:

1. **Analyzing NSCLC clinical trial data** to identify medication name columns and extract unique drug names
2. **Web scraping oncology databases** to gather comprehensive NSCLC drug name databases including FDA approvals, clinical trials, and cancer drug compendiums
3. **LLM-powered classification** with NSCLC-specific context to accurately categorize cancer medications
4. **Oncology expert validation** workflows to ensure accuracy for cancer treatments

## Goals and Success Metrics

### Primary Goals
- **Automate NSCLC medication list augmentation** for the conmeds_defaults.yml file reducing manual effort by 80%
- **Improve NSCLC medication matching accuracy** by expanding drug name coverage from current 2-10 names per class to 20-50+ names per class
- **Create a reproducible process** that can later be extended to other disease indications

### Success Metrics
- **Medication matching improvement**: Increase percentage of matched medications by ≥30% in NSCLC datasets
- **Coverage expansion**: Achieve ≥95% coverage of NSCLC medications found in the MSK CHORD 2024 dataset
- **Processing time reduction**: Reduce manual review time from days to hours for NSCLC pipeline updates
- **False negative reduction**: Decrease LLM classification false negatives by ≥50% for cancer drugs
- **Drug class completeness**: Expand from 54 to 70+ drug classes relevant to NSCLC treatment

### Evaluation Criteria
1. **Data source identification accuracy**: Which medication column was correctly identified
2. **Name expansion count**: Number of generic/brand names added beyond initial examples
3. **Before/after matching rates**: Quantified improvement in medication capture

## Current Workflow

### Manual Process
1. **Clinical scientists** define NSCLC-relevant drug classes for lung cancer treatment
2. **NSCLC data specifications** include cancer-specific `taking_{drug-class}` variables:
   - Chemotherapy: `taking_paclitaxel`, `taking_carboplatin`, `taking_pemetrexed`, `taking_docetaxel`
   - Immunotherapy: `taking_pembrolizumab`, `taking_nivolumab`, `taking_atezolizumab`, `taking_durvalumab`
   - Targeted therapy: `taking_osimertinib`, `taking_erlotinib`, `taking_crizotinib`, `taking_alectinib`
3. **NSCLC conmeds_defaults.yml** file at `packages/plombier/src/plombier/pipelines/nsclc/`:
   - **Current state**: 54 drug classes with 2-10 medication names each
   - **Focus**: Primarily cancer drugs but missing many newer agents
4. **Data scientists** manually review NSCLC clinical trial data (e.g., MSK CHORD dataset)
5. **Manual research** via oncology resources to identify additional NSCLC drug names
6. **Manual or LLM sorting** of medications into appropriate cancer drug classes

### Technical Implementation
- **Plombier pipeline** processes medications using regex matching (meds_enrich.py:158)
- **Pattern matching** uses word boundaries: `\b({core_pattern})\b`
- **Boolean indicators** generated for each `taking_{drug-class}` variable

## Proposed Solution

### Architecture Overview

```
Raw Data → Column Detection → Name Extraction → Web Research → LLM Classification → Validation → Updated conmeds.yml
```

### Core Components

#### 1. Data Source Analyzer
**Purpose:** Automatically identify medication name columns in raw datasets

**Features:**
- Multi-column detection algorithm
- Confidence scoring for column selection
- Support for various data formats (CSV, TSV, etc.)

**Input:** Raw clinical datasets
**Output:** Identified medication column(s) with confidence scores

#### 2. Medication Name Extractor
**Purpose:** Extract unique medication names from identified columns

**Features:**
- Data cleaning and normalization
- Duplicate removal and standardization
- Statistical analysis of name frequency

**Input:** Raw medication column data
**Output:** Cleaned, unique medication name list

#### 3. Web Research Engine
**Purpose:** Gather comprehensive medication information via web scraping

**Features:**
- Multiple data source integration (FDA databases, drug reference sites)
- Generic-brand name mapping
- Drug class classification context
- Rate limiting and respectful scraping

**Input:** Medication names from raw data
**Output:** Enriched medication database with context

#### 4. LLM Classification System
**Purpose:** Intelligently classify medications into drug classes

**Features:**
- Enhanced prompting with web-scraped context
- Confidence scoring for classifications
- Batch processing capabilities
- Human-in-the-loop validation

**Input:** Medication names + web context + existing drug class definitions
**Output:** Classified medications with confidence scores

#### 5. Validation & Quality Assurance
**Purpose:** Ensure accuracy and completeness of classifications

**Features:**
- Pharmacist review workflow
- Automated consistency checks
- Version control for conmeds.yml updates
- Rollback capabilities

### Data Assets

#### Required Inputs
1. **NSCLC Data Specifications**: Subset of `specs.csv` containing NSCLC-relevant medication variables
   - **Focus medications**: Lines 178-214 covering NSCLC-specific treatments
   - **Drug categories**: Chemotherapy (paclitaxel, carboplatin, pemetrexed), Immunotherapy (pembrolizumab, nivolumab), Targeted therapy (osimertinib, crizotinib, alectinib)
2. **Current NSCLC conmeds_defaults.yml**: Located at `packages/plombier/src/plombier/pipelines/nsclc/conmeds_defaults.yml`:
   - **Current coverage**: 54 drug classes with 2-10 medication names each
   - **Example entries**:
     - `taking_pembrolizumab: [pembrolizumab, Keytruda]`
     - `taking_osimertinib: [osimertinib, Tagrisso]`
     - `taking_paclitaxel: [paclitaxel, Abraxane, Taxol, Paclitaxel Loaded Polymeric Micelle, Paclitaxel Poliglumex, Nab Paclitaxel]`
3. **NSCLC Raw Data Example**:
   - MSK CHORD 2024: `s3://unlearnai-prod-data-acquisition-team/20250728_msk_chord_2024/data_timeline_treatment.txt`
   - Medication column: AGENT
   - Contains real-world NSCLC treatment data from Memorial Sloan Kettering

#### Generated Outputs
1. **Enhanced NSCLC conmeds_defaults.yml** with expanded medication lists (target: 200+ additional drug names)
2. **NSCLC classification report** showing confidence scores and additions per drug class
3. **Validation log** for NSCLC medications with oncology expert review notes
4. **Coverage analysis** comparing before/after augmentation for NSCLC pipeline

## Technical Requirements

### System Requirements
- **Python environment** with pandas, polars for data processing
- **Web scraping capabilities** with rate limiting and error handling
- **LLM integration** (OpenAI API or similar)
- **Version control** for configuration file management
- **Logging and monitoring** for process tracking

### Integration Points
- **Existing Plombier pipeline** (meds_enrich.py)
- **Google Sheets API** for data specifications
- **Cloud storage** (S3) for raw data access
- **GitHub** for version control of conmeds.yml files

### Performance Requirements
- **Processing time**: Complete augmentation within 2 hours for typical datasets
- **Accuracy**: ≥95% precision for medication classification
- **Reliability**: 99% uptime for critical processing periods
- **Scalability**: Handle datasets with 10,000+ unique medication names

## Implementation Plan

### Phase 1: NSCLC Data Analysis
- Analyze MSK CHORD 2024 NSCLC dataset
- Extract unique medication names from AGENT column
- Identify gaps between raw data and current conmeds_defaults.yml
- Create baseline metrics for current matching rates

### Phase 2: NSCLC Drug Database Building
- Web scrape NSCLC-specific resources:
  - FDA.gov NSCLC approvals
  - ClinicalTrials.gov NSCLC studies
  - NCCN NSCLC guidelines
  - OncoKB lung cancer drugs
- Build comprehensive NSCLC medication database
- Map generic names to all brand variations

### Phase 3: NSCLC Classification & Augmentation
- Configure LLM with NSCLC/oncology context
- Classify medications into appropriate drug classes
- Generate expanded conmeds_defaults.yml for NSCLC
- Create confidence scores for each classification

### Phase 4: Validation & Deployment
- Oncology expert review of classifications
- Test augmented file with MSK CHORD dataset
- Measure improvement in medication matching
- Deploy enhanced NSCLC conmeds_defaults.yml
- Document process for future disease expansions

## Risk Assessment

### Technical Risks
- **Web scraping limitations**: Rate limits, anti-bot measures
- **LLM accuracy**: False classifications requiring manual review
- **Data quality**: Inconsistent medication naming in raw data

### Mitigation Strategies
- Multiple data source redundancy
- Human validation workflows
- Robust error handling and logging
- Gradual rollout with careful monitoring

## Success Criteria

### Minimum Viable Product (MVP) - NSCLC Focus
- Successfully identify AGENT column in MSK CHORD dataset
- Add 100+ new medication names to NSCLC conmeds_defaults.yml
- Achieve 30% improvement in medication matching for NSCLC data
- Document clear process for oncology medication augmentation

### Full Success - NSCLC Pipeline
- 95% medication matching accuracy for MSK CHORD NSCLC dataset
- Expand from 54 to 70+ drug classes with 20+ names each
- Fully automated NSCLC augmentation requiring <2 hours manual review
- Reproducible process ready for expansion to other cancer indications
- Deployment-ready enhanced NSCLC conmeds_defaults.yml file

## Appendix

### Code References
- **Medication enrichment logic**: `meds_enrich.py:158` - Pattern matching implementation
- **Regex patterns**: Word boundary matching `\b({core_pattern})\b`
- **Data processing**: Polars-based grouping and aggregation

### Data Examples
- **Single column example**: MSK CHORD 2024 dataset with AGENT column
- **Multi-column example**: Dataset with DRUGDTXT (correct) vs ACTTRTXT (treatment assignment - incorrect)

### Current NSCLC Drug Class Examples

#### From Current NSCLC conmeds_defaults.yml (54 drug classes)

**Chemotherapy Agents**
```yaml
taking_paclitaxel: [paclitaxel, Abraxane, Taxol, Paclitaxel Loaded Polymeric Micelle, Paclitaxel Poliglumex, Nab Paclitaxel]
taking_carboplatin: [carboplatin, Paraplatin]
taking_pemetrexed: [pemetrexed, pemetrexed disodium, Alimta, Ciambra, Pemfexy, Pemrydi Rtu]
taking_docetaxel: [docetaxel, taxotere]
taking_gemcitabine: [gemcitabine, Gemcitabine Hydrochloride, Gemzar]
```

**Immunotherapy Agents**
```yaml
taking_pembrolizumab: [pembrolizumab, Keytruda]
taking_nivolumab: [nivolumab, Opdivo, Opdualag]
taking_atezolizumab: [atezolizumab, Tecentriq]
taking_durvalumab: [durvalumab, Imfinzi]
taking_ipilimumab: [ipilimumab, Yervoy]
```

**Targeted Therapy Agents**
```yaml
taking_osimertinib: [osimertinib, Tagrisso]
taking_erlotinib: [erlotinib, Erlotinib Hydrochloride, Tarceva]
taking_crizotinib: [crizotinib, Xalkori]
taking_alectinib: [alectinib, Alecensa]
taking_lorlatinib: [lorlatinib, Lorbrena]
```

### Coverage Analysis

**NSCLC Pipeline Current State**
- **Total drug classes**: 54 in conmeds_defaults.yml
- **NSCLC-specific medications**: ~37 cancer treatment classes
- **Current medication density**: 2-10 names per drug class
- **Total medication names**: ~250 across all classes

**NSCLC Expansion Opportunity**
- **Target drug classes**: 70+ (adding newer immunotherapies, ADCs, combination regimens)
- **Target names per class**: 20-50 (including trial names, international brands, abbreviations)
- **Expected total names**: 1000+ medication variations
- **Key gaps to address**:
  - Newer checkpoint inhibitors (e.g., tislelizumab, sintilimab)
  - Antibody-drug conjugates (e.g., trastuzumab deruxtecan)
  - Combination regimen names (e.g., "carbo/pem", "nivo/ipi")
  - Clinical trial drug codes (e.g., "MEDI4736" for durvalumab)

### Stakeholders
- **Clinical Scientists**: Define NSCLC drug classes and validate cancer medication classifications
- **Data Scientists**: Review and approve NSCLC medication augmentations
- **Software Engineers**: Implement the automation system for NSCLC pipeline
- **Oncology Experts**: Validate NSCLC-specific drug classifications
- **Quality Assurance**: Ensure accuracy of cancer medication mappings

## Next Steps

### Immediate Actions
1. **Access verification**: Confirm access to MSK CHORD 2024 dataset
2. **Repository setup**: Clone plombier repository and locate NSCLC conmeds_defaults.yml
3. **Resource gathering**: Compile list of NSCLC drug information sources
4. **Team alignment**: Review PRD with stakeholders and gather feedback

### Future Expansion
After successful NSCLC implementation, the process can be extended to:
- Other cancer indications (PCa, breast, colorectal)
- Cardiovascular pipelines (CAD, BP)
- Metabolic diseases (T2D, obesity)
- Neurological conditions (AD, PD)


## Conclusion

This focused approach on NSCLC provides a manageable scope for developing and validating the medication augmentation system. The NSCLC pipeline serves as an ideal proof-of-concept due to its well-defined drug classes, availability of quality data (MSK CHORD), and clear clinical importance. Success with NSCLC will establish a reproducible framework for expanding to all 23 disease pipelines in the plombier ecosystem.
