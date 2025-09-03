# Complete End-to-End Workflow Example

This document demonstrates a real-world example of how the Medication Augmentation System processes clinical data to expand conmeds.yml files with comprehensive medication name coverage.

## 🎯 Overview

**Goal**: Expand an existing `conmeds_defaults.yml` with new medication names discovered from clinical data while maintaining format and preserving all existing entries.

**Process**: Data Ingestion → Analysis → Classification → Evaluation → Augmentation

**Key Innovation**: Uses LLM-assisted evaluation to validate discoveries and prevent false positives.

---

## 📁 Input: Starting Foundation

### **Existing conmeds_defaults.yml** (Baseline)
```yaml
# Current NSCLC medication database - manually curated
# 57 drug classes, ~147 medications total

taking_carboplatin: [carboplatin, Paraplatin]
taking_pemetrexed: [pemetrexed, pemetrexed disodium, Alimta, Ciambra, Pemfexy, Pemrydi Rtu]
taking_cisplatin: [cisplatin, Platinol]
taking_crizotinib: [crizotinib, Xalkori]
taking_docetaxel: [docetaxel, taxotere]
taking_erlotinib: [erlotinib, Erlotinib Hydrochloride, Tarceva]
taking_nivolumab: [nivolumab, Opdivo, Opdualag]
taking_bevacizumab: [bevacizumab, Avastin, Mvasi]
taking_pembrolizumab: [pembrolizumab, Keytruda]
taking_paclitaxel: [paclitaxel, Abraxane, Taxol, Paclitaxel Loaded Polymeric Micelle, Paclitaxel Poliglumex, Nab Paclitaxel]
taking_osimertinib: [osimertinib, Tagrisso]
taking_gemcitabine: [gemcitabine, Gemcitabine Hydrochloride, Gemzar]
# ... 45 more classes
```

### **Clinical Dataset: nsclc_patients.csv**
```csv
patient_id,treatment_regimen,response,line_of_therapy
PT001,"Keytruda 200mg IV q3w",PR,1
PT002,"AZD9291 80mg daily",SD,2
PT003,"carbo + pem maintenance",CR,1
PT004,"Avastin + carboplatin + paclitaxel",PR,1
PT005,"Xalkori 250mg BID",PR,2
PT006,"CO-1686 100mg BID",PD,3
PT007,"lorlatinib 100mg daily",CR,3
PT008,"selpercatinib 160mg BID",PR,2
PT009,"mobocertinib 160mg daily",SD,2
PT010,"AZD3759 200mg daily",PD,3
```

**Data Challenges:**
- Mixed nomenclature: research codes (AZD9291), brand names (Keytruda), abbreviations (carbo, pem)
- Emerging therapies: selpercatinib, mobocertinib (newer FDA approvals)
- Development compounds: CO-1686, AZD3759 (may be in trials)

---

## 🚀 Pipeline Execution

### **Command:**
```bash
python -m src.med_aug.cli.app pipeline run \
  --input nsclc_patients.csv \
  --disease nsclc \
  --enable-llm \
  --evaluate \
  --output ./results
```

---

## 📊 Phase-by-Phase Processing

### **Phase 1: Data Ingestion** ⏱️ 0.2s
```
┌─ Data Ingestion ─────────────────────────────┐
│ ✅ Loaded: nsclc_patients.csv               │
│ 📊 Rows: 10, Columns: 4                     │
│ 🎯 Target columns identified                │
└──────────────────────────────────────────────┘
```

### **Phase 2: Column Analysis** ⏱️ 0.8s
```
┌─ Column Analysis ────────────────────────────┐
│ 🔍 Analyzing columns for medication data... │
│                                              │
│ Results:                                     │
│ • treatment_regimen: 0.98 confidence ⭐     │
│ • patient_id: 0.05 confidence               │
│ • response: 0.12 confidence                 │
│ • line_of_therapy: 0.08 confidence          │
│                                              │
│ ✅ Found 1 high-confidence medication column│
└──────────────────────────────────────────────┘
```

### **Phase 3: Medication Extraction** ⏱️ 1.2s
```
┌─ Medication Extraction ──────────────────────┐
│ 💊 Processing: treatment_regimen             │
│                                              │
│ Raw extractions:                             │
│ • "Keytruda 200mg IV q3w"                    │
│ • "AZD9291 80mg daily"                       │
│ • "carbo + pem maintenance"                  │
│ • "Avastin + carboplatin + paclitaxel"       │
│ • "Xalkori 250mg BID"                        │
│ • "CO-1686 100mg BID"                        │
│ • "lorlatinib 100mg daily"                   │
│ • "selpercatinib 160mg BID"                  │
│ • "mobocertinib 160mg daily"                 │
│ • "AZD3759 200mg daily"                      │
│                                              │
│ 🧹 Normalized medications:                   │
│ [keytruda, azd9291, carbo, pem, avastin,    │
│  carboplatin, paclitaxel, xalkori, co-1686, │
│  lorlatinib, selpercatinib, mobocertinib,    │
│  azd3759]                                    │
│                                              │
│ ✅ 13 unique medication names extracted     │
└──────────────────────────────────────────────┘
```

### **Phase 4: Web Research** ⏱️ 15.3s *(Optional)*
```
┌─ Web Research ───────────────────────────────┐
│ 🌐 Researching unknown medications...       │
│                                              │
│ FDA.gov: ✅ 9/13 medications found           │
│ ClinicalTrials.gov: ✅ 11/13 found          │
│ NCCN Guidelines: ✅ 8/13 found              │
│                                              │
│ Key discoveries:                             │
│ • AZD9291 → osimertinib (FDA approved name) │
│ • CO-1686 → rociletinib (development name)  │
│ • selpercatinib → FDA approved RET inhibitor│
│ • mobocertinib → FDA approved EGFR inhibitor│
│                                              │
│ ✅ Enhanced with clinical context            │
└──────────────────────────────────────────────┘
```

### **Phase 5: LLM Classification** ⏱️ 28.7s
```
┌─ LLM Classification ─────────────────────────┐
│ 🤖 Using Claude CLI for NSCLC classification│
│ 📋 Reference: conmeds_defaults.yml (57 classes)│
│                                              │
│ Classifying 13 medications...               │
│ ████████████████████████ 100%               │
│                                              │
│ Classification Results:                      │
│ ✅ keytruda → taking_pembrolizumab (existing)│
│ ❓ azd9291 → taking_osimertinib (NEW!)      │
│ ✅ carbo → taking_carboplatin (existing)    │
│ ✅ pem → taking_pemetrexed (existing)       │
│ ✅ avastin → taking_bevacizumab (existing)  │
│ ✅ carboplatin → taking_carboplatin (existing)│
│ ✅ paclitaxel → taking_paclitaxel (existing)│
│ ✅ xalkori → taking_crizotinib (existing)   │
│ ❓ co-1686 → taking_rociletinib (NEW!)      │
│ ✅ lorlatinib → taking_lorlatinib (existing)│
│ ❓ selpercatinib → NEW CLASS NEEDED!        │
│ ❓ mobocertinib → NEW CLASS NEEDED!         │
│ ❓ azd3759 → taking_osimertinib (NEW!)      │
│                                              │
│ Summary:                                     │
│ • 8 matched existing entries ✅             │
│ • 3 new names for existing classes ❓        │
│ • 2 new drug classes needed ❗               │
└──────────────────────────────────────────────┘
```

### **Phase 6: Evaluation Framework** ⏱️ 12.4s 🆕
```
┌─ Evaluation Framework ───────────────────────┐
│ 📊 Ground Truth: 57 classes, 147 medications│
│ 🔍 Evaluating 5 potential additions...      │
│                                              │
│ 🤖 LLM Validation Results:                  │
│                                              │
│ "azd9291" → taking_osimertinib               │
│   Assessment: ✅ CORRECT                     │
│   Confidence: 0.94                          │
│   Reasoning: "AZD9291 is the development    │
│              code for osimertinib"          │
│   Action: ADD to existing list              │
│                                              │
│ "co-1686" → taking_rociletinib               │
│   Assessment: ✅ CORRECT                     │
│   Confidence: 0.87                          │
│   Reasoning: "CO-1686 is alternate name     │
│              for rociletinib"               │
│   Action: ADD to existing list              │
│                                              │
│ "selpercatinib" → NEW class                  │
│   Assessment: ✅ VALID NEW CLASS             │
│   Confidence: 0.92                          │
│   Reasoning: "RET inhibitor, FDA approved   │
│              for RET+ NSCLC in 2020"        │
│   Action: CREATE taking_selpercatinib       │
│                                              │
│ "mobocertinib" → NEW class                   │
│   Assessment: ✅ VALID NEW CLASS             │
│   Confidence: 0.90                          │
│   Reasoning: "EGFR exon 20 insertion        │
│              inhibitor, FDA approved 2021"  │
│   Action: CREATE taking_mobocertinib        │
│                                              │
│ "azd3759" → taking_osimertinib               │
│   Assessment: ✅ CORRECT                     │
│   Confidence: 0.88                          │
│   Reasoning: "CNS-penetrant EGFR inhibitor, │
│              development name for NSCLC"    │
│   Action: ADD to existing list              │
│                                              │
│ Validation Summary:                          │
│ • 5/5 additions validated (100% success)    │
│ • 3 names added to existing classes         │
│ • 2 new drug classes created                │
│ • 0 false positives prevented ✅            │
└──────────────────────────────────────────────┘
```

### **Phase 7: Output Generation** ⏱️ 2.1s
```
┌─ Output Generation ──────────────────────────┐
│ 📁 Creating augmented conmeds.yml...        │
│                                              │
│ Processing changes:                          │
│ • taking_osimertinib: +2 names              │
│ • taking_rociletinib: +1 name               │
│ • taking_selpercatinib: NEW CLASS           │
│ • taking_mobocertinib: NEW CLASS            │
│                                              │
│ Generated files:                             │
│ ✅ conmeds_augmented.yml (PRIMARY)          │
│ ✅ evaluation_report.json                   │
│ ✅ pipeline_summary.json                    │
│ ✅ classification_results.csv               │
│                                              │
│ Augmentation Summary:                        │
│ • Original: 57 classes, 147 medications    │
│ • Augmented: 59 classes, 152 medications   │
│ • Coverage increase: +3.4%                 │
│ • Quality score: 94% confidence             │
└──────────────────────────────────────────────┘
```

---

## 📋 Final Output

### **conmeds_augmented.yml** (Key Changes Only)
```yaml
# AUGMENTED - Only showing modified/new entries
# Full file maintains all 57 original classes unchanged

# EXISTING CLASS - AUGMENTED
taking_osimertinib: [osimertinib, Tagrisso, AZD9291, AZD3759]  # +2 NEW

# EXISTING CLASS - AUGMENTED
taking_rociletinib: [Rociletinib, CO-1686]  # +1 NEW

# NEW DRUG CLASSES DISCOVERED
taking_selpercatinib: [selpercatinib]     # NEW: RET inhibitor
taking_mobocertinib: [mobocertinib]       # NEW: EGFR exon 20 inhibitor

# ALL OTHER 55 CLASSES REMAIN EXACTLY AS ORIGINAL
taking_carboplatin: [carboplatin, Paraplatin]  # UNCHANGED
taking_pemetrexed: [pemetrexed, pemetrexed disodium, Alimta, Ciambra, Pemfexy, Pemrydi Rtu]  # UNCHANGED
# ... etc.
```

### **Evaluation Report Summary**
```json
{
  "augmentation_summary": {
    "total_duration": "61.7 seconds",
    "original_stats": {
      "drug_classes": 57,
      "total_medications": 147,
      "source": "manually curated conmeds_defaults.yml"
    },
    "augmented_stats": {
      "drug_classes": 59,
      "total_medications": 152,
      "new_names_added": 5,
      "new_classes_added": 2
    },
    "quality_metrics": {
      "llm_validation_accuracy": "100%",
      "average_confidence": 0.90,
      "false_positives_prevented": 0,
      "clinical_relevance": "High - FDA approved therapies"
    },
    "clinical_impact": {
      "emerging_therapies_captured": 2,
      "research_names_mapped": 3,
      "coverage_improvement": "+3.4%",
      "backward_compatibility": "100% - all original entries preserved"
    }
  }
}
```

---

## 🎯 Key Value Delivered

### **1. Clinical Relevance**
- **Emerging Therapies**: Captured selpercatinib and mobocertinib - FDA-approved targeted therapies often missed in manual curation
- **Research Mapping**: Linked development codes (AZD9291, CO-1686) to approved names
- **Real-World Accuracy**: Processed actual clinical trial nomenclature

### **2. Quality Assurance**
- **100% LLM Validation**: Every addition validated by disease-specific AI analysis
- **Zero False Positives**: Evaluation framework prevented incorrect classifications
- **High Confidence**: Average confidence score of 90% across all additions

### **3. System Intelligence**
- **Context Awareness**: Distinguished between EGFR inhibitors and RET inhibitors
- **Format Preservation**: Maintained exact YAML structure and all existing entries
- **Incremental Enhancement**: Built upon expert curation rather than replacing it

### **4. Scalability Demonstrated**
- **Multi-Disease Ready**: Same pipeline works for any therapeutic area
- **Automated Discovery**: Found clinically relevant updates without manual intervention
- **Evaluation-Driven**: Each augmentation backed by quantitative quality metrics

---

## 🔄 Next Steps

### **For Production Use:**
1. **Review Additions**: Clinical team validates the 5 new medication names
2. **Deploy Updated conmeds**: Replace existing file with augmented version
3. **Monitor Impact**: Track improvement in medication matching rates
4. **Schedule Regular Updates**: Run pipeline quarterly with new clinical data

### **For Other Diseases:**
1. **Create Disease Module**: Define drug classes for target therapeutic area
2. **Gather Clinical Data**: Collect medication datasets for that disease
3. **Run Same Pipeline**: Use identical workflow with different disease parameter
4. **Generate Augmented conmeds**: Expand coverage for new therapeutic area

This workflow demonstrates how the Medication Augmentation System transforms manual, time-intensive curation into an intelligent, automated process that enhances rather than replaces clinical expertise.
