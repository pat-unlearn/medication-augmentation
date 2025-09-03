"""Non-Small Cell Lung Cancer (NSCLC) disease module."""

from typing import List
from ..base import DiseaseModule, DrugClassConfig


class NSCLCModule(DiseaseModule):
    """Non-Small Cell Lung Cancer disease module implementation."""
    
    @property
    def name(self) -> str:
        """Disease identifier."""
        return "nsclc"
    
    @property
    def display_name(self) -> str:
        """Human-readable disease name."""
        return "Non-Small Cell Lung Cancer"
    
    @property
    def drug_classes(self) -> List[DrugClassConfig]:
        """NSCLC-specific drug class configurations."""
        return [
            # Chemotherapy agents
            DrugClassConfig(
                name="chemotherapy",
                keywords=[
                    "carboplatin", "paclitaxel", "pemetrexed", "docetaxel", 
                    "gemcitabine", "cisplatin", "etoposide", "vinorelbine",
                    "irinotecan", "topotecan", "vincristine", "abraxane",
                    "taxol", "paraplatin", "alimta", "taxotere", "gemzar"
                ],
                confidence_threshold=0.8,
                web_sources=["fda", "nccn", "clinicaltrials"]
            ),
            
            # Immunotherapy agents
            DrugClassConfig(
                name="immunotherapy",
                keywords=[
                    "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
                    "ipilimumab", "cemiplimab", "keytruda", "opdivo", "tecentriq",
                    "imfinzi", "yervoy", "libtayo", "opdualag", "medi4736",
                    "tremelimumab", "tislelizumab", "sintilimab", "camrelizumab"
                ],
                confidence_threshold=0.85,
                web_sources=["fda", "nccn", "oncokb"]
            ),
            
            # Targeted therapy agents
            DrugClassConfig(
                name="targeted_therapy",
                keywords=[
                    "osimertinib", "erlotinib", "afatinib", "gefitinib", "dacomitinib",
                    "crizotinib", "alectinib", "brigatinib", "ceritinib", "lorlatinib",
                    "entrectinib", "larotrectinib", "sotorasib", "adagrasib", "amivantamab",
                    "mobocertinib", "capmatinib", "tepotinib", "savolitinib", "pralsetinib",
                    "selpercatinib", "tagrisso", "tarceva", "gilotrif", "iressa", "vizimpro",
                    "xalkori", "alecensa", "alunbrig", "zykadia", "lorbrena", "rozlytrek",
                    "vitrakvi", "lumakras", "krazati", "rybrevant", "exkivity", "tabrecta",
                    "tepmetko", "gavreto", "retevmo"
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb", "clinicaltrials"]
            ),
            
            # Anti-angiogenic agents
            DrugClassConfig(
                name="anti_angiogenic",
                keywords=[
                    "bevacizumab", "ramucirumab", "avastin", "cyramza",
                    "nintedanib", "ofev", "vargatef", "regorafenib",
                    "stivarga", "axitinib", "inlyta", "sunitinib", "sutent"
                ],
                confidence_threshold=0.85,
                web_sources=["fda", "nccn"]
            ),
            
            # Antibody-drug conjugates
            DrugClassConfig(
                name="antibody_drug_conjugates",
                keywords=[
                    "trastuzumab deruxtecan", "enhertu", "sacituzumab govitecan",
                    "trodelvy", "datopotamab deruxtecan", "dato-dxd", "patritumab deruxtecan"
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb", "clinicaltrials"]
            ),
            
            # KRAS inhibitors (new class)
            DrugClassConfig(
                name="kras_inhibitors",
                keywords=[
                    "sotorasib", "adagrasib", "lumakras", "krazati",
                    "mrtx849", "gdc-6036", "bi-1823911", "jdq443"
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb", "clinicaltrials"]
            ),
            
            # EGFR inhibitors (specific subclass)
            DrugClassConfig(
                name="egfr_inhibitors",
                keywords=[
                    "osimertinib", "erlotinib", "afatinib", "gefitinib", "dacomitinib",
                    "mobocertinib", "lazertinib", "tagrisso", "tarceva", "gilotrif",
                    "iressa", "vizimpro", "exkivity", "leclaza", "amivantamab", "rybrevant"
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb"]
            ),
            
            # ALK inhibitors (specific subclass)
            DrugClassConfig(
                name="alk_inhibitors",
                keywords=[
                    "crizotinib", "alectinib", "brigatinib", "ceritinib", "lorlatinib",
                    "ensartinib", "xalkori", "alecensa", "alunbrig", "zykadia", "lorbrena"
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb"]
            ),
            
            # ROS1 inhibitors
            DrugClassConfig(
                name="ros1_inhibitors",
                keywords=[
                    "crizotinib", "entrectinib", "ceritinib", "lorlatinib",
                    "repotrectinib", "taletrectinib", "xalkori", "rozlytrek",
                    "zykadia", "lorbrena"
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb"]
            ),
            
            # MET inhibitors
            DrugClassConfig(
                name="met_inhibitors",
                keywords=[
                    "capmatinib", "tepotinib", "savolitinib", "crizotinib",
                    "tabrecta", "tepmetko", "xalkori", "elzovantinib",
                    "bozitinib", "amg-337"
                ],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb", "clinicaltrials"]
            )
        ]
    
    def get_web_sources(self) -> List[str]:
        """NSCLC-specific data sources for web scraping."""
        return [
            "https://www.fda.gov/drugs/resources-information-approved-drugs/oncology-cancer-hematologic-malignancies-approval-notifications",
            "https://clinicaltrials.gov/search?cond=Non-Small%20Cell%20Lung%20Cancer",
            "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1450",
            "https://www.oncokb.org/cancer-genes/lung",
            "https://www.asco.org/research-guidelines/quality-guidelines/guidelines/thoracic-cancer#/9236",
            "https://www.esmo.org/guidelines/guidelines-by-topic/lung-and-chest-tumours/clinical-practice-living-guidelines-metastatic-non-small-cell-lung-cancer",
            "https://www.lungcancerresearchfoundation.org/research/treatments-clinical-trials/",
        ]
    
    def get_llm_context(self) -> str:
        """NSCLC-specific context for LLM classification."""
        return """You are a clinical oncologist specializing in Non-Small Cell Lung Cancer (NSCLC) treatment.
Your expertise includes the latest FDA-approved therapies and clinical trial agents for NSCLC.

Current NSCLC treatment landscape (2024-2025):

FIRST-LINE TREATMENTS:
- Immunotherapy: Pembrolizumab (Keytruda) +/- chemotherapy for PD-L1 positive
- Targeted therapy by mutation:
  * EGFR+: Osimertinib (Tagrisso), Erlotinib (Tarceva), Afatinib (Gilotrif)
  * ALK+: Alectinib (Alecensa), Brigatinib (Alunbrig), Lorlatinib (Lorbrena)
  * ROS1+: Crizotinib (Xalkori), Entrectinib (Rozlytrek), Ceritinib (Zykadia)
  * BRAF V600E: Dabrafenib + Trametinib
  * MET ex14: Capmatinib (Tabrecta), Tepotinib (Tepmetko)
  * RET+: Selpercatinib (Retevmo), Pralsetinib (Gavreto)
  * KRAS G12C: Sotorasib (Lumakras), Adagrasib (Krazati)
  * NTRK: Larotrectinib (Vitrakvi), Entrectinib (Rozlytrek)

IMMUNOTHERAPY COMBINATIONS:
- Nivolumab (Opdivo) + Ipilimumab (Yervoy)
- Durvalumab (Imfinzi) + Tremelimumab
- Atezolizumab (Tecentriq) + Bevacizumab (Avastin) + Chemotherapy

RECENT APPROVALS (2023-2025):
- Amivantamab (Rybrevant) for EGFR exon 20 insertions
- Mobocertinib (Exkivity) for EGFR exon 20 insertions
- Adagrasib (Krazati) for KRAS G12C mutations
- Trastuzumab deruxtecan (Enhertu) for HER2-mutant NSCLC
- Lazertinib for EGFR mutations
- Datopotamab deruxtecan (Dato-DXd) in clinical trials

CHEMOTHERAPY BACKBONE:
- Platinum-based: Carboplatin, Cisplatin
- Taxanes: Paclitaxel, Docetaxel, Nab-paclitaxel (Abraxane)
- Others: Pemetrexed (Alimta), Gemcitabine (Gemzar), Etoposide

Classify medications into these categories:
- Chemotherapy: Traditional cytotoxic agents
- Immunotherapy: Checkpoint inhibitors, immune modulators
- Targeted Therapy: Small molecule inhibitors targeting specific mutations
- EGFR Inhibitors: Specific for EGFR mutations
- ALK Inhibitors: Specific for ALK rearrangements
- KRAS Inhibitors: Specific for KRAS mutations (especially G12C)
- Anti-angiogenic: VEGF pathway inhibitors
- Antibody-Drug Conjugates: Targeted delivery systems
- ROS1 Inhibitors: For ROS1 rearrangements
- MET Inhibitors: For MET alterations

Consider generic names, brand names, trial designations, and common abbreviations.
Many drugs may belong to multiple categories (e.g., Crizotinib is both ALK and ROS1 inhibitor)."""
    
    def validate_medication(self, medication: str, drug_class: str) -> bool:
        """
        NSCLC-specific medication validation.
        
        Args:
            medication: Medication name to validate
            drug_class: Drug class to validate against
            
        Returns:
            True if medication is valid for the drug class
        """
        medication_lower = medication.lower().strip()
        
        # Known NSCLC medications by class (comprehensive list)
        known_nsclc_meds = {
            "chemotherapy": [
                "carboplatin", "paclitaxel", "pemetrexed", "docetaxel", "gemcitabine",
                "cisplatin", "etoposide", "vinorelbine", "irinotecan", "topotecan",
                "abraxane", "taxol", "paraplatin", "alimta", "taxotere", "gemzar"
            ],
            "immunotherapy": [
                "pembrolizumab", "keytruda", "nivolumab", "opdivo", "atezolizumab",
                "tecentriq", "durvalumab", "imfinzi", "ipilimumab", "yervoy",
                "cemiplimab", "libtayo", "tremelimumab", "tislelizumab", "sintilimab"
            ],
            "targeted_therapy": [
                "osimertinib", "tagrisso", "erlotinib", "tarceva", "afatinib", "gilotrif",
                "gefitinib", "iressa", "dacomitinib", "vizimpro", "crizotinib", "xalkori",
                "alectinib", "alecensa", "brigatinib", "alunbrig", "ceritinib", "zykadia",
                "lorlatinib", "lorbrena", "entrectinib", "rozlytrek", "larotrectinib",
                "vitrakvi", "sotorasib", "lumakras", "adagrasib", "krazati",
                "amivantamab", "rybrevant", "mobocertinib", "exkivity",
                "capmatinib", "tabrecta", "tepotinib", "tepmetko",
                "selpercatinib", "retevmo", "pralsetinib", "gavreto"
            ],
            "egfr_inhibitors": [
                "osimertinib", "tagrisso", "erlotinib", "tarceva", "afatinib", "gilotrif",
                "gefitinib", "iressa", "dacomitinib", "vizimpro", "mobocertinib", "exkivity",
                "amivantamab", "rybrevant", "lazertinib", "leclaza"
            ],
            "alk_inhibitors": [
                "crizotinib", "xalkori", "alectinib", "alecensa", "brigatinib", "alunbrig",
                "ceritinib", "zykadia", "lorlatinib", "lorbrena", "ensartinib"
            ],
            "kras_inhibitors": [
                "sotorasib", "lumakras", "adagrasib", "krazati", "mrtx849", "gdc-6036"
            ],
            "anti_angiogenic": [
                "bevacizumab", "avastin", "ramucirumab", "cyramza", "nintedanib", "ofev"
            ],
            "antibody_drug_conjugates": [
                "trastuzumab deruxtecan", "enhertu", "sacituzumab govitecan", "trodelvy",
                "datopotamab deruxtecan", "dato-dxd", "patritumab deruxtecan"
            ]
        }
        
        # Check if medication is known for this drug class
        if drug_class in known_nsclc_meds:
            for known_med in known_nsclc_meds[drug_class]:
                if known_med in medication_lower or medication_lower in known_med:
                    return True
        
        # Check against keywords in drug class config
        drug_class_config = self.get_drug_class_by_name(drug_class)
        if drug_class_config:
            for keyword in drug_class_config.keywords:
                if keyword.lower() in medication_lower or medication_lower in keyword.lower():
                    return True
        
        # For unknown medications in unknown classes, be permissive to allow discovery
        # For known classes with unknown medications, return False to maintain accuracy
        if drug_class in known_nsclc_meds:
            return False
        
        # For completely unknown drug classes, be permissive
        return True
    
    def get_medication_notes(self, medication: str) -> str:
        """Get NSCLC-specific notes about a medication."""
        medication_lower = medication.lower()
        
        notes = []
        
        # Multi-target drugs
        if medication_lower in ["crizotinib", "xalkori"]:
            notes.append("Targets both ALK and ROS1 rearrangements")
        elif medication_lower in ["ceritinib", "zykadia", "lorlatinib", "lorbrena"]:
            notes.append("Active against both ALK and ROS1")
        elif medication_lower in ["entrectinib", "rozlytrek"]:
            notes.append("Targets ROS1, NTRK, and ALK")
        
        # Resistance patterns
        if medication_lower in ["osimertinib", "tagrisso"]:
            notes.append("3rd generation EGFR-TKI, active against T790M resistance")
        elif medication_lower in ["lorlatinib", "lorbrena"]:
            notes.append("3rd generation ALK inhibitor, crosses blood-brain barrier")
        elif medication_lower in ["alectinib", "alecensa"]:
            notes.append("Preferred 1st-line ALK inhibitor, excellent CNS penetration")
        
        # Special populations
        if medication_lower in ["pemetrexed", "alimta"]:
            notes.append("Preferred for non-squamous histology")
        elif medication_lower in ["gemcitabine", "gemzar"]:
            notes.append("Preferred for squamous histology")
        
        # Combination therapies
        if medication_lower in ["pembrolizumab", "keytruda"]:
            notes.append("Can be used as monotherapy (PD-L1 ≥50%) or with chemotherapy")
        elif medication_lower in ["bevacizumab", "avastin"]:
            notes.append("Usually combined with chemotherapy and/or immunotherapy")
        
        return "; ".join(notes) if notes else ""


# Register the module for auto-discovery
MODULE_CLASS = NSCLCModule