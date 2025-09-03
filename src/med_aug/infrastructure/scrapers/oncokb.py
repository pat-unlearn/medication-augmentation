"""OncoKB database integration for precision oncology information."""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from .base import BaseScraper, ScraperConfig, ScraperResult

logger = structlog.get_logger()


class OncoKBScraper(BaseScraper):
    """Scraper for OncoKB precision oncology knowledge base."""

    def __init__(self, client=None, api_key: Optional[str] = None):
        """
        Initialize OncoKB scraper.

        Args:
            client: HTTP client
            api_key: OncoKB API key (required for full access)
        """
        config = ScraperConfig(
            base_url="https://www.oncokb.org/api/v1",
            rate_limit=0.5,  # OncoKB rate limits
            timeout=30,
            user_agent="MedicationAugmentation/1.0 (Educational/Research)",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else "",
            },
        )
        super().__init__(config, client)
        self.api_key = api_key

    async def scrape_medication_info(self, medication_name: str) -> ScraperResult:
        """
        Get OncoKB information for a medication.

        Args:
            medication_name: Name of the medication

        Returns:
            Scraper result with OncoKB data
        """
        try:
            # OncoKB focuses on targeted therapy and biomarkers
            # Mock data based on OncoKB's typical structure

            data = {
                "drug": medication_name,
                "oncokb_levels": self._get_oncokb_levels(medication_name),
                "alterations": self._get_drug_alterations(medication_name),
                "indications": self._get_fda_indications(medication_name),
                "resistance_mechanisms": self._get_resistance_mechanisms(
                    medication_name
                ),
                "clinical_evidence": self._get_clinical_evidence(medication_name),
            }

            return ScraperResult(
                success=True,
                data=data,
                url=f"{self.config.base_url}/drugs/{medication_name}",
            )

        except Exception as e:
            logger.error(
                "oncokb_scrape_failed", medication=medication_name, error=str(e)
            )
            return ScraperResult(
                success=False, data={}, url=self.config.base_url, error=str(e)
            )

    def _get_oncokb_levels(self, medication_name: str) -> List[Dict[str, Any]]:
        """Get OncoKB evidence levels for drug-alteration pairs."""
        medication_lower = medication_name.lower()

        levels = []

        # Map drugs to OncoKB evidence levels
        if "osimertinib" in medication_lower:
            levels = [
                {
                    "alteration": "EGFR L858R",
                    "cancer_type": "Non-Small Cell Lung Cancer",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                },
                {
                    "alteration": "EGFR Exon 19 deletion",
                    "cancer_type": "Non-Small Cell Lung Cancer",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                },
                {
                    "alteration": "EGFR T790M",
                    "cancer_type": "Non-Small Cell Lung Cancer",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                },
            ]
        elif "alectinib" in medication_lower:
            levels = [
                {
                    "alteration": "ALK Fusion",
                    "cancer_type": "Non-Small Cell Lung Cancer",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                }
            ]
        elif "sotorasib" in medication_lower or "adagrasib" in medication_lower:
            levels = [
                {
                    "alteration": "KRAS G12C",
                    "cancer_type": "Non-Small Cell Lung Cancer",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                }
            ]
        elif "pembrolizumab" in medication_lower:
            levels = [
                {
                    "alteration": "TMB-H",
                    "cancer_type": "All Solid Tumors",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                },
                {
                    "alteration": "MSI-H",
                    "cancer_type": "All Solid Tumors",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                },
            ]
        elif "dabrafenib" in medication_lower:
            levels = [
                {
                    "alteration": "BRAF V600E",
                    "cancer_type": "Non-Small Cell Lung Cancer",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                }
            ]
        elif "entrectinib" in medication_lower:
            levels = [
                {
                    "alteration": "ROS1 Fusion",
                    "cancer_type": "Non-Small Cell Lung Cancer",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                },
                {
                    "alteration": "NTRK Fusion",
                    "cancer_type": "All Solid Tumors",
                    "level": "1",
                    "description": "FDA-recognized biomarker",
                },
            ]

        return levels

    def _get_drug_alterations(self, medication_name: str) -> List[Dict[str, Any]]:
        """Get alterations targeted by the drug."""
        medication_lower = medication_name.lower()

        alteration_map = {
            "osimertinib": {
                "gene": "EGFR",
                "alterations": [
                    {"name": "L858R", "type": "Missense"},
                    {"name": "Exon 19 deletion", "type": "In-frame deletion"},
                    {"name": "T790M", "type": "Missense"},
                    {"name": "C797S", "type": "Resistance mutation"},
                ],
            },
            "alectinib": {
                "gene": "ALK",
                "alterations": [
                    {"name": "EML4-ALK", "type": "Fusion"},
                    {"name": "ALK G1202R", "type": "Resistance mutation"},
                    {"name": "ALK I1171T", "type": "Resistance mutation"},
                ],
            },
            "sotorasib": {
                "gene": "KRAS",
                "alterations": [
                    {"name": "G12C", "type": "Missense"},
                    {"name": "G12C + secondary KRAS", "type": "Resistance"},
                ],
            },
            "pembrolizumab": {
                "gene": "Multiple",
                "alterations": [
                    {"name": "PD-L1 expression", "type": "Biomarker"},
                    {"name": "TMB-H", "type": "Biomarker"},
                    {"name": "MSI-H/dMMR", "type": "Biomarker"},
                ],
            },
            "amivantamab": {
                "gene": "EGFR/MET",
                "alterations": [
                    {"name": "EGFR Exon 20 insertion", "type": "In-frame insertion"},
                    {"name": "MET amplification", "type": "Copy number alteration"},
                ],
            },
        }

        for drug, info in alteration_map.items():
            if drug in medication_lower:
                return [info]

        return []

    def _get_fda_indications(self, medication_name: str) -> List[Dict[str, Any]]:
        """Get FDA-approved indications with biomarkers."""
        medication_lower = medication_name.lower()

        indications = []

        if "osimertinib" in medication_lower:
            indications = [
                {
                    "indication": "First-line treatment of metastatic NSCLC",
                    "biomarker": "EGFR exon 19 deletions or exon 21 L858R",
                    "approval_date": "2018-04-18",
                    "line_of_therapy": "First-line",
                },
                {
                    "indication": "Metastatic EGFR T790M mutation-positive NSCLC",
                    "biomarker": "EGFR T790M",
                    "approval_date": "2015-11-13",
                    "line_of_therapy": "Second-line",
                },
                {
                    "indication": "Adjuvant treatment after tumor resection",
                    "biomarker": "EGFR exon 19 deletions or exon 21 L858R",
                    "approval_date": "2020-12-18",
                    "line_of_therapy": "Adjuvant",
                },
            ]
        elif "pembrolizumab" in medication_lower:
            indications = [
                {
                    "indication": "First-line treatment of metastatic NSCLC",
                    "biomarker": "PD-L1 TPS ≥50%",
                    "approval_date": "2016-10-24",
                    "line_of_therapy": "First-line",
                },
                {
                    "indication": "With chemotherapy for metastatic NSCLC",
                    "biomarker": "No EGFR or ALK aberrations",
                    "approval_date": "2018-08-20",
                    "line_of_therapy": "First-line",
                },
            ]
        elif "sotorasib" in medication_lower:
            indications = [
                {
                    "indication": "KRAS G12C-mutated locally advanced or metastatic NSCLC",
                    "biomarker": "KRAS G12C",
                    "approval_date": "2021-05-28",
                    "line_of_therapy": "Second-line or later",
                }
            ]

        return indications

    def _get_resistance_mechanisms(self, medication_name: str) -> List[Dict[str, Any]]:
        """Get known resistance mechanisms."""
        medication_lower = medication_name.lower()

        resistance_map = {
            "osimertinib": [
                {
                    "mechanism": "EGFR C797S mutation",
                    "frequency": "~25%",
                    "description": "Prevents drug binding",
                },
                {
                    "mechanism": "MET amplification",
                    "frequency": "15-20%",
                    "description": "Bypass pathway activation",
                },
                {
                    "mechanism": "HER2 amplification",
                    "frequency": "5-10%",
                    "description": "Alternative signaling",
                },
                {
                    "mechanism": "Small cell transformation",
                    "frequency": "3-5%",
                    "description": "Histologic transformation",
                },
            ],
            "alectinib": [
                {
                    "mechanism": "ALK G1202R mutation",
                    "frequency": "20-30%",
                    "description": "Gatekeeper mutation",
                },
                {
                    "mechanism": "ALK I1171 mutations",
                    "frequency": "10-15%",
                    "description": "Multiple variants",
                },
            ],
            "sotorasib": [
                {
                    "mechanism": "Secondary KRAS mutations",
                    "frequency": "10-20%",
                    "description": "G12C/G13D, Y96D",
                },
                {
                    "mechanism": "Bypass pathway activation",
                    "frequency": "Variable",
                    "description": "EGFR, MET, FGFR activation",
                },
            ],
        }

        for drug, mechanisms in resistance_map.items():
            if drug in medication_lower:
                return mechanisms

        return []

    def _get_clinical_evidence(self, medication_name: str) -> List[Dict[str, Any]]:
        """Get key clinical trial evidence."""
        medication_lower = medication_name.lower()

        evidence = []

        if "osimertinib" in medication_lower:
            evidence = [
                {
                    "trial": "FLAURA",
                    "phase": "Phase 3",
                    "comparison": "Osimertinib vs erlotinib/gefitinib",
                    "primary_endpoint": "PFS",
                    "result": "18.9 vs 10.2 months (HR 0.46)",
                    "publication": "NEJM 2018",
                },
                {
                    "trial": "ADAURA",
                    "phase": "Phase 3",
                    "setting": "Adjuvant",
                    "primary_endpoint": "DFS",
                    "result": "89% vs 52% at 24 months",
                    "publication": "NEJM 2020",
                },
            ]
        elif "pembrolizumab" in medication_lower:
            evidence = [
                {
                    "trial": "KEYNOTE-024",
                    "phase": "Phase 3",
                    "comparison": "Pembrolizumab vs chemotherapy",
                    "primary_endpoint": "PFS",
                    "result": "10.3 vs 6.0 months (HR 0.50)",
                    "publication": "NEJM 2016",
                },
                {
                    "trial": "KEYNOTE-189",
                    "phase": "Phase 3",
                    "comparison": "Pembrolizumab + chemo vs chemo",
                    "primary_endpoint": "OS",
                    "result": "22.0 vs 10.6 months",
                    "publication": "NEJM 2018",
                },
            ]
        elif "alectinib" in medication_lower:
            evidence = [
                {
                    "trial": "ALEX",
                    "phase": "Phase 3",
                    "comparison": "Alectinib vs crizotinib",
                    "primary_endpoint": "PFS",
                    "result": "34.8 vs 10.9 months (HR 0.43)",
                    "publication": "NEJM 2020",
                }
            ]

        return evidence

    async def get_biomarker_summary(self, gene: str) -> Dict[str, Any]:
        """
        Get summary of a biomarker/gene.

        Args:
            gene: Gene name (e.g., 'EGFR', 'ALK')

        Returns:
            Biomarker summary
        """
        try:
            biomarker_data = {
                "EGFR": {
                    "full_name": "Epidermal Growth Factor Receptor",
                    "frequency_nsclc": "10-15% (Western), 30-50% (Asian)",
                    "mutation_types": [
                        "Exon 19 deletions (45%)",
                        "L858R (40%)",
                        "T790M (acquired resistance)",
                        "Exon 20 insertions (10%)",
                        "Uncommon mutations (5%)",
                    ],
                    "targeted_therapies": [
                        "osimertinib",
                        "erlotinib",
                        "gefitinib",
                        "afatinib",
                        "dacomitinib",
                        "amivantamab",
                        "mobocertinib",
                        "lazertinib",
                    ],
                    "testing_methods": ["NGS", "PCR", "Liquid biopsy"],
                },
                "ALK": {
                    "full_name": "Anaplastic Lymphoma Kinase",
                    "frequency_nsclc": "3-5%",
                    "fusion_partners": ["EML4 (most common)", "KIF5B", "TFG", "Others"],
                    "targeted_therapies": [
                        "alectinib",
                        "brigatinib",
                        "lorlatinib",
                        "crizotinib",
                        "ceritinib",
                        "ensartinib",
                    ],
                    "testing_methods": ["FISH", "IHC", "NGS"],
                },
                "KRAS": {
                    "full_name": "Kirsten Rat Sarcoma viral oncogene",
                    "frequency_nsclc": "25-30%",
                    "mutation_types": [
                        "G12C (40% of KRAS)",
                        "G12V (20%)",
                        "G12D (15%)",
                        "Others",
                    ],
                    "targeted_therapies": ["sotorasib (G12C)", "adagrasib (G12C)"],
                    "testing_methods": ["NGS", "PCR"],
                },
                "ROS1": {
                    "full_name": "ROS proto-oncogene 1",
                    "frequency_nsclc": "1-2%",
                    "fusion_partners": ["CD74", "SLC34A2", "Others"],
                    "targeted_therapies": [
                        "entrectinib",
                        "crizotinib",
                        "ceritinib",
                        "lorlatinib",
                        "repotrectinib",
                    ],
                    "testing_methods": ["FISH", "NGS", "IHC"],
                },
                "BRAF": {
                    "full_name": "B-Raf proto-oncogene",
                    "frequency_nsclc": "2-4%",
                    "mutation_types": ["V600E (50%)", "Non-V600 (50%)"],
                    "targeted_therapies": ["dabrafenib + trametinib (V600E)"],
                    "testing_methods": ["NGS", "PCR"],
                },
                "MET": {
                    "full_name": "Mesenchymal-epithelial transition factor",
                    "frequency_nsclc": "3-4% (exon 14 skipping)",
                    "alteration_types": [
                        "Exon 14 skipping mutations",
                        "MET amplification",
                    ],
                    "targeted_therapies": [
                        "capmatinib",
                        "tepotinib",
                        "savolitinib",
                        "amivantamab",
                    ],
                    "testing_methods": ["NGS", "FISH for amplification"],
                },
                "RET": {
                    "full_name": "Rearranged during transfection",
                    "frequency_nsclc": "1-2%",
                    "fusion_partners": ["KIF5B", "CCDC6", "Others"],
                    "targeted_therapies": ["selpercatinib", "pralsetinib"],
                    "testing_methods": ["NGS", "FISH"],
                },
                "HER2": {
                    "full_name": "Human epidermal growth factor receptor 2",
                    "frequency_nsclc": "2-3%",
                    "alteration_types": [
                        "Exon 20 insertions",
                        "Point mutations",
                        "Amplification (rare)",
                    ],
                    "targeted_therapies": [
                        "trastuzumab deruxtecan",
                        "poziotinib",
                        "pyrotinib",
                    ],
                    "testing_methods": ["NGS", "IHC", "FISH"],
                },
                "NTRK": {
                    "full_name": "Neurotrophic receptor tyrosine kinase",
                    "frequency_nsclc": "<1%",
                    "genes": ["NTRK1", "NTRK2", "NTRK3"],
                    "targeted_therapies": [
                        "larotrectinib",
                        "entrectinib",
                        "repotrectinib",
                    ],
                    "testing_methods": ["NGS", "IHC screening"],
                },
            }

            return biomarker_data.get(
                gene.upper(), {"error": f"Biomarker {gene} not found in database"}
            )

        except Exception as e:
            logger.error("biomarker_summary_failed", gene=gene, error=str(e))
            return {}

    async def search_medications(
        self, query: str, limit: int = 10
    ) -> List[ScraperResult]:
        """
        Search OncoKB for medications.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results
        """
        results = []

        try:
            # OncoKB targeted therapy drugs
            oncokb_drugs = [
                "osimertinib",
                "alectinib",
                "sotorasib",
                "adagrasib",
                "pembrolizumab",
                "dabrafenib",
                "trametinib",
                "entrectinib",
                "larotrectinib",
                "selpercatinib",
                "pralsetinib",
                "capmatinib",
                "tepotinib",
                "amivantamab",
                "trastuzumab deruxtecan",
            ]

            for drug in oncokb_drugs[:limit]:
                if query.lower() in drug.lower():
                    result = await self.scrape_medication_info(drug)
                    results.append(result)

            return results

        except Exception as e:
            logger.error("oncokb_search_failed", query=query, error=str(e))
            return []
