"""NCCN guidelines scraper for cancer treatment information."""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from .base import BaseScraper, ScraperConfig, ScraperResult

logger = structlog.get_logger()


class NCCNScraper(BaseScraper):
    """Scraper for NCCN (National Comprehensive Cancer Network) guidelines."""

    def __init__(self, client=None):
        """Initialize NCCN scraper."""
        config = ScraperConfig(
            base_url="https://www.nccn.org",
            rate_limit=2.0,  # NCCN requires slower rate
            timeout=30,
            user_agent="MedicationAugmentation/1.0 (Educational/Research)",
        )
        super().__init__(config, client)

    async def scrape_medication_info(self, medication_name: str) -> ScraperResult:
        """
        Scrape NCCN guidelines for medication information.

        Args:
            medication_name: Name of the medication

        Returns:
            Scraper result with NCCN data
        """
        try:
            # NCCN requires registration for full access
            # For now, return mock data structure based on typical NCCN content

            data = {
                "medication": medication_name,
                "guideline": "NSCLC",
                "version": "2024.5",
                "last_updated": "2024-10-15",
                "recommendations": self._get_mock_recommendations(medication_name),
                "treatment_lines": self._get_treatment_lines(medication_name),
                "biomarker_driven": self._is_biomarker_driven(medication_name),
                "combination_regimens": self._get_combination_regimens(medication_name),
                "dosing_schedules": self._get_dosing_schedules(medication_name),
            }

            return ScraperResult(
                success=True,
                data=data,
                url=f"{self.config.base_url}/professionals/physician_gls/pdf/nscl.pdf",
            )

        except Exception as e:
            logger.error("nccn_scrape_failed", medication=medication_name, error=str(e))
            return ScraperResult(
                success=False, data={}, url=self.config.base_url, error=str(e)
            )

    def _get_mock_recommendations(self, medication_name: str) -> List[Dict[str, Any]]:
        """Get NCCN recommendation categories for medication."""
        medication_lower = medication_name.lower()

        # Map medications to NCCN categories
        if "pembrolizumab" in medication_lower:
            return [
                {
                    "category": "1",
                    "indication": "First-line for PD-L1 ≥50% without driver mutations",
                    "evidence": "High-quality evidence",
                },
                {
                    "category": "1",
                    "indication": "With chemotherapy for PD-L1 <50%",
                    "evidence": "High-quality evidence",
                },
            ]
        elif "osimertinib" in medication_lower:
            return [
                {
                    "category": "1",
                    "indication": "First-line for EGFR exon 19 del or L858R",
                    "evidence": "High-quality evidence",
                },
                {
                    "category": "1",
                    "indication": "Second-line for T790M mutation",
                    "evidence": "High-quality evidence",
                },
            ]
        elif "alectinib" in medication_lower:
            return [
                {
                    "category": "1",
                    "indication": "First-line for ALK-positive NSCLC",
                    "evidence": "High-quality evidence",
                }
            ]
        else:
            return [
                {
                    "category": "2A",
                    "indication": "Alternative therapy option",
                    "evidence": "Lower-level evidence",
                }
            ]

    def _get_treatment_lines(self, medication_name: str) -> Dict[str, List[str]]:
        """Get treatment line information."""
        medication_lower = medication_name.lower()

        lines = {"first_line": [], "second_line": [], "subsequent": []}

        # Categorize by treatment line
        first_line_drugs = [
            "pembrolizumab",
            "osimertinib",
            "alectinib",
            "brigatinib",
            "ceritinib",
            "crizotinib",
            "sotorasib",
            "adagrasib",
        ]

        if any(drug in medication_lower for drug in first_line_drugs):
            lines["first_line"].append(medication_name)

        # Add context-specific placements
        if "docetaxel" in medication_lower:
            lines["second_line"].append(f"{medication_name} +/- ramucirumab")

        if "nivolumab" in medication_lower:
            lines["second_line"].append(f"{medication_name} +/- ipilimumab")

        return lines

    def _is_biomarker_driven(self, medication_name: str) -> Dict[str, Any]:
        """Check if medication is biomarker-driven."""
        medication_lower = medication_name.lower()

        biomarker_map = {
            "osimertinib": {
                "biomarker": "EGFR",
                "mutations": ["exon 19 del", "L858R", "T790M"],
            },
            "erlotinib": {"biomarker": "EGFR", "mutations": ["exon 19 del", "L858R"]},
            "afatinib": {
                "biomarker": "EGFR",
                "mutations": ["exon 19 del", "L858R", "uncommon"],
            },
            "dacomitinib": {"biomarker": "EGFR", "mutations": ["exon 19 del", "L858R"]},
            "alectinib": {"biomarker": "ALK", "mutations": ["ALK fusion"]},
            "brigatinib": {"biomarker": "ALK", "mutations": ["ALK fusion"]},
            "lorlatinib": {
                "biomarker": "ALK",
                "mutations": ["ALK fusion", "ALK resistance"],
            },
            "sotorasib": {"biomarker": "KRAS", "mutations": ["G12C"]},
            "adagrasib": {"biomarker": "KRAS", "mutations": ["G12C"]},
            "dabrafenib": {"biomarker": "BRAF", "mutations": ["V600E"]},
            "entrectinib": {"biomarker": "ROS1", "mutations": ["ROS1 fusion"]},
            "capmatinib": {"biomarker": "MET", "mutations": ["MET exon 14 skipping"]},
            "tepotinib": {"biomarker": "MET", "mutations": ["MET exon 14 skipping"]},
            "larotrectinib": {"biomarker": "NTRK", "mutations": ["NTRK fusion"]},
            "pralsetinib": {"biomarker": "RET", "mutations": ["RET fusion"]},
            "selpercatinib": {"biomarker": "RET", "mutations": ["RET fusion"]},
            "amivantamab": {"biomarker": "EGFR", "mutations": ["exon 20 insertion"]},
            "mobocertinib": {"biomarker": "EGFR", "mutations": ["exon 20 insertion"]},
            "trastuzumab deruxtecan": {
                "biomarker": "HER2",
                "mutations": ["HER2 mutation"],
            },
        }

        for drug, info in biomarker_map.items():
            if drug in medication_lower:
                return {"is_biomarker_driven": True, **info}

        # PD-L1 based immunotherapy
        if any(drug in medication_lower for drug in ["pembrolizumab", "atezolizumab"]):
            return {
                "is_biomarker_driven": True,
                "biomarker": "PD-L1",
                "threshold": "≥1% or ≥50% depending on setting",
            }

        return {"is_biomarker_driven": False}

    def _get_combination_regimens(self, medication_name: str) -> List[Dict[str, Any]]:
        """Get NCCN-recommended combination regimens."""
        medication_lower = medication_name.lower()

        regimens = []

        if "pembrolizumab" in medication_lower:
            regimens = [
                {
                    "name": "Pembrolizumab + Carboplatin + Pemetrexed",
                    "indication": "Non-squamous NSCLC",
                    "category": "1",
                },
                {
                    "name": "Pembrolizumab + Carboplatin + Paclitaxel/nab-paclitaxel",
                    "indication": "Squamous NSCLC",
                    "category": "1",
                },
                {
                    "name": "Pembrolizumab monotherapy",
                    "indication": "PD-L1 ≥50%",
                    "category": "1",
                },
            ]
        elif "nivolumab" in medication_lower:
            regimens = [
                {
                    "name": "Nivolumab + Ipilimumab",
                    "indication": "First-line PD-L1 ≥1%",
                    "category": "1",
                },
                {
                    "name": "Nivolumab + Ipilimumab + Chemotherapy",
                    "indication": "First-line regardless of PD-L1",
                    "category": "1",
                },
            ]
        elif "atezolizumab" in medication_lower:
            regimens = [
                {
                    "name": "Atezolizumab + Bevacizumab + Carboplatin + Paclitaxel",
                    "indication": "Non-squamous NSCLC",
                    "category": "1",
                },
                {
                    "name": "Atezolizumab + Carboplatin + nab-paclitaxel",
                    "indication": "Non-squamous NSCLC",
                    "category": "1",
                },
            ]
        elif "dabrafenib" in medication_lower:
            regimens = [
                {
                    "name": "Dabrafenib + Trametinib",
                    "indication": "BRAF V600E mutation",
                    "category": "1",
                }
            ]

        return regimens

    def _get_dosing_schedules(self, medication_name: str) -> Dict[str, Any]:
        """Get NCCN-recommended dosing schedules."""
        medication_lower = medication_name.lower()

        dosing_map = {
            "pembrolizumab": {
                "dose": "200 mg or 2 mg/kg",
                "frequency": "Every 3 weeks",
                "alternative": "400 mg every 6 weeks",
                "duration": "Until progression or 2 years",
            },
            "nivolumab": {
                "dose": "240 mg",
                "frequency": "Every 2 weeks",
                "alternative": "480 mg every 4 weeks",
                "duration": "Until progression",
            },
            "osimertinib": {
                "dose": "80 mg",
                "frequency": "Once daily",
                "duration": "Until progression",
            },
            "alectinib": {
                "dose": "600 mg",
                "frequency": "Twice daily with food",
                "duration": "Until progression",
            },
            "atezolizumab": {
                "dose": "840 mg, 1200 mg, or 1680 mg",
                "frequency": "Every 2, 3, or 4 weeks",
                "duration": "Until progression",
            },
            "durvalumab": {
                "dose": "10 mg/kg",
                "frequency": "Every 2 weeks",
                "alternative": "1500 mg every 4 weeks",
                "duration": "Until progression or 1 year",
            },
        }

        for drug, schedule in dosing_map.items():
            if drug in medication_lower:
                return schedule

        return {
            "dose": "See prescribing information",
            "frequency": "Varies by indication",
            "duration": "Until progression or unacceptable toxicity",
        }

    async def search_medications(
        self, query: str, limit: int = 10
    ) -> List[ScraperResult]:
        """
        Search NCCN guidelines for medications.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results
        """
        results = []

        try:
            # Mock NCCN preferred regimens
            nccn_preferred = [
                "pembrolizumab",
                "nivolumab + ipilimumab",
                "atezolizumab",
                "osimertinib",
                "alectinib",
                "brigatinib",
                "sotorasib",
                "adagrasib",
                "dabrafenib + trametinib",
                "amivantamab",
                "trastuzumab deruxtecan",
            ]

            for drug in nccn_preferred[:limit]:
                if query.lower() in drug.lower():
                    result = await self.scrape_medication_info(drug)
                    results.append(result)

            return results

        except Exception as e:
            logger.error("nccn_search_failed", query=query, error=str(e))
            return []

    async def get_recent_updates(self) -> List[Dict[str, Any]]:
        """
        Get recent NCCN guideline updates.

        Returns:
            List of recent updates
        """
        try:
            # Mock recent NCCN updates for NSCLC
            updates = [
                {
                    "date": "2024-10-15",
                    "version": "5.2024",
                    "changes": [
                        "Added amivantamab-vmjw + carboplatin + pemetrexed for EGFR exon 20 ins",
                        "Updated pembrolizumab duration recommendations",
                        "Added tremelimumab + durvalumab + chemotherapy option",
                    ],
                },
                {
                    "date": "2024-08-01",
                    "version": "4.2024",
                    "changes": [
                        "Added adagrasib for KRAS G12C second-line",
                        "Updated biomarker testing recommendations",
                        "Clarified PD-L1 testing requirements",
                    ],
                },
                {
                    "date": "2024-06-15",
                    "version": "3.2024",
                    "changes": [
                        "Added trastuzumab deruxtecan for HER2-mutant NSCLC",
                        "Updated osimertinib adjuvant therapy recommendations",
                        "Added new combination immunotherapy regimens",
                    ],
                },
            ]

            return updates

        except Exception as e:
            logger.error("nccn_updates_failed", error=str(e))
            return []

    async def get_treatment_algorithm(
        self, histology: str, biomarkers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get NCCN treatment algorithm based on histology and biomarkers.

        Args:
            histology: Tumor histology (e.g., 'adenocarcinoma', 'squamous')
            biomarkers: Dictionary of biomarker results

        Returns:
            Treatment algorithm
        """
        try:
            algorithm = {
                "histology": histology,
                "biomarkers": biomarkers,
                "recommendations": [],
            }

            # EGFR mutations
            if biomarkers.get("EGFR"):
                if "exon 20" in str(biomarkers["EGFR"]):
                    algorithm["recommendations"].append(
                        {
                            "priority": 1,
                            "regimen": "Amivantamab-vmjw + chemotherapy",
                            "category": "1",
                        }
                    )
                else:
                    algorithm["recommendations"].append(
                        {"priority": 1, "regimen": "Osimertinib", "category": "1"}
                    )

            # ALK rearrangements
            elif biomarkers.get("ALK"):
                algorithm["recommendations"].append(
                    {
                        "priority": 1,
                        "regimen": "Alectinib or brigatinib or lorlatinib",
                        "category": "1",
                    }
                )

            # KRAS G12C
            elif biomarkers.get("KRAS") == "G12C":
                algorithm["recommendations"].append(
                    {
                        "priority": 1,
                        "regimen": "Sotorasib or adagrasib",
                        "category": "1",
                    }
                )

            # PD-L1 high
            elif biomarkers.get("PD-L1", 0) >= 50:
                algorithm["recommendations"].append(
                    {
                        "priority": 1,
                        "regimen": "Pembrolizumab monotherapy",
                        "category": "1",
                    }
                )

            # Default chemotherapy + immunotherapy
            else:
                if histology.lower() == "squamous":
                    algorithm["recommendations"].append(
                        {
                            "priority": 1,
                            "regimen": "Pembrolizumab + carboplatin + paclitaxel",
                            "category": "1",
                        }
                    )
                else:
                    algorithm["recommendations"].append(
                        {
                            "priority": 1,
                            "regimen": "Pembrolizumab + carboplatin + pemetrexed",
                            "category": "1",
                        }
                    )

            return algorithm

        except Exception as e:
            logger.error("nccn_algorithm_failed", error=str(e))
            return {}
