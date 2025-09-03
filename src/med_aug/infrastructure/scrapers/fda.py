"""FDA Orange Book and drug database scraper.

Integrates with FDA's public APIs to retrieve medication information:
- OpenFDA Drug Database API
- Orange Book database
"""

from typing import List, Dict, Any, Optional
from urllib.parse import quote
import structlog
from .base import BaseScraper, ScraperConfig, ScraperResult

logger = structlog.get_logger()


class FDAScraper(BaseScraper):
    """Scraper for FDA Orange Book and drug databases."""

    def __init__(self, client=None):
        """Initialize FDA scraper."""
        config = ScraperConfig(
            base_url="https://api.fda.gov",
            rate_limit=1.0,  # FDA rate limit: 1 request per second for unauthenticated requests
            timeout=30,
            user_agent="MedicationAugmentation/1.0 (Educational/Research)",
        )
        super().__init__(config, client)

    async def scrape_medication_info(self, medication_name: str) -> ScraperResult:
        """
        Scrape FDA information for a specific medication using OpenFDA API.

        Args:
            medication_name: Name of the medication

        Returns:
            Scraper result with FDA data
        """
        try:
            # Clean medication name for search
            clean_name = medication_name.strip().lower()

            # Try multiple search strategies
            fda_data = await self._search_openfda_drugs(clean_name)
            if not fda_data:
                # Fallback to Orange Book search
                fda_data = await self._search_orange_book(clean_name)

            if not fda_data:
                return ScraperResult(
                    success=False,
                    data={},
                    url=f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{quote(clean_name)}",
                    error=f"No FDA data found for {medication_name}",
                )

            return ScraperResult(
                success=True,
                data=fda_data,
                url=f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{quote(clean_name)}",
            )

        except Exception as e:
            logger.error("fda_scrape_failed", medication=medication_name, error=str(e))
            return ScraperResult(
                success=False, data={}, url="https://api.fda.gov", error=str(e)
            )

    async def _search_openfda_drugs(
        self, medication_name: str
    ) -> Optional[Dict[str, Any]]:
        """Search OpenFDA drug database."""
        try:
            # Search by brand name first
            search_terms = [
                f"openfda.brand_name:{quote(medication_name)}",
                f"openfda.generic_name:{quote(medication_name)}",
                f"openfda.substance_name:{quote(medication_name)}",
            ]

            for search_term in search_terms:
                url = f"{self.config.base_url}/drug/label.json?search={search_term}&limit=5"

                async with self.rate_limit():
                    response = await self.client.get(
                        url, headers={"User-Agent": self.config.user_agent}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("results"):
                            result = data["results"][0]  # Take first result
                            return self._parse_openfda_result(result, medication_name)
                    elif response.status_code == 404:
                        continue  # Try next search term
                    else:
                        logger.warning(
                            "fda_api_error", status=response.status_code, url=url
                        )

            return None

        except Exception as e:
            logger.error(
                "openfda_search_failed", medication=medication_name, error=str(e)
            )
            return None

    async def _search_orange_book(
        self, medication_name: str
    ) -> Optional[Dict[str, Any]]:
        """Search FDA Orange Book database."""
        try:
            # Orange Book API endpoint
            url = f"{self.config.base_url}/drug/drugsfda.json?search=products.brand_name:{quote(medication_name)}&limit=5"

            async with self.rate_limit():
                response = await self.client.get(
                    url, headers={"User-Agent": self.config.user_agent}
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("results"):
                        result = data["results"][0]
                        return self._parse_orange_book_result(result, medication_name)
                elif response.status_code != 404:
                    logger.warning(
                        "orange_book_api_error", status=response.status_code, url=url
                    )

            return None

        except Exception as e:
            logger.error(
                "orange_book_search_failed", medication=medication_name, error=str(e)
            )
            return None

    def _parse_openfda_result(
        self, result: Dict[str, Any], medication_name: str
    ) -> Dict[str, Any]:
        """Parse OpenFDA API result into standardized format."""
        openfda = result.get("openfda", {})

        return {
            "name": medication_name,
            "source": "OpenFDA",
            "fda_application_number": (
                openfda.get("application_number", [""])[0]
                if openfda.get("application_number")
                else ""
            ),
            "active_ingredient": (
                openfda.get("generic_name", [""])[0]
                if openfda.get("generic_name")
                else ""
            ),
            "brand_names": openfda.get("brand_name", []),
            "generic_names": openfda.get("generic_name", []),
            "manufacturer": (
                openfda.get("manufacturer_name", [""])[0]
                if openfda.get("manufacturer_name")
                else ""
            ),
            "dosage_forms": openfda.get("dosage_form", []),
            "routes": openfda.get("route", []),
            "substance_names": openfda.get("substance_name", []),
            "product_numbers": openfda.get("product_ndc", []),
            "indications_and_usage": (
                result.get("indications_and_usage", [""])[0]
                if result.get("indications_and_usage")
                else ""
            ),
            "warnings": (
                result.get("warnings", [""])[0] if result.get("warnings") else ""
            ),
            "purpose": result.get("purpose", [""])[0] if result.get("purpose") else "",
        }

    def _parse_orange_book_result(
        self, result: Dict[str, Any], medication_name: str
    ) -> Dict[str, Any]:
        """Parse Orange Book API result into standardized format."""
        products = result.get("products", [{}])[0] if result.get("products") else {}

        return {
            "name": medication_name,
            "source": "FDA Orange Book",
            "fda_application_number": result.get("application_number", ""),
            "active_ingredient": (
                products.get("active_ingredients", [{}])[0].get("name", "")
                if products.get("active_ingredients")
                else ""
            ),
            "brand_names": (
                [products.get("brand_name", "")] if products.get("brand_name") else []
            ),
            "generic_names": (
                [products.get("generic_name", "")]
                if products.get("generic_name")
                else []
            ),
            "manufacturer": result.get("sponsor_name", ""),
            "dosage_forms": (
                [products.get("dosage_form", "")] if products.get("dosage_form") else []
            ),
            "strength": (
                [products.get("active_ingredients", [{}])[0].get("strength", "")]
                if products.get("active_ingredients")
                else []
            ),
            "routes": [products.get("route", "")] if products.get("route") else [],
            "approval_date": products.get("marketing_start_date", ""),
            "marketing_status": products.get("marketing_status", ""),
            "therapeutic_equivalence": products.get("te_code", ""),
            "reference_drug": products.get("reference_drug", "No"),
        }

    async def search_medications(
        self, query: str, limit: int = 10
    ) -> List[ScraperResult]:
        """
        Search FDA database for medications.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results
        """
        results: List[ScraperResult] = []

        try:
            # Search OpenFDA for medications matching query
            search_terms = [
                f"openfda.brand_name:{quote(query)}*",
                f"openfda.generic_name:{quote(query)}*",
                f"openfda.substance_name:{quote(query)}*",
            ]

            seen_names = set()

            for search_term in search_terms:
                if len(results) >= limit:
                    break

                url = f"{self.config.base_url}/drug/label.json?search={search_term}&limit={limit}"

                async with self.rate_limit():
                    response = await self.client.get(
                        url, headers={"User-Agent": self.config.user_agent}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        for result in data.get("results", []):
                            if len(results) >= limit:
                                break

                            openfda = result.get("openfda", {})
                            brand_names = openfda.get("brand_name", [])
                            generic_names = openfda.get("generic_name", [])

                            # Use first brand name or generic name
                            name = (
                                brand_names[0]
                                if brand_names
                                else generic_names[0]
                                if generic_names
                                else "Unknown"
                            )

                            if name.lower() not in seen_names and name != "Unknown":
                                seen_names.add(name.lower())
                                parsed_data = self._parse_openfda_result(result, name)
                                results.append(
                                    ScraperResult(
                                        success=True, data=parsed_data, url=url
                                    )
                                )

            return results[:limit]

        except Exception as e:
            logger.error("fda_search_failed", query=query, error=str(e))
            return []

    async def get_recent_approvals(self, days: int = 365) -> List[ScraperResult]:
        """
        Get recently FDA-approved medications.

        Args:
            days: Number of days to look back

        Returns:
            List of recently approved medications
        """
        results = []

        try:
            # FDA recent approvals page
            approvals_url = f"{self.config.base_url}/drugs/new-drugs-fda-cders-new-molecular-entities-and-new-therapeutic-biological-products"

            # Mock recent approvals for NSCLC
            recent_nsclc_drugs = [
                {
                    "name": "amivantamab-vmjw",
                    "brand_name": "Rybrevant",
                    "approval_date": "2024-03-15",
                    "indication": "EGFR exon 20 insertion mutations",
                },
                {
                    "name": "adagrasib",
                    "brand_name": "Krazati",
                    "approval_date": "2024-01-20",
                    "indication": "KRAS G12C-mutated NSCLC",
                },
                {
                    "name": "trastuzumab deruxtecan",
                    "brand_name": "Enhertu",
                    "approval_date": "2024-02-10",
                    "indication": "HER2-mutant NSCLC",
                },
            ]

            for drug_info in recent_nsclc_drugs:
                data = {
                    "name": drug_info["name"],
                    "brand_names": [drug_info["brand_name"]],
                    "generic_names": [drug_info["name"]],
                    "approval_date": drug_info["approval_date"],
                    "indications": [drug_info["indication"]],
                    "drug_class": ["targeted_therapy"],
                    "fda_fast_track": True,
                    "orphan_drug": False,
                }

                results.append(
                    ScraperResult(success=True, data=data, url=approvals_url)
                )

            return results

        except Exception as e:
            logger.error("fda_recent_approvals_failed", error=str(e))
            return []

    async def get_orange_book_data(self, medication_name: str) -> Dict[str, Any]:
        """
        Get Orange Book specific data for a medication.

        Args:
            medication_name: Name of medication

        Returns:
            Orange Book data
        """
        try:
            # Mock Orange Book data
            data = {
                "active_ingredient": medication_name.lower(),
                "proprietary_name": medication_name.title(),
                "dosage_form_route": "TABLET;ORAL",
                "strength": "100MG",
                "reference_listed_drug": True,
                "reference_standard": "RS",
                "therapeutic_equivalence_code": "AB",
                "application_number": "NDA123456",
                "product_number": "001",
                "approval_date": "2024-01-15",
                "applicant": "PHARMA CORP",
                "patent_data": [
                    {
                        "patent_number": "9,999,999",
                        "expiration_date": "2035-01-15",
                        "drug_substance_claim": True,
                    }
                ],
                "exclusivity_data": [
                    {"exclusivity_code": "NCE", "expiration_date": "2029-01-15"}
                ],
            }

            return data

        except Exception as e:
            logger.error("orange_book_failed", medication=medication_name, error=str(e))
            return {}

    async def get_drug_label(self, medication_name: str) -> Dict[str, Any]:
        """
        Get FDA drug label information.

        Args:
            medication_name: Name of medication

        Returns:
            Drug label data
        """
        try:
            # Mock label data
            data = {
                "drug_name": medication_name,
                "indications_and_usage": "For the treatment of metastatic non-small cell lung cancer",
                "dosage_and_administration": "Recommended dose is 200mg once daily",
                "dosage_forms_and_strengths": ["Tablets: 100mg, 200mg"],
                "contraindications": ["Hypersensitivity to drug substance"],
                "warnings_and_precautions": [
                    "Hepatotoxicity",
                    "Pneumonitis",
                    "QTc prolongation",
                ],
                "adverse_reactions": [
                    "Fatigue",
                    "Nausea",
                    "Decreased appetite",
                    "Rash",
                ],
                "drug_interactions": [
                    "Strong CYP3A4 inhibitors",
                    "Strong CYP3A4 inducers",
                ],
                "clinical_pharmacology": {
                    "mechanism_of_action": "Selective inhibitor of mutant forms",
                    "pharmacokinetics": "Oral bioavailability >80%",
                },
            }

            return data

        except Exception as e:
            logger.error("drug_label_failed", medication=medication_name, error=str(e))
            return {}
