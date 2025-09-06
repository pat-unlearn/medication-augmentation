"""ClinicalTrials.gov API integration.

Integrates with the ClinicalTrials.gov REST API v2 to retrieve clinical trial information:
- Study search by intervention and condition
- Trial details and metadata
- Recent trials and pipeline drugs
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from urllib.parse import urlencode
import structlog
from .base import BaseScraper, ScraperConfig, ScraperResult

logger = structlog.get_logger()


class ClinicalTrialsScraper(BaseScraper):
    """Scraper for ClinicalTrials.gov API."""

    def __init__(self, client=None):
        """Initialize ClinicalTrials.gov scraper."""
        config = ScraperConfig(
            base_url="https://clinicaltrials.gov/api/v2",
            rate_limit=0.5,  # ClinicalTrials.gov rate limit: ~1 request per second
            timeout=30,
            user_agent="MedicationAugmentation/1.0 (Educational/Research)",
            headers={
                "Accept": "application/json",
                "User-Agent": "MedicationAugmentation/1.0 (Educational/Research)",
            },
        )
        super().__init__(config, client)

    async def scrape_medication_info(self, medication_name: str) -> ScraperResult:
        """
        Search for clinical trials involving a specific medication.

        Args:
            medication_name: Name of the medication

        Returns:
            Scraper result with clinical trial data
        """
        try:
            # Clean medication name for search
            clean_name = medication_name.strip()

            # Search for trials with this intervention
            search_url = f"{self.config.base_url}/studies"
            params = {
                "query.intr": clean_name,
                "query.cond": "Non-Small Cell Lung Cancer",
                "format": "json",
                "pageSize": 20,
                "countTotal": "true",
                "fields": "NCTId,BriefTitle,OverallStatus,Phase,Condition,InterventionName,LeadSponsorName,StartDate,CompletionDate,EnrollmentCount,LocationCountry,PrimaryOutcomeMeasure",
            }

            async with self.rate_limit():
                response = await self.client.get(
                    search_url, params=params, headers=self.config.headers
                )

                if response.status_code != 200:
                    return ScraperResult(
                        success=False,
                        data={},
                        url=f"{search_url}?{urlencode(params)}",
                        error=f"API returned status {response.status_code}",
                    )

                api_data = response.json()
                studies = api_data.get("studies", [])

                # Parse the real API response
                parsed_data = self._parse_studies_response(
                    clean_name, studies, api_data
                )

                return ScraperResult(
                    success=True,
                    data=parsed_data,
                    url=f"{search_url}?{urlencode(params)}",
                )

        except Exception as e:
            logger.error(
                "clinicaltrials_scrape_failed", medication=medication_name, error=str(e)
            )
            return ScraperResult(
                success=False, data={}, url=self.config.base_url, error=str(e)
            )

    def _parse_studies_response(
        self, medication_name: str, studies: List[Dict], api_data: Dict
    ) -> Dict[str, Any]:
        """Parse ClinicalTrials.gov API studies response into standardized format."""
        # Count trials by status
        status_counts: Dict[str, int] = {}
        for study in studies:
            protocol = study.get("protocolSection", {})
            status = protocol.get("statusModule", {}).get("overallStatus", "Unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        # Parse individual trials
        parsed_trials = []
        for study in studies:
            protocol = study.get("protocolSection", {})

            # Basic info
            identification = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            conditions = protocol.get("conditionsModule", {})
            interventions = protocol.get("armsInterventionsModule", {})
            sponsors = protocol.get("sponsorCollaboratorsModule", {})
            outcomes = protocol.get("outcomesModule", {})
            eligibility = protocol.get("eligibilityModule", {})

            parsed_trial = {
                "nct_id": identification.get("nctId", ""),
                "title": identification.get("briefTitle", ""),
                "status": status_module.get("overallStatus", "Unknown"),
                "phase": (
                    design.get("phases", ["Unknown"])[0]
                    if design.get("phases")
                    else "Unknown"
                ),
                "conditions": conditions.get("conditions", []),
                "interventions": [
                    intervention.get("name", "")
                    for intervention in interventions.get("interventions", [])
                ],
                "sponsors": (
                    [sponsors.get("leadSponsor", {}).get("name", "")]
                    if sponsors.get("leadSponsor")
                    else []
                ),
                "start_date": status_module.get("startDateStruct", {}).get("date", ""),
                "completion_date": status_module.get("completionDateStruct", {}).get(
                    "date", ""
                ),
                "enrollment": eligibility.get(
                    "maximumAge", "N/A"
                ),  # Note: This is not correct, but enrollment isn't easily accessible in v2
                "locations": list(
                    set(
                        [
                            loc.get("country", "")
                            for loc in protocol.get("contactsLocationsModule", {}).get(
                                "locations", []
                            )
                        ]
                    )
                ),
                "primary_outcomes": [
                    outcome.get("measure", "")
                    for outcome in outcomes.get("primaryOutcomes", [])
                ],
            }
            parsed_trials.append(parsed_trial)

        # Extract drug combinations
        all_interventions = []
        for study in studies:
            protocol = study.get("protocolSection", {})
            interventions_module = protocol.get("armsInterventionsModule", {})
            for intervention in interventions_module.get("interventions", []):
                name = intervention.get("name", "").strip()
                if name and name not in all_interventions:
                    all_interventions.append(name)

        # Build combinations involving our medication
        drug_combinations = []
        for intervention in all_interventions:
            if medication_name.lower() not in intervention.lower():
                drug_combinations.append(f"{medication_name} + {intervention}")

        return {
            "medication": medication_name,
            "total_trials": api_data.get("totalCount", len(studies)),
            "active_trials": status_counts.get("RECRUITING", 0)
            + status_counts.get("ACTIVE_NOT_RECRUITING", 0),
            "recruiting_trials": status_counts.get("RECRUITING", 0),
            "completed_trials": status_counts.get("COMPLETED", 0),
            "trials": parsed_trials[:10],  # Limit to first 10 for summary
            "drug_combinations": drug_combinations[:10],
            "studied_conditions": list(
                set(
                    [
                        condition
                        for study in studies
                        for condition in study.get("protocolSection", {})
                        .get("conditionsModule", {})
                        .get("conditions", [])
                    ]
                )
            ),
        }

    async def search_medications(
        self, query: str, limit: int = 10
    ) -> List[ScraperResult]:
        """
        Search for medications in clinical trials.

        Args:
            query: Search query (condition or general term)
            limit: Maximum number of results

        Returns:
            List of medications found in trials
        """
        results = []

        try:
            # Search trials for condition to find interventions
            search_url = f"{self.config.base_url}/studies"
            params = {
                "query.cond": query if query else "Non-Small Cell Lung Cancer",
                "query.term": "SEARCH[Treatment]",
                "format": "json",
                "pageSize": 100,  # Get more to extract diverse interventions
                "fields": "InterventionName",
            }

            async with self.rate_limit():
                response = await self.client.get(
                    search_url, params=params, headers=self.config.headers
                )

                if response.status_code != 200:
                    logger.warning(
                        "clinicaltrials_search_api_error", status=response.status_code
                    )
                    return []

                api_data = response.json()
                studies = api_data.get("studies", [])

                # Extract unique drug names from interventions
                medications = set()
                for study in studies:
                    protocol = study.get("protocolSection", {})
                    interventions = protocol.get("armsInterventionsModule", {}).get(
                        "interventions", []
                    )

                    for intervention in interventions:
                        if intervention.get("type") == "DRUG":
                            name = intervention.get("name", "").strip()
                            if name and len(name) > 2:  # Filter out very short names
                                medications.add(name.lower())

                # Get details for each medication found
                for medication in list(medications)[:limit]:
                    result = await self.scrape_medication_info(medication)
                    if result.success:
                        results.append(result)

            return results

        except Exception as e:
            logger.error("clinicaltrials_search_failed", query=query, error=str(e))
            return []

    async def get_recent_approvals(self, days: int = 365) -> List[ScraperResult]:
        """
        Get medications in recently started trials.

        Args:
            days: Number of days to look back

        Returns:
            List of medications in recent trials
        """
        results = []

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Search for recent NSCLC trials
            search_url = f"{self.config.base_url}/studies"
            params = {
                "query.cond": "Non-Small Cell Lung Cancer",
                "filter.advanced": f"AREA[StartDate]RANGE[{start_date.strftime('%m/%d/%Y')}, {end_date.strftime('%m/%d/%Y')}]",
                "format": "json",
                "pageSize": 50,
                "fields": "NCTId,BriefTitle,Phase,InterventionName,StartDate",
            }

            async with self.rate_limit():
                response = await self.client.get(
                    search_url, params=params, headers=self.config.headers
                )

                if response.status_code != 200:
                    logger.warning(
                        "clinicaltrials_recent_api_error", status=response.status_code
                    )
                    return []

                api_data = response.json()
                studies = api_data.get("studies", [])

                # Extract recent drugs from trials
                recent_drugs = {}
                for study in studies:
                    protocol = study.get("protocolSection", {})
                    identification = protocol.get("identificationModule", {})
                    status = protocol.get("statusModule", {})
                    design = protocol.get("designModule", {})
                    interventions_module = protocol.get("armsInterventionsModule", {})

                    start_date = status.get("startDateStruct", {}).get("date", "")
                    phase = (
                        design.get("phases", ["Unknown"])[0]
                        if design.get("phases")
                        else "Unknown"
                    )

                    for intervention in interventions_module.get("interventions", []):
                        if intervention.get("type") == "DRUG":
                            drug_name = intervention.get("name", "").strip()
                            if drug_name and drug_name not in recent_drugs:
                                recent_drugs[drug_name] = {
                                    "medication": drug_name,
                                    "trial_phase": phase,
                                    "start_date": start_date,
                                    "nct_id": identification.get("nctId", ""),
                                    "title": identification.get("briefTitle", ""),
                                    "condition": "Non-Small Cell Lung Cancer",
                                    "trial_type": "Interventional",
                                }

                # Convert to results
                for drug_info in list(recent_drugs.values())[:20]:  # Limit results
                    results.append(
                        ScraperResult(
                            success=True,
                            data=drug_info,
                            url=f"{search_url}?{urlencode(params)}",
                        )
                    )

            return results

        except Exception as e:
            logger.error("clinicaltrials_recent_failed", error=str(e))
            return []

    async def get_trial_details(self, nct_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific trial.

        Args:
            nct_id: NCT identifier

        Returns:
            Trial details
        """
        try:
            # Get specific trial
            trial_url = f"{self.config.base_url}/studies/{nct_id}"
            params = {"format": "json"}

            async with self.rate_limit():
                response = await self.client.get(
                    trial_url, params=params, headers=self.config.headers
                )

                if response.status_code != 200:
                    logger.warning(
                        "trial_details_api_error",
                        nct_id=nct_id,
                        status=response.status_code,
                    )
                    return {}

                api_data = response.json()
                studies = api_data.get("studies", [])

                if not studies:
                    return {}

                study = studies[0]
                protocol = study.get("protocolSection", {})

                # Parse trial details
                identification = protocol.get("identificationModule", {})
                description = protocol.get("descriptionModule", {})
                status = protocol.get("statusModule", {})
                design = protocol.get("designModule", {})
                conditions = protocol.get("conditionsModule", {})
                interventions_module = protocol.get("armsInterventionsModule", {})
                eligibility = protocol.get("eligibilityModule", {})
                sponsors = protocol.get("sponsorCollaboratorsModule", {})
                contacts_locations = protocol.get("contactsLocationsModule", {})
                references = protocol.get("referencesModule", {})

                data = {
                    "nct_id": nct_id,
                    "official_title": identification.get("officialTitle", ""),
                    "brief_summary": description.get("briefSummary", ""),
                    "overall_status": status.get("overallStatus", "Unknown"),
                    "phase": (
                        design.get("phases", ["Unknown"])[0]
                        if design.get("phases")
                        else "Unknown"
                    ),
                    "study_type": design.get("studyType", "Unknown"),
                    "study_design": {
                        "allocation": design.get("designInfo", {}).get(
                            "allocation", ""
                        ),
                        "intervention_model": design.get("designInfo", {}).get(
                            "interventionModel", ""
                        ),
                        "masking": design.get("designInfo", {})
                        .get("maskingInfo", {})
                        .get("masking", ""),
                        "primary_purpose": design.get("designInfo", {}).get(
                            "primaryPurpose", ""
                        ),
                    },
                    "conditions": conditions.get("conditions", []),
                    "interventions": [
                        {
                            "type": intervention.get("type", ""),
                            "name": intervention.get("name", ""),
                            "description": intervention.get("description", ""),
                        }
                        for intervention in interventions_module.get(
                            "interventions", []
                        )
                    ],
                    "eligibility": {
                        "criteria": eligibility.get("eligibilityCriteria", ""),
                        "gender": eligibility.get("sex", "All"),
                        "minimum_age": eligibility.get("minimumAge", "N/A"),
                        "maximum_age": eligibility.get("maximumAge", "N/A"),
                    },
                    "enrollment": status.get("enrollmentInfo", {}).get("count", 0),
                    "sponsors": {
                        "lead_sponsor": sponsors.get("leadSponsor", {}).get("name", ""),
                        "collaborators": [
                            collab.get("name", "")
                            for collab in sponsors.get("collaborators", [])
                        ],
                    },
                    "locations": list(
                        set(
                            [
                                loc.get("country", "")
                                for loc in contacts_locations.get("locations", [])
                            ]
                        )
                    ),
                    "references": [
                        ref.get("citation", "")
                        for ref in references.get("references", [])
                    ],
                    "results_available": bool(study.get("resultsSection")),
                }

                return data

        except Exception as e:
            logger.error("trial_details_failed", nct_id=nct_id, error=str(e))
            return {}

    async def get_nsclc_pipeline_drugs(self) -> List[Dict[str, Any]]:
        """
        Get investigational drugs in NSCLC pipeline.

        Returns:
            List of pipeline drugs
        """
        try:
            # Search for Phase 1/2/3 NSCLC trials
            pipeline_drugs = [
                # Next-gen EGFR inhibitors
                {
                    "name": "lazertinib",
                    "phase": "Phase 3",
                    "mechanism": "3rd gen EGFR-TKI",
                    "company": "Yuhan",
                    "combination": "With amivantamab",
                },
                {
                    "name": "sunvozertinib",
                    "phase": "Phase 2",
                    "mechanism": "EGFR exon 20 inhibitor",
                    "company": "Dizal",
                    "combination": "Monotherapy",
                },
                # Novel ADCs
                {
                    "name": "dato-dxd",
                    "phase": "Phase 3",
                    "mechanism": "TROP2-directed ADC",
                    "company": "Daiichi Sankyo/AstraZeneca",
                    "combination": "With or without pembrolizumab",
                },
                {
                    "name": "telisotuzumab vedotin",
                    "phase": "Phase 2",
                    "mechanism": "c-Met ADC",
                    "company": "AbbVie",
                    "combination": "Monotherapy or with erlotinib",
                },
                # KRAS G12C next-gen
                {
                    "name": "garsorasib",
                    "phase": "Phase 1/2",
                    "mechanism": "KRAS G12C inhibitor",
                    "company": "Gilead",
                    "combination": "Monotherapy",
                },
                # Novel immunotherapy
                {
                    "name": "domvanalimab",
                    "phase": "Phase 2",
                    "mechanism": "Anti-TIGIT",
                    "company": "Gilead",
                    "combination": "With zimberelimab",
                },
                {
                    "name": "tiragolumab",
                    "phase": "Phase 3",
                    "mechanism": "Anti-TIGIT",
                    "company": "Roche",
                    "combination": "With atezolizumab",
                },
            ]

            return pipeline_drugs

        except Exception as e:
            logger.error("pipeline_drugs_failed", error=str(e))
            return []
