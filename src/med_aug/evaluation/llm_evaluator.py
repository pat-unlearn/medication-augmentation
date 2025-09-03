"""LLM-assisted evaluation for medication classification quality assessment."""

from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass

from ..llm.service import LLMService
from .ground_truth import GroundTruthEntry
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMEvaluationResult:
    """Result of LLM evaluation for a medication classification."""

    medication: str
    predicted_class: str
    llm_assessment: str  # "correct", "incorrect", "ambiguous", "unknown"
    confidence: float
    reasoning: str
    alternative_classes: List[str] = None
    clinical_notes: Optional[str] = None


class LLMAssistedEvaluator:
    """Uses LLM to assist in evaluating medication classifications."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize with LLM service."""
        self.llm = llm_service or LLMService()

        # Evaluation prompts
        self.evaluation_prompt_template = """You are a clinical pharmacist and medication expert. Your task is to evaluate whether a medication has been correctly classified into a drug class for NSCLC (Non-Small Cell Lung Cancer) treatment.

**Medication to Evaluate:** {medication}
**Predicted Drug Class:** {predicted_class}
**Context:** {context}

Please evaluate this classification and respond with a JSON object:

{{
  "assessment": "correct|incorrect|ambiguous|unknown",
  "confidence": 0.95,
  "reasoning": "Detailed explanation of your assessment",
  "alternative_classes": ["alternative1", "alternative2"],
  "clinical_notes": "Any additional clinical context or notes"
}}

**Assessment Guidelines:**
- "correct": The medication clearly belongs to the predicted drug class
- "incorrect": The medication clearly does NOT belong to the predicted drug class
- "ambiguous": The medication could reasonably belong to multiple classes
- "unknown": You cannot determine the classification (insufficient information)

**Drug Classes for Reference:**
{drug_classes}

Focus on NSCLC treatment context. Consider:
1. Mechanism of action
2. Clinical indications
3. FDA approvals for lung cancer
4. Standard treatment protocols
5. Drug naming conventions (-mab, -nib, -platin, etc.)"""

        self.batch_evaluation_prompt = """You are evaluating multiple medication classifications for NSCLC treatment. For each medication, assess whether it's correctly classified.

**Medications to Evaluate:**
{medication_list}

**Available Drug Classes:**
{drug_classes}

For each medication, respond with a JSON array of objects:
[
  {{
    "medication": "drug_name",
    "predicted_class": "class_name",
    "assessment": "correct|incorrect|ambiguous|unknown",
    "confidence": 0.95,
    "reasoning": "Brief explanation",
    "alternative_classes": []
  }}
]

Focus on NSCLC context and clinical accuracy."""

    async def evaluate_classifications(
        self,
        classifications: Dict[str, str],  # medication -> predicted_class
        confidence_scores: Optional[Dict[str, float]] = None,
        batch_size: int = 10,
        focus_on_ambiguous: bool = True,
    ) -> List[LLMEvaluationResult]:
        """
        Use LLM to evaluate medication classifications.

        Args:
            classifications: Medication -> drug class predictions
            confidence_scores: Confidence scores for predictions
            batch_size: Number of medications to evaluate per LLM call
            focus_on_ambiguous: Whether to prioritize low-confidence predictions

        Returns:
            List of LLM evaluation results
        """
        logger.info(
            "llm_evaluation_started", total_classifications=len(classifications)
        )

        # Prioritize evaluations
        sorted_medications = self._prioritize_evaluations(
            classifications, confidence_scores, focus_on_ambiguous
        )

        # Get drug class context
        drug_classes_info = await self._get_drug_class_context()

        all_results = []

        # Process in batches
        for i in range(0, len(sorted_medications), batch_size):
            batch = sorted_medications[i : i + batch_size]

            try:
                batch_results = await self._evaluate_batch(
                    batch, classifications, drug_classes_info
                )
                all_results.extend(batch_results)

                logger.info(
                    "batch_evaluated",
                    batch_size=len(batch),
                    total_processed=len(all_results),
                )

            except Exception as e:
                logger.error("batch_evaluation_failed", batch_start=i, error=str(e))
                # Continue with next batch
                continue

        logger.info("llm_evaluation_completed", total_results=len(all_results))
        return all_results

    async def evaluate_false_positives(
        self,
        false_positives: Dict[str, List[str]],  # drug_class -> [medications]
        max_per_class: int = 5,
    ) -> Dict[str, List[LLMEvaluationResult]]:
        """
        Evaluate false positive classifications to identify potential valid discoveries.

        This is crucial for expanding our ground truth with legitimately correct
        classifications that weren't in the original dataset.
        """
        logger.info(
            "evaluating_false_positives",
            classes=len(false_positives),
            total_fps=sum(len(meds) for meds in false_positives.values()),
        )

        drug_classes_info = await self._get_drug_class_context()
        fp_evaluations = {}

        for drug_class, medications in false_positives.items():
            # Limit evaluations per class to manage API costs
            meds_to_evaluate = medications[:max_per_class]

            evaluations = []
            for medication in meds_to_evaluate:
                try:
                    result = await self._evaluate_single_classification(
                        medication, drug_class, drug_classes_info
                    )
                    evaluations.append(result)

                except Exception as e:
                    logger.error(
                        "fp_evaluation_failed",
                        medication=medication,
                        drug_class=drug_class,
                        error=str(e),
                    )

            fp_evaluations[drug_class] = evaluations

        return fp_evaluations

    async def suggest_ground_truth_updates(
        self, llm_evaluations: List[LLMEvaluationResult], min_confidence: float = 0.8
    ) -> List[GroundTruthEntry]:
        """
        Suggest new ground truth entries based on LLM evaluations.

        This helps expand our ground truth dataset with high-confidence LLM validations.
        """
        suggestions = []

        for result in llm_evaluations:
            if (
                result.llm_assessment == "correct"
                and result.confidence >= min_confidence
            ):
                # This was marked as a false positive but LLM says it's correct
                suggestion = GroundTruthEntry(
                    medication=result.medication,
                    drug_class=result.predicted_class,
                    confidence=result.confidence,
                    source="llm_validation",
                    notes=f"LLM Assessment: {result.reasoning}",
                    validated_by="claude_llm",
                )
                suggestions.append(suggestion)

        logger.info("ground_truth_suggestions", count=len(suggestions))
        return suggestions

    async def evaluate_ambiguous_cases(
        self,
        ambiguous_medications: List[str],
        possible_classes: Dict[str, List[str]],  # medication -> [possible_classes]
    ) -> Dict[str, LLMEvaluationResult]:
        """Evaluate medications with ambiguous classifications."""
        results = {}
        drug_classes_info = await self._get_drug_class_context()

        for medication in ambiguous_medications:
            classes = possible_classes.get(medication, [])

            if len(classes) > 1:
                # Ask LLM to choose between multiple valid options
                context = f"This medication could belong to multiple classes: {', '.join(classes)}"

                try:
                    result = await self._evaluate_with_context(
                        medication, classes[0], context, drug_classes_info
                    )
                    results[medication] = result

                except Exception as e:
                    logger.error(
                        "ambiguous_evaluation_failed",
                        medication=medication,
                        error=str(e),
                    )

        return results

    def _prioritize_evaluations(
        self,
        classifications: Dict[str, str],
        confidence_scores: Optional[Dict[str, float]],
        focus_on_ambiguous: bool,
    ) -> List[str]:
        """Prioritize which medications to evaluate first."""
        medications = list(classifications.keys())

        if confidence_scores and focus_on_ambiguous:
            # Sort by confidence (lowest first for ambiguous cases)
            medications.sort(key=lambda med: confidence_scores.get(med, 0.5))

        return medications

    async def _get_drug_class_context(self) -> str:
        """Get formatted drug class information for LLM context."""
        from ..diseases import disease_registry

        try:
            nsclc_module = disease_registry.get_module("nsclc")
            if not nsclc_module:
                return "Standard NSCLC drug classes"

            class_info = []
            for drug_class in nsclc_module.drug_classes:
                keywords_str = ", ".join(drug_class.keywords[:3])  # First 3 keywords
                class_info.append(
                    f"- {drug_class.name}: {drug_class.description} (Keywords: {keywords_str})"
                )

            return "\\n".join(class_info)

        except Exception as e:
            logger.error("drug_class_context_failed", error=str(e))
            return "Standard NSCLC drug classes for lung cancer treatment"

    async def _evaluate_batch(
        self,
        medications: List[str],
        classifications: Dict[str, str],
        drug_classes_info: str,
    ) -> List[LLMEvaluationResult]:
        """Evaluate a batch of medications."""
        medication_list = []
        for med in medications:
            drug_class = classifications[med]
            medication_list.append(f"- {med} → {drug_class}")

        prompt = self.batch_evaluation_prompt.format(
            medication_list="\\n".join(medication_list), drug_classes=drug_classes_info
        )

        try:
            response = await self.llm.generate(prompt)

            # Parse JSON response
            evaluations_data = json.loads(response.content)

            results = []
            for eval_data in evaluations_data:
                result = LLMEvaluationResult(
                    medication=eval_data["medication"],
                    predicted_class=eval_data["predicted_class"],
                    llm_assessment=eval_data["assessment"],
                    confidence=eval_data["confidence"],
                    reasoning=eval_data["reasoning"],
                    alternative_classes=eval_data.get("alternative_classes", []),
                )
                results.append(result)

            return results

        except json.JSONDecodeError as e:
            logger.error("llm_response_parse_failed", error=str(e))
            return []
        except Exception as e:
            logger.error("batch_evaluation_error", error=str(e))
            return []

    async def _evaluate_single_classification(
        self,
        medication: str,
        predicted_class: str,
        drug_classes_info: str,
        context: str = "",
    ) -> LLMEvaluationResult:
        """Evaluate a single medication classification."""
        prompt = self.evaluation_prompt_template.format(
            medication=medication,
            predicted_class=predicted_class,
            context=context,
            drug_classes=drug_classes_info,
        )

        response = await self.llm.generate(prompt)

        try:
            eval_data = json.loads(response.content)

            return LLMEvaluationResult(
                medication=medication,
                predicted_class=predicted_class,
                llm_assessment=eval_data["assessment"],
                confidence=eval_data["confidence"],
                reasoning=eval_data["reasoning"],
                alternative_classes=eval_data.get("alternative_classes", []),
                clinical_notes=eval_data.get("clinical_notes"),
            )

        except json.JSONDecodeError:
            # Fallback: treat as unknown if can't parse
            return LLMEvaluationResult(
                medication=medication,
                predicted_class=predicted_class,
                llm_assessment="unknown",
                confidence=0.0,
                reasoning=f"Could not parse LLM response: {response.content[:100]}",
            )

    async def _evaluate_with_context(
        self,
        medication: str,
        predicted_class: str,
        context: str,
        drug_classes_info: str,
    ) -> LLMEvaluationResult:
        """Evaluate with additional context."""
        return await self._evaluate_single_classification(
            medication, predicted_class, drug_classes_info, context
        )

    def create_evaluation_summary(
        self, llm_evaluations: List[LLMEvaluationResult]
    ) -> Dict[str, Any]:
        """Create summary of LLM evaluation results."""
        total = len(llm_evaluations)
        if total == 0:
            return {}

        assessments = [r.llm_assessment for r in llm_evaluations]
        assessment_counts = {
            "correct": assessments.count("correct"),
            "incorrect": assessments.count("incorrect"),
            "ambiguous": assessments.count("ambiguous"),
            "unknown": assessments.count("unknown"),
        }

        avg_confidence = sum(r.confidence for r in llm_evaluations) / total

        # Find high-confidence corrections (false positives that are actually correct)
        potential_ground_truth = [
            r
            for r in llm_evaluations
            if r.llm_assessment == "correct" and r.confidence > 0.8
        ]

        # Find clear errors needing correction
        clear_errors = [
            r
            for r in llm_evaluations
            if r.llm_assessment == "incorrect" and r.confidence > 0.8
        ]

        summary = {
            "total_evaluated": total,
            "assessment_distribution": assessment_counts,
            "assessment_percentages": {
                k: round(v / total * 100, 1) for k, v in assessment_counts.items()
            },
            "average_confidence": round(avg_confidence, 3),
            "potential_ground_truth_additions": len(potential_ground_truth),
            "clear_errors_found": len(clear_errors),
            "top_alternative_suggestions": self._get_top_alternatives(llm_evaluations),
            "clinical_insights": self._extract_clinical_insights(llm_evaluations),
        }

        return summary

    def _get_top_alternatives(
        self, evaluations: List[LLMEvaluationResult]
    ) -> List[Dict[str, Any]]:
        """Extract top alternative class suggestions."""
        alternatives = []

        for result in evaluations:
            if (
                result.llm_assessment == "incorrect"
                and result.alternative_classes
                and result.confidence > 0.7
            ):
                alternatives.append(
                    {
                        "medication": result.medication,
                        "current_class": result.predicted_class,
                        "suggested_classes": result.alternative_classes,
                        "reasoning": result.reasoning,
                    }
                )

        return alternatives[:10]  # Top 10 suggestions

    def _extract_clinical_insights(
        self, evaluations: List[LLMEvaluationResult]
    ) -> List[str]:
        """Extract key clinical insights from LLM reasoning."""
        # This could be enhanced with more sophisticated text analysis
        # For now, just collect unique clinical notes
        clinical_notes = set()
        for result in evaluations:
            if result.clinical_notes:
                clinical_notes.add(result.clinical_notes)

        return list(clinical_notes)[:5]  # Top 5 unique insights
