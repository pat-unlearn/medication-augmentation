"""Pipeline phase for evaluating medication classification quality."""

from typing import Dict, List, Any
from datetime import datetime

from .phases import PipelinePhase, PhaseResult
from ..evaluation import MedicationEvaluator, GroundTruthManager, ClassificationReport
from ..evaluation.llm_evaluator import LLMAssistedEvaluator, LLMEvaluationResult
from ..core.logging import get_logger, get_performance_logger

logger = get_logger(__name__)
perf_logger = get_performance_logger(__name__)


class EvaluationPhase(PipelinePhase):
    """Phase for evaluating classification quality and identifying false positives/negatives."""

    def __init__(self):
        """Initialize evaluation phase."""
        super().__init__("evaluation", required=False)
        self.dependencies = ["llm_classification", "validation"]

        # Initialize evaluators
        self.ground_truth_manager = GroundTruthManager()
        self.evaluator = MedicationEvaluator(self.ground_truth_manager)
        self.llm_evaluator = LLMAssistedEvaluator()

    async def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Check if we have classifications to evaluate."""
        return (
            context.get("llm_classifications") is not None
            or context.get("classification_results") is not None
        )

    async def execute(self, context: Dict[str, Any]) -> PhaseResult:
        """Evaluate classification quality with LLM assistance."""
        logger.info("evaluation_phase_started")
        perf_logger.start_operation("evaluation")

        start_time = datetime.now()

        try:
            # Get classifications from context
            classifications = self._extract_classifications(context)
            confidence_scores = context.get("llm_confidence_scores", {})

            if not classifications:
                logger.warning("no_classifications_to_evaluate")
                return self._create_empty_result()

            logger.info(
                "evaluating_classifications",
                total_classifications=len(classifications),
                has_confidence_scores=len(confidence_scores) > 0,
            )

            # Step 1: Standard evaluation against ground truth
            standard_report = await self._run_standard_evaluation(
                classifications, confidence_scores
            )

            # Step 2: LLM-assisted evaluation of problematic cases
            llm_evaluations = await self._run_llm_evaluation(
                standard_report, classifications, confidence_scores
            )

            # Step 3: Generate actionable insights
            insights = await self._generate_insights(standard_report, llm_evaluations)

            # Step 4: Update ground truth with high-confidence LLM validations
            ground_truth_updates = await self._suggest_ground_truth_updates(
                llm_evaluations
            )

            # Prepare results
            evaluation_results = {
                "standard_evaluation": standard_report.to_dict(),
                "llm_evaluation_summary": self.llm_evaluator.create_evaluation_summary(
                    llm_evaluations
                ),
                "actionable_insights": insights,
                "ground_truth_suggestions": [
                    {
                        "medication": entry.medication,
                        "drug_class": entry.drug_class,
                        "confidence": entry.confidence,
                        "reasoning": entry.notes,
                    }
                    for entry in ground_truth_updates
                ],
                "evaluation_metrics": {
                    "total_classifications": len(classifications),
                    "evaluation_coverage": (
                        len(llm_evaluations) / len(classifications)
                        if classifications
                        else 0
                    ),
                    "false_positive_rate": self._calculate_fp_rate(standard_report),
                    "false_negative_rate": self._calculate_fn_rate(standard_report),
                    "precision": standard_report.metrics.weighted_precision,
                    "recall": standard_report.metrics.weighted_recall,
                    "f1_score": standard_report.metrics.weighted_f1,
                },
            }

            # Store results in context for other phases
            context["evaluation_results"] = evaluation_results
            context["classification_report"] = standard_report
            context["llm_evaluations"] = llm_evaluations

            duration = perf_logger.end_operation(
                "evaluation",
                classifications_evaluated=len(classifications),
                llm_evaluations_completed=len(llm_evaluations),
                f1_score=standard_report.metrics.weighted_f1,
            )

            logger.info(
                "evaluation_phase_completed",
                duration=duration,
                f1_score=standard_report.metrics.weighted_f1,
                precision=standard_report.metrics.weighted_precision,
                recall=standard_report.metrics.weighted_recall,
            )

            return PhaseResult(
                phase_name="evaluation",
                success=True,
                data=evaluation_results,
                metrics={
                    "classifications_evaluated": len(classifications),
                    "f1_score": standard_report.metrics.weighted_f1,
                    "precision": standard_report.metrics.weighted_precision,
                    "recall": standard_report.metrics.weighted_recall,
                    "false_positive_rate": evaluation_results["evaluation_metrics"][
                        "false_positive_rate"
                    ],
                    "false_negative_rate": evaluation_results["evaluation_metrics"][
                        "false_negative_rate"
                    ],
                    "ground_truth_suggestions": len(ground_truth_updates),
                },
                execution_time=(datetime.now() - start_time).total_seconds(),
                artifacts={
                    "evaluation_report.json": evaluation_results,
                    "classification_errors.json": {
                        "false_positives": standard_report.false_positives,
                        "false_negatives": standard_report.false_negatives,
                        "misclassifications": standard_report.misclassifications,
                    },
                },
            )

        except Exception as e:
            duration = perf_logger.end_operation("evaluation", error=str(e))
            logger.error("evaluation_phase_failed", error=str(e), duration=duration)

            return PhaseResult(
                phase_name="evaluation",
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    def _extract_classifications(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Extract medication classifications from context."""
        # Try LLM classifications first
        llm_classifications = context.get("llm_classifications", {})
        if llm_classifications:
            # Convert from drug_class -> [medications] to medication -> drug_class
            classifications = {}
            for drug_class, medications in llm_classifications.items():
                for medication in medications:
                    classifications[medication] = drug_class
            return classifications

        # Fall back to validation results
        classification_results = context.get("classification_results", {})
        if classification_results:
            return {
                med: result["drug_class"]
                for med, result in classification_results.items()
                if result.get("valid", False)
            }

        return {}

    async def _run_standard_evaluation(
        self, classifications: Dict[str, str], confidence_scores: Dict[str, float]
    ) -> ClassificationReport:
        """Run standard precision/recall evaluation against ground truth."""
        logger.info("running_standard_evaluation")

        report = self.evaluator.evaluate_classification_results(
            predictions=classifications,
            confidence_scores=confidence_scores,
            minimum_confidence=0.5,
        )

        logger.info(
            "standard_evaluation_completed",
            macro_f1=report.metrics.macro_f1,
            weighted_f1=report.metrics.weighted_f1,
            total_fp=sum(len(fps) for fps in report.false_positives.values()),
            total_fn=sum(len(fns) for fns in report.false_negatives.values()),
        )

        return report

    async def _run_llm_evaluation(
        self,
        standard_report: ClassificationReport,
        classifications: Dict[str, str],
        confidence_scores: Dict[str, float],
    ) -> List[LLMEvaluationResult]:
        """Run LLM-assisted evaluation focusing on problematic cases."""
        logger.info("running_llm_evaluation")

        # Focus on false positives (potential ground truth expansions)
        fp_evaluations = await self.llm_evaluator.evaluate_false_positives(
            standard_report.false_positives,
            max_per_class=5,  # Limit for cost control
        )

        # Flatten results
        all_llm_evaluations = []
        for drug_class, evaluations in fp_evaluations.items():
            all_llm_evaluations.extend(evaluations)

        # Also evaluate some low-confidence classifications
        low_conf_meds = [
            med
            for med, conf in confidence_scores.items()
            if 0.3 <= conf <= 0.7  # Ambiguous range
        ][:10]  # Limit for cost

        if low_conf_meds:
            low_conf_classifications = {
                med: classifications[med]
                for med in low_conf_meds
                if med in classifications
            }

            additional_evaluations = await self.llm_evaluator.evaluate_classifications(
                low_conf_classifications,
                confidence_scores,
                batch_size=5,
                focus_on_ambiguous=True,
            )
            all_llm_evaluations.extend(additional_evaluations)

        logger.info(
            "llm_evaluation_completed", total_evaluations=len(all_llm_evaluations)
        )

        return all_llm_evaluations

    async def _generate_insights(
        self,
        standard_report: ClassificationReport,
        llm_evaluations: List[LLMEvaluationResult],
    ) -> Dict[str, Any]:
        """Generate actionable insights combining standard and LLM evaluation."""
        insights = {
            "priority_fixes": [],
            "ground_truth_expansion_opportunities": [],
            "classification_improvements": [],
            "coverage_gaps": [],
            "quality_summary": {},
        }

        # Priority fixes from standard evaluation
        problematic_classes = standard_report.metrics.get_problematic_classes(
            min_f1=0.7
        )
        for drug_class in problematic_classes[:5]:
            metrics = standard_report.metrics.class_metrics[drug_class]
            insights["priority_fixes"].append(
                {
                    "drug_class": drug_class,
                    "issue": f"Low F1 score: {metrics.f1_score:.3f}",
                    "recommendation": f"Review {metrics.false_negatives} false negatives and {metrics.false_positives} false positives",
                }
            )

        # Ground truth expansion from LLM evaluation
        for llm_eval in llm_evaluations:
            if llm_eval.llm_assessment == "correct" and llm_eval.confidence > 0.8:
                insights["ground_truth_expansion_opportunities"].append(
                    {
                        "medication": llm_eval.medication,
                        "drug_class": llm_eval.predicted_class,
                        "llm_confidence": llm_eval.confidence,
                        "reasoning": llm_eval.reasoning,
                    }
                )

        # Classification improvements from LLM suggestions
        for llm_eval in llm_evaluations:
            if (
                llm_eval.llm_assessment == "incorrect"
                and llm_eval.alternative_classes
                and llm_eval.confidence > 0.7
            ):
                insights["classification_improvements"].append(
                    {
                        "medication": llm_eval.medication,
                        "current_class": llm_eval.predicted_class,
                        "suggested_class": llm_eval.alternative_classes[0],
                        "reasoning": llm_eval.reasoning,
                    }
                )

        # Coverage gaps
        if standard_report.unclassified_medications:
            insights["coverage_gaps"] = [
                {
                    "medication": med,
                    "recommendation": "Add classification rules or include in ground truth",
                }
                for med in standard_report.unclassified_medications[:10]
            ]

        # Quality summary
        insights["quality_summary"] = {
            "overall_f1": standard_report.metrics.weighted_f1,
            "precision": standard_report.metrics.weighted_precision,
            "recall": standard_report.metrics.weighted_recall,
            "coverage": standard_report.metrics.classification_coverage,
            "llm_validation_rate": (
                len([e for e in llm_evaluations if e.llm_assessment == "correct"])
                / len(llm_evaluations)
                if llm_evaluations
                else 0
            ),
            "potential_ground_truth_additions": len(
                insights["ground_truth_expansion_opportunities"]
            ),
        }

        return insights

    async def _suggest_ground_truth_updates(
        self, llm_evaluations: List[LLMEvaluationResult]
    ) -> List:
        """Suggest ground truth updates based on LLM evaluation."""
        return await self.llm_evaluator.suggest_ground_truth_updates(
            llm_evaluations, min_confidence=0.8
        )

    def _calculate_fp_rate(self, report: ClassificationReport) -> float:
        """Calculate false positive rate."""
        total_fp = sum(len(fps) for fps in report.false_positives.values())
        total_predictions = report.metrics.total_classified
        return total_fp / total_predictions if total_predictions > 0 else 0.0

    def _calculate_fn_rate(self, report: ClassificationReport) -> float:
        """Calculate false negative rate."""
        total_fn = sum(len(fns) for fns in report.false_negatives.values())
        total_actual = sum(m.support for m in report.metrics.class_metrics.values())
        return total_fn / total_actual if total_actual > 0 else 0.0

    def _create_empty_result(self) -> PhaseResult:
        """Create empty result when no classifications available."""
        return PhaseResult(
            phase_name="evaluation",
            success=True,
            data={"message": "No classifications available for evaluation"},
            metrics={},
            execution_time=0.0,
        )
