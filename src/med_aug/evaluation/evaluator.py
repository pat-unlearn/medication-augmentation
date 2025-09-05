"""Main evaluation engine for medication classification."""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter

from .metrics import EvaluationMetrics, ClassificationMetrics, ClassificationReport
from .ground_truth import GroundTruthManager
from ..core.logging import get_logger

logger = get_logger(__name__)


class MedicationEvaluator:
    """Evaluates medication classification results against ground truth."""

    def __init__(self, ground_truth_manager: Optional[GroundTruthManager] = None):
        """Initialize evaluator with ground truth data."""
        self.ground_truth = ground_truth_manager or GroundTruthManager()

    def evaluate_classification_results(
        self,
        predictions: Dict[str, str],  # medication -> predicted_drug_class
        confidence_scores: Optional[Dict[str, float]] = None,
        minimum_confidence: float = 0.5,
    ) -> ClassificationReport:
        """
        Evaluate classification predictions against ground truth.

        Args:
            predictions: Dictionary mapping medication names to predicted drug classes
            confidence_scores: Optional confidence scores for each prediction
            minimum_confidence: Minimum confidence to consider a classification valid

        Returns:
            Detailed classification report with metrics and error analysis
        """
        logger.info("evaluation_started", total_predictions=len(predictions))

        # Load ground truth data
        ground_truth = self.ground_truth.get_ground_truth_mappings()

        if not ground_truth:
            logger.warning(
                "no_ground_truth_data", message="No ground truth data available"
            )
            return self._create_empty_report(predictions)

        # Filter predictions by confidence if provided
        if confidence_scores:
            high_confidence_predictions = {
                med: drug_class
                for med, drug_class in predictions.items()
                if confidence_scores.get(med, 0.0) >= minimum_confidence
            }
            low_confidence_predictions = {
                med: drug_class
                for med, drug_class in predictions.items()
                if confidence_scores.get(med, 0.0) < minimum_confidence
            }
            logger.info(
                "confidence_filtering",
                high_confidence=len(high_confidence_predictions),
                low_confidence=len(low_confidence_predictions),
            )
        else:
            high_confidence_predictions = predictions
            low_confidence_predictions = {}

        # Calculate metrics for each drug class
        class_metrics = self._calculate_class_metrics(
            high_confidence_predictions, ground_truth
        )

        # Identify specific errors
        false_positives, false_negatives, misclassifications = (
            self._identify_classification_errors(
                high_confidence_predictions, ground_truth
            )
        )

        # Find unclassified medications
        unclassified = self._find_unclassified_medications(
            predictions, ground_truth, confidence_scores, minimum_confidence
        )

        # Create overall metrics
        total_medications = len(ground_truth)
        total_classified = len(high_confidence_predictions)
        total_unclassified = len(unclassified)

        metrics = EvaluationMetrics(
            class_metrics=class_metrics,
            total_medications=total_medications,
            total_classified=total_classified,
            total_unclassified=total_unclassified,
        )

        # Create detailed report
        report = ClassificationReport(
            metrics=metrics,
            false_positives=false_positives,
            false_negatives=false_negatives,
            unclassified_medications=unclassified,
            misclassifications=misclassifications,
        )

        logger.info(
            "evaluation_completed",
            macro_f1=metrics.macro_f1,
            weighted_f1=metrics.weighted_f1,
            coverage=metrics.classification_coverage,
        )

        return report

    def _calculate_class_metrics(
        self, predictions: Dict[str, str], ground_truth: Dict[str, str]
    ) -> Dict[str, ClassificationMetrics]:
        """Calculate precision, recall, F1 for each drug class."""
        # Get all drug classes from both predictions and ground truth
        all_classes = set(predictions.values()) | set(ground_truth.values())

        class_metrics = {}

        for drug_class in all_classes:
            # Get medications for this class
            predicted_meds = {
                med for med, cls in predictions.items() if cls == drug_class
            }
            actual_meds = {
                med for med, cls in ground_truth.items() if cls == drug_class
            }

            # Calculate confusion matrix elements
            true_positives = len(predicted_meds & actual_meds)
            false_positives = len(predicted_meds - actual_meds)
            false_negatives = len(actual_meds - predicted_meds)

            metrics = ClassificationMetrics(
                drug_class=drug_class,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                total_predictions=len(predicted_meds),
                total_ground_truth=len(actual_meds),
            )

            class_metrics[drug_class] = metrics

        return class_metrics

    def _identify_classification_errors(
        self, predictions: Dict[str, str], ground_truth: Dict[str, str]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[Dict[str, str]]]:
        """Identify specific false positives, false negatives, and misclassifications."""
        false_positives = defaultdict(list)
        false_negatives = defaultdict(list)
        misclassifications = []

        # Get all medications from both sets
        all_medications = set(predictions.keys()) | set(ground_truth.keys())

        for medication in all_medications:
            predicted_class = predictions.get(medication)
            actual_class = ground_truth.get(medication)

            if actual_class is None:
                # Medication not in ground truth - could be new discovery or false positive
                if predicted_class:
                    false_positives[predicted_class].append(medication)

            elif predicted_class is None:
                # Medication in ground truth but not predicted - false negative
                false_negatives[actual_class].append(medication)

            elif predicted_class != actual_class:
                # Medication misclassified
                false_negatives[actual_class].append(medication)
                false_positives[predicted_class].append(medication)
                misclassifications.append(
                    {
                        "medication": medication,
                        "predicted": predicted_class,
                        "actual": actual_class,
                    }
                )

        return dict(false_positives), dict(false_negatives), misclassifications

    def _find_unclassified_medications(
        self,
        all_predictions: Dict[str, str],
        ground_truth: Dict[str, str],
        confidence_scores: Optional[Dict[str, float]],
        minimum_confidence: float,
    ) -> List[str]:
        """Find medications that should have been classified but weren't."""
        unclassified = []

        for medication in ground_truth.keys():
            # Check if medication was predicted
            if medication not in all_predictions:
                unclassified.append(medication)
            # Check if prediction was below confidence threshold
            elif (
                confidence_scores
                and confidence_scores.get(medication, 0.0) < minimum_confidence
            ):
                unclassified.append(medication)

        return unclassified

    def _create_empty_report(self, predictions: Dict[str, str]) -> ClassificationReport:
        """Create an empty report when no ground truth is available."""
        empty_metrics = EvaluationMetrics(
            class_metrics={},
            total_medications=len(predictions),
            total_classified=len(predictions),
            total_unclassified=0,
        )

        return ClassificationReport(
            metrics=empty_metrics,
            false_positives={},
            false_negatives={},
            unclassified_medications=[],
            misclassifications=[],
        )

    def compare_before_after(
        self,
        before_predictions: Dict[str, str],
        after_predictions: Dict[str, str],
        confidence_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compare classification results before and after augmentation.

        Returns:
            Dictionary with improvement metrics and analysis
        """
        logger.info("comparison_started")

        before_report = self.evaluate_classification_results(before_predictions)
        after_report = self.evaluate_classification_results(
            after_predictions, confidence_scores
        )

        comparison = {
            "before_metrics": before_report.metrics.to_dict()["overall_metrics"],
            "after_metrics": after_report.metrics.to_dict()["overall_metrics"],
            "improvements": {},
            "new_classifications": {},
            "lost_classifications": {},
            "quality_changes": {},
        }

        # Calculate improvements
        before_m = before_report.metrics
        after_m = after_report.metrics

        comparison["improvements"] = {
            "precision_change": after_m.weighted_precision
            - before_m.weighted_precision,
            "recall_change": after_m.weighted_recall - before_m.weighted_recall,
            "f1_change": after_m.weighted_f1 - before_m.weighted_f1,
            "coverage_change": after_m.classification_coverage
            - before_m.classification_coverage,
            "new_medications_classified": after_m.total_classified
            - before_m.total_classified,
        }

        # Find new and lost classifications
        before_meds = set(before_predictions.keys())
        after_meds = set(after_predictions.keys())

        new_meds = after_meds - before_meds
        lost_meds = before_meds - after_meds

        comparison["new_classifications"] = {
            "count": len(new_meds),
            "medications": list(new_meds)[:10],  # Show first 10
            "by_class": Counter([after_predictions[med] for med in new_meds]),
        }

        comparison["lost_classifications"] = {
            "count": len(lost_meds),
            "medications": list(lost_meds)[:10],
            "by_class": Counter([before_predictions[med] for med in lost_meds]),
        }

        # Quality changes for overlapping medications
        common_meds = before_meds & after_meds
        quality_changes = {"improved": [], "degraded": [], "unchanged": 0}

        ground_truth = self.ground_truth.get_ground_truth_mappings()

        for med in common_meds:
            before_class = before_predictions[med]
            after_class = after_predictions[med]
            actual_class = ground_truth.get(med)

            if actual_class:
                before_correct = before_class == actual_class
                after_correct = after_class == actual_class

                if not before_correct and after_correct:
                    quality_changes["improved"].append(med)
                elif before_correct and not after_correct:
                    quality_changes["degraded"].append(med)
                else:
                    quality_changes["unchanged"] += 1

        comparison["quality_changes"] = quality_changes

        logger.info(
            "comparison_completed",
            f1_improvement=comparison["improvements"]["f1_change"],
            coverage_improvement=comparison["improvements"]["coverage_change"],
        )

        return comparison

    def generate_validation_suggestions(
        self, report: ClassificationReport, top_n: int = 10
    ) -> Dict[str, List[str]]:
        """Generate specific suggestions for improving classification accuracy."""
        suggestions = {
            "add_to_ground_truth": [],
            "review_false_positives": [],
            "investigate_patterns": [],
            "improve_coverage": [],
        }

        # Suggest adding high-confidence correct predictions to ground truth
        ground_truth = self.ground_truth.get_ground_truth_mappings()

        # Look for patterns in false positives that might indicate new valid medications
        fp_by_class = report.false_positives
        for drug_class, medications in fp_by_class.items():
            if len(medications) > 3:  # Pattern threshold
                suggestions["investigate_patterns"].append(
                    f"Multiple false positives in {drug_class}: {', '.join(medications[:3])}..."
                )

        # Suggest reviewing most common false positives
        all_fps = [med for meds in report.false_positives.values() for med in meds]
        fp_counts = Counter(all_fps)

        for medication, count in fp_counts.most_common(top_n):
            if count > 1:  # Appeared as FP in multiple classes
                suggestions["review_false_positives"].append(
                    f"'{medication}' incorrectly classified {count} times - review validity"
                )

        # Suggest improvements for coverage
        if report.unclassified_medications:
            unclass_sample = report.unclassified_medications[:top_n]
            suggestions["improve_coverage"].extend(
                [f"Add classification rules for: {med}" for med in unclass_sample]
            )

        return suggestions
