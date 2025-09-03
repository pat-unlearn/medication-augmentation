"""Evaluation metrics for medication classification."""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
import json
from ..core.mixins import DictMixin


@dataclass
class ClassificationMetrics:
    """Metrics for a single drug class."""

    drug_class: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_predictions: int = 0
    total_ground_truth: int = 0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def support(self) -> int:
        """Number of actual instances in ground truth."""
        return self.total_ground_truth

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "drug_class": self.drug_class,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "support": self.support,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class EvaluationMetrics:
    """Overall evaluation metrics for the medication classification system."""

    class_metrics: Dict[str, ClassificationMetrics]
    total_medications: int
    total_classified: int
    total_unclassified: int

    @property
    def macro_precision(self) -> float:
        """Average precision across all classes."""
        if not self.class_metrics:
            return 0.0
        return sum(m.precision for m in self.class_metrics.values()) / len(
            self.class_metrics
        )

    @property
    def macro_recall(self) -> float:
        """Average recall across all classes."""
        if not self.class_metrics:
            return 0.0
        return sum(m.recall for m in self.class_metrics.values()) / len(
            self.class_metrics
        )

    @property
    def macro_f1(self) -> float:
        """Average F1 across all classes."""
        if not self.class_metrics:
            return 0.0
        return sum(m.f1_score for m in self.class_metrics.values()) / len(
            self.class_metrics
        )

    @property
    def weighted_precision(self) -> float:
        """Precision weighted by support (ground truth counts)."""
        if not self.class_metrics:
            return 0.0

        total_support = sum(m.support for m in self.class_metrics.values())
        if total_support == 0:
            return 0.0

        weighted_sum = sum(m.precision * m.support for m in self.class_metrics.values())
        return weighted_sum / total_support

    @property
    def weighted_recall(self) -> float:
        """Recall weighted by support."""
        if not self.class_metrics:
            return 0.0

        total_support = sum(m.support for m in self.class_metrics.values())
        if total_support == 0:
            return 0.0

        weighted_sum = sum(m.recall * m.support for m in self.class_metrics.values())
        return weighted_sum / total_support

    @property
    def weighted_f1(self) -> float:
        """F1 weighted by support."""
        if not self.class_metrics:
            return 0.0

        total_support = sum(m.support for m in self.class_metrics.values())
        if total_support == 0:
            return 0.0

        weighted_sum = sum(m.f1_score * m.support for m in self.class_metrics.values())
        return weighted_sum / total_support

    @property
    def overall_accuracy(self) -> float:
        """Overall classification accuracy."""
        if self.total_medications == 0:
            return 0.0

        total_correct = sum(m.true_positives for m in self.class_metrics.values())
        return total_correct / self.total_medications

    @property
    def classification_coverage(self) -> float:
        """Percentage of medications that were classified."""
        if self.total_medications == 0:
            return 0.0
        return self.total_classified / self.total_medications

    def get_problematic_classes(self, min_f1: float = 0.7) -> List[str]:
        """Get drug classes with F1 score below threshold."""
        return [
            drug_class
            for drug_class, metrics in self.class_metrics.items()
            if metrics.f1_score < min_f1
        ]

    def get_top_classes(
        self, metric: str = "f1_score", n: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top N classes by specified metric."""
        metric_values = [
            (drug_class, getattr(metrics, metric))
            for drug_class, metrics in self.class_metrics.items()
        ]
        return sorted(metric_values, key=lambda x: x[1], reverse=True)[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_metrics": {
                "macro_precision": round(self.macro_precision, 4),
                "macro_recall": round(self.macro_recall, 4),
                "macro_f1": round(self.macro_f1, 4),
                "weighted_precision": round(self.weighted_precision, 4),
                "weighted_recall": round(self.weighted_recall, 4),
                "weighted_f1": round(self.weighted_f1, 4),
                "overall_accuracy": round(self.overall_accuracy, 4),
                "classification_coverage": round(self.classification_coverage, 4),
            },
            "summary_stats": {
                "total_medications": self.total_medications,
                "total_classified": self.total_classified,
                "total_unclassified": self.total_unclassified,
                "num_drug_classes": len(self.class_metrics),
            },
            "class_metrics": {
                drug_class: metrics.to_dict()
                for drug_class, metrics in self.class_metrics.items()
            },
            "problematic_classes": self.get_problematic_classes(),
            "top_classes_by_f1": self.get_top_classes("f1_score"),
        }


@dataclass
class ClassificationReport:
    """Detailed classification report with specific errors."""

    metrics: EvaluationMetrics
    false_positives: Dict[str, List[str]]  # drug_class -> [medication_names]
    false_negatives: Dict[str, List[str]]  # drug_class -> [medication_names]
    unclassified_medications: List[str]
    misclassifications: List[
        Dict[str, str]
    ]  # [{'medication': str, 'predicted': str, 'actual': str}]

    def get_actionable_insights(self) -> Dict[str, List[str]]:
        """Get actionable insights for improving classification."""
        insights = {
            "high_priority_fixes": [],
            "false_negative_patterns": [],
            "false_positive_patterns": [],
            "coverage_improvements": [],
        }

        # High priority: classes with low F1 and high support
        for drug_class, metrics in self.metrics.class_metrics.items():
            if metrics.f1_score < 0.7 and metrics.support > 10:
                insights["high_priority_fixes"].append(
                    f"{drug_class}: F1={metrics.f1_score:.3f}, Support={metrics.support}"
                )

        # False negative patterns
        fn_counts = Counter()
        for drug_class, medications in self.false_negatives.items():
            for med in medications:
                fn_counts[med] += 1

        for medication, count in fn_counts.most_common(5):
            insights["false_negative_patterns"].append(
                f"'{medication}' missed {count} times across classes"
            )

        # False positive patterns
        fp_counts = Counter()
        for drug_class, medications in self.false_positives.items():
            for med in medications:
                fp_counts[med] += 1

        for medication, count in fp_counts.most_common(5):
            insights["false_positive_patterns"].append(
                f"'{medication}' incorrectly classified {count} times"
            )

        # Coverage improvements
        if self.unclassified_medications:
            insights["coverage_improvements"].append(
                f"{len(self.unclassified_medications)} medications remain unclassified"
            )
            # Show most common unclassified medications
            unclass_sample = self.unclassified_medications[:5]
            insights["coverage_improvements"].append(
                f"Top unclassified: {', '.join(unclass_sample)}"
            )

        return insights

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": self.metrics.to_dict(),
            "detailed_errors": {
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "unclassified_medications": self.unclassified_medications,
                "misclassifications": self.misclassifications,
            },
            "actionable_insights": self.get_actionable_insights(),
        }

    def save_report(self, file_path: str) -> None:
        """Save detailed report to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        print("=== MEDICATION CLASSIFICATION EVALUATION REPORT ===\n")

        print("OVERALL METRICS:")
        print(f"  Macro F1:        {self.metrics.macro_f1:.4f}")
        print(f"  Weighted F1:     {self.metrics.weighted_f1:.4f}")
        print(f"  Precision:       {self.metrics.weighted_precision:.4f}")
        print(f"  Recall:          {self.metrics.weighted_recall:.4f}")
        print(f"  Coverage:        {self.metrics.classification_coverage:.4f}")
        print(f"  Total Medications: {self.metrics.total_medications}")
        print()

        print("TOP PERFORMING CLASSES:")
        for drug_class, f1 in self.metrics.get_top_classes("f1_score", 3):
            support = self.metrics.class_metrics[drug_class].support
            print(f"  {drug_class}: F1={f1:.4f}, Support={support}")
        print()

        print("PROBLEMATIC CLASSES (F1 < 0.7):")
        problematic = self.metrics.get_problematic_classes()
        for drug_class in problematic[:5]:
            metrics = self.metrics.class_metrics[drug_class]
            print(
                f"  {drug_class}: F1={metrics.f1_score:.4f}, P={metrics.precision:.4f}, R={metrics.recall:.4f}"
            )
        print()

        print("ACTIONABLE INSIGHTS:")
        insights = self.get_actionable_insights()
        for category, items in insights.items():
            if items:
                print(f"  {category.replace('_', ' ').title()}:")
                for item in items[:3]:  # Show top 3
                    print(f"    - {item}")
        print()
