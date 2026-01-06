"""
Go/No-Go Assessment for PALM Training

Provides clear, quantitative criteria for deciding whether to scale training.
Based on context rot resistance, source grounding, and generation quality.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import json


@dataclass
class GoNoGoCriteria:
    """Thresholds for go/no-go decision."""
    # Source Ablation: How much should faithfulness drop when source is corrupted?
    # Higher = model uses source more = good
    min_source_ablation_drop: float = 0.10  # At least 10% drop
    
    # Distractor Resistance: Accuracy with 4 distractors
    min_distractor_accuracy: float = 0.60  # At least 60% correct
    
    # Length Robustness: Max degradation from focused to full context
    max_length_degradation: float = 0.25  # At most 25% worse
    
    # Semantic Needle: Accuracy on semantic (non-lexical) matching
    min_semantic_needle_accuracy: float = 0.50  # At least 50%
    
    # Generation Quality: Maximum repetition rate
    max_repetition_rate: float = 0.30  # At most 30% repetition
    
    # Distinct-2: Minimum diversity
    min_distinct_2: float = 0.15  # At least 15% unique bigrams


@dataclass
class CriterionResult:
    """Result of checking a single criterion."""
    name: str
    threshold: float
    actual: float
    passed: bool
    direction: str  # "min" or "max"
    importance: str  # "critical", "important", "nice_to_have"
    
    def format(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        op = ">=" if self.direction == "min" else "<="
        return f"  [{status}] {self.name}: {self.actual:.3f} {op} {self.threshold:.3f}"


@dataclass
class GoNoGoAssessment:
    """Complete go/no-go assessment."""
    criteria_results: List[CriterionResult]
    critical_passed: int
    critical_total: int
    important_passed: int
    important_total: int
    recommendation: str  # "GO", "NO_GO", "CONDITIONAL"
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "critical_passed": self.critical_passed,
            "critical_total": self.critical_total,
            "important_passed": self.important_passed,
            "important_total": self.important_total,
            "all_passed": all(c.passed for c in self.criteria_results),
            "criteria": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "actual": c.actual,
                    "threshold": c.threshold,
                }
                for c in self.criteria_results
            ],
        }
    
    def format_report(self) -> str:
        """Generate formatted report for console output."""
        lines = [
            "",
            "=" * 70,
            "GO / NO-GO ASSESSMENT",
            "=" * 70,
            "",
        ]
        
        # Group by importance
        critical = [c for c in self.criteria_results if c.importance == "critical"]
        important = [c for c in self.criteria_results if c.importance == "important"]
        nice = [c for c in self.criteria_results if c.importance == "nice_to_have"]
        
        if critical:
            lines.append("CRITICAL CRITERIA:")
            lines.extend([c.format() for c in critical])
            lines.append("")
        
        if important:
            lines.append("IMPORTANT CRITERIA:")
            lines.extend([c.format() for c in important])
            lines.append("")
        
        if nice:
            lines.append("NICE TO HAVE:")
            lines.extend([c.format() for c in nice])
            lines.append("")
        
        lines.extend([
            "-" * 70,
            f"Critical: {self.critical_passed}/{self.critical_total} passed",
            f"Important: {self.important_passed}/{self.important_total} passed",
            "",
            "=" * 70,
            f"RECOMMENDATION: {self.recommendation}",
            "=" * 70,
            "",
            "Reasoning:",
            self.reasoning,
            "",
        ])
        
        return "\n".join(lines)


def assess_go_nogo(
    context_rot_result=None,
    ablation_result=None,
    staggered_result=None,
    criteria: Optional[GoNoGoCriteria] = None,
) -> GoNoGoAssessment:
    """
    Assess whether training results meet go/no-go criteria.
    
    Args:
        context_rot_result: Result from ContextRotEvaluator.run_full_suite()
        ablation_result: Result from SourceAblationEvaluator.evaluate()
        staggered_result: Result from StaggeredEvaluator (for degeneration metrics)
        criteria: Custom thresholds (uses defaults if None)
    
    Returns:
        GoNoGoAssessment with recommendation
    """
    criteria = criteria or GoNoGoCriteria()
    results = []
    
    # CRITICAL: Source Ablation Drop
    if ablation_result is not None:
        # Calculate faithfulness drop when source is corrupted
        normal_faith = ablation_result.normal.entity_precision
        random_faith = ablation_result.random.entity_precision
        ablation_drop = normal_faith - random_faith
        
        results.append(CriterionResult(
            name="Source Ablation Drop",
            threshold=criteria.min_source_ablation_drop,
            actual=ablation_drop,
            passed=ablation_drop >= criteria.min_source_ablation_drop,
            direction="min",
            importance="critical",
        ))
    
    # CRITICAL: Distractor Resistance
    if context_rot_result is not None:
        distractor_acc_4 = context_rot_result.distractor_accuracy_by_count.get(4, 0)
        
        results.append(CriterionResult(
            name="Distractor Accuracy (4 distractors)",
            threshold=criteria.min_distractor_accuracy,
            actual=distractor_acc_4,
            passed=distractor_acc_4 >= criteria.min_distractor_accuracy,
            direction="min",
            importance="critical",
        ))
    
    # IMPORTANT: Length Degradation
    if context_rot_result is not None:
        length_deg = context_rot_result.length_degradation
        
        results.append(CriterionResult(
            name="Length Degradation",
            threshold=criteria.max_length_degradation,
            actual=length_deg,
            passed=length_deg <= criteria.max_length_degradation,
            direction="max",
            importance="important",
        ))
    
    # IMPORTANT: Semantic Needle Accuracy
    if context_rot_result is not None:
        semantic_acc = context_rot_result.semantic_needle_accuracy
        
        results.append(CriterionResult(
            name="Semantic Needle Accuracy",
            threshold=criteria.min_semantic_needle_accuracy,
            actual=semantic_acc,
            passed=semantic_acc >= criteria.min_semantic_needle_accuracy,
            direction="min",
            importance="important",
        ))
    
    # NICE TO HAVE: Repetition Rate
    if staggered_result is not None and staggered_result.degeneration is not None:
        rep_rate = staggered_result.degeneration.repetition_rate
        
        results.append(CriterionResult(
            name="Repetition Rate",
            threshold=criteria.max_repetition_rate,
            actual=rep_rate,
            passed=rep_rate <= criteria.max_repetition_rate,
            direction="max",
            importance="nice_to_have",
        ))
    
    # NICE TO HAVE: Distinct-2
    if staggered_result is not None and staggered_result.degeneration is not None:
        distinct_2 = staggered_result.degeneration.distinct_2gram
        
        results.append(CriterionResult(
            name="Distinct-2",
            threshold=criteria.min_distinct_2,
            actual=distinct_2,
            passed=distinct_2 >= criteria.min_distinct_2,
            direction="min",
            importance="nice_to_have",
        ))
    
    # Compute Summary
    critical = [r for r in results if r.importance == "critical"]
    important = [r for r in results if r.importance == "important"]
    
    critical_passed = sum(r.passed for r in critical)
    critical_total = len(critical)
    important_passed = sum(r.passed for r in important)
    important_total = len(important)
    
    # Decision logic
    if critical_total == 0:
        recommendation = "INSUFFICIENT_DATA"
        reasoning = "Not enough evaluation data to make a recommendation. Run context rot and ablation tests first."
    elif critical_passed < critical_total:
        recommendation = "NO_GO"
        failed_critical = [r.name for r in critical if not r.passed]
        reasoning = f"Failed critical criteria: {', '.join(failed_critical)}. Model is not reliably using source context. Consider: (1) increasing SAE weight, (2) more grounded QA data, (3) longer training."
    elif important_passed < important_total * 0.5:
        recommendation = "CONDITIONAL"
        failed_important = [r.name for r in important if not r.passed]
        reasoning = f"Passed critical criteria but failed important criteria: {', '.join(failed_important)}. Consider addressing these issues before scaling, or proceed with caution."
    else:
        recommendation = "GO"
        reasoning = "Model demonstrates reliable source grounding and context rot resistance. Ready to scale to longer sequences."
    
    return GoNoGoAssessment(
        criteria_results=results,
        critical_passed=critical_passed,
        critical_total=critical_total,
        important_passed=important_passed,
        important_total=important_total,
        recommendation=recommendation,
        reasoning=reasoning,
    )


def save_go_nogo_assessment(assessment: GoNoGoAssessment, output_path: str):
    """Save assessment to JSON."""
    with open(output_path, "w") as f:
        json.dump(assessment.to_dict(), f, indent=2)
    print(f"Saved go/no-go assessment to {output_path}")
