"""
Source Ablation Sensitivity Test for PALM.

Tests whether the model is truly grounded by ablating the source context.

Compares three conditions:
1. Normal: real source → generate
2. Random: random tokens (same length) → generate
3. Dropped: empty/minimal source → generate

Key insight:
- If truly grounded: faithfulness drops SHARPLY with ablation
- LM coherence (PPL, diversity) should degrade LESS than faithfulness
- If faithfulness doesn't drop much → model just outputs plausible stuff, not using source

Usage:
    from palm.evaluation import SourceAblationEvaluator
    
    evaluator = SourceAblationEvaluator(model, tokenizer)
    result = evaluator.evaluate(eval_samples)
    print(evaluator.format_results(result))
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import json
import os

from .comprehensive import (
    compute_faithfulness_metrics,
    compute_degeneration_metrics,
    compute_perplexity,
    compute_distinct_ngrams,
    extract_entities_simple,
)


@dataclass
class ConditionMetrics:
    """Metrics for a single ablation condition."""
    condition_name: str
    
    # Faithfulness (compared to ORIGINAL source)
    entity_precision: float
    entity_recall: float
    hallucination_rate: float
    copy_rate: float
    
    # Coherence/quality
    perplexity: float
    distinct_2: float
    distinct_3: float
    repetition_rate: float
    
    # Sample outputs for inspection
    sample_outputs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            f"{self.condition_name}/entity_precision": self.entity_precision,
            f"{self.condition_name}/hallucination_rate": self.hallucination_rate,
            f"{self.condition_name}/perplexity": self.perplexity,
            f"{self.condition_name}/distinct_2": self.distinct_2,
            f"{self.condition_name}/repetition_rate": self.repetition_rate,
        }


@dataclass
class SourceAblationResult:
    """
    Results from source ablation sensitivity test.
    
    Contains metrics for all three conditions plus sensitivity analysis.
    """
    # Metrics for each condition
    normal: ConditionMetrics
    random: ConditionMetrics
    dropped: ConditionMetrics
    
    # Sensitivity deltas (how much metrics change with ablation)
    faithfulness_drop_random: float  # Should be LARGE if truly grounded
    faithfulness_drop_dropped: float
    
    coherence_drop_random: float  # Should be SMALLER than faithfulness drop
    coherence_drop_dropped: float
    
    # PPL changes
    ppl_increase_random: float
    ppl_increase_dropped: float
    
    # Diagnosis
    is_truly_grounded: bool
    grounding_strength: float  # Ratio of faith drop to coherence drop
    diagnosis_message: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        result.update(self.normal.to_dict())
        result.update(self.random.to_dict())
        result.update(self.dropped.to_dict())
        result.update({
            "sensitivity/faithfulness_drop_random": self.faithfulness_drop_random,
            "sensitivity/faithfulness_drop_dropped": self.faithfulness_drop_dropped,
            "sensitivity/coherence_drop_random": self.coherence_drop_random,
            "sensitivity/coherence_drop_dropped": self.coherence_drop_dropped,
            "sensitivity/ppl_increase_random": self.ppl_increase_random,
            "sensitivity/ppl_increase_dropped": self.ppl_increase_dropped,
            "diagnosis/is_truly_grounded": float(self.is_truly_grounded),
            "diagnosis/grounding_strength": self.grounding_strength,
        })
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SourceAblationEvaluator:
    """
    Test if model is truly grounded by ablating the source.
    
    The key causality test:
    - If scrambling the source doesn't hurt faithfulness much, the model
      isn't truly using the source - it's just outputting plausible text.
    - If faithfulness drops sharply but coherence remains, the model IS
      properly grounded and depends on the source for factual content.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        max_gen_tokens: int = 50,
        min_source_length: int = 10,
    ):
        """
        Args:
            model: PALM model to evaluate
            tokenizer: Tokenizer
            max_gen_tokens: Maximum tokens to generate per sample
            min_source_length: Minimum tokens for "dropped" source condition
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_gen_tokens = max_gen_tokens
        self.min_source_length = min_source_length
        self.device = next(model.parameters()).device
        self.vocab_size = tokenizer.vocab_size
    
    def _generate_random_source(self, original_source_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate random tokens with same length as original source.
        
        Avoids special tokens to ensure realistic "gibberish" that the model
        should NOT be able to extract meaningful information from.
        """
        seq_len = original_source_ids.size(1)
        
        # Sample random tokens, avoiding special tokens
        # Most vocabularies have special tokens at the start and end
        min_token = 100  # Skip first 100 (usually special tokens)
        max_token = min(self.vocab_size - 100, self.vocab_size)  # Skip last 100
        
        if max_token <= min_token:
            min_token = 0
            max_token = self.vocab_size
        
        random_ids = torch.randint(
            min_token, max_token, (1, seq_len), 
            device=self.device, dtype=original_source_ids.dtype
        )
        
        return random_ids
    
    def _generate_dropped_source(self) -> torch.Tensor:
        """
        Generate minimal/empty source.
        
        Uses a minimal prompt that provides no useful information.
        """
        # Minimal prompt - provides structure but no content
        minimal_text = "Context: [none]\n\nAnswer:"
        return self.tokenizer.encode(minimal_text, return_tensors="pt").to(self.device)
    
    def _get_perplexity(
        self,
        source_ids: torch.Tensor,
        target_text: str,
    ) -> float:
        """Compute perplexity for generation given source."""
        if not target_text:
            return 0.0
        
        source_text = self.tokenizer.decode(source_ids[0], skip_special_tokens=True)
        full_text = source_text + " " + target_text
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
        
        source_len = source_ids.size(1)
        labels = full_ids.clone()
        labels[:, :source_len] = -100
        
        with torch.no_grad():
            try:
                outputs = self.model(
                    full_ids, 
                    labels=labels, 
                    source_len=torch.tensor([source_len], device=self.device)
                )
                _, _, lm_loss, _ = outputs
                if lm_loss is not None:
                    return compute_perplexity(lm_loss.item())
            except Exception:
                pass
        
        return 0.0
    
    @torch.no_grad()
    def _evaluate_condition(
        self,
        source_ids: torch.Tensor,
        original_source_text: str,  # For faithfulness comparison
        condition_name: str,
        target_text: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate from source and compute metrics.
        
        Faithfulness is ALWAYS computed against the ORIGINAL source text,
        not the ablated source. This is the key: we want to see if the model
        can still produce faithful output when the source is corrupted.
        
        Args:
            source_ids: The (possibly ablated) source token IDs
            original_source_text: The original source text for faithfulness comparison
            condition_name: Name of this condition ("normal", "random", "dropped")
            target_text: Optional target text for PPL computation
            
        Returns:
            Tuple of (metrics_dict, generated_text)
        """
        self.model.eval()
        source_len = source_ids.size(1)
        
        # Generate with sampling and repetition penalty for realistic evaluation
        # Greedy decoding amplifies repetition; sampling shows true model behavior
        generated = self.model.generate(
            source_ids,
            max_length=source_len + self.max_gen_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            use_cache=True,
        )
        
        gen_tokens = generated[0, source_len:].tolist()
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        # Compute faithfulness against ORIGINAL source
        faith_metrics = compute_faithfulness_metrics(original_source_text, gen_text)
        
        # Compute degeneration
        degen_metrics = compute_degeneration_metrics(
            gen_tokens,
            expected_len=self.max_gen_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Get PPL if we have target
        ppl = 0.0
        if target_text:
            ppl = self._get_perplexity(source_ids, target_text)
        
        return {
            "entity_precision": faith_metrics.entity_precision,
            "entity_recall": faith_metrics.entity_recall,
            "hallucination_rate": faith_metrics.entity_hallucination_rate,
            "copy_rate": faith_metrics.copy_rate,
            "perplexity": ppl,
            "distinct_2": degen_metrics.distinct_2gram,
            "distinct_3": degen_metrics.distinct_3gram,
            "repetition_rate": degen_metrics.repetition_rate,
        }, gen_text
    
    def evaluate(
        self,
        eval_samples: List[Dict[str, str]],
        num_examples_to_keep: int = 3,
    ) -> SourceAblationResult:
        """
        Run source ablation evaluation on a set of samples.
        
        Args:
            eval_samples: List of {"source": str, "target": str} dicts
            num_examples_to_keep: Number of example outputs to store for inspection
            
        Returns:
            SourceAblationResult with all metrics and diagnosis
        """
        if not eval_samples:
            raise ValueError("No eval samples provided")
        
        # Accumulators for each condition
        normal_metrics = {k: [] for k in [
            "entity_precision", "entity_recall", "hallucination_rate", 
            "copy_rate", "perplexity", "distinct_2", "distinct_3", "repetition_rate"
        ]}
        random_metrics = {k: [] for k in normal_metrics}
        dropped_metrics = {k: [] for k in normal_metrics}
        
        normal_outputs = []
        random_outputs = []
        dropped_outputs = []
        
        for sample in eval_samples:
            source_text = sample["source"]
            target_text = sample.get("target", "")
            
            # 1. Normal condition
            source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(self.device)
            metrics, gen_text = self._evaluate_condition(
                source_ids, source_text, "normal", target_text
            )
            for k, v in metrics.items():
                normal_metrics[k].append(v)
            normal_outputs.append(gen_text)
            
            # 2. Random source condition
            random_ids = self._generate_random_source(source_ids)
            metrics, gen_text = self._evaluate_condition(
                random_ids, source_text, "random", target_text
            )
            for k, v in metrics.items():
                random_metrics[k].append(v)
            random_outputs.append(gen_text)
            
            # 3. Dropped source condition
            dropped_ids = self._generate_dropped_source()
            metrics, gen_text = self._evaluate_condition(
                dropped_ids, source_text, "dropped", target_text
            )
            for k, v in metrics.items():
                dropped_metrics[k].append(v)
            dropped_outputs.append(gen_text)
        
        # Aggregate metrics
        def avg(lst): 
            return sum(lst) / len(lst) if lst else 0.0
        
        def make_condition_metrics(name, metrics_dict, outputs):
            return ConditionMetrics(
                condition_name=name,
                entity_precision=avg(metrics_dict["entity_precision"]),
                entity_recall=avg(metrics_dict["entity_recall"]),
                hallucination_rate=avg(metrics_dict["hallucination_rate"]),
                copy_rate=avg(metrics_dict["copy_rate"]),
                perplexity=avg(metrics_dict["perplexity"]),
                distinct_2=avg(metrics_dict["distinct_2"]),
                distinct_3=avg(metrics_dict["distinct_3"]),
                repetition_rate=avg(metrics_dict["repetition_rate"]),
                sample_outputs=outputs[:num_examples_to_keep],
            )
        
        normal = make_condition_metrics("normal", normal_metrics, normal_outputs)
        random = make_condition_metrics("random", random_metrics, random_outputs)
        dropped = make_condition_metrics("dropped", dropped_metrics, dropped_outputs)
        
        # Compute sensitivity deltas
        faith_drop_random = normal.entity_precision - random.entity_precision
        faith_drop_dropped = normal.entity_precision - dropped.entity_precision
        
        # Use distinct-2 as coherence proxy (1 - repetition also works)
        coherence_drop_random = normal.distinct_2 - random.distinct_2
        coherence_drop_dropped = normal.distinct_2 - dropped.distinct_2
        
        # PPL increases
        ppl_increase_random = random.perplexity - normal.perplexity if normal.perplexity > 0 else 0
        ppl_increase_dropped = dropped.perplexity - normal.perplexity if normal.perplexity > 0 else 0
        
        # Diagnosis
        # Grounding strength: how much MORE faithfulness drops than coherence
        # High ratio = faithfulness depends on source, coherence doesn't
        epsilon = 0.01  # Avoid division by zero
        grounding_strength = faith_drop_random / (abs(coherence_drop_random) + epsilon)
        
        # Criteria for "truly grounded":
        # 1. Faithfulness drops significantly with random source (> 0.1)
        # 2. Grounding strength > 2 (faithfulness drops more than coherence)
        # 3. Faithfulness with random source is notably lower than normal
        is_truly_grounded = (
            faith_drop_random > 0.1 and  # Significant faith drop
            grounding_strength > 1.5 and  # Faith drops more than coherence
            random.entity_precision < normal.entity_precision * 0.8  # At least 20% relative drop
        )
        
        # Generate diagnosis message
        if is_truly_grounded:
            diagnosis = (
                f"✅ Model IS truly grounded (strength: {grounding_strength:.2f}x). "
                f"Faithfulness drops {faith_drop_random:.3f} with corrupted source, "
                f"while coherence only drops {abs(coherence_drop_random):.3f}."
            )
        else:
            if faith_drop_random < 0.1:
                diagnosis = (
                    f"❌ Model is NOT truly grounded. Faithfulness barely changes "
                    f"({faith_drop_random:.3f}) when source is corrupted - "
                    f"the model may just be outputting plausible text without using the source."
                )
            elif grounding_strength < 1.5:
                diagnosis = (
                    f"⚠️ Weak grounding (strength: {grounding_strength:.2f}x). "
                    f"Coherence and faithfulness drop similarly - "
                    f"the model may be partially grounded but also relies on general knowledge."
                )
            else:
                diagnosis = (
                    f"⚠️ Inconclusive grounding test. Strength: {grounding_strength:.2f}x, "
                    f"faith drop: {faith_drop_random:.3f}"
                )
        
        return SourceAblationResult(
            normal=normal,
            random=random,
            dropped=dropped,
            faithfulness_drop_random=faith_drop_random,
            faithfulness_drop_dropped=faith_drop_dropped,
            coherence_drop_random=coherence_drop_random,
            coherence_drop_dropped=coherence_drop_dropped,
            ppl_increase_random=ppl_increase_random,
            ppl_increase_dropped=ppl_increase_dropped,
            is_truly_grounded=is_truly_grounded,
            grounding_strength=grounding_strength,
            diagnosis_message=diagnosis,
        )
    
    def format_results(self, result: SourceAblationResult) -> str:
        """Format results as readable summary."""
        lines = [
            "",
            "=" * 70,
            "SOURCE ABLATION SENSITIVITY TEST",
            "=" * 70,
            "",
            "Condition          Entity Prec  Halluc Rate  Distinct-2  Repetition",
            "-" * 70,
            f"Normal (real)      {result.normal.entity_precision:>11.3f}  {result.normal.hallucination_rate:>11.3f}  "
            f"{result.normal.distinct_2:>10.3f}  {result.normal.repetition_rate:>10.3f}",
            f"Random tokens      {result.random.entity_precision:>11.3f}  {result.random.hallucination_rate:>11.3f}  "
            f"{result.random.distinct_2:>10.3f}  {result.random.repetition_rate:>10.3f}",
            f"Dropped source     {result.dropped.entity_precision:>11.3f}  {result.dropped.hallucination_rate:>11.3f}  "
            f"{result.dropped.distinct_2:>10.3f}  {result.dropped.repetition_rate:>10.3f}",
            "",
            "Sensitivity Analysis:",
            f"  Faithfulness drop (random):    {result.faithfulness_drop_random:+.3f}",
            f"  Faithfulness drop (dropped):   {result.faithfulness_drop_dropped:+.3f}",
            f"  Coherence drop (random):       {result.coherence_drop_random:+.3f}",
            f"  Coherence drop (dropped):      {result.coherence_drop_dropped:+.3f}",
            "",
            f"DIAGNOSIS:",
            f"  {result.diagnosis_message}",
            "",
        ]
        
        # Add sample outputs if available
        if result.normal.sample_outputs:
            lines.extend([
                "Sample Outputs:",
                "-" * 70,
                "",
                "Normal condition:",
            ])
            for i, output in enumerate(result.normal.sample_outputs[:2]):
                lines.append(f"  [{i+1}] {output[:100]}...")
            
            lines.append("\nRandom source condition:")
            for i, output in enumerate(result.random.sample_outputs[:2]):
                lines.append(f"  [{i+1}] {output[:100]}...")
            
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def plot_ablation_results(
    result: SourceAblationResult,
    output_path: str = "source_ablation.png",
) -> None:
    """
    Plot source ablation results.
    
    Args:
        result: SourceAblationResult from evaluation
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    conditions = ['Normal', 'Random', 'Dropped']
    x = np.arange(len(conditions))
    width = 0.35
    
    # 1. Faithfulness comparison
    ax1 = axes[0]
    faith_values = [
        result.normal.entity_precision,
        result.random.entity_precision,
        result.dropped.entity_precision,
    ]
    colors = ['#27AE60', '#E74C3C', '#E74C3C']
    bars = ax1.bar(x, faith_values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Entity Precision (Faithfulness)')
    ax1.set_title('Faithfulness by Condition')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, faith_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Coherence comparison (Distinct-2)
    ax2 = axes[1]
    d2_values = [
        result.normal.distinct_2,
        result.random.distinct_2,
        result.dropped.distinct_2,
    ]
    colors = ['#F39C12', '#3498DB', '#3498DB']
    bars = ax2.bar(x, d2_values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Distinct-2 (Coherence)')
    ax2.set_title('Coherence by Condition')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions)
    ax2.set_ylim(0, 1)
    
    for bar, val in zip(bars, d2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Sensitivity deltas
    ax3 = axes[2]
    deltas = ['Faith Drop\n(Random)', 'Faith Drop\n(Dropped)', 
              'Coherence Drop\n(Random)', 'Coherence Drop\n(Dropped)']
    values = [
        result.faithfulness_drop_random,
        result.faithfulness_drop_dropped,
        result.coherence_drop_random,
        result.coherence_drop_dropped,
    ]
    colors = ['#E74C3C' if v > 0 else '#27AE60' for v in values]
    
    bars = ax3.bar(range(len(deltas)), values, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Change from Normal')
    ax3.set_title('Sensitivity to Source Ablation')
    ax3.set_xticks(range(len(deltas)))
    ax3.set_xticklabels(deltas, fontsize=9)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars, values):
        offset = 0.02 if val >= 0 else -0.05
        ax3.text(bar.get_x() + bar.get_width()/2, val + offset,
                f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontweight='bold', fontsize=9)
    
    # Add grounding diagnosis
    grounding_text = "✅ GROUNDED" if result.is_truly_grounded else "❌ NOT GROUNDED"
    fig.suptitle(f'Source Ablation Test - {grounding_text}\nGrounding Strength: {result.grounding_strength:.2f}x',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved ablation plot to {output_path}")


def save_ablation_results(
    result: SourceAblationResult,
    output_file: str = "source_ablation_results.json",
) -> None:
    """Save ablation results to JSON file."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.to_json())
    print(f"Saved ablation results to {output_file}")


# GRADUAL ABLATION SENSITIVITY CURVE
# Good visualization: X = % corrupted, Y = quality metric
# Compare baseline vs PALM to show PALM degrades more sharply (uses source more)

@dataclass
class GradualAblationPoint:
    """Single point on the ablation curve."""
    corruption_pct: float  # 0.0 to 1.0
    entity_precision: float
    hallucination_rate: float
    distinct_2: float
    logprob_delta: float  # Change in log probability vs 0% corruption
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class GradualAblationCurve:
    """Full ablation curve for one model."""
    model_name: str
    points: List[GradualAblationPoint]
    
    # Summary statistics
    auc_faithfulness: float  # Area under faithfulness curve (higher = less sensitive)
    sensitivity_slope: float  # Linear slope of faithfulness vs corruption
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "points": [p.to_dict() for p in self.points],
            "auc_faithfulness": self.auc_faithfulness,
            "sensitivity_slope": self.sensitivity_slope,
        }


@dataclass
class AblationCurveComparison:
    """Comparison of ablation curves between baseline and PALM."""
    baseline_curve: GradualAblationCurve
    palm_curve: GradualAblationCurve
    
    # Key metrics
    sensitivity_ratio: float  # PALM sensitivity / baseline sensitivity
    is_palm_more_grounded: bool  # PALM degrades faster = uses source more
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline_curve.to_dict(),
            "palm": self.palm_curve.to_dict(),
            "sensitivity_ratio": self.sensitivity_ratio,
            "is_palm_more_grounded": self.is_palm_more_grounded,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class GradualAblationEvaluator:
    """
    Evaluate source sensitivity with gradual corruption levels.
    
    Creates the "killer demo" curve:
    - X-axis: % of source corrupted (0%, 10%, 25%, 50%, 75%, 100%)
    - Y-axis: Quality metric (faithfulness, logprob delta)
    - Compare: Baseline vs PALM
    - Key insight: PALM should degrade MORE sharply (it actually uses the source!)
    """
    
    def __init__(
        self,
        tokenizer,
        max_gen_tokens: int = 50,
        corruption_levels: List[float] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer (shared between models)
            max_gen_tokens: Maximum tokens to generate
            corruption_levels: List of corruption percentages (default: 0 to 100%)
        """
        self.tokenizer = tokenizer
        self.max_gen_tokens = max_gen_tokens
        self.corruption_levels = corruption_levels or [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        self.vocab_size = tokenizer.vocab_size
    
    def _corrupt_source(
        self,
        source_ids: torch.Tensor,
        corruption_pct: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Corrupt a percentage of source tokens with random tokens.
        
        Args:
            source_ids: Original source token IDs [1, seq_len]
            corruption_pct: Fraction of tokens to corrupt (0.0 to 1.0)
            device: Device to place tensor on
            
        Returns:
            Corrupted source token IDs
        """
        if corruption_pct <= 0:
            return source_ids.clone()
        
        seq_len = source_ids.size(1)
        num_to_corrupt = int(seq_len * corruption_pct)
        
        if num_to_corrupt >= seq_len:
            # Full corruption - all random
            min_token = 100
            max_token = min(self.vocab_size - 100, self.vocab_size)
            return torch.randint(min_token, max_token, (1, seq_len), device=device)
        
        # Partial corruption - randomly select positions
        corrupted = source_ids.clone()
        positions = torch.randperm(seq_len)[:num_to_corrupt]
        
        min_token = 100
        max_token = min(self.vocab_size - 100, self.vocab_size)
        random_tokens = torch.randint(min_token, max_token, (num_to_corrupt,), device=device)
        
        corrupted[0, positions] = random_tokens
        return corrupted
    
    def _get_logprob(
        self,
        model: torch.nn.Module,
        source_ids: torch.Tensor,
        target_text: str,
        device: torch.device,
    ) -> float:
        """Compute log probability of target given source."""
        if not target_text:
            return 0.0
        
        source_text = self.tokenizer.decode(source_ids[0], skip_special_tokens=True)
        full_text = source_text + " " + target_text
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(device)
        
        source_len = source_ids.size(1)
        labels = full_ids.clone()
        labels[:, :source_len] = -100
        
        with torch.no_grad():
            try:
                outputs = model(
                    full_ids,
                    labels=labels,
                    source_len=torch.tensor([source_len], device=device)
                )
                _, _, lm_loss, _ = outputs
                if lm_loss is not None:
                    # Return negative loss as logprob (higher = better)
                    return -lm_loss.item()
            except Exception:
                pass
        
        return 0.0
    
    @torch.no_grad()
    def _evaluate_at_corruption_level(
        self,
        model: torch.nn.Module,
        eval_samples: List[Dict[str, str]],
        corruption_pct: float,
        baseline_logprobs: List[float],  # Logprobs at 0% corruption
        device: torch.device,
    ) -> GradualAblationPoint:
        """Evaluate model at a specific corruption level."""
        model.eval()
        
        all_precision = []
        all_halluc = []
        all_d2 = []
        all_logprob_delta = []
        
        for i, sample in enumerate(eval_samples):
            source_text = sample["source"]
            target_text = sample.get("target", "")
            
            # Get original source IDs
            source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(device)
            
            # Corrupt source
            corrupted_ids = self._corrupt_source(source_ids, corruption_pct, device)
            source_len = corrupted_ids.size(1)
            
            # Generate
            generated = model.generate(
                corrupted_ids,
                max_length=source_len + self.max_gen_tokens,
                do_sample=False,
                use_cache=True,
            )
            
            gen_tokens = generated[0, source_len:].tolist()
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Compute faithfulness against ORIGINAL source
            faith_metrics = compute_faithfulness_metrics(source_text, gen_text)
            all_precision.append(faith_metrics.entity_precision)
            all_halluc.append(faith_metrics.entity_hallucination_rate)
            
            # Compute diversity
            all_d2.append(compute_distinct_ngrams(gen_tokens, 2))
            
            # Compute logprob delta (if we have target and baseline)
            if target_text and i < len(baseline_logprobs):
                current_logprob = self._get_logprob(model, corrupted_ids, target_text, device)
                logprob_delta = current_logprob - baseline_logprobs[i]
                all_logprob_delta.append(logprob_delta)
        
        return GradualAblationPoint(
            corruption_pct=corruption_pct,
            entity_precision=sum(all_precision) / len(all_precision) if all_precision else 0,
            hallucination_rate=sum(all_halluc) / len(all_halluc) if all_halluc else 0,
            distinct_2=sum(all_d2) / len(all_d2) if all_d2 else 1,
            logprob_delta=sum(all_logprob_delta) / len(all_logprob_delta) if all_logprob_delta else 0,
        )
    
    def _get_baseline_logprobs(
        self,
        model: torch.nn.Module,
        eval_samples: List[Dict[str, str]],
        device: torch.device,
    ) -> List[float]:
        """Get logprobs at 0% corruption for delta computation."""
        logprobs = []
        for sample in eval_samples:
            source_text = sample["source"]
            target_text = sample.get("target", "")
            
            if target_text:
                source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(device)
                logprob = self._get_logprob(model, source_ids, target_text, device)
                logprobs.append(logprob)
            else:
                logprobs.append(0.0)
        
        return logprobs
    
    def _compute_curve_stats(self, points: List[GradualAblationPoint]) -> Tuple[float, float]:
        """Compute AUC and slope for faithfulness curve."""
        if len(points) < 2:
            return 0.0, 0.0
        
        # Sort by corruption percentage
        sorted_points = sorted(points, key=lambda p: p.corruption_pct)
        
        # AUC using trapezoidal rule
        auc = 0.0
        for i in range(len(sorted_points) - 1):
            p1, p2 = sorted_points[i], sorted_points[i + 1]
            width = p2.corruption_pct - p1.corruption_pct
            height = (p1.entity_precision + p2.entity_precision) / 2
            auc += width * height
        
        # Linear regression slope
        x = [p.corruption_pct for p in sorted_points]
        y = [p.entity_precision for p in sorted_points]
        n = len(x)
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denom
        
        return auc, slope
    
    def evaluate_single_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        eval_samples: List[Dict[str, str]],
    ) -> GradualAblationCurve:
        """
        Evaluate a single model across all corruption levels.
        
        Args:
            model: Model to evaluate
            model_name: Name for the curve (e.g., "baseline", "palm")
            eval_samples: List of {"source": str, "target": str}
            
        Returns:
            GradualAblationCurve with all points
        """
        device = next(model.parameters()).device
        
        # Get baseline logprobs at 0% corruption
        baseline_logprobs = self._get_baseline_logprobs(model, eval_samples, device)
        
        points = []
        for corruption_pct in self.corruption_levels:
            print(f"  {model_name}: {int(corruption_pct * 100)}% corruption...")
            point = self._evaluate_at_corruption_level(
                model, eval_samples, corruption_pct, baseline_logprobs, device
            )
            points.append(point)
        
        # Compute summary statistics
        auc, slope = self._compute_curve_stats(points)
        
        return GradualAblationCurve(
            model_name=model_name,
            points=points,
            auc_faithfulness=auc,
            sensitivity_slope=slope,  # Negative slope = degrades with corruption
        )
    
    def compare_models(
        self,
        baseline_model: torch.nn.Module,
        palm_model: torch.nn.Module,
        eval_samples: List[Dict[str, str]],
    ) -> AblationCurveComparison:
        """
        Compare baseline vs PALM model ablation sensitivity.
        
        This is the KILLER DEMO: shows PALM degrades more sharply
        when source is corrupted, proving it actually uses the source.
        
        Args:
            baseline_model: Standard LM without PALM modifications
            palm_model: PALM model with source attention
            eval_samples: Evaluation samples
            
        Returns:
            AblationCurveComparison with both curves
        """
        print("\n" + "=" * 60)
        print("GRADUAL SOURCE ABLATION COMPARISON")
        print("=" * 60)
        
        # Evaluate baseline
        print("\nEvaluating baseline model...")
        baseline_curve = self.evaluate_single_model(
            baseline_model, "baseline", eval_samples
        )
        
        # Evaluate PALM
        print("\nEvaluating PALM model...")
        palm_curve = self.evaluate_single_model(
            palm_model, "palm", eval_samples
        )
        
        # Compare sensitivities (more negative slope = more sensitive to corruption)
        # PALM should have more negative slope (degrades faster when source corrupted)
        baseline_sensitivity = abs(baseline_curve.sensitivity_slope)
        palm_sensitivity = abs(palm_curve.sensitivity_slope)
        
        sensitivity_ratio = palm_sensitivity / (baseline_sensitivity + 1e-6)
        is_palm_more_grounded = palm_sensitivity > baseline_sensitivity * 1.2  # 20% threshold
        
        return AblationCurveComparison(
            baseline_curve=baseline_curve,
            palm_curve=palm_curve,
            sensitivity_ratio=sensitivity_ratio,
            is_palm_more_grounded=is_palm_more_grounded,
        )
    
    def format_comparison(self, comparison: AblationCurveComparison) -> str:
        """Format comparison results as readable summary."""
        lines = [
            "",
            "=" * 70,
            "GRADUAL SOURCE ABLATION SENSITIVITY CURVE",
            "=" * 70,
            "",
            "Corruption %     Baseline Faith    PALM Faith    Delta",
            "-" * 70,
        ]
        
        baseline_points = {p.corruption_pct: p for p in comparison.baseline_curve.points}
        palm_points = {p.corruption_pct: p for p in comparison.palm_curve.points}
        
        for pct in sorted(baseline_points.keys()):
            bp = baseline_points[pct]
            pp = palm_points.get(pct)
            if pp:
                delta = pp.entity_precision - bp.entity_precision
                lines.append(
                    f"{int(pct * 100):>10}%     {bp.entity_precision:>13.3f}    "
                    f"{pp.entity_precision:>10.3f}    {delta:>+8.3f}"
                )
        
        lines.extend([
            "",
            "Summary Statistics:",
            f"  Baseline AUC:       {comparison.baseline_curve.auc_faithfulness:.3f}",
            f"  PALM AUC:           {comparison.palm_curve.auc_faithfulness:.3f}",
            f"  Baseline slope:     {comparison.baseline_curve.sensitivity_slope:.4f}",
            f"  PALM slope:         {comparison.palm_curve.sensitivity_slope:.4f}",
            f"  Sensitivity ratio:  {comparison.sensitivity_ratio:.2f}x",
            "",
        ])
        
        if comparison.is_palm_more_grounded:
            lines.append("✅ PALM IS MORE GROUNDED: Degrades more sharply when source corrupted")
            lines.append("   This proves PALM actually uses the source for generation!")
        else:
            lines.append("⚠️ PALM sensitivity similar to baseline - check training")
        
        lines.append("=" * 70)
        return "\n".join(lines)


def plot_ablation_sensitivity_curve(
    comparison: AblationCurveComparison,
    output_path: str = "ablation_sensitivity_curve.png",
) -> None:
    """
    Plot the KILLER DEMO: ablation sensitivity curves comparing baseline vs PALM.
    
    Shows:
    - X-axis: % of source corrupted
    - Y-axis: Faithfulness (entity precision)
    - Two curves: baseline (should be flat) vs PALM (should drop sharply)
    
    Args:
        comparison: AblationCurveComparison from evaluator
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    baseline_x = [p.corruption_pct * 100 for p in comparison.baseline_curve.points]
    baseline_faith = [p.entity_precision for p in comparison.baseline_curve.points]
    baseline_logprob = [p.logprob_delta for p in comparison.baseline_curve.points]
    
    palm_x = [p.corruption_pct * 100 for p in comparison.palm_curve.points]
    palm_faith = [p.entity_precision for p in comparison.palm_curve.points]
    palm_logprob = [p.logprob_delta for p in comparison.palm_curve.points]
    
    # ===== Plot 1: Faithfulness vs Corruption =====
    ax1 = axes[0]
    
    # Plot curves
    ax1.plot(baseline_x, baseline_faith, 'o-', color='#3498DB', linewidth=2.5, 
             markersize=10, label='Baseline', markeredgecolor='white', markeredgewidth=2)
    ax1.plot(palm_x, palm_faith, 's-', color='#E74C3C', linewidth=2.5,
             markersize=10, label='PALM', markeredgecolor='white', markeredgewidth=2)
    
    # Fill between to show the gap
    ax1.fill_between(palm_x, palm_faith, baseline_faith, 
                     where=[p >= b for p, b in zip(palm_faith, baseline_faith)],
                     alpha=0.3, color='#27AE60', label='PALM advantage')
    ax1.fill_between(palm_x, palm_faith, baseline_faith,
                     where=[p < b for p, b in zip(palm_faith, baseline_faith)],
                     alpha=0.3, color='#E74C3C', label='Baseline advantage')
    
    ax1.set_xlabel('Source Corruption (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Faithfulness (Entity Precision)', fontsize=12, fontweight='bold')
    ax1.set_title('Source Ablation Sensitivity\n(Steeper drop = uses source more)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(0, 1.05)
    
    # Add annotation for key insight
    if comparison.is_palm_more_grounded:
        ax1.annotate('PALM uses source\nmore → degrades faster',
                    xy=(75, palm_faith[-2] if len(palm_faith) > 1 else 0.5),
                    xytext=(50, 0.3),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2),
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='#E74C3C'))
    
    # ===== Plot 2: LogProb Delta vs Corruption =====
    ax2 = axes[1]
    
    ax2.plot(baseline_x, baseline_logprob, 'o-', color='#3498DB', linewidth=2.5,
             markersize=10, label='Baseline', markeredgecolor='white', markeredgewidth=2)
    ax2.plot(palm_x, palm_logprob, 's-', color='#E74C3C', linewidth=2.5,
             markersize=10, label='PALM', markeredgecolor='white', markeredgewidth=2)
    
    ax2.set_xlabel('Source Corruption (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Log Probability Delta', fontsize=12, fontweight='bold')
    ax2.set_title('Perplexity Change with Corruption\n(More negative = relies on source)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 105)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Main title
    grounded_text = "✅ PALM More Grounded" if comparison.is_palm_more_grounded else "⚠️ Similar Sensitivity"
    fig.suptitle(f'Source–Target Coherence: Ablation Sensitivity Curve\n{grounded_text} '
                 f'(Sensitivity Ratio: {comparison.sensitivity_ratio:.2f}x)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved ablation sensitivity curve to {output_path}")


def save_ablation_curve_comparison(
    comparison: AblationCurveComparison,
    output_file: str = "ablation_curve_comparison.json",
) -> None:
    """Save ablation curve comparison to JSON file."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(comparison.to_json())
    print(f"Saved ablation curve comparison to {output_file}")

