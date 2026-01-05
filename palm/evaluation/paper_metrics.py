"""
Paper-based evaluation metrics from the PALM paper:
"Decoder-Only or Encoder-Decoder? Interpreting Language Model as a Regularized Encoder-Decoder"
(Fu et al., 2023 - arxiv.org/abs/2304.04052)

Implements:
1. Stepwise Hallucination Analysis - measures hallucination ratio at each generation step
2. Numerical Sensitivity Analysis - measures output sensitivity to source perturbations

These metrics provide deeper insights into:
- When and where hallucinations occur during generation
- How strongly the model depends on source tokens at each position
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import json
import os


# STEPWISE HALLUCINATION ANALYSIS
# From Section 4.2 of the paper: "Hallucination Ratio (HR) Analysis"
#
# The hallucination ratio at step i is defined as:
#   HR_i = 1 - [ Σ_j max(1(t_ji ∈ r_j), C[s_jk, t_ji]) / Σ_j 1(|t_j| ≥ i) ]
#
# Where:
#   - t_ji is the i-th generated token for sample j
#   - r_j is the reference (gold) tokens for sample j
#   - s_jk is the k-th source token for sample j
#   - C[p, q] is an alignment score using word co-occurrence with sigmoid normalization


@dataclass
class StepwiseHallucinationResult:
    """Results from stepwise hallucination analysis."""
    # Hallucination ratio at each step [HR_1, HR_2, ..., HR_max]
    hallucination_ratios: List[float]
    
    # Summary statistics
    avg_hallucination_ratio: float  # Average across all steps
    max_hallucination_ratio: float  # Maximum (worst step)
    min_hallucination_ratio: float  # Minimum (best step)
    peak_hallucination_step: int    # Step with highest hallucination
    
    # Sample-level statistics
    num_samples: int
    max_generation_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hallucination_ratios": self.hallucination_ratios,
            "avg_hallucination_ratio": self.avg_hallucination_ratio,
            "max_hallucination_ratio": self.max_hallucination_ratio,
            "min_hallucination_ratio": self.min_hallucination_ratio,
            "peak_hallucination_step": self.peak_hallucination_step,
            "num_samples": self.num_samples,
            "max_generation_length": self.max_generation_length,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def compute_alignment_score(
    token: str,
    source_tokens: List[str],
    generated_tokens: List[str],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Compute alignment score C[p, q] between a generated token and source.
    
    From the paper:
    C[p, q] = sigmoid((#co-occurrences(p in source) * #co-occurrences(q in generated) - α) / β)
    
    Args:
        token: The generated token to check
        source_tokens: List of source tokens
        generated_tokens: List of all generated tokens
        alpha: Offset parameter (default 1.0)
        beta: Scaling parameter (default 1.0)
        
    Returns:
        Alignment score between 0 and 1
    """
    # Count occurrences
    in_source = 1 if token.lower() in [s.lower() for s in source_tokens] else 0
    in_generated = 1 if token.lower() in [g.lower() for g in generated_tokens] else 0
    
    # Compute alignment using sigmoid normalization
    cval = (in_source * in_generated) - alpha
    alignment_score = 1 / (1 + math.exp(-cval / max(beta, 1e-6)))
    
    return alignment_score


def compute_stepwise_hallucination_ratio(
    model: torch.nn.Module,
    tokenizer,
    dataset: List[Dict[str, str]],
    device: torch.device,
    max_gen_length: int = 64,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> StepwiseHallucinationResult:
    """
    Compute stepwise hallucination ratio as defined in the PALM paper.
    
    Measures the hallucination ratio at each generation step, showing when
    the model is most likely to hallucinate during the generation process.
    
    Args:
        model: PALM model to evaluate
        tokenizer: Tokenizer for encoding/decoding
        dataset: List of {"source"/"prompt": str, "target"/"completion": str} dicts
        device: Device to run on
        max_gen_length: Maximum generation length
        alpha: Alignment score offset parameter
        beta: Alignment score scaling parameter
        
    Returns:
        StepwiseHallucinationResult with HR at each step
    """
    model.eval()
    
    all_sources: List[str] = []
    all_refs: List[str] = []
    all_generated_tokens: List[List[str]] = []
    
    # Get the underlying model for generation if wrapped by PEFT
    gen_model = model
    if hasattr(model, 'base_model'):
        base = model.base_model
        if hasattr(base, 'model'):
            gen_model = base.model
    
    # Generate for each sample
    for item in dataset:
        # Handle different key names
        source = item.get("prompt") or item.get("source", "")
        reference = item.get("completion") or item.get("target", "")
        all_sources.append(source)
        all_refs.append(reference)
        
        input_ids = tokenizer.encode(source, return_tensors="pt").to(device)
        source_len = input_ids.size(1)
        
        with torch.no_grad():
            dev_type = device.type
            with torch.amp.autocast(device_type=dev_type, enabled=(dev_type == "cuda")):
                generated_ids = gen_model.generate(
                    input_ids=input_ids,
                    max_length=source_len + max_gen_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
        
        # Decode only generated tokens (after source)
        gen_tokens = generated_ids[0, source_len:].tolist()
        decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        tokens = decoded_text.split()
        all_generated_tokens.append(tokens)
    
    # Compute HR at each step
    max_step = max(len(seq) for seq in all_generated_tokens) if all_generated_tokens else 0
    hallucination_ratios: List[float] = []
    
    for i in range(1, max_step + 1):
        denom = 0
        alignment_sum = 0.0
        
        for idx, gen_tokens in enumerate(all_generated_tokens):
            if len(gen_tokens) < i:
                continue
            
            denom += 1
            t_ji = gen_tokens[i - 1]  # i-th token (1-indexed)
            source_tokens = all_sources[idx].split()
            ref_tokens = all_refs[idx].split()
            
            # Check if token is in gold reference
            in_gold = t_ji.lower() in [r.lower() for r in ref_tokens]
            
            # Compute alignment score with source
            alignment_score = compute_alignment_score(
                t_ji, source_tokens, gen_tokens, alpha, beta
            )
            
            # max(1(t_ji ∈ r_j), C[s_jk, t_ji])
            is_aligned = max(float(in_gold), alignment_score)
            alignment_sum += is_aligned
        
        if denom > 0:
            # HR_i = 1 - alignment_sum / denom
            hr_i = 1.0 - alignment_sum / denom
        else:
            hr_i = 0.0
        
        hallucination_ratios.append(hr_i)
    
    # Compute summary statistics
    if hallucination_ratios:
        avg_hr = sum(hallucination_ratios) / len(hallucination_ratios)
        max_hr = max(hallucination_ratios)
        min_hr = min(hallucination_ratios)
        peak_step = hallucination_ratios.index(max_hr) + 1  # 1-indexed
    else:
        avg_hr = 0.0
        max_hr = 0.0
        min_hr = 0.0
        peak_step = 0
    
    return StepwiseHallucinationResult(
        hallucination_ratios=hallucination_ratios,
        avg_hallucination_ratio=avg_hr,
        max_hallucination_ratio=max_hr,
        min_hallucination_ratio=min_hr,
        peak_hallucination_step=peak_step,
        num_samples=len(dataset),
        max_generation_length=max_step,
    )


# NUMERICAL SENSITIVITY ANALYSIS
# From Section 4.3 of the paper: "Sensitivity Analysis"
#
# Measures how sensitive the output distribution at step i is to small
# perturbations in the source embeddings at position j.
#
# Sensitivity S_i = (1/|S|) Σ_j ||Δz_i|| / ||Δx_j||
#
# Where:
#   - x_j is the embedding at source position j
#   - z_i is the output logits at generation step i
#   - Δ indicates change due to perturbation


@dataclass
class NumericalSensitivityResult:
    """Results from numerical sensitivity analysis."""
    # Sensitivity at each generation step [S_1, S_2, ..., S_max]
    sensitivities: List[float]
    
    # Summary statistics
    avg_sensitivity: float  # Average across all steps
    max_sensitivity: float  # Maximum sensitivity
    min_sensitivity: float  # Minimum sensitivity
    peak_sensitivity_step: int  # Step with highest sensitivity
    
    # Configuration
    source_length: int
    num_gen_steps: int
    perturbation_scale: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensitivities": self.sensitivities,
            "avg_sensitivity": self.avg_sensitivity,
            "max_sensitivity": self.max_sensitivity,
            "min_sensitivity": self.min_sensitivity,
            "peak_sensitivity_step": self.peak_sensitivity_step,
            "source_length": self.source_length,
            "num_gen_steps": self.num_gen_steps,
            "perturbation_scale": self.perturbation_scale,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def numerical_sensitivity_analysis(
    model: torch.nn.Module,
    tokenizer,
    source_text: str,
    device: torch.device,
    perturb_scale: float = 1e-3,
    max_tokens: int = 50,
) -> NumericalSensitivityResult:
    """
    Perform numerical sensitivity analysis as described in the PALM paper.
    
    Measures how much the output distribution changes when source embeddings
    are perturbed. Higher sensitivity indicates stronger source-target dependency.
    
    Args:
        model: PALM model to analyze
        tokenizer: Tokenizer
        source_text: Source text to analyze
        device: Device to run on
        perturb_scale: Scale of random perturbations
        max_tokens: Maximum generation steps to analyze
        
    Returns:
        NumericalSensitivityResult with sensitivity at each step
    """
    model.eval()
    
    # Get the underlying model if wrapped by PEFT
    base_model = model
    if hasattr(model, 'base_model'):
        base = model.base_model
        if hasattr(base, 'model'):
            base_model = base.model
    
    # Encode source
    input_ids = tokenizer.encode(source_text, return_tensors="pt").to(device)
    seq_len = input_ids.size(1)
    
    # Get embedding layer (handle different model structures)
    if hasattr(base_model, 'embeddings'):
        embeddings_layer = base_model.embeddings.word_embeddings
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
        embeddings_layer = base_model.model.embed_tokens
    else:
        raise ValueError("Could not find embedding layer in model")
    
    hidden_dim = embeddings_layer.embedding_dim
    original_embedding_weight = embeddings_layer.weight.data.clone()
    
    # Generate baseline stepwise logits
    stepwise_base_logits: List[torch.Tensor] = []
    gen_ids = input_ids.clone()
    
    for i in range(max_tokens):
        with torch.no_grad():
            dev_type = device.type
            with torch.amp.autocast(device_type=dev_type, enabled=(dev_type == "cuda")):
                try:
                    out = base_model(gen_ids, source_len=torch.tensor([seq_len], device=device))
                    logits = out[0]  # [batch, seq, vocab]
                except Exception:
                    # Fallback for models that don't accept source_len
                    out = base_model(gen_ids)
                    logits = out[0] if isinstance(out, tuple) else out.logits
        
        stepwise_base_logits.append(logits[:, -1, :].clone())
        next_token = torch.argmax(stepwise_base_logits[-1], dim=-1).unsqueeze(-1)
        gen_ids = torch.cat([gen_ids, next_token], dim=-1)
        
        # Stop if EOS generated
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    actual_gen_steps = len(stepwise_base_logits)
    
    # Generate random perturbations for each source position
    random_perturbations = torch.randn(seq_len, hidden_dim, device=device) * perturb_scale
    
    # Compute sensitivity at each generation step
    sensitivities: List[float] = []
    
    for i in range(actual_gen_steps):
        sum_ratios = 0.0
        
        for j in range(seq_len):
            # Reset embeddings
            embeddings_layer.weight.data = original_embedding_weight.clone()
            
            # Perturb embedding for token at position j
            tok_id = input_ids[0, j].item()
            if tok_id < embeddings_layer.num_embeddings:
                embeddings_layer.weight.data[tok_id] += random_perturbations[j]
            
            # Decode up to step i with perturbed embeddings
            pert_gen_ids = input_ids.clone()
            final_step_logits = None
            
            for step_idx in range(i + 1):
                with torch.no_grad():
                    dev_type = device.type
                    with torch.amp.autocast(device_type=dev_type, enabled=(dev_type == "cuda")):
                        try:
                            out_pert = base_model(
                                pert_gen_ids, 
                                source_len=torch.tensor([seq_len], device=device)
                            )
                            logits_pert = out_pert[0]
                        except Exception:
                            out_pert = base_model(pert_gen_ids)
                            logits_pert = out_pert[0] if isinstance(out_pert, tuple) else out_pert.logits
                
                final_step_logits = logits_pert[:, -1, :]
                
                if step_idx < i:
                    next_token_pert = torch.argmax(final_step_logits, dim=-1).unsqueeze(-1)
                    pert_gen_ids = torch.cat([pert_gen_ids, next_token_pert], dim=-1)
            
            # Compute sensitivity: ||Δz_i|| / ||Δx_j||
            delta_logits = final_step_logits - stepwise_base_logits[i]
            logit_norm = delta_logits.norm(p=2).item()
            perturb_norm = random_perturbations[j].norm(p=2).item()
            
            if perturb_norm > 1e-10:
                ratio = logit_norm / perturb_norm
            else:
                ratio = 0.0
            
            sum_ratios += ratio
        
        # Average sensitivity across all source positions
        avg_sensitivity = sum_ratios / seq_len if seq_len > 0 else 0.0
        sensitivities.append(avg_sensitivity)
    
    # Restore original embeddings
    embeddings_layer.weight.data = original_embedding_weight.clone()
    
    # Compute summary statistics
    if sensitivities:
        avg_sens = sum(sensitivities) / len(sensitivities)
        max_sens = max(sensitivities)
        min_sens = min(sensitivities)
        peak_step = sensitivities.index(max_sens) + 1  # 1-indexed
    else:
        avg_sens = 0.0
        max_sens = 0.0
        min_sens = 0.0
        peak_step = 0
    
    return NumericalSensitivityResult(
        sensitivities=sensitivities,
        avg_sensitivity=avg_sens,
        max_sensitivity=max_sens,
        min_sensitivity=min_sens,
        peak_sensitivity_step=peak_step,
        source_length=seq_len,
        num_gen_steps=actual_gen_steps,
        perturbation_scale=perturb_scale,
    )


# COMBINED PAPER METRICS EVALUATOR
@dataclass
class PaperMetricsResult:
    """Combined results from all paper-based metrics."""
    stepwise_hallucination: Optional[StepwiseHallucinationResult] = None
    numerical_sensitivity: Optional[NumericalSensitivityResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.stepwise_hallucination:
            result["stepwise_hallucination"] = self.stepwise_hallucination.to_dict()
        if self.numerical_sensitivity:
            result["numerical_sensitivity"] = self.numerical_sensitivity.to_dict()
        return result
    
    def to_wandb_dict(self, prefix: str = "paper") -> Dict[str, float]:
        """Convert to flat dict for wandb logging."""
        result = {}
        if self.stepwise_hallucination:
            sh = self.stepwise_hallucination
            result.update({
                f"{prefix}/hr/avg": sh.avg_hallucination_ratio,
                f"{prefix}/hr/max": sh.max_hallucination_ratio,
                f"{prefix}/hr/min": sh.min_hallucination_ratio,
                f"{prefix}/hr/peak_step": sh.peak_hallucination_step,
            })
        if self.numerical_sensitivity:
            ns = self.numerical_sensitivity
            result.update({
                f"{prefix}/sensitivity/avg": ns.avg_sensitivity,
                f"{prefix}/sensitivity/max": ns.max_sensitivity,
                f"{prefix}/sensitivity/min": ns.min_sensitivity,
                f"{prefix}/sensitivity/peak_step": ns.peak_sensitivity_step,
            })
        return result


class PaperMetricsEvaluator:
    """
    Evaluator for paper-based PALM metrics.
    
    Combines:
    1. Stepwise Hallucination Analysis
    2. Numerical Sensitivity Analysis
    
    Usage:
        evaluator = PaperMetricsEvaluator(model, tokenizer)
        result = evaluator.evaluate(eval_samples)
        wandb.log(result.to_wandb_dict())
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        max_gen_length: int = 64,
        perturb_scale: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Args:
            model: PALM model to evaluate
            tokenizer: Tokenizer
            max_gen_length: Maximum generation length for hallucination analysis
            perturb_scale: Perturbation scale for sensitivity analysis
            alpha: Alignment score offset parameter
            beta: Alignment score scaling parameter
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_gen_length = max_gen_length
        self.perturb_scale = perturb_scale
        self.alpha = alpha
        self.beta = beta
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def run_stepwise_hallucination(
        self,
        eval_samples: List[Dict[str, str]],
    ) -> StepwiseHallucinationResult:
        """Run stepwise hallucination analysis."""
        return compute_stepwise_hallucination_ratio(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=eval_samples,
            device=self.device,
            max_gen_length=self.max_gen_length,
            alpha=self.alpha,
            beta=self.beta,
        )
    
    def run_sensitivity_analysis(
        self,
        source_text: str,
        max_tokens: int = 20,
    ) -> NumericalSensitivityResult:
        """Run numerical sensitivity analysis on a single source."""
        return numerical_sensitivity_analysis(
            model=self.model,
            tokenizer=self.tokenizer,
            source_text=source_text,
            device=self.device,
            perturb_scale=self.perturb_scale,
            max_tokens=max_tokens,
        )
    
    def evaluate(
        self,
        eval_samples: List[Dict[str, str]],
        run_hallucination: bool = True,
        run_sensitivity: bool = True,
        sensitivity_sample_idx: int = 0,
        sensitivity_max_tokens: int = 20,
    ) -> PaperMetricsResult:
        """
        Run all paper-based evaluations.
        
        Args:
            eval_samples: List of {"source": str, "target": str} dicts
            run_hallucination: Whether to run stepwise hallucination analysis
            run_sensitivity: Whether to run numerical sensitivity analysis
            sensitivity_sample_idx: Which sample to use for sensitivity (expensive)
            sensitivity_max_tokens: Max tokens for sensitivity analysis
            
        Returns:
            PaperMetricsResult with all computed metrics
        """
        result = PaperMetricsResult()
        
        if run_hallucination:
            print("Running stepwise hallucination analysis...")
            result.stepwise_hallucination = self.run_stepwise_hallucination(eval_samples)
            print(f"  Average HR: {result.stepwise_hallucination.avg_hallucination_ratio:.3f}")
            print(f"  Peak HR at step {result.stepwise_hallucination.peak_hallucination_step}: "
                  f"{result.stepwise_hallucination.max_hallucination_ratio:.3f}")
        
        if run_sensitivity and eval_samples:
            source_text = eval_samples[sensitivity_sample_idx].get("source", "")
            if source_text:
                print(f"Running numerical sensitivity analysis on sample {sensitivity_sample_idx}...")
                result.numerical_sensitivity = self.run_sensitivity_analysis(
                    source_text, sensitivity_max_tokens
                )
                print(f"  Average sensitivity: {result.numerical_sensitivity.avg_sensitivity:.4f}")
                print(f"  Peak sensitivity at step {result.numerical_sensitivity.peak_sensitivity_step}")
        
        return result
    
    def format_results(self, result: PaperMetricsResult) -> str:
        """Format results as readable summary."""
        lines = [
            "",
            "=" * 70,
            "PALM PAPER-BASED METRICS",
            "=" * 70,
        ]
        
        if result.stepwise_hallucination:
            sh = result.stepwise_hallucination
            lines.extend([
                "",
                "STEPWISE HALLUCINATION ANALYSIS",
                "-" * 70,
                f"  Samples analyzed:     {sh.num_samples}",
                f"  Max generation steps: {sh.max_generation_length}",
                "",
                f"  Average HR:           {sh.avg_hallucination_ratio:.3f}",
                f"  Min HR:               {sh.min_hallucination_ratio:.3f}",
                f"  Max HR:               {sh.max_hallucination_ratio:.3f}",
                f"  Peak at step:         {sh.peak_hallucination_step}",
                "",
                "  HR by step (first 10):",
            ])
            for i, hr in enumerate(sh.hallucination_ratios[:10], 1):
                lines.append(f"    Step {i:2d}: {hr:.3f} {'█' * int(hr * 20)}")
        
        if result.numerical_sensitivity:
            ns = result.numerical_sensitivity
            lines.extend([
                "",
                "NUMERICAL SENSITIVITY ANALYSIS",
                "-" * 70,
                f"  Source length:        {ns.source_length}",
                f"  Generation steps:     {ns.num_gen_steps}",
                f"  Perturbation scale:   {ns.perturbation_scale}",
                "",
                f"  Average sensitivity:  {ns.avg_sensitivity:.4f}",
                f"  Min sensitivity:      {ns.min_sensitivity:.4f}",
                f"  Max sensitivity:      {ns.max_sensitivity:.4f}",
                f"  Peak at step:         {ns.peak_sensitivity_step}",
                "",
                "  Sensitivity by step (first 10):",
            ])
            for i, sens in enumerate(ns.sensitivities[:10], 1):
                bar_len = min(int(sens * 100), 30)
                lines.append(f"    Step {i:2d}: {sens:.4f} {'█' * bar_len}")
        
        lines.extend(["", "=" * 70])
        return "\n".join(lines)


# PLOTTING FUNCTIONS
def plot_stepwise_hallucination(
    result: StepwiseHallucinationResult,
    output_path: str = "stepwise_hallucination.png",
) -> None:
    """
    Plot stepwise hallucination ratio curve.
    
    Args:
        result: StepwiseHallucinationResult from analysis
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    steps = list(range(1, len(result.hallucination_ratios) + 1))
    hrs = result.hallucination_ratios
    
    # Plot HR curve
    ax.plot(steps, hrs, 'o-', color='#E74C3C', linewidth=2, markersize=6,
            markeredgecolor='white', markeredgewidth=1.5, label='Hallucination Ratio')
    
    # Fill area under curve
    ax.fill_between(steps, hrs, alpha=0.3, color='#E74C3C')
    
    # Highlight peak
    peak_idx = result.peak_hallucination_step - 1
    if 0 <= peak_idx < len(hrs):
        ax.scatter([result.peak_hallucination_step], [result.max_hallucination_ratio],
                   s=200, c='#C0392B', zorder=5, edgecolor='white', linewidth=3)
        ax.annotate(f'Peak: {result.max_hallucination_ratio:.3f}',
                   xy=(result.peak_hallucination_step, result.max_hallucination_ratio),
                   xytext=(result.peak_hallucination_step + 2, result.max_hallucination_ratio + 0.05),
                   fontsize=11, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2))
    
    # Average line
    ax.axhline(y=result.avg_hallucination_ratio, color='#3498DB', linestyle='--',
               linewidth=2, label=f'Average: {result.avg_hallucination_ratio:.3f}')
    
    ax.set_xlabel('Generation Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hallucination Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Stepwise Hallucination Analysis\n(Higher = more hallucination at that step)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, len(steps) + 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved stepwise hallucination plot to {output_path}")


def plot_sensitivity_analysis(
    result: NumericalSensitivityResult,
    output_path: str = "sensitivity_analysis.png",
) -> None:
    """
    Plot numerical sensitivity analysis curve.
    
    Args:
        result: NumericalSensitivityResult from analysis
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    steps = list(range(1, len(result.sensitivities) + 1))
    sens = result.sensitivities
    
    # Plot sensitivity curve
    ax.plot(steps, sens, 's-', color='#27AE60', linewidth=2, markersize=6,
            markeredgecolor='white', markeredgewidth=1.5, label='Sensitivity')
    
    # Fill area under curve
    ax.fill_between(steps, sens, alpha=0.3, color='#27AE60')
    
    # Highlight peak
    peak_idx = result.peak_sensitivity_step - 1
    if 0 <= peak_idx < len(sens):
        ax.scatter([result.peak_sensitivity_step], [result.max_sensitivity],
                   s=200, c='#1E8449', zorder=5, edgecolor='white', linewidth=3)
        ax.annotate(f'Peak: {result.max_sensitivity:.4f}',
                   xy=(result.peak_sensitivity_step, result.max_sensitivity),
                   xytext=(result.peak_sensitivity_step + 2, result.max_sensitivity * 1.1),
                   fontsize=11, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#1E8449', lw=2))
    
    # Average line
    ax.axhline(y=result.avg_sensitivity, color='#3498DB', linestyle='--',
               linewidth=2, label=f'Average: {result.avg_sensitivity:.4f}')
    
    ax.set_xlabel('Generation Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sensitivity (||Δlogits|| / ||Δembedding||)', fontsize=12, fontweight='bold')
    ax.set_title('Numerical Sensitivity Analysis\n(Higher = more dependent on source)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(steps) + 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved sensitivity analysis plot to {output_path}")


def plot_combined_paper_metrics(
    result: PaperMetricsResult,
    output_path: str = "paper_metrics.png",
) -> None:
    """
    Plot both stepwise hallucination and sensitivity analysis side by side.
    
    Args:
        result: PaperMetricsResult with both analyses
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Stepwise Hallucination
    if result.stepwise_hallucination:
        ax1 = axes[0]
        sh = result.stepwise_hallucination
        
        steps = list(range(1, len(sh.hallucination_ratios) + 1))
        hrs = sh.hallucination_ratios
        
        ax1.plot(steps, hrs, 'o-', color='#E74C3C', linewidth=2, markersize=5)
        ax1.fill_between(steps, hrs, alpha=0.3, color='#E74C3C')
        ax1.axhline(y=sh.avg_hallucination_ratio, color='#3498DB', linestyle='--', linewidth=2)
        
        ax1.set_xlabel('Generation Step', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Hallucination Ratio', fontsize=11, fontweight='bold')
        ax1.set_title(f'Stepwise Hallucination\nAvg: {sh.avg_hallucination_ratio:.3f}',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
    
    # Plot 2: Sensitivity Analysis
    if result.numerical_sensitivity:
        ax2 = axes[1]
        ns = result.numerical_sensitivity
        
        steps = list(range(1, len(ns.sensitivities) + 1))
        sens = ns.sensitivities
        
        ax2.plot(steps, sens, 's-', color='#27AE60', linewidth=2, markersize=5)
        ax2.fill_between(steps, sens, alpha=0.3, color='#27AE60')
        ax2.axhline(y=ns.avg_sensitivity, color='#3498DB', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Generation Step', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Sensitivity', fontsize=11, fontweight='bold')
        ax2.set_title(f'Numerical Sensitivity\nAvg: {ns.avg_sensitivity:.4f}',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    fig.suptitle('PALM Paper-Based Metrics', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved combined paper metrics plot to {output_path}")


# SAVE/LOAD FUNCTIONS
def save_paper_metrics(
    result: PaperMetricsResult,
    output_file: str = "paper_metrics.json",
) -> None:
    """Save paper metrics results to JSON file."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Saved paper metrics to {output_file}")


def load_paper_metrics(input_file: str) -> Optional[PaperMetricsResult]:
    """Load paper metrics from JSON file."""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        result = PaperMetricsResult()
        
        if "stepwise_hallucination" in data:
            sh_data = data["stepwise_hallucination"]
            result.stepwise_hallucination = StepwiseHallucinationResult(**sh_data)
        
        if "numerical_sensitivity" in data:
            ns_data = data["numerical_sensitivity"]
            result.numerical_sensitivity = NumericalSensitivityResult(**ns_data)
        
        return result
    except Exception as e:
        print(f"Failed to load paper metrics: {e}")
        return None
