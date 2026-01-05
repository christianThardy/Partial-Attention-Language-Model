"""
Lambda (SAE Weight) Sweep for PALM.

Provides infrastructure for controlled comparison of different SAE weight values (λ).

The sweep uses:
- Same random seed
- Same data slice
- Same number of steps
- Fixed λ (no ramp) for each run

This produces the "first paper-ready figure": a clean tradeoff curve showing
higher λ ⇒ better grounding, worse fluency/diversity.

Usage:
    from palm.evaluation import LambdaSweepRunner, plot_lambda_tradeoff
    
    runner = LambdaSweepRunner(model_fn, tokenizer, train_dataloader, eval_samples)
    results = runner.run_sweep(lambda_values=[0.0, 0.01, 0.05, 0.1])
    plot_lambda_tradeoff(results, "lambda_tradeoff.png")
"""

import json
import os
import gc
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class LambdaSweepConfig:
    """Configuration for a single λ sweep run."""
    lambda_value: float
    seed: int = 42
    max_steps: int = 500
    eval_every: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class LambdaSweepResult:
    """Results from a single λ value run."""
    lambda_value: float
    seed: int
    max_steps: int
    
    # Final metrics
    final_lm_loss: float
    final_perplexity: float
    baseline_perplexity: float
    ppl_drift: float
    
    # Faithfulness
    faithfulness_score: float  # entity_precision
    hallucination_rate: float
    
    # Degeneration
    distinct_2: float
    distinct_3: float
    repetition_rate: float
    
    # Composite
    palm_score: float
    
    # Training progression (for plotting)
    step_history: List[int] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    ppl_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), separators=(',', ':'))
    
    @classmethod
    def from_json_line(cls, line: str) -> "LambdaSweepResult":
        return cls(**json.loads(line))


def save_sweep_results(
    results: List[LambdaSweepResult],
    output_file: str = "lambda_sweep_results.jsonl"
) -> None:
    """Save sweep results to JSONL file."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result.to_json_line() + "\n")


def load_sweep_results(input_file: str) -> List[LambdaSweepResult]:
    """Load sweep results from JSONL file."""
    results = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(LambdaSweepResult.from_json_line(line))
    return results


class LambdaSweepRunner:
    """
    Run controlled sweep over λ (SAE weight) values.
    
    Each run uses identical:
    - Random seed
    - Data
    - Number of steps
    - Model architecture
    
    Only the SAE weight differs, enabling clean comparison.
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        tokenizer,
        train_dataloader,
        eval_samples: List[Dict[str, str]],
        device: Optional[torch.device] = None,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-4,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model_factory: Callable that returns a fresh model instance
            tokenizer: Tokenizer
            train_dataloader: Training data loader (will be reused)
            eval_samples: List of {"source": str, "target": str} for evaluation
            device: Device to train on
            gradient_accumulation_steps: Gradient accumulation
            learning_rate: Learning rate
            max_grad_norm: Gradient clipping
        """
        self.model_factory = model_factory
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_samples = eval_samples
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _get_baseline_ppl(self, model: nn.Module) -> float:
        """Get baseline perplexity before training."""
        from .lightweight import compute_generation_metrics_fast
        from .comprehensive import compute_perplexity
        
        model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for sample in self.eval_samples[:4]:  # Use first 4 for baseline
                source_text = sample["source"]
                target_text = sample.get("target", "")
                
                if not target_text:
                    continue
                
                full_text = source_text + target_text
                source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(self.device)
                full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
                
                source_len = source_ids.size(1)
                labels = full_ids.clone()
                labels[:, :source_len] = -100
                
                outputs = model(full_ids, labels=labels, source_len=torch.tensor([source_len], device=self.device))
                _, _, lm_loss, _ = outputs
                
                if lm_loss is not None:
                    total_loss += lm_loss.item()
                    count += 1
        
        if count == 0:
            return 10.0  # Default baseline
        
        return compute_perplexity(total_loss / count)
    
    def _evaluate(
        self,
        model: nn.Module,
        baseline_ppl: float,
    ) -> Dict[str, float]:
        """Run evaluation and return metrics."""
        from .lightweight import compute_generation_metrics_fast
        from .comprehensive import (
            compute_perplexity,
            compute_distinct_ngrams,
            compute_repetition_rate,
        )
        
        model.eval()
        
        total_loss = 0.0
        count = 0
        all_faith = []
        all_hal = []
        all_rep = []
        all_d2 = []
        all_d3 = []
        
        with torch.no_grad():
            for sample in self.eval_samples:
                source_text = sample["source"]
                target_text = sample.get("target", "")
                
                source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(self.device)
                source_len = source_ids.size(1)
                
                # Get loss
                if target_text:
                    full_text = source_text + target_text
                    full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
                    labels = full_ids.clone()
                    labels[:, :source_len] = -100
                    
                    outputs = model(full_ids, labels=labels, source_len=torch.tensor([source_len], device=self.device))
                    _, _, lm_loss, _ = outputs
                    
                    if lm_loss is not None:
                        total_loss += lm_loss.item()
                        count += 1
                
                # Generate
                generated = model.generate(
                    source_ids,
                    max_length=source_len + 50,
                    do_sample=False,
                    use_cache=True,
                )
                
                gen_tokens = generated[0, source_len:].tolist()
                gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                # Compute metrics
                rep, d2, prec, hal = compute_generation_metrics_fast(
                    gen_tokens, source_text, gen_text
                )
                d3 = compute_distinct_ngrams(gen_tokens, 3)
                
                all_rep.append(rep)
                all_d2.append(d2)
                all_d3.append(d3)
                all_faith.append(prec)
                all_hal.append(hal)
        
        # Aggregate
        avg_loss = total_loss / max(count, 1)
        ppl = compute_perplexity(avg_loss)
        
        avg_faith = sum(all_faith) / len(all_faith) if all_faith else 1.0
        avg_hal = sum(all_hal) / len(all_hal) if all_hal else 0.0
        avg_rep = sum(all_rep) / len(all_rep) if all_rep else 0.0
        avg_d2 = sum(all_d2) / len(all_d2) if all_d2 else 1.0
        avg_d3 = sum(all_d3) / len(all_d3) if all_d3 else 1.0
        
        ppl_drift = max(0, ppl - baseline_ppl)
        
        # PALM score
        palm_score = avg_faith - 1.0 * avg_hal - 0.5 * avg_rep - 0.1 * ppl_drift
        
        return {
            "lm_loss": avg_loss,
            "perplexity": ppl,
            "ppl_drift": ppl_drift,
            "faithfulness": avg_faith,
            "hallucination": avg_hal,
            "repetition_rate": avg_rep,
            "distinct_2": avg_d2,
            "distinct_3": avg_d3,
            "palm_score": palm_score,
        }
    
    def _train_single_lambda(
        self,
        lambda_value: float,
        seed: int,
        max_steps: int,
        eval_every: int,
    ) -> LambdaSweepResult:
        """Train with a single λ value."""
        from torch.optim import AdamW
        
        print(f"\n{'='*60}")
        print(f"λ = {lambda_value}")
        print(f"{'='*60}")
        
        # Set seed
        self._set_seed(seed)
        
        # Create fresh model
        model = self.model_factory()
        model = model.to(self.device)
        
        # Set SAE weight (fixed, no ramp)
        base_model = model.base_model.model if hasattr(model, 'base_model') else model
        if hasattr(base_model, 'model'):
            base_model = base_model.model
        base_model.sae_weight = lambda_value
        
        # Get baseline PPL
        baseline_ppl = self._get_baseline_ppl(model)
        print(f"  Baseline PPL: {baseline_ppl:.2f}")
        
        # Create optimizer
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        # Training loop
        step_history = []
        loss_history = []
        ppl_history = []
        
        model.train()
        global_step = 0
        optimizer.zero_grad()
        
        data_iter = iter(self.train_dataloader)
        
        progress = tqdm(range(max_steps), desc=f"λ={lambda_value}")
        for step in progress:
            # Get batch (cycle if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            source_len = batch["source_len"].to(self.device)
            
            # Forward
            _, combined_loss, loss, sae_loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                source_len=source_len,
            )
            
            # Backward
            scaled_loss = combined_loss / self.gradient_accumulation_steps
            scaled_loss.backward()
            
            # Optimizer step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Periodic eval
            if (step + 1) % eval_every == 0:
                metrics = self._evaluate(model, baseline_ppl)
                step_history.append(step + 1)
                loss_history.append(metrics["lm_loss"])
                ppl_history.append(metrics["perplexity"])
                model.train()
        
        # Final evaluation
        final_metrics = self._evaluate(model, baseline_ppl)
        
        # Cleanup
        del model
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return LambdaSweepResult(
            lambda_value=lambda_value,
            seed=seed,
            max_steps=max_steps,
            final_lm_loss=final_metrics["lm_loss"],
            final_perplexity=final_metrics["perplexity"],
            baseline_perplexity=baseline_ppl,
            ppl_drift=final_metrics["ppl_drift"],
            faithfulness_score=final_metrics["faithfulness"],
            hallucination_rate=final_metrics["hallucination"],
            distinct_2=final_metrics["distinct_2"],
            distinct_3=final_metrics["distinct_3"],
            repetition_rate=final_metrics["repetition_rate"],
            palm_score=final_metrics["palm_score"],
            step_history=step_history,
            loss_history=loss_history,
            ppl_history=ppl_history,
        )
    
    def run_sweep(
        self,
        lambda_values: List[float] = None,
        seed: int = 42,
        max_steps: int = 500,
        eval_every: int = 100,
        output_dir: str = "./sweeps",
    ) -> List[LambdaSweepResult]:
        """
        Run controlled sweep over λ values.
        
        Args:
            lambda_values: List of λ values to test (default: [0.0, 0.01, 0.05, 0.1])
            seed: Random seed for reproducibility
            max_steps: Max training steps per run
            eval_every: Evaluate every N steps
            output_dir: Directory to save results
            
        Returns:
            List of LambdaSweepResult, one per λ value
        """
        if lambda_values is None:
            lambda_values = [0.0, 0.01, 0.05, 0.1]
        
        print(f"\n{'='*70}")
        print(f"LAMBDA SWEEP: {len(lambda_values)} values")
        print(f"λ ∈ {lambda_values}")
        print(f"Seed: {seed}, Steps: {max_steps}")
        print(f"{'='*70}")
        
        results = []
        
        for λ in lambda_values:
            result = self._train_single_lambda(
                lambda_value=λ,
                seed=seed,
                max_steps=max_steps,
                eval_every=eval_every,
            )
            results.append(result)
            
            # Save intermediate results
            os.makedirs(output_dir, exist_ok=True)
            save_sweep_results(results, f"{output_dir}/lambda_sweep.jsonl")
        
        # Print summary
        print(f"\n{'='*70}")
        print("SWEEP COMPLETE - SUMMARY")
        print(f"{'='*70}")
        print(f"{'λ':>8} {'PPL':>8} {'Drift':>8} {'Faith':>8} {'Hal':>8} {'D2':>8} {'PALM':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r.lambda_value:>8.3f} {r.final_perplexity:>8.2f} {r.ppl_drift:>8.2f} "
                  f"{r.faithfulness_score:>8.3f} {r.hallucination_rate:>8.3f} "
                  f"{r.distinct_2:>8.3f} {r.palm_score:>8.3f}")
        print("=" * 70)
        
        return results


def plot_lambda_tradeoff(
    results: List[LambdaSweepResult],
    output_path: str = "lambda_tradeoff.png",
) -> None:
    """
    Plot Pareto curve: λ vs (PPL drift, faithfulness, diversity).
    
    This is the "first paper-ready figure"!
    
    Args:
        results: List of LambdaSweepResult from sweep
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    if not results:
        print("No results to plot")
        return
    
    # Sort by lambda
    results = sorted(results, key=lambda r: r.lambda_value)
    
    lambdas = [r.lambda_value for r in results]
    ppl_drifts = [r.ppl_drift for r in results]
    faithfulness = [r.faithfulness_score for r in results]
    hallucination = [r.hallucination_rate for r in results]
    distinct_2 = [r.distinct_2 for r in results]
    palm_scores = [r.palm_score for r in results]
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Common styling
    marker_size = 120
    line_width = 2.5
    
    # 1. PPL Drift vs λ
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(lambdas, ppl_drifts, 'o-', color='#E74C3C', markersize=10, linewidth=line_width)
    ax1.scatter(lambdas, ppl_drifts, s=marker_size, c='#E74C3C', zorder=5)
    ax1.set_xlabel('SAE Weight (λ)', fontsize=12)
    ax1.set_ylabel('PPL Drift from Baseline', fontsize=12)
    ax1.set_title('Fluency Cost', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.01, max(lambdas) * 1.1)
    
    # 2. Faithfulness vs λ
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(lambdas, faithfulness, 's-', color='#27AE60', markersize=10, linewidth=line_width)
    ax2.scatter(lambdas, faithfulness, s=marker_size, c='#27AE60', zorder=5, marker='s')
    ax2.set_xlabel('SAE Weight (λ)', fontsize=12)
    ax2.set_ylabel('Entity Precision', fontsize=12)
    ax2.set_title('Grounding Benefit', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.01, max(lambdas) * 1.1)
    
    # 3. Diversity vs λ
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(lambdas, distinct_2, '^-', color='#F39C12', markersize=10, linewidth=line_width)
    ax3.scatter(lambdas, distinct_2, s=marker_size, c='#F39C12', zorder=5, marker='^')
    ax3.set_xlabel('SAE Weight (λ)', fontsize=12)
    ax3.set_ylabel('Distinct-2', fontsize=12)
    ax3.set_title('Diversity Impact', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.01, max(lambdas) * 1.1)
    
    # 4. Tradeoff: PPL Drift vs Faithfulness (the key plot!)
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(ppl_drifts, faithfulness, c=lambdas, cmap='viridis', 
                         s=marker_size * 1.5, edgecolors='black', linewidths=1.5)
    ax4.plot(ppl_drifts, faithfulness, 'k--', alpha=0.5, linewidth=1.5)
    
    # Add λ labels
    for i, (x, y, l) in enumerate(zip(ppl_drifts, faithfulness, lambdas)):
        ax4.annotate(f'λ={l}', (x, y), textcoords="offset points", 
                    xytext=(10, 5), fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('PPL Drift (↓ better fluency)', fontsize=12)
    ax4.set_ylabel('Entity Precision (↑ better grounding)', fontsize=12)
    ax4.set_title('Tradeoff Curve: Fluency vs Grounding', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('λ', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 5. PALM Score vs λ
    ax5 = fig.add_subplot(gs[1, 1])
    colors = ['#2ECC71' if s > 0 else '#E74C3C' for s in palm_scores]
    ax5.bar(range(len(lambdas)), palm_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_xticks(range(len(lambdas)))
    ax5.set_xticklabels([f'{l}' for l in lambdas])
    ax5.set_xlabel('SAE Weight (λ)', fontsize=12)
    ax5.set_ylabel('PALM Composite Score', fontsize=12)
    ax5.set_title('Composite Score', fontsize=14, fontweight='bold')
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Hallucination vs λ
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(lambdas, hallucination, 'd-', color='#9B59B6', markersize=10, linewidth=line_width)
    ax6.scatter(lambdas, hallucination, s=marker_size, c='#9B59B6', zorder=5, marker='d')
    ax6.set_xlabel('SAE Weight (λ)', fontsize=12)
    ax6.set_ylabel('Hallucination Rate', fontsize=12)
    ax6.set_title('Hallucination Reduction', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-0.01, max(lambdas) * 1.1)
    
    # Add title
    fig.suptitle('SAE Weight (λ) Sweep Analysis\nHigher λ → Better Grounding, Worse Fluency/Diversity', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved λ tradeoff plot to {output_path}")


def plot_lambda_training_curves(
    results: List[LambdaSweepResult],
    output_path: str = "lambda_training_curves.png",
) -> None:
    """
    Plot training curves for each λ value.
    
    Args:
        results: List of LambdaSweepResult from sweep
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.")
        return
    
    if not results or not any(r.step_history for r in results):
        print("No training history to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis([i / len(results) for i in range(len(results))])
    
    for i, result in enumerate(sorted(results, key=lambda r: r.lambda_value)):
        if result.step_history:
            label = f'λ={result.lambda_value}'
            axes[0].plot(result.step_history, result.loss_history, 
                        color=colors[i], label=label, linewidth=2)
            axes[1].plot(result.step_history, result.ppl_history,
                        color=colors[i], label=label, linewidth=2)
    
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('LM Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Perplexity Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_path}")
