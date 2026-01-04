"""
Checkpoint Scoreboard for PALM.

Saves a JSON line per checkpoint containing all key metrics for Pareto frontier analysis.
This enables tracking phase changes during training:
- Early training: LM coherence improves
- Later training: faithfulness improves
- Too much SAE: often hurts distinctness

Usage:
    from palm.evaluation import CheckpointScoreboard, save_checkpoint_scoreboard
    
    scoreboard = CheckpointScoreboard(...)
    save_checkpoint_scoreboard(scoreboard, "checkpoints/scoreboard.jsonl")
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any


@dataclass
class CheckpointScoreboard:
    """
    Single JSON line per checkpoint for Pareto analysis.
    
    Contains all key PALM metrics organized by category:
    - Metadata (step, epoch, paths)
    - LM quality (loss, PPL, drift)
    - SAE metrics (loss, ratio, weight)
    - Mask compliance (future leakage, source attention - global + per-layer)
    - Degeneration (distinct-n, repetition)
    - Faithfulness (entity precision, hallucination, copy rate)
    - Composite (PALM score)
    """
    # Metadata
    step: int
    epoch: int
    checkpoint_path: Optional[str] = None
    timestamp: Optional[str] = None
    
    # LM quality
    lm_loss: float = 0.0
    perplexity: float = 0.0
    ppl_drift: float = 0.0  # vs baseline
    
    # SAE
    sae_loss: float = 0.0
    sae_ratio: float = 0.0  # sae_loss / total_loss
    sae_weight: float = 0.0  # current λ
    
    # Mask compliance - global
    future_leakage_global: float = 0.0
    source_attention_mass_global: float = 0.0
    
    # Mask compliance - per-layer (for detailed analysis)
    future_leakage_per_layer: List[float] = field(default_factory=list)
    source_attention_mass_per_layer: List[float] = field(default_factory=list)
    
    # Degeneration
    distinct_1: float = 1.0
    distinct_2: float = 1.0
    distinct_3: float = 1.0
    repetition_rate: float = 0.0
    avg_entropy: float = 0.0
    
    # Faithfulness
    entity_precision: float = 1.0
    entity_recall: float = 1.0
    entity_hallucination_rate: float = 0.0
    copy_rate: float = 0.0
    source_coverage: float = 0.0
    
    # Composite
    palm_score: float = 0.0
    
    # Training phase info
    training_phase: str = ""  # "LM_WARMUP", "SAE_RAMP", "STABLE"
    
    def to_json_line(self) -> str:
        """Convert to JSON line (single line, no pretty printing)."""
        return json.dumps(asdict(self), separators=(',', ':'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_json_line(cls, line: str) -> "CheckpointScoreboard":
        """Parse from JSON line."""
        data = json.loads(line)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointScoreboard":
        """Create from dictionary."""
        return cls(**data)


def save_checkpoint_scoreboard(
    scoreboard: CheckpointScoreboard,
    output_file: str = "checkpoint_scoreboard.jsonl"
) -> None:
    """
    Append checkpoint scoreboard to JSONL file.
    
    Args:
        scoreboard: CheckpointScoreboard to save
        output_file: Path to JSONL file (will be created if doesn't exist)
    """
    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Append JSON line
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(scoreboard.to_json_line() + "\n")


def load_checkpoint_scoreboard(
    input_file: str = "checkpoint_scoreboard.jsonl"
) -> List[CheckpointScoreboard]:
    """
    Load all checkpoint scoreboards from JSONL file.
    
    Args:
        input_file: Path to JSONL file
        
    Returns:
        List of CheckpointScoreboard objects, sorted by step
    """
    scoreboards = []
    
    if not os.path.exists(input_file):
        return scoreboards
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    scoreboards.append(CheckpointScoreboard.from_json_line(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue
    
    # Sort by step
    scoreboards.sort(key=lambda x: x.step)
    return scoreboards


def get_pareto_frontier(
    scoreboards: List[CheckpointScoreboard],
    x_metric: str = "ppl_drift",
    y_metric: str = "entity_precision",
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> List[CheckpointScoreboard]:
    """
    Extract Pareto frontier from scoreboards.
    
    A point is on the Pareto frontier if no other point is strictly better
    on both metrics.
    
    Args:
        scoreboards: List of CheckpointScoreboard objects
        x_metric: Metric name for X axis (e.g., "ppl_drift")
        y_metric: Metric name for Y axis (e.g., "entity_precision")
        minimize_x: If True, lower X is better
        maximize_y: If True, higher Y is better
        
    Returns:
        List of CheckpointScoreboard objects on the Pareto frontier
    """
    if not scoreboards:
        return []
    
    # Get values
    points = []
    for sb in scoreboards:
        x = getattr(sb, x_metric, 0)
        y = getattr(sb, y_metric, 0)
        points.append((x, y, sb))
    
    # Find Pareto frontier
    frontier = []
    for x1, y1, sb1 in points:
        is_dominated = False
        for x2, y2, sb2 in points:
            if sb1 is sb2:
                continue
            
            # Check if sb2 dominates sb1
            x_better = (x2 < x1) if minimize_x else (x2 > x1)
            x_equal = (x2 == x1)
            y_better = (y2 > y1) if maximize_y else (y2 < y1)
            y_equal = (y2 == y1)
            
            # sb2 dominates if it's better or equal on both, and strictly better on at least one
            if ((x_better or x_equal) and (y_better or y_equal) and 
                (x_better or y_better)):
                is_dominated = True
                break
        
        if not is_dominated:
            frontier.append(sb1)
    
    # Sort frontier by x metric
    frontier.sort(key=lambda sb: getattr(sb, x_metric, 0), reverse=not minimize_x)
    return frontier


def plot_pareto_analysis(
    scoreboards: List[CheckpointScoreboard],
    output_path: str = "pareto_analysis.png",
    show_frontier: bool = True,
) -> None:
    """
    Plot Pareto analysis: PPL drift vs Faithfulness with training progression.
    
    Args:
        scoreboards: List of CheckpointScoreboard objects
        output_path: Path to save the plot
        show_frontier: If True, highlight Pareto frontier points
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    if not scoreboards:
        print("No scoreboards to plot")
        return
    
    # Extract data
    steps = [sb.step for sb in scoreboards]
    ppl_drift = [sb.ppl_drift for sb in scoreboards]
    faithfulness = [sb.entity_precision for sb in scoreboards]
    distinct_2 = [sb.distinct_2 for sb in scoreboards]
    sae_weights = [sb.sae_weight for sb in scoreboards]
    palm_scores = [sb.palm_score for sb in scoreboards]
    
    # Get Pareto frontier
    frontier = get_pareto_frontier(scoreboards) if show_frontier else []
    frontier_steps = {sb.step for sb in frontier}
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. PPL Drift vs Step (colored by SAE weight)
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(steps, ppl_drift, c=sae_weights, cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('PPL Drift from Baseline')
    ax1.set_title('Fluency Cost Over Training')
    plt.colorbar(scatter1, ax=ax1, label='SAE Weight (λ)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Faithfulness vs Step
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(steps, faithfulness, c=sae_weights, cmap='viridis', s=50, alpha=0.7)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Entity Precision (Faithfulness)')
    ax2.set_title('Grounding Over Training')
    plt.colorbar(scatter2, ax=ax2, label='SAE Weight (λ)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distinct-2 vs Step
    ax3 = axes[0, 2]
    scatter3 = ax3.scatter(steps, distinct_2, c=sae_weights, cmap='viridis', s=50, alpha=0.7)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Distinct-2 (Diversity)')
    ax3.set_title('Diversity Over Training')
    plt.colorbar(scatter3, ax=ax3, label='SAE Weight (λ)')
    ax3.grid(True, alpha=0.3)
    
    # 4. PPL Drift vs Faithfulness (Pareto plot!)
    ax4 = axes[1, 0]
    colors = ['red' if sb.step in frontier_steps else 'blue' for sb in scoreboards]
    sizes = [100 if sb.step in frontier_steps else 50 for sb in scoreboards]
    ax4.scatter(ppl_drift, faithfulness, c=colors, s=sizes, alpha=0.7)
    
    # Connect frontier points
    if frontier:
        frontier_x = [sb.ppl_drift for sb in frontier]
        frontier_y = [sb.entity_precision for sb in frontier]
        ax4.plot(frontier_x, frontier_y, 'r--', linewidth=2, label='Pareto Frontier')
    
    ax4.set_xlabel('PPL Drift (lower = better fluency)')
    ax4.set_ylabel('Entity Precision (higher = better grounding)')
    ax4.set_title('Pareto Frontier: Fluency vs Grounding')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. PALM Score vs Step
    ax5 = axes[1, 1]
    ax5.plot(steps, palm_scores, 'g-o', markersize=5, alpha=0.7)
    ax5.set_xlabel('Training Step')
    ax5.set_ylabel('PALM Composite Score')
    ax5.set_title('PALM Score Over Training')
    ax5.grid(True, alpha=0.3)
    
    # 6. SAE Weight Schedule
    ax6 = axes[1, 2]
    ax6.plot(steps, sae_weights, 'purple', linewidth=2)
    ax6.fill_between(steps, sae_weights, alpha=0.3, color='purple')
    ax6.set_xlabel('Training Step')
    ax6.set_ylabel('SAE Weight (λ)')
    ax6.set_title('SAE Weight Schedule')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Pareto analysis to {output_path}")


def summarize_scoreboards(scoreboards: List[CheckpointScoreboard]) -> str:
    """
    Generate a text summary of training progression from scoreboards.
    
    Args:
        scoreboards: List of CheckpointScoreboard objects
        
    Returns:
        Formatted summary string
    """
    if not scoreboards:
        return "No checkpoints recorded."
    
    lines = [
        "=" * 80,
        "CHECKPOINT SCOREBOARD SUMMARY",
        "=" * 80,
        "",
        f"Total checkpoints: {len(scoreboards)}",
        f"Steps: {scoreboards[0].step} → {scoreboards[-1].step}",
        f"Epochs: {scoreboards[0].epoch} → {scoreboards[-1].epoch}",
        "",
        "Metric Progression (first → last):",
        f"  PPL:           {scoreboards[0].perplexity:.2f} → {scoreboards[-1].perplexity:.2f}",
        f"  PPL Drift:     {scoreboards[0].ppl_drift:.2f} → {scoreboards[-1].ppl_drift:.2f}",
        f"  Faithfulness:  {scoreboards[0].entity_precision:.3f} → {scoreboards[-1].entity_precision:.3f}",
        f"  Hallucination: {scoreboards[0].entity_hallucination_rate:.3f} → {scoreboards[-1].entity_hallucination_rate:.3f}",
        f"  Distinct-2:    {scoreboards[0].distinct_2:.3f} → {scoreboards[-1].distinct_2:.3f}",
        f"  PALM Score:    {scoreboards[0].palm_score:.3f} → {scoreboards[-1].palm_score:.3f}",
        "",
    ]
    
    # Find best checkpoints
    best_palm = max(scoreboards, key=lambda sb: sb.palm_score)
    best_faith = max(scoreboards, key=lambda sb: sb.entity_precision)
    best_ppl = min(scoreboards, key=lambda sb: sb.perplexity)
    
    lines.extend([
        "Best Checkpoints:",
        f"  Best PALM Score:    step {best_palm.step} (score: {best_palm.palm_score:.3f})",
        f"  Best Faithfulness:  step {best_faith.step} (precision: {best_faith.entity_precision:.3f})",
        f"  Best Perplexity:    step {best_ppl.step} (PPL: {best_ppl.perplexity:.2f})",
        "",
        "=" * 80,
    ])
    
    return "\n".join(lines)
