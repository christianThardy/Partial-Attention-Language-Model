from .generation import generate_text
from .metrics import calculate_rouge, calculate_bert_score, evaluate_generations, evaluate_information_extraction
from .lightweight import (
    # Cluster dataclasses
    DegenerationCluster,
    FaithfulnessCluster,
    LossCluster,
    MaskComplianceCluster,
    StaggeredEvalResult,
    # Evaluators
    StaggeredEvaluator,
    LightweightEvaluator,
    LightweightEvalResult,
    # Utilities
    create_eval_samples_from_dataloader,
    compute_generation_metrics_fast,
    compute_mask_compliance_fast,
    compute_palm_score_fast,
)
from .comprehensive import (
    # Loss decomposition
    LossDecomposition,
    compute_loss_decomposition,
    compute_perplexity,
    # Degeneration metrics
    DegenerationMetrics,
    compute_degeneration_metrics,
    compute_distinct_ngrams,
    compute_repetition_rate,
    # Faithfulness metrics
    FaithfulnessMetrics,
    compute_faithfulness_metrics,
    extract_entities_simple,
    # Mask compliance
    MaskComplianceMetrics,
    compute_mask_compliance,
    # PALM score
    PALMScore,
    compute_palm_score,
    # Ablation evaluation
    AblationResult,
    AblationEvaluator,
    # Smoke tests
    compare_generations_cached_vs_uncached,
    # QA evaluations
    evaluate_answerable_from_source,
    evaluate_distractor_swap,
)

# =============================================================================
# NEW: Checkpoint Scoreboard for Pareto Analysis
# =============================================================================
from .checkpoint_scoreboard import (
    CheckpointScoreboard,
    save_checkpoint_scoreboard,
    load_checkpoint_scoreboard,
    get_pareto_frontier,
    plot_pareto_analysis,
    summarize_scoreboards,
)

# =============================================================================
# NEW: Lambda (SAE Weight) Sweep Infrastructure
# =============================================================================
from .lambda_sweep import (
    LambdaSweepConfig,
    LambdaSweepResult,
    LambdaSweepRunner,
    save_sweep_results,
    load_sweep_results,
    plot_lambda_tradeoff,
    plot_lambda_training_curves,
)

# =============================================================================
# NEW: Source Ablation Sensitivity Test
# =============================================================================
from .source_ablation import (
    ConditionMetrics,
    SourceAblationResult,
    SourceAblationEvaluator,
    plot_ablation_results,
    save_ablation_results,
    # Gradual Ablation Sensitivity Curve
    GradualAblationPoint,
    GradualAblationCurve,
    AblationCurveComparison,
    GradualAblationEvaluator,
    plot_ablation_sensitivity_curve,
    save_ablation_curve_comparison,
)

# =============================================================================
# Paper-Based Metrics (Stepwise Hallucination & Sensitivity Analysis)
# =============================================================================
# From: "Decoder-Only or Encoder-Decoder? Interpreting Language Model as a
#        Regularized Encoder-Decoder" (Fu et al., 2023)
from .paper_metrics import (
    # Stepwise Hallucination Analysis
    StepwiseHallucinationResult,
    compute_stepwise_hallucination_ratio,
    compute_alignment_score,
    # Numerical Sensitivity Analysis
    NumericalSensitivityResult,
    numerical_sensitivity_analysis,
    # Combined Evaluator
    PaperMetricsResult,
    PaperMetricsEvaluator,
    # Plotting
    plot_stepwise_hallucination,
    plot_sensitivity_analysis,
    plot_combined_paper_metrics,
    # Save/Load
    save_paper_metrics,
    load_paper_metrics,
)

# =============================================================================
# Context Rot Evaluation Suite
# =============================================================================
from .context_rot import (
    # Levenshtein utilities
    normalized_levenshtein_score,
    # Repeated Words (Chroma benchmark)
    RepeatedWordsResult,
    RepeatedWordsSummary,
    # Needle tests
    NeedleTestResult,
    DistractorTestResult,
    FocusedVsFullResult,
    ContextRotSuiteResult,
    # Evaluator
    ContextRotEvaluator,
    # Plotting & I/O
    plot_context_rot_results,
    plot_repeated_words_curve,
    save_context_rot_results,
    load_context_rot_results,
)

# =============================================================================
# Lightweight Context Rot Probes (During-Training)
# =============================================================================
from .context_rot_probes import (
    ContextRotProbeResult,
    run_context_rot_probes,
    format_probe_results,
)

# =============================================================================
# Go/No-Go Assessment
# =============================================================================
from .go_nogo import (
    GoNoGoCriteria,
    CriterionResult,
    GoNoGoAssessment,
    assess_go_nogo,
    save_go_nogo_assessment,
)