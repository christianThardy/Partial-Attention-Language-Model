from .config import PALMConfig
from .model import PALMModel, transfer_weights_to_palm, bootstrap_palm_components
from .data import load_and_split_dataset, preprocess_function, create_data_loaders
from .training import PALMTrainer, collate_fn, apply_lora, apply_qlora, maybe_apply_lora

# Optional imports that require additional dependencies
try:
    from .evaluation import (
        generate_text,
        evaluate_generations,
        evaluate_information_extraction,
        # Staggered evaluation (preferred)
        StaggeredEvaluator,
        StaggeredEvalResult,
        # Legacy lightweight evaluation
        LightweightEvaluator,
        LightweightEvalResult,
        # Utilities
        create_eval_samples_from_dataloader,
        # =================================================================
        # NEW: Checkpoint Scoreboard for Pareto Analysis
        # =================================================================
        CheckpointScoreboard,
        save_checkpoint_scoreboard,
        load_checkpoint_scoreboard,
        get_pareto_frontier,
        plot_pareto_analysis,
        summarize_scoreboards,
        # =================================================================
        # NEW: Lambda (SAE Weight) Sweep Infrastructure
        # =================================================================
        LambdaSweepConfig,
        LambdaSweepResult,
        LambdaSweepRunner,
        save_sweep_results,
        load_sweep_results,
        plot_lambda_tradeoff,
        plot_lambda_training_curves,
        # =================================================================
        # NEW: Source Ablation Sensitivity Test
        # =================================================================
        ConditionMetrics,
        SourceAblationResult,
        SourceAblationEvaluator,
        plot_ablation_results,
        save_ablation_results,
        # Gradual Ablation Sensitivity Curve
        GradualAblationEvaluator,
        AblationCurveComparison,
        plot_ablation_sensitivity_curve,
        save_ablation_curve_comparison,
        # =================================================================
        # Paper-Based Metrics (Fu et al., 2023)
        # Stepwise Hallucination & Numerical Sensitivity Analysis
        # =================================================================
        StepwiseHallucinationResult,
        compute_stepwise_hallucination_ratio,
        NumericalSensitivityResult,
        numerical_sensitivity_analysis,
        PaperMetricsResult,
        PaperMetricsEvaluator,
        plot_stepwise_hallucination,
        plot_sensitivity_analysis,
        plot_combined_paper_metrics,
        save_paper_metrics,
        load_paper_metrics,
        # =================================================================
        # Context Rot Evaluation Suite
        # =================================================================
        ContextRotSuiteResult,
        ContextRotEvaluator,
        plot_context_rot_results,
        save_context_rot_results,
        # Lightweight probes for during-training
        ContextRotProbeResult,
        run_context_rot_probes,
        format_probe_results,
        # =================================================================
        # Go/No-Go Assessment
        # =================================================================
        GoNoGoCriteria,
        GoNoGoAssessment,
        assess_go_nogo,
        save_go_nogo_assessment,
    )
except ImportError:
    pass  # Evaluation dependencies not installed
