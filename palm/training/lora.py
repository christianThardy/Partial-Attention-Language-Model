"""
LoRA and QLoRA Utilities for PALM

Provides functions to apply LoRA/QLoRA adapters to PALMModel for efficient finetuning.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_palm_lora_target_modules() -> List[str]:
    """
    Get the target modules for LoRA based on PALMModel architecture.
    
    Targets all attention projections in both regular and partial attention modules.
    This is critical for adapting the pretrained backbone to PALM's new masking regime.
    
    Returns:
        List of module name patterns to target with LoRA
    """
    return [
        # Regular self-attention projections
        "attention.query",
        "attention.key", 
        "attention.value",
        "attention.dense",
        # Partial attention projections (key for adapting to new masking regime)
        "partial_attention.query",
        "partial_attention.key",
        "partial_attention.value", 
        "partial_attention.dense",
    ]


def apply_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    modules_to_save: Optional[List[str]] = None,
):
    """
    Apply LoRA adapters to PALMModel for efficient finetuning.
    
    Args:
        model: PALMModel instance
        r: LoRA rank (higher = more capacity but more params)
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout for LoRA layers
        target_modules: Modules to target (defaults to attention projections)
        modules_to_save: Modules to train fully (not with LoRA)
    
    Returns:
        Model wrapped with PEFT LoRA adapters
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "PEFT library required for LoRA. Install with: pip install peft"
        )
    
    if target_modules is None:
        target_modules = get_palm_lora_target_modules()
    
    if modules_to_save is None:
        modules_to_save = ["lm_head", "sae_head"]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    logger.info(f"LoRA adapters applied with r={r}, alpha={lora_alpha}")
    logger.info(f"Target modules: {target_modules}")
    logger.info(f"Modules trained fully: {modules_to_save}")
    
    return model


def apply_qlora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    modules_to_save: Optional[List[str]] = None,
):
    """
    Apply QLoRA (quantized LoRA) adapters to PALMModel for memory-efficient finetuning.
    
    Requires the model to be loaded in 4-bit or 8-bit quantization.
    
    Args:
        model: PALMModel instance (should be quantized)
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout for LoRA layers
        target_modules: Modules to target (defaults to attention projections)
        modules_to_save: Modules to train fully
    
    Returns:
        Model wrapped with PEFT QLoRA adapters
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        raise ImportError(
            "PEFT library required for QLoRA. Install with: pip install peft bitsandbytes"
        )
    
    if target_modules is None:
        target_modules = get_palm_lora_target_modules()
    
    if modules_to_save is None:
        modules_to_save = ["lm_head", "sae_head"]
    
    # Prepare model for quantized training
    model = prepare_model_for_kbit_training(model)
    
    qlora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    
    model = get_peft_model(model, qlora_config)
    model.print_trainable_parameters()
    
    logger.info(f"QLoRA adapters applied with r={r}, alpha={lora_alpha}")
    logger.info(f"Target modules: {target_modules}")
    
    return model


def maybe_apply_lora(
    model,
    use_lora: bool = False,
    use_qlora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """
    Conditionally apply LoRA or QLoRA based on flags.
    
    Args:
        model: PALMModel instance
        use_lora: Whether to apply LoRA
        use_qlora: Whether to apply QLoRA (takes precedence if both True)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout for LoRA layers
    
    Returns:
        Model (potentially wrapped with PEFT adapters)
    """
    if use_qlora:
        return apply_qlora(model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    elif use_lora:
        return apply_lora(model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    else:
        logger.info("No LoRA/QLoRA applied - training full model")
        return model

