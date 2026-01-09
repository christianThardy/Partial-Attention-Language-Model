import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# PARTIAL ATTENTION WARMUP
# Based on: "Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off"
# Key insight: New components (partial attention, Fp network) need to "catch up"
# to pretrained weights before full training. Otherwise, random gradients from
# new components can destabilize pretrained representations.
#
# Strategy:
# 1. Freeze everything EXCEPT partial_attention.* and Fp_* for K steps
# 2. Train only these components with a focused learning rate
# 3. After warmup, unfreeze and proceed with normal training
#
# IMPORTANT: Warmup LR Guidelines
# - Default 1e-4 is conservative and safe
# - 5e-4 can cause instability (NaN/Inf in attention) during early steps
# - If you see "Non-finite values" warnings, REDUCE warmup_lr
# - The Fp network is particularly sensitive to high LR since it transforms
#   source states before they go through K/V projections

class PartialAttentionWarmup:
    """
    Manages the partial attention warmup phase where only partial_attention
    and Fp network components are trained while everything else is frozen.
    
    This allows the PALM-specific components to learn coherent representations
    before the pretrained backbone starts updating, preventing instability.
    
    IMPORTANT: Use warmup_lr <= 1e-4 to avoid numerical instability.
    Higher LR (e.g., 5e-4) can cause NaN/Inf in early training steps.
    
    Usage:
        warmup = PartialAttentionWarmup(warmup_steps=500, warmup_lr=1e-4)
        
        for step in training_loop:
            # Check phase and get appropriate parameters
            if warmup.is_active(step):
                # During warmup: only partial_attn params are trainable
                warmup.apply_warmup_freeze(model)
            elif warmup.just_completed(step):
                # Transition: unfreeze everything, recreate optimizer
                warmup.end_warmup(model)
                optimizer = create_optimizer(model)  # Recreate with all params
            
            # Normal training step...
    """
    
    # Parameter patterns that should train during warmup
    WARMUP_TRAINABLE_PATTERNS = [
        'partial_attention.query',
        'partial_attention.key',
        'partial_attention.value',
        'partial_attention.dense',
        'partial_attention.Fp_',
        'partial_attn_norm',
    ]
    
    def __init__(
        self,
        warmup_steps: int = 500,
        warmup_lr: float = 1e-4,
        enabled: bool = True,
    ):
        """
        Args:
            warmup_steps: Number of steps to run partial attention warmup
            warmup_lr: Learning rate for warmup phase. Default 1e-4 is conservative.
                      CAUTION: Values > 2e-4 can cause NaN/Inf in attention during
                      early steps. If you see "Non-finite values" warnings, reduce this.
            enabled: Whether warmup is active (False to skip entirely)
        """
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.enabled = enabled
        self._warmup_applied = False
        self._warmup_completed = False
        self._frozen_params: Dict[str, bool] = {}  # Track original requires_grad state
    
    def is_active(self, step: int) -> bool:
        """Check if warmup phase is currently active."""
        return self.enabled and step < self.warmup_steps and not self._warmup_completed
    
    def just_completed(self, step: int) -> bool:
        """Check if warmup just completed at this step (transition point)."""
        if not self.enabled:
            return False
        if step == self.warmup_steps and not self._warmup_completed:
            self._warmup_completed = True
            return True
        return False
    
    def get_phase_name(self, step: int) -> str:
        """Get current phase name for logging."""
        if self.is_active(step):
            return "PARTIAL_ATTN_WARMUP"
        return "NORMAL"
    
    def _is_warmup_trainable(self, param_name: str) -> bool:
        """Check if parameter should be trainable during warmup."""
        return any(pattern in param_name for pattern in self.WARMUP_TRAINABLE_PATTERNS)
    
    def apply_warmup_freeze(self, model: nn.Module) -> Tuple[int, int]:
        """
        Apply warmup freeze: only partial_attention and Fp params are trainable.
        
        Returns:
            (num_frozen, num_trainable) parameter counts
        """
        if self._warmup_applied:
            return 0, 0  # Already applied
        
        frozen_count = 0
        trainable_count = 0
        
        for name, param in model.named_parameters():
            # Store original state for restoration
            self._frozen_params[name] = param.requires_grad
            
            if self._is_warmup_trainable(name):
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        
        self._warmup_applied = True
        logger.info(f"Partial attention warmup applied: {frozen_count} frozen, {trainable_count} trainable")
        return frozen_count, trainable_count
    
    def end_warmup(self, model: nn.Module) -> int:
        """
        End warmup phase: restore original requires_grad states.
        
        Returns:
            Number of parameters restored
        """
        if not self._warmup_applied:
            return 0
        
        restored_count = 0
        for name, param in model.named_parameters():
            if name in self._frozen_params:
                param.requires_grad = self._frozen_params[name]
                restored_count += 1
        
        self._frozen_params.clear()
        self._warmup_applied = False
        logger.info(f"Partial attention warmup ended: {restored_count} parameters restored")
        return restored_count
    
    def get_warmup_params(self, model: nn.Module) -> List[nn.Parameter]:
        """
        Get list of parameters that should train during warmup.
        Useful for creating a focused optimizer.
        """
        return [
            param for name, param in model.named_parameters()
            if self._is_warmup_trainable(name) and param.requires_grad
        ]
    
    def create_warmup_optimizer(
        self, 
        model: nn.Module,
        optimizer_class=None,
    ) -> torch.optim.Optimizer:
        """
        Create an optimizer for the warmup phase with only partial_attention params.
        
        Args:
            model: The model
            optimizer_class: Optimizer class (default: AdamW)
        
        Returns:
            Optimizer configured for warmup phase
        """
        if optimizer_class is None:
            optimizer_class = torch.optim.AdamW
        
        warmup_params = self.get_warmup_params(model)
        
        if not warmup_params:
            raise ValueError("No warmup-trainable parameters found!")
        
        optimizer = optimizer_class(
            warmup_params,
            lr=self.warmup_lr,
            weight_decay=0.01,
        )
        
        logger.info(f"Created warmup optimizer: {len(warmup_params)} params @ lr={self.warmup_lr}")
        return optimizer
    
    def get_progress(self, step: int) -> float:
        """Get warmup progress as fraction (0.0 to 1.0)."""
        if not self.enabled or self.warmup_steps <= 0:
            return 1.0
        return min(step / self.warmup_steps, 1.0)
    
    def format_status(self, step: int) -> str:
        """Format current warmup status for logging."""
        if not self.enabled:
            return "Warmup disabled"
        if self._warmup_completed:
            return "Warmup completed"
        if step < self.warmup_steps:
            progress = self.get_progress(step)
            return f"Warmup: {step}/{self.warmup_steps} ({progress*100:.1f}%)"
        return "Warmup complete"


def get_partial_attention_param_count(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in partial attention components.
    
    Returns:
        Dict with parameter counts for each component type
    """
    counts = {
        'partial_attention_qkv': 0,
        'partial_attention_dense': 0,
        'fp_network': 0,
        'partial_attn_norm': 0,
        'total_partial': 0,
        'total_model': 0,
    }
    
    for name, param in model.named_parameters():
        numel = param.numel()
        counts['total_model'] += numel
        
        if 'partial_attention.query' in name or 'partial_attention.key' in name or 'partial_attention.value' in name:
            counts['partial_attention_qkv'] += numel
            counts['total_partial'] += numel
        elif 'partial_attention.dense' in name:
            counts['partial_attention_dense'] += numel
            counts['total_partial'] += numel
        elif 'Fp_' in name:
            counts['fp_network'] += numel
            counts['total_partial'] += numel
        elif 'partial_attn_norm' in name:
            counts['partial_attn_norm'] += numel
            counts['total_partial'] += numel
    
    return counts


def collate_fn_instruct(batch):
    """
    Use if you have instruction-oriented data.
    """
    # Initialize dictionaries to store the batched data
    batched_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "source_len": []
    }
    # Iterate through the batch and append each item to the corresponding list
    for item in batch:
        for key in batched_data:
            # Ensure each item is a tensor and append it to the list
            batched_data[key].append(torch.tensor(item[key]))
    # Stack tensors, making sure all elements are tensors and have the same shape
    for key in batched_data:
        batched_data[key] = torch.stack(batched_data[key])
    return batched_data

def collate_fn_base(batch):
    """
    Use if you have pretraining style data.
    """
    # Identify all keys in the batch
    keys = batch[0].keys()  # e.g. ["input_ids","attention_mask","labels","source_len"]
    batched_data = {key: [item[key] for item in batch] for key in keys}
    
    # Convert lists to tensors
    for key in batched_data:
        # Force them to integer (long) Tensors
        batched_data[key] = torch.tensor(batched_data[key], dtype=torch.long)
    return batched_data

# Alias for backward compatibility
collate_fn = collate_fn_base

def init_custom_layer_weights(module):
    """
    Recursively initialize custom layers (partial attention, lm/sae heads) 
    to encourage faster learning from scratch.
    """
    for name, submodule in module.named_children():
        # Initialize only "custom" modules or submodules you consider new. 
        # This example checks partial attention & heads, skipping loaded pretrained weights.
        if isinstance(submodule, PALMPartialAttention) or name in ["lm_head", "sae_head"]:
            for param_name, param in submodule.named_parameters(recurse=False):
                if "weight" in param_name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in param_name:
                    nn.init.zeros_(param)
        else:
            init_custom_layer_weights(submodule)

def is_custom_param(param_name):
    """
    Checks if a parameter name belongs to newly added custom modules
    such as partial_attention blocks, Fp submodules, or lm/sae heads.
    """
    if any(x in param_name for x in ["partial_attention", "lm_head", "sae_head", "Fp"]):
        return True
    return False

def freeze_selected_layers(model_, freeze_embeddings=True, freeze_up_to_layer_idx=0):
    """
    Freezes or unfreezes embeddings and some portion of layers.
    If freeze_up_to_layer_idx=12, for example, layers [0..11] are frozen,
    and layers [12..end] are trainable.
    """
    real_model = model_.module if hasattr(model_, 'module') else model_

    # Freeze/unfreeze embeddings
    if freeze_embeddings:
        for param in real_model.embeddings.parameters():
            param.requires_grad = False
            
    # Freeze/unfreeze layers
    for idx, layer in enumerate(real_model.layers):
        for param in layer.parameters():
            param.requires_grad = (idx >= freeze_up_to_layer_idx)
            
     # Always keep LM and SAE heads trainable
    for param in real_model.lm_head.parameters():
        param.requires_grad = True
    for param in real_model.sae_head.parameters():
        param.requires_grad = True

def continuous_unfreeze(model, epoch, total_epochs, NUM_LAYERS):
    """
    Continuously unfreeze layers based on current epoch. 
    Allows a smoother transition than chunk-based freezing.
    """
    real_model = model.module if hasattr(model, 'module') else model
    
    layers_to_unfreeze = int((epoch / float(total_epochs)) * NUM_LAYERS)
    # Freeze embeddings until last epoch, or adjust logic as desired:
    freeze_embeddings = (epoch < (total_epochs - 1))
    
    freeze_selected_layers(
        model,
        freeze_embeddings=freeze_embeddings,
        freeze_up_to_layer_idx=real_model.config.num_hidden_layers - layers_to_unfreeze
    )
