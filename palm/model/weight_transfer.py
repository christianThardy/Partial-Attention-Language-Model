"""
Weight Transfer Utilities for PALM

Supports transferring pretrained weights from various HuggingFace models
(Llama, Qwen, Mistral, Phi, Gemma, Falcon, etc.) to PALMModel architecture.
"""

import gc
import logging
from typing import Dict, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


# =============================================================================
# ARCHITECTURE DETECTION
# =============================================================================

def detect_model_architecture(model_name_or_config: Union[str, "PretrainedConfig"]) -> str:
    """
    Detect the architecture type from a model name or config.
    
    Args:
        model_name_or_config: HuggingFace model name or config object
    
    Returns:
        Architecture family identifier: 'llama', 'qwen', 'mistral', 'phi', 'gemma', 'falcon'
    """
    if isinstance(model_name_or_config, str):
        config = AutoConfig.from_pretrained(model_name_or_config, trust_remote_code=True)
    else:
        config = model_name_or_config
    
    arch = getattr(config, 'model_type', '').lower()
    architectures = getattr(config, 'architectures', [])
    
    # Map to architecture families
    if 'llama' in arch or any('llama' in a.lower() for a in architectures):
        return 'llama'
    elif 'qwen' in arch or any('qwen' in a.lower() for a in architectures):
        return 'qwen'
    elif 'mistral' in arch or any('mistral' in a.lower() for a in architectures):
        return 'mistral'
    elif 'phi' in arch or any('phi' in a.lower() for a in architectures):
        return 'phi'
    elif 'gemma' in arch or any('gemma' in a.lower() for a in architectures):
        return 'gemma'
    elif 'falcon' in arch or any('falcon' in a.lower() for a in architectures):
        return 'falcon'
    else:
        logger.warning(f"Unknown architecture '{arch}', defaulting to 'llama' mapping")
        return 'llama'


# =============================================================================
# WEIGHT KEY MAPPINGS
# =============================================================================

def get_weight_mapping(arch_type: str, num_layers: int) -> Dict[str, Optional[str]]:
    """
    Get weight key mappings from source model to PALM model.
    
    Args:
        arch_type: Architecture family ('llama', 'qwen', 'mistral', etc.)
        num_layers: Number of transformer layers to map
    
    Returns:
        Dict mapping source keys to PALM keys (None means skip)
    """
    mapping = {}
    
    if arch_type in ['llama', 'mistral', 'gemma']:
        mapping['model.embed_tokens.weight'] = 'embeddings.word_embeddings.weight'
        mapping['lm_head.weight'] = 'lm_head.weight'
        mapping['model.norm.weight'] = None  # PALM uses per-layer norms
        
        for i in range(num_layers):
            src_prefix = f'model.layers.{i}'
            dst_prefix = f'layers.{i}'
            
            # Self-attention
            mapping[f'{src_prefix}.self_attn.q_proj.weight'] = f'{dst_prefix}.attention.query.weight'
            mapping[f'{src_prefix}.self_attn.k_proj.weight'] = f'{dst_prefix}.attention.key.weight'
            mapping[f'{src_prefix}.self_attn.v_proj.weight'] = f'{dst_prefix}.attention.value.weight'
            mapping[f'{src_prefix}.self_attn.o_proj.weight'] = f'{dst_prefix}.attention.dense.weight'
            
            # Layer norms
            mapping[f'{src_prefix}.input_layernorm.weight'] = f'{dst_prefix}.attention.LayerNorm.weight'
            
            # MLP
            mapping[f'{src_prefix}.mlp.gate_proj.weight'] = f'{dst_prefix}.intermediate.dense.weight'
            mapping[f'{src_prefix}.mlp.down_proj.weight'] = f'{dst_prefix}.output.dense.weight'
            
    elif arch_type == 'qwen':
        mapping['transformer.wte.weight'] = 'embeddings.word_embeddings.weight'
        mapping['lm_head.weight'] = 'lm_head.weight'
        
        for i in range(num_layers):
            src_prefix = f'transformer.h.{i}'
            dst_prefix = f'layers.{i}'
            
            # Qwen v1 style
            mapping[f'{src_prefix}.attn.q_proj.weight'] = f'{dst_prefix}.attention.query.weight'
            mapping[f'{src_prefix}.attn.k_proj.weight'] = f'{dst_prefix}.attention.key.weight'
            mapping[f'{src_prefix}.attn.v_proj.weight'] = f'{dst_prefix}.attention.value.weight'
            mapping[f'{src_prefix}.attn.c_proj.weight'] = f'{dst_prefix}.attention.dense.weight'
            
            # Qwen2 style
            mapping[f'model.layers.{i}.self_attn.q_proj.weight'] = f'{dst_prefix}.attention.query.weight'
            mapping[f'model.layers.{i}.self_attn.k_proj.weight'] = f'{dst_prefix}.attention.key.weight'
            mapping[f'model.layers.{i}.self_attn.v_proj.weight'] = f'{dst_prefix}.attention.value.weight'
            mapping[f'model.layers.{i}.self_attn.o_proj.weight'] = f'{dst_prefix}.attention.dense.weight'
            
            mapping[f'{src_prefix}.ln_1.weight'] = f'{dst_prefix}.attention.LayerNorm.weight'
            
            mapping[f'{src_prefix}.mlp.w1.weight'] = f'{dst_prefix}.intermediate.dense.weight'
            mapping[f'{src_prefix}.mlp.w2.weight'] = f'{dst_prefix}.output.dense.weight'
            
            # Qwen2 MLP
            mapping[f'model.layers.{i}.mlp.gate_proj.weight'] = f'{dst_prefix}.intermediate.dense.weight'
            mapping[f'model.layers.{i}.mlp.down_proj.weight'] = f'{dst_prefix}.output.dense.weight'
            
    elif arch_type == 'phi':
        mapping['model.embed_tokens.weight'] = 'embeddings.word_embeddings.weight'
        mapping['lm_head.weight'] = 'lm_head.weight'
        
        for i in range(num_layers):
            src_prefix = f'model.layers.{i}'
            dst_prefix = f'layers.{i}'
            
            mapping[f'{src_prefix}.self_attn.q_proj.weight'] = f'{dst_prefix}.attention.query.weight'
            mapping[f'{src_prefix}.self_attn.k_proj.weight'] = f'{dst_prefix}.attention.key.weight'
            mapping[f'{src_prefix}.self_attn.v_proj.weight'] = f'{dst_prefix}.attention.value.weight'
            mapping[f'{src_prefix}.self_attn.dense.weight'] = f'{dst_prefix}.attention.dense.weight'
            
            mapping[f'{src_prefix}.mlp.fc1.weight'] = f'{dst_prefix}.intermediate.dense.weight'
            mapping[f'{src_prefix}.mlp.fc2.weight'] = f'{dst_prefix}.output.dense.weight'
            
    elif arch_type == 'falcon':
        mapping['transformer.word_embeddings.weight'] = 'embeddings.word_embeddings.weight'
        mapping['lm_head.weight'] = 'lm_head.weight'
        
        for i in range(num_layers):
            src_prefix = f'transformer.h.{i}'
            dst_prefix = f'layers.{i}'
            
            # Note: Falcon uses fused QKV which requires special handling
            mapping[f'{src_prefix}.self_attention.query_key_value.weight'] = None
            mapping[f'{src_prefix}.self_attention.dense.weight'] = f'{dst_prefix}.attention.dense.weight'
            
            mapping[f'{src_prefix}.mlp.dense_h_to_4h.weight'] = f'{dst_prefix}.intermediate.dense.weight'
            mapping[f'{src_prefix}.mlp.dense_4h_to_h.weight'] = f'{dst_prefix}.output.dense.weight'
    
    return mapping


# =============================================================================
# WEIGHT TRANSFER
# =============================================================================

def transfer_weights_to_palm(
    palm_model: torch.nn.Module,
    source_model_name: str,
    device: str = 'cpu',
    dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    """
    Transfer pretrained weights from any supported HuggingFace model to PALMModel.
    
    Compatible weights from the source model's self-attention and MLP layers are
    transferred to PALM's corresponding layers. Components unique to PALM 
    (partial_attention, sae_head, Fp, language_embeddings, position_embeddings)
    remain randomly initialized.
    
    Args:
        palm_model: Initialized PALMModel instance
        source_model_name: HuggingFace model name/path (e.g., "meta-llama/Llama-3.2-3B")
        device: Device to load source model on temporarily ('cpu' recommended to avoid OOM)
        dtype: Data type for loading source model
    
    Returns:
        palm_model with transferred weights
        
    Supported architectures:
        - Llama (all versions)
        - Qwen / Qwen2
        - Mistral
        - Phi
        - Gemma
        - Falcon
    """
    logger.info(f"Loading pretrained weights from: {source_model_name}")
    
    # Detect architecture
    arch_type = detect_model_architecture(source_model_name)
    logger.info(f"Detected architecture: {arch_type}")
    
    # Load source model
    logger.info("Loading source model (this may take a while for large models)...")
    source_config = AutoConfig.from_pretrained(source_model_name, trust_remote_code=True)
    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Get state dicts
    source_state = source_model.state_dict()
    palm_state = palm_model.state_dict()
    
    # Get number of layers to transfer (min of source and PALM)
    source_layers = getattr(source_config, 'num_hidden_layers', 
                           getattr(source_config, 'n_layer', 28))
    palm_layers = len(palm_model.layers)
    num_layers = min(source_layers, palm_layers)
    
    logger.info(f"Source has {source_layers} layers, PALM has {palm_layers} layers")
    logger.info(f"Will transfer weights for {num_layers} layers")
    
    # Get weight mapping
    mapping = get_weight_mapping(arch_type, num_layers)
    
    # Transfer weights
    transferred = 0
    skipped = 0
    shape_mismatch = 0
    
    for src_key, dst_key in mapping.items():
        if dst_key is None:
            skipped += 1
            continue
            
        if src_key not in source_state:
            skipped += 1
            continue
            
        if dst_key not in palm_state:
            skipped += 1
            continue
        
        src_weight = source_state[src_key]
        dst_weight = palm_state[dst_key]
        
        if src_weight.shape == dst_weight.shape:
            palm_state[dst_key] = src_weight.clone().to(dst_weight.dtype)
            transferred += 1
        else:
            # Handle shape mismatches
            if 'embed' in dst_key and src_weight.shape[0] >= dst_weight.shape[0]:
                # Vocabulary size mismatch - truncate if source is larger
                palm_state[dst_key] = src_weight[:dst_weight.shape[0]].clone().to(dst_weight.dtype)
                transferred += 1
                logger.info(f"Truncated {src_key} from {src_weight.shape} to {dst_weight.shape}")
            elif len(src_weight.shape) == len(dst_weight.shape) == 2:
                # Linear layer mismatch - transfer overlapping portion
                min_in = min(src_weight.shape[1], dst_weight.shape[1])
                min_out = min(src_weight.shape[0], dst_weight.shape[0])
                palm_state[dst_key][:min_out, :min_in] = src_weight[:min_out, :min_in].clone().to(dst_weight.dtype)
                transferred += 1
                logger.info(f"Partial transfer {src_key} {src_weight.shape} -> {dst_key} {dst_weight.shape}")
            else:
                shape_mismatch += 1
                logger.warning(f"Shape mismatch: {src_key} {src_weight.shape} vs {dst_key} {dst_weight.shape}")
    
    # Load the modified state dict
    palm_model.load_state_dict(palm_state, strict=False)
    
    # Clean up source model to free memory
    del source_model
    del source_state
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info(f"Weight transfer complete:")
    logger.info(f"  - Transferred: {transferred} tensors")
    logger.info(f"  - Skipped: {skipped} tensors")
    logger.info(f"  - Shape mismatches: {shape_mismatch} tensors")
    
    # Log which PALM components are initialized fresh
    fresh_components = ['partial_attention', 'sae_head', 'Fp', 'language_embeddings', 'position_embeddings']
    logger.info(f"Components initialized fresh (not from pretrained): {fresh_components}")
    
    return palm_model

