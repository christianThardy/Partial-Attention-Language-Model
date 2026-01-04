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
    
    # Bootstrap PALM-specific components from pretrained weights (architecture-aware)
    palm_state = bootstrap_palm_components(palm_model, source_state, palm_state, arch_type=arch_type)
    
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
    fresh_components = ['language_embeddings', 'position_embeddings']
    logger.info(f"Components initialized fresh (not from pretrained): {fresh_components}")
    
    return palm_model


# =============================================================================
# PALM COMPONENT BOOTSTRAPPING
# =============================================================================

def _get_attention_key_patterns(arch_type: str, layer_idx: int) -> Dict[str, str]:
    """
    Get source attention key patterns for a given architecture.
    
    Returns dict mapping projection type to source key pattern.
    """
    i = layer_idx
    
    if arch_type in ['llama', 'mistral', 'gemma']:
        return {
            'q': f'model.layers.{i}.self_attn.q_proj.weight',
            'k': f'model.layers.{i}.self_attn.k_proj.weight',
            'v': f'model.layers.{i}.self_attn.v_proj.weight',
            'o': f'model.layers.{i}.self_attn.o_proj.weight',
            'norm': f'model.layers.{i}.input_layernorm.weight',
            'norm_bias': f'model.layers.{i}.input_layernorm.bias',
        }
    elif arch_type == 'qwen':
        # Try Qwen2 style first (more common now), fallback patterns handled at call site
        return {
            'q': f'model.layers.{i}.self_attn.q_proj.weight',
            'k': f'model.layers.{i}.self_attn.k_proj.weight',
            'v': f'model.layers.{i}.self_attn.v_proj.weight',
            'o': f'model.layers.{i}.self_attn.o_proj.weight',
            'norm': f'model.layers.{i}.input_layernorm.weight',
            'norm_bias': f'model.layers.{i}.input_layernorm.bias',
            # Qwen1 fallbacks
            'q_alt': f'transformer.h.{i}.attn.q_proj.weight',
            'k_alt': f'transformer.h.{i}.attn.k_proj.weight',
            'v_alt': f'transformer.h.{i}.attn.v_proj.weight',
            'o_alt': f'transformer.h.{i}.attn.c_proj.weight',
            'norm_alt': f'transformer.h.{i}.ln_1.weight',
        }
    elif arch_type == 'phi':
        return {
            'q': f'model.layers.{i}.self_attn.q_proj.weight',
            'k': f'model.layers.{i}.self_attn.k_proj.weight',
            'v': f'model.layers.{i}.self_attn.v_proj.weight',
            'o': f'model.layers.{i}.self_attn.dense.weight',
            'norm': f'model.layers.{i}.input_layernorm.weight',
            'norm_bias': f'model.layers.{i}.input_layernorm.bias',
        }
    elif arch_type == 'falcon':
        # Falcon uses fused QKV - can't bootstrap partial attention from it
        return {
            'q': None,  # Fused QKV not supported
            'k': None,
            'v': None,
            'o': f'transformer.h.{i}.self_attention.dense.weight',
            'norm': f'transformer.h.{i}.input_layernorm.weight',
            'norm_bias': f'transformer.h.{i}.input_layernorm.bias',
        }
    else:
        # Default to Llama-style (most common)
        return {
            'q': f'model.layers.{i}.self_attn.q_proj.weight',
            'k': f'model.layers.{i}.self_attn.k_proj.weight',
            'v': f'model.layers.{i}.self_attn.v_proj.weight',
            'o': f'model.layers.{i}.self_attn.o_proj.weight',
            'norm': f'model.layers.{i}.input_layernorm.weight',
            'norm_bias': f'model.layers.{i}.input_layernorm.bias',
        }


def bootstrap_palm_components(
    palm_model: torch.nn.Module,
    source_state: Dict[str, torch.Tensor],
    palm_state: Dict[str, torch.Tensor],
    arch_type: str = 'llama'
) -> Dict[str, torch.Tensor]:
    """
    Bootstrap PALM-specific components from pretrained weights for better initialization.
    
    Strategy:
    1. sae_head ← clone from lm_head (both predict tokens from hidden states)
    2. partial_attention Q/K/V/dense ← clone from regular attention Q/K/V/o_proj
       - Uses repeat_interleave for GQA→MHA expansion (not slicing)
    3. partial_attention LayerNorm ← clone from input_layernorm (prevents activation shock)
    4. Fp network ← identity-like init with Fp_linear2 = 0 (Fp(x) ≈ x initially)
    
    This provides a much better starting point than random initialization:
    - Lower initial SAE loss (starting from reasonable token prediction)
    - Faster convergence (partial attention already "knows" how to attend)
    - No activation shock (LayerNorm scales match pretrained model)
    - More stable training (Fp outputs 0 initially, residual dominates)
    
    Supports architectures: Llama, Mistral, Gemma, Qwen, Qwen2, Phi, Falcon (partial)
    
    Args:
        palm_model: The PALM model (for config access)
        source_state: State dict from source pretrained model
        palm_state: State dict from PALM model (will be modified in-place)
        arch_type: Architecture type ('llama', 'qwen', 'mistral', 'phi', 'gemma', 'falcon')
    
    Returns:
        Modified palm_state dict
    """
    num_layers = palm_model.config.num_hidden_layers
    bootstrapped_components = []
    
    # 1. SAE HEAD ← CLONE FROM LM_HEAD
    # Both project hidden_size → vocab_size, both predict tokens
    if 'lm_head.weight' in source_state and 'sae_head.weight' in palm_state:
        src_weight = source_state['lm_head.weight']
        dst_dtype = palm_state['sae_head.weight'].dtype
        palm_state['sae_head.weight'] = src_weight.clone().to(dst_dtype)
        bootstrapped_components.append('sae_head')
        logger.info("✓ Bootstrapped sae_head from lm_head")
    
    # 2. PARTIAL ATTENTION Q/K/V/DENSE ← CLONE FROM REGULAR ATTENTION
    # This gives partial attention a head start with coherent attention patterns
    partial_attention_bootstrapped = 0
    for i in range(num_layers):
        # Get architecture-specific key patterns
        patterns = _get_attention_key_patterns(arch_type, i)
        
        # Map source attention → partial attention (architecture-aware)
        mappings = [
            (patterns['q'], f'layers.{i}.partial_attention.query.weight'),
            (patterns['k'], f'layers.{i}.partial_attention.key.weight'),
            (patterns['v'], f'layers.{i}.partial_attention.value.weight'),
            (patterns['o'], f'layers.{i}.partial_attention.dense.weight'),
        ]
        
        # Add fallback patterns for Qwen1 if primary not found
        if arch_type == 'qwen':
            fallback_mappings = [
                (patterns.get('q_alt'), f'layers.{i}.partial_attention.query.weight'),
                (patterns.get('k_alt'), f'layers.{i}.partial_attention.key.weight'),
                (patterns.get('v_alt'), f'layers.{i}.partial_attention.value.weight'),
                (patterns.get('o_alt'), f'layers.{i}.partial_attention.dense.weight'),
            ]
        else:
            fallback_mappings = []
        
        for src_key, dst_key in mappings:
            # Skip if source key is None (e.g., Falcon's fused QKV)
            if src_key is None:
                continue
            
            # Try primary key, then fallback for Qwen1
            actual_src_key = src_key
            if src_key not in source_state and fallback_mappings:
                # Find matching fallback
                for fb_src, fb_dst in fallback_mappings:
                    if fb_dst == dst_key and fb_src and fb_src in source_state:
                        actual_src_key = fb_src
                        break
            
            if actual_src_key not in source_state or dst_key not in palm_state:
                continue
                
            src_w = source_state[actual_src_key]
            dst_w = palm_state[dst_key]
            dst_dtype = dst_w.dtype
            
            if src_w.shape == dst_w.shape:
                palm_state[dst_key] = src_w.clone().to(dst_dtype)
            else:
                # Handle GQA → MHA expansion using repeat_interleave
                # Source GQA has fewer K/V heads than PALM's full MHA
                # We repeat each source head to fill the destination heads
                if dst_w.shape[0] > src_w.shape[0] and dst_w.shape[0] % src_w.shape[0] == 0:
                    # GQA → MHA: repeat K/V heads to match destination
                    num_repeats = dst_w.shape[0] // src_w.shape[0]
                    expanded = src_w.repeat_interleave(num_repeats, dim=0)
                    # Handle input dimension if also different
                    if expanded.shape[1] != dst_w.shape[1]:
                        min_in = min(expanded.shape[1], dst_w.shape[1])
                        palm_state[dst_key][:, :min_in] = expanded[:, :min_in].clone().to(dst_dtype)
                    else:
                        palm_state[dst_key] = expanded.clone().to(dst_dtype)
                    logger.debug(f"GQA→MHA expand {actual_src_key} {src_w.shape} -> {dst_key} {dst_w.shape} (repeat {num_repeats}x)")
                else:
                    # Fallback: partial transfer for other mismatches
                    min_out = min(src_w.shape[0], dst_w.shape[0])
                    min_in = min(src_w.shape[1], dst_w.shape[1])
                    palm_state[dst_key][:min_out, :min_in] = src_w[:min_out, :min_in].clone().to(dst_dtype)
                    logger.debug(f"Partial bootstrap {actual_src_key} {src_w.shape} -> {dst_key} {dst_w.shape}")
            
            partial_attention_bootstrapped += 1
    
    if partial_attention_bootstrapped > 0:
        bootstrapped_components.append('partial_attention')
        logger.info(f"✓ Bootstrapped partial_attention Q/K/V/dense for {num_layers} layers "
                   f"({partial_attention_bootstrapped} tensors)")
    
    # 3. LAYERNORM CLONING ← CLONE INPUT_LAYERNORM TO PARTIAL_ATTENTION
    # The trained LayerNorm weights compensate for activation drift in that specific layer.
    # Using default LayerNorm (scale=1.0) with cloned attention weights causes "activation shock".
    # Clone input_layernorm.weight to partial_attention's LayerNorm for proper scaling.
    layernorm_bootstrapped = 0
    for i in range(num_layers):
        # Get architecture-specific LayerNorm key
        patterns = _get_attention_key_patterns(arch_type, i)
        src_norm_key = patterns.get('norm')
        src_bias_key = patterns.get('norm_bias')
        dst_norm_key = f'layers.{i}.partial_attention.LayerNorm.weight'
        dst_bias_key = f'layers.{i}.partial_attention.LayerNorm.bias'
        
        # Try primary key, then fallback (for Qwen1)
        if src_norm_key and src_norm_key not in source_state:
            alt_key = patterns.get('norm_alt')
            if alt_key and alt_key in source_state:
                src_norm_key = alt_key
        
        if src_norm_key and src_norm_key in source_state and dst_norm_key in palm_state:
            src_norm = source_state[src_norm_key]
            dst_dtype = palm_state[dst_norm_key].dtype
            palm_state[dst_norm_key] = src_norm.clone().to(dst_dtype)
            layernorm_bootstrapped += 1
        
        # Also clone bias if present
        if src_bias_key and src_bias_key in source_state and dst_bias_key in palm_state:
            palm_state[dst_bias_key] = source_state[src_bias_key].clone().to(palm_state[dst_bias_key].dtype)
    
    if layernorm_bootstrapped > 0:
        bootstrapped_components.append('partial_attention_LayerNorm')
        logger.info(f"✓ Cloned input_layernorm to partial_attention.LayerNorm for {layernorm_bootstrapped} layers")
    
    # 4. Fp NETWORK ← IDENTITY-LIKE INITIALIZATION
    # Goal: Fp(x) ≈ x initially, so partial attention starts by just copying source
    # Paper uses Pl = Pl2 + Pl1 (residual), so we want:
    #   - Fp_linear1 ≈ identity (with small noise)
    #   - Fp_linear2 ≈ small values (residual dominates)
    fp_initialized = 0
    for i in range(num_layers):
        fp_prefix = f'layers.{i}.partial_attention'
        
        # Fp_linear1: Initialize close to identity
        fp1_key = f'{fp_prefix}.Fp_linear1.weight'
        if fp1_key in palm_state:
            weight = palm_state[fp1_key]
            hidden_size = weight.shape[0]
            device = weight.device
            dtype = weight.dtype
            
            # Small random noise + scaled identity for residual path
            # The identity component ensures Fp(x) ≈ x initially
            identity_like = torch.eye(hidden_size, device=device, dtype=dtype)
            palm_state[fp1_key] = 0.01 * torch.randn_like(weight) + 0.1 * identity_like
            fp_initialized += 1
        
        # Fp_linear2: Initialize to ZERO (not small noise)
        # With ReLU activation between Fp_linear1 and Fp_linear2, small noise can still
        # produce non-zero outputs. Zero init ensures Fp block outputs exactly 0 initially,
        # letting the residual connection dominate completely.
        fp2_key = f'{fp_prefix}.Fp_linear2.weight'
        if fp2_key in palm_state:
            palm_state[fp2_key] = torch.zeros_like(palm_state[fp2_key])
            fp_initialized += 1
        
        # Zero biases if present (lets the weight matrices do the work)
        for bias_key in [f'{fp_prefix}.Fp_linear1.bias', f'{fp_prefix}.Fp_linear2.bias']:
            if bias_key in palm_state:
                palm_state[bias_key] = torch.zeros_like(palm_state[bias_key])
    
    if fp_initialized > 0:
        bootstrapped_components.append('Fp_network')
        logger.info(f"✓ Initialized Fp network as near-identity (residual-dominant) "
                   f"({fp_initialized} tensors)")
    
    # Load the modified state dict back to model
    palm_model.load_state_dict(palm_state, strict=False)
    
    if bootstrapped_components:
        logger.info(f"PALM component bootstrapping complete: {bootstrapped_components}")
    
    return palm_state

