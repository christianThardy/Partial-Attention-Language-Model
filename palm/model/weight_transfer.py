"""
Weight Transfer Utilities for PALM

Supports transferring pretrained weights from various HuggingFace models
(Llama, Qwen, Mistral, Phi, Gemma, Falcon, Seed/Hermes, etc.) to PALMModel architecture.

Key features:
- Transfers RoPE (Rotary Position Embeddings) inv_freq for preserving positional understanding
- Bootstraps PALM-specific components (partial attention, SAE head) from pretrained weights
- Handles GQA → MHA expansion for partial attention

Updated for modern PALM architecture:
- SwiGLU MLP (gate_proj, up_proj, down_proj)
- Pre-Norm with RMSNorm (attn_norm, partial_attn_norm, mlp_norm, final_norm)
"""

import gc
import logging
from typing import Dict, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


# ARCHITECTURE DETECTION
def detect_model_architecture(model_name_or_config: Union[str, "PretrainedConfig"]) -> str:
    """
    Detect the architecture type from a model name or config.
    
    Returns:
        Architecture family identifier: 'llama', 'qwen', 'mistral', 'phi', 'gemma', 'falcon', 'seed'
    """
    if isinstance(model_name_or_config, str):
        config = AutoConfig.from_pretrained(model_name_or_config, trust_remote_code=True)
    else:
        config = model_name_or_config
    
    arch = getattr(config, 'model_type', '').lower()
    architectures = getattr(config, 'architectures', [])
    
    if 'llama' in arch or any('llama' in a.lower() for a in architectures):
        return 'llama'
    elif 'seed' in arch or any('seed' in a.lower() for a in architectures):
        # ByteDance Seed architecture (used by NousResearch Hermes 4.x)
        # Uses Llama-compatible weight structure with minor differences
        return 'seed'
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


# WEIGHT KEY MAPPINGS
def get_weight_mapping(arch_type: str, num_layers: int) -> Dict[str, Optional[str]]:
    """
    Get weight key mappings from source model to PALM model.
    
    Updated for modern PALM architecture:
    - SwiGLU MLP: gate_proj, up_proj, down_proj
    - Pre-Norm: attn_norm, mlp_norm, final_norm
    
    Returns:
        Dict mapping source keys to PALM keys (None means skip)
    """
    mapping = {}
    
    if arch_type in ['llama', 'mistral', 'gemma', 'seed']:
        # Embeddings and LM head
        mapping['model.embed_tokens.weight'] = 'embeddings.word_embeddings.weight'
        mapping['lm_head.weight'] = 'lm_head.weight'
        
        # Final norm (maps to PALM's final_norm for Pre-Norm architecture)
        mapping['model.norm.weight'] = 'final_norm.weight'
        
        for i in range(num_layers):
            src_prefix = f'model.layers.{i}'
            dst_prefix = f'layers.{i}'
            
            # Self-attention projections
            mapping[f'{src_prefix}.self_attn.q_proj.weight'] = f'{dst_prefix}.attention.query.weight'
            mapping[f'{src_prefix}.self_attn.k_proj.weight'] = f'{dst_prefix}.attention.key.weight'
            mapping[f'{src_prefix}.self_attn.v_proj.weight'] = f'{dst_prefix}.attention.value.weight'
            mapping[f'{src_prefix}.self_attn.o_proj.weight'] = f'{dst_prefix}.attention.dense.weight'
            
            # Seed/Hermes models have attention_bias=True, handle bias weights
            if arch_type == 'seed':
                mapping[f'{src_prefix}.self_attn.q_proj.bias'] = f'{dst_prefix}.attention.query.bias'
                mapping[f'{src_prefix}.self_attn.k_proj.bias'] = f'{dst_prefix}.attention.key.bias'
                mapping[f'{src_prefix}.self_attn.v_proj.bias'] = f'{dst_prefix}.attention.value.bias'
                # Note: Seed has attention_out_bias=False, so no o_proj bias
            
            # Pre-attention norm → PALM's attn_norm (RMSNorm)
            mapping[f'{src_prefix}.input_layernorm.weight'] = f'{dst_prefix}.attn_norm.weight'
            
            # Post-attention / MLP norm → PALM's mlp_norm (RMSNorm)
            mapping[f'{src_prefix}.post_attention_layernorm.weight'] = f'{dst_prefix}.mlp_norm.weight'
            
            # SwiGLU MLP (Llama-style gate_proj, up_proj, down_proj → PALM's SwiGLU)
            mapping[f'{src_prefix}.mlp.gate_proj.weight'] = f'{dst_prefix}.mlp.gate_proj.weight'
            mapping[f'{src_prefix}.mlp.up_proj.weight'] = f'{dst_prefix}.mlp.up_proj.weight'
            mapping[f'{src_prefix}.mlp.down_proj.weight'] = f'{dst_prefix}.mlp.down_proj.weight'
            
    elif arch_type == 'qwen':
        # Qwen v1 and v2 embeddings
        mapping['transformer.wte.weight'] = 'embeddings.word_embeddings.weight'
        mapping['model.embed_tokens.weight'] = 'embeddings.word_embeddings.weight'
        mapping['lm_head.weight'] = 'lm_head.weight'
        
        # Final norm
        mapping['transformer.ln_f.weight'] = 'final_norm.weight'
        mapping['model.norm.weight'] = 'final_norm.weight'
        
        for i in range(num_layers):
            dst_prefix = f'layers.{i}'
            
            # Qwen v1 style
            src_v1 = f'transformer.h.{i}'
            mapping[f'{src_v1}.attn.q_proj.weight'] = f'{dst_prefix}.attention.query.weight'
            mapping[f'{src_v1}.attn.k_proj.weight'] = f'{dst_prefix}.attention.key.weight'
            mapping[f'{src_v1}.attn.v_proj.weight'] = f'{dst_prefix}.attention.value.weight'
            mapping[f'{src_v1}.attn.c_proj.weight'] = f'{dst_prefix}.attention.dense.weight'
            mapping[f'{src_v1}.ln_1.weight'] = f'{dst_prefix}.attn_norm.weight'
            mapping[f'{src_v1}.ln_2.weight'] = f'{dst_prefix}.mlp_norm.weight'
            mapping[f'{src_v1}.mlp.w1.weight'] = f'{dst_prefix}.mlp.gate_proj.weight'
            mapping[f'{src_v1}.mlp.w2.weight'] = f'{dst_prefix}.mlp.down_proj.weight'
            mapping[f'{src_v1}.mlp.w3.weight'] = f'{dst_prefix}.mlp.up_proj.weight'
            
            # Qwen2 style
            src_v2 = f'model.layers.{i}'
            mapping[f'{src_v2}.self_attn.q_proj.weight'] = f'{dst_prefix}.attention.query.weight'
            mapping[f'{src_v2}.self_attn.k_proj.weight'] = f'{dst_prefix}.attention.key.weight'
            mapping[f'{src_v2}.self_attn.v_proj.weight'] = f'{dst_prefix}.attention.value.weight'
            mapping[f'{src_v2}.self_attn.o_proj.weight'] = f'{dst_prefix}.attention.dense.weight'
            mapping[f'{src_v2}.input_layernorm.weight'] = f'{dst_prefix}.attn_norm.weight'
            mapping[f'{src_v2}.post_attention_layernorm.weight'] = f'{dst_prefix}.mlp_norm.weight'
            mapping[f'{src_v2}.mlp.gate_proj.weight'] = f'{dst_prefix}.mlp.gate_proj.weight'
            mapping[f'{src_v2}.mlp.up_proj.weight'] = f'{dst_prefix}.mlp.up_proj.weight'
            mapping[f'{src_v2}.mlp.down_proj.weight'] = f'{dst_prefix}.mlp.down_proj.weight'
            
    elif arch_type == 'phi':
        mapping['model.embed_tokens.weight'] = 'embeddings.word_embeddings.weight'
        mapping['lm_head.weight'] = 'lm_head.weight'
        mapping['model.final_layernorm.weight'] = 'final_norm.weight'
        
        for i in range(num_layers):
            src_prefix = f'model.layers.{i}'
            dst_prefix = f'layers.{i}'
            
            mapping[f'{src_prefix}.self_attn.q_proj.weight'] = f'{dst_prefix}.attention.query.weight'
            mapping[f'{src_prefix}.self_attn.k_proj.weight'] = f'{dst_prefix}.attention.key.weight'
            mapping[f'{src_prefix}.self_attn.v_proj.weight'] = f'{dst_prefix}.attention.value.weight'
            mapping[f'{src_prefix}.self_attn.dense.weight'] = f'{dst_prefix}.attention.dense.weight'
            
            mapping[f'{src_prefix}.input_layernorm.weight'] = f'{dst_prefix}.attn_norm.weight'
            
            # Phi uses different MLP structure - may need adaptation
            # Phi-2 style (fc1/fc2 → needs conversion to SwiGLU)
            mapping[f'{src_prefix}.mlp.fc1.weight'] = f'{dst_prefix}.mlp.up_proj.weight'
            mapping[f'{src_prefix}.mlp.fc2.weight'] = f'{dst_prefix}.mlp.down_proj.weight'
            
            # Phi-3 style (SwiGLU)
            mapping[f'{src_prefix}.mlp.gate_up_proj.weight'] = None  # Fused, needs special handling
            mapping[f'{src_prefix}.mlp.down_proj.weight'] = f'{dst_prefix}.mlp.down_proj.weight'
            
    elif arch_type == 'falcon':
        mapping['transformer.word_embeddings.weight'] = 'embeddings.word_embeddings.weight'
        mapping['lm_head.weight'] = 'lm_head.weight'
        mapping['transformer.ln_f.weight'] = 'final_norm.weight'
        
        for i in range(num_layers):
            src_prefix = f'transformer.h.{i}'
            dst_prefix = f'layers.{i}'
            
            # Note: Falcon uses fused QKV which requires special handling
            mapping[f'{src_prefix}.self_attention.query_key_value.weight'] = None
            mapping[f'{src_prefix}.self_attention.dense.weight'] = f'{dst_prefix}.attention.dense.weight'
            
            mapping[f'{src_prefix}.input_layernorm.weight'] = f'{dst_prefix}.attn_norm.weight'
            
            # Falcon MLP
            mapping[f'{src_prefix}.mlp.dense_h_to_4h.weight'] = f'{dst_prefix}.mlp.up_proj.weight'
            mapping[f'{src_prefix}.mlp.dense_4h_to_h.weight'] = f'{dst_prefix}.mlp.down_proj.weight'
    
    return mapping


def _get_rope_key_patterns(arch_type: str, layer_idx: int) -> Dict[str, str]:
    """
    Get source RoPE (inv_freq) key patterns for a given architecture.
    
    Most modern models store inv_freq as a buffer in the rotary embedding class.
    """
    i = layer_idx
    
    if arch_type in ['llama', 'mistral', 'gemma', 'seed']:
        return {
            'inv_freq': f'model.layers.{i}.self_attn.rotary_emb.inv_freq',
        }
    elif arch_type == 'qwen':
        return {
            'inv_freq': f'model.layers.{i}.self_attn.rotary_emb.inv_freq',
            'inv_freq_alt': f'transformer.h.{i}.attn.rotary_emb.inv_freq',
        }
    elif arch_type == 'phi':
        return {
            'inv_freq': f'model.layers.{i}.self_attn.rotary_emb.inv_freq',
        }
    else:
        return {
            'inv_freq': f'model.layers.{i}.self_attn.rotary_emb.inv_freq',
        }


# WEIGHT TRANSFER
def transfer_weights_to_palm(
    palm_model: torch.nn.Module,
    source_model_name: str,
    device: str = 'cpu',
    dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    """
    Transfer pretrained weights from any supported HuggingFace model to PALMModel.
    
    Key transfers:
    1. Word embeddings, attention projections, MLP weights
    2. RoPE inv_freq parameters (preserves pretrained positional understanding)
    3. Bootstraps PALM-specific components from pretrained weights
    
    Components initialized fresh (not from pretrained):
    - language_embeddings (PALM-specific source/target differentiation)
    - partial_attention Fp network (initialized as near-identity)
    - partial_attn_norm (cloned from attn_norm)
    
    Args:
        palm_model: Initialized PALMModel instance
        source_model_name: HuggingFace model name/path
        device: Device to load source model on temporarily
        dtype: Data type for loading source model
    
    Returns:
        palm_model with transferred weights
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
    
    # Get number of layers to transfer
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
                palm_state[dst_key] = src_weight[:dst_weight.shape[0]].clone().to(dst_weight.dtype)
                transferred += 1
                logger.info(f"Truncated {src_key} from {src_weight.shape} to {dst_weight.shape}")
            elif len(src_weight.shape) == len(dst_weight.shape) == 2:
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
    
    # Transfer RoPE inv_freq parameters
    rope_transferred = transfer_rope_parameters(palm_model, source_state, arch_type, num_layers)
    
    # Bootstrap PALM-specific components from pretrained weights
    palm_state = bootstrap_palm_components(palm_model, source_state, palm_state, arch_type=arch_type)
    
    # Clean up source model
    del source_model
    del source_state
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info(f"Weight transfer complete:")
    logger.info(f"  - Transferred: {transferred} tensors")
    logger.info(f"  - RoPE inv_freq: {rope_transferred} layers")
    logger.info(f"  - Skipped: {skipped} tensors")
    logger.info(f"  - Shape mismatches: {shape_mismatch} tensors")
    
    # Components initialized fresh
    fresh_components = ['language_embeddings', 'partial_attn_norm']
    logger.info(f"Components initialized fresh (not from pretrained): {fresh_components}")
    
    return palm_model


def transfer_rope_parameters(
    palm_model: torch.nn.Module,
    source_state: Dict[str, torch.Tensor],
    arch_type: str,
    num_layers: int
) -> int:
    """
    Transfer RoPE inv_freq parameters from source model to PALM.
    
    This preserves the pretrained positional understanding from the backbone model.
    The inv_freq determines the rotation frequencies used in RoPE.
    
    Args:
        palm_model: The PALM model to update
        source_state: State dict from source model
        arch_type: Architecture type
        num_layers: Number of layers to transfer
    
    Returns:
        Number of layers where RoPE was transferred
    """
    transferred = 0
    
    for i in range(num_layers):
        patterns = _get_rope_key_patterns(arch_type, i)
        
        # Find source inv_freq
        src_inv_freq = None
        for key_name in ['inv_freq', 'inv_freq_alt']:
            src_key = patterns.get(key_name)
            if src_key and src_key in source_state:
                src_inv_freq = source_state[src_key]
                break
        
        if src_inv_freq is None:
            continue
        
        # Transfer to main attention
        attn_rotary = palm_model.layers[i].attention.rotary_emb
        if hasattr(attn_rotary, 'inv_freq'):
            if attn_rotary.inv_freq.shape == src_inv_freq.shape:
                attn_rotary.inv_freq.copy_(src_inv_freq)
                # Clear cached cos/sin so they're recomputed with new inv_freq
                attn_rotary._cos_cached = None
                attn_rotary._sin_cached = None
                attn_rotary._cached_seq_len = 0
            else:
                logger.debug(f"RoPE shape mismatch layer {i}: src {src_inv_freq.shape} vs dst {attn_rotary.inv_freq.shape}")
        
        # Transfer to partial attention (same frequencies for consistency)
        partial_attn_rotary = palm_model.layers[i].partial_attention.rotary_emb
        if hasattr(partial_attn_rotary, 'inv_freq'):
            if partial_attn_rotary.inv_freq.shape == src_inv_freq.shape:
                partial_attn_rotary.inv_freq.copy_(src_inv_freq)
                partial_attn_rotary._cos_cached = None
                partial_attn_rotary._sin_cached = None
                partial_attn_rotary._cached_seq_len = 0
        
        transferred += 1
    
    if transferred > 0:
        logger.info(f"✓ Transferred RoPE inv_freq to {transferred} layers (both attention and partial_attention)")
    
    return transferred


# PALM COMPONENT BOOTSTRAPPING
def _get_attention_key_patterns(arch_type: str, layer_idx: int) -> Dict[str, str]:
    """Get source attention key patterns for a given architecture."""
    i = layer_idx
    
    if arch_type in ['llama', 'mistral', 'gemma', 'seed']:
        return {
            'q': f'model.layers.{i}.self_attn.q_proj.weight',
            'k': f'model.layers.{i}.self_attn.k_proj.weight',
            'v': f'model.layers.{i}.self_attn.v_proj.weight',
            'o': f'model.layers.{i}.self_attn.o_proj.weight',
            'attn_norm': f'model.layers.{i}.input_layernorm.weight',
        }
    elif arch_type == 'qwen':
        return {
            'q': f'model.layers.{i}.self_attn.q_proj.weight',
            'k': f'model.layers.{i}.self_attn.k_proj.weight',
            'v': f'model.layers.{i}.self_attn.v_proj.weight',
            'o': f'model.layers.{i}.self_attn.o_proj.weight',
            'attn_norm': f'model.layers.{i}.input_layernorm.weight',
            'q_alt': f'transformer.h.{i}.attn.q_proj.weight',
            'k_alt': f'transformer.h.{i}.attn.k_proj.weight',
            'v_alt': f'transformer.h.{i}.attn.v_proj.weight',
            'o_alt': f'transformer.h.{i}.attn.c_proj.weight',
            'attn_norm_alt': f'transformer.h.{i}.ln_1.weight',
        }
    elif arch_type == 'phi':
        return {
            'q': f'model.layers.{i}.self_attn.q_proj.weight',
            'k': f'model.layers.{i}.self_attn.k_proj.weight',
            'v': f'model.layers.{i}.self_attn.v_proj.weight',
            'o': f'model.layers.{i}.self_attn.dense.weight',
            'attn_norm': f'model.layers.{i}.input_layernorm.weight',
        }
    elif arch_type == 'falcon':
        return {
            'q': None,
            'k': None,
            'v': None,
            'o': f'transformer.h.{i}.self_attention.dense.weight',
            'attn_norm': f'transformer.h.{i}.input_layernorm.weight',
        }
    else:
        return {
            'q': f'model.layers.{i}.self_attn.q_proj.weight',
            'k': f'model.layers.{i}.self_attn.k_proj.weight',
            'v': f'model.layers.{i}.self_attn.v_proj.weight',
            'o': f'model.layers.{i}.self_attn.o_proj.weight',
            'attn_norm': f'model.layers.{i}.input_layernorm.weight',
        }


def bootstrap_palm_components(
    palm_model: torch.nn.Module,
    source_state: Dict[str, torch.Tensor],
    palm_state: Dict[str, torch.Tensor],
    arch_type: str = 'llama'
) -> Dict[str, torch.Tensor]:
    """
    Bootstrap PALM-specific components from pretrained weights.
    
    Strategy:
    1. partial_attention Q/K/V/dense ← clone from regular attention
    2. partial_attn_norm ← clone from attn_norm (Pre-Norm architecture)
    3. Fp network ← identity-like init (Fp(x) ≈ x initially)
    
    Note: SAE head is tied to lm_head, so no separate bootstrapping needed.
    """
    num_layers = palm_model.config.num_hidden_layers
    bootstrapped_components = []
    
    # 1. PARTIAL ATTENTION Q/K/V/DENSE ← CLONE FROM REGULAR ATTENTION
    partial_attention_bootstrapped = 0
    for i in range(num_layers):
        patterns = _get_attention_key_patterns(arch_type, i)
        
        mappings = [
            (patterns['q'], f'layers.{i}.partial_attention.query.weight'),
            (patterns['k'], f'layers.{i}.partial_attention.key.weight'),
            (patterns['v'], f'layers.{i}.partial_attention.value.weight'),
            (patterns['o'], f'layers.{i}.partial_attention.dense.weight'),
        ]
        
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
            if src_key is None:
                continue
            
            actual_src_key = src_key
            if src_key not in source_state and fallback_mappings:
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
                if dst_w.shape[0] > src_w.shape[0] and dst_w.shape[0] % src_w.shape[0] == 0:
                    num_repeats = dst_w.shape[0] // src_w.shape[0]
                    expanded = src_w.repeat_interleave(num_repeats, dim=0)
                    if expanded.shape[1] != dst_w.shape[1]:
                        min_in = min(expanded.shape[1], dst_w.shape[1])
                        palm_state[dst_key][:, :min_in] = expanded[:, :min_in].clone().to(dst_dtype)
                    else:
                        palm_state[dst_key] = expanded.clone().to(dst_dtype)
                    logger.debug(f"GQA→MHA expand {actual_src_key} {src_w.shape} -> {dst_key} {dst_w.shape}")
                else:
                    min_out = min(src_w.shape[0], dst_w.shape[0])
                    min_in = min(src_w.shape[1], dst_w.shape[1])
                    palm_state[dst_key][:min_out, :min_in] = src_w[:min_out, :min_in].clone().to(dst_dtype)
            
            partial_attention_bootstrapped += 1
    
    if partial_attention_bootstrapped > 0:
        bootstrapped_components.append('partial_attention')
        logger.info(f"✓ Bootstrapped partial_attention Q/K/V/dense for {num_layers} layers")
    
    # 2. PARTIAL_ATTN_NORM ← CLONE FROM ATTN_NORM (Pre-Norm architecture)
    norm_bootstrapped = 0
    for i in range(num_layers):
        # Clone attn_norm to partial_attn_norm
        src_norm_key = f'layers.{i}.attn_norm.weight'
        dst_norm_key = f'layers.{i}.partial_attn_norm.weight'
        
        if src_norm_key in palm_state and dst_norm_key in palm_state:
            palm_state[dst_norm_key] = palm_state[src_norm_key].clone()
            norm_bootstrapped += 1
    
    if norm_bootstrapped > 0:
        bootstrapped_components.append('partial_attn_norm')
        logger.info(f"✓ Cloned attn_norm to partial_attn_norm for {norm_bootstrapped} layers")
    
    # 3. Fp NETWORK ← IDENTITY-LIKE INITIALIZATION
    # Goal: Fp(x) ≈ x initially, so partial attention starts coherent
    # 
    # The Fp network is: P = Fp_linear2(SiLU(Fp_linear1(x))) + SiLU(Fp_linear1(x))
    # For identity-like behavior, we need SiLU(Fp_linear1(x)) ≈ x
    # 
    # SiLU(x) = x * sigmoid(x) ≈ x for x >> 0, but ≈ 0 for x << 0
    # For normalized hidden states (mean ≈ 0), half values get squashed!
    #
    # Solution: Scale up Fp_linear1 so SiLU doesn't dampen too much.
    # With Fp_linear1 ≈ 1.7*I, SiLU(1.7*x) ≈ x for typical normalized inputs.
    # Then Fp_linear2 ≈ 0 gives P = 0 + SiLU(1.7*x) ≈ x
    #
    # Also add very small noise for symmetry breaking.
    fp_initialized = 0
    FP_SCALE = 1.7  # Compensates for SiLU dampening on normalized inputs
    FP_NOISE = 0.001  # Small noise for symmetry breaking
    
    for i in range(num_layers):
        fp_prefix = f'layers.{i}.partial_attention'
        
        fp1_key = f'{fp_prefix}.Fp_linear1.weight'
        if fp1_key in palm_state:
            weight = palm_state[fp1_key]
            hidden_size = weight.shape[0]
            device = weight.device
            dtype = weight.dtype
            
            # Scaled identity + tiny noise
            identity_scaled = FP_SCALE * torch.eye(hidden_size, device=device, dtype=dtype)
            noise = FP_NOISE * torch.randn_like(weight)
            palm_state[fp1_key] = identity_scaled + noise
            fp_initialized += 1
        
        fp2_key = f'{fp_prefix}.Fp_linear2.weight'
        if fp2_key in palm_state:
            # Zero init for Fp_linear2 so P ≈ Pl1
            palm_state[fp2_key] = FP_NOISE * torch.randn_like(palm_state[fp2_key])
            fp_initialized += 1
        
        for bias_key in [f'{fp_prefix}.Fp_linear1.bias', f'{fp_prefix}.Fp_linear2.bias']:
            if bias_key in palm_state:
                palm_state[bias_key] = torch.zeros_like(palm_state[bias_key])
    
    if fp_initialized > 0:
        bootstrapped_components.append('Fp_network')
        logger.info(f"✓ Initialized Fp network as near-identity ({fp_initialized} tensors)")
    
    # Load the modified state dict back to model
    palm_model.load_state_dict(palm_state, strict=False)
    
    if bootstrapped_components:
        logger.info(f"PALM component bootstrapping complete: {bootstrapped_components}")
    
    return palm_state
