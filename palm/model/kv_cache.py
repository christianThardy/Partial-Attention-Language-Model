"""
KV Cache Optimization Strategies for PALM

This module implements two complementary KV cache optimization strategies:

1. Cross-Layer KV Sharing (Strategy #3):
   - Shares Partial Attention KV across layer groups
   - The source grounding signal doesn't need 28 unique transformations
   - Expected savings: ~37% KV reduction with 4 groups, ~25% with 2 groups

2. Hybrid Multi-Granularity Cache (Strategy #1):
   - Full precision for system prompt + most recent turn
   - 4-bit quantization for older conversation turns
   - PALM's SAE loss prepares the model for compressed representations
   - Expected savings: ~75% for older turns

Usage:
    config = PALMConfig(
        # Cross-Layer KV Sharing config
        kv_sharing_groups=4,           # Number of layer groups (4 = 7 layers each for 28 layers)
        share_partial_kv=True,         # Enable cross-layer sharing for Partial Attention
        
        # Hybrid Granularity Cache config
        enable_kv_quantization=True,   # Enable 4-bit quantization for older turns
        quantize_after_turns=1,        # Quantize KV after this many turns
    )
"""

import math
import logging
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class KVCacheConfig:
    """Configuration for KV cache optimization strategies."""
    
    # Cross-Layer KV Sharing (Strategy #3)
    kv_sharing_groups: int = 1  # 1 = no sharing, 4 = share across 4 groups
    share_partial_kv: bool = False  # Only share Partial Attention KV (not Self-Attention)
    
    # Hybrid Granularity Cache (Strategy #1)
    enable_kv_quantization: bool = False
    quantize_after_turns: int = 1  # Keep recent N turns at full precision
    quantization_bits: int = 4  # 4-bit quantization for older turns
    
    # General settings
    num_hidden_layers: int = 28
    num_kv_heads: int = 8
    head_dim: int = 128
    
    @classmethod
    def from_palm_config(cls, config) -> "KVCacheConfig":
        """Create KVCacheConfig from PALMConfig."""
        return cls(
            kv_sharing_groups=getattr(config, 'kv_sharing_groups', 1),
            share_partial_kv=getattr(config, 'share_partial_kv', False),
            enable_kv_quantization=getattr(config, 'enable_kv_quantization', False),
            quantize_after_turns=getattr(config, 'quantize_after_turns', 1),
            quantization_bits=getattr(config, 'quantization_bits', 4),
            num_hidden_layers=config.num_hidden_layers,
            num_kv_heads=getattr(config, 'num_kv_heads', config.num_attention_heads),
            head_dim=config.hidden_size // config.num_attention_heads,
        )


# =============================================================================
# STRATEGY #1: HYBRID MULTI-GRANULARITY CACHE (HMC)
# =============================================================================

class QuantizedKVCache:
    """
    4-bit quantized KV cache using symmetric per-channel quantization.
    
    PALM's SAE (Source Auto-Encoding) loss trains the model to be robust to
    source reconstructions, making it uniquely prepared for KV quantization.
    
    Quantization scheme:
    - Symmetric quantization: values mapped to [-7, 7] for 4-bit
    - Per-channel scaling: each head has its own scale factor
    - Dequantization on-the-fly during attention computation
    """
    
    def __init__(
        self,
        bits: int = 4,
        device: torch.device = None,
    ):
        self.bits = bits
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Quantization range for symmetric quantization
        self.qmax = (1 << (bits - 1)) - 1  # 7 for 4-bit
        self.qmin = -self.qmax  # -7 for 4-bit
        
        # Storage for quantized values
        self.quantized_keys: Optional[torch.Tensor] = None  # int8 storage
        self.quantized_values: Optional[torch.Tensor] = None
        self.key_scales: Optional[torch.Tensor] = None  # Per-channel scales
        self.value_scales: Optional[torch.Tensor] = None
        
        # Shape info
        self.batch_size = 0
        self.num_heads = 0
        self.seq_len = 0
        self.head_dim = 0
    
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to int8 with per-channel (per-head) scaling.
        
        Args:
            tensor: Shape (batch, num_heads, seq_len, head_dim)
        
        Returns:
            quantized: int8 tensor
            scales: Per-head scale factors (batch, num_heads, 1, 1)
        """
        # Compute per-head max absolute value
        abs_max = tensor.abs().amax(dim=(-2, -1), keepdim=True)  # (batch, heads, 1, 1)
        
        # Avoid division by zero
        abs_max = torch.clamp(abs_max, min=1e-8)
        
        # Compute scale and quantize
        scales = abs_max / self.qmax
        quantized = torch.round(tensor / scales).clamp(self.qmin, self.qmax).to(torch.int8)
        
        return quantized, scales
    
    def dequantize(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        Dequantize int8 tensor back to float.
        
        Args:
            quantized: int8 tensor (batch, num_heads, seq_len, head_dim)
            scales: Per-head scales (batch, num_heads, 1, 1)
        
        Returns:
            Dequantized float tensor
        """
        return quantized.float() * scales
    
    def update(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Update cache with new quantized KV pairs.
        
        Args:
            keys: (batch, num_heads, seq_len, head_dim)
            values: (batch, num_heads, seq_len, head_dim)
        """
        q_keys, k_scales = self.quantize(keys)
        q_values, v_scales = self.quantize(values)
        
        if self.quantized_keys is None:
            self.quantized_keys = q_keys
            self.quantized_values = q_values
            self.key_scales = k_scales
            self.value_scales = v_scales
        else:
            # Concatenate along sequence dimension
            self.quantized_keys = torch.cat([self.quantized_keys, q_keys], dim=2)
            self.quantized_values = torch.cat([self.quantized_values, q_values], dim=2)
            # Update scales (use max to handle varying scales)
            self.key_scales = torch.max(self.key_scales, k_scales)
            self.value_scales = torch.max(self.value_scales, v_scales)
        
        # Update shape info
        self.batch_size, self.num_heads, self.seq_len, self.head_dim = self.quantized_keys.shape
    
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dequantized KV tensors."""
        if self.quantized_keys is None:
            return None, None
        
        keys = self.dequantize(self.quantized_keys, self.key_scales)
        values = self.dequantize(self.quantized_values, self.value_scales)
        
        return keys, values
    
    def clear(self):
        """Clear the cache."""
        self.quantized_keys = None
        self.quantized_values = None
        self.key_scales = None
        self.value_scales = None
        self.seq_len = 0


class HybridGranularityCache:
    """
    Hybrid Multi-Granularity Cache (HMC) for PALM.
    
    Maintains different precision levels for different parts of the conversation:
    - System prompt: Full precision (always important for grounding)
    - Recent turns: Full precision (high temporal relevance)
    - Older turns: 4-bit quantized (PALM's SAE prepares for this)
    
    The key insight is that PALM's SAE loss trains the model to reconstruct
    source tokens from hidden representations, making it robust to the
    compression introduced by quantization.
    """
    
    def __init__(
        self,
        num_layers: int,
        quantize_after_turns: int = 1,
        quantization_bits: int = 4,
        device: torch.device = None,
    ):
        self.num_layers = num_layers
        self.quantize_after_turns = quantize_after_turns
        self.quantization_bits = quantization_bits
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Per-layer caches
        # Each layer has: (full_precision_kv, quantized_kv)
        # Full precision: Tuple of (key, value) tensors
        # Quantized: QuantizedKVCache instance
        self.full_precision_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(num_layers)
        ]
        self.quantized_caches: List[QuantizedKVCache] = [
            QuantizedKVCache(bits=quantization_bits, device=device) 
            for _ in range(num_layers)
        ]
        
        # Turn boundary tracking
        self.turn_boundaries: List[int] = []  # Sequence positions where turns start
        self.current_turn = 0
        self.system_prompt_end = 0  # End of system prompt (never quantized)
    
    def set_system_prompt_end(self, position: int):
        """Mark where system prompt ends (this part is never quantized)."""
        self.system_prompt_end = position
    
    def mark_turn_boundary(self, position: int):
        """Mark start of a new conversation turn."""
        self.turn_boundaries.append(position)
        self.current_turn = len(self.turn_boundaries)
    
    def update_layer(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        quantize: bool = False,
    ):
        """
        Update cache for a specific layer.
        
        Args:
            layer_idx: Which layer's cache to update
            keys: New key tensor (batch, num_heads, new_seq_len, head_dim)
            values: New value tensor
            quantize: Whether to quantize these KVs (for older turns)
        """
        if quantize:
            self.quantized_caches[layer_idx].update(keys, values)
        else:
            if self.full_precision_caches[layer_idx] is None:
                self.full_precision_caches[layer_idx] = (keys, values)
            else:
                old_k, old_v = self.full_precision_caches[layer_idx]
                self.full_precision_caches[layer_idx] = (
                    torch.cat([old_k, keys], dim=2),
                    torch.cat([old_v, values], dim=2),
                )
    
    def get_layer_kv(
        self,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get combined KV for a layer (quantized + full precision).
        
        Returns concatenated (keys, values) with older turns dequantized.
        """
        keys_parts = []
        values_parts = []
        
        # Get quantized KV (older turns)
        q_keys, q_values = self.quantized_caches[layer_idx].get_kv()
        if q_keys is not None:
            keys_parts.append(q_keys)
            values_parts.append(q_values)
        
        # Get full precision KV (recent turns)
        if self.full_precision_caches[layer_idx] is not None:
            fp_keys, fp_values = self.full_precision_caches[layer_idx]
            keys_parts.append(fp_keys)
            values_parts.append(fp_values)
        
        if not keys_parts:
            return None, None
        
        return (
            torch.cat(keys_parts, dim=2),
            torch.cat(values_parts, dim=2),
        )
    
    def maybe_quantize_old_turns(self):
        """
        Move older turns from full precision to quantized storage.
        
        Called after each new turn is added. Keeps only the most recent
        `quantize_after_turns` turns at full precision.
        """
        if self.current_turn <= self.quantize_after_turns:
            return  # Not enough turns to quantize yet
        
        turns_to_quantize = self.current_turn - self.quantize_after_turns
        if turns_to_quantize <= 0:
            return
        
        # Find the cutoff position (after system prompt, before recent turns)
        if turns_to_quantize < len(self.turn_boundaries):
            cutoff = self.turn_boundaries[turns_to_quantize]
        else:
            return  # Nothing to quantize
        
        # Ensure we don't quantize system prompt
        cutoff = max(cutoff, self.system_prompt_end)
        
        for layer_idx in range(self.num_layers):
            fp_cache = self.full_precision_caches[layer_idx]
            if fp_cache is None:
                continue
            
            keys, values = fp_cache
            seq_len = keys.shape[2]
            
            if cutoff >= seq_len:
                continue  # Nothing to quantize for this layer
            
            # Split: [to_quantize | keep_full_precision]
            keys_to_quantize = keys[:, :, :cutoff, :]
            values_to_quantize = values[:, :, :cutoff, :]
            
            keys_keep = keys[:, :, cutoff:, :]
            values_keep = values[:, :, cutoff:, :]
            
            # Quantize the older portion
            self.quantized_caches[layer_idx].update(keys_to_quantize, values_to_quantize)
            
            # Update full precision cache with only recent portion
            self.full_precision_caches[layer_idx] = (keys_keep, values_keep)
    
    def clear(self):
        """Clear all caches."""
        for i in range(self.num_layers):
            self.full_precision_caches[i] = None
            self.quantized_caches[i].clear()
        self.turn_boundaries = []
        self.current_turn = 0
        self.system_prompt_end = 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        full_precision_bytes = 0
        quantized_bytes = 0
        
        for i in range(self.num_layers):
            if self.full_precision_caches[i] is not None:
                k, v = self.full_precision_caches[i]
                # Assuming bfloat16 (2 bytes per element)
                full_precision_bytes += k.numel() * 2 + v.numel() * 2
            
            qc = self.quantized_caches[i]
            if qc.quantized_keys is not None:
                # int8 (1 byte) + scales (2 bytes per head)
                quantized_bytes += qc.quantized_keys.numel()
                quantized_bytes += qc.quantized_values.numel()
                quantized_bytes += qc.key_scales.numel() * 2
                quantized_bytes += qc.value_scales.numel() * 2
        
        return {
            "full_precision_mb": full_precision_bytes / (1024 * 1024),
            "quantized_mb": quantized_bytes / (1024 * 1024),
            "total_mb": (full_precision_bytes + quantized_bytes) / (1024 * 1024),
            "compression_ratio": full_precision_bytes / max(quantized_bytes, 1) if quantized_bytes > 0 else 0,
        }


# =============================================================================
# STRATEGY #3: CROSS-LAYER KV SHARING
# =============================================================================

class CrossLayerKVManager:
    """
    Manages cross-layer KV sharing for Partial Attention.
    
    Key insight: PALM's Partial Attention source KV is:
    - Static during generation (already reused via past_key_value)
    - Processed through Fp network (learned transformation of source)
    - The "grounding signal" - doesn't need unique transformations per layer
    
    This manager groups layers and shares Partial Attention KV within groups:
    - 4 groups for 28 layers: layers 0-6, 7-13, 14-20, 21-27 share KV
    - 2 groups for 28 layers: layers 0-13, 14-27 share KV
    
    Expected savings:
    - 4 groups: 75% Partial Attention KV reduction (~37% total)
    - 2 groups: 50% Partial Attention KV reduction (~25% total)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_groups: int = 4,
        device: torch.device = None,
    ):
        """
        Args:
            num_layers: Total number of transformer layers
            num_groups: Number of layer groups for KV sharing
            device: Torch device
        """
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Compute layer-to-group mapping
        self.layers_per_group = num_layers // num_groups
        self.layer_to_group = {}
        for layer_idx in range(num_layers):
            group_idx = min(layer_idx // self.layers_per_group, num_groups - 1)
            self.layer_to_group[layer_idx] = group_idx
        
        # Shared KV storage: one per group
        # Each entry: (key, value) or None
        self.shared_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(num_groups)
        ]
        
        # Track which layers have contributed to shared KV
        self.contributor_layers: List[Optional[int]] = [None for _ in range(num_groups)]
        
        logger.info(f"CrossLayerKVManager: {num_layers} layers in {num_groups} groups "
                   f"({self.layers_per_group} layers/group)")
    
    def get_group_for_layer(self, layer_idx: int) -> int:
        """Get the group index for a given layer."""
        return self.layer_to_group.get(layer_idx, 0)
    
    def get_representative_layer(self, group_idx: int) -> int:
        """Get the first layer in a group (the representative that computes KV)."""
        return group_idx * self.layers_per_group
    
    def is_representative_layer(self, layer_idx: int) -> bool:
        """Check if this layer should compute the shared KV for its group."""
        group_idx = self.get_group_for_layer(layer_idx)
        return layer_idx == self.get_representative_layer(group_idx)
    
    def should_compute_kv(self, layer_idx: int) -> bool:
        """
        Determine if a layer should compute its own Partial Attention KV.
        
        Returns True if:
        - This is the first (representative) layer in its group, OR
        - The group's shared KV hasn't been computed yet
        """
        group_idx = self.get_group_for_layer(layer_idx)
        return self.shared_kv[group_idx] is None
    
    def store_shared_kv(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Store KV computed by a representative layer for sharing with group.
        
        Args:
            layer_idx: The layer that computed this KV
            keys: (batch, num_kv_heads, source_len, head_dim)
            values: (batch, num_kv_heads, source_len, head_dim)
        """
        group_idx = self.get_group_for_layer(layer_idx)
        self.shared_kv[group_idx] = (keys.detach(), values.detach())
        self.contributor_layers[group_idx] = layer_idx
    
    def get_shared_kv(
        self,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get shared KV for a layer from its group.
        
        Returns None if the group's KV hasn't been computed yet.
        """
        group_idx = self.get_group_for_layer(layer_idx)
        return self.shared_kv[group_idx]
    
    def clear(self):
        """Clear all shared KV caches."""
        for i in range(self.num_groups):
            self.shared_kv[i] = None
            self.contributor_layers[i] = None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stored_groups = sum(1 for kv in self.shared_kv if kv is not None)
        total_bytes = 0
        
        for kv in self.shared_kv:
            if kv is not None:
                keys, values = kv
                # Assuming bfloat16 (2 bytes)
                total_bytes += keys.numel() * 2 + values.numel() * 2
        
        # Calculate savings vs no sharing
        # Without sharing, each layer would store its own KV
        savings_factor = self.num_layers / self.num_groups if self.num_groups > 0 else 1
        
        return {
            "groups_stored": stored_groups,
            "total_groups": self.num_groups,
            "memory_mb": total_bytes / (1024 * 1024),
            "savings_factor": savings_factor,
            "partial_kv_reduction_pct": (1 - 1/savings_factor) * 100,
        }


# =============================================================================
# COMBINED PALM CACHE
# =============================================================================

class PALMCache:
    """
    Combined KV cache manager for PALM supporting both optimization strategies.
    
    This class coordinates:
    1. Cross-Layer KV Sharing for Partial Attention
    2. Hybrid Multi-Granularity quantization for older turns
    
    Both strategies work together:
    - Cross-layer sharing reduces the NUMBER of unique KV tensors
    - Quantization reduces the SIZE of stored KV tensors
    
    For Self-Attention (causal attention over generated tokens):
    - Standard per-layer KV cache
    - Optionally quantized for older turns
    
    For Partial Attention (cross-attention to source):
    - Shared across layer groups
    - Optionally quantized (source is "old" by definition)
    """
    
    def __init__(
        self,
        config: Union[KVCacheConfig, Any],
        device: torch.device = None,
    ):
        """
        Args:
            config: KVCacheConfig or PALMConfig
            device: Torch device
        """
        if not isinstance(config, KVCacheConfig):
            config = KVCacheConfig.from_palm_config(config)
        
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Strategy #3: Cross-Layer KV Sharing
        self.cross_layer_manager: Optional[CrossLayerKVManager] = None
        if config.share_partial_kv and config.kv_sharing_groups > 1:
            self.cross_layer_manager = CrossLayerKVManager(
                num_layers=config.num_hidden_layers,
                num_groups=config.kv_sharing_groups,
                device=self.device,
            )
        
        # Strategy #1: Hybrid Granularity Cache
        self.hybrid_cache: Optional[HybridGranularityCache] = None
        if config.enable_kv_quantization:
            self.hybrid_cache = HybridGranularityCache(
                num_layers=config.num_hidden_layers,
                quantize_after_turns=config.quantize_after_turns,
                quantization_bits=config.quantization_bits,
                device=self.device,
            )
        
        # Standard per-layer Self-Attention KV cache
        # Used when neither optimization is enabled, or for non-quantized portions
        self.self_attn_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(config.num_hidden_layers)
        ]
        
        # Standard per-layer Partial Attention KV cache (used when cross-layer sharing disabled)
        self.partial_attn_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(config.num_hidden_layers)
        ]
    
    # =========================================================================
    # Self-Attention Cache Methods
    # =========================================================================
    
    def update_self_attn(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Update Self-Attention KV cache for a layer."""
        if self.hybrid_cache is not None:
            # Use hybrid granularity cache
            self.hybrid_cache.update_layer(layer_idx, keys, values, quantize=False)
        else:
            # Standard cache
            if self.self_attn_cache[layer_idx] is None:
                self.self_attn_cache[layer_idx] = (keys, values)
            else:
                old_k, old_v = self.self_attn_cache[layer_idx]
                self.self_attn_cache[layer_idx] = (
                    torch.cat([old_k, keys], dim=2),
                    torch.cat([old_v, values], dim=2),
                )
    
    def get_self_attn(
        self,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get Self-Attention KV for a layer."""
        if self.hybrid_cache is not None:
            return self.hybrid_cache.get_layer_kv(layer_idx)
        return self.self_attn_cache[layer_idx]
    
    # =========================================================================
    # Partial Attention Cache Methods
    # =========================================================================
    
    def should_compute_partial_kv(self, layer_idx: int) -> bool:
        """
        Check if this layer should compute its Partial Attention KV.
        
        Returns False if the KV can be obtained from cross-layer sharing.
        """
        if self.cross_layer_manager is not None:
            return self.cross_layer_manager.should_compute_kv(layer_idx)
        # Without sharing, always compute (unless cached)
        return self.partial_attn_cache[layer_idx] is None
    
    def update_partial_attn(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Update Partial Attention KV cache for a layer."""
        if self.cross_layer_manager is not None:
            # Store for cross-layer sharing
            self.cross_layer_manager.store_shared_kv(layer_idx, keys, values)
        else:
            # Standard per-layer cache
            self.partial_attn_cache[layer_idx] = (keys, values)
    
    def get_partial_attn(
        self,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get Partial Attention KV for a layer."""
        if self.cross_layer_manager is not None:
            return self.cross_layer_manager.get_shared_kv(layer_idx)
        return self.partial_attn_cache[layer_idx]
    
    # =========================================================================
    # Turn Management (for Hybrid Granularity)
    # =========================================================================
    
    def set_system_prompt_end(self, position: int):
        """Mark where system prompt ends."""
        if self.hybrid_cache is not None:
            self.hybrid_cache.set_system_prompt_end(position)
    
    def mark_turn_boundary(self, position: int):
        """Mark start of a new conversation turn."""
        if self.hybrid_cache is not None:
            self.hybrid_cache.mark_turn_boundary(position)
            self.hybrid_cache.maybe_quantize_old_turns()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def clear(self):
        """Clear all caches."""
        if self.cross_layer_manager is not None:
            self.cross_layer_manager.clear()
        if self.hybrid_cache is not None:
            self.hybrid_cache.clear()
        
        for i in range(len(self.self_attn_cache)):
            self.self_attn_cache[i] = None
            self.partial_attn_cache[i] = None
    
    def get_past_key_values(self) -> List[Tuple]:
        """
        Get past_key_values in the format expected by PALMModel.
        
        Returns: List of ((self_attn_k, self_attn_v), (partial_attn_k, partial_attn_v))
        """
        past_key_values = []
        for layer_idx in range(self.config.num_hidden_layers):
            self_attn_kv = self.get_self_attn(layer_idx)
            partial_attn_kv = self.get_partial_attn(layer_idx)
            past_key_values.append((self_attn_kv, partial_attn_kv))
        return past_key_values
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get combined memory statistics."""
        stats = {
            "cross_layer_sharing_enabled": self.cross_layer_manager is not None,
            "quantization_enabled": self.hybrid_cache is not None,
        }
        
        if self.cross_layer_manager is not None:
            stats["cross_layer"] = self.cross_layer_manager.get_memory_stats()
        
        if self.hybrid_cache is not None:
            stats["hybrid_granularity"] = self.hybrid_cache.get_memory_stats()
        
        # Standard cache memory
        standard_bytes = 0
        for cache in [self.self_attn_cache, self.partial_attn_cache]:
            for kv in cache:
                if kv is not None:
                    k, v = kv
                    standard_bytes += k.numel() * 2 + v.numel() * 2  # bfloat16
        
        stats["standard_cache_mb"] = standard_bytes / (1024 * 1024)
        
        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_palm_cache(
    config,
    enable_cross_layer_sharing: bool = None,
    num_sharing_groups: int = None,
    enable_quantization: bool = None,
    quantize_after_turns: int = None,
    device: torch.device = None,
) -> PALMCache:
    """
    Factory function to create a PALMCache with specified optimizations.
    
    Args:
        config: PALMConfig instance
        enable_cross_layer_sharing: Override config's share_partial_kv
        num_sharing_groups: Override config's kv_sharing_groups
        enable_quantization: Override config's enable_kv_quantization
        quantize_after_turns: Override config's quantize_after_turns
        device: Torch device
    
    Returns:
        Configured PALMCache instance
    """
    kv_config = KVCacheConfig.from_palm_config(config)
    
    # Apply overrides
    if enable_cross_layer_sharing is not None:
        kv_config.share_partial_kv = enable_cross_layer_sharing
    if num_sharing_groups is not None:
        kv_config.kv_sharing_groups = num_sharing_groups
    if enable_quantization is not None:
        kv_config.enable_kv_quantization = enable_quantization
    if quantize_after_turns is not None:
        kv_config.quantize_after_turns = quantize_after_turns
    
    return PALMCache(kv_config, device=device)

