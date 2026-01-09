import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    Used by Llama, Mistral, Qwen, and other modern LLMs.
    Faster than LayerNorm (no mean subtraction) with similar stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then back
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads to match query heads for GQA.
    
    Args:
        hidden_states: (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each KV head
    
    Returns:
        Tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


# ROTARY POSITION EMBEDDINGS (RoPE)
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    RoPE encodes positional information by rotating query and key vectors
    in a way that their dot product depends on relative position.
    
    This is the standard implementation used by Llama, Qwen, Mistral, etc.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
        rope_type: str = "default",
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type
        
        # Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0
    
    @torch.no_grad()
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cached cos/sin values if sequence length exceeds cache."""
        if seq_len <= self._cached_seq_len and self._cos_cached is not None:
            return

        # IMPORTANT:
        # Cache only what we need (or modestly more), NOT max_position_embeddings up front.
        # Caching to max_position_embeddings would allocate huge buffers per layer
        # (and PALM has two RoPE modules per layer), causing OOM for long-context configs.
        #
        # Performance note:
        # During incremental decoding, max position often increases by 1 each step.
        # If we resized the cache to exactly seq_len every time, we'd reallocate and
        # recompute O(n) trig tables per step (O(n^2) overall). To avoid that, we grow
        # the cache exponentially (doubling) so resizes happen only O(log n) times.
        target_len = int(seq_len)
        if self._cached_seq_len <= 0:
            new_len = target_len
        else:
            new_len = self._cached_seq_len
            while new_len < target_len:
                new_len *= 2

        # Respect configured maximum (still no *pre*-allocation to it).
        if self.max_position_embeddings is not None:
            new_len = min(new_len, int(self.max_position_embeddings))

        self._cached_seq_len = new_len
        
        # Create position indices
        t = torch.arange(self._cached_seq_len, device=device, dtype=torch.float32)
        
        # Apply scaling if needed
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor
        
        # Compute frequencies: outer product of positions and inverse frequencies
        # Shape: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq.to(device))
        
        # Create [cos, sin] for each position
        # Shape: [seq_len, dim] (interleaved or concatenated depending on implementation)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin values for the given positions.
        
        Args:
            x: Input tensor (for dtype/device reference)
            position_ids: Position indices [batch_size, seq_len]
        
        Returns:
            Tuple of (cos, sin) tensors of shape [batch_size, seq_len, dim]
        """
        seq_len = position_ids.max().item() + 1
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        # Gather cos/sin values for the specific positions
        # position_ids: [batch_size, seq_len] -> gather from cached values
        cos = self._cos_cached[position_ids]  # [batch_size, seq_len, dim]
        sin = self._sin_cached[position_ids]  # [batch_size, seq_len, dim]
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine part of rotary embedding (batch, seq_len, head_dim)
        sin: Sine part of rotary embedding (batch, seq_len, head_dim)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting with heads
    
    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Add head dimension for broadcasting: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply rotation: x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# PALM ATTENTION WITH ROPE AND SDPA
class PALMAttention(nn.Module):
    """
    Self-Attention with Group Query Attention (GQA), RoPE, and SDPA support.
    
    Key features:
    - GQA reduces KV heads while keeping all query heads
    - RoPE applied to Q and K before dot product (preserves pretrained positional understanding)
    - SDPA for efficient attention computation (FlashAttention when available)
    - Pre-norm architecture: normalization happens OUTSIDE this module
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads  # Query heads
        self.num_kv_heads = getattr(config, 'num_kv_heads', config.num_attention_heads)  # KV heads (GQA)
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads  # How many times to repeat KV
        
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # Size per head
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # Total Q size
        self.kv_head_size = self.num_kv_heads * self.attention_head_size  # Total KV size (smaller with GQA)
        
        # Query projection: full num_attention_heads
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        # Key/Value projections: reduced num_kv_heads (GQA)
        self.key = nn.Linear(config.hidden_size, self.kv_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.kv_head_size, bias=False)

        # Initialize attention projection with proper scaling
        nn.init.normal_(self.query.weight, mean=0.0, std=0.02/math.sqrt(2 * config.num_hidden_layers))
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02/math.sqrt(2 * config.num_hidden_layers))
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02/math.sqrt(2 * config.num_hidden_layers))

        # Dropout probability for attention
        self.attention_dropout = config.attention_probs_dropout_prob
        
        # Output projection
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Scale factor for attention scores
        self.attention_scale = math.sqrt(self.attention_head_size)
        
        # RoPE: Rotary Position Embeddings
        rope_base = getattr(config, 'rope_theta', 10000.0)
        rope_scaling = getattr(config, 'rope_scaling', None)
        scaling_factor = 1.0
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            scaling_factor = rope_scaling.get('factor', 1.0)
        
        self.rotary_emb = RotaryEmbedding(
            dim=self.attention_head_size,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_base,
            scaling_factor=scaling_factor,
        )

    def transpose_for_scores(self, x, num_heads):
        """Reshape for multi-head attention: (batch, seq, hidden) -> (batch, heads, seq, head_dim)"""
        new_x_shape = x.size()[:-1] + (num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size) - ALREADY NORMALIZED
            attention_mask: Attention mask in additive form (0=attend, -10000=mask)
            position_ids: Position indices for RoPE [batch_size, seq_len]
                         Supports SPE: reset to 0 at target start for separate positional encoding
            past_key_value: Tuple of (past_key, past_value) for KV caching
            use_cache: Whether to return key/value for caching
        
        Returns:
            attention_output: Output tensor (WITHOUT residual connection - caller adds it)
            present_key_value: Tuple of (key, value) if use_cache=True, else None
        """
        try:
            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            
            logger.debug(f"Hidden states shape: {hidden_states.shape}")
            
            # Create default position_ids if not provided (0, 1, 2, ...)
            if position_ids is None:
                if past_key_value is not None:
                    # Incremental decoding: position is past_length
                    past_length = past_key_value[0].shape[2]
                    position_ids = torch.arange(past_length, past_length + seq_len, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                else:
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Compute Q (full heads), K, V (potentially fewer heads with GQA) for current input
            query_layer = self.transpose_for_scores(self.query(hidden_states), self.num_attention_heads)
            key_layer = self.transpose_for_scores(self.key(hidden_states), self.num_kv_heads)
            value_layer = self.transpose_for_scores(self.value(hidden_states), self.num_kv_heads)
            
            # Apply RoPE to query and key
            cos, sin = self.rotary_emb(query_layer, position_ids)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)
    
            # If we have cached key/values, concatenate with current
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key_layer = torch.cat([past_key, key_layer], dim=2)
                value_layer = torch.cat([past_value, value_layer], dim=2)
            
            # Store for caching if requested (cache the smaller KV)
            present_key_value = (key_layer, value_layer) if use_cache else None
    
            logger.debug(f"Query layer shape: {query_layer.shape}")
            logger.debug(f"Key layer shape: {key_layer.shape}")
            logger.debug(f"Value layer shape: {value_layer.shape}")
    
            # GQA: Repeat KV heads to match query heads
            key_layer_expanded = repeat_kv(key_layer, self.num_kv_groups)
            value_layer_expanded = repeat_kv(value_layer, self.num_kv_groups)
            
            # Use SDPA (Scaled Dot-Product Attention) for efficient computation
            # SDPA automatically selects FlashAttention, Memory-Efficient, or Math backend
            dropout_p = self.attention_dropout if self.training else 0.0
            
            # Convert additive mask to SDPA format if provided
            # SDPA expects: True = mask out, or additive float mask
            if attention_mask is not None:
                # Our mask is additive (-10000 for masked, 0 for attend)
                # SDPA can handle this directly as an additive bias
                if attention_mask.dim() == 4:
                    # Already in correct shape [batch, 1, seq, seq] or [batch, heads, seq, seq]
                    attn_mask = attention_mask
                else:
                    attn_mask = attention_mask.unsqueeze(1)
            else:
                attn_mask = None
            
            # SDPA call - handles scaling internally
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer_expanded,
                value_layer_expanded,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,  # We handle causality via our custom mask
                scale=1.0 / self.attention_scale,  # SDPA uses scale directly, not sqrt
            )
            
            logger.debug(f"Context layer shape after SDPA: {context_layer.shape}")
            
            # Reshape: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
    
            logger.debug(f"Context layer shape after reshaping: {context_layer.shape}")
    
            # Output projection (NO residual connection here - Pre-Norm does it outside)
            attention_output = self.dense(context_layer)
            attention_output = self.resid_dropout(attention_output)
    
            logger.debug(f"Attention output shape: {attention_output.shape}")

            # Guard for any non-finite
            if not torch.isfinite(attention_output).all():
                attention_output = torch.where(
                    torch.isfinite(attention_output),
                    attention_output,
                    torch.zeros_like(attention_output)
                )
                logger.warning("Non-finite values detected in attention output, zeroing them")

            return attention_output, present_key_value
    
        except Exception as e:
            logger.error(f"Error in PALMAttention forward pass: {str(e)}")
            logger.error(f"Input shapes - hidden_states: {hidden_states.shape} "
                         f"attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}")
            raise


class PALMPartialAttention(nn.Module):
    """
    Partial Attention with Group Query Attention (GQA), RoPE, and SDPA support.
    
    Key features:
    - GQA reduces KV heads while keeping all query heads
    - RoPE applied to Q and K for consistent positional encoding with main attention
    - SDPA for efficient attention computation
    - Attends ONLY to source tokens (fixes attention degeneration problem)
    - Pre-norm architecture: normalization happens OUTSIDE this module
    
    When num_kv_heads < num_attention_heads:
    - Queries: num_attention_heads heads (full expressiveness)
    - Keys/Values: num_kv_heads heads (repeated to match queries)
    - KV cache shrinks by factor of (num_attention_heads / num_kv_heads)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads  # Query heads
        self.num_kv_heads = getattr(config, 'num_kv_heads', config.num_attention_heads)  # KV heads (GQA)
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads  # How many times to repeat KV
        
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # Size per head
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # Total Q size
        self.kv_head_size = self.num_kv_heads * self.attention_head_size  # Total KV size (smaller with GQA)

        # Query projection: full num_attention_heads
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        # Key/Value projections: reduced num_kv_heads (GQA)
        self.key = nn.Linear(config.hidden_size, self.kv_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.kv_head_size, bias=False)

        # Dropout probability for attention
        self.attention_dropout = config.attention_probs_dropout_prob
        
        # Output projection
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # MLP applied to the source states (Fp network from paper)
        # Paper uses Pl = Pl2 + Pl1 (residual connection after two linear layers)
        # Using SiLU for consistency with modern architectures
        self.Fp_linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.Fp_activation = nn.SiLU()  # Modern activation (matches Llama)
        self.Fp_dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.Fp_linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.Fp_dropout2 = nn.Dropout(config.hidden_dropout_prob)
        
        # Scale factor for attention scores
        self.attention_scale = math.sqrt(self.attention_head_size)
        
        # RoPE for partial attention (uses source positions only)
        rope_base = getattr(config, 'rope_theta', 10000.0)
        rope_scaling = getattr(config, 'rope_scaling', None)
        scaling_factor = 1.0
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            scaling_factor = rope_scaling.get('factor', 1.0)
        
        self.rotary_emb = RotaryEmbedding(
            dim=self.attention_head_size,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_base,
            scaling_factor=scaling_factor,
        )

    def transpose_for_scores(self, x, num_heads):
        """Reshape for multi-head attention: (batch, seq, hidden) -> (batch, heads, seq, head_dim)"""
        new_x_shape = x.size()[:-1] + (num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        source_position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size) - ALREADY NORMALIZED
            source_states: Source tensor of shape (batch, source_len, hidden_size)
            attention_mask: Not used for partial attention (all positions attend to all source)
            position_ids: Position indices for query RoPE [batch_size, seq_len]
            source_position_ids: Position indices for source RoPE [batch_size, source_len]
            past_key_value: Tuple of (past_key, past_value) from source - can be reused
            use_cache: Whether to return key/value for caching
            shared_kv: Optional pre-computed (key, value) from cross-layer sharing.
                       When provided, skips Fp network computation and uses these KVs directly.
                       Used for cross-layer KV sharing optimization (Strategy #3).
        
        Returns:
            attention_output: Output tensor (WITHOUT residual connection - caller adds it)
            present_key_value: Tuple of (key, value) if use_cache=True, else None
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Create default position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Determine source of K/V: shared_kv > past_key_value > compute from source_states
        # Priority order for cross-layer sharing optimization:
        # 1. shared_kv: Pre-computed from representative layer (cross-layer sharing)
        # 2. past_key_value: Cached from previous forward pass (standard caching)
        # 3. Compute: Run Fp network on source_states (first pass, no sharing)
        
        computed_kv = False  # Track if we computed KV (for sharing with other layers)
        
        if shared_kv is not None:
            # Cross-layer sharing: use KV from representative layer
            key_layer, value_layer = shared_kv
        elif past_key_value is not None:
            # Reuse cached source K/V (already in num_kv_heads format)
            key_layer, value_layer = past_key_value
        else:
            if source_states is None:
                raise ValueError("source_states required when past_key_value is None and shared_kv is None")
            
            source_len = source_states.shape[1]
            
            # Create source position IDs if not provided (0, 1, 2, ..., source_len-1)
            if source_position_ids is None:
                source_position_ids = torch.arange(source_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Compute K/V from source states through Fp network
            # Paper: Pl = Fp(source) with residual connection: Pl = Pl2 + Pl1
            Pl1 = self.Fp_linear1(source_states)
            Pl1 = self.Fp_activation(Pl1)
            Pl1 = self.Fp_dropout1(Pl1)
            Pl2 = self.Fp_linear2(Pl1)
            Pl2 = self.Fp_dropout2(Pl2)
            P = Pl2 + Pl1  # Residual connection as per paper
            
            # Clamp Fp output to prevent extreme values during warmup
            # This prevents attention score overflow while Fp is learning
            P = torch.clamp(P, min=-100.0, max=100.0)
            
            # Project to num_kv_heads (not num_attention_heads) for GQA
            key_layer = self.transpose_for_scores(self.key(P), self.num_kv_heads)
            value_layer = self.transpose_for_scores(self.value(P), self.num_kv_heads)
            
            # Apply RoPE to key (source positions: 0, 1, 2, ..., source_len-1)
            cos_k, sin_k = self.rotary_emb(key_layer, source_position_ids)
            # Only rotate keys, we'll rotate queries separately with their positions
            key_layer = (key_layer * cos_k.unsqueeze(1)) + (rotate_half(key_layer) * sin_k.unsqueeze(1))
            
            computed_kv = True  # Mark that we computed fresh KV
        
        # Store for caching if requested (cache the smaller KV)
        # Only cache if we computed it ourselves (not from shared_kv)
        # Note: We return KV even when use_cache=False if we computed fresh KV,
        # because the caller may need it for cross-layer sharing
        present_key_value = (key_layer, value_layer) if (use_cache or computed_kv) and not shared_kv else None
    
        # Compute query matrix from full hidden states (always recomputed, full heads)
        query_layer = self.transpose_for_scores(self.query(hidden_states), self.num_attention_heads)
        
        # Apply RoPE to query with full sequence positions
        cos_q, sin_q = self.rotary_emb(query_layer, position_ids)
        query_layer = (query_layer * cos_q.unsqueeze(1)) + (rotate_half(query_layer) * sin_q.unsqueeze(1))

        # GQA: Repeat KV heads to match query heads
        key_layer_expanded = repeat_kv(key_layer, self.num_kv_groups)
        value_layer_expanded = repeat_kv(value_layer, self.num_kv_groups)

        # Use SDPA for efficient attention computation
        # NOTE: Partial attention does NOT use any mask - all tokens attend to ALL source tokens
        dropout_p = self.attention_dropout if self.training else 0.0
        
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer_expanded,
            value_layer_expanded,
            attn_mask=None,  # No mask for partial attention
            dropout_p=dropout_p,
            is_causal=False,
            scale=1.0 / self.attention_scale,
        )
        
        # Reshape: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
    
        # Output projection (NO residual connection here - Pre-Norm does it outside)
        attention_output = self.dense(context_layer)
        attention_output = self.resid_dropout(attention_output)
    
        # Guard for non-finite values (can occur during warmup with fresh Fp weights)
        if not torch.isfinite(attention_output).all():
            attention_output = torch.where(
                torch.isfinite(attention_output),
                attention_output,
                torch.zeros_like(attention_output)
            )
            logger.warning("Non-finite values detected in partial attention output, zeroing them")
    
        return attention_output, present_key_value
