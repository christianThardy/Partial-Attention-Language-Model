import os
import logging
import math
from typing import Optional, Tuple, List, Dict, Any

from .attention import PALMAttention, PALMPartialAttention, RMSNorm
from .embeddings import PALMEmbeddings
from .kv_cache import PALMCache, CrossLayerKVManager, create_palm_cache, KVCacheConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from safetensors.torch import save_file as safe_save_file
from transformers import GenerationConfig

import glob

# Logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) MLP block.
    
    Used by Llama, Mistral, PaLM 2, and other modern LLMs.
    Provides better learning capacity than standard GELU MLP for same parameter count.
    
    Architecture: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate and up projections (both to intermediate size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        
        # Down projection back to hidden size
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up, then project down
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)
        output = self.dropout(output)
        return output


class PALMLayer(nn.Module):
    """
    Single PALM transformer layer with Pre-Normalization architecture.
    
    Pre-Norm pattern (modern standard):
        x = x + Attention(Norm(x))
        x = x + PartialAttention(Norm(x))
        x = x + MLP(Norm(x))
    
    This provides better gradient flow and training stability at scale.
    
    KV Cache Optimizations:
    - Cross-layer KV sharing: Partial Attention KV can be shared across layer groups
    - Hybrid granularity: Older conversation turns can be quantized
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx  # For cross-layer KV sharing

        # Pre-normalization layers (RMSNorm for efficiency)
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.partial_attn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Attention mechanism with RoPE
        self.attention = PALMAttention(config)

        # Partial attention mechanism for handling source input sequences
        self.partial_attention = PALMPartialAttention(config)

        # SwiGLU MLP (modern standard)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        source_len: Optional[int] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        shared_partial_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask in additive form
            position_ids: Position IDs for RoPE with SPE
            source_len: Length of source sequence for extracting source from attention output
            past_key_value: Tuple of ((attn_key, attn_value), (partial_key, partial_value))
            use_cache: Whether to return key/values for caching
            shared_partial_kv: Optional pre-computed Partial Attention KV from cross-layer sharing.
                              When provided, skips Fp network computation for Partial Attention.
        
        Returns:
            layer_output: Output tensor
            present_key_value: Tuple of caches if use_cache=True, else None
            computed_partial_kv: The Partial Attention KV computed by this layer (for cross-layer sharing)
                                Returns None if using shared_kv or past_key_value
        """
        # Unpack past key values if provided
        past_attn_kv = None
        past_partial_kv = None
        if past_key_value is not None:
            past_attn_kv, past_partial_kv = past_key_value
        
        # ============ Self-Attention Block (Pre-Norm) ============
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        
        attention_output, present_attn_kv = self.attention(
            hidden_states, 
            attention_mask, 
            position_ids=position_ids,
            past_key_value=past_attn_kv, 
            use_cache=use_cache
        )
        hidden_states = residual + attention_output

        # Extract source_hidden_states from THIS LAYER's output for partial attention
        # Only needed if we're computing fresh KV (no shared_kv, no past_partial_kv)
        source_hidden_states = None
        source_position_ids = None
        needs_source = (shared_partial_kv is None and past_partial_kv is None)
        
        if needs_source:
            if source_len is None:
                source_len = self.config.fixed_source_length
            if isinstance(source_len, torch.Tensor):
                max_source_len = int(source_len.max().item())
            else:
                max_source_len = int(source_len)
            max_source_len = min(max_source_len, hidden_states.size(1))
            source_hidden_states = hidden_states[:, :max_source_len]
            
            # Source positions for partial attention: 0, 1, 2, ..., source_len-1
            if position_ids is not None:
                source_position_ids = position_ids[:, :max_source_len]
        
        # ============ Partial Attention Block (Pre-Norm) ============
        residual = hidden_states
        hidden_states = self.partial_attn_norm(hidden_states)
        
        partial_attention_output, present_partial_kv = self.partial_attention(
            hidden_states,
            source_hidden_states,
            attention_mask,
            position_ids=position_ids,
            source_position_ids=source_position_ids,
            past_key_value=past_partial_kv,
            use_cache=use_cache,
            shared_kv=shared_partial_kv,
        )
        hidden_states = residual + partial_attention_output
        
        # Track if this layer computed fresh Partial Attention KV
        # (for cross-layer sharing - representative layers compute, others reuse)
        computed_partial_kv = present_partial_kv if needs_source else None
        
        # Pack present key values
        present_key_value = None
        if use_cache:
            present_key_value = (present_attn_kv, present_partial_kv)
        
        # ============ MLP Block (Pre-Norm) ============
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states, present_key_value, computed_partial_kv
    

class PALMModel(nn.Module):
    """
    Partial Attention Language Model (PALM) with modern architecture.
    
    Key architectural features:
    - RoPE (Rotary Position Embeddings): Preserves pretrained positional understanding
    - SPE (Separate Positional Encoding): Resets positions at source→target boundary
    - Tied SAE head: sae_head shares weights with lm_head for stronger coupling
    - GQA (Grouped Query Attention): Memory-efficient attention with reduced KV heads
    - Pre-Normalization: Better gradient flow and training stability
    - RMSNorm: Faster than LayerNorm with similar stability
    - SwiGLU: Better MLP capacity than GELU
    - SDPA: Efficient attention (FlashAttention when available)
    """
    
    # HuggingFace/PEFT compatibility attributes
    _supports_cache_class = False
    supports_gradient_checkpointing = True
    _supports_sdpa = True  # Now supports SDPA!
    _supports_flash_attn_2 = True  # Via SDPA backend selection
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Gradient checkpointing for memory efficiency
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

        # Embedding layer (word + language embeddings, NO positional - RoPE handles that)
        self.embeddings = PALMEmbeddings(config)

        # Stack of transformer layers with RoPE (each layer knows its index for cross-layer sharing)
        self.layers = nn.ModuleList([
            PALMLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        
        # Cross-Layer KV Sharing Manager (Strategy #3)
        # Only enabled when share_partial_kv=True and kv_sharing_groups > 1
        self.cross_layer_kv_manager: Optional[CrossLayerKVManager] = None
        if getattr(config, 'share_partial_kv', False) and getattr(config, 'kv_sharing_groups', 1) > 1:
            self.cross_layer_kv_manager = CrossLayerKVManager(
                num_layers=config.num_hidden_layers,
                num_groups=config.kv_sharing_groups,
            )
        
        # Final RMSNorm before lm_head (required for Pre-Norm architecture)
        self.final_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # SAE head: TIED to lm_head weights for stronger SAE↔generation coupling
        # This prevents the "escape valve" problem where SAE learns through a separate path
        # Both heads now share the same learned token embeddings
        self._tie_sae_head = getattr(config, 'tie_sae_head', True)
        if self._tie_sae_head:
            # Create a reference (no separate weights - they share lm_head.weight)
            self.sae_head = None  # Will use lm_head directly
        else:
            # Fallback: separate SAE head (original behavior)
            self.sae_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight for combining SAE loss
        self.sae_weight = config.sae_weight if hasattr(config, 'sae_weight') else 0.5
        
        # Logit softcapping to prevent numerical instability (Gemma 2 style)
        self.logit_softcap = getattr(config, 'logit_softcap', 0.0)
        
        # Generation config for PEFT/HuggingFace compatibility
        self.generation_config = GenerationConfig(
            max_length=getattr(config, 'max_length', 512),
            max_new_tokens=None,
            min_length=getattr(config, 'min_length', 1),
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=getattr(config, 'pad_token_id', None),
            eos_token_id=getattr(config, 'eos_token_id', None),
            bos_token_id=getattr(config, 'bos_token_id', None),
        )
    
    def get_sae_head(self) -> nn.Linear:
        """Get the SAE head (tied to lm_head or separate)."""
        if self._tie_sae_head:
            return self.lm_head  # Use lm_head for SAE (tied weights)
        return self.sae_head

    def create_bidirectional_attention_mask(self, input_ids, source_len=None):
        """
        Create attention mask with per-sample source lengths.
        
        Mask structure per sample:
            - Source tokens (0 to source_len-1): bidirectional attention
            - Target tokens (source_len to end): causal attention + can attend to all source
        """
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        if source_len is None:
            source_len = self.config.fixed_source_length
        
        if isinstance(source_len, int):
            source_len = torch.full((batch_size,), source_len, dtype=torch.long, device=device)
        else:
            source_len = source_len.to(device)
            source_len = torch.clamp(source_len, max=seq_length)
        
        pos = torch.arange(seq_length, device=device).unsqueeze(0)
        source_len_expanded = source_len.unsqueeze(1)
        is_source = pos < source_len_expanded
        
        causal = torch.tril(torch.ones((seq_length, seq_length), device=device, dtype=torch.bool))
        causal = causal.unsqueeze(0).expand(batch_size, -1, -1)
        
        is_source_query = is_source.unsqueeze(2)
        is_source_key = is_source.unsqueeze(1)
        
        source_to_source = is_source_query & is_source_key
        is_target_query = ~is_source_query
        target_to_source = is_target_query & is_source_key
        target_to_target = is_target_query & (~is_source_key) & causal
        
        attend_mask = source_to_source | target_to_source | target_to_target
        
        mask = attend_mask.unsqueeze(1).float()
        mask = (1.0 - mask) * -10000.0
        
        return mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        source_len: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
        position_offset: Optional[int] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass with RoPE and tied SAE head.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask (2D padding or 4D full mask)
            labels: Labels for loss computation
            source_len: Per-sample source lengths [batch_size] for mask creation and SAE loss
            past_key_values: List of past key/value tuples for each layer
            use_cache: Whether to return key/values for caching
            position_offset: Position offset for incremental decoding
            inputs_embeds: Pre-computed embeddings (alternative to input_ids, used by PEFT)
            **kwargs: Additional arguments for HuggingFace compatibility
        
        Returns:
            lm_logits, combined_loss, loss, sae_loss[, past_key_values]
        """
        try:
            # Handle inputs_embeds (used by PEFT) or input_ids
            if inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
                if input_ids is None:
                    input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
                # Need to compute position_ids even with pre-computed embeddings
                position_ids = None
            else:
                input_ids = torch.tensor(input_ids) if not isinstance(input_ids, torch.Tensor) else input_ids
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                batch_size, seq_length = input_ids.shape
                device = input_ids.device
            
            # Track if source_len was explicitly provided (for SAE loss computation)
            source_len_provided = source_len is not None
            
            # Handle source_len
            if source_len is None:
                source_len = torch.full(
                    (batch_size,), 
                    min(self.config.fixed_source_length, seq_length),
                    dtype=torch.long, 
                    device=device
                )
            elif isinstance(source_len, int):
                source_len = torch.full((batch_size,), source_len, dtype=torch.long, device=device)
            else:
                source_len = source_len.to(device)
            
            # Position offset for incremental decoding
            if position_offset is None:
                position_offset = 0
            
            # Handle attention mask
            padding_mask = None
            if attention_mask is not None:
                attention_mask = torch.tensor(attention_mask) if not isinstance(attention_mask, torch.Tensor) else attention_mask
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                if attention_mask.dim() == 2:
                    padding_mask = attention_mask
                    attention_mask = None
            
            # Create the bidirectional attention mask
            if attention_mask is None:
                attention_mask = self.create_bidirectional_attention_mask(input_ids, source_len)
                
                if padding_mask is not None:
                    expanded_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                    padding_mask_additive = (1.0 - expanded_padding_mask.float()) * -10000.0
                    attention_mask = attention_mask + padding_mask_additive
    
            # Ensure labels are tensors
            if labels is not None:
                labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
    
            # Embedding lookup with position IDs for RoPE
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                # Compute position_ids for RoPE even with pre-computed embeddings
                _, position_ids = self.embeddings(input_ids, source_len=source_len, position_offset=position_offset)
            else:
                hidden_states, position_ids = self.embeddings(
                    input_ids, source_len=source_len, position_offset=position_offset
                )
            
            # Calculate max source length for layers
            max_source_len = min(source_len.max().item(), seq_length)
            max_source_len = int(max_source_len)
    
            # Initialize present key values list
            present_key_values = [] if use_cache else None
            
            # Clear cross-layer KV manager for new forward pass (not during incremental decoding)
            if self.cross_layer_kv_manager is not None and past_key_values is None:
                self.cross_layer_kv_manager.clear()
    
            # Pass through each layer with RoPE position_ids
            for i, layer in enumerate(self.layers):
                layer_past_kv = past_key_values[i] if past_key_values is not None else None
                
                # Cross-layer KV sharing: get shared KV for this layer's group
                shared_partial_kv = None
                if self.cross_layer_kv_manager is not None and past_key_values is None:
                    # During initial pass (not incremental), check if we should reuse KV
                    shared_partial_kv = self.cross_layer_kv_manager.get_shared_kv(i)
                
                # Use gradient checkpointing if enabled and training
                if self.gradient_checkpointing and self.training and not use_cache:
                    def create_custom_forward(module, src_len, pos_ids, shared_kv):
                        def custom_forward(hidden_states, attention_mask):
                            outputs = module(
                                hidden_states, 
                                attention_mask,
                                position_ids=pos_ids,
                                source_len=src_len,
                                past_key_value=None, 
                                use_cache=False,
                                shared_partial_kv=shared_kv,
                            )
                            return outputs[0]
                        return custom_forward
                    
                    hidden_states = gradient_checkpoint(
                        create_custom_forward(layer, max_source_len, position_ids, shared_partial_kv),
                        hidden_states,
                        attention_mask,
                        use_reentrant=False
                    )
                    present_kv = None
                    computed_partial_kv = None
                else:
                    hidden_states, present_kv, computed_partial_kv = layer(
                        hidden_states, 
                        attention_mask,
                        position_ids=position_ids,
                        source_len=max_source_len if past_key_values is None else None,
                        past_key_value=layer_past_kv,
                        use_cache=use_cache,
                        shared_partial_kv=shared_partial_kv,
                    )
                
                # Cross-layer KV sharing: store computed KV for other layers in the group
                if (self.cross_layer_kv_manager is not None and 
                    computed_partial_kv is not None and
                    past_key_values is None):
                    # This is a representative layer - store its KV for sharing
                    self.cross_layer_kv_manager.store_shared_kv(i, *computed_partial_kv)
                
                if use_cache:
                    present_key_values.append(present_kv)
            
            # Apply final normalization (required for Pre-Norm architecture)
            hidden_states = self.final_norm(hidden_states)
    
            # Compute logits for language modeling
            lm_logits = self.lm_head(hidden_states)
            
            # Apply logit softcapping if enabled
            if self.logit_softcap > 0:
                lm_logits = self.logit_softcap * torch.tanh(lm_logits / self.logit_softcap)

            # Initialize loss variables
            loss = None
            sae_loss = None
            combined_loss = None
            
            # Compute loss if labels are provided
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

                # Calculate SAE loss only if source_len was explicitly provided
                if source_len_provided:
                    max_source_len = min(source_len.max().item(), hidden_states.size(1))
                    max_source_len = int(max_source_len)
                    
                    # Use tied SAE head (same as lm_head)
                    sae_head = self.get_sae_head()
                    sae_logits = sae_head(hidden_states[:, :max_source_len])
                    
                    # Apply softcapping to SAE logits as well
                    if self.logit_softcap > 0:
                        sae_logits = self.logit_softcap * torch.tanh(sae_logits / self.logit_softcap)
                    
                    sae_labels = input_ids[:, :max_source_len].clone()
                    
                    # Mask out positions beyond each sample's actual source_len
                    range_tensor = torch.arange(max_source_len, device=device).unsqueeze(0)
                    source_mask = range_tensor >= source_len.unsqueeze(1)
                    sae_labels[source_mask] = -100
                    
                    sae_loss = loss_fct(sae_logits.view(-1, self.config.vocab_size), sae_labels.view(-1))
                    
                    # Combine losses
                    combined_loss = loss + self.sae_weight * sae_loss
                else:
                    combined_loss = loss
                    sae_loss = torch.tensor(0.0, device=loss.device)
            
            if use_cache:
                return lm_logits, combined_loss, loss, sae_loss, present_key_values
            return lm_logits, combined_loss, loss, sae_loss   
    
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes - input_ids: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'not a tensor'}, "
                  f"attention_mask: {attention_mask.shape if isinstance(attention_mask, torch.Tensor) else 'not a tensor'}, "
                  f"labels: {labels.shape if isinstance(labels, torch.Tensor) else 'not a tensor'},")
            raise

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text with optional KV caching."""
        max_length = max_length if max_length is not None else getattr(self.config, 'max_length', 512)
        min_length = min_length if min_length is not None else getattr(self.config, 'min_length', 1)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        device = next(self.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)

        # Store the original source length (the prompt length)
        original_source_length = input_ids.shape[1]
        
        # Temporarily override fixed_source_length
        saved_fixed_source_length = self.config.fixed_source_length
        self.config.fixed_source_length = original_source_length
        
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        generated_sequence = input_ids
        past_key_values = None

        try:
            while True:
                if use_cache and past_key_values is not None:
                    model_input_ids = generated_sequence[:, -1:]
                    position_offset = generated_sequence.shape[1] - 1
                    full_seq_len = generated_sequence.shape[1]
                    attention_mask = self._create_incremental_attention_mask(
                        full_seq_len, original_source_length, device
                    )
                else:
                    model_input_ids = generated_sequence
                    position_offset = 0
                    attention_mask = self.create_bidirectional_attention_mask(generated_sequence)

                with torch.no_grad():
                    outputs = self(
                        model_input_ids,
                        attention_mask=attention_mask,
                        source_len=torch.tensor([original_source_length], device=device),
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        position_offset=position_offset
                    )
                
                if use_cache:
                    lm_logits, _, _, _, past_key_values = outputs
                else:
                    lm_logits = outputs[0]
                
                next_token_logits = lm_logits[:, -1, :]

                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits,
                    cur_len=generated_sequence.shape[1],
                    max_length=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    input_ids=generated_sequence
                )

                next_token_logits = next_token_logits / temperature
                next_token_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
                generated_sequence = torch.cat([generated_sequence, next_tokens.unsqueeze(-1)], dim=-1)

                if unfinished_sequences.max() == 0 or generated_sequence.shape[1] >= max_length:
                    break
        finally:
            self.config.fixed_source_length = saved_fixed_source_length

        return generated_sequence
    
    def _create_incremental_attention_mask(self, full_seq_len: int, source_length: int, device: torch.device):
        """Create attention mask for incremental decoding."""
        mask = torch.zeros((1, 1, 1, full_seq_len), device=device)
        return mask
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """Prepare inputs for generation (required by PEFT/HuggingFace)."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }
    
    def adjust_logits_during_generation(self, logits, cur_len, max_length, min_length, repetition_penalty, input_ids):
        """Adjust token logits during generation."""
        if repetition_penalty != 1.0:
            unique_tokens = torch.unique(input_ids)
            logits.scatter_(
                1,
                unique_tokens.unsqueeze(0).expand(logits.size(0), -1),
                torch.where(
                    logits.gather(1, unique_tokens.unsqueeze(0).expand(logits.size(0), -1)) < 0,
                    logits.gather(1, unique_tokens.unsqueeze(0).expand(logits.size(0), -1)) * repetition_penalty,
                    logits.gather(1, unique_tokens.unsqueeze(0).expand(logits.size(0), -1)) / repetition_penalty
                )
            )
        if cur_len < min_length:
            logits[:, self.config.eos_token_id] = float('-inf')

        return logits
    
    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """Filter a distribution of logits using top-k and/or top-p filtering."""
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            topk_values, _ = torch.topk(logits, top_k)
            indices_to_remove = logits < topk_values[..., -1, None]
            logits.masked_fill_(indices_to_remove, filter_value)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits.masked_fill_(indices_to_remove, filter_value)
            
        return logits
    
    def save_pretrained(
        self, save_directory, is_main_process=True, state_dict=None, save_function=torch.save, 
        push_to_hub=False, max_shard_size="5GB", safe_serialization=True, variant=None, 
        token=None, save_peft_format=True, **kwargs
    ):
        """Save model and configuration."""
        if not is_main_process:
            return None

        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if not hasattr(self, 'config'):
            raise AttributeError("Model doesn't have a config attribute")

        state_dict = state_dict or (
            self.module.state_dict() if hasattr(self, 'module') else self.state_dict()
        )
        
        with open(os.path.join(save_directory, '.gitattributes'), 'w') as f:
            f.write('*.bin filter=lfs diff=lfs merge=lfs -text\n')
            f.write('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
        
        if safe_serialization:
            from safetensors.torch import save_file
            save_file(
                state_dict,
                os.path.join(save_directory, 'model.safetensors'),
                metadata={"format": "pt"}
            )
        else:
            save_function(
                state_dict,
                os.path.join(save_directory, 'pytorch_model.bin')
            )

        model_to_save = self.module if hasattr(self, 'module') else self
        if hasattr(model_to_save, 'config'):
            model_to_save.config.save_pretrained(save_directory)

        if hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(save_directory)
        
        if push_to_hub and hasattr(self, '_push_to_hub'):
            commit_message = kwargs.pop("commit_message", "Upload model")
            private = kwargs.pop("private", False)
            return self._push_to_hub(
                save_directory,
                commit_message=commit_message,
                private=private,
                **kwargs
            )

        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        from ..config import PALMConfig
        config = kwargs.get("config") or cls._load_config(pretrained_model_name_or_path)
        model = cls(config)
       
        state_dict = kwargs.get("state_dict") or cls._find_and_load_state_dict(
            pretrained_model_name_or_path
        )
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
           
        return model
       
    @staticmethod
    def _load_config(path):
        """Helper method for config loading."""
        from ..config import PALMConfig
        config_file = os.path.join(path, "config.json")
        if not os.path.exists(config_file):
            raise OSError(f"Config file not found in {path}")
        return PALMConfig.from_json_file(config_file)
       
    @staticmethod
    def _find_and_load_state_dict(path):
        """Helper method for state dict loading."""
        file_types = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"]
        for pattern in file_types:
            files = glob.glob(os.path.join(path, pattern))
            if files:
                if pattern == "*.safetensors":
                    from safetensors.torch import load_file
                    return load_file(files[0])
                return torch.load(files[0], map_location="cpu")
        return None
