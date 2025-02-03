import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Logger object
logger = logging.getLogger()
# Set the level of the logger. Possible values: DEBUG, INFO, WARNING, ERROR, CRITICAL
logger.setLevel(logging.DEBUG)
# Handler that writes log messages to the notebook's output
handler = logging.StreamHandler()
# Set the format for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(handler)


class PALMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads # Number of attention heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # Size per attention head
        self.all_head_size = self.num_attention_heads * self.attention_head_size # Total size for all attention heads

        # Pre-attention layer norm
        self.pre_attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Linear layers to project hidden states into query, key, and value representations
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Initialize attention projection with proper scaling
        nn.init.normal_(self.query.weight, mean=0.0, std=0.02/math.sqrt(2 * config.num_hidden_layers))
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02/math.sqrt(2 * config.num_hidden_layers))
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02/math.sqrt(2 * config.num_hidden_layers))

        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # Linear layer for output of the attention mechanism
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization for stability and improved training
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Scale factor for attention scores
        self.attention_scale = math.sqrt(self.attention_head_size)

    def transpose_for_scores(self, x):
        # Reshape input tensor for multi-head attention and permute dimensions
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # Permute dimensions to (batch, heads, seq_len, head_size)

    def forward(self, hidden_states, attention_mask=None, use_cache=False, past=None):
        try:
            logger.debug(f"Hidden states shape: {hidden_states.shape}") # Log shape of hidden states
            
            # Pre-norm
            normed_hidden_states = self.pre_attention_norm(hidden_states) # Apply normalization before attention
            
            # Project Q, K, V
            query_layer = self.transpose_for_scores(self.query(hidden_states)) # Compute query matrix

            # Apply caching
            if use_cache and past is not None:
                past_key, past_value = past
                new_key = self.transpose_for_scores(self.key(normed_hidden_states))
                new_value = self.transpose_for_scores(self.value(normed_hidden_states))
                key_layer = torch.cat([past_key, new_key], dim=2)
                value_layer = torch.cat([past_value, new_value], dim=2)
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states)) # Compute key matrix
                value_layer = self.transpose_for_scores(self.value(hidden_states)) # Compute value matrix
    
            logger.debug(f"Query layer shape: {query_layer.shape}")
            logger.debug(f"Key layer shape: {key_layer.shape}")
            logger.debug(f"Value layer shape: {value_layer.shape}")

            # Compute QKᵀ / sqrt(d_k): 
            # Attention scores between query and key layers
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / self.attention_scale
    
            logger.debug(f"Attention scores shape: {attention_scores.shape}")
    
            if attention_mask is not None:
                logger.debug(f"Original attention mask shape: {attention_mask.shape}")
                
                # Only unsqueeze if attention_mask is 2D. If it's already 4D, skip it
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                elif attention_mask.dim() != 4:
                    raise ValueError(f"attention_mask must be 2D or 4D, got {attention_mask.dim()}D shape.")
                
                # Convert mask to the same dtype as hidden_states
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
                
                # Convert 1-> keep (add 0), 0-> mask out (add large negative)
                attention_mask = (1.0 - attention_mask) * -1e4
                attention_scores = attention_scores + attention_mask
    
                logger.debug(f"Reshaped attention mask shape: {attention_mask.shape}")
    
            # Stable softmax in fp32
            device_type_for_autocast = hidden_states.device.type
            with torch.amp.autocast(device_type=device_type_for_autocast, enabled=False):
                attention_probs = F.softmax(attention_scores.float(), dim=-1)
                attention_probs = attention_probs.to(hidden_states.dtype)
                logger.debug(f"Attention probs shape: {attention_probs.shape}")

            if self.training:
                attention_probs = self.dropout(attention_probs)

            # Compute context layer by applying attention to the value layer
            # Weighted sum over value vectors
            context_layer = torch.matmul(attention_probs, value_layer)
            
            logger.debug(f"Context layer shape before permute: {context_layer.shape}")

            # Permute dimensions back
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            
            logger.debug(f"Context layer shape after permute: {context_layer.shape}")
    
            # Reshape context layer to combine attention heads
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
    
            logger.debug(f"Context layer shape after reshaping: {context_layer.shape}")
    
            # Output projection + residual + layernorm
            attention_output = self.dense(context_layer)
            attention_output = self.dropout(attention_output)
            attention_output = self.LayerNorm(attention_output + hidden_states)
    
            logger.debug(f"Attention output shape: {attention_output.shape}")

            # Guard for any non-finite
            if not torch.isfinite(attention_output).all():
                attention_output = torch.where(
                    torch.isfinite(attention_output),
                    attention_output,
                    torch.zeros_like(attention_output)
                )
                logger.warning("Non-finite values detected in attention output, zeroing them")

            # KV caching parameter to return both the output and the new cached states
            if use_cache:
                new_past = (key_layer, value_layer)
                return attention_output, new_past
    
            return attention_output
    
        except Exception as e:
            logger.error(f"Error in PALMAttention forward pass: {str(e)}")
            logger.error(f"Input shapes - hidden_states: {hidden_states.shape} "
                         f"attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}")
            raise


class PALMPartialAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads # Number of attention heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # Size per attention head
        self.all_head_size = self.num_attention_heads * self.attention_head_size # Total size for all attention heads

        # Linear layers to project hidden states into query, key, and value representations
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob) # Dropout layer for regularization
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # Linear layer for output of the attention mechanism
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # Layer normalization for stability and improved training

        # MLP applied to the source states
        self.Fp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def transpose_for_scores(self, x):
        # Reshape input tensor for multi-head attention and permute dimensions, (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # Permute dimensions to (batch, heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, source_states, source_len, attention_mask=None):
        """
        hidden_states: (B, T, H)
        source_states: (B, T, H)  - same shape as hidden_states
        source_len:    (B,)       - each example can have a different source boundary
        attention_mask: optional (B, 1, T, T) or similar
        """
        # Transform entire source_states (B,T,H) => Fp => P
        P = self.Fp(source_states)

        # Build a mask that is 1.0 where position < source_len[i], else 0.0
        B, T, H = P.shape
        rng = torch.arange(T, device=P.device).unsqueeze(0) # shape (1, T)
        source_mask = (rng < source_len.unsqueeze(1))       # shape (B, T) of bools

        # Zero out tokens beyond each example's source boundary via simple elementwise multiply
        source_mask_3d = source_mask.unsqueeze(-1).to(P.dtype) # (B, T, 1)
        P = P * source_mask_3d  # beyond source_len => zeroed

        if use_cache and past is not None:
            past_key, past_value = past
            new_key = self.transpose_for_scores(self.key(P))
            new_value = self.transpose_for_scores(self.value(P))
            key_layer = torch.cat([past_key, new_key], dim=2)
            value_layer = torch.cat([past_value, new_value], dim=2)
        else:
            # Create K,V from masked P, but Q is from the entire hidden_states
            key_layer = self.transpose_for_scores(self.key(P)) # Compute key matrix from the transformed source states
            value_layer = self.transpose_for_scores(self.value(P)) # Compute value matrix from the transformed source states

        query_layer = self.transpose_for_scores(self.query(hidden_states)) # Compute query matrix
        
        # Compute attention scores: shape (B, nHeads, T, T)
        d_k = self.attention_head_size
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(d_k)

        # Combine partial mask with the attention_mask:
        # Turn 'source_mask' into shape (B,1,1,T) to exclude K beyond source_len
        partial_mask_4d = source_mask.unsqueeze(1).unsqueeze(2).to(attention_scores.dtype) 
        attention_scores += (1.0 - partial_mask_4d) * -1e4 # Multiply to block out positions where partial_mask=0

        # If we have a standard mask, broadcast it similarly:
        if attention_mask is not None:
            # Ensure attention_mask has the correct shape (4D tensor)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() != 4:
                raise ValueError(f"attention_mask must be 2D or 4D, got {attention_mask.dim()}D shape.")
                
            attention_mask = attention_mask.to(attention_scores.dtype)
            # Convert attention mask to float and scale it to large negative values where mask is 0
            attention_mask = (1.0 - attention_mask) * -1e4
            attention_scores += attention_mask
            
        # Softmax in fp32 for stability
        device_type_for_autocast = hidden_states.device.type
        with torch.amp.autocast(device_type=device_type_for_autocast, enabled=False):
            attention_probs = F.softmax(attention_scores.float(), dim=-1)
            attention_probs = attention_probs.to(hidden_states.dtype)

        if self.training:
            # Apply dropout to the attention probabilities
            attention_probs = self.dropout(attention_probs)

        # Weighted sum -> context_layer
        # Compute context layer by applying attention to the value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # Permute dimensions back
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # Reshape the context layer

        # Pass context layer through residual and layer norm
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        # Zero out any non‐finite values
        if not torch.isfinite(attention_output).all():
            attention_output = torch.where(
                torch.isfinite(attention_output),
                attention_output,
                torch.zeros_like(attention_output)
            )
            logger.warning("Non-finite values detected in partial attention output, zeroing them")

        if use_cache:
            new_past = (key_layer, value_layer)
            return attention_output, new_past
            
        # Return the final attention output
        return attention_output
    
