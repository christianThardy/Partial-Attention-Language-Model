import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Logger object
logger = logging.getLogger(__name__)
# Set the level of the logger. Possible values: DEBUG, INFO, WARNING, ERROR, CRITICAL
# logger.setLevel(logging.DEBUG)
# # Handler that writes log messages to the notebook's output
# handler = logging.StreamHandler()
# # Set the format for the log messages
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# # Add the handler to the logger
# logger.addHandler(handler)
logger.setLevel(logging.WARNING)  # Changed from DEBUG to reduce overhead during training


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


    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask in additive form (0=attend, -10000=mask)
            past_key_value: Tuple of (past_key, past_value) for KV caching
            use_cache: Whether to return key/value for caching
        
        Returns:
            attention_output: Output tensor
            present_key_value: Tuple of (key, value) if use_cache=True, else None
        """
        try:
            logger.debug(f"Hidden states shape: {hidden_states.shape}")
            
            # Compute Q, K, V for current input
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
    
            # If we have cached key/values, concatenate with current
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key_layer = torch.cat([past_key, key_layer], dim=2)
                value_layer = torch.cat([past_value, value_layer], dim=2)
            
            # Store for caching if requested
            present_key_value = (key_layer, value_layer) if use_cache else None
    
            logger.debug(f"Query layer shape: {query_layer.shape}")
            logger.debug(f"Key layer shape: {key_layer.shape}")
            logger.debug(f"Value layer shape: {value_layer.shape}")
    
            # Calculate attention scores
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    
            logger.debug(f"Attention scores shape: {attention_scores.shape}")
    
            if attention_mask is not None:
                logger.debug(f"Original attention mask shape: {attention_mask.shape}")
                
                # Only unsqueeze if attention_mask is 2D. If it's already 4D, skip it
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                elif attention_mask.dim() != 4:
                    raise ValueError(f"attention_mask must be 2D or 4D, got {attention_mask.dim()}D shape.")
                
                # Mask is already in additive form (0 = attend, -10000 = mask out)
                attention_mask = attention_mask.to(dtype=torch.float32)
    
                logger.debug(f"Reshaped attention mask shape: {attention_mask.shape}")
    
                attention_scores = attention_scores + attention_mask
    
            # Compute attention probabilities
            attention_probs = F.softmax(attention_scores, dim=-1)
            logger.debug(f"Attention probs shape: {attention_probs.shape}")
    
            attention_probs = self.dropout(attention_probs)

            # Compute context layer
            context_layer = torch.matmul(attention_probs, value_layer)
            
            logger.debug(f"Context layer shape before permute: {context_layer.shape}")
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            logger.debug(f"Context layer shape after permute: {context_layer.shape}")
    
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
    
            logger.debug(f"Context layer shape after reshaping: {context_layer.shape}")
    
            # Output projection and residual connection
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
    
            return attention_output, present_key_value
    
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

    def forward(self, hidden_states, source_states, attention_mask=None, past_key_value=None, use_cache=False):
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            source_states: Source tensor of shape (batch, source_len, hidden_size)
            attention_mask: Not used for partial attention (all positions attend to all source)
            past_key_value: Tuple of (past_key, past_value) from source - can be reused
            use_cache: Whether to return key/value for caching
        
        Returns:
            attention_output: Output tensor
            present_key_value: Tuple of (key, value) if use_cache=True, else None
        """
        # For partial attention, source K/V can be cached and reused since source doesn't change
        if past_key_value is not None:
            # Reuse cached source K/V
            key_layer, value_layer = past_key_value
        else:
            # Compute K/V from source states
            P = self.Fp(source_states)
            key_layer = self.transpose_for_scores(self.key(P))
            value_layer = self.transpose_for_scores(self.value(P))
        
        # Store for caching if requested
        present_key_value = (key_layer, value_layer) if use_cache else None
    
        # Compute query matrix from full hidden states (always recomputed)
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Calculate attention scores: (batch, heads, seq_len, source_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    
        # NOTE: Partial attention does NOT use the causal/bidirectional mask
        # All tokens can attend to ALL source tokens - this is the key mechanism
    
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
    
        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        
        if context_layer.dim() != 4:
            context_layer = context_layer.view(hidden_states.size(0), -1, self.num_attention_heads, self.attention_head_size)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
    
        # Output projection and residual connection
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)
    
        return attention_output, present_key_value
