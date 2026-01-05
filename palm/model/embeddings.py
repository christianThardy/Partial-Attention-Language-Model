import torch
import torch.nn as nn
from typing import Optional, Tuple


class PALMEmbeddings(nn.Module):
    """
    PALM Embeddings with RoPE-compatible design.
    
    Key changes from original:
    - NO positional embeddings (RoPE handles positions in attention layers)
    - Language embeddings to differentiate source vs target
    - Computes position_ids with SPE (Separate Positional Encoding) for RoPE
    
    SPE (from PALM paper):
    - Source positions: 0, 1, 2, ..., source_len-1
    - Target positions: 0, 1, 2, ... (reset after source boundary)
    
    This allows both source and target to use the full positional resolution
    and enables the model to better differentiate between source and target.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fixed_source_length = getattr(config, 'fixed_source_length', 100)
        
        # Embedding layer for word tokens
        padding_idx = getattr(config, 'pad_token_id', None)
        if padding_idx is not None and padding_idx >= config.vocab_size:
            padding_idx = None  # Invalid padding_idx, disable it
        
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=padding_idx
        )
        
        # Language embedding to differentiate source (0) vs target (1)
        # This replaces positional information for source/target discrimination
        self.language_embeddings = nn.Embedding(2, config.hidden_size)
        
        # Layer normalization to stabilize embeddings
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        source_len: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute embeddings and position IDs for RoPE.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            source_len: Per-sample source lengths [batch_size] tensor, int, or None
            position_offset: Offset for incremental decoding
        
        Returns:
            embeddings: Word + language embeddings [batch_size, seq_length, hidden_size]
            position_ids: Position IDs for RoPE with SPE [batch_size, seq_length]
            
        SPE (Separate Positional Encoding) for RoPE:
            - Source positions: 0, 1, 2, ..., source_len-1
            - Target positions: 0, 1, 2, ... (reset to 0 after source boundary)
        
        Language embeddings:
            - Source tokens (positions < source_len): language_id = 0
            - Target tokens (positions >= source_len): language_id = 1
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Handle source_len - convert to per-sample tensor
        if source_len is None:
            source_len = torch.full((batch_size,), self.fixed_source_length, dtype=torch.long, device=device)
        elif isinstance(source_len, int):
            source_len = torch.full((batch_size,), source_len, dtype=torch.long, device=device)
        else:
            source_len = source_len.to(device)
        
        # Clamp source_len to valid range
        source_len = torch.clamp(source_len, min=0, max=seq_length + position_offset)
        
        # COMPUTE POSITION IDS WITH SPE (Separate Positional Encoding)
        # Source: positions 0, 1, 2, ..., source_len-1
        # Target: positions 0, 1, 2, ... (reset after source boundary)
        
        pos_range = torch.arange(seq_length, dtype=torch.long, device=device)
        absolute_positions = pos_range.unsqueeze(0) + position_offset  # [1, seq_length]
        
        # Expand source_len for broadcasting: [batch_size, 1]
        source_len_expanded = source_len.unsqueeze(1)
        
        # Determine which positions are source vs target
        is_source = absolute_positions < source_len_expanded  # [batch_size, seq_length]
        
        # SPE: Reset position counter at target boundary
        # Source positions: keep as-is (0, 1, 2, ...)
        # Target positions: reset to (absolute_position - source_len), i.e., (0, 1, 2, ...)
        position_ids = torch.where(
            is_source,
            absolute_positions.expand(batch_size, -1),  # Source: use absolute position
            absolute_positions - source_len_expanded     # Target: reset (subtract source_len)
        )
        
        # Ensure position_ids are non-negative (can happen with position_offset during generation)
        position_ids = torch.clamp(position_ids, min=0)
        
        # COMPUTE EMBEDDINGS (word + language, NO positional)
        # Language IDs: 0 for source, 1 for target
        language_ids = (~is_source).long()  # [batch_size, seq_length]

        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        language_embeddings = self.language_embeddings(language_ids)

        # Combine embeddings (NO positional embeddings - RoPE handles this)
        embeddings = word_embeddings + language_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings, position_ids
