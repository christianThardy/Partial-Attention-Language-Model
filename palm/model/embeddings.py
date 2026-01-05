import torch
import torch.nn as nn

class PALMEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fixed_source_length = getattr(config, 'fixed_source_length', 100)
        
        # Embedding layer for word tokens
        # Ensure padding_idx is valid (within vocab bounds) or None
        padding_idx = getattr(config, 'pad_token_id', None)
        if padding_idx is not None and padding_idx >= config.vocab_size:
            padding_idx = None  # Invalid padding_idx, disable it
        
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=padding_idx
        )
        # Embedding layer for positional information (e.g., position of each token in the sequence)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Embedding layer for language information (0 for source, 1 for target)
        # 2 embeddings: one for source, one for target
        self.language_embeddings = nn.Embedding(2, config.hidden_size)  # 2 for source and target
        
        # Layer normalization to stabilize and accelerate training by normalizing the input of each layer
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, source_len=None, position_offset=0):
        """
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            source_len: Per-sample source lengths [batch_size] tensor, int, or None (falls back to fixed_source_length)
            position_offset: Offset to add to position IDs (for incremental decoding)
        
        Implements Separate Positional Encoding (SPE) from the PALM paper:
            - Source positions: 0, 1, 2, ..., source_len-1
            - Target positions: 0, 1, 2, ... (reset after source)
        
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
        
        # Create position indices [batch_size, seq_length]
        # Using Separate Positional Encoding (SPE): positions reset to 0 at target start
        pos_range = torch.arange(seq_length, dtype=torch.long, device=device)  # [seq_length]
        absolute_positions = pos_range.unsqueeze(0) + position_offset  # [1, seq_length]
        
        # Expand source_len for broadcasting: [batch_size, 1]
        source_len_expanded = source_len.unsqueeze(1)
        
        # Determine which positions are source vs target
        is_source = absolute_positions < source_len_expanded  # [batch_size, seq_length]
        
        # Source positions: just the absolute position (0, 1, 2, ...)
        # Target positions: reset to (absolute_position - source_len), i.e., (0, 1, 2, ...) after source
        position_ids = torch.where(
            is_source,
            absolute_positions.expand(batch_size, -1),  # Source: use absolute position
            absolute_positions - source_len_expanded     # Target: reset (subtract source_len)
        )
        
        # Clamp position_ids to valid embedding range
        position_ids = torch.clamp(position_ids, min=0, max=self.config.max_position_embeddings - 1)
        
        # Language IDs: 0 for source, 1 for target
        language_ids = (~is_source).long()  # [batch_size, seq_length]

        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        language_embeddings = self.language_embeddings(language_ids)

        # Sum embeddings
        embeddings = word_embeddings + position_embeddings + language_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
