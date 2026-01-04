import torch
import torch.nn as nn

class PALMEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding layer for word tokens
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # Embedding layer for positional information (e.g., position of each token in the sequence)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Embedding layer for language information (0 for source, 1 for target)
        # 2 embeddings: one for source, one for target
        self.language_embeddings = nn.Embedding(2, config.hidden_size)  # 2 for source and target
        
        # Layer normalization to stabilize and accelerate training by normalizing the input of each layer
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Fixed length for source sequence; used to distinguish between source and target positions
        self.fixed_source_length = config.fixed_source_length

    def forward(self, input_ids, position_offset=0):
        """
        Args:
            input_ids: Token IDs
            position_offset: Offset to add to position IDs (for incremental decoding)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        seq_length = input_ids.size(1)
        
        # Separate Positional Encoding (SPE)
        # Generate position IDs ranging from 0 to seq_length-1, plus offset
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) + position_offset
        
        # Adjust position_ids if positions exceed the fixed source length
        # For target positions, reset to start from 0
        if position_offset < self.fixed_source_length:
            # Some positions may be source, some may be target
            target_start = max(0, self.fixed_source_length - position_offset)
            if target_start < seq_length:
                # Positions beyond source_length get reset for SPE
                position_ids[target_start:] = torch.arange(
                    seq_length - target_start, 
                    dtype=torch.long, 
                    device=input_ids.device
                )
        else:
            # All positions are target positions
            position_ids = torch.arange(
                position_offset - self.fixed_source_length,
                position_offset - self.fixed_source_length + seq_length,
                dtype=torch.long, 
                device=input_ids.device
            )
        
        # Language IDs: 0 for source, 1 for target
        language_ids = torch.zeros_like(input_ids)
        
        # Determine which positions are target based on absolute position
        for i in range(seq_length):
            if position_offset + i >= self.fixed_source_length:
                language_ids[:, i] = 1

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
    