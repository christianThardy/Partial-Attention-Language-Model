import torch
import torch.nn as nn

class PALMEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding layer for word tokens
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id
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

    def forward(self, input_ids, source_len=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension if input is a single sequence
        batch_size, seq_length = input_ids.shape # Get sequence length of the input
        
        # Separate Positional Encoding (SPE)
        # Generate position IDs ranging from 0 to seq_length-1
        word_embeddings = self.word_embeddings(input_ids)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)

        if source_len is not None:
            # if source_len is int, use min(...); if Tensor, clamp it
            if isinstance(source_len, int):
                source_len_clamped = min(source_len, seq_length)
            else:
                source_len_clamped = torch.clamp(source_len, max=seq_length)
            
            # Create a broadcasted range [0..seq_length-1] repeated batch_size times
            range_tensor = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
            range_tensor = range_tensor.expand(batch_size, seq_length)
            
            language_mask = (range_tensor >= source_len_clamped.unsqueeze(1)) if isinstance(source_len_clamped, torch.Tensor) else (range_tensor >= source_len_clamped)
            language_ids = language_mask.long()
        else:
            language_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=input_ids.device)

        # Get embeddings for the language (source/target)
        language_embeddings = self.language_embeddings(language_ids)

        # Sum word, position, and language embeddings to form the final embedding representation
        embeddings = word_embeddings + position_embeddings + language_embeddings
        
        # Apply layer normalization to the combined embeddings
        embeddings = self.LayerNorm(embeddings)

        # Apply dropout for regularization
        embeddings = self.dropout(embeddings)

        return embeddings # Final embedding tensor
    
