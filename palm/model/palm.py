
import os
import logging

from .attention import PALMAttention, PALMPartialAttention
from .embeddings import PALMEmbeddings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from safetensors.torch import save_file as safe_save_file
from transformers import GenerationConfig

import glob

# Logger object
logger = logging.getLogger(__name__)
# # Set the level of the logger. Possible values: DEBUG, INFO, WARNING, ERROR, CRITICAL
# logger.setLevel(logging.DEBUG)
# # Handler that writes log messages to the notebook's output
# handler = logging.StreamHandler()
# # Set the format for the log messages
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# # Add the handler to the logger
# logger.addHandler(handler)
logger.setLevel(logging.WARNING)  # Changed from DEBUG to reduce overhead during training


class PALMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Linear layer to project hidden states to a larger intermediate size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # Activation function (GELU) to introduce non-linearity
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        # Apply linear transformation and activation function to the hidden states
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    

class PALMOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Linear layer to project the intermediate representation back to the original hidden size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Layer normalization for stability and improved training
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # Apply linear transformation and dropout to the hidden states
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add input tensor to the transformed hidden states (residual connection) and normalize
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class PALMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Attention mechanism
        self.attention = PALMAttention(config)

        # Partial attention mechanism for handling specific input sequences
        self.partial_attention = PALMPartialAttention(config)

        # Intermediate layer for processing the attention output
        self.intermediate = PALMIntermediate(config)

        # Output layer to produce the final output for this layer
        self.output = PALMOutput(config)

    def forward(self, hidden_states, attention_mask=None, source_hidden_states=None,
                past_key_value=None, use_cache=False):
        """
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask in additive form
            source_hidden_states: Source states for partial attention
            past_key_value: Tuple of ((attn_key, attn_value), (partial_key, partial_value))
            use_cache: Whether to return key/values for caching
        
        Returns:
            layer_output: Output tensor
            present_key_value: Tuple of caches if use_cache=True, else None
        """
        # Unpack past key values if provided
        past_attn_kv = None
        past_partial_kv = None
        if past_key_value is not None:
            past_attn_kv, past_partial_kv = past_key_value
        
        # Apply attention mechanism with caching
        attention_output, present_attn_kv = self.attention(
            hidden_states, attention_mask, 
            past_key_value=past_attn_kv, use_cache=use_cache
        )

        # Apply partial attention using source hidden states
        # For incremental decoding with cache, source_hidden_states is None because K/V are cached
        if source_hidden_states is None and past_partial_kv is None:
            # Fallback: extract from hidden_states if possible
            source_hidden_states = hidden_states[:, :self.config.fixed_source_length]
        
        partial_attention_output, present_partial_kv = self.partial_attention(
            attention_output,
            source_hidden_states,  # Can be None if past_partial_kv is provided
            attention_mask,
            past_key_value=past_partial_kv,
            use_cache=use_cache
        )
        
        # Pack present key values
        present_key_value = None
        if use_cache:
            present_key_value = (present_attn_kv, present_partial_kv)
        
        # Process output of the partial attention with the intermediate layer
        intermediate_output = self.intermediate(partial_attention_output)

        # Apply output layer to produce the final output for this layer
        layer_output = self.output(intermediate_output, partial_attention_output)
        return layer_output, present_key_value
    

class PALMModel(nn.Module):
    # HuggingFace/PEFT compatibility attributes
    _supports_cache_class = False
    supports_gradient_checkpointing = True
    _supports_sdpa = False
    _supports_flash_attn_2 = False
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Gradient checkpointing for memory efficiency
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

        # Embedding layer for input tokens
        self.embeddings = PALMEmbeddings(config)

        # Stack of transformer layers
        self.layers = nn.ModuleList([PALMLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Linear layer for language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Linear layer for sequence autoencoding head
        self.sae_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight for combining SAE loss
        self.sae_weight = config.sae_weight if hasattr(config, 'sae_weight') else 0.5  # Weight for SAE loss
        
        # Generation config for PEFT/HuggingFace compatibility
        # This is required when PEFT wraps the model for generation
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

    def create_bidirectional_attention_mask(self, input_ids, source_len=None):
        """
        Create attention mask with per-sample source lengths.
        
        Args:
            input_ids: [batch_size, seq_length]
            source_len: tensor of shape [batch_size] with per-sample source lengths,
                       OR an integer for uniform source length across batch,
                       OR None to use config.fixed_source_length
        
        Returns:
            mask: [batch_size, 1, seq_length, seq_length] in additive form
                  (0 = attend, -10000 = mask)
        
        Mask structure per sample:
            - Source tokens (0 to source_len-1): bidirectional attention
            - Target tokens (source_len to end): causal attention + can attend to all source
        """
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Handle source_len argument
        if source_len is None:
            source_len = self.config.fixed_source_length
        
        # Convert source_len to tensor if it's an integer
        if isinstance(source_len, int):
            source_len = torch.full((batch_size,), source_len, dtype=torch.long, device=device)
        else:
            source_len = source_len.to(device)
            # Clamp source_len to not exceed seq_length
            source_len = torch.clamp(source_len, max=seq_length)
        
        # Create position indices [1, seq_length]
        pos = torch.arange(seq_length, device=device).unsqueeze(0)
        
        # Expand source_len to [batch_size, 1] for broadcasting
        source_len_expanded = source_len.unsqueeze(1)
        
        # Create masks for source and target regions
        # is_source[b, i] = True if position i is in source region for sample b
        is_source = pos < source_len_expanded  # [batch_size, seq_length]
        
        # Create causal mask
        causal = torch.tril(torch.ones((seq_length, seq_length), device=device, dtype=torch.bool))
        causal = causal.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Build attention pattern
        is_source_query = is_source.unsqueeze(2)  # [batch_size, seq_length, 1]
        is_source_key = is_source.unsqueeze(1)    # [batch_size, 1, seq_length]
        
        # Source tokens can attend to all source tokens (bidirectional)
        source_to_source = is_source_query & is_source_key
        
        # Target tokens can attend to source tokens
        is_target_query = ~is_source_query
        target_to_source = is_target_query & is_source_key
        
        # Target tokens use causal attention for target-to-target
        target_to_target = is_target_query & (~is_source_key) & causal
        
        # Combine all attention patterns
        attend_mask = source_to_source | target_to_source | target_to_target
        
        # Convert to additive mask format
        mask = attend_mask.unsqueeze(1).float()
        mask = (1.0 - mask) * -10000.0
        
        return mask

    def forward(self, input_ids=None, attention_mask=None, labels=None, source_len=None,
                past_key_values=None, use_cache=False, position_offset=None,
                inputs_embeds=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        """
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask (2D padding or 4D full mask)
            labels: Labels for loss computation
            source_len: Per-sample source lengths [batch_size] for mask creation and SAE loss
            past_key_values: List of past key/value tuples for each layer
            use_cache: Whether to return key/values for caching
            position_offset: Position offset for incremental decoding (for embeddings)
            inputs_embeds: Pre-computed embeddings (alternative to input_ids, used by PEFT)
            output_attentions: Not used, for HuggingFace compatibility
            output_hidden_states: Not used, for HuggingFace compatibility
            return_dict: Not used, for HuggingFace compatibility
            **kwargs: Additional arguments for compatibility
        
        Returns:
            lm_logits, combined_loss, loss, sae_loss[, past_key_values]
        """
        try:
            # Handle inputs_embeds (used by PEFT) or input_ids
            if inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
                # If input_ids not provided but we need them for labels/SAE, this will fail gracefully
                if input_ids is None:
                    input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
            else:
                # Ensure input_ids is a tensor and has the correct dimensions
                input_ids = torch.tensor(input_ids) if not isinstance(input_ids, torch.Tensor) else input_ids
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                batch_size, seq_length = input_ids.shape
                device = input_ids.device
            
            # Track if source_len was explicitly provided (for SAE loss computation)
            source_len_provided = source_len is not None
            
            # Handle source_len - default to fixed_source_length if not provided
            # (needed for attention mask creation, but SAE loss only computed if explicitly provided)
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
            
            # Determine position offset for embeddings
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
            
            # Create the bidirectional attention mask with per-sample source_len
            if attention_mask is None:
                attention_mask = self.create_bidirectional_attention_mask(input_ids, source_len)
                
                if padding_mask is not None:
                    expanded_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                    padding_mask_additive = (1.0 - expanded_padding_mask.float()) * -10000.0
                    attention_mask = attention_mask + padding_mask_additive
    
            # Ensure labels are tensors and have the correct dimensions
            if labels is not None:
                labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
    
            # Embedding lookup with position offset (or use pre-computed inputs_embeds)
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embeddings(input_ids, position_offset=position_offset)
            
            # For first forward pass (no cache), get source hidden states
            # For incremental decoding, source K/V is already cached in partial attention
            if past_key_values is None:
                # Use max source_len for source_hidden_states extraction
                max_source_len = min(source_len.max().item(), seq_length)
                source_hidden_states = hidden_states[:, :int(max_source_len)]
            else:
                # Incremental: source states not needed as K/V are cached
                source_hidden_states = None
    
            # Initialize present key values list
            present_key_values = [] if use_cache else None
    
            # Pass through each layer with caching
            for i, layer in enumerate(self.layers):
                layer_past_kv = past_key_values[i] if past_key_values is not None else None
                
                # Use gradient checkpointing if enabled and training (saves memory)
                if self.gradient_checkpointing and self.training and not use_cache:
                    # Gradient checkpointing wrapper - recomputes activations during backward
                    def create_custom_forward(module):
                        def custom_forward(hidden_states, attention_mask, source_hidden_states):
                            outputs = module(hidden_states, attention_mask, 
                                           source_hidden_states=source_hidden_states,
                                           past_key_value=None, use_cache=False)
                            return outputs[0]  # Only return hidden_states
                        return custom_forward
                    
                    hidden_states = gradient_checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                        source_hidden_states,
                        use_reentrant=False
                    )
                    present_kv = None
                else:
                    hidden_states, present_kv = layer(
                        hidden_states, 
                        attention_mask,
                        source_hidden_states=source_hidden_states,
                        past_key_value=layer_past_kv,
                        use_cache=use_cache
                    )
                
                if use_cache:
                    present_key_values.append(present_kv)
    
            # Compute logits for language modeling
            lm_logits = self.lm_head(hidden_states)

            # Initialize loss variables
            loss = None
            sae_loss = None
            combined_loss = None
            
            # Compute loss if labels are provided
            if labels is not None:
                # Use ignore_index=-100 to exclude masked tokens (prompt + padding) from loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

                # Calculate SAE loss only if source_len was explicitly provided
                if source_len_provided:
                    # Compute SAE logits only for source portion
                    max_source_len = min(source_len.max().item(), hidden_states.size(1))
                    max_source_len = int(max_source_len)
                    
                    sae_logits = self.sae_head(hidden_states[:, :max_source_len])
                    sae_labels = input_ids[:, :max_source_len].clone()
                    
                    # Mask out positions beyond each sample's actual source_len with -100
                    # Vectorized version for efficiency
                    range_tensor = torch.arange(max_source_len, device=device).unsqueeze(0)
                    source_mask = range_tensor >= source_len.unsqueeze(1)
                    sae_labels[source_mask] = -100
                    
                    sae_loss = loss_fct(sae_logits.view(-1, self.config.vocab_size), sae_labels.view(-1))
                    
                    # Combine losses
                    combined_loss = loss + self.sae_weight * sae_loss
                else:
                    # No SAE loss if source_len not provided (backward compatibility)
                    combined_loss = loss
                    sae_loss = torch.tensor(0.0, device=loss.device)
            
            # Return with or without cache
            if use_cache:
                return lm_logits, combined_loss, loss, sae_loss, present_key_values
            return lm_logits, combined_loss, loss, sae_loss   
    
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes - input_ids: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'not a tensor'}, "
                  f"attention_mask: {attention_mask.shape if isinstance(attention_mask, torch.Tensor) else 'not a tensor'}, "
                  f"labels: {labels.shape if isinstance(labels, torch.Tensor) else 'not a tensor'},")
            raise

    def generate(self, input_ids, max_length=None, min_length=None, do_sample=True, 
                 temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0, 
                 pad_token_id=None, eos_token_id=None, attention_mask=None, 
                 use_cache=True, **kwargs):
        """
        Generate text with optional KV caching for faster inference.
        
        Args:
            use_cache: If True, use KV caching for faster generation (default: True)
        """
        # Set default values for generation parameters
        max_length = max_length if max_length is not None else getattr(self.config, 'max_length', 512)
        min_length = min_length if min_length is not None else getattr(self.config, 'min_length', 1)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # Ensure input_ids are on the correct device
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        # IMPORTANT: Store the original source length (the prompt length)
        original_source_length = input_ids.shape[1]
        
        # Temporarily override fixed_source_length for this generation
        saved_fixed_source_length = self.config.fixed_source_length
        self.config.fixed_source_length = original_source_length
        
        # Initialize sequence tracking
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        generated_sequence = input_ids
        past_key_values = None

        try:
            while True:
                # Determine what to pass to forward
                if use_cache and past_key_values is not None:
                    # Only pass the last token when using cache
                    model_input_ids = generated_sequence[:, -1:]
                    # Position offset is the position of the new token
                    position_offset = generated_sequence.shape[1] - 1
                    # Create attention mask for single new token attending to full sequence
                    full_seq_len = generated_sequence.shape[1]
                    attention_mask = self._create_incremental_attention_mask(
                        full_seq_len, original_source_length, device
                    )
                else:
                    # First step or no caching: pass full sequence
                    model_input_ids = generated_sequence
                    position_offset = 0
                    attention_mask = self.create_bidirectional_attention_mask(generated_sequence)

                # Forward pass without gradients
                with torch.no_grad():
                    outputs = self(
                        model_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        position_offset=position_offset
                    )
                
                # Get logits and optionally update cache
                if use_cache:
                    lm_logits, _, _, _, past_key_values = outputs
                else:
                    lm_logits = outputs[0]
                
                # Get the next token logits (last position)
                next_token_logits = lm_logits[:, -1, :]

                # Adjust logits for generation
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits,
                    cur_len=generated_sequence.shape[1],
                    max_length=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    input_ids=generated_sequence
                )

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k and top-p filtering
                next_token_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # Handle finished sequences
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # Update unfinished sequences
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

                # Append next tokens to the sequence
                generated_sequence = torch.cat([generated_sequence, next_tokens.unsqueeze(-1)], dim=-1)

                # Stop if we've reached max_length or all sequences are finished
                if unfinished_sequences.max() == 0 or generated_sequence.shape[1] >= max_length:
                    break
        finally:
            # Restore original fixed_source_length
            self.config.fixed_source_length = saved_fixed_source_length

        return generated_sequence
    
    def _create_incremental_attention_mask(self, full_seq_len, source_length, device):
        """
        Create attention mask for a single new token attending to the full sequence.
        
        For incremental decoding, the new token can attend to:
        - All source tokens (bidirectionally)
        - All previous target tokens (causally - but they're all in the past)
        - Itself
        
        Returns:
            Attention mask of shape [1, 1, 1, full_seq_len] (0 = attend, -10000 = mask)
        """
        # The new token can attend to everything before it (including all source and generated tokens)
        # So the mask is all zeros (attend to everything)
        mask = torch.zeros((1, 1, 1, full_seq_len), device=device)
        return mask
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """
        Prepare inputs for generation (required by PEFT/HuggingFace generation utilities).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            past_key_values: Cached key/values from previous forward pass
            attention_mask: Optional attention mask
            **kwargs: Additional arguments (position_offset, use_cache, etc.)
        
        Returns:
            Dictionary with model inputs
        """
        # If past_key_values exist, only use the last token
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
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[0]):
                for previous_token in set(input_ids[i].tolist()):
                    # If score < 0 then repetition penalty has to multiply it by repetition penalty
                    if logits[i, previous_token] < 0:
                        logits[i, previous_token] *= repetition_penalty
                    else:
                        logits[i, previous_token] /= repetition_penalty

        # Prevent generation of tokens before min_length
        if cur_len < min_length:
            logits[:, self.config.eos_token_id] = float('-inf')

        return logits
    
    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """Filter a distribution of logits using top-k and/or top-p (nucleus) filtering."""
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value

        return logits
    
    def save_pretrained(self, save_directory, is_main_process=True, state_dict=None, save_function=torch.save, push_to_hub=False, max_shard_size="5GB", safe_serialization=True, variant=None, token=None, save_peft_format=True, **kwargs):
        """Save a model and its configuration file to a directory, so that it can be re-loaded using the `from_pretrained` class method."""
        if not is_main_process:
            return None

        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # Get state_dict
        if state_dict is None:
            state_dict = self.state_dict()

        # Handle the case for DataParallel
        if hasattr(self, 'module'):
            state_dict = self.module.state_dict()

        # Save model
        model_to_save = self.module if hasattr(self, 'module') else self

        # Implement model weight sharding if max_shard_size is specified
        if max_shard_size is not None:
            # Implement _shard_checkpoint or remove this logic if not needed
            # shards, index = self._shard_checkpoint(state_dict, max_shard_size)
            # for shard_file, shard in shards.items():
            #     self._save_shard(shard, save_directory, shard_file, safe_serialization)
            # if index is not None:
            #     save_function(index, os.path.join(save_directory, 'pytorch_model.bin.index.json'))
            pass
        else:
            # Use safe serialization if specified
            if safe_serialization:
                safe_save_file(state_dict, os.path.join(save_directory, 'model.safetensors'), metadata={"format": "pt"})
            else:
                save_function(state_dict, os.path.join(save_directory, 'pytorch_model.bin'))

        # Save config
        if hasattr(model_to_save, 'config') and hasattr(model_to_save.config, 'save_pretrained'):
            model_to_save.config.save_pretrained(save_directory)
        else:
            print("Warning: Model doesn't have a config with save_pretrained method. Config not saved.")

        # Handle push to hub
        if push_to_hub:
            if hasattr(self, '_push_to_hub'):
                return self._push_to_hub(save_directory, token=token, **kwargs)
            else:
                print("Warning: _push_to_hub method not implemented. Model not pushed to hub.")

        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.get("config", None)
        state_dict = kwargs.get("state_dict", None)

        # If config is not provided, try to load it
        if config is None:
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_file):
                config = cls.config_class.from_json_file(config_file)
            else:
                raise OSError(f"Config file not found in {pretrained_model_name_or_path}")

        # Instantiate model
        model = cls(config)

        if state_dict is None:
            # Look for various file types
            file_types = ["*.bin", "*.pt", "*.pth", "*.ckpt", "*.safetensors"]
            found_files = []
            for file_type in file_types:
                found_files.extend(glob.glob(os.path.join(pretrained_model_name_or_path, file_type)))
            
            if not found_files:
                logger.warning(f"No model weights found in {pretrained_model_name_or_path}. "
                               "Initializing model with random weights.")
                return model
            else:
                # Use the first file found
                state_dict = torch.load(found_files[0], map_location="cpu")

        # Load the state dict if it exists
        if state_dict:
            model.load_state_dict(state_dict, strict=False)

        return model
  