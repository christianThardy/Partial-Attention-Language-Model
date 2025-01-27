
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file as safe_save_file

from attention import PALMAttention, PALMPartialAttention
from embeddings import PALMEmbeddings

import glob

# Logger object
logger = logging.getLogger()
# Level of the logger. Values: DEBUG, INFO, WARNING, ERROR, CRITICAL
logger.setLevel(logging.DEBUG)
# Handler writes log messages to the notebook's output
handler = logging.StreamHandler()
# Set the format for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class PALMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Linear layer to project hidden states to a larger intermediate size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # Activation function (GELU) to introduce non-linearity
        self.intermediate_act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        # Apply linear transformation and activation function to the hidden states
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
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

        # Partial attention mechanism for handling source input sequences
        self.partial_attention = PALMPartialAttention(config)

        # Intermediate layer for processing the attention output
        self.intermediate = PALMIntermediate(config)

        # Output layer to produce the final output for this layer
        self.output = PALMOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        # Standard self-attention (the "ATTl" block in the paper)
        attention_output = self.attention(
            hidden_states, 
            attention_mask=attention_mask
        )

        # Apply partial attention using a subset of the hidden states 
        # (focusing only on the source portion)
        partial_attention_output = self.partial_attention(
            hidden_states=attention_output,
            source_states=attention_output,
            source_len=source_len,
            attention_mask=attention_mask
        )
        
        # Process output of the intermediate layer with the partial attention
        intermediate_output = self.intermediate(partial_attention_output)
        layer_output = self.output(intermediate_output, partial_attention_output)

        # Clean up any leftover NaNs
        if not torch.isfinite(layer_output).all():
            layer_output = torch.where(
                torch.isfinite(layer_output),
                layer_output,
                torch.zeros_like(layer_output)
            )
            logger.warning("Non-finite values detected in final PALMLayer output, zeroing them")
            
        return layer_output
    

class PALMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layer for input tokens
        self.embeddings = PALMEmbeddings(config)

        # Stack of transformer layers
        self.layers = nn.ModuleList([PALMLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Linear layers for language modeling, source autoencoding heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sae_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Scale embeddings by sqrt(d_model)
        self.embed_scale = math.sqrt(config.hidden_size)
       
        # Initialize weights with proper scaling
        self._init_weights()
       
        self.gradient_checkpointing = config.gradient_checkpointing

    def _init_weights(self):
        # Initialize embedding weights
        nn.init.normal_(self.embeddings.word_embeddings.weight, mean=0.0,
                       std=0.02/math.sqrt(2 * self.config.num_hidden_layers))
       
        # Initialize output heads
        nn.init.normal_(self.lm_head.weight, mean=0.0,
                       std=0.02/math.sqrt(2 * self.config.num_hidden_layers))
        nn.init.normal_(self.sae_head.weight, mean=0.0,
                       std=0.02/math.sqrt(2 * self.config.num_hidden_layers))

    def create_bidirectional_attention_mask(self, input_ids, source_len):
        batch_size, seq_length = input_ids.size()
    
        # Create a mask for bidirectional attention on the source sequence and causal attention on the target
        mask = torch.zeros((batch_size, 1, seq_length, seq_length), device=input_ids.device)
    
        # Define length of the source sequence
        source_len = min(source_len, seq_length)
    
        # Apply bidirectional attention to the source sequence
        mask[:, :, :source_len, :source_len] = 1
    
        # If source is shorter than sequence length, add causal mask for the target sequence
        # Apply causal attention to the target sequence
        if source_length < seq_length:
            target_length = seq_length - source_len
            causal_mask = torch.tril(torch.ones((target_length, target_length), device=input_ids.device))
            mask[:, :, source_length:, source_length:] = causal_mask
            # Allow target sequence to attend to all of source sequence
            mask[:, :, source_length:, :source_length] = 1
    
        # Convert the mask to a form suitable for additive attention
        # So convert 0s to -10000.0 and 1s to 0.0
        mask = (1.0 - mask) * -10000.0
        return mask

    def forward(self, input_ids, attention_mask=None, labels=None, source_len=None):
        try:
            # Ensure input_ids is a tensor and has the correct dimensions
            input_ids = torch.tensor(input_ids) if not isinstance(input_ids, torch.Tensor) else input_ids
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # Add batch dimension
            
            # Create or validate attention mask
            if attention_mask is None:
                attention_mask = self.create_bidirectional_attention_mask(input_ids)
            else:
                attention_mask = torch.tensor(attention_mask) if not isinstance(attention_mask, torch.Tensor) else attention_mask
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
    
            # Ensure labels are tensors and have the correct dimensions
            if labels is not None:
                labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)  # Add batch dimension
    
            # Embedding lookup
            #print("Forward pass: Starting embeddings")
            hidden_states = self.embeddings(input_ids)
            #print("Forward pass: Embeddings complete")
    
            # Pass through each layer
            for layer in self.layers:
                #print(f"Forward pass: Starting layer {layer}")
                hidden_states = layer(hidden_states, attention_mask)
                #print(f"Forward pass: Layer {layer} complete")
    
            # Compute logits for language modeling
            #print("Forward pass: Starting lm_head")
            lm_logits = self.lm_head(hidden_states)
            #print("Forward pass: lm_head complete")

            # Compute logits for sequence autoencoding
            #print("Forward pass: Starting sae_head")
            sae_logits = self.sae_head(hidden_states[:, :self.config.fixed_source_length])
            #print("Forward pass: sae_head complete")

            # Initialize loss variables
            loss = None
            sae_loss = None
            combined_loss = None
            
            # Compute loss if labels are provided
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                #print(f"Forward pass: Loss calculated: {loss.item()}")

                # Calculate SAE loss if source length is provided
                if source_len is not None:
                    sae_labels = input_ids[:, :self.config.fixed_source_length]
                    sae_loss = loss_fct(sae_logits.view(-1, self.config.vocab_size), sae_labels.reshape(-1))

                # Combine losses
                combined_loss = loss + self.sae_weight * sae_loss
                
            return lm_logits, combined_loss, loss, sae_loss   
    
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes - input_ids: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'not a tensor'}, "
                  f"attention_mask: {attention_mask.shape if isinstance(attention_mask, torch.Tensor) else 'not a tensor'}, "
                  f"labels: {labels.shape if isinstance(labels, torch.Tensor) else 'not a tensor'},")
            raise

    def generate(
        self, input_ids, max_length=None, min_length=None, do_sample=True, temperature=1.0, 
        top_k=50, top_p=1.0, repetition_penalty=1.0, pad_token_id=None, eos_token_id=None, 
        attention_mask=None, **kwargs
    ):
        # Set default values for generation parameters
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # Ensure input_ids are on the correct device
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_bidirectional_attention_mask(input_ids)
        
        # Initialize sequence tracking and keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        
        # Initialize generated sequence with the input sequence
        generated_sequence = input_ids

        while True:
            # Prepare model inputs
            model_inputs = {
                "input_ids": generated_sequence,
                "attention_mask": attention_mask,
            }

            # Forward pass without gradients
            with torch.no_grad():
                outputs = self(**model_inputs)
            
            # Get the next token logits
            next_token_logits = outputs[0][:, -1, :]

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
            # Finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Update unfinished sequences
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # Append next tokens to the sequence
            generated_sequence = torch.cat([generated_sequence, next_tokens.unsqueeze(-1)], dim=-1)

            # Update attention mask
            new_attention_mask = self.create_bidirectional_attention_mask(generated_sequence)
            attention_mask = new_attention_mask

            # Stop if we've reached max_length or all sequences are finished
            if unfinished_sequences.max() == 0 or generated_sequence.shape[1] >= max_length:
                break

        return generated_sequence
    
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
    
    def save_pretrained(
        self, save_directory, is_main_process=True, state_dict=None, save_function=torch.save, push_to_hub=False, 
        max_shard_size="5GB", safe_serialization=True, variant=None, token=None, save_peft_format=True, **kwargs):
        """Save a model and its configuration file to a directory, so that it can be re-loaded using the 
        `from_pretrained` class method."""
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
  
