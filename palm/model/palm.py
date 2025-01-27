
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
            # Create input ids, attention mask and labels
            input_ids = self._validate_input(input_ids)
            attention_mask = self._prepare_attention_mask(input_ids, attention_mask)
            labels = self._prepare_labels(input_ids, labels)

            # If source_len isn't specified, treat entire seq_length as source
            if source_len is not None:
                batch_size, seq_len = input_ids.shape
                device_ = input_ids.device
                source_len = torch.full(
                    (batch_size,),
                    seq_len,
                    dtype=torch.long,
                    device=device_
                )

            hidden_states = self.embeddings(input_ids, source_len=source_len)
            hidden_states = hidden_states * self.embed_scale
    
            for layer_module in self.layers:
                if self.gradient_checkpointing and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer_module, hidden_states, attention_mask, source_len
                    )
                else:
                    hidden_states = layer_module(hidden_states, attention_mask, source_len)

                if not torch.isfinite(hidden_states).all():
                    raise ValueError("NaN or Inf detected in hidden_states")

            # Use entire hidden_states for sae_head, not slicing
            lm_logits = self.lm_head(hidden_states)
            sae_logits = self.sae_head(hidden_states)  # shape [B, T, vocab_size]

            # Compute losses with per‐sample mask
            loss, sae_loss, combined_loss = self._compute_loss(
                lm_logits, sae_logits, labels, source_len
            )  
            return lm_logits, combined_loss, loss, sae_loss   
    
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes - input_ids: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'not a tensor'}, "
                  f"attention_mask: {attention_mask.shape if isinstance(attention_mask, torch.Tensor) else 'not a tensor'}, "
                  f"labels: {labels.shape if isinstance(labels, torch.Tensor) else 'not a tensor'},")
            raise

    def _validate_input(self, input_ids):
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if hasattr(self, 'module'):
            device = next(self.module.parameters()).device
        else:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                pass
        return input_ids.to(device)

    def _prepare_attention_mask(self, input_ids, attention_mask):
        if attention_mask is None:
            attention_mask = self.create_bidirectional_attention_mask(input_ids, input_ids.size(1))
        return attention_mask.to(input_ids.device)

    def _prepare_labels(self, input_ids, labels):
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            return labels.to(input_ids.device)
        return None

    # Do a per‐sample masked loss for SAE
    def _compute_loss(self, lm_logits, sae_logits, labels, source_len):
        """
        Implementation for each sample's source_len. Create a mask that selects
        only [0..source_len[i]-1] per batch example for the SAE loss.
        """
        if labels is None:
            return None, None, None

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        device_type_ = lm_logits.device.type

        with torch.amp.autocast(device_type=device_type_, enabled=False):
            lm_logits = lm_logits.float()
            sae_logits = sae_logits.float()
            labels = labels.long()

            # 1) Normal LM loss over all tokens
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1)
            )
            # Build a boolean mask for each sample's source portion
            B, T, _ = sae_logits.shape   # e.g. [8, 1024, vocab_size]
            range_tensor = torch.arange(T, device=labels.device).unsqueeze(0)  # shape [1, T]
            # shape [B, T], True if j < source_len[i], else False
            mask = range_tensor < source_len.unsqueeze(1)

            # Create a copy of labels for SAE, ignoring positions beyond source_len
            sae_labels = labels.clone()
            sae_labels[~mask] = -100  # set them to ignore_index
            sae_loss = loss_fct(
                sae_logits.view(-1, sae_logits.size(-1)),
                sae_labels.view(-1)
            )
            
        # Zero out inf/nan
        if not torch.isfinite(loss):
            logger.warning("Loss is non-finite; zeroing it out.")
            loss = torch.zeros(1, device=loss.device, dtype=loss.dtype)
        if not torch.isfinite(sae_loss):
            logger.warning("SAE loss is non-finite; zeroing it out.")
            sae_loss = torch.zeros(1, device=sae_loss.device, dtype=sae_loss.dtype)

        sae_weight = getattr(self.config, 'sae_weight', 0.5)
        combined_loss = loss + sae_weight * sae_loss

        if not torch.isfinite(combined_loss):
            logger.warning("Combined loss is non-finite; zeroing it out.")
            combined_loss = torch.zeros(1, device=combined_loss.device, dtype=combined_loss.dtype)

        if not torch.isfinite(combined_loss):
            raise ValueError("Combined loss is NaN or Inf")

        return loss, sae_loss, combined_loss

    def generate(
        self, input_ids, max_length=None, min_length=None, do_sample=True, temperature=1.0, 
        top_k=50, top_p=1.0, repetition_penalty=1.0, pad_token_id=None, eos_token_id=None, 
        attention_mask=None, **kwargs
    ):
        """Generate method with improved memory efficiency and vectorized operations."""
        # Set default values for generation parameters
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # Ensure input_ids are on the correct device
        dev = next(self.parameters()).device
        input_ids = input_ids.to(dev, non_blocking=True)

        batch_size, seq_len = input_ids.shape
        cumulative_attention_mask = torch.zeros(
            (batch_size, 1, max_length, max_length),
            device=dev,
            dtype=torch.bool
        )
        cumulative_attention_mask[:, :, :seq_len, :seq_len] = True
        
        # Initialize sequence tracking and keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, device=dev, dtype=torch.long)
        
        # Initialize generated sequence with the input sequence
        generated_sequence = input_ids
        
        next_tokens = torch.zeros(batch_size, device=dev, dtype=torch.long)

        for _ in range(max_length - seq_len):
            current_length = generated_sequence.size(1)
            cumulative_attention_mask[:, :, current_length-1, :current_length] = True


            model_inputs = {
                "input_ids": generated_sequence,
                "attention_mask": cumulative_attention_mask[:, :, :current_length, :current_length]
            }

            # Specify device_type for autocast
            generation_device_type = dev.type
            with torch.no_grad(), torch.amp.autocast(device_type=generation_device_type,
                                                     enabled=(generation_device_type == "cuda")):
                outputs = self(**model_inputs)
                next_token_logits = outputs[0][:, -1, :]

                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits,
                    current_length,
                    max_length,
                    min_length,
                    repetition_penalty,
                    generated_sequence
                )
                next_token_logits.div_(temperature)
                next_token_logits = self.top_k_top_p_filtering(
                    next_token_logits,
                    top_k=top_k,
                    top_p=top_p
                )
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = next_token_logits.argmax(dim=-1)

            next_tokens = torch.where(
                unfinished_sequences.bool(),
                next_tokens,
                pad_token_id * torch.ones_like(next_tokens)
            )
            unfinished_sequences.mul_((next_tokens != eos_token_id).long())

            generated_sequence = torch.cat([
                generated_sequence,
                next_tokens.unsqueeze(-1)
            ], dim=-1)

            if unfinished_sequences.max() == 0:
                break
        return generated_sequence
    
    def adjust_logits_during_generation(self, logits, cur_len, max_length, min_length, repetition_penalty, input_ids):
        """Adjust token logits during generation. Optimized adjustment with vectorized operations."""
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            # Vectorized repetition penalty application
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
        # Prevent generation of tokens before min_length
        if cur_len < min_length:
            logits[:, self.config.eos_token_id] = float('-inf')

        return logits
    
    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """Filter a distribution of logits using top-k and/or top-p (nucleus) filtering."""
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            topk_values, _ = torch.topk(logits, top_k)
            indices_to_remove = logits < topk_values[..., -1, None]
            logits.masked_fill_(indices_to_remove, filter_value)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors to original indexing
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits.masked_fill_(indices_to_remove, filter_value)
            
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

        # Validate configuration
        if not hasattr(self, 'config'):
            raise AttributeError("Model doesn't have a config attribute")

        # Efficient state dict handling
        state_dict = state_dict or (
            self.module.state_dict() if hasattr(self, 'module') else self.state_dict()
        )
        
        # Set up LFS tracking for large files
        with open(os.path.join(save_directory, '.gitattributes'), 'w') as f:
            f.write('*.bin filter=lfs diff=lfs merge=lfs -text\n')
            f.write('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
        
        # Optimize file saving with sharding support
        if safe_serialization:
            from safetensors.torch import save_file
            save_file(
                state_dict,
                os.path.join(save_directory, 'model.safetensors'),
                metadata={"format": "pt"}
            )
        else:
            if max_shard_size and max_shard_size != "5GB":
                from transformers.modeling_utils import shard_checkpoint
                sharded_state_dict = shard_checkpoint(state_dict, max_shard_size=max_shard_size)
                for shard_file, shard in sharded_state_dict.items():
                    save_function(shard, os.path.join(save_directory, shard_file))
            else:
                save_function(
                    state_dict,
                    os.path.join(save_directory, 'pytorch_model.bin')
                )

        # Save model config if available
        model_to_save = self.module if hasattr(self, 'module') else self
        if hasattr(model_to_save, 'config'):
            model_to_save.config.save_pretrained(save_directory)

       # Save tokenizer if available
        if hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(save_directory)
        
        # Handle hub pushing with metadata
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
        """Helper method for efficient config loading."""
        config_file = os.path.join(path, "config.json")
        if not os.path.exists(config_file):
            raise OSError(f"Config file not found in {path}")
        return PALMConfig.from_json_file(config_file)
       
    @staticmethod
    def _find_and_load_state_dict(path):
        """Helper method for efficient state dict loading."""
        file_types = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"]
        for pattern in file_types:
            files = glob.glob(os.path.join(path, pattern))
            if files:
                return torch.load(files[0], map_location="cpu")
        return None
  
