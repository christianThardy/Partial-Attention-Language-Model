from transformers import PretrainedConfig, AutoConfig


class PALMConfig(PretrainedConfig):
    '''Define configuration class for the partial 
        attention language model architecture'''
    # Specify the model type for identification in the broader framework
    model_type = 'llama'

    def __init__(
        self,
        base_model_name="meta-llama/Llama-3.2-3B", # Default base model
        vocab_size=128256, # Vocabulary size, defining number of tokens available
        hidden_size=3072, # Size of hidden layers in the model
        num_hidden_layers=28, # Number of hidden layers in the model
        num_attention_heads=24, # Number of attention heads for multi-head attention mechanism
        intermediate_size=8192, # Size of the intermediate feed-forward layer in transformer blocks
        hidden_act="silu", # Activation function used in hidden layers
        hidden_dropout_prob=0.1, # Dropout probability for hidden layers
        attention_probs_dropout_prob=0.1, # Dropout probability for attention probabilities
        max_position_embeddings=131072, # Maximum number of position embeddings (sequence length)
        initializer_range=0.02, # Range for weight initialization
        layer_norm_eps=1e-5, # Epsilon parameter for layer normalization to avoid division by zero
        pad_token_id=None, # Token ID used for padding sequences
        bos_token_id=128000, # Token ID for the beginning of a sequence
        eos_token_id=128001, # Token ID for the end of a sequence
        tie_word_embeddings=True, # Enable tying input and output embeddings to reduce parameters
        torch_dtype="bfloat16", # Use bfloat16 data type for efficient training and inference
        rope_scaling={ # Configure RoPE (Rotary Position Embedding) scaling factors
            "factor": 32.0, # General scaling factor for embeddings
            "high_freq_factor": 4.0, # High-frequency scaling adjustment
            "low_freq_factor": 1.0, # Low-frequency scaling adjustment
            "original_max_position_embeddings": 8192, # Base sequence length for scaling
            "rope_type": "llama3" 
        },
        rope_theta=500000.0, # Theta value for rotary embeddings, impacting periodicity
        use_cache=True, # Enable caching for improved decoding efficiency
        attention_bias=False, # Disable bias in attention layers for simplicity
        mlp_bias=False, # Disable bias in MLP layers to reduce parameter count
        rms_norm_eps=1e-05, # Small value to prevent division by zero in RMS normalization
        transformers_version="4.45.0.dev0",
        architectures=["LlamaForCausalLM"], # Supported architectures for the model
        sae_weight=0.5, # Weight for source-autoencoder attention loss
        gradient_checkpointing=True, # Enable gradient checkpointing to save memory
        **kwargs
    ):
        # Call the parent class (PretrainedConfig) constructor with specific token IDs
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        # Assign initialization parameters to instance variables
        self.base_model_name = base_model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype
        self.pretraining_tp = pretraining_tp
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.rms_norm_eps = rms_norm_eps
        self.transformers_version = transformers_version
        self.architectures = architectures
        self.layer_norm_eps = layer_norm_eps  # Ensure this is set
        self.sae_weight = sae_weight
        self.gradient_checkpointing = gradient_checkpointing

        # Load base model config
        base_config = AutoConfig.from_pretrained(base_model_name)

        # Copy attributes from base_config that are not already set
        for key, value in base_config.to_dict().items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    # Method to create an instance of PALMConfig from a pre-trained model's configuration
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        return config
    
