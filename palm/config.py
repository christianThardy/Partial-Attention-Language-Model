from transformers import PretrainedConfig, AutoConfig


class PALMConfig(PretrainedConfig):
    '''Define configuration class for the partial 
        attention language model architecture'''
    # Specify the model type for identification in the broader framework
    model_type = 'llama'

    def __init__(
        self,
        base_model_name="meta-llama/Llama-3.2-3B", # Default base model, adjust as needed
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
        pad_token_id=128257, # Token ID used for padding sequences
        bos_token_id=1, # Token ID for the beginning of a sequence
        eos_token_id=2, # Token ID for the end of a sequence
        # Training hyperparameters
        learning_rate=5e-5, # Learning rate for optimizer
        warmup_steps=50, # Number of warmup steps for learning rate scheduler
        num_train_epochs=70, # Number of training epochs
        gradient_accumulation_steps=10, # Steps for gradient accumulation
        train_batch_size=64, # Training batch size
        logging_steps=100, # Log metrics every N steps
        sae_weight=0.5, # Weight for SAE loss in combined loss
        max_length=512, # Max generation length
        min_length=1, # Min generation length
        gradient_checkpointing=False, # Whether to use gradient checkpointing
        fixed_source_length=100, # Default fixed source length for attention mask
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
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_batch_size = train_batch_size
        self.logging_steps = logging_steps
        self.sae_weight = sae_weight
        self.max_length = max_length
        self.min_length = min_length
        self.fixed_source_length = fixed_source_length
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
