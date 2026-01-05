from transformers import PretrainedConfig, AutoConfig


class PALMConfig(PretrainedConfig):
    """
    Configuration class for Partial Attention Language Model (PALM).
    
    Key architectural features:
    - RoPE (Rotary Position Embeddings): Parameters inherited from base model
    - SPE (Separate Positional Encoding): Position reset at source→target boundary
    - Tied SAE head: SAE head shares weights with LM head
    - GQA (Grouped Query Attention): Supports num_kv_heads < num_attention_heads
    """
    model_type = 'llama'

    def __init__(
        self,
        base_model_name="meta-llama/Llama-3.2-3B",
        vocab_size=None,
        hidden_size=3072,
        num_hidden_layers=28,
        num_attention_heads=24,
        num_kv_heads=None,
        intermediate_size=8192,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=131072,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        # RoPE parameters (inherited from base model)
        rope_theta=10000.0,
        rope_scaling=None,
        # PALM-specific parameters
        tie_sae_head=True,  # If True, SAE head shares weights with LM head
        sae_weight=0.5,
        logit_softcap=30.0,
        fixed_source_length=100,
        # Training hyperparameters
        learning_rate=5e-5,
        warmup_steps=50,
        num_train_epochs=70,
        gradient_accumulation_steps=10,
        train_batch_size=64,
        logging_steps=100,
        max_length=512,
        min_length=1,
        gradient_checkpointing=False,
        **kwargs
    ):
        # Load base model config to inherit architecture parameters
        base_config = AutoConfig.from_pretrained(base_model_name)
        
        # Use base model values for critical params if not explicitly provided
        if vocab_size is None:
            vocab_size = base_config.vocab_size
        if pad_token_id is None:
            pad_token_id = getattr(base_config, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = getattr(base_config, 'eos_token_id', vocab_size - 1)
        if bos_token_id is None:
            bos_token_id = getattr(base_config, 'bos_token_id', 1)
        if eos_token_id is None:
            eos_token_id = getattr(base_config, 'eos_token_id', 2)
        
        # Handle token IDs that might be lists
        def _ensure_int(token_id):
            if isinstance(token_id, list):
                return token_id[0] if token_id else None
            return token_id
        
        pad_token_id = _ensure_int(pad_token_id)
        bos_token_id = _ensure_int(bos_token_id)
        eos_token_id = _ensure_int(eos_token_id)
        
        # Inherit RoPE parameters from base model if not provided
        # These are critical for preserving pretrained positional understanding
        if rope_theta == 10000.0:  # Use default only if not explicitly set
            rope_theta = getattr(base_config, 'rope_theta', 10000.0)
        if rope_scaling is None:
            rope_scaling = getattr(base_config, 'rope_scaling', None)
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Model architecture
        self.base_model_name = base_model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        
        # RoPE parameters
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        
        # PALM-specific parameters
        self.tie_sae_head = tie_sae_head
        self.sae_weight = sae_weight
        self.logit_softcap = logit_softcap
        self.fixed_source_length = fixed_source_length
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_batch_size = train_batch_size
        self.logging_steps = logging_steps
        self.max_length = max_length
        self.min_length = min_length
        self.gradient_checkpointing = gradient_checkpointing

        # Copy remaining attributes from base_config
        for key, value in base_config.to_dict().items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        return config
