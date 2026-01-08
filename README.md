# Partial Attention Language Model (PALM)

As generations grow longer, LLMs often exhibit **context rot**. This **attention degradation** causes a loss of grounding in earlier context, which in turn reduces adherence to retrieved evidence, constraints and ultimately results in instruction drift, hallucination, and forgetting.

In standard decoder-only transformers, this degradation is driven by how the attention and residual stream become increasingly dominated by the model's own continuation, weakening effective access to the original source (input) tokens as generation continues.

**PALM** (≠ [PaLM](https://en.wikipedia.org/wiki/PaLM)) is a decoder-only architecture that preserves source connectivity throughout decoding by introducing a partial attention pathway: generated tokens can attend directly to the full source prefix at every step, while maintaining causal structure over the generated continuation. 

This creates a stable channel for conditioning on prompt instructions, system constraints and retrieved documents. Improving source–target coherence and reducing attention degeneration during conditional text generation.

## How it works

<img width="2041" height="1035" alt="Image" src="https://github.com/user-attachments/assets/2f8916fe-0532-42fb-9725-f2d6c36fd62d" />

## Overview

This PALM implementation operationalizes and modernizes research that re-interprets decoder-only architectures as regularized encoder-decoders (Fu et al.), but for autoregressive chat models. 

By treating PALM as an architectural wrapper, partial attention mechanisms are integrated into existing dense, open-weights (Llama, Mistral, Qwen, Hermes, etc.) while preserving their fundamental learned representations.

PALM enhances standard decoder-only models by introducing:

- **Bidirectional attention over source tokens** - The attention mask allows source tokens (think of this as the models context window or system prompt) to attend to each other bidirectionally (like an encoder), while target tokens attend causally to previous tokens and fully to all source tokens. 

  - **TLDR**; the prompt/context attends to itself fully, capturing richer contextual representations.

- **Partial attention mechanism** - Each layer applies an additional attention operation where query vectors from the full sequence attend to key/value vectors derived solely from the source portion, processed through a learned transformation (Fp network). This maintains a persistent connection to the original prompt throughout generation. 

  - **TLDR**; a dedicated cross-attention-like module where all positions attend to the processed source embeddings.

- **Source Auto-Encoding (SAE) auxiliary loss** - The SAE head predicts the source tokens from their hidden representations, acting as a regularizer:
   ```
   combined_loss = lm_loss + λ * sae_loss
   ```
   The `sae_weight` (λ) hyperparameter controls the strength of this regularization. 
   
    - **TLDR**; regularizes the model to reconstruct the source, encouraging faithful representations

- **Separate Positional Encoding (SPE)** - positions reset at the boundary between source and target to better delineate the two regions.

- **Language embeddings** - learned embeddings distinguish source (prompt) from target (generation).

---

## Start training in 10 Minutes

```bash
# Clone and install
git clone https://github.com/christianThardy/Partial-Attention-Language-Model.git
cd Partial-Attention-Language-Model
pip install torch transformers peft bitsandbytes wandb tqdm safetensors datasets

# Run the training notebook
jupyter notebook notebooks/finetune_palm.ipynb
```

The notebook walks through:
1. Loading a base model (Llama/Qwen/etc.) and transferring weights to PALM
2. Fine-tuning on a QA dataset with LoRA
3. Evaluating generation quality and source faithfulness

**Lower VRAM?** Use QLoRA (8GB+) or reduce batch size. See the notebook for configuration.

## Quick Start

```python
from palm import PALMConfig, PALMModel, transfer_weights_to_palm

# Initialize configuration (inherits from any HuggingFace model)
config = PALMConfig(
    base_model_name="meta-llama/Llama-3.3-70B-Instruct",
    fixed_source_length=128,
    sae_weight=0.5,
)

# Create model and transfer pretrained weights
model = PALMModel(config)
model = transfer_weights_to_palm(model, config.base_model_name)

# Forward pass with source/target distinction
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    source_len=source_lengths,  # per-sample source lengths
)
lm_logits, combined_loss, lm_loss, sae_loss = outputs
```

## Features

### Weight Transfer

PALM doesn't train from scratch, it bootstraps from any compatible open-weight model:

```python
from palm import PALMModel, PALMConfig, transfer_weights_to_palm

# Supports: Llama, Qwen, Mistral, Phi, Gemma, Falcon, Hermes/Seed
model = PALMModel(PALMConfig(base_model_name="Qwen/Qwen2.5-3B"))
model = transfer_weights_to_palm(model, "NousResearch/Hermes-4.3-36B")
```

**What transfers:** Self-attention projections (Q, K, V, O), MLP weights, embeddings, LM head; the bulk of the model.

**What's fresh/bootstrapped:** Language and position embeddings are fresh. PALM-specific components are initialized from pretrained weights then fine-tuned:

- partial attention Q, K, V, O ← cloned from self-attention
- SAE head ← cloned from LM head
- LayerNorms ← cloned from input norms
- Fp network ← identity-like (residual-dominant)

### LoRA / QLoRA Fine-tuning

```python
from palm import apply_lora, apply_qlora

# Standard LoRA
model = apply_lora(model, r=16, lora_alpha=32)

# Quantized LoRA for memory efficiency
model = apply_qlora(model, r=16, lora_alpha=32)
```

Targets all attention projections in both self-attention and partial attention modules.

### Training

```python
from palm import PALMTrainer

trainer = PALMTrainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    config=config,
)
trainer.train()
```

Includes gradient accumulation, learning rate scheduling, and WandB integration.

### Generation with KV Caching

```python
generated = model.generate(
    input_ids=prompt_ids,
    max_length=256,
    temperature=0.7,
    top_p=0.9,
    use_cache=True,  # KV caching for efficient autoregressive generation
)
```

### Evaluation Suite

```python
from palm.evaluation import (
    StaggeredEvaluator,
    CheckpointScoreboard,
    LambdaSweepRunner,
    SourceAblationEvaluator,
)

# Lightweight evaluation during training
evaluator = StaggeredEvaluator(model, tokenizer)
results = evaluator.evaluate(eval_samples)

# Hyperparameter analysis
sweep = LambdaSweepRunner(config, lambda_values=[0.1, 0.3, 0.5, 0.7, 1.0])
sweep_results = sweep.run(train_loader, eval_loader)

# Source ablation to test sensitivity to prompt
ablation = SourceAblationEvaluator(model, tokenizer)
ablation_results = ablation.evaluate(samples)
```

## Status

This is a research implementation. The architecture is complete and trainable, but:
- Benchmarks against baselines are ongoing
- Hyperparameter tuning (especially `sae_weight`) is task-dependent

## References

Fu, Lam, Yu, Cho So, Hu, Liu, Collier. *Decoder-Only or Encoder-Decoder? Interpreting Language Model as a Regularized Encoder-Decoder*. 2023. [[arXiv]](https://arxiv.org/pdf/2304.04052)
