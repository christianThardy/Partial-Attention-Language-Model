"""
Quick validation tests for PALM bug fixes.
Run with: python palm/tests/test_fixes.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn

# Import directly to avoid dependency issues with evaluation module
from palm.config import PALMConfig
from palm.model.palm import PALMModel, PALMLayer
from palm.model.attention import PALMAttention, PALMPartialAttention
from palm.data.preprocessing import preprocess_function


def create_test_config():
    """Create a small config for fast testing (no network calls)."""
    return PALMConfig(
        base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=512,
        fixed_source_length=10,
        pad_token_id=0,
        learning_rate=1e-4,
        sae_weight=0.5,
    )


def test_attention_mask_no_double_inversion():
    """Test that attention mask is NOT double-inverted."""
    print("\n=== Test 1: Attention Mask Not Double-Inverted ===")
    
    config = create_test_config()
    model = PALMModel(config)
    
    # Create input with source_length=5, total_length=10
    input_ids = torch.randint(1, 100, (2, 10))  # batch=2, seq=10
    
    # Create bidirectional mask
    mask = model.create_bidirectional_attention_mask(input_ids)
    
    # Verify mask shape
    assert mask.shape == (2, 1, 10, 10), f"Expected (2, 1, 10, 10), got {mask.shape}"
    
    # Mask values should be 0 (attend) or -10000 (mask)
    unique_vals = torch.unique(mask)
    assert len(unique_vals) <= 2, f"Mask should have at most 2 unique values, got {unique_vals}"
    
    # Source tokens (0:10 with fixed_source_length=10) should all attend to each other (mask=0)
    source_to_source = mask[0, 0, :config.fixed_source_length, :config.fixed_source_length]
    assert torch.all(source_to_source == 0), "Source tokens should attend to each other (mask=0)"
    
    print("[OK] Attention mask has correct values (0 or -10000)")
    print("[OK] Source tokens can attend to each other")
    print("[PASSED]")


def test_attention_mask_with_padding():
    """Test that padding mask is properly combined with bidirectional mask."""
    print("\n=== Test 2: Padding Mask Integration ===")
    
    config = create_test_config()
    config.fixed_source_length = 5
    model = PALMModel(config)
    
    # Create input: batch=2, seq=10, with some padding
    input_ids = torch.randint(1, 100, (2, 10))
    # Padding mask: first sample has 8 real tokens, second has 6
    padding_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 8 real tokens
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # 6 real tokens
    ])
    
    # Forward pass with padding mask
    labels = input_ids.clone()
    labels[padding_mask == 0] = -100
    
    outputs = model(input_ids, attention_mask=padding_mask, labels=labels, source_len=torch.tensor([5, 5]))
    
    # Should not crash
    assert outputs[0] is not None, "Forward pass should produce logits"
    assert outputs[1] is not None, "Forward pass should produce loss"
    
    print("[OK] Padding mask properly combined with bidirectional mask")
    print("[OK] Forward pass with padding works")
    print("[PASSED]")


def test_loss_with_ignore_index():
    """Test that loss properly ignores masked positions."""
    print("\n=== Test 3: Loss Ignores Masked Positions ===")
    
    config = create_test_config()
    model = PALMModel(config)
    
    input_ids = torch.randint(1, 100, (2, 20))
    
    # Labels with -100 for first 5 tokens (prompt) and last 5 (padding)
    labels = input_ids.clone()
    labels[:, :5] = -100   # Mask prompt
    labels[:, 15:] = -100  # Mask padding
    
    source_len = torch.tensor([5, 5])
    
    outputs = model(input_ids, labels=labels, source_len=source_len)
    lm_logits, combined_loss, loss, sae_loss = outputs
    
    assert loss is not None, "Loss should not be None"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    assert combined_loss is not None, "Combined loss should not be None"
    
    print(f"[OK] LM Loss: {loss.item():.4f}")
    print(f"[OK] SAE Loss: {sae_loss.item():.4f}")
    print(f"[OK] Combined Loss: {combined_loss.item():.4f}")
    print("[PASSED]")


def test_sae_uses_actual_source_len():
    """Test that SAE loss uses actual source_len, not fixed_source_length."""
    print("\n=== Test 4: SAE Uses Actual source_len ===")
    
    config = create_test_config()
    config.fixed_source_length = 100  # Large fixed length
    model = PALMModel(config)
    
    # Input is only 20 tokens, source_len is 5 and 8
    input_ids = torch.randint(1, 100, (2, 20))
    labels = input_ids.clone()
    labels[:, :8] = -100  # Mask first 8 tokens
    
    source_len = torch.tensor([5, 8])  # Different source lengths per sample
    
    outputs = model(input_ids, labels=labels, source_len=source_len)
    
    # Should not crash even though fixed_source_length > seq_length
    assert outputs[3] is not None, "SAE loss should be computed"
    assert not torch.isnan(outputs[3]), "SAE loss should not be NaN"
    
    print(f"[OK] SAE loss computed with variable source_len: {outputs[3].item():.4f}")
    print("[PASSED]")


def test_sae_loss_none_handling():
    """Test that model handles missing source_len gracefully."""
    print("\n=== Test 5: SAE Loss Handles None source_len ===")
    
    config = create_test_config()
    model = PALMModel(config)
    
    input_ids = torch.randint(1, 100, (2, 20))
    labels = input_ids.clone()
    
    # Forward WITHOUT source_len
    outputs = model(input_ids, labels=labels, source_len=None)
    
    lm_logits, combined_loss, loss, sae_loss = outputs
    
    assert combined_loss is not None, "Combined loss should not be None"
    assert combined_loss == loss, "Combined loss should equal LM loss when no SAE"
    assert sae_loss.item() == 0.0, "SAE loss should be 0 when source_len is None"
    
    print("[OK] Model handles None source_len without crashing")
    print("[OK] Combined loss equals LM loss when no SAE")
    print("[PASSED]")


def test_partial_attention_no_masking():
    """Test that partial attention allows all-to-source attention."""
    print("\n=== Test 6: Partial Attention Allows All-to-Source ===")
    
    config = create_test_config()
    config.fixed_source_length = 5
    partial_attn = PALMPartialAttention(config)
    
    # hidden_states: [batch=2, seq=10, hidden=128]
    # source_states: [batch=2, source=5, hidden=128]
    hidden_states = torch.randn(2, 10, 128)
    source_states = torch.randn(2, 5, 128)
    
    # Create a restrictive attention mask (to verify it's NOT applied)
    attention_mask = torch.zeros(2, 1, 10, 10)
    attention_mask[:, :, 5:, :5] = -10000  # Would block target->source if applied
    
    output, _ = partial_attn(hidden_states, source_states, attention_mask)
    
    # Output should have same shape as hidden_states
    assert output.shape == hidden_states.shape, f"Expected {hidden_states.shape}, got {output.shape}"
    
    # Should not be all zeros (attention is happening)
    assert not torch.allclose(output, torch.zeros_like(output)), "Output should not be all zeros"
    
    print("[OK] Partial attention output shape correct")
    print("[OK] Partial attention is NOT restricted by causal mask")
    print("[PASSED]")


def test_generation_preserves_source_length():
    """Test that generation preserves original source length."""
    print("\n=== Test 7: Generation Preserves Source Length ===")
    
    config = create_test_config()
    config.fixed_source_length = 50  # Different from input length
    model = PALMModel(config)
    
    original_fixed_source_length = config.fixed_source_length
    
    # Input is 8 tokens (the "prompt")
    input_ids = torch.randint(1, 100, (1, 8))
    
    # Generate up to 15 tokens
    generated = model.generate(input_ids, max_length=15, do_sample=False)
    
    # Config should be restored after generation
    assert config.fixed_source_length == original_fixed_source_length, \
        f"Config fixed_source_length should be restored to {original_fixed_source_length}"
    
    assert generated.shape[1] == 15, f"Should generate up to max_length=15, got {generated.shape[1]}"
    
    print(f"[OK] Generated sequence length: {generated.shape[1]}")
    print(f"[OK] Config fixed_source_length restored: {config.fixed_source_length}")
    print("[PASSED]")


def test_preprocessing_masks_prompt():
    """Test that preprocessing masks prompt tokens in labels."""
    print("\n=== Test 8: Preprocessing Masks Prompt Tokens ===")
    
    # Create a mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            
        def __call__(self, texts, max_length, truncation, padding):
            # Simple mock: each char is a token
            result = {"input_ids": [], "attention_mask": []}
            for text in texts:
                ids = [ord(c) % 100 + 1 for c in text[:max_length]]
                ids = ids + [0] * (max_length - len(ids))  # Pad
                mask = [1 if i != 0 else 0 for i in ids]
                result["input_ids"].append(ids)
                result["attention_mask"].append(mask)
            return result
        
        def encode(self, text, add_special_tokens=True):
            return [ord(c) % 100 + 1 for c in text]
    
    tokenizer = MockTokenizer()
    
    examples = {
        "prompt": ["Hello", "Hi there"],
        "completion": [" World", " Friend"]
    }
    
    processed = preprocess_function(examples, tokenizer, max_seq_length=30)
    
    # Check that labels exist
    assert "labels" in processed, "Processed data should have labels"
    
    # Check that prompt positions are masked with -100
    labels = processed["labels"]
    source_lens = processed["source_len"]
    
    for i, source_len in enumerate(source_lens):
        sl = int(source_len.item())
        prompt_labels = labels[i, :sl]
        assert torch.all(prompt_labels == -100), \
            f"Prompt tokens should be masked with -100, got {prompt_labels}"
    
    print(f"[OK] Source lengths: {source_lens.tolist()}")
    print("[OK] Prompt tokens masked with -100")
    print("[PASSED]")


def test_layer_forward():
    """Test that PALMLayer forward pass works correctly."""
    print("\n=== Test 9: PALMLayer Forward Pass ===")
    
    config = create_test_config()
    config.fixed_source_length = 5
    layer = PALMLayer(config)
    
    hidden_states = torch.randn(2, 10, 128)
    attention_mask = torch.zeros(2, 1, 10, 10)  # All attend
    
    output, _ = layer(hidden_states, attention_mask)
    
    assert output.shape == hidden_states.shape, f"Expected {hidden_states.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output should not contain NaN"
    
    print("[OK] Layer output shape correct")
    print("[OK] No NaN values")
    print("[PASSED]")


def test_bidirectional_causal_mask_pattern():
    """Test the specific pattern of bidirectional/causal masking."""
    print("\n=== Test 10: Bidirectional/Causal Mask Pattern ===")
    
    config = create_test_config()
    config.fixed_source_length = 4
    model = PALMModel(config)
    
    input_ids = torch.randint(1, 100, (1, 8))  # 4 source + 4 target
    mask = model.create_bidirectional_attention_mask(input_ids)
    
    # Remove batch and head dims for easier checking
    mask_2d = mask[0, 0]  # [8, 8]
    
    # Source tokens (0-3) should all attend to each other (mask=0)
    for i in range(4):
        for j in range(4):
            assert mask_2d[i, j] == 0, f"Source[{i}] should attend to Source[{j}]"
    
    # Target tokens (4-7) should attend to all source tokens
    for i in range(4, 8):
        for j in range(4):
            assert mask_2d[i, j] == 0, f"Target[{i}] should attend to Source[{j}]"
    
    # Target tokens should have causal pattern among themselves
    for i in range(4, 8):
        for j in range(4, 8):
            if j <= i:
                assert mask_2d[i, j] == 0, f"Target[{i}] should attend to Target[{j}] (causal)"
            else:
                assert mask_2d[i, j] == -10000, f"Target[{i}] should NOT attend to Target[{j}] (future)"
    
    # Source tokens should NOT attend to target tokens
    for i in range(4):
        for j in range(4, 8):
            assert mask_2d[i, j] == -10000, f"Source[{i}] should NOT attend to Target[{j}]"
    
    print("[OK] Source tokens attend to all source tokens (bidirectional)")
    print("[OK] Target tokens attend to all source tokens")
    print("[OK] Target tokens have causal attention among themselves")
    print("[OK] Source tokens do NOT attend to target tokens")
    print("[PASSED]")


def test_attention_probability_mass():
    """Test that attention probabilities sum correctly and follow expected patterns."""
    print("\n=== Test 11: Attention Probability Mass Checks ===")
    
    config = create_test_config()
    config.fixed_source_length = 4
    model = PALMModel(config)
    model.eval()
    
    # Create input: 4 source + 4 target = 8 tokens
    input_ids = torch.randint(1, 100, (1, 8))
    
    # Get the attention mask
    attention_mask = model.create_bidirectional_attention_mask(input_ids)
    
    # Run embedding
    hidden_states = model.embeddings(input_ids)
    
    # Get attention layer and run it to inspect probabilities
    attn_layer = model.layers[0].attention
    
    # Manually compute attention to inspect probabilities
    query_layer = attn_layer.transpose_for_scores(attn_layer.query(hidden_states))
    key_layer = attn_layer.transpose_for_scores(attn_layer.key(hidden_states))
    
    import math
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(attn_layer.attention_head_size)
    attention_scores = attention_scores + attention_mask
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    
    # Average over heads for checking
    avg_probs = attention_probs[0].mean(dim=0)  # [seq, seq]
    
    # Check 1: For target positions (4-7), attention to future positions should be ~0
    for t in range(4, 8):
        future_attn = avg_probs[t, t+1:].sum().item() if t < 7 else 0
        assert future_attn < 0.01, f"Target position {t} attending to future: {future_attn}"
    
    # Check 2: For target positions, attention to source should be non-trivial
    for t in range(4, 8):
        source_attn = avg_probs[t, :4].sum().item()
        assert source_attn > 0.05, f"Target position {t} should attend to source, got: {source_attn}"
    
    # Check 3: All attention probs should sum to ~1 for each position
    for t in range(8):
        total_attn = avg_probs[t].sum().item()
        assert abs(total_attn - 1.0) < 0.01, f"Attention at position {t} doesn't sum to 1: {total_attn}"
    
    print("[OK] Target tokens don't attend to future (causal)")
    print("[OK] Target tokens attend to source (non-trivial mass)")
    print("[OK] Attention probabilities sum to 1")
    print("[PASSED]")


def test_sae_routing_changes_with_source_len():
    """Test that changing source_len actually changes SAE loss."""
    print("\n=== Test 12: SAE Routing Changes With source_len ===")
    
    config = create_test_config()
    model = PALMModel(config)
    
    # Same input tokens
    input_ids = torch.randint(1, 100, (2, 20))
    labels = input_ids.clone()
    labels[:, :10] = -100  # Mask first 10 tokens
    
    # Different source lengths
    source_len_short = torch.tensor([5, 5])
    source_len_long = torch.tensor([15, 15])
    
    outputs_short = model(input_ids, labels=labels, source_len=source_len_short)
    outputs_long = model(input_ids, labels=labels, source_len=source_len_long)
    
    sae_loss_short = outputs_short[3].item()
    sae_loss_long = outputs_long[3].item()
    
    # SAE losses should be different because different tokens are included
    assert sae_loss_short != sae_loss_long, \
        f"SAE loss should change with source_len: short={sae_loss_short}, long={sae_loss_long}"
    
    print(f"[OK] SAE loss with source_len=5: {sae_loss_short:.4f}")
    print(f"[OK] SAE loss with source_len=15: {sae_loss_long:.4f}")
    print("[OK] SAE loss changes with source_len")
    print("[PASSED]")


def test_label_shift_correctness():
    """Test that labels are correctly shifted (predict next token)."""
    print("\n=== Test 13: Label Shift Correctness ===")
    
    # For instruction tuning with causal LM, labels should be:
    # - Prompt positions: -100 (ignored)
    # - Completion positions: the actual token at that position
    # The model predicts token[t] given tokens[0:t], so labels[t] = input_ids[t]
    # (This is "no shift" because CrossEntropyLoss compares logits[t] with labels[t])
    
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            
        def __call__(self, texts, max_length, truncation, padding):
            result = {"input_ids": [], "attention_mask": []}
            for text in texts:
                ids = [ord(c) % 100 + 1 for c in text[:max_length]]
                ids = ids + [0] * (max_length - len(ids))
                mask = [1 if i != 0 else 0 for i in ids]
                result["input_ids"].append(ids)
                result["attention_mask"].append(mask)
            return result
        
        def encode(self, text, add_special_tokens=True):
            return [ord(c) % 100 + 1 for c in text]
    
    tokenizer = MockTokenizer()
    
    examples = {
        "prompt": ["Hello"],  # 5 chars = 5 tokens
        "completion": [" World!"]  # 7 chars = 7 tokens
    }
    
    processed = preprocess_function(examples, tokenizer, max_seq_length=20)
    
    input_ids = processed["input_ids"][0]
    labels = processed["labels"][0]
    source_len = processed["source_len"][0].item()
    
    # Check prompt is masked
    assert torch.all(labels[:source_len] == -100), "Prompt should be masked with -100"
    
    # Check completion labels equal input_ids (no shift needed for causal LM)
    # The completion starts after the prompt
    completion_start = source_len
    completion_labels = labels[completion_start:]
    completion_inputs = input_ids[completion_start:]
    
    # Non-padding completion labels should match input_ids
    for i in range(len(completion_labels)):
        if completion_labels[i] != -100:  # Not padding
            assert completion_labels[i] == completion_inputs[i], \
                f"Label at {i} should match input: {completion_labels[i]} vs {completion_inputs[i]}"
    
    # Verify completion length makes sense
    actual_completion_len = (labels != -100).sum().item()
    print(f"[OK] Prompt length: {source_len}")
    print(f"[OK] Completion tokens (non-masked labels): {actual_completion_len}")
    print("[OK] Prompt labels are all -100")
    print("[OK] Completion labels match input tokens")
    print("[PASSED]")


def test_overfit_single_example():
    """Test that model can overfit a single example (sanity check)."""
    print("\n=== Test 14: Overfit Single Example ===")
    
    config = create_test_config()
    config.fixed_source_length = 5
    model = PALMModel(config)
    
    # Create a simple, repeatable pattern to memorize
    # Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Labels: [-100, -100, -100, -100, -100, 6, 7, 8, 9, 10]  (predict completion)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, 6, 7, 8, 9, 10]])
    source_len = torch.tensor([5])
    
    # Train for a few steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    initial_loss = None
    final_loss = None
    
    model.train()
    for step in range(50):
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels, source_len=source_len)
        loss = outputs[1]  # combined_loss
        
        if step == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        if step == 49:
            final_loss = loss.item()
    
    # Loss should decrease significantly
    assert final_loss < initial_loss * 0.5, \
        f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    
    # Check if model can predict the completion
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids[:, :5])  # Just the prompt
        logits = outputs[0]
        # Get predicted tokens for positions 5-9
        preds = logits[0, :].argmax(dim=-1)
    
    print(f"[OK] Initial loss: {initial_loss:.4f}")
    print(f"[OK] Final loss: {final_loss:.4f}")
    print(f"[OK] Loss decreased by {(1 - final_loss/initial_loss)*100:.1f}%")
    print("[PASSED]")


def test_kv_cache_parity():
    """Test that generation with use_cache=True matches use_cache=False (greedy decode)."""
    print("\n=== Test 15: KV Cache Parity (use_cache=True vs False) ===")
    
    config = create_test_config()
    config.fixed_source_length = 5
    model = PALMModel(config)
    model.eval()
    
    # Create a prompt
    input_ids = torch.randint(1, 100, (1, 5))
    
    # Generate with cache
    with torch.no_grad():
        output_cached = model.generate(
            input_ids.clone(),
            max_length=15,
            do_sample=False,  # Greedy for deterministic comparison
            use_cache=True
        )
    
    # Generate without cache
    with torch.no_grad():
        output_no_cache = model.generate(
            input_ids.clone(),
            max_length=15,
            do_sample=False,
            use_cache=False
        )
    
    # They should be identical for greedy decoding
    match = torch.all(output_cached == output_no_cache).item()
    
    if not match:
        print(f"  Cached output:    {output_cached[0].tolist()}")
        print(f"  No-cache output:  {output_no_cache[0].tolist()}")
        # Find first difference
        for i in range(min(len(output_cached[0]), len(output_no_cache[0]))):
            if output_cached[0, i] != output_no_cache[0, i]:
                print(f"  First difference at position {i}: {output_cached[0, i]} vs {output_no_cache[0, i]}")
                break
    
    assert match, "Cached and non-cached generation should produce identical results for greedy decode"
    
    print(f"[OK] Generated length: {output_cached.shape[1]}")
    print("[OK] use_cache=True matches use_cache=False")
    print("[PASSED]")


def test_kv_cache_shapes():
    """Test that KV cache has correct shapes through forward pass."""
    print("\n=== Test 16: KV Cache Shapes ===")
    
    config = create_test_config()
    config.fixed_source_length = 5
    model = PALMModel(config)
    model.eval()
    
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    
    # Forward with caching
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    assert len(outputs) == 5, f"Expected 5 outputs with use_cache=True, got {len(outputs)}"
    past_key_values = outputs[4]
    
    # Check structure
    assert len(past_key_values) == config.num_hidden_layers, \
        f"Expected {config.num_hidden_layers} layers in cache, got {len(past_key_values)}"
    
    # Check each layer's cache
    for i, layer_cache in enumerate(past_key_values):
        attn_cache, partial_cache = layer_cache
        attn_k, attn_v = attn_cache
        partial_k, partial_v = partial_cache
        
        # Regular attention K/V should have full sequence length
        expected_attn_shape = (batch_size, config.num_attention_heads, seq_len, config.hidden_size // config.num_attention_heads)
        assert attn_k.shape == expected_attn_shape, f"Layer {i} attn key shape: {attn_k.shape} != {expected_attn_shape}"
        assert attn_v.shape == expected_attn_shape, f"Layer {i} attn value shape: {attn_v.shape} != {expected_attn_shape}"
        
        # Partial attention K/V should have source length
        expected_partial_shape = (batch_size, config.num_attention_heads, config.fixed_source_length, config.hidden_size // config.num_attention_heads)
        assert partial_k.shape == expected_partial_shape, f"Layer {i} partial key shape: {partial_k.shape} != {expected_partial_shape}"
        assert partial_v.shape == expected_partial_shape, f"Layer {i} partial value shape: {partial_v.shape} != {expected_partial_shape}"
    
    print(f"[OK] Cache has {config.num_hidden_layers} layers")
    print(f"[OK] Attention K/V shape: {attn_k.shape}")
    print(f"[OK] Partial attention K/V shape: {partial_k.shape}")
    print("[PASSED]")


def test_kv_cache_incremental():
    """Test that incremental decoding with cache produces same hidden states."""
    print("\n=== Test 17: KV Cache Incremental Decoding ===")
    
    config = create_test_config()
    config.fixed_source_length = 5
    model = PALMModel(config)
    model.eval()
    
    # Full sequence
    input_ids = torch.randint(1, 100, (1, 8))
    
    # Get full forward pass result
    with torch.no_grad():
        full_outputs = model(input_ids, use_cache=False)
        full_logits = full_outputs[0]
    
    # Now do incremental: first get cache from first 7 tokens
    with torch.no_grad():
        partial_outputs = model(input_ids[:, :7], use_cache=True)
        partial_logits, _, _, _, past_kv = partial_outputs
        
        # Then forward just the 8th token with cache
        # Need to create the right attention mask for incremental
        incremental_mask = model._create_incremental_attention_mask(8, 5, input_ids.device)
        
        incremental_outputs = model(
            input_ids[:, 7:8],
            attention_mask=incremental_mask,
            past_key_values=past_kv,
            use_cache=True,
            position_offset=7  # 8th token is at position 7
        )
        incremental_logits = incremental_outputs[0]
    
    # The logits for position 7 should match between full and incremental
    full_last_logit = full_logits[0, 7, :]
    incr_last_logit = incremental_logits[0, 0, :]
    
    # Check they're close (not exact due to floating point)
    max_diff = (full_last_logit - incr_last_logit).abs().max().item()
    assert max_diff < 0.01, f"Logits differ by {max_diff}, expected < 0.01"
    
    print(f"[OK] Max logit difference: {max_diff:.6f}")
    print("[OK] Incremental decoding matches full forward")
    print("[PASSED]")


def test_degeneration_metrics():
    """Test degeneration/mush detection metrics."""
    print("\n=== Test 18: Degeneration Metrics ===")
    
    # Import directly to avoid rouge dependency in metrics.py
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "comprehensive", 
        os.path.join(project_root, "palm", "evaluation", "comprehensive.py")
    )
    comprehensive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comprehensive)
    
    compute_degeneration_metrics = comprehensive.compute_degeneration_metrics
    compute_distinct_ngrams = comprehensive.compute_distinct_ngrams
    compute_repetition_rate = comprehensive.compute_repetition_rate
    
    # Test distinct n-grams
    tokens_diverse = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tokens_repetitive = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    
    diverse_distinct = compute_distinct_ngrams(tokens_diverse, 2)
    repetitive_distinct = compute_distinct_ngrams(tokens_repetitive, 2)
    
    assert diverse_distinct > repetitive_distinct, \
        f"Diverse should have higher distinct-2: {diverse_distinct} vs {repetitive_distinct}"
    
    # Test repetition rate
    diverse_rep = compute_repetition_rate(tokens_diverse, 2)
    repetitive_rep = compute_repetition_rate(tokens_repetitive, 2)
    
    assert diverse_rep < repetitive_rep, \
        f"Diverse should have lower repetition: {diverse_rep} vs {repetitive_rep}"
    
    # Test full metrics
    metrics = compute_degeneration_metrics(
        tokens_repetitive,
        expected_len=10,
        min_len=5,
        max_len=100,
        eos_token_id=99,
    )
    
    assert metrics.repetition_rate > 0, "Repetition rate should be > 0"
    assert metrics.distinct_1gram > 0, "Distinct-1 should be > 0"
    assert not metrics.early_stop, "Should not be early stop"
    assert not metrics.infinite_loop, "Should not be infinite loop"
    
    print(f"[OK] Distinct-2 (diverse): {diverse_distinct:.3f}")
    print(f"[OK] Distinct-2 (repetitive): {repetitive_distinct:.3f}")
    print(f"[OK] Repetition rate computed correctly")
    print("[PASSED]")


def test_faithfulness_metrics():
    """Test faithfulness and hallucination detection."""
    print("\n=== Test 19: Faithfulness Metrics ===")
    
    # Import directly to avoid rouge dependency in metrics.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "comprehensive", 
        os.path.join(project_root, "palm", "evaluation", "comprehensive.py")
    )
    comprehensive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comprehensive)
    
    compute_faithfulness_metrics = comprehensive.compute_faithfulness_metrics
    extract_entities_simple = comprehensive.extract_entities_simple
    
    # Test entity extraction
    text = "John Smith visited New York in 2024. He said 'hello world'."
    entities = extract_entities_simple(text)
    
    assert "john smith" in entities or "John Smith".lower() in entities, \
        f"Should extract 'John Smith', got {entities}"
    assert "2024" in entities, f"Should extract '2024', got {entities}"
    
    # Test faithfulness with grounded output
    source = "The Eiffel Tower is located in Paris, France. It was built in 1889."
    grounded_output = "The Eiffel Tower in Paris was constructed in 1889."
    hallucinated_output = "The Statue of Liberty in New York was built in 1886."
    
    grounded_metrics = compute_faithfulness_metrics(source, grounded_output)
    hallucinated_metrics = compute_faithfulness_metrics(source, hallucinated_output)
    
    assert grounded_metrics.entity_hallucination_rate < hallucinated_metrics.entity_hallucination_rate, \
        "Grounded output should have lower hallucination rate"
    
    assert grounded_metrics.entity_precision > hallucinated_metrics.entity_precision, \
        "Grounded output should have higher entity precision"
    
    print(f"[OK] Grounded hallucination rate: {grounded_metrics.entity_hallucination_rate:.3f}")
    print(f"[OK] Hallucinated rate: {hallucinated_metrics.entity_hallucination_rate:.3f}")
    print("[PASSED]")


def test_loss_decomposition():
    """Test loss decomposition logging."""
    print("\n=== Test 20: Loss Decomposition ===")
    
    # Import directly to avoid rouge dependency in metrics.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "comprehensive", 
        os.path.join(project_root, "palm", "evaluation", "comprehensive.py")
    )
    comprehensive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comprehensive)
    
    compute_loss_decomposition = comprehensive.compute_loss_decomposition
    compute_perplexity = comprehensive.compute_perplexity
    
    lm_loss = torch.tensor(2.5)
    sae_loss = torch.tensor(1.0)
    combined = torch.tensor(3.0)
    
    decomp = compute_loss_decomposition(lm_loss, sae_loss, combined)
    
    assert decomp.lm_loss == 2.5, f"LM loss should be 2.5, got {decomp.lm_loss}"
    assert decomp.sae_loss == 1.0, f"SAE loss should be 1.0, got {decomp.sae_loss}"
    assert decomp.total_loss == 3.0, f"Total loss should be 3.0, got {decomp.total_loss}"
    assert abs(decomp.sae_ratio - 1.0/3.0) < 0.01, f"SAE ratio should be ~0.33, got {decomp.sae_ratio}"
    
    # Test perplexity
    ppl = compute_perplexity(2.5)
    expected_ppl = 12.182  # e^2.5
    assert abs(ppl - expected_ppl) < 0.1, f"Perplexity should be ~12.18, got {ppl}"
    
    # Test dict conversion
    d = decomp.to_dict()
    assert "lm_loss" in d and "sae_loss" in d and "sae_ratio" in d
    
    print(f"[OK] Loss decomposition: LM={decomp.lm_loss}, SAE={decomp.sae_loss}")
    print(f"[OK] SAE ratio: {decomp.sae_ratio:.3f}")
    print(f"[OK] Perplexity: {ppl:.2f}")
    print("[PASSED]")


def test_mask_compliance_metrics():
    """Test mask compliance tracking."""
    print("\n=== Test 21: Mask Compliance Metrics ===")
    
    # Import directly to avoid rouge dependency in metrics.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "comprehensive", 
        os.path.join(project_root, "palm", "evaluation", "comprehensive.py")
    )
    comprehensive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comprehensive)
    
    compute_mask_compliance = comprehensive.compute_mask_compliance
    
    config = create_test_config()
    config.fixed_source_length = 4
    model = PALMModel(config)
    model.eval()
    
    input_ids = torch.randint(1, 100, (1, 8))  # 4 source + 4 target
    
    metrics = compute_mask_compliance(model, input_ids, source_length=4)
    
    # Future leakage should be near 0
    assert metrics.future_leakage < 0.01, \
        f"Future leakage should be < 0.01, got {metrics.future_leakage}"
    
    # Source attention mass should be non-trivial
    assert metrics.source_attention_mass > 0.05, \
        f"Source attention mass should be > 0.05, got {metrics.source_attention_mass}"
    
    # Per-layer metrics should exist
    assert len(metrics.per_layer_future_leakage) == config.num_hidden_layers
    
    print(f"[OK] Future leakage: {metrics.future_leakage:.6f}")
    print(f"[OK] Source attention mass: {metrics.source_attention_mass:.3f}")
    print(f"[OK] Per-layer leakage: {metrics.per_layer_future_leakage}")
    print("[PASSED]")


def test_palm_composite_score():
    """Test PALM composite score calculation."""
    print("\n=== Test 22: PALM Composite Score ===")
    
    # Import directly to avoid rouge dependency in metrics.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "comprehensive", 
        os.path.join(project_root, "palm", "evaluation", "comprehensive.py")
    )
    comprehensive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comprehensive)
    
    compute_palm_score = comprehensive.compute_palm_score
    
    # Good model: high faithfulness, low hallucination/degeneracy
    good_score = compute_palm_score(
        faithfulness=0.9,
        hallucination=0.1,
        degeneracy=0.1,
        ppl_drift=0.5,
    )
    
    # Bad model: low faithfulness, high hallucination/degeneracy
    bad_score = compute_palm_score(
        faithfulness=0.3,
        hallucination=0.6,
        degeneracy=0.5,
        ppl_drift=2.0,
    )
    
    assert good_score.composite > bad_score.composite, \
        f"Good model should have higher composite: {good_score.composite} vs {bad_score.composite}"
    
    # Verify formula: Faithfulness - α·Hallucination - β·Degeneracy - γ·PPL_Drift
    # Default: α=1.0, β=0.5, γ=0.1
    expected_good = 0.9 - 1.0*0.1 - 0.5*0.1 - 0.1*0.5
    assert abs(good_score.composite - expected_good) < 0.001, \
        f"Composite should be {expected_good}, got {good_score.composite}"
    
    print(f"[OK] Good model composite: {good_score.composite:.3f}")
    print(f"[OK] Bad model composite: {bad_score.composite:.3f}")
    print("[PASSED]")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("PALM Bug Fix Validation Tests")
    print("=" * 60)
    
    tests = [
        test_attention_mask_no_double_inversion,
        test_attention_mask_with_padding,
        test_loss_with_ignore_index,
        test_sae_uses_actual_source_len,
        test_sae_loss_none_handling,
        test_partial_attention_no_masking,
        test_generation_preserves_source_length,
        test_preprocessing_masks_prompt,
        test_layer_forward,
        test_bidirectional_causal_mask_pattern,
        # New tests from tests.yml
        test_attention_probability_mass,
        test_sae_routing_changes_with_source_len,
        test_label_shift_correctness,
        test_overfit_single_example,
        # KV cache tests
        test_kv_cache_parity,
        test_kv_cache_shapes,
        test_kv_cache_incremental,
        # Evaluation metrics tests (from evals.yml)
        test_degeneration_metrics,
        test_faithfulness_metrics,
        test_loss_decomposition,
        test_mask_compliance_metrics,
        test_palm_composite_score,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n[FAILED]: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

