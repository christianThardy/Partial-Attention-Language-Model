"""
Quick test for RoPE and tied SAE head changes.
Run with: python palm/tests/test_rope_changes.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch


def test_rope_and_tied_sae():
    """Test RoPE implementation and tied SAE head."""
    print("=== Testing RoPE and Tied SAE Head Changes ===\n")
    
    # Test config creation
    from palm.config import PALMConfig
    config = PALMConfig(
        base_model_name='meta-llama/Llama-3.2-3B',
        vocab_size=1000,  # Small for testing
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,  # GQA
        intermediate_size=256,
        max_position_embeddings=512,
        fixed_source_length=10,
    )
    print(f'1. Config created:')
    print(f'   rope_theta: {config.rope_theta}')
    print(f'   tie_sae_head: {config.tie_sae_head}')
    print(f'   [OK]')

    # Test model creation
    from palm.model.palm import PALMModel
    model = PALMModel(config)
    print(f'\n2. Model created:')
    print(f'   sae_head is None: {model.sae_head is None}')
    print(f'   Using tied SAE: {model._tie_sae_head}')
    print(f'   [OK]')
    
    # Test RoPE is present in attention
    print(f'\n3. RoPE in attention layers:')
    for i, layer in enumerate(model.layers):
        has_rope_attn = hasattr(layer.attention, 'rotary_emb')
        has_rope_partial = hasattr(layer.partial_attention, 'rotary_emb')
        print(f'   Layer {i}: attention RoPE={has_rope_attn}, partial_attention RoPE={has_rope_partial}')
        assert has_rope_attn, f"Layer {i} attention missing rotary_emb"
        assert has_rope_partial, f"Layer {i} partial_attention missing rotary_emb"
    print(f'   [OK]')

    # Test forward pass with RoPE
    print(f'\n4. Forward pass with RoPE:')
    input_ids = torch.randint(0, 1000, (2, 20))
    source_len = torch.tensor([10, 10])
    labels = input_ids.clone()
    labels[:, :10] = -100  # Mask source from loss

    outputs = model(input_ids, labels=labels, source_len=source_len)
    lm_logits, combined_loss, loss, sae_loss = outputs
    print(f'   lm_logits shape: {lm_logits.shape}')
    print(f'   combined_loss: {combined_loss.item():.4f}')
    print(f'   lm_loss: {loss.item():.4f}')
    print(f'   sae_loss: {sae_loss.item():.4f}')
    print(f'   [OK]')

    # Test embeddings return position_ids with SPE
    print(f'\n5. Embeddings with SPE (Separate Positional Encoding):')
    embeddings, position_ids = model.embeddings(input_ids, source_len=source_len)
    print(f'   embeddings shape: {embeddings.shape}')
    print(f'   position_ids shape: {position_ids.shape}')
    print(f'   position_ids[0]: {position_ids[0].tolist()}')
    # Verify SPE: source 0-9, target resets to 0-9
    expected_pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert position_ids[0].tolist() == expected_pos, f"SPE failed: {position_ids[0].tolist()} != {expected_pos}"
    print(f'   SPE verified: source 0-9, target resets to 0-9')
    print(f'   [OK]')
    
    # Test tied SAE head
    print(f'\n6. Tied SAE head verification:')
    sae_head = model.get_sae_head()
    assert sae_head is model.lm_head, "SAE head should be same as lm_head"
    print(f'   get_sae_head() returns lm_head: True')
    print(f'   [OK]')
    
    # Test generation
    print(f'\n7. Generation with RoPE:')
    model.eval()
    with torch.no_grad():
        generated = model.generate(input_ids[:1, :10], max_length=15, do_sample=False)
    print(f'   Input length: 10, Output length: {generated.shape[1]}')
    print(f'   [OK]')
    
    print(f'\n{"="*50}')
    print(f'ALL TESTS PASSED!')
    print(f'{"="*50}')


if __name__ == "__main__":
    test_rope_and_tied_sae()

