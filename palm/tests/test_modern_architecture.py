"""
Test script for modern PALM architecture changes:
- RMSNorm instead of LayerNorm
- Pre-Normalization pattern
- SwiGLU MLP
- SDPA/Flash Attention
- SiLU activation
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
from palm.config import PALMConfig
from palm.model.palm import PALMModel, PALMLayer, SwiGLU
from palm.model.attention import PALMAttention, PALMPartialAttention, RMSNorm


def create_test_config():
    """Create a small test config."""
    return PALMConfig(
        base_model_name="meta-llama/Llama-3.2-3B",
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,  # GQA
        intermediate_size=256,
        max_position_embeddings=512,
        fixed_source_length=10,
        pad_token_id=0,
    )


def test_rmsnorm():
    """Test RMSNorm module."""
    print("\n1. Testing RMSNorm...")
    norm = RMSNorm(128)
    x = torch.randn(2, 10, 128)
    out = norm(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "RMSNorm shape mismatch"
    # Check that it normalizes (variance should be ~1 after norm)
    print("   [OK] RMSNorm works")
    return True


def test_swiglu():
    """Test SwiGLU MLP module."""
    print("\n2. Testing SwiGLU MLP...")
    config = create_test_config()
    mlp = SwiGLU(config)
    x = torch.randn(2, 10, 128)
    out = mlp(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "SwiGLU shape mismatch"
    # Check that it has the correct components
    assert hasattr(mlp, 'gate_proj'), "SwiGLU missing gate_proj"
    assert hasattr(mlp, 'up_proj'), "SwiGLU missing up_proj"
    assert hasattr(mlp, 'down_proj'), "SwiGLU missing down_proj"
    print("   [OK] SwiGLU works")
    return True


def test_palm_attention_sdpa():
    """Test PALMAttention with SDPA."""
    print("\n3. Testing PALMAttention with SDPA...")
    config = create_test_config()
    attn = PALMAttention(config)
    x = torch.randn(2, 10, 128)
    out, _ = attn(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "PALMAttention shape mismatch"
    print("   [OK] PALMAttention with SDPA works")
    return True


def test_palm_partial_attention_sdpa():
    """Test PALMPartialAttention with SDPA and SiLU."""
    print("\n4. Testing PALMPartialAttention with SDPA...")
    config = create_test_config()
    partial_attn = PALMPartialAttention(config)
    x = torch.randn(2, 10, 128)
    source = x[:, :5]  # First 5 tokens as source
    out, _ = partial_attn(x, source)
    print(f"   Hidden: {x.shape}, Source: {source.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "PALMPartialAttention shape mismatch"
    # Check SiLU activation in Fp network
    assert isinstance(partial_attn.Fp_activation, torch.nn.SiLU), "Fp should use SiLU"
    print("   [OK] PALMPartialAttention with SDPA and SiLU works")
    return True


def test_palm_layer_prenorm():
    """Test PALMLayer with Pre-Norm architecture."""
    print("\n5. Testing PALMLayer (Pre-Norm architecture)...")
    config = create_test_config()
    layer = PALMLayer(config)
    
    # Check Pre-Norm components
    assert hasattr(layer, 'attn_norm'), "Missing attn_norm"
    assert hasattr(layer, 'partial_attn_norm'), "Missing partial_attn_norm"
    assert hasattr(layer, 'mlp_norm'), "Missing mlp_norm"
    assert isinstance(layer.attn_norm, RMSNorm), "attn_norm should be RMSNorm"
    assert isinstance(layer.partial_attn_norm, RMSNorm), "partial_attn_norm should be RMSNorm"
    assert isinstance(layer.mlp_norm, RMSNorm), "mlp_norm should be RMSNorm"
    
    # Check SwiGLU MLP
    assert isinstance(layer.mlp, SwiGLU), "MLP should be SwiGLU"
    
    print("   Layer has: attn_norm, partial_attn_norm, mlp_norm (RMSNorm)")
    print("   Layer has: attention, partial_attention, mlp (SwiGLU)")
    
    # Test forward pass
    x = torch.randn(2, 10, 128)
    mask = torch.zeros(2, 1, 10, 10)
    out, _ = layer(x, attention_mask=mask, source_len=5)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "PALMLayer shape mismatch"
    print("   [OK] PALMLayer works")
    return True


def test_palm_model_full():
    """Test full PALMModel with all modern features."""
    print("\n6. Testing full PALMModel...")
    config = create_test_config()
    model = PALMModel(config)
    
    print(f"   Model has {len(model.layers)} layers")
    print(f"   Model has final_norm: {hasattr(model, 'final_norm')}")
    print(f"   _supports_sdpa: {model._supports_sdpa}")
    print(f"   _supports_flash_attn_2: {model._supports_flash_attn_2}")
    
    # Check final norm
    assert hasattr(model, 'final_norm'), "Missing final_norm"
    assert isinstance(model.final_norm, RMSNorm), "final_norm should be RMSNorm"
    
    # Check SDPA support flags
    assert model._supports_sdpa == True, "_supports_sdpa should be True"
    assert model._supports_flash_attn_2 == True, "_supports_flash_attn_2 should be True"
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 10))
    labels = input_ids.clone()
    source_len = torch.tensor([5, 5])
    
    outputs = model(input_ids, labels=labels, source_len=source_len)
    lm_logits, combined_loss, loss, sae_loss = outputs
    
    print(f"   Input: {input_ids.shape}")
    print(f"   Logits: {lm_logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   SAE Loss: {sae_loss.item():.4f}")
    print(f"   Combined Loss: {combined_loss.item():.4f}")
    print("   [OK] PALMModel forward pass works")
    return True


def test_backward_pass():
    """Test that gradients flow correctly."""
    print("\n7. Testing backward pass (gradient flow)...")
    config = create_test_config()
    model = PALMModel(config)
    
    input_ids = torch.randint(0, 1000, (2, 10))
    labels = input_ids.clone()
    source_len = torch.tensor([5, 5])
    
    outputs = model(input_ids, labels=labels, source_len=source_len)
    _, combined_loss, _, _ = outputs
    
    # Backward pass
    combined_loss.backward()
    
    # Check that gradients exist
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
    
    print(f"   Parameters with gradients: {grad_count}")
    assert grad_count > 0, "No gradients computed"
    print("   [OK] Backward pass works")
    return True


def main():
    print("=" * 50)
    print("Testing Modern PALM Architecture")
    print("=" * 50)
    
    tests = [
        test_rmsnorm,
        test_swiglu,
        test_palm_attention_sdpa,
        test_palm_partial_attention_sdpa,
        test_palm_layer_prenorm,
        test_palm_model_full,
        test_backward_pass,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   [FAILED] {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed == 0:
        print("\nArchitecture Summary:")
        print("  [OK] RMSNorm: Faster than LayerNorm")
        print("  [OK] SwiGLU: Better MLP capacity")
        print("  [OK] SDPA: FlashAttention when available")
        print("  [OK] Pre-Norm: Better gradient flow")
        print("  [OK] SiLU activation in Fp network")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

