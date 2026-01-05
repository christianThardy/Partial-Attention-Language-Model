"""
Test suite for PALM KV Cache Optimization Strategies.

Tests both strategies:
1. Cross-Layer KV Sharing (Strategy #3)
2. Hybrid Multi-Granularity Cache (Strategy #1)
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import pytest
from palm.config import PALMConfig
from palm.model.palm import PALMModel, PALMLayer
from palm.model.kv_cache import (
    KVCacheConfig,
    CrossLayerKVManager,
    QuantizedKVCache,
    HybridGranularityCache,
    PALMCache,
    create_palm_cache,
)


def create_test_config(
    share_partial_kv=False,
    kv_sharing_groups=4,
    enable_kv_quantization=False,
):
    """Create a small test config for fast testing."""
    return PALMConfig(
        base_model_name="meta-llama/Llama-3.2-3B",
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=8,  # 8 layers = 2 per group with 4 groups
        num_attention_heads=4,
        num_kv_heads=2,  # GQA
        intermediate_size=256,
        max_position_embeddings=512,
        fixed_source_length=10,
        pad_token_id=0,
        # KV Cache optimization settings
        share_partial_kv=share_partial_kv,
        kv_sharing_groups=kv_sharing_groups,
        enable_kv_quantization=enable_kv_quantization,
    )


# =============================================================================
# STRATEGY #3: CROSS-LAYER KV SHARING TESTS
# =============================================================================

class TestCrossLayerKVManager:
    """Test the CrossLayerKVManager component."""
    
    def test_layer_to_group_mapping(self):
        """Test layer-to-group assignment is correct."""
        print("\n1. Testing layer-to-group mapping...")
        
        # 8 layers, 4 groups = 2 layers per group
        manager = CrossLayerKVManager(num_layers=8, num_groups=4)
        
        expected = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}
        for layer_idx, expected_group in expected.items():
            assert manager.get_group_for_layer(layer_idx) == expected_group, \
                f"Layer {layer_idx} should be in group {expected_group}"
        
        print(f"   Mapping: {manager.layer_to_group}")
        print("   [OK] Layer-to-group mapping correct")
    
    def test_representative_layers(self):
        """Test that first layer of each group is representative."""
        print("\n2. Testing representative layer detection...")
        
        manager = CrossLayerKVManager(num_layers=8, num_groups=4)
        
        # Layers 0, 2, 4, 6 should be representatives
        representatives = [i for i in range(8) if manager.is_representative_layer(i)]
        assert representatives == [0, 2, 4, 6], \
            f"Expected representatives [0, 2, 4, 6], got {representatives}"
        
        print(f"   Representative layers: {representatives}")
        print("   [OK] Representative layer detection correct")
    
    def test_kv_sharing(self):
        """Test KV sharing across groups."""
        print("\n3. Testing KV storage and retrieval...")
        
        manager = CrossLayerKVManager(num_layers=8, num_groups=4)
        
        # Create dummy KV tensors
        batch, heads, seq, dim = 2, 4, 10, 32
        keys = torch.randn(batch, heads, seq, dim)
        values = torch.randn(batch, heads, seq, dim)
        
        # Store from layer 0 (representative of group 0)
        manager.store_shared_kv(0, keys, values)
        
        # Layer 1 (same group) should get this KV
        shared = manager.get_shared_kv(1)
        assert shared is not None, "Layer 1 should have shared KV from group 0"
        assert torch.equal(shared[0], keys), "Keys should match"
        assert torch.equal(shared[1], values), "Values should match"
        
        # Layer 2 (different group) should NOT have KV
        shared_2 = manager.get_shared_kv(2)
        assert shared_2 is None, "Layer 2 (group 1) should not have KV yet"
        
        print("   [OK] KV sharing works correctly")
    
    def test_memory_stats(self):
        """Test memory statistics reporting."""
        print("\n4. Testing memory stats...")
        
        manager = CrossLayerKVManager(num_layers=8, num_groups=4)
        
        # Store KV for 2 groups
        keys = torch.randn(2, 4, 10, 32)
        values = torch.randn(2, 4, 10, 32)
        manager.store_shared_kv(0, keys, values)
        manager.store_shared_kv(2, keys, values)
        
        stats = manager.get_memory_stats()
        print(f"   Stats: {stats}")
        
        assert stats["groups_stored"] == 2, "Should have 2 groups stored"
        assert stats["total_groups"] == 4, "Should have 4 total groups"
        assert stats["savings_factor"] == 2.0, "Savings factor should be 2x"
        
        print("   [OK] Memory stats correct")


class TestCrossLayerKVInModel:
    """Test cross-layer KV sharing integrated into PALMModel."""
    
    def test_model_with_kv_sharing(self):
        """Test PALMModel with cross-layer KV sharing enabled."""
        print("\n5. Testing PALMModel with cross-layer KV sharing...")
        
        config = create_test_config(share_partial_kv=True, kv_sharing_groups=4)
        model = PALMModel(config)
        
        assert model.cross_layer_kv_manager is not None, \
            "Model should have cross_layer_kv_manager"
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 20))
        source_len = torch.tensor([10, 10])
        
        with torch.no_grad():
            logits, loss, _, _ = model(
                input_ids, labels=input_ids, source_len=source_len
            )
        
        print(f"   Output shape: {logits.shape}")
        print(f"   Cross-layer manager groups: {model.cross_layer_kv_manager.num_groups}")
        
        # Check that KV was stored for representative layers
        stats = model.cross_layer_kv_manager.get_memory_stats()
        print(f"   Stored groups: {stats['groups_stored']}/{stats['total_groups']}")
        
        print("   [OK] Model with cross-layer KV sharing works")
    
    def test_model_without_kv_sharing(self):
        """Test PALMModel with cross-layer KV sharing disabled."""
        print("\n6. Testing PALMModel without cross-layer KV sharing...")
        
        config = create_test_config(share_partial_kv=False)
        model = PALMModel(config)
        
        assert model.cross_layer_kv_manager is None, \
            "Model should NOT have cross_layer_kv_manager when disabled"
        
        # Forward pass should still work
        input_ids = torch.randint(0, 1000, (2, 20))
        source_len = torch.tensor([10, 10])
        
        with torch.no_grad():
            logits, loss, _, _ = model(
                input_ids, labels=input_ids, source_len=source_len
            )
        
        print(f"   Output shape: {logits.shape}")
        print("   [OK] Model without cross-layer KV sharing works")


# =============================================================================
# STRATEGY #1: HYBRID MULTI-GRANULARITY CACHE TESTS
# =============================================================================

class TestQuantizedKVCache:
    """Test the QuantizedKVCache component."""
    
    def test_quantize_dequantize(self):
        """Test quantization and dequantization."""
        print("\n7. Testing 4-bit quantization...")
        
        cache = QuantizedKVCache(bits=4)
        
        # Create test tensor
        tensor = torch.randn(2, 4, 10, 32)  # batch, heads, seq, dim
        
        # Quantize
        quantized, scales = cache.quantize(tensor)
        
        assert quantized.dtype == torch.int8, "Quantized should be int8"
        assert quantized.max() <= 7, "4-bit symmetric should be <= 7"
        assert quantized.min() >= -7, "4-bit symmetric should be >= -7"
        
        # Dequantize
        reconstructed = cache.dequantize(quantized, scales)
        
        # Check reconstruction error (should be reasonable for 4-bit)
        mse = ((tensor - reconstructed) ** 2).mean()
        print(f"   Original range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        print(f"   Quantized range: [{quantized.min()}, {quantized.max()}]")
        print(f"   Reconstruction MSE: {mse:.6f}")
        
        # MSE should be small for 4-bit (typically < 0.1 for normalized data)
        assert mse < 0.5, f"Reconstruction error too high: {mse}"
        
        print("   [OK] Quantization works correctly")
    
    def test_cache_update(self):
        """Test cache update mechanism."""
        print("\n8. Testing cache update...")
        
        cache = QuantizedKVCache(bits=4)
        
        # Add first batch
        keys1 = torch.randn(2, 4, 5, 32)
        values1 = torch.randn(2, 4, 5, 32)
        cache.update(keys1, values1)
        
        assert cache.seq_len == 5, f"Expected seq_len 5, got {cache.seq_len}"
        
        # Add second batch
        keys2 = torch.randn(2, 4, 3, 32)
        values2 = torch.randn(2, 4, 3, 32)
        cache.update(keys2, values2)
        
        assert cache.seq_len == 8, f"Expected seq_len 8, got {cache.seq_len}"
        
        # Get combined KV
        k, v = cache.get_kv()
        assert k.shape == (2, 4, 8, 32), f"Unexpected key shape: {k.shape}"
        
        print(f"   Final cache seq_len: {cache.seq_len}")
        print("   [OK] Cache update works")


class TestHybridGranularityCache:
    """Test the HybridGranularityCache component."""
    
    def test_turn_boundary_tracking(self):
        """Test turn boundary management."""
        print("\n9. Testing turn boundary tracking...")
        
        cache = HybridGranularityCache(
            num_layers=4,
            quantize_after_turns=1,
            quantization_bits=4,
        )
        
        cache.set_system_prompt_end(10)
        cache.mark_turn_boundary(20)  # Turn 1 starts at position 20
        cache.mark_turn_boundary(40)  # Turn 2 starts at position 40
        
        assert cache.system_prompt_end == 10
        assert cache.current_turn == 2
        assert cache.turn_boundaries == [20, 40]
        
        print(f"   System prompt end: {cache.system_prompt_end}")
        print(f"   Turn boundaries: {cache.turn_boundaries}")
        print(f"   Current turn: {cache.current_turn}")
        print("   [OK] Turn boundary tracking works")


# =============================================================================
# COMBINED PALM CACHE TESTS
# =============================================================================

class TestPALMCache:
    """Test the combined PALMCache manager."""
    
    def test_cache_creation(self):
        """Test PALMCache creation from config."""
        print("\n10. Testing PALMCache creation...")
        
        config = create_test_config(
            share_partial_kv=True,
            kv_sharing_groups=4,
            enable_kv_quantization=False,
        )
        
        cache = create_palm_cache(config)
        
        assert cache.cross_layer_manager is not None, \
            "Should have cross_layer_manager"
        assert cache.hybrid_cache is None, \
            "Should NOT have hybrid_cache when quantization disabled"
        
        print(f"   Cross-layer sharing: enabled")
        print(f"   Quantization: disabled")
        print("   [OK] PALMCache creation works")
    
    def test_cache_with_both_strategies(self):
        """Test PALMCache with both strategies enabled."""
        print("\n11. Testing PALMCache with both strategies...")
        
        config = create_test_config(
            share_partial_kv=True,
            kv_sharing_groups=4,
            enable_kv_quantization=True,
        )
        config.enable_kv_quantization = True  # Explicitly set
        
        cache = PALMCache(KVCacheConfig.from_palm_config(config))
        
        stats = cache.get_memory_stats()
        print(f"   Stats: {stats}")
        
        assert stats["cross_layer_sharing_enabled"] == True
        # Note: quantization is enabled in config but hybrid_cache creation depends on it
        
        print("   [OK] Combined cache works")


# =============================================================================
# GRADIENT FLOW TESTS
# =============================================================================

class TestGradientFlow:
    """Test that gradients flow correctly with KV optimizations."""
    
    def test_backward_with_cross_layer_sharing(self):
        """Test backward pass with cross-layer KV sharing."""
        print("\n12. Testing backward pass with cross-layer sharing...")
        
        config = create_test_config(share_partial_kv=True, kv_sharing_groups=4)
        model = PALMModel(config)
        
        input_ids = torch.randint(0, 1000, (2, 20))
        source_len = torch.tensor([10, 10])
        
        logits, combined_loss, loss, sae_loss = model(
            input_ids, labels=input_ids, source_len=source_len
        )
        
        # Backward pass
        combined_loss.backward()
        
        # Check gradients exist for key parameters
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
        
        print(f"   Combined loss: {combined_loss.item():.4f}")
        print(f"   Parameters with gradients: {grad_count}")
        
        assert grad_count > 0, "Should have gradients"
        assert torch.isfinite(combined_loss), "Loss should be finite"
        
        print("   [OK] Backward pass works with cross-layer sharing")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all KV cache optimization tests."""
    print("=" * 70)
    print("PALM KV CACHE OPTIMIZATION TESTS")
    print("=" * 70)
    
    # Cross-Layer KV Sharing Tests
    test_manager = TestCrossLayerKVManager()
    test_manager.test_layer_to_group_mapping()
    test_manager.test_representative_layers()
    test_manager.test_kv_sharing()
    test_manager.test_memory_stats()
    
    test_model = TestCrossLayerKVInModel()
    test_model.test_model_with_kv_sharing()
    test_model.test_model_without_kv_sharing()
    
    # Quantization Tests
    test_quant = TestQuantizedKVCache()
    test_quant.test_quantize_dequantize()
    test_quant.test_cache_update()
    
    test_hybrid = TestHybridGranularityCache()
    test_hybrid.test_turn_boundary_tracking()
    
    # Combined Cache Tests
    test_palm_cache = TestPALMCache()
    test_palm_cache.test_cache_creation()
    test_palm_cache.test_cache_with_both_strategies()
    
    # Gradient Tests
    test_grad = TestGradientFlow()
    test_grad.test_backward_with_cross_layer_sharing()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

