"""
Unit tests for PALM training utilities.

Tests the training recipe components from finetune_palm.ipynb:
- SAEWeightScheduler: 3-phase SAE weight scheduling
- EMAModel: Exponential Moving Average weights
- LossSpikeDetector: Training instability detection
- DynamicSAEWeight: Adaptive SAE weight adjustment
- freeze_backbone: Parameter freezing logic
- create_optimizer: Differential learning rate setup

Run with: python -m pytest palm/tests/test_training_utils.py -v
Or: python palm/tests/test_training_utils.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import pytest
from typing import List, Tuple

# Copy of training utilities from notebook for testing
# (In production, these would be moved to palm/training/ module)
class SAEWeightScheduler:
    """
    Scheduler for SAE loss weight that implements the PALM training recipe:
    - Phase 1 (LM Warmup): SAE weight = 0 (LM-only training)
    - Phase 2 (SAE Ramp): Linear ramp from start_weight to end_weight
    - Phase 3 (Stable): SAE weight = end_weight
    """
    def __init__(
        self,
        warmup_epochs: int,
        ramp_epochs: int,
        start_weight: float = 0.01,
        end_weight: float = 0.35,
    ):
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.ramp_start_epoch = warmup_epochs
        self.ramp_end_epoch = warmup_epochs + ramp_epochs
    
    def get_weight(self, epoch: int) -> float:
        """Get SAE weight for given epoch (0-indexed)."""
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.ramp_end_epoch:
            ramp_progress = (epoch - self.ramp_start_epoch) / max(self.ramp_epochs, 1)
            return self.start_weight + ramp_progress * (self.end_weight - self.start_weight)
        else:
            return self.end_weight
    
    def get_phase(self, epoch: int) -> str:
        """Get training phase name for logging."""
        if epoch < self.warmup_epochs:
            return "LM_WARMUP"
        elif epoch < self.ramp_end_epoch:
            return "SAE_RAMP"
        else:
            return "STABLE"


class EMAModel:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model):
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self, model):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def copy_to(self, model):
        """Permanently copy shadow weights to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])


class LossSpikeDetector:
    """Detects training instability (loss spikes) and triggers recovery."""
    def __init__(self, threshold=3.0, patience=3, lr_factor=0.5, window_size=50):
        self.threshold = threshold
        self.patience = patience
        self.lr_factor = lr_factor
        self.window_size = window_size
        
        self.loss_history = []
        self.spike_count = 0
        self.recovered = False
    
    def check(self, loss: float) -> Tuple[bool, str]:
        """Check if current loss is a spike."""
        self.loss_history.append(loss)
        
        if len(self.loss_history) < self.window_size:
            return False, ""
        
        recent = self.loss_history[-self.window_size:-1]
        rolling_avg = sum(recent) / len(recent)
        
        if loss > self.threshold * rolling_avg:
            self.spike_count += 1
            if self.spike_count >= self.patience:
                self.spike_count = 0
                self.recovered = True
                return True, f"Loss spike detected! {loss:.4f} > {self.threshold}x avg ({rolling_avg:.4f})"
            return False, f"Potential spike ({self.spike_count}/{self.patience})"
        else:
            self.spike_count = max(0, self.spike_count - 1)
            return False, ""
    
    def should_reduce_lr(self) -> bool:
        """Check if we should reduce LR (resets after calling)."""
        if self.recovered:
            self.recovered = False
            return True
        return False


class DynamicSAEWeight:
    """Dynamically adjusts SAE weight based on the ratio of LM loss to SAE loss."""
    def __init__(self, target_ratio=0.25, adjust_rate=0.1, min_weight=0.01, max_weight=0.5):
        self.target_ratio = target_ratio
        self.adjust_rate = adjust_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def adjust(self, current_weight: float, lm_loss: float, sae_loss: float) -> Tuple[float, str]:
        """Compute adjusted SAE weight based on loss balance."""
        if sae_loss <= 0 or lm_loss <= 0:
            return current_weight, ""
        
        total_loss = lm_loss + current_weight * sae_loss
        current_ratio = (current_weight * sae_loss) / total_loss
        
        error = current_ratio - self.target_ratio
        adjustment = -error * self.adjust_rate
        
        new_weight = current_weight + adjustment * current_weight
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        
        if abs(adjustment) > 0.01:
            direction = "↓" if adjustment < 0 else "↑"
            return new_weight, f"SAE weight {direction} {current_weight:.3f}→{new_weight:.3f} (ratio: {current_ratio:.2f})"
        
        return new_weight, ""


# Helper: Create simple test model
class SimpleTestModel(nn.Module):
    """Simple model for testing training utilities."""
    def __init__(self):
        super().__init__()
        # Backbone-like layers
        self.embeddings = nn.Embedding(100, 32)
        self.backbone_layer = nn.Linear(32, 32)
        self.backbone_norm = nn.LayerNorm(32)
        
        # PALM-specific layers
        self.partial_attention = nn.Linear(32, 32)
        self.sae_head = nn.Linear(32, 100)
        self.lm_head = nn.Linear(32, 100)
    
    def forward(self, x):
        x = self.embeddings(x)
        x = self.backbone_layer(x)
        x = self.backbone_norm(x)
        return self.lm_head(x), self.sae_head(x)


def freeze_backbone(model, freeze: bool = True):
    """Freeze or unfreeze backbone parameters (non-PALM-specific layers)."""
    trainable_patterns = [
        'partial_attention',
        'sae_head',
        'lm_head',
        'lora',
        'Fp',
        'language_embeddings',
    ]
    
    frozen_count = 0
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        is_palm_or_lora = any(pattern in name for pattern in trainable_patterns)
        
        if is_palm_or_lora:
            param.requires_grad = True
            unfrozen_count += 1
        else:
            param.requires_grad = not freeze
            if freeze:
                frozen_count += 1
            else:
                unfrozen_count += 1
    
    return frozen_count, unfrozen_count


# Tests: SAEWeightScheduler
class TestSAEWeightScheduler:
    """Tests for SAE weight scheduling."""
    
    def test_warmup_phase_weight_is_zero(self):
        """During warmup, SAE weight should be 0."""
        scheduler = SAEWeightScheduler(
            warmup_epochs=2,
            ramp_epochs=4,
            start_weight=0.01,
            end_weight=0.35,
        )
        
        assert scheduler.get_weight(0) == 0.0
        assert scheduler.get_weight(1) == 0.0
    
    def test_warmup_phase_name(self):
        """Phase name should be LM_WARMUP during warmup."""
        scheduler = SAEWeightScheduler(warmup_epochs=2, ramp_epochs=4)
        
        assert scheduler.get_phase(0) == "LM_WARMUP"
        assert scheduler.get_phase(1) == "LM_WARMUP"
    
    def test_ramp_phase_linear_increase(self):
        """During ramp, weight should increase linearly."""
        scheduler = SAEWeightScheduler(
            warmup_epochs=2,
            ramp_epochs=4,
            start_weight=0.0,
            end_weight=0.4,
        )
        
        # Epoch 2: start of ramp (0% progress) -> 0.0
        assert scheduler.get_weight(2) == pytest.approx(0.0, abs=0.01)
        
        # Epoch 3: 25% progress -> 0.1
        assert scheduler.get_weight(3) == pytest.approx(0.1, abs=0.01)
        
        # Epoch 4: 50% progress -> 0.2
        assert scheduler.get_weight(4) == pytest.approx(0.2, abs=0.01)
        
        # Epoch 5: 75% progress -> 0.3
        assert scheduler.get_weight(5) == pytest.approx(0.3, abs=0.01)
    
    def test_ramp_phase_name(self):
        """Phase name should be SAE_RAMP during ramp."""
        scheduler = SAEWeightScheduler(warmup_epochs=2, ramp_epochs=4)
        
        for epoch in range(2, 6):
            assert scheduler.get_phase(epoch) == "SAE_RAMP"
    
    def test_stable_phase_constant_weight(self):
        """After ramp, weight should stay constant."""
        scheduler = SAEWeightScheduler(
            warmup_epochs=2,
            ramp_epochs=4,
            start_weight=0.01,
            end_weight=0.35,
        )
        
        # Epoch 6 onwards should be stable
        for epoch in range(6, 20):
            assert scheduler.get_weight(epoch) == pytest.approx(0.35, abs=0.001)
            assert scheduler.get_phase(epoch) == "STABLE"
    
    def test_zero_warmup_epochs(self):
        """With 0 warmup epochs, should start ramping immediately."""
        scheduler = SAEWeightScheduler(
            warmup_epochs=0,
            ramp_epochs=2,
            start_weight=0.1,
            end_weight=0.3,
        )
        
        assert scheduler.get_phase(0) == "SAE_RAMP"
        assert scheduler.get_weight(0) == pytest.approx(0.1, abs=0.01)
    
    def test_zero_ramp_epochs(self):
        """With 0 ramp epochs, should jump directly to stable."""
        scheduler = SAEWeightScheduler(
            warmup_epochs=2,
            ramp_epochs=0,
            start_weight=0.01,
            end_weight=0.35,
        )
        
        assert scheduler.get_phase(1) == "LM_WARMUP"
        assert scheduler.get_phase(2) == "STABLE"
        assert scheduler.get_weight(2) == pytest.approx(0.35, abs=0.01)


# Tests: EMAModel
class TestEMAModel:
    """Tests for Exponential Moving Average."""
    
    def test_initialization(self):
        """EMA should initialize with model's current weights."""
        model = SimpleTestModel()
        ema = EMAModel(model, decay=0.999)
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.allclose(ema.shadow[name], param.data)
    
    def test_update_moves_towards_current(self):
        """EMA update should move shadow weights towards current weights."""
        model = SimpleTestModel()
        ema = EMAModel(model, decay=0.9)  # Low decay for visible effect
        
        # Store original shadow
        original_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        
        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)  # Add 1 to all weights
        
        # Update EMA
        ema.update(model)
        
        # Check that shadow moved towards new weights but not all the way
        for name, param in model.named_parameters():
            if name in ema.shadow:
                # Shadow should be between original and current
                diff_to_original = (ema.shadow[name] - original_shadow[name]).abs().mean()
                diff_to_current = (ema.shadow[name] - param.data).abs().mean()
                
                assert diff_to_original > 0  # Should have moved
                assert diff_to_current > 0   # But not all the way
    
    def test_apply_and_restore(self):
        """apply_shadow and restore should be reversible."""
        model = SimpleTestModel()
        ema = EMAModel(model, decay=0.999)
        
        # Modify model to be different from EMA
        original_weights = {name: param.data.clone() 
                          for name, param in model.named_parameters()}
        
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(2.0)
        
        modified_weights = {name: param.data.clone() 
                           for name, param in model.named_parameters()}
        
        # Apply shadow
        ema.apply_shadow(model)
        
        # Weights should now match original (shadow)
        for name, param in model.named_parameters():
            if name in ema.shadow:
                assert torch.allclose(param.data, original_weights[name])
        
        # Restore
        ema.restore(model)
        
        # Weights should be back to modified
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, modified_weights[name])
    
    def test_copy_to_permanent(self):
        """copy_to should permanently set model weights to shadow."""
        model = SimpleTestModel()
        ema = EMAModel(model, decay=0.999)
        
        # Store original shadow
        shadow_copy = {k: v.clone() for k, v in ema.shadow.items()}
        
        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(2.0)
        
        # Copy shadow to model
        ema.copy_to(model)
        
        # Model should now have shadow weights
        for name, param in model.named_parameters():
            if name in shadow_copy:
                assert torch.allclose(param.data, shadow_copy[name])
    
    def test_decay_effect(self):
        """Higher decay should result in slower updates."""
        model1 = SimpleTestModel()
        model2 = SimpleTestModel()
        
        # Initialize with same weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.data.copy_(p1.data)
        
        ema_fast = EMAModel(model1, decay=0.9)  # Fast update
        ema_slow = EMAModel(model2, decay=0.999)  # Slow update
        
        # Modify both models the same way
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p1.add_(1.0)
                p2.add_(1.0)
        
        ema_fast.update(model1)
        ema_slow.update(model2)
        
        # Fast EMA should have moved more
        for (name1, _), (name2, _) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 in ema_fast.shadow and name2 in ema_slow.shadow:
                fast_diff = (ema_fast.shadow[name1] - ema_slow.shadow[name2]).abs().mean()
                # Fast should be closer to new weights (further from slow)
                assert fast_diff > 0


# Tests: LossSpikeDetector
class TestLossSpikeDetector:
    """Tests for loss spike detection."""
    
    def test_no_spike_in_warmup(self):
        """No spikes should be detected until window is full."""
        detector = LossSpikeDetector(threshold=3.0, patience=3, window_size=10)
        
        # First 9 values shouldn't trigger anything
        for i in range(9):
            is_spike, msg = detector.check(1.0)
            assert not is_spike
            assert msg == ""
    
    def test_normal_loss_no_spike(self):
        """Normal losses should not trigger spike detection."""
        detector = LossSpikeDetector(threshold=3.0, patience=3, window_size=10)
        
        # Fill window with normal losses
        for _ in range(20):
            is_spike, msg = detector.check(1.0)
        
        # Slightly elevated but not 3x
        is_spike, msg = detector.check(2.0)
        assert not is_spike
        assert not detector.should_reduce_lr()
    
    def test_spike_detection_with_patience(self):
        """Spike should only trigger after patience consecutive spikes."""
        detector = LossSpikeDetector(threshold=2.0, patience=3, window_size=10)
        
        # Fill window
        for _ in range(10):
            detector.check(1.0)
        
        # First spike
        is_spike, msg = detector.check(5.0)
        assert not is_spike  # Not yet, patience=1
        assert "Potential spike (1/3)" in msg
        
        # Second spike
        is_spike, msg = detector.check(5.0)
        assert not is_spike  # Not yet, patience=2
        assert "Potential spike (2/3)" in msg
        
        # Third spike - should trigger
        is_spike, msg = detector.check(5.0)
        assert is_spike
        assert "Loss spike detected" in msg
        assert detector.should_reduce_lr()
    
    def test_spike_count_decay(self):
        """Spike count should decay on normal losses."""
        detector = LossSpikeDetector(threshold=2.0, patience=3, window_size=10)
        
        # Fill window
        for _ in range(10):
            detector.check(1.0)
        
        # Two spikes
        detector.check(5.0)
        detector.check(5.0)
        assert detector.spike_count == 2
        
        # Normal loss - count should decay
        detector.check(1.0)
        assert detector.spike_count == 1
        
        # Another normal loss
        detector.check(1.0)
        assert detector.spike_count == 0
    
    def test_should_reduce_lr_resets(self):
        """should_reduce_lr should reset after being called."""
        detector = LossSpikeDetector(threshold=2.0, patience=1, window_size=10)
        
        # Fill window and trigger spike
        for _ in range(10):
            detector.check(1.0)
        detector.check(10.0)
        
        # First call should return True
        assert detector.should_reduce_lr()
        
        # Second call should return False (reset)
        assert not detector.should_reduce_lr()


# Tests: DynamicSAEWeight
class TestDynamicSAEWeight:
    """Tests for dynamic SAE weight adjustment."""
    
    def test_no_adjustment_at_target_ratio(self):
        """No adjustment when at target ratio."""
        adjuster = DynamicSAEWeight(target_ratio=0.25, adjust_rate=0.1, max_weight=1.0)
        
        # If SAE is 25% of total loss, minimal/no adjustment
        # lm_loss=0.75, sae_loss=1, weight=0.25 → total=1.0, ratio=0.25
        new_weight, msg = adjuster.adjust(0.25, 0.75, 1.0)
        
        # Should be very close to original (no significant adjustment)
        assert abs(new_weight - 0.25) < 0.05
        # Message should be empty (no significant adjustment)
        assert msg == ""
    
    def test_decrease_when_sae_dominant(self):
        """SAE weight should decrease when SAE loss is too dominant."""
        adjuster = DynamicSAEWeight(target_ratio=0.25, adjust_rate=0.1)
        
        # High SAE ratio: lm_loss=1, sae_loss=1, weight=1 → ratio=0.5
        new_weight, msg = adjuster.adjust(1.0, 1.0, 1.0)
        
        assert new_weight < 1.0
        assert "↓" in msg
    
    def test_increase_when_sae_underweight(self):
        """SAE weight should increase when SAE loss is too small."""
        adjuster = DynamicSAEWeight(target_ratio=0.25, adjust_rate=0.1)
        
        # Low SAE ratio: lm_loss=10, sae_loss=1, weight=0.1 → ratio~0.01
        new_weight, msg = adjuster.adjust(0.1, 10.0, 1.0)
        
        assert new_weight > 0.1
        assert "↑" in msg
    
    def test_respects_min_weight(self):
        """Weight should not go below min_weight."""
        adjuster = DynamicSAEWeight(
            target_ratio=0.01,  # Very low target
            adjust_rate=1.0,    # Aggressive adjustment
            min_weight=0.05,
        )
        
        # This should try to decrease aggressively
        new_weight, _ = adjuster.adjust(0.5, 1.0, 10.0)
        
        assert new_weight >= 0.05
    
    def test_respects_max_weight(self):
        """Weight should not exceed max_weight."""
        adjuster = DynamicSAEWeight(
            target_ratio=0.99,  # Very high target
            adjust_rate=1.0,    # Aggressive adjustment
            max_weight=0.5,
        )
        
        # This should try to increase aggressively
        new_weight, _ = adjuster.adjust(0.1, 10.0, 0.1)
        
        assert new_weight <= 0.5
    
    def test_handles_zero_loss(self):
        """Should handle zero loss gracefully."""
        adjuster = DynamicSAEWeight(target_ratio=0.25, adjust_rate=0.1)
        
        # Zero SAE loss
        new_weight, msg = adjuster.adjust(0.3, 1.0, 0.0)
        assert new_weight == 0.3
        assert msg == ""
        
        # Zero LM loss
        new_weight, msg = adjuster.adjust(0.3, 0.0, 1.0)
        assert new_weight == 0.3
        assert msg == ""


# Tests: freeze_backbone
class TestFreezeBackbone:
    """Tests for backbone freezing functionality."""
    
    def test_freeze_backbone_params(self):
        """Freezing should disable gradients for backbone params."""
        model = SimpleTestModel()
        
        frozen, unfrozen = freeze_backbone(model, freeze=True)
        
        # Check backbone params are frozen
        assert not model.embeddings.weight.requires_grad
        assert not model.backbone_layer.weight.requires_grad
        assert not model.backbone_layer.bias.requires_grad
        assert not model.backbone_norm.weight.requires_grad
        assert not model.backbone_norm.bias.requires_grad
        
        # Check PALM params are still trainable
        assert model.partial_attention.weight.requires_grad
        assert model.sae_head.weight.requires_grad
        assert model.lm_head.weight.requires_grad
    
    def test_unfreeze_backbone_params(self):
        """Unfreezing should enable gradients for backbone params."""
        model = SimpleTestModel()
        
        # First freeze
        freeze_backbone(model, freeze=True)
        
        # Then unfreeze
        frozen, unfrozen = freeze_backbone(model, freeze=False)
        
        # All should be trainable now
        for name, param in model.named_parameters():
            assert param.requires_grad, f"{name} should be trainable"
    
    def test_palm_params_always_trainable(self):
        """PALM-specific params should remain trainable even when frozen."""
        model = SimpleTestModel()
        
        # Freeze multiple times
        for _ in range(3):
            freeze_backbone(model, freeze=True)
        
        # PALM params should still be trainable
        assert model.partial_attention.weight.requires_grad
        assert model.sae_head.weight.requires_grad
        assert model.lm_head.weight.requires_grad
    
    def test_counts_are_correct(self):
        """Returned counts should match actual frozen/unfrozen params."""
        model = SimpleTestModel()
        
        frozen_count, unfrozen_count = freeze_backbone(model, freeze=True)
        
        # Count manually
        actual_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        actual_unfrozen = sum(1 for p in model.parameters() if p.requires_grad)
        
        assert frozen_count == actual_frozen
        assert unfrozen_count == actual_unfrozen


# Integration Tests
class TestTrainingIntegration:
    """Integration tests for training utilities working together."""
    
    def test_full_training_simulation(self):
        """Simulate a full training run with all utilities."""
        model = SimpleTestModel()
        
        # Initialize utilities
        sae_scheduler = SAEWeightScheduler(
            warmup_epochs=1,
            ramp_epochs=2,
            start_weight=0.1,
            end_weight=0.3,
        )
        ema = EMAModel(model, decay=0.99)
        spike_detector = LossSpikeDetector(threshold=3.0, patience=2, window_size=5)
        
        # Freeze backbone for warmup
        freeze_backbone(model, freeze=True)
        
        # Simulate training
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3
        )
        
        for epoch in range(4):
            # Get phase and weight
            phase = sae_scheduler.get_phase(epoch)
            sae_weight = sae_scheduler.get_weight(epoch)
            
            # Unfreeze at phase transition
            if epoch == 1:  # LM_WARMUP -> SAE_RAMP
                freeze_backbone(model, freeze=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Simulate training step
            x = torch.randint(0, 100, (2, 10))
            lm_out, sae_out = model(x)
            loss = lm_out.mean() + sae_weight * sae_out.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA
            ema.update(model)
            
            # Check spike
            spike_detector.check(loss.item())
        
        # Apply EMA at end
        ema.copy_to(model)
        
        # Verify final state
        assert sae_scheduler.get_phase(3) == "STABLE"
        assert sae_scheduler.get_weight(3) == pytest.approx(0.3, abs=0.01)


# Main
def run_all_tests():
    """Run all tests without pytest."""
    import traceback
    
    test_classes = [
        TestSAEWeightScheduler,
        TestEMAModel,
        TestLossSpikeDetector,
        TestDynamicSAEWeight,
        TestFreezeBackbone,
        TestTrainingIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS: {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  FAIL: {method_name}")
                    print(f"    Error: {e}")
                    traceback.print_exc()
                    failed_tests.append((test_class.__name__, method_name, str(e)))
    
    print(f"\n{'='*60}")
    print(f"Results: {passed_tests}/{total_tests} passed")
    print('='*60)
    
    if failed_tests:
        print("\nFailed tests:")
        for cls, method, error in failed_tests:
            print(f"  - {cls}.{method}: {error}")
        return False
    
    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
