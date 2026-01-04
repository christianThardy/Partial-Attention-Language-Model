class DynamicSAEWeight:
    def __init__(self, target_ratio=0.25, adjust_rate=0.1, min_weight=0.01, max_weight=0.5):
        self.target_ratio = target_ratio
        self.adjust_rate = adjust_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def adjust(self, current_weight, lm_loss, sae_loss):
        if sae_loss <= 0 or lm_loss <= 0:
            return current_weight, ''
        
        total_loss = lm_loss + current_weight * sae_loss
        current_ratio = (current_weight * sae_loss) / total_loss
        
        error = current_ratio - self.target_ratio
        adjustment = -error * self.adjust_rate
        
        new_weight = current_weight + adjustment * current_weight
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        
        print(f'lm_loss={lm_loss}, sae_loss={sae_loss}, weight={current_weight}')
        print(f'total_loss={total_loss}, current_ratio={current_ratio}')
        print(f'error={error}, adjustment={adjustment}')
        print(f'new_weight (before clamp)={current_weight + adjustment * current_weight}')
        print(f'new_weight (after clamp)={new_weight}')
        
        if abs(adjustment) > 0.01:
            return new_weight, 'adjusted'
        return new_weight, ''

adj = DynamicSAEWeight(target_ratio=0.25, adjust_rate=0.1)
new_weight, msg = adj.adjust(1.0, 3.0, 1.0)
print(f'Result: new_weight={new_weight}, msg="{msg}"')

