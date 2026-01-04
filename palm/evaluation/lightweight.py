"""
Lightweight evaluation for efficient periodic eval during training.

Designed to be cheap while still capturing key PALM metrics with staggered computation:

Cluster 1 - Degeneration (output quality):
  - repetition_rate, distinct_2gram, distinct_3gram

Cluster 2 - Faithfulness (source grounding):
  - entity_precision, hallucination_rate, copy_rate

Cluster 3 - Loss-based (every epoch, cheap):
  - perplexity, sae_ratio, ppl_drift

Cluster 4 - Mask Compliance (infrequent validation):
  - future_leakage, source_attention_mass

PALM Score - computed when enough data available, uses cached cluster values
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .comprehensive import (
    compute_distinct_ngrams,
    compute_repetition_rate,
    extract_entities_simple,
    compute_ngram_overlap,
    compute_perplexity,
)


@dataclass
class DegenerationCluster:
    """Cluster 1: Output quality metrics."""
    repetition_rate: float = 0.0
    distinct_2gram: float = 1.0
    distinct_3gram: float = 1.0
    
    def to_dict(self, prefix: str = "degen") -> Dict[str, float]:
        return {
            f"{prefix}/repetition_rate": self.repetition_rate,
            f"{prefix}/distinct_2gram": self.distinct_2gram,
            f"{prefix}/distinct_3gram": self.distinct_3gram,
        }


@dataclass
class FaithfulnessCluster:
    """Cluster 2: Source grounding metrics."""
    entity_precision: float = 1.0
    hallucination_rate: float = 0.0
    copy_rate: float = 0.0
    
    def to_dict(self, prefix: str = "faith") -> Dict[str, float]:
        return {
            f"{prefix}/entity_precision": self.entity_precision,
            f"{prefix}/hallucination_rate": self.hallucination_rate,
            f"{prefix}/copy_rate": self.copy_rate,
        }


@dataclass
class LossCluster:
    """Cluster 3: Loss-based metrics (cheap, every epoch)."""
    perplexity: float = 0.0
    sae_ratio: float = 0.0
    ppl_drift: float = 0.0
    
    def to_dict(self, prefix: str = "loss") -> Dict[str, float]:
        return {
            f"{prefix}/perplexity": self.perplexity,
            f"{prefix}/sae_ratio": self.sae_ratio,
            f"{prefix}/ppl_drift": self.ppl_drift,
        }


@dataclass
class MaskComplianceCluster:
    """Cluster 4: Attention mask validation (infrequent)."""
    future_leakage: float = 0.0
    source_attention_mass: float = 0.0
    
    def to_dict(self, prefix: str = "mask") -> Dict[str, float]:
        return {
            f"{prefix}/future_leakage": self.future_leakage,
            f"{prefix}/source_attention_mass": self.source_attention_mass,
        }


@dataclass 
class StaggeredEvalResult:
    """Result from staggered evaluation with cluster tracking."""
    # Which clusters were computed this round
    computed_clusters: List[str] = field(default_factory=list)
    
    # Cluster results (None if not computed this round)
    degeneration: Optional[DegenerationCluster] = None
    faithfulness: Optional[FaithfulnessCluster] = None
    loss: Optional[LossCluster] = None
    mask_compliance: Optional[MaskComplianceCluster] = None
    
    # PALM score (computed when both degen and faith available)
    palm_score: Optional[float] = None
    
    def to_wandb_dict(self, prefix: str = "comprehensive") -> Dict[str, float]:
        """Convert computed clusters to wandb dict."""
        result = {}
        if self.degeneration:
            result.update(self.degeneration.to_dict(f"{prefix}/degen"))
        if self.faithfulness:
            result.update(self.faithfulness.to_dict(f"{prefix}/faith"))
        if self.loss:
            result.update(self.loss.to_dict(f"{prefix}/loss"))
        if self.mask_compliance:
            result.update(self.mask_compliance.to_dict(f"{prefix}/mask"))
        if self.palm_score is not None:
            result[f"{prefix}/palm_score"] = self.palm_score
        return result


# Legacy dataclass for backwards compatibility
@dataclass
class LightweightEvalResult:
    """Aggregated metrics from lightweight eval (legacy, use StaggeredEvalResult)."""
    perplexity: float
    sae_ratio: float
    repetition_rate: float
    distinct_2gram: float
    entity_precision: float
    hallucination_rate: float
    future_leakage: float
    source_attention_mass: float
    palm_score: float
    
    def to_wandb_dict(self, prefix: str = "eval") -> Dict[str, float]:
        return {
            f"{prefix}/perplexity": self.perplexity,
            f"{prefix}/sae_ratio": self.sae_ratio,
            f"{prefix}/repetition_rate": self.repetition_rate,
            f"{prefix}/distinct_2gram": self.distinct_2gram,
            f"{prefix}/entity_precision": self.entity_precision,
            f"{prefix}/hallucination_rate": self.hallucination_rate,
            f"{prefix}/future_leakage": self.future_leakage,
            f"{prefix}/source_attention_mass": self.source_attention_mass,
            f"{prefix}/palm_score": self.palm_score,
        }


def compute_generation_metrics_fast(
    generated_tokens: List[int],
    source_text: str,
    output_text: str,
) -> Tuple[float, float, float, float]:
    """
    Compute degeneration + faithfulness metrics in one pass.
    
    Returns: (repetition_rate, distinct_2gram, entity_precision, hallucination_rate)
    """
    # Degeneration
    rep_rate = compute_repetition_rate(generated_tokens, n=3)
    dist_2 = compute_distinct_ngrams(generated_tokens, 2)
    
    # Faithfulness
    source_entities = extract_entities_simple(source_text)
    output_entities = extract_entities_simple(output_text)
    
    if output_entities:
        precision = len(output_entities & source_entities) / len(output_entities)
        halluc = 1.0 - precision
    else:
        precision = 1.0
        halluc = 0.0
    
    return rep_rate, dist_2, precision, halluc


def compute_mask_compliance_fast(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    source_length: int,
    sample_layers: int = 3,
) -> Tuple[float, float]:
    """
    Fast mask compliance check - samples a few layers instead of all.
    
    Returns: (future_leakage, source_attention_mass)
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    seq_len = input_ids.size(1)
    if seq_len <= source_length:
        return 0.0, 1.0  # No target tokens to check
    
    # Get attention mask and embeddings
    attention_mask = model.create_bidirectional_attention_mask(input_ids)
    hidden_states = model.embeddings(input_ids)
    
    # Sample evenly-spaced layers
    num_layers = len(model.layers)
    layer_indices = [
        i * num_layers // sample_layers 
        for i in range(sample_layers)
    ]
    
    future_leakages = []
    source_masses = []
    
    with torch.no_grad():
        for layer_idx in layer_indices:
            layer = model.layers[layer_idx]
            attn = layer.attention
            
            # Compute attention scores
            query = attn.transpose_for_scores(attn.query(hidden_states))
            key = attn.transpose_for_scores(attn.key(hidden_states))
            
            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores / math.sqrt(attn.attention_head_size)
            scores = scores + attention_mask
            probs = F.softmax(scores, dim=-1)  # [batch, heads, seq, seq]
            
            # Average over heads
            avg_probs = probs.mean(dim=1)[0]  # [seq, seq]
            
            # Check target positions only
            for t in range(source_length, seq_len):
                # Future leakage
                if t < seq_len - 1:
                    future_leakages.append(avg_probs[t, t+1:].sum().item())
                # Source attention
                source_masses.append(avg_probs[t, :source_length].sum().item())
    
    avg_leak = sum(future_leakages) / len(future_leakages) if future_leakages else 0.0
    avg_source = sum(source_masses) / len(source_masses) if source_masses else 0.0
    
    return avg_leak, avg_source


def compute_palm_score_fast(
    entity_precision: float,
    hallucination_rate: float,
    repetition_rate: float,
    ppl_drift: float,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
) -> float:
    """Compute PALM composite score."""
    return entity_precision - alpha * hallucination_rate - beta * repetition_rate - gamma * ppl_drift


class LightweightEvaluator:
    """
    Efficient evaluator for periodic use during training.
    
    Usage:
        evaluator = LightweightEvaluator(model, tokenizer, eval_samples)
        result = evaluator.run(baseline_ppl=baseline_ppl)
        wandb.log(result.to_wandb_dict())
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        eval_samples: List[Dict[str, str]],
        max_gen_tokens: int = 50,
    ):
        """
        Args:
            model: PALM model
            tokenizer: Tokenizer
            eval_samples: List of {"source": ..., "target": ...} dicts (5-10 samples ideal)
            max_gen_tokens: Max tokens to generate per sample
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eval_samples = eval_samples
        self.max_gen_tokens = max_gen_tokens
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def run(
        self,
        baseline_ppl: Optional[float] = None,
        check_mask_compliance: bool = True,
    ) -> LightweightEvalResult:
        """
        Run lightweight evaluation.
        
        Args:
            baseline_ppl: Reference perplexity for PPL drift (optional)
            check_mask_compliance: Whether to check attention mask (can skip for speed)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_sae_loss = 0.0
        num_loss_samples = 0
        
        all_rep_rates = []
        all_dist_2 = []
        all_precision = []
        all_halluc = []
        
        for sample in self.eval_samples:
            source_text = sample["source"]
            target_text = sample.get("target", "")
            
            # Encode source
            source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(self.device)
            source_len = source_ids.size(1)
            
            # Get loss if we have target
            if target_text:
                full_text = source_text + target_text
                full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
                labels = full_ids.clone()
                labels[:, :source_len] = -100
                
                outputs = self.model(
                    full_ids,
                    labels=labels,
                    source_len=torch.tensor([source_len], device=self.device),
                )
                _, combined_loss, lm_loss, sae_loss = outputs
                
                if lm_loss is not None:
                    total_loss += lm_loss.item()
                    num_loss_samples += 1
                if sae_loss is not None:
                    total_sae_loss += sae_loss.item()
            
            # Generate
            generated = self.model.generate(
                source_ids,
                max_length=source_len + self.max_gen_tokens,
                do_sample=False,
                use_cache=True,
            )
            
            gen_tokens = generated[0, source_len:].tolist()
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Compute metrics
            rep, dist2, prec, hal = compute_generation_metrics_fast(
                gen_tokens, source_text, gen_text
            )
            all_rep_rates.append(rep)
            all_dist_2.append(dist2)
            all_precision.append(prec)
            all_halluc.append(hal)
        
        # Aggregate
        avg_loss = total_loss / max(num_loss_samples, 1)
        avg_sae = total_sae_loss / max(num_loss_samples, 1)
        ppl = compute_perplexity(avg_loss)
        sae_ratio = avg_sae / (avg_loss + avg_sae) if (avg_loss + avg_sae) > 0 else 0.0
        
        avg_rep = sum(all_rep_rates) / len(all_rep_rates) if all_rep_rates else 0.0
        avg_dist2 = sum(all_dist_2) / len(all_dist_2) if all_dist_2 else 1.0
        avg_prec = sum(all_precision) / len(all_precision) if all_precision else 1.0
        avg_hal = sum(all_halluc) / len(all_halluc) if all_halluc else 0.0
        
        # Mask compliance (on first sample only)
        future_leak = 0.0
        source_mass = 0.0
        if check_mask_compliance and self.eval_samples:
            source_text = self.eval_samples[0]["source"]
            source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(self.device)
            # Add some target tokens for mask check
            target_stub = " The answer is"
            full_ids = self.tokenizer.encode(source_text + target_stub, return_tensors="pt").to(self.device)
            source_len = source_ids.size(1)
            
            future_leak, source_mass = compute_mask_compliance_fast(
                self.model, full_ids, source_len, sample_layers=3
            )
        
        # PALM score
        ppl_drift = max(0, ppl - baseline_ppl) if baseline_ppl else 0.0
        palm_score = compute_palm_score_fast(avg_prec, avg_hal, avg_rep, ppl_drift)
        
        return LightweightEvalResult(
            perplexity=ppl,
            sae_ratio=sae_ratio,
            repetition_rate=avg_rep,
            distinct_2gram=avg_dist2,
            entity_precision=avg_prec,
            hallucination_rate=avg_hal,
            future_leakage=future_leak,
            source_attention_mass=source_mass,
            palm_score=palm_score,
        )


class StaggeredEvaluator:
    """
    Staggered evaluator that computes metric clusters on different schedules.
    
    Default schedule:
    - Loss cluster: every epoch (cheap, uses existing forward pass data)
    - Degeneration cluster: every 2 epochs (even epochs: 0, 2, 4, ...)
    - Faithfulness cluster: every 2 epochs, offset (odd epochs: 1, 3, 5, ...)
    - Mask compliance: epoch 0 only (validation)
    - PALM score: whenever we have recent data from both degen and faith
    
    Usage:
        evaluator = StaggeredEvaluator(model, tokenizer, eval_samples)
        result = evaluator.run_for_epoch(epoch, eval_loss, eval_sae_loss, baseline_ppl)
        wandb.log(result.to_wandb_dict())
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        eval_samples: List[Dict[str, str]],
        max_gen_tokens: int = 50,
        degen_every: int = 2,
        faith_every: int = 2,
        faith_offset: int = 1,
        mask_epochs: List[int] = None,
    ):
        """
        Args:
            model: PALM model
            tokenizer: Tokenizer
            eval_samples: List of {"source": ..., "target": ...} dicts
            max_gen_tokens: Max tokens to generate per sample
            degen_every: Compute degeneration every N epochs
            faith_every: Compute faithfulness every N epochs
            faith_offset: Offset for faithfulness (to stagger from degen)
            mask_epochs: Specific epochs to check mask compliance (default: [0])
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eval_samples = eval_samples
        self.max_gen_tokens = max_gen_tokens
        self.device = next(model.parameters()).device
        
        # Schedules
        self.degen_every = degen_every
        self.faith_every = faith_every
        self.faith_offset = faith_offset
        self.mask_epochs = mask_epochs or [0]
        
        # Cached cluster values for PALM score computation
        self._cached_degen: Optional[DegenerationCluster] = None
        self._cached_faith: Optional[FaithfulnessCluster] = None
        self._cached_loss: Optional[LossCluster] = None
        self._cached_mask: Optional[MaskComplianceCluster] = None
    
    def _get_generation_model(self):
        """
        Get the underlying model suitable for generation.
        
        Handles PEFT-wrapped models by extracting the base PALMModel
        which has a proper generate() method.
        """
        model = self.model
        
        # If wrapped by PEFT, get the underlying PALMModel
        # PEFT wraps as: PeftModelForCausalLM -> LoraModel -> PALMModel
        if hasattr(model, 'base_model'):
            base = model.base_model
            if hasattr(base, 'model'):
                # This is the PALMModel
                return base.model
            return base
        
        return model
    
    def _should_compute_degen(self, epoch: int) -> bool:
        return epoch % self.degen_every == 0
    
    def _should_compute_faith(self, epoch: int) -> bool:
        return (epoch - self.faith_offset) % self.faith_every == 0
    
    def _should_compute_mask(self, epoch: int) -> bool:
        return epoch in self.mask_epochs
    
    @torch.no_grad()
    def _compute_degeneration(self) -> DegenerationCluster:
        """Compute degeneration metrics via generation."""
        self.model.eval()
        
        all_rep = []
        all_d2 = []
        all_d3 = []
        
        for sample in self.eval_samples:
            try:
                source_text = sample["source"]
                source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(self.device)
                source_len = source_ids.size(1)
                
                # Get the underlying model for generation if wrapped by PEFT
                gen_model = self._get_generation_model()
                
                generated = gen_model.generate(
                    source_ids,
                    max_length=source_len + self.max_gen_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                gen_tokens = generated[0, source_len:].tolist()
                
                all_rep.append(compute_repetition_rate(gen_tokens, n=3))
                all_d2.append(compute_distinct_ngrams(gen_tokens, 2))
                all_d3.append(compute_distinct_ngrams(gen_tokens, 3))
            except Exception as e:
                # Log but don't fail - use defaults for this sample
                import logging
                logging.warning(f"Degeneration eval failed for sample: {e}")
                continue
        
        return DegenerationCluster(
            repetition_rate=sum(all_rep) / len(all_rep) if all_rep else 0.0,
            distinct_2gram=sum(all_d2) / len(all_d2) if all_d2 else 1.0,
            distinct_3gram=sum(all_d3) / len(all_d3) if all_d3 else 1.0,
        )
    
    @torch.no_grad()
    def _compute_faithfulness(self) -> FaithfulnessCluster:
        """Compute faithfulness metrics via generation."""
        self.model.eval()
        
        all_prec = []
        all_hal = []
        all_copy = []
        
        for sample in self.eval_samples:
            try:
                source_text = sample["source"]
                source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(self.device)
                source_len = source_ids.size(1)
                
                # Get the underlying model for generation if wrapped by PEFT
                gen_model = self._get_generation_model()
                
                generated = gen_model.generate(
                    source_ids,
                    max_length=source_len + self.max_gen_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                gen_tokens = generated[0, source_len:].tolist()
                gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                # Entity-based metrics
                source_entities = extract_entities_simple(source_text)
                output_entities = extract_entities_simple(gen_text)
                
                if output_entities:
                    prec = len(output_entities & source_entities) / len(output_entities)
                    hal = 1.0 - prec
                else:
                    prec = 1.0
                    hal = 0.0
                
                all_prec.append(prec)
                all_hal.append(hal)
                all_copy.append(compute_ngram_overlap(source_text, gen_text, n=3))
            except Exception as e:
                # Log but don't fail - use defaults for this sample
                import logging
                logging.warning(f"Faithfulness eval failed for sample: {e}")
                continue
        
        return FaithfulnessCluster(
            entity_precision=sum(all_prec) / len(all_prec) if all_prec else 1.0,
            hallucination_rate=sum(all_hal) / len(all_hal) if all_hal else 0.0,
            copy_rate=sum(all_copy) / len(all_copy) if all_copy else 0.0,
        )
    
    def _compute_loss(
        self,
        eval_loss: float,
        eval_sae_loss: float,
        baseline_ppl: Optional[float],
    ) -> LossCluster:
        """Compute loss-based metrics (cheap, from existing eval loop data)."""
        ppl = compute_perplexity(eval_loss)
        sae_ratio = eval_sae_loss / (eval_loss + eval_sae_loss) if (eval_loss + eval_sae_loss) > 0 else 0.0
        ppl_drift = max(0, ppl - baseline_ppl) if baseline_ppl else 0.0
        
        return LossCluster(
            perplexity=ppl,
            sae_ratio=sae_ratio,
            ppl_drift=ppl_drift,
        )
    
    @torch.no_grad()
    def _compute_mask_compliance(self) -> MaskComplianceCluster:
        """Compute mask compliance on first sample."""
        if not self.eval_samples:
            return MaskComplianceCluster()
        
        try:
            source_text = self.eval_samples[0]["source"]
            source_ids = self.tokenizer.encode(source_text, return_tensors="pt").to(self.device)
            target_stub = " The answer is"
            full_ids = self.tokenizer.encode(source_text + target_stub, return_tensors="pt").to(self.device)
            source_len = source_ids.size(1)
            
            # Get the underlying model for mask compliance check
            base_model = self._get_generation_model()
            
            future_leak, source_mass = compute_mask_compliance_fast(
                base_model, full_ids, source_len, sample_layers=3
            )
            
            return MaskComplianceCluster(
                future_leakage=future_leak,
                source_attention_mass=source_mass,
            )
        except Exception as e:
            import logging
            logging.warning(f"Mask compliance eval failed: {e}")
            return MaskComplianceCluster()
    
    def _compute_palm_score(self) -> Optional[float]:
        """Compute PALM score if we have cached degen and faith data."""
        if self._cached_degen is None or self._cached_faith is None:
            return None
        
        ppl_drift = self._cached_loss.ppl_drift if self._cached_loss else 0.0
        
        return compute_palm_score_fast(
            entity_precision=self._cached_faith.entity_precision,
            hallucination_rate=self._cached_faith.hallucination_rate,
            repetition_rate=self._cached_degen.repetition_rate,
            ppl_drift=ppl_drift,
        )
    
    def run_for_epoch(
        self,
        epoch: int,
        eval_loss: float,
        eval_sae_loss: float,
        baseline_ppl: Optional[float] = None,
    ) -> StaggeredEvalResult:
        """
        Run staggered evaluation for a given epoch.
        
        Only computes clusters scheduled for this epoch.
        Returns result with computed clusters and PALM score if available.
        
        Note: This method is designed to be defensive - individual cluster
        computations can fail without causing the entire evaluation to fail.
        """
        result = StaggeredEvalResult(computed_clusters=[])
        
        # Loss cluster: always computed (cheap)
        try:
            result.loss = self._compute_loss(eval_loss, eval_sae_loss, baseline_ppl)
            self._cached_loss = result.loss
            result.computed_clusters.append("loss")
        except Exception as e:
            import logging
            logging.warning(f"Loss cluster computation failed: {e}")
        
        # Degeneration cluster: check schedule
        if self._should_compute_degen(epoch):
            try:
                result.degeneration = self._compute_degeneration()
                self._cached_degen = result.degeneration
                result.computed_clusters.append("degeneration")
            except Exception as e:
                import logging
                logging.warning(f"Degeneration cluster computation failed: {e}")
        
        # Faithfulness cluster: check schedule
        if self._should_compute_faith(epoch):
            try:
                result.faithfulness = self._compute_faithfulness()
                self._cached_faith = result.faithfulness
                result.computed_clusters.append("faithfulness")
            except Exception as e:
                import logging
                logging.warning(f"Faithfulness cluster computation failed: {e}")
        
        # Mask compliance: check schedule
        if self._should_compute_mask(epoch):
            try:
                result.mask_compliance = self._compute_mask_compliance()
                self._cached_mask = result.mask_compliance
                result.computed_clusters.append("mask_compliance")
            except Exception as e:
                import logging
                logging.warning(f"Mask compliance cluster computation failed: {e}")
        
        # PALM score: compute if we have both degen and faith cached
        try:
            result.palm_score = self._compute_palm_score()
        except Exception as e:
            import logging
            logging.warning(f"PALM score computation failed: {e}")
        
        return result
    
    def get_schedule_info(self, num_epochs: int) -> str:
        """Return human-readable schedule info."""
        lines = [
            f"Staggered Evaluation Schedule ({num_epochs} epochs):",
            f"  Loss cluster: every epoch",
            f"  Degeneration: epochs {list(range(0, num_epochs, self.degen_every))}",
            f"  Faithfulness: epochs {list(range(self.faith_offset, num_epochs, self.faith_every))}",
            f"  Mask compliance: epochs {[e for e in self.mask_epochs if e < num_epochs]}",
        ]
        return "\n".join(lines)


def create_eval_samples_from_dataloader(
    dataloader,
    tokenizer,
    num_samples: int = 8,
) -> List[Dict[str, str]]:
    """
    Extract eval samples from a dataloader for lightweight eval.
    
    Returns list of {"source": ..., "target": ...} dicts.
    """
    samples = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        source_lens = batch["source_len"]
        
        for i in range(input_ids.size(0)):
            if len(samples) >= num_samples:
                return samples
            
            ids = input_ids[i].tolist()
            src_len = source_lens[i].item()
            
            # Decode source and target
            source_ids = ids[:src_len]
            target_ids = [t for t in ids[src_len:] if t != tokenizer.pad_token_id]
            
            source_text = tokenizer.decode(source_ids, skip_special_tokens=True)
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            
            if source_text.strip() and target_text.strip():
                samples.append({"source": source_text, "target": target_text})
    
    return samples

