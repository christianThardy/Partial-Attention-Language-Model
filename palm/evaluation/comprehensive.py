"""
Comprehensive evaluation module for PALM.

Implements the full eval strategy from evals.yml:
- Loss decomposition & logging
- Degeneration/mush detection (repetition, entropy, length)
- Faithfulness & hallucination metrics
- Mask compliance tracking
- PALLM composite score
- Ablation evaluation runner
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import re


@dataclass
class LossDecomposition:
    """Structured loss breakdown for logging."""
    lm_loss: float
    sae_loss: float
    total_loss: float
    sae_ratio: float  # sae_loss / total_loss
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "lm_loss": self.lm_loss,
            "sae_loss": self.sae_loss,
            "total_loss": self.total_loss,
            "sae_ratio": self.sae_ratio,
        }


@dataclass
class DegenerationMetrics:
    """Metrics for detecting mushy/degenerate generation."""
    repetition_rate: float      # 1 - distinct_ngrams / total_ngrams
    distinct_1gram: float       # Ratio of unique unigrams
    distinct_2gram: float       # Ratio of unique bigrams
    distinct_3gram: float       # Ratio of unique trigrams
    avg_entropy: float          # Average token entropy during generation
    length_ratio: float         # actual_len / expected_len
    early_stop: bool            # Stopped before min_length
    infinite_loop: bool         # Hit max_length without EOS
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "repetition_rate": self.repetition_rate,
            "distinct_1gram": self.distinct_1gram,
            "distinct_2gram": self.distinct_2gram,
            "distinct_3gram": self.distinct_3gram,
            "avg_entropy": self.avg_entropy,
            "length_ratio": self.length_ratio,
            "early_stop": float(self.early_stop),
            "infinite_loop": float(self.infinite_loop),
        }


@dataclass
class FaithfulnessMetrics:
    """Metrics for prompt-tethering and hallucination detection."""
    entity_precision: float     # Entities in output that are in source
    entity_recall: float        # Entities in source that are in output
    entity_hallucination_rate: float  # Entities in output NOT in source
    copy_rate: float            # N-gram overlap with source (potential over-copying)
    source_coverage: float      # Fraction of source content referenced
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "entity_precision": self.entity_precision,
            "entity_recall": self.entity_recall,
            "entity_hallucination_rate": self.entity_hallucination_rate,
            "copy_rate": self.copy_rate,
            "source_coverage": self.source_coverage,
        }


@dataclass
class MaskComplianceMetrics:
    """Metrics for verifying attention mask correctness."""
    future_leakage: float           # Attention mass to future positions (should be ~0)
    source_attention_mass: float    # Attention from target to source (should be non-trivial)
    per_layer_future_leakage: List[float] = field(default_factory=list)
    per_head_future_leakage: List[List[float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "future_leakage": self.future_leakage,
            "source_attention_mass": self.source_attention_mass,
            "per_layer_future_leakage": self.per_layer_future_leakage,
        }


@dataclass
class PALMScore:
    """Composite PALLM score from evals.yml."""
    faithfulness: float
    hallucination: float
    degeneracy: float
    ppl_drift: float
    composite: float  # Faithfulness - α·Hallucination - β·Degeneracy - γ·PPL_Drift
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "hallucination": self.hallucination,
            "degeneracy": self.degeneracy,
            "ppl_drift": self.ppl_drift,
            "composite": self.composite,
        }


# Loss Decomposition
def compute_loss_decomposition(
    lm_loss: torch.Tensor,
    sae_loss: torch.Tensor,
    combined_loss: torch.Tensor,
) -> LossDecomposition:
    """Decompose and log loss components."""
    lm = lm_loss.item() if lm_loss is not None else 0.0
    sae = sae_loss.item() if sae_loss is not None else 0.0
    total = combined_loss.item() if combined_loss is not None else lm
    sae_ratio = sae / total if total > 0 else 0.0
    
    return LossDecomposition(
        lm_loss=lm,
        sae_loss=sae,
        total_loss=total,
        sae_ratio=sae_ratio,
    )


def compute_perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    return math.exp(min(loss, 100))  # Cap to avoid overflow


# =============================================================================
# Degeneration / Mush Detection
# =============================================================================

def compute_distinct_ngrams(tokens: List[int], n: int) -> float:
    """Compute distinct-n metric: unique n-grams / total n-grams."""
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def compute_repetition_rate(tokens: List[int], n: int = 3) -> float:
    """Compute repetition rate: 1 - distinct_ngrams."""
    return 1.0 - compute_distinct_ngrams(tokens, n)


def compute_token_entropy(logits: torch.Tensor) -> float:
    """Compute average entropy across token positions."""
    # logits: [seq_len, vocab_size]
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # [seq_len]
    return entropy.mean().item()


def detect_length_pathologies(
    generated_len: int,
    expected_len: int,
    min_len: int,
    max_len: int,
    has_eos: bool,
) -> Tuple[float, bool, bool]:
    """Detect length-related generation issues."""
    length_ratio = generated_len / expected_len if expected_len > 0 else 1.0
    early_stop = generated_len < min_len
    infinite_loop = generated_len >= max_len and not has_eos
    return length_ratio, early_stop, infinite_loop


def compute_degeneration_metrics(
    generated_tokens: List[int],
    logits: Optional[torch.Tensor] = None,
    expected_len: int = 50,
    min_len: int = 10,
    max_len: int = 512,
    eos_token_id: Optional[int] = None,
) -> DegenerationMetrics:
    """Compute comprehensive degeneration metrics."""
    has_eos = eos_token_id in generated_tokens if eos_token_id else False
    length_ratio, early_stop, infinite_loop = detect_length_pathologies(
        len(generated_tokens), expected_len, min_len, max_len, has_eos
    )
    
    avg_entropy = 0.0
    if logits is not None:
        avg_entropy = compute_token_entropy(logits)
    
    return DegenerationMetrics(
        repetition_rate=compute_repetition_rate(generated_tokens, n=3),
        distinct_1gram=compute_distinct_ngrams(generated_tokens, 1),
        distinct_2gram=compute_distinct_ngrams(generated_tokens, 2),
        distinct_3gram=compute_distinct_ngrams(generated_tokens, 3),
        avg_entropy=avg_entropy,
        length_ratio=length_ratio,
        early_stop=early_stop,
        infinite_loop=infinite_loop,
    )


# Faithfulness & Hallucination Detection
def extract_entities_simple(text: str) -> set:
    """
    Simple entity extraction: capitalized words, numbers, quoted phrases.
    For production, use spaCy NER.
    """
    entities = set()
    # Capitalized words (potential proper nouns)
    entities.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
    # Numbers
    entities.update(re.findall(r'\b\d+(?:\.\d+)?\b', text))
    # Quoted strings
    entities.update(re.findall(r'"([^"]+)"', text))
    entities.update(re.findall(r"'([^']+)'", text))
    return {e.lower() for e in entities if len(e) > 1}


def compute_ngram_overlap(source: str, output: str, n: int = 3) -> float:
    """Compute n-gram overlap between source and output (copy rate)."""
    def get_ngrams(text: str, n: int) -> set:
        words = text.lower().split()
        if len(words) < n:
            return set()
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}
    
    source_ngrams = get_ngrams(source, n)
    output_ngrams = get_ngrams(output, n)
    
    if not output_ngrams:
        return 0.0
    return len(source_ngrams & output_ngrams) / len(output_ngrams)


def compute_faithfulness_metrics(
    source_text: str,
    output_text: str,
    reference_entities: Optional[set] = None,
) -> FaithfulnessMetrics:
    """Compute faithfulness and hallucination metrics."""
    source_entities = reference_entities or extract_entities_simple(source_text)
    output_entities = extract_entities_simple(output_text)
    
    # Entity precision: what fraction of output entities are in source
    if output_entities:
        entities_in_source = output_entities & source_entities
        entity_precision = len(entities_in_source) / len(output_entities)
        entity_hallucination_rate = 1.0 - entity_precision
    else:
        entity_precision = 1.0
        entity_hallucination_rate = 0.0
    
    # Entity recall: what fraction of source entities appear in output
    if source_entities:
        entity_recall = len(output_entities & source_entities) / len(source_entities)
    else:
        entity_recall = 1.0
    
    # Copy rate: n-gram overlap (detecting over-copying)
    copy_rate = compute_ngram_overlap(source_text, output_text, n=3)
    
    # Source coverage: word-level overlap
    source_words = set(source_text.lower().split())
    output_words = set(output_text.lower().split())
    if source_words:
        source_coverage = len(source_words & output_words) / len(source_words)
    else:
        source_coverage = 0.0
    
    return FaithfulnessMetrics(
        entity_precision=entity_precision,
        entity_recall=entity_recall,
        entity_hallucination_rate=entity_hallucination_rate,
        copy_rate=copy_rate,
        source_coverage=source_coverage,
    )


# Mask Compliance
def compute_mask_compliance(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    source_length: int,
) -> MaskComplianceMetrics:
    """
    Compute attention mask compliance metrics.
    
    Verifies:
    - Target tokens don't attend to future (causal compliance)
    - Target tokens DO attend to source (partial attention purpose)
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    seq_len = input_ids.size(1)
    attention_mask = model.create_bidirectional_attention_mask(input_ids)
    # PALMEmbeddings returns (embeddings, position_ids) for RoPE-compatible models
    emb_out = model.embeddings(input_ids)
    hidden_states = emb_out[0] if isinstance(emb_out, (tuple, list)) else emb_out
    
    future_leakages = []
    source_masses = []
    per_layer_leakage = []
    per_head_leakage = []
    
    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attention
        
        # Compute attention scores manually
        # Note: transpose_for_scores requires num_heads - queries use num_attention_heads, keys use num_kv_heads
        query = attn.transpose_for_scores(attn.query(hidden_states), attn.num_attention_heads)
        key = attn.transpose_for_scores(attn.key(hidden_states), attn.num_kv_heads)
        
        # GQA: Repeat KV heads to match query heads
        from palm.model.attention import repeat_kv
        key = repeat_kv(key, attn.num_kv_groups)
        
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(attn.attention_head_size)
        scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1)  # [batch, heads, seq, seq]
        
        # Per-head analysis
        head_leakages = []
        for h in range(probs.size(1)):
            head_probs = probs[0, h]  # [seq, seq]
            
            # For target positions, check attention to future
            layer_future_leak = 0.0
            layer_source_mass = 0.0
            count = 0
            
            for t in range(source_length, seq_len):
                # Future leakage: attention to positions > t
                if t < seq_len - 1:
                    future_leak = head_probs[t, t+1:].sum().item()
                    layer_future_leak += future_leak
                
                # Source attention mass
                source_mass = head_probs[t, :source_length].sum().item()
                layer_source_mass += source_mass
                count += 1
            
            if count > 0:
                head_leakages.append(layer_future_leak / count)
                layer_future_leak /= count
                layer_source_mass /= count
        
        per_head_leakage.append(head_leakages)
        per_layer_leakage.append(sum(head_leakages) / len(head_leakages) if head_leakages else 0.0)
        future_leakages.append(layer_future_leak)
        source_masses.append(layer_source_mass)
    
    return MaskComplianceMetrics(
        future_leakage=sum(future_leakages) / len(future_leakages) if future_leakages else 0.0,
        source_attention_mass=sum(source_masses) / len(source_masses) if source_masses else 0.0,
        per_layer_future_leakage=per_layer_leakage,
        per_head_future_leakage=per_head_leakage,
    )


# PALLM Composite Score
def compute_palm_score(
    faithfulness: float,
    hallucination: float,
    degeneracy: float,
    ppl_drift: float,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
) -> PALMScore:
    """
    Compute composite PALLM score.
    
    PALLM Score = Faithfulness - α·Hallucination - β·Degeneracy - γ·PPL_Drift
    """
    composite = faithfulness - alpha * hallucination - beta * degeneracy - gamma * ppl_drift
    
    return PALMScore(
        faithfulness=faithfulness,
        hallucination=hallucination,
        degeneracy=degeneracy,
        ppl_drift=ppl_drift,
        composite=composite,
    )


# Ablation Evaluation Runner
@dataclass
class AblationResult:
    """Results for a single ablation configuration."""
    config_name: str
    perplexity: float
    faithfulness: FaithfulnessMetrics
    degeneration: DegenerationMetrics
    mask_compliance: Optional[MaskComplianceMetrics]
    palm_score: PALMScore


class AblationEvaluator:
    """
    Run evaluation across ablation configurations:
    - Base model (no changes)
    - SAE only
    - Partial attention only
    - SAE + Partial attention (full method)
    """
    
    def __init__(self, base_model, tokenizer, eval_prompts: List[Dict[str, str]]):
        """
        Args:
            base_model: The PALM model to evaluate
            tokenizer: Tokenizer for text processing
            eval_prompts: List of {"source": ..., "expected": ...} dicts
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
    
    def evaluate_single_config(
        self,
        model: torch.nn.Module,
        config_name: str,
        use_partial_attention: bool = True,
        use_sae: bool = True,
        reference_ppl: Optional[float] = None,
    ) -> AblationResult:
        """Evaluate a single ablation configuration."""
        model.eval()
        device = next(model.parameters()).device
        
        all_faithfulness = []
        all_degeneration = []
        total_loss = 0.0
        num_samples = 0
        
        for prompt_data in self.eval_prompts:
            source = prompt_data["source"]
            expected = prompt_data.get("expected", "")
            
            # Encode and generate
            input_ids = self.tokenizer.encode(source, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Get loss for perplexity
                if expected:
                    full_text = source + expected
                    full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(device)
                    source_len = torch.tensor([input_ids.size(1)])
                    labels = full_ids.clone()
                    labels[:, :input_ids.size(1)] = -100
                    
                    outputs = model(full_ids, labels=labels, source_len=source_len if use_sae else None)
                    loss = outputs[2]
                    if loss is not None:
                        total_loss += loss.item()
                        num_samples += 1
                
                # Generate
                generated = model.generate(
                    input_ids,
                    max_length=input_ids.size(1) + 100,
                    do_sample=False,
                )
            
            generated_text = self.tokenizer.decode(
                generated[0, input_ids.size(1):], 
                skip_special_tokens=True
            )
            
            # Compute metrics
            faithfulness = compute_faithfulness_metrics(source, generated_text)
            all_faithfulness.append(faithfulness)
            
            generated_tokens = generated[0, input_ids.size(1):].tolist()
            degeneration = compute_degeneration_metrics(
                generated_tokens,
                expected_len=len(expected.split()) if expected else 50,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            all_degeneration.append(degeneration)
        
        # Aggregate metrics
        avg_ppl = compute_perplexity(total_loss / num_samples) if num_samples > 0 else float('inf')
        ppl_drift = avg_ppl - reference_ppl if reference_ppl else 0.0
        
        avg_faithfulness = FaithfulnessMetrics(
            entity_precision=sum(f.entity_precision for f in all_faithfulness) / len(all_faithfulness),
            entity_recall=sum(f.entity_recall for f in all_faithfulness) / len(all_faithfulness),
            entity_hallucination_rate=sum(f.entity_hallucination_rate for f in all_faithfulness) / len(all_faithfulness),
            copy_rate=sum(f.copy_rate for f in all_faithfulness) / len(all_faithfulness),
            source_coverage=sum(f.source_coverage for f in all_faithfulness) / len(all_faithfulness),
        )
        
        avg_degeneration = DegenerationMetrics(
            repetition_rate=sum(d.repetition_rate for d in all_degeneration) / len(all_degeneration),
            distinct_1gram=sum(d.distinct_1gram for d in all_degeneration) / len(all_degeneration),
            distinct_2gram=sum(d.distinct_2gram for d in all_degeneration) / len(all_degeneration),
            distinct_3gram=sum(d.distinct_3gram for d in all_degeneration) / len(all_degeneration),
            avg_entropy=sum(d.avg_entropy for d in all_degeneration) / len(all_degeneration),
            length_ratio=sum(d.length_ratio for d in all_degeneration) / len(all_degeneration),
            early_stop=any(d.early_stop for d in all_degeneration),
            infinite_loop=any(d.infinite_loop for d in all_degeneration),
        )
        
        # Mask compliance (only for full model)
        mask_compliance = None
        if use_partial_attention and len(self.eval_prompts) > 0:
            source = self.eval_prompts[0]["source"]
            input_ids = self.tokenizer.encode(source, return_tensors="pt").to(device)
            mask_compliance = compute_mask_compliance(model, input_ids, input_ids.size(1))
        
        # Compute PALM score
        palm_score = compute_palm_score(
            faithfulness=avg_faithfulness.entity_precision,
            hallucination=avg_faithfulness.entity_hallucination_rate,
            degeneracy=avg_degeneration.repetition_rate,
            ppl_drift=max(0, ppl_drift),
        )
        
        return AblationResult(
            config_name=config_name,
            perplexity=avg_ppl,
            faithfulness=avg_faithfulness,
            degeneration=avg_degeneration,
            mask_compliance=mask_compliance,
            palm_score=palm_score,
        )
    
    def run_ablation_grid(self) -> Dict[str, AblationResult]:
        """
        Run full ablation grid evaluation.
        
        Returns dict with results for:
        - "base": No modifications (SAE weight = 0, standard attention)
        - "sae_only": SAE enabled, standard attention
        - "partial_only": Partial attention, SAE disabled
        - "full": Both SAE and partial attention
        """
        results = {}
        
        # Store original config
        original_sae_weight = self.base_model.sae_weight
        
        # 1. Base model (disable SAE)
        self.base_model.sae_weight = 0.0
        results["base"] = self.evaluate_single_config(
            self.base_model, "base",
            use_partial_attention=False,
            use_sae=False,
        )
        reference_ppl = results["base"].perplexity
        
        # 2. SAE only
        self.base_model.sae_weight = original_sae_weight
        results["sae_only"] = self.evaluate_single_config(
            self.base_model, "sae_only",
            use_partial_attention=False,
            use_sae=True,
            reference_ppl=reference_ppl,
        )
        
        # 3. Partial attention only
        self.base_model.sae_weight = 0.0
        results["partial_only"] = self.evaluate_single_config(
            self.base_model, "partial_only",
            use_partial_attention=True,
            use_sae=False,
            reference_ppl=reference_ppl,
        )
        
        # 4. Full method
        self.base_model.sae_weight = original_sae_weight
        results["full"] = self.evaluate_single_config(
            self.base_model, "full",
            use_partial_attention=True,
            use_sae=True,
            reference_ppl=reference_ppl,
        )
        
        return results
    
    def format_results(self, results: Dict[str, AblationResult]) -> str:
        """Format ablation results as a readable table."""
        lines = [
            "=" * 80,
            "PALM Ablation Evaluation Results",
            "=" * 80,
            "",
            f"{'Config':<15} {'PPL':>8} {'Faith':>8} {'Halluc':>8} {'Repet':>8} {'PALM':>8}",
            "-" * 80,
        ]
        
        for name, result in results.items():
            lines.append(
                f"{name:<15} "
                f"{result.perplexity:>8.2f} "
                f"{result.faithfulness.entity_precision:>8.3f} "
                f"{result.faithfulness.entity_hallucination_rate:>8.3f} "
                f"{result.degeneration.repetition_rate:>8.3f} "
                f"{result.palm_score.composite:>8.3f}"
            )
        
        lines.extend(["", "=" * 80])
        return "\n".join(lines)


# Smoke Test Generation Comparison
def compare_generations_cached_vs_uncached(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    """
    Compare generations with use_cache=True vs False.
    
    Returns dict with:
    - matches: number of prompts with identical outputs
    - mismatches: list of (prompt, cached_output, uncached_output)
    """
    model.eval()
    device = next(model.parameters()).device
    
    matches = 0
    mismatches = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            cached = model.generate(
                input_ids.clone(),
                max_length=input_ids.size(1) + max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            uncached = model.generate(
                input_ids.clone(),
                max_length=input_ids.size(1) + max_new_tokens,
                do_sample=False,
                use_cache=False,
            )
        
        if torch.all(cached == uncached):
            matches += 1
        else:
            cached_text = tokenizer.decode(cached[0], skip_special_tokens=True)
            uncached_text = tokenizer.decode(uncached[0], skip_special_tokens=True)
            mismatches.append({
                "prompt": prompt,
                "cached": cached_text,
                "uncached": uncached_text,
            })
    
    return {
        "total": len(prompts),
        "matches": matches,
        "match_rate": matches / len(prompts) if prompts else 1.0,
        "mismatches": mismatches,
    }


# QA-based Faithfulness Evaluation
def evaluate_answerable_from_source(
    model: torch.nn.Module,
    tokenizer,
    qa_examples: List[Dict[str, str]],
) -> Dict[str, float]:
    """
    Evaluate if model answers are faithful to provided passages.
    
    qa_examples: List of {"passage": ..., "question": ..., "answer": ...}
    
    Returns accuracy and hallucination rate.
    """
    model.eval()
    device = next(model.parameters()).device
    
    correct = 0
    hallucinated = 0
    total = len(qa_examples)
    
    for example in qa_examples:
        passage = example["passage"]
        question = example["question"]
        expected_answer = example["answer"].lower()
        
        prompt = f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=input_ids.size(1) + 50,
                do_sample=False,
            )
        
        answer = tokenizer.decode(generated[0, input_ids.size(1):], skip_special_tokens=True)
        answer_lower = answer.lower().strip()
        
        # Check if expected answer is in generated answer
        if expected_answer in answer_lower:
            correct += 1
        
        # Check for hallucinated entities
        answer_entities = extract_entities_simple(answer)
        passage_entities = extract_entities_simple(passage)
        question_entities = extract_entities_simple(question)
        allowed_entities = passage_entities | question_entities
        
        hallucinated_entities = answer_entities - allowed_entities
        if hallucinated_entities:
            hallucinated += 1
    
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "hallucination_rate": hallucinated / total if total > 0 else 0.0,
        "correct": correct,
        "hallucinated": hallucinated,
        "total": total,
    }


def evaluate_distractor_swap(
    model: torch.nn.Module,
    tokenizer,
    distractor_examples: List[Dict[str, str]],
) -> Dict[str, float]:
    """
    Evaluate if model correctly uses the right passage when distractors are present.
    
    distractor_examples: List of {
        "correct_passage": ...,
        "distractor_passage": ...,
        "question": ...,
        "answer": ...,
    }
    """
    model.eval()
    device = next(model.parameters()).device
    
    correct_first = 0
    correct_second = 0
    total = len(distractor_examples)
    
    for example in distractor_examples:
        correct_passage = example["correct_passage"]
        distractor = example["distractor_passage"]
        question = example["question"]
        expected = example["answer"].lower()
        
        # Test with correct passage first
        prompt1 = f"Passage A: {correct_passage}\n\nPassage B: {distractor}\n\nQuestion: {question}\n\nAnswer:"
        input_ids1 = tokenizer.encode(prompt1, return_tensors="pt").to(device)
        
        # Test with distractor first
        prompt2 = f"Passage A: {distractor}\n\nPassage B: {correct_passage}\n\nQuestion: {question}\n\nAnswer:"
        input_ids2 = tokenizer.encode(prompt2, return_tensors="pt").to(device)
        
        with torch.no_grad():
            gen1 = model.generate(input_ids1, max_length=input_ids1.size(1) + 50, do_sample=False)
            gen2 = model.generate(input_ids2, max_length=input_ids2.size(1) + 50, do_sample=False)
        
        answer1 = tokenizer.decode(gen1[0, input_ids1.size(1):], skip_special_tokens=True).lower()
        answer2 = tokenizer.decode(gen2[0, input_ids2.size(1):], skip_special_tokens=True).lower()
        
        if expected in answer1:
            correct_first += 1
        if expected in answer2:
            correct_second += 1
    
    return {
        "accuracy_correct_first": correct_first / total if total > 0 else 0.0,
        "accuracy_distractor_first": correct_second / total if total > 0 else 0.0,
        "consistency": (correct_first == correct_second) / total if total > 0 else 0.0,
        "total": total,
    }
