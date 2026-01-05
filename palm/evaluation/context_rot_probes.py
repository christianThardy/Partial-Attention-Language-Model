"""
Lightweight Context Rot Probes for During-Training Evaluation

These are fast (~30s) probes that run every epoch to catch context rot early.
For full evaluation, use context_rot.py after training.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import time
import torch
from tqdm import tqdm


@dataclass
class ContextRotProbeResult:
    """Lightweight probe results for single epoch."""
    # Focused vs Full (quick 2-sample test)
    focused_correct: int
    full_correct: int
    focused_full_delta: float  # focused - full, positive = length hurts
    
    # Distractor (quick 2-sample test)
    distractor_correct: int
    distractor_total: int
    distractor_accuracy: float
    
    # Repeated Words (quick check at 2 lengths)
    repeated_words_short_score: float  # ~50 words
    repeated_words_long_score: float   # ~200 words
    repeated_words_degradation: float  # short - long (positive = degrading)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "focused_full_delta": self.focused_full_delta,
            "distractor_accuracy": self.distractor_accuracy,
            "focused_correct": self.focused_correct,
            "full_correct": self.full_correct,
            "repeated_words_short": self.repeated_words_short_score,
            "repeated_words_long": self.repeated_words_long_score,
            "repeated_words_degradation": self.repeated_words_degradation,
        }


def _levenshtein_score(gold: str, pred: str) -> float:
    """Quick Levenshtein similarity score."""
    if not gold or not pred:
        return 0.0
    # Use simple ratio for speed
    if len(gold) < len(pred):
        gold, pred = pred, gold
    if len(gold) == 0:
        return 1.0
    
    # Count matching characters in order (simplified)
    matches = 0
    pred_idx = 0
    for char in gold:
        if pred_idx < len(pred) and char == pred[pred_idx]:
            matches += 1
            pred_idx += 1
    
    return matches / len(gold)


# Quick test cases (subset for speed)
QUICK_NEEDLE_TESTS = [
    {
        "needle": "The access code for the server room is 8472.",
        "question": "What is the access code for the server room?",
        "answer": "8472",
    },
    {
        "needle": "The project deadline was moved to March 15th by the client.",
        "question": "When is the project deadline?",
        "answer": "March 15th",
    },
]

QUICK_DISTRACTORS = [
    "The backup code for the storage facility is 3921.",
    "The original deadline was February 28th before the extension.",
]

FILLER_TEXT = """The most important thing about building systems is understanding the tradeoffs involved. 
Every decision has consequences, and the best engineers are those who can anticipate these consequences 
and design accordingly. This requires both technical skill and the wisdom that comes from experience.

When working on complex projects, it's easy to get lost in the details and forget the bigger picture. 
Regular check-ins and documentation help maintain perspective. The goal is not perfection but progress, 
and sometimes good enough is better than perfect if it ships on time.

Communication is often undervalued in technical work. The best code in the world is useless if no one 
understands how to use it or why it was written that way. Clear documentation and thoughtful API design 
are investments that pay dividends over the lifetime of a project."""


def run_context_rot_probes(
    model: torch.nn.Module,
    tokenizer,
    device: Optional[str] = None,
    max_gen_tokens: int = 32,
    show_progress: bool = True,
) -> ContextRotProbeResult:
    """
    Run lightweight context rot probes (~30 seconds).
    
    Designed to run every epoch during training to catch issues early.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        device: Device to run on (defaults to model's device)
        max_gen_tokens: Maximum tokens to generate per sample
        show_progress: Whether to show tqdm progress bar
    
    Returns:
        ContextRotProbeResult with quick metrics
    """
    device = device or next(model.parameters()).device
    model.eval()
    
    # Track generation times for diagnostics
    gen_times = []
    
    def generate(prompt: str, desc: str = "") -> str:
        start = time.time()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        elapsed = time.time() - start
        gen_times.append((desc, elapsed, inputs["input_ids"].shape[1]))
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip().lower()
    
    def check_answer(response: str, expected: str) -> bool:
        return expected.lower() in response.lower()
    
    # Total probes: 2 focused + 2 full + 2 distractor + 2 repeated = 8
    total_probes = 8
    probe_iter = range(total_probes)
    if show_progress:
        probe_iter = tqdm(probe_iter, desc="Context Rot Probes", leave=False)
    probe_idx = iter(probe_iter)
    
    # Probe 1: Focused vs Full Context (2 samples)
    focused_correct = 0
    full_correct = 0
    
    for test in QUICK_NEEDLE_TESTS:
        # Focused (just the needle)
        next(probe_idx)  # Advance progress
        focused_prompt = f"""Context: {test["needle"]}

Question: {test["question"]}

Answer:"""
        
        focused_response = generate(focused_prompt, "focused")
        if check_answer(focused_response, test["answer"]):
            focused_correct += 1
        
        # Full (needle buried in filler)
        next(probe_idx)  # Advance progress
        full_context = f"{FILLER_TEXT}\n\n{test['needle']}\n\n{FILLER_TEXT}"
        full_prompt = f"""Context: {full_context}

Question: {test["question"]}

Answer:"""
        
        full_response = generate(full_prompt, "full_context")
        if check_answer(full_response, test["answer"]):
            full_correct += 1
    
    focused_full_delta = (focused_correct - full_correct) / len(QUICK_NEEDLE_TESTS)
    
    # Probe 2: Distractor Confusion (2 samples with 1 distractor each)
    distractor_correct = 0
    
    for i, test in enumerate(QUICK_NEEDLE_TESTS):
        next(probe_idx)  # Advance progress
        distractor = QUICK_DISTRACTORS[i % len(QUICK_DISTRACTORS)]
        
        # Needle with distractor
        context = f"{distractor}\n\n{FILLER_TEXT[:200]}\n\n{test['needle']}"
        
        prompt = f"""Context: {context}

Question: {test["question"]}

Answer with only the specific information asked:"""
        
        response = generate(prompt, "distractor")
        if check_answer(response, test["answer"]):
            distractor_correct += 1
    
    distractor_accuracy = distractor_correct / len(QUICK_NEEDLE_TESTS)
    
    # Probe 3: Repeated Words (quick 2-length test)
    def test_repeated_words(num_words: int, probe_name: str) -> float:
        """Test at a specific word count."""
        common = "apple"
        modified = "apples"
        modified_pos = num_words // 2
        
        words = [common] * num_words
        words[modified_pos] = modified
        gold_text = " ".join(words)
        
        prompt = f"""Reproduce the following text exactly:

{gold_text}

Reproduced:"""
        
        # Need enough tokens
        output = generate(prompt, probe_name)
        return _levenshtein_score(gold_text, output)
    
    # Test at short (50 words) and long (200 words)
    next(probe_idx)  # Advance progress
    repeated_short = test_repeated_words(50, "repeat_50")
    next(probe_idx)  # Advance progress
    repeated_long = test_repeated_words(200, "repeat_200")
    repeated_degradation = repeated_short - repeated_long
    
    # Log timing summary if any generations were slow
    if gen_times and show_progress:
        total_time = sum(t[1] for t in gen_times)
        slowest = max(gen_times, key=lambda x: x[1])
        if total_time > 60:  # Only warn if surprisingly slow
            tqdm.write(f"  ⏱️ Probe timing: {total_time:.1f}s total, slowest={slowest[0]} ({slowest[1]:.1f}s, {slowest[2]} input tokens)")
    
    return ContextRotProbeResult(
        focused_correct=focused_correct,
        full_correct=full_correct,
        focused_full_delta=focused_full_delta,
        distractor_correct=distractor_correct,
        distractor_total=len(QUICK_NEEDLE_TESTS),
        distractor_accuracy=distractor_accuracy,
        repeated_words_short_score=repeated_short,
        repeated_words_long_score=repeated_long,
        repeated_words_degradation=repeated_degradation,
    )


def format_probe_results(result: ContextRotProbeResult) -> str:
    """One-line summary for training logs."""
    return (
        f"Context Rot Probe: "
        f"focused_full_delta={result.focused_full_delta:+.2f}, "
        f"distractor_acc={result.distractor_accuracy:.1%}, "
        f"repeat_lev={result.repeated_words_short_score:.2f}->{result.repeated_words_long_score:.2f}"
    )
