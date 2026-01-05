"""
Context Rot Evaluation Suite for PALM

Based on Kelly Hong's research at Chroma (research.trychroma.com):
- Models degrade with increasing context length
- Semantic matching harder than lexical matching
- Distractors cause uncertainty and hallucination
- Shuffled haystacks can paradoxically help

This module implements tests to validate PALM's resistance to context rot.

NEW: Repeated Words Test (from Chroma's context-rot repo)
- Model must replicate a sequence of repeated words with one modified word
- Uses Normalized Levenshtein Score to measure exact reproduction
- Tracks performance degradation across input lengths
"""

import random
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import json
import numpy as np

import torch
import torch.nn.functional as F

# Levenshtein distance (inline to avoid dependency)
def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_levenshtein_score(gold: str, pred: str) -> float:
    """
    Compute normalized Levenshtein score (1 = perfect match, 0 = completely different).
    
    From Chroma's context-rot: 1 - (edit_distance / max_len)
    """
    if not gold or not pred:
        return 0.0
    distance = _levenshtein_distance(gold, pred)
    max_len = max(len(gold), len(pred))
    return 1.0 - (distance / max_len)


# TEST DATA: Needles, Questions, and Distractors
# Lexical needle-question pairs (direct word overlap)
LEXICAL_NEEDLES = [
    {
        "needle": "The secret code for the vault is 7429.",
        "question": "What is the secret code for the vault?",
        "answer": "7429",
        "type": "lexical",
    },
    {
        "needle": "The meeting is scheduled for Tuesday at 3pm in Room 204.",
        "question": "When and where is the meeting scheduled?",
        "answer": "Tuesday at 3pm in Room 204",
        "type": "lexical",
    },
    {
        "needle": "Dr. Sarah Chen discovered the high-temperature superconductor in 2019.",
        "question": "Who discovered the high-temperature superconductor and when?",
        "answer": "Dr. Sarah Chen in 2019",
        "type": "lexical",
    },
]

# Semantic needle-question pairs (no direct word overlap)
SEMANTIC_NEEDLES = [
    {
        "needle": "I had an interesting friend from my college humanities courses who wrote daily and advised me to try writing at least once a week. Looking back, it's the most useful habit I've developed.",
        "question": "What was the best writing advice you got from your college classmate?",
        "answer": "to write at least once a week",
        "type": "semantic",
    },
    {
        "needle": "The quarterly financial review revealed that our European operations contributed 45 million, with Asian markets adding another 38 million to the total.",
        "question": "How much revenue came from international markets?",
        "answer": "83 million (45 from Europe + 38 from Asia)",
        "type": "semantic",
    },
    {
        "needle": "After extensive testing, the engineering team determined that the thermal threshold should never exceed 85 degrees Celsius during normal operation.",
        "question": "What's the maximum safe operating temperature?",
        "answer": "85 degrees Celsius",
        "type": "semantic",
    },
]

# Distractor variants (similar but wrong)
def generate_distractors(needle: Dict, num_distractors: int = 4) -> List[str]:
    """Generate distractors that are similar but incorrect."""
    distractors = []
    
    if "code" in needle["needle"].lower():
        codes = ["8531", "2947", "6183", "9024"]
        for code in codes[:num_distractors]:
            distractors.append(f"The backup code for the storage is {code}.")
    
    elif "meeting" in needle["needle"].lower():
        variants = [
            "The conference is planned for Monday at 2pm in Room 105.",
            "The workshop was held on Wednesday at 4pm in the main hall.",
            "The review session is set for Friday at 10am in Room 302.",
            "The standup happens daily at 9am in the open area.",
        ]
        distractors = variants[:num_distractors]
    
    elif "superconductor" in needle["needle"].lower():
        variants = [
            "Prof. Michael Lee published groundbreaking work on semiconductors in 2018.",
            "The low-temperature superconductor was theorized by Dr. James Wong in 2020.",
            "Dr. Sarah Chen's earlier work on insulators was completed in 2015.",
            "The research team at MIT announced a superconductor breakthrough in 2021.",
        ]
        distractors = variants[:num_distractors]
    
    elif "writing" in needle["needle"].lower():
        variants = [
            "My professor always said the key to good writing is reading extensively every day.",
            "A famous author once mentioned that writing in the morning produces the best work.",
            "The worst writing tip I received was to edit while writing the first draft.",
            "My high school teacher recommended writing for at least two hours daily.",
        ]
        distractors = variants[:num_distractors]
    
    elif "financial" in needle["needle"].lower() or "revenue" in needle["question"].lower():
        variants = [
            "The domestic operations showed a revenue of 52 million this quarter.",
            "Last year's European contribution was 41 million with Asia at 35 million.",
            "Projected international revenue for next quarter is estimated at 90 million.",
            "The financial audit noted discrepancies of 3 million in the Asian accounts.",
        ]
        distractors = variants[:num_distractors]
    
    elif "temperature" in needle["needle"].lower() or "thermal" in needle["needle"].lower():
        variants = [
            "The cooling system activates when ambient temperature reaches 75 degrees.",
            "Previous models had a thermal limit of 90 degrees Celsius.",
            "The stress test showed components failing at 95 degrees Celsius.",
            "Optimal performance occurs between 60 and 70 degrees Celsius.",
        ]
        distractors = variants[:num_distractors]
    
    else:
        # Generic distractors
        distractors = [
            "This is some related but incorrect information.",
            "Another piece of similar but wrong content.",
            "Yet more plausible but inaccurate details.",
            "Additional misleading but believable text.",
        ][:num_distractors]
    
    return distractors


# Haystack filler content (Paul Graham essays style - neutral content)
HAYSTACK_PARAGRAPHS = [
    "The most dangerous thing about technology is how it changes our habits of thought. We don't notice these changes because they happen gradually, like the proverbial frog in boiling water.",
    "Startups are not just small versions of big companies. They're a fundamentally different type of organization, optimized for different goals. Big companies optimize for efficiency; startups optimize for learning.",
    "The best founders are people who genuinely want to solve a problem, not people who want to start a company. The desire to start a company is often a warning sign.",
    "Programming is like writing in that both are forms of thinking on paper. But unlike writing, programming gives you immediate feedback. You can run your program and see if it works.",
    "The reason we have so many unsuccessful startups is that starting a startup has become too easy. This sounds like a paradox, but it makes sense when you think about it.",
    "One of the most valuable skills you can have is the ability to explain technical concepts to non-technical people. This is harder than it sounds because it requires understanding both the concept and the audience.",
    "The future is already here, it's just not evenly distributed. This observation by William Gibson is becoming more true every year as technology creates wider gaps between early adopters and everyone else.",
    "Reading old books is valuable because it lets you see which ideas have stood the test of time. If something was written 200 years ago and is still relevant, it's probably onto something important.",
    "The best way to predict the future is to invent it. But the second best way is to pay attention to what's working in small pockets and imagine it spreading.",
    "Most people overestimate what they can do in a day and underestimate what they can do in a year. This is why consistent small efforts compound into large achievements.",
]


# RESULT DATACLASSES
@dataclass
class RepeatedWordsResult:
    """Result of a single repeated words test."""
    num_words: int  # Total word count
    modified_word_position: int  # Position of the modified word (0-indexed)
    token_count: int  # Input length in tokens
    gold_text: str  # Expected output
    model_output: str  # What the model produced
    levenshtein_score: float  # 0-1, how close the match was
    modified_word_present: bool  # Did output contain the modified word
    correct_position: bool  # Was modified word at correct position


@dataclass
class RepeatedWordsSummary:
    """Summary of repeated words test across input lengths."""
    # Score by input token count bin
    score_by_token_bin: Dict[str, float]  # {"100-500": 0.95, "500-1000": 0.90, ...}
    # Overall metrics
    overall_levenshtein: float
    modified_word_accuracy: float
    position_accuracy: float
    # For plotting
    token_bins: List[int]  # Bin centers (for x-axis)
    avg_scores: List[float]  # Average scores (for y-axis)
    
    def get_degradation_slope(self) -> float:
        """Compute slope of degradation curve (negative = degrading)."""
        if len(self.token_bins) < 2:
            return 0.0
        # Linear regression on log-transformed token counts
        log_tokens = [math.log10(max(t, 1)) for t in self.token_bins]
        n = len(log_tokens)
        mean_x = sum(log_tokens) / n
        mean_y = sum(self.avg_scores) / n
        
        numerator = sum((log_tokens[i] - mean_x) * (self.avg_scores[i] - mean_y) for i in range(n))
        denominator = sum((log_tokens[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        return numerator / denominator


@dataclass
class NeedleTestResult:
    """Result of a single needle-in-haystack test."""
    needle_type: str  # "lexical" or "semantic"
    needle_position: float  # 0.0 = start, 1.0 = end
    haystack_length: int  # in tokens
    question: str
    expected_answer: str
    model_answer: str
    answer_found: bool
    confidence: float  # 0-1, how confident we are the answer is correct


@dataclass
class DistractorTestResult:
    """Result of a distractor confusion test."""
    num_distractors: int
    correct_answer: str
    model_answer: str
    picked_distractor: bool
    abstained: bool
    correct: bool


@dataclass
class FocusedVsFullResult:
    """Compare performance on focused vs full context."""
    focused_correct: bool
    focused_answer: str
    full_correct: bool
    full_answer: str
    performance_delta: float  # focused_score - full_score


@dataclass 
class ContextRotSuiteResult:
    """Full context rot evaluation results."""
    # Needle-in-haystack
    lexical_needle_accuracy: float
    semantic_needle_accuracy: float
    needle_results: List[NeedleTestResult]
    
    # Distractor tests
    distractor_accuracy_by_count: Dict[int, float]  # {0: 0.95, 1: 0.85, 4: 0.70}
    distractor_results: List[DistractorTestResult]
    
    # Focused vs Full
    focused_accuracy: float
    full_accuracy: float
    length_degradation: float  # How much worse is full vs focused
    focused_vs_full_results: List[FocusedVsFullResult]
    
    # Repeated Words (NEW - from Chroma context-rot)
    repeated_words_summary: Optional[RepeatedWordsSummary] = None
    repeated_words_results: Optional[List[RepeatedWordsResult]] = None
    
    # Summary
    context_rot_resistance_score: float = 0.0  # 0-1 composite
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "lexical_needle_accuracy": self.lexical_needle_accuracy,
            "semantic_needle_accuracy": self.semantic_needle_accuracy,
            "needle_accuracy_delta": self.lexical_needle_accuracy - self.semantic_needle_accuracy,
            "distractor_accuracy_0": self.distractor_accuracy_by_count.get(0, 0),
            "distractor_accuracy_1": self.distractor_accuracy_by_count.get(1, 0),
            "distractor_accuracy_4": self.distractor_accuracy_by_count.get(4, 0),
            "distractor_degradation": self.distractor_accuracy_by_count.get(0, 0) - self.distractor_accuracy_by_count.get(4, 0),
            "focused_accuracy": self.focused_accuracy,
            "full_accuracy": self.full_accuracy,
            "length_degradation": self.length_degradation,
            "context_rot_resistance_score": self.context_rot_resistance_score,
        }
        
        # Add repeated words metrics if available
        if self.repeated_words_summary:
            rw = self.repeated_words_summary
            d["repeated_words_overall_levenshtein"] = rw.overall_levenshtein
            d["repeated_words_degradation_slope"] = rw.get_degradation_slope()
            d["repeated_words_modified_word_accuracy"] = rw.modified_word_accuracy
            # Add per-bin scores
            for bin_label, score in rw.score_by_token_bin.items():
                d[f"repeated_words_{bin_label}"] = score
        
        return d


# CONTEXT ROT EVALUATOR
class ContextRotEvaluator:
    """
    Evaluates model resistance to context rot using tests from Chroma research.
    
    Tests:
    1. Lexical vs Semantic Needle: Can model find info with/without word overlap?
    2. Distractor Confusion: Does model pick wrong answer when similar facts present?
    3. Focused vs Full: Does performance degrade with irrelevant context?
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        max_gen_tokens: int = 64,
        haystack_base_tokens: int = 256,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_gen_tokens = max_gen_tokens
        self.haystack_base_tokens = haystack_base_tokens
        self.device = device or next(model.parameters()).device
        
    def _generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate text from prompt."""
        max_tokens = max_tokens or self.max_gen_tokens
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_tokens,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    def _build_haystack(
        self,
        needle: str,
        position: float = 0.5,
        num_tokens: int = 256,
        distractors: List[str] = None,
    ) -> str:
        """
        Build a haystack with needle inserted at given position.
        
        Args:
            needle: The fact to insert
            position: 0.0 = start, 0.5 = middle, 1.0 = end
            num_tokens: Approximate haystack size
            distractors: Optional list of distractor sentences to include
        """
        # Build filler paragraphs
        filler = []
        current_tokens = 0
        para_idx = 0
        
        while current_tokens < num_tokens:
            para = HAYSTACK_PARAGRAPHS[para_idx % len(HAYSTACK_PARAGRAPHS)]
            filler.append(para)
            current_tokens += len(self.tokenizer.encode(para))
            para_idx += 1
        
        # Insert distractors randomly if provided
        if distractors:
            for d in distractors:
                insert_pos = random.randint(0, len(filler))
                filler.insert(insert_pos, d)
        
        # Insert needle at position
        needle_idx = int(len(filler) * position)
        filler.insert(needle_idx, needle)
        
        return "\n\n".join(filler)
    
    def _check_answer(self, model_answer: str, expected: str, question: str) -> Tuple[bool, float]:
        """Check if model answer contains expected answer."""
        model_lower = model_answer.lower()
        expected_lower = expected.lower()
        
        # Direct containment
        if expected_lower in model_lower:
            return True, 1.0
        
        # Check for key numbers/entities
        expected_tokens = set(re.findall(r'\b\w+\b', expected_lower))
        model_tokens = set(re.findall(r'\b\w+\b', model_lower))
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'was', 'were', 'in', 'at', 'on', 'to', 'for'}
        expected_tokens -= stopwords
        model_tokens -= stopwords
        
        if not expected_tokens:
            return False, 0.0
            
        overlap = len(expected_tokens & model_tokens) / len(expected_tokens)
        return overlap > 0.6, overlap
    
    def _check_abstention(self, answer: str) -> bool:
        """Check if model abstained from answering."""
        abstention_phrases = [
            "i don't know",
            "i cannot find",
            "not mentioned",
            "no information",
            "unable to determine",
            "not specified",
            "cannot answer",
            "not provided",
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in abstention_phrases)
    
    # TEST 1: Needle-in-Haystack (Lexical vs Semantic)
    def test_needle_in_haystack(
        self,
        needle_type: str = "both",  # "lexical", "semantic", or "both"
        positions: List[float] = [0.1, 0.5, 0.9],
        num_samples: int = 6,
    ) -> List[NeedleTestResult]:
        """
        Test needle retrieval with lexical vs semantic matching.
        
        Lexical: Question words appear in needle
        Semantic: Must infer connection (no word overlap)
        """
        results = []
        
        needles = []
        if needle_type in ["lexical", "both"]:
            needles.extend(LEXICAL_NEEDLES)
        if needle_type in ["semantic", "both"]:
            needles.extend(SEMANTIC_NEEDLES)
        
        for needle_data in needles[:num_samples]:
            for position in positions:
                # Build haystack
                haystack = self._build_haystack(
                    needle=needle_data["needle"],
                    position=position,
                    num_tokens=self.haystack_base_tokens,
                )
                
                # Build prompt
                prompt = f"""Context:
{haystack}

Question: {needle_data["question"]}

Answer:"""
                
                # Generate
                answer = self._generate(prompt)
                
                # Check
                found, confidence = self._check_answer(
                    answer, needle_data["answer"], needle_data["question"]
                )
                
                results.append(NeedleTestResult(
                    needle_type=needle_data["type"],
                    needle_position=position,
                    haystack_length=len(self.tokenizer.encode(haystack)),
                    question=needle_data["question"],
                    expected_answer=needle_data["answer"],
                    model_answer=answer,
                    answer_found=found,
                    confidence=confidence,
                ))
        
        return results
    
    # TEST 2: Distractor Confusion
    def test_distractor_confusion(
        self,
        distractor_counts: List[int] = [0, 1, 4],
        num_samples: int = 6,
    ) -> List[DistractorTestResult]:
        """
        Test if model picks correct answer when distractors present.
        
        Distractors are semantically similar but factually wrong.
        """
        results = []
        
        all_needles = LEXICAL_NEEDLES + SEMANTIC_NEEDLES
        
        for needle_data in all_needles[:num_samples]:
            for num_distractors in distractor_counts:
                # Generate distractors
                distractors = []
                if num_distractors > 0:
                    distractors = generate_distractors(needle_data, num_distractors)
                
                # Build haystack
                haystack = self._build_haystack(
                    needle=needle_data["needle"],
                    position=0.5,
                    num_tokens=self.haystack_base_tokens,
                    distractors=distractors,
                )
                
                # Build prompt
                prompt = f"""Context:
{haystack}

Question: {needle_data["question"]}

Answer with only the specific information asked. If unsure, say "I cannot determine from the context."

Answer:"""
                
                # Generate
                answer = self._generate(prompt)
                
                # Check
                correct, _ = self._check_answer(
                    answer, needle_data["answer"], needle_data["question"]
                )
                abstained = self._check_abstention(answer)
                
                # Check if picked a distractor
                picked_distractor = False
                if not correct and not abstained:
                    for d in distractors:
                        # Check if distractor content appears in answer
                        if any(word in answer.lower() for word in d.lower().split()[:5]):
                            picked_distractor = True
                            break
                
                results.append(DistractorTestResult(
                    num_distractors=num_distractors,
                    correct_answer=needle_data["answer"],
                    model_answer=answer,
                    picked_distractor=picked_distractor,
                    abstained=abstained,
                    correct=correct,
                ))
        
        return results
    
    # TEST 3: Focused vs Full Context
    def test_focused_vs_full(
        self,
        num_samples: int = 6,
        full_context_tokens: int = 512,
    ) -> List[FocusedVsFullResult]:
        """
        Compare performance with focused context vs full context.
        
        Focused: Only relevant information (~100 tokens)
        Full: Relevant + lots of irrelevant (~500+ tokens)
        """
        results = []
        
        all_needles = LEXICAL_NEEDLES + SEMANTIC_NEEDLES
        
        for needle_data in all_needles[:num_samples]:
            # Focused prompt (just the needle)
            focused_prompt = f"""Context:
{needle_data["needle"]}

Question: {needle_data["question"]}

Answer:"""
            
            focused_answer = self._generate(focused_prompt)
            focused_correct, _ = self._check_answer(
                focused_answer, needle_data["answer"], needle_data["question"]
            )
            
            # Full prompt (needle buried in haystack)
            haystack = self._build_haystack(
                needle=needle_data["needle"],
                position=0.5,
                num_tokens=full_context_tokens,
            )
            
            full_prompt = f"""Context:
{haystack}

Question: {needle_data["question"]}

Answer:"""
            
            full_answer = self._generate(full_prompt)
            full_correct, _ = self._check_answer(
                full_answer, needle_data["answer"], needle_data["question"]
            )
            
            results.append(FocusedVsFullResult(
                focused_correct=focused_correct,
                focused_answer=focused_answer,
                full_correct=full_correct,
                full_answer=full_answer,
                performance_delta=float(focused_correct) - float(full_correct),
            ))
        
        return results
    
    # TEST 4: Repeated Words (Chroma context-rot benchmark)
    def test_repeated_words(
        self,
        common_word: str = "apple",
        modified_word: str = "apples",
        word_counts: List[int] = [50, 100, 200, 400, 800],
        positions_per_count: int = 3,  # Test 3 positions per word count
    ) -> Tuple[List[RepeatedWordsResult], RepeatedWordsSummary]:
        """
        Test model's ability to exactly replicate repeated text.
        
        From Chroma's context-rot: Model must reproduce a sequence of repeated
        words with one word modified. Uses Levenshtein score to measure fidelity.
        
        Args:
            common_word: The word to repeat (e.g., "apple")
            modified_word: The word to insert once (e.g., "apples")
            word_counts: Different sequence lengths to test
            positions_per_count: How many positions to test per word count
            
        Returns:
            List of individual results and a summary with binned scores
        """
        results = []
        
        for num_words in word_counts:
            # Test at different positions (start, middle, end)
            test_positions = [
                0,  # Start
                num_words // 2,  # Middle
                num_words - 1,  # End
            ][:positions_per_count]
            
            for modified_pos in test_positions:
                # Build the gold sequence
                words = [common_word] * num_words
                words[modified_pos] = modified_word
                gold_text = " ".join(words)
                
                # Build prompt asking model to reproduce exactly
                prompt = f"""Please reproduce the following text EXACTLY as written, character for character:

{gold_text}

Reproduced text:"""
                
                # Count input tokens
                input_tokens = len(self.tokenizer.encode(prompt))
                
                # Generate - need enough tokens to reproduce the full text
                max_output = min(input_tokens + 100, 4096)
                model_output = self._generate(prompt, max_tokens=max_output)
                
                # Compute Levenshtein score
                lev_score = normalized_levenshtein_score(gold_text, model_output)
                
                # Check if modified word is present
                if modified_pos == num_words - 1:
                    # End position: look for " apples" (space before)
                    unique_pattern = f" {modified_word}"
                else:
                    # Other positions: look for "apples " (space after)
                    unique_pattern = f"{modified_word} "
                
                modified_present = unique_pattern in model_output
                
                # Check if at correct position
                correct_pos = False
                if modified_present:
                    try:
                        gold_idx = gold_text.index(unique_pattern)
                        output_idx = model_output.index(unique_pattern)
                        correct_pos = abs(gold_idx - output_idx) < 10  # Allow small offset
                    except ValueError:
                        correct_pos = False
                
                results.append(RepeatedWordsResult(
                    num_words=num_words,
                    modified_word_position=modified_pos,
                    token_count=input_tokens,
                    gold_text=gold_text[:100] + "...",  # Truncate for storage
                    model_output=model_output[:100] + "...",
                    levenshtein_score=lev_score,
                    modified_word_present=modified_present,
                    correct_position=correct_pos,
                ))
        
        # Create summary with binned scores
        summary = self._summarize_repeated_words(results)
        
        return results, summary
    
    def _summarize_repeated_words(
        self,
        results: List[RepeatedWordsResult],
    ) -> RepeatedWordsSummary:
        """Summarize repeated words results with binned scores by token count."""
        if not results:
            return RepeatedWordsSummary(
                score_by_token_bin={},
                overall_levenshtein=0.0,
                modified_word_accuracy=0.0,
                position_accuracy=0.0,
                token_bins=[],
                avg_scores=[],
            )
        
        # Define token bins (log scale like Chroma)
        min_tokens = max(min(r.token_count for r in results), 1)
        max_tokens = max(r.token_count for r in results)
        num_bins = min(8, len(set(r.token_count for r in results)))
        
        if min_tokens >= max_tokens:
            bins = [min_tokens, max_tokens + 1]
        else:
            bins = list(np.logspace(np.log10(min_tokens), np.log10(max_tokens), num_bins + 1))
        
        # Bin the results
        score_by_bin = {}
        bin_centers = []
        avg_scores = []
        
        for i in range(len(bins) - 1):
            left, right = bins[i], bins[i + 1]
            bin_results = [r for r in results if left <= r.token_count < right]
            
            if bin_results:
                avg_score = sum(r.levenshtein_score for r in bin_results) / len(bin_results)
                bin_label = f"{int(left)}-{int(right)}"
                score_by_bin[bin_label] = avg_score
                bin_centers.append(int(np.sqrt(left * right)))  # Geometric mean
                avg_scores.append(avg_score)
        
        # Overall metrics
        overall_lev = sum(r.levenshtein_score for r in results) / len(results)
        modified_acc = sum(r.modified_word_present for r in results) / len(results)
        position_acc = sum(r.correct_position for r in results) / len(results)
        
        return RepeatedWordsSummary(
            score_by_token_bin=score_by_bin,
            overall_levenshtein=overall_lev,
            modified_word_accuracy=modified_acc,
            position_accuracy=position_acc,
            token_bins=bin_centers,
            avg_scores=avg_scores,
        )
    
    # FULL SUITE
    def run_full_suite(
        self,
        needle_samples: int = 6,
        distractor_samples: int = 6,
        focused_full_samples: int = 6,
        run_repeated_words: bool = True,
        repeated_words_counts: List[int] = [50, 100, 200, 400],
    ) -> ContextRotSuiteResult:
        """Run all context rot tests and return comprehensive results."""
        
        # Test 1: Needle-in-haystack
        needle_results = self.test_needle_in_haystack(
            needle_type="both",
            positions=[0.1, 0.5, 0.9],
            num_samples=needle_samples,
        )
        
        lexical_results = [r for r in needle_results if r.needle_type == "lexical"]
        semantic_results = [r for r in needle_results if r.needle_type == "semantic"]
        
        lexical_acc = sum(r.answer_found for r in lexical_results) / max(len(lexical_results), 1)
        semantic_acc = sum(r.answer_found for r in semantic_results) / max(len(semantic_results), 1)
        
        # Test 2: Distractor confusion
        distractor_results = self.test_distractor_confusion(
            distractor_counts=[0, 1, 4],
            num_samples=distractor_samples,
        )
        
        distractor_acc_by_count = {}
        for count in [0, 1, 4]:
            count_results = [r for r in distractor_results if r.num_distractors == count]
            if count_results:
                distractor_acc_by_count[count] = sum(r.correct for r in count_results) / len(count_results)
        
        # Test 3: Focused vs Full
        focused_full_results = self.test_focused_vs_full(
            num_samples=focused_full_samples,
        )
        
        focused_acc = sum(r.focused_correct for r in focused_full_results) / max(len(focused_full_results), 1)
        full_acc = sum(r.full_correct for r in focused_full_results) / max(len(focused_full_results), 1)
        length_degradation = focused_acc - full_acc
        
        # Test 4: Repeated Words (Chroma benchmark)
        repeated_words_results = None
        repeated_words_summary = None
        if run_repeated_words:
            try:
                repeated_words_results, repeated_words_summary = self.test_repeated_words(
                    word_counts=repeated_words_counts,
                    positions_per_count=3,
                )
            except Exception as e:
                print(f"  Repeated words test failed: {e}")
        
        # Compute composite score
        # Weights: semantic needle (0.25), distractor resistance (0.30), 
        #          length robustness (0.25), repeated words (0.20)
        distractor_resistance = 1.0 - (distractor_acc_by_count.get(0, 0) - distractor_acc_by_count.get(4, 0))
        length_robustness = 1.0 - max(0, length_degradation)
        
        # Repeated words score: penalize negative slope (degradation with length)
        repeated_words_score = 0.5  # Default if not run
        if repeated_words_summary:
            slope = repeated_words_summary.get_degradation_slope()
            # Slope is typically negative (degradation). Less negative = better.
            # Score: 1.0 if slope >= 0, scales down as slope gets more negative
            repeated_words_score = max(0, min(1, 1.0 + slope))  # slope in [-1, 0] range typically
            # Also factor in overall Levenshtein
            repeated_words_score = 0.5 * repeated_words_score + 0.5 * repeated_words_summary.overall_levenshtein
        
        composite = (
            0.25 * semantic_acc +
            0.30 * distractor_resistance +
            0.25 * length_robustness +
            0.20 * repeated_words_score
        )
        
        return ContextRotSuiteResult(
            lexical_needle_accuracy=lexical_acc,
            semantic_needle_accuracy=semantic_acc,
            needle_results=needle_results,
            distractor_accuracy_by_count=distractor_acc_by_count,
            distractor_results=distractor_results,
            focused_accuracy=focused_acc,
            full_accuracy=full_acc,
            length_degradation=length_degradation,
            focused_vs_full_results=focused_full_results,
            repeated_words_summary=repeated_words_summary,
            repeated_words_results=repeated_words_results,
            context_rot_resistance_score=composite,
        )
    
    def format_results(self, result: ContextRotSuiteResult) -> str:
        """Format results for console output."""
        lines = [
            "",
            "=" * 70,
            "CONTEXT ROT EVALUATION RESULTS",
            "=" * 70,
            "",
            "1. NEEDLE-IN-HAYSTACK",
            "-" * 40,
            f"   Lexical matching accuracy:  {result.lexical_needle_accuracy:.1%}",
            f"   Semantic matching accuracy: {result.semantic_needle_accuracy:.1%}",
            f"   Delta (lexical - semantic): {result.lexical_needle_accuracy - result.semantic_needle_accuracy:+.1%}",
            "",
            "2. DISTRACTOR CONFUSION",
            "-" * 40,
        ]
        
        for count, acc in sorted(result.distractor_accuracy_by_count.items()):
            lines.append(f"   {count} distractors: {acc:.1%}")
        
        degradation = result.distractor_accuracy_by_count.get(0, 0) - result.distractor_accuracy_by_count.get(4, 0)
        lines.extend([
            f"   Degradation (0 -> 4): {degradation:+.1%}",
            "",
            "3. FOCUSED vs FULL CONTEXT",
            "-" * 40,
            f"   Focused context accuracy: {result.focused_accuracy:.1%}",
            f"   Full context accuracy:    {result.full_accuracy:.1%}",
            f"   Length degradation:       {result.length_degradation:+.1%}",
        ])
        
        # Add Repeated Words results if available
        if result.repeated_words_summary:
            rw = result.repeated_words_summary
            lines.extend([
                "",
                "4. REPEATED WORDS (Chroma Benchmark)",
                "-" * 40,
                f"   Overall Levenshtein:      {rw.overall_levenshtein:.3f}",
                f"   Modified word accuracy:   {rw.modified_word_accuracy:.1%}",
                f"   Position accuracy:        {rw.position_accuracy:.1%}",
                f"   Degradation slope:        {rw.get_degradation_slope():+.4f}",
                "",
                "   Score by input length:",
            ])
            for bin_label, score in rw.score_by_token_bin.items():
                lines.append(f"     {bin_label} tokens: {score:.3f}")
        
        lines.extend([
            "",
            "=" * 70,
            f"CONTEXT ROT RESISTANCE SCORE: {result.context_rot_resistance_score:.3f}",
            "=" * 70,
            "",
        ])
        
        return "\n".join(lines)

# PLOTTING
def plot_context_rot_results(result: ContextRotSuiteResult, output_path: str):
    """Create visualization of context rot results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    # Determine layout based on whether we have repeated words data
    has_rw = result.repeated_words_summary is not None
    
    if has_rw:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Lexical vs Semantic
    ax1 = axes[0]
    x = ["Lexical", "Semantic"]
    y = [result.lexical_needle_accuracy, result.semantic_needle_accuracy]
    colors = ["#4CAF50", "#FF9800"]
    bars = ax1.bar(x, y, color=colors, edgecolor="black", linewidth=1.2)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Needle-in-Haystack\n(Lexical vs Semantic)")
    for bar, val in zip(bars, y):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.1%}", 
                ha="center", fontweight="bold")
    
    # Plot 2: Distractor degradation
    ax2 = axes[1]
    counts = sorted(result.distractor_accuracy_by_count.keys())
    accs = [result.distractor_accuracy_by_count[c] for c in counts]
    ax2.plot(counts, accs, marker="o", markersize=10, linewidth=2, color="#2196F3")
    ax2.fill_between(counts, accs, alpha=0.3, color="#2196F3")
    ax2.set_xlabel("Number of Distractors")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Distractor Resistance")
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(counts)
    
    # Plot 3: Focused vs Full
    ax3 = axes[2]
    x = ["Focused\n(~100 tokens)", "Full\n(~500 tokens)"]
    y = [result.focused_accuracy, result.full_accuracy]
    colors = ["#4CAF50", "#f44336"]
    bars = ax3.bar(x, y, color=colors, edgecolor="black", linewidth=1.2)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Context Length Impact")
    for bar, val in zip(bars, y):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.1%}", 
                ha="center", fontweight="bold")
    
    # Plot 4: Repeated Words - Levenshtein by Input Length (THE KEY PLOT!)
    if has_rw:
        ax4 = axes[3]
        rw = result.repeated_words_summary
        
        if rw.token_bins and rw.avg_scores:
            # Main line plot
            ax4.plot(rw.token_bins, rw.avg_scores, 
                    marker='o', markersize=8, linewidth=2.5, 
                    color="#90B8B6", markeredgecolor="white", markeredgewidth=1.5)
            
            # Log scale for x-axis (like Chroma)
            ax4.set_xscale('log')
            ax4.set_xlabel('Input Length (Tokens)', fontsize=11)
            ax4.set_ylabel('Avg Normalized Levenshtein Score', fontsize=11)
            ax4.set_title('Repeated Words: Performance by Length\n(Higher = Better)', fontsize=12)
            
            # Y-axis limits
            ax4.set_ylim(0, 1.05)
            ax4.grid(True, alpha=0.3)
            
            # Add degradation slope annotation
            slope = rw.get_degradation_slope()
            ax4.text(0.95, 0.05, f"Slope: {slope:+.3f}", 
                    transform=ax4.transAxes, fontsize=10,
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax4.text(0.5, 0.5, "No repeated words data", 
                    ha="center", va="center", transform=ax4.transAxes)
    
    plt.suptitle(f"Context Rot Resistance Score: {result.context_rot_resistance_score:.3f}", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved context rot plot to {output_path}")


def plot_repeated_words_curve(
    result: ContextRotSuiteResult, 
    output_path: str,
    model_name: str = "PALM",
    baseline_summary: Optional[RepeatedWordsSummary] = None,
):
    """
    Create the signature Chroma-style plot: Levenshtein score vs input length.
    
    Optionally includes a baseline for comparison.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    if not result.repeated_words_summary:
        print("No repeated words data to plot")
        return
    
    rw = result.repeated_words_summary
    
    plt.figure(figsize=(10, 6))
    
    # Plot PALM results
    if rw.token_bins and rw.avg_scores:
        plt.plot(rw.token_bins, rw.avg_scores, 
                marker='o', markersize=8, linewidth=2.5, 
                color="#2FB874", label=f"{model_name} (slope: {rw.get_degradation_slope():+.3f})")
    
    # Plot baseline if provided
    if baseline_summary and baseline_summary.token_bins and baseline_summary.avg_scores:
        plt.plot(baseline_summary.token_bins, baseline_summary.avg_scores,
                marker='s', markersize=8, linewidth=2.5, linestyle='--',
                color="#EA5412", label=f"Baseline (slope: {baseline_summary.get_degradation_slope():+.3f})")
    
    plt.xscale('log')
    plt.xlabel('Input Length (Tokens)', fontsize=12)
    plt.ylabel('Average Normalized Levenshtein Score', fontsize=12)
    plt.title('Repeated Words: Performance Degradation by Input Length', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved repeated words curve to {output_path}")


def save_context_rot_results(result: ContextRotSuiteResult, output_path: str):
    """Save results to JSON."""
    data = result.to_dict()
    
    # Add detailed results
    data["needle_results_detail"] = [
        {
            "type": r.needle_type,
            "position": r.needle_position,
            "found": r.answer_found,
            "confidence": r.confidence,
        }
        for r in result.needle_results
    ]
    
    data["distractor_results_detail"] = [
        {
            "num_distractors": r.num_distractors,
            "correct": r.correct,
            "abstained": r.abstained,
            "picked_distractor": r.picked_distractor,
        }
        for r in result.distractor_results
    ]
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved context rot results to {output_path}")


def load_context_rot_results(input_path: str) -> Dict:
    """Load results from JSON."""
    with open(input_path, "r") as f:
        return json.load(f)
