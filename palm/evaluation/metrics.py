import torch
import math
import logging

logger = logging.getLogger(__name__)

# Optional dependencies for evaluation metrics
# These are only needed for specific evaluation functions, not core functionality
_ROUGE_AVAILABLE = False
_BERT_SCORE_AVAILABLE = False
_EVALUATE_AVAILABLE = False
_TQDM_AVAILABLE = False

try:
    from rouge import Rouge
    _ROUGE_AVAILABLE = True
except ImportError:
    Rouge = None

try:
    from bert_score import score as bert_score_fn
    _BERT_SCORE_AVAILABLE = True
except ImportError:
    bert_score_fn = None

try:
    import evaluate
    _EVALUATE_AVAILABLE = True
except ImportError:
    evaluate = None

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    # Fallback tqdm that just returns the iterable
    def tqdm(iterable, **kwargs):
        return iterable


def calculate_rouge(reference, hypothesis):
    """
    Calculates ROUGE-L F1 score between single 'reference' and 'hypothesis'.
    Uses python-rouge's get_scores method.
    
    Requires: pip install rouge
    """
    if not _ROUGE_AVAILABLE:
        raise ImportError(
            "Rouge is required for calculate_rouge. Install with: pip install rouge"
        )
    
    # Initialize ROUGE metric object
    rouge = Rouge()

    # Calculate ROUGE scores between the hypothesis and reference texts
    scores = rouge.get_scores(hypothesis, reference)[0]

    # Return F1 score of the ROUGE-L metric (longest common subsequence)
    return scores['rouge-l']['f']

def calculate_bert_score(reference, hypothesis):
    """
    Calculates BERTScore F1 between single 'reference' and 'hypothesis'.
    Uses bert_score library's score() method. Returns float F1.
    
    Requires: pip install bert-score
    """
    if not _BERT_SCORE_AVAILABLE:
        raise ImportError(
            "bert_score is required for calculate_bert_score. Install with: pip install bert-score"
        )
    
    # Calculate BERTScore F1 score between the hypothesis and reference texts
    _, _, f1 = bert_score_fn([hypothesis], [reference], lang="en")
    # Return F1 score as a scalar value
    return f1.item()

def evaluate_generations(reference, palm_output, baseline_output):
    """
    Evaluates two model outputs (palm_output, baseline_output) against a reference,
    returning their ROUGE-L and BERTScore metrics.
    """
    # Calculate ROUGE-L and BERT scores for the PALM model output
    palm_rouge = calculate_rouge(reference, palm_output)
    baseline_rouge = calculate_rouge(reference, baseline_output)
    
    palm_bert_score = calculate_bert_score(reference, palm_output)
    baseline_bert_score = calculate_bert_score(reference, baseline_output)
    
    # Return a dictionary containing evaluation scores for both models
    return {
        "palm_rouge": palm_rouge,
        "baseline_rouge": baseline_rouge,
        "palm_bert_score": palm_bert_score,
        "baseline_bert_score": baseline_bert_score
    }

def evaluate_information_extraction(true_info, palm_output, baseline_output):
    """
    Compares extracted answers from palm_output / baseline_output vs. true_info.
    Returns a simple accuracy measure for each model.
    """
    # Helper function to extract answers from the model output text
    def extract_answers(output):
        lines = output.split('\n')
        answers = [line.split(': ', 1)[1] if ': ' in line else '' for line in lines if line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]
        return answers

    # Extract answers from PALM and baseline outputs
    palm_answers = extract_answers(palm_output)
    baseline_answers = extract_answers(baseline_output)

    # Calculate the number of correct answers for PALM and baseline outputs
    palm_correct = sum(1 for true, pred in zip(true_info, palm_answers) if true.lower() in pred.lower())
    baseline_correct = sum(1 for true, pred in zip(true_info, baseline_answers) if true.lower() in pred.lower())

    # Return a dictionary containing accuracy metrics for both models
    return {
        "palm_accuracy": palm_correct / len(true_info),
        "baseline_accuracy": baseline_correct / len(true_info)
    }

def evaluate_model_perplexity(model, dataloader, device):
    """
    Computes the average cross-entropy loss across the given dataloader,
    returning (avg_loss, perplexity). Minimizes overhead by disabling grad.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            source_len = batch["source_len"].to(device, non_blocking=True)

            # Forward pass (lm_logits, combined_loss, loss, sae_loss) = model(...)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                source_len=source_len
            )
            # outputs[1] is combined_loss
            total_loss += outputs[1].item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def compute_text_generation_metrics(predictions, references):
    """
    Computes multiple text generation metrics (e.g. BLEU, METEOR, ROUGE) in one pass
    using Hugging Face evaluate method. Returns dict of scores.
    
    Requires: pip install evaluate
    """
    metrics_dict = {}

    if not _EVALUATE_AVAILABLE:
        logger.warning("evaluate module not installed. Install with: pip install evaluate")
        return metrics_dict

    # Example: Using evaluate.load for BLEU
    try:
        bleu_metric = evaluate.load("bleu")
        preds_tok = [p.strip().split() for p in predictions]
        refs_tok = [[r.strip().split()] for r in references]  # HF BLEU expects list-of-lists
        bleu_score = bleu_metric.compute(predictions=preds_tok, references=refs_tok)["bleu"]
        metrics_dict["BLEU"] = bleu_score
    except Exception:
        pass  # skip if 'bleu' not installed or other error

    # METEOR
    try:
        meteor_metric = evaluate.load("meteor")
        meteor_score = meteor_metric.compute(predictions=predictions, references=references)["meteor"]
        metrics_dict["METEOR"] = meteor_score
    except Exception:
        pass

    # ROUGE
    try:
        rouge_metric = evaluate.load("rouge")
        rouge_score = rouge_metric.compute(predictions=predictions, references=references)
        # use rougeL F1 as an example
        metrics_dict["ROUGEL"] = rouge_score["rougeL"].mid.fmeasure
    except Exception:
        pass

    return metrics_dict

def evaluate_generation_on_dataset(
    model,
    tokenizer,
    dataset,
    device,
    max_gen_length=128,
    temperature=1.0,
    top_p=0.9
):
    """
    Generates model outputs for each sample's prompt (or source) in 'dataset'
    and compares them to references. Returns aggregated HF metrics (BLEU, METEOR, ROUGE).
    Adjust logic for how 'prompt'/'completion' columns are used as needed.
    """
    model.eval()
    predictions, references = [], []

    for item in tqdm(dataset, desc="Generating for evaluation"):
        prompt = item.get("prompt", "")
        reference = item.get("completion", "")
        references.append(reference)

        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # For advanced usage, pass in custom generation args
                output_ids = model.generate(
                    input_ids,
                    max_length=max_gen_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

        pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred_text)

    return compute_text_generation_metrics(predictions, references)

def compute_stepwise_hallucination_ratio(
    model,
    tokenizer,
    dataset,
    device,
    max_gen_length=64,
    alpha=1.0,
    beta=1.0
):
    """
    Example placeholder to measure how "aligned" tokens are with the source or 
    reference. Your snippet can remain minimal or be extended if you use it. We 
    preserve it here for completeness.
    """
    model.eval()
    # Just a stub from your advanced example:
    # Real implementation would generate tokens, check alignment, etc.
    # Return an empty list if not in usage
    return []

def numerical_sensitivity_analysis(
    model,
    tokenizer,
    source_text,
    device,
    perturb_scale=1e-3,
    max_tokens=50
):
    """
    Another placeholder for analyzing hidden-state sensitivity to small input 
    perturbations.
    """
    model.eval()
    # Stub from your advanced example
    return []

def evaluate_multiple_models(
    model_names,
    dataset,
    device,
    max_gen_length=128,
    temperature=1.0,
    top_p=0.9
):
    """
    Example function for comparing multiple checkpoint names or model paths on a single 
    dataset.
    """
    final_results = {}
    # Implementation would loop over model_names, load, generate, compute metrics, etc.
    # Omitted for brevity or left minimal:
    for mname in model_names:
        final_results[mname] = {}
    return final_results
