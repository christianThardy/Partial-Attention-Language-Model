import torch
from torch.utils.data import DataLoader


def preprocess_function(examples, tokenizer, max_seq_length=1024):
    """
    Preprocess function that:
    
      - Combines prompt + completion (if data has them),
        or uses 'text' if it just has a single field
      - Tokenizes
      - Sets input_ids, attention_mask
      - Creates 'labels' identical to input_ids for causal LM
      - Creates 'source_len' to identify portion of the sequence considered 'source'
      
    Modify as needed if your dataset has different columns.
    """
    # If your data has "text" (pretrained dataset usually) only:
    if "text" in examples:
        texts = examples["text"]
        model_inputs = tokenizer(texts, max_length=max_seq_length, truncation=True, padding="max_length")
        # Create labels from input_ids
        model_inputs["labels"] = [ids[:] for ids in model_inputs["input_ids"]]
        # Construct source_len as half or however you define
        pad_id = tokenizer.pad_token_id
        source_lens = []
        for i_ids in model_inputs["input_ids"]:
            unpadded_len = sum(1 for t in i_ids if t != pad_id)
            # source_lens.append(unpadded_len // 2)  # Half of unpadded length
            source_lens.append(unpadded_len)  # Entire unpadded length
        model_inputs["source_len"] = source_lens
        return model_inputs

    # Otherwise, if you have "prompt" + "completion" style
    # Combine them:
    prompts = examples.get("prompt", [])
    completions = examples.get("completion", [])
    combined_texts = []
    for p, c in zip(prompts, completions):
        # You can tweak how you combine them
        combined_texts.append(f"{p}\n{c}")

    model_inputs = tokenizer(combined_texts, max_length=max_seq_length, truncation=True, padding="max_length")
    model_inputs["labels"] = [ids[:] for ids in model_inputs["input_ids"]]

    # Example source_len (just assume prompt is half):
    pad_id = tokenizer.pad_token_id
    source_lens = []
    for text in prompts:
        # naive approach: length of the prompt portion
        tokens_prompt = tokenizer.encode(text, add_special_tokens=False)
        source_lens.append(min(len(tokens_prompt), max_seq_length))
    model_inputs["source_len"] = source_lens

    # Combine prompts and completions into a single string for each example
    combined = [f"{prompt}\n\nAssistant: {completion}" for prompt, completion in zip(prompts, completions)]
    
    # Tokenize combined strings, truncating or padding them to the max_seq_length
    model_inputs = tokenizer(combined, max_length=max_seq_length, truncation=True, padding="max_length")
    
    # Calculate source lengths BEFORE converting to tensors (tokenize prompts separately)
    source_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    
    # Convert input_ids and attention_mask to PyTorch tensors
    input_ids = torch.tensor(model_inputs["input_ids"])
    attention_mask = torch.tensor(model_inputs["attention_mask"])
    
    # Create labels: mask prompt tokens with -100 so loss is only computed on completions
    # This is standard practice for instruction-tuning
    labels = input_ids.clone()
    for i, source_len in enumerate(source_lengths):
        # Mask all tokens in the prompt portion (set to -100, ignored by CrossEntropyLoss)
        labels[i, :source_len] = -100
    
    # Also mask padding tokens in labels
    labels[attention_mask == 0] = -100
    
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = attention_mask
    model_inputs["labels"] = labels
    model_inputs["source_len"] = torch.tensor(source_lengths)
    
    # Return processed inputs ready for model consumption
    return model_inputs

def create_data_loaders(
    train_dataset,
    eval_dataset,
    train_batch_size,
    eval_batch_size,
    collate_fn,
    num_workers=4,
    pin_memory=False,
    persistent_workers=False
):
    """
    Utility for creating train + eval dataloaders from processed datasets.
    """
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    return train_dataloader, eval_dataloader
