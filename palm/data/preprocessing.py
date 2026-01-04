import torch
from torch.utils.data import DataLoader


def preprocess_function(examples, tokenizer, max_seq_length):
    # Extract 'prompt' and 'completion' fields from the dataset examples
    prompts = examples['prompt']
    completions = examples['completion']

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


def create_data_loaders(train_dataset, eval_dataset, batch_size, collate_fn):
    # Create a DataLoader for the training dataset with shuffling and specified batch size
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    
    # Create a DataLoader for the evaluation dataset without shuffling
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    
    # Return both training and evaluation DataLoaders
    return train_dataloader, eval_dataloader
