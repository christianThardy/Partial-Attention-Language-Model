import torch

def collate_fn_instrcut(batch):
    # Initialize dictionaries to store the batched data
    batched_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "source_len": []
    }
    # Iterate through the batch and append each item to the corresponding list
    for item in batch:
        for key in batched_data:
            # Ensure each item is a tensor and append it to the list
            batched_data[key].append(torch.tensor(item[key]))
    # Stack tensors, making sure all elements are tensors and have the same shape
    for key in batched_data:
        batched_data[key] = torch.stack(batched_data[key])

    return batched_data

def collate_fn_base(batch):
    # Identify all keys in the batch
    keys = batch[0].keys()  # e.g. ["input_ids","attention_mask","labels","source_len"]
    batched_data = {key: [item[key] for item in batch] for key in keys}
    
    # Convert lists to tensors
    for key in batched_data:
        # Force them to integer (long) Tensors
        batched_data[key] = torch.tensor(batched_data[key], dtype=torch.long)

    return batched_data

def init_custom_layer_weights(module):
    """
    Recursively initialize custom layers (partial attention, lm/sae heads) 
    to encourage faster learning from scratch.
    """
    for name, submodule in module.named_children():
        # Initialize only "custom" modules or submodules you consider new. 
        # This example checks partial attention & heads, skipping loaded pretrained weights.
        if isinstance(submodule, PALMPartialAttention) or name in ["lm_head", "sae_head"]:
            for param_name, param in submodule.named_parameters(recurse=False):
                if "weight" in param_name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in param_name:
                    nn.init.zeros_(param)
        else:
            init_custom_layer_weights(submodule)

def is_custom_param(param_name):
    # Here, treat partial_attention, lm_head, sae_head, or 'Fp' submodule as custom
    if any(x in param_name for x in ["partial_attention", "lm_head", "sae_head", "Fp"]):
        return True
    return False

def freeze_selected_layers(model_, freeze_embeddings=True, freeze_up_to_layer_idx=0):
    real_model = model_.module if hasattr(model_, 'module') else model_
    if freeze_embeddings:
        for param in real_model.embeddings.parameters():
            param.requires_grad = False
    for idx, layer in enumerate(real_model.layers):
        for param in layer.parameters():
            param.requires_grad = (idx >= freeze_up_to_layer_idx)
    for param in real_model.lm_head.parameters():
        param.requires_grad = True
    for param in real_model.sae_head.parameters():
        param.requires_grad = True
