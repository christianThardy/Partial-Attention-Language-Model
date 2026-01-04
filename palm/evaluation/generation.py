import torch


def generate_text(model, tokenizer, prompt, max_length=1024, temperature=0.7, top_p=0.9, model_type=None):
    try:
        # Determine device: if DataParallel, fetch device from the first replica
        if isinstance(model, torch.nn.DataParallel):
            dev = next(model.module.parameters()).device
        else:
            dev = next(model.parameters()).device

        # Encode the input prompt into token IDs using the tokenizer, converting it to a tensor
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
        print(f"input_ids shape: {input_ids.shape}")  # Debug print

        # If model_type is "PALM" and the model has a bidirectional attention mask method
        attention_mask = None
        if model_type == "PALM" and hasattr(model, 'create_bidirectional_attention_mask'):
            attention_mask = model.create_bidirectional_attention_mask(input_ids)
            print(f"attention_mask shape: {attention_mask.shape}")  # Debug print

        # Generation with no_grad + optional autocast for GPU efficiency
        with torch.no_grad():  # Disable gradient computation for inference
            with torch.amp.autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
                if model_type == "PALM":
                    # PALM-specific generate (requires custom attention_mask usage)
                    output = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True, # Enable sampling for more diverse text generation
                        pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                else:
                    # Fallback for GPT-2 style models or others
                    # Use eos_token_id as pad if no separate pad_token_id
                    fallback_pad_id = (
                        tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    )
                    output = model.generate(
                        input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=fallback_pad_id
                    )
                    
        # Decode generated tokens back into text, skipping special tokens like <PAD> or <EOS>
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        print(f"Error in generate_text: {str(e)}")
        # Try printing relevant device info
        try:
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Input device: {input_ids.device}")  # might fail if input_ids not defined
        except:
            pass
        return None
