import os
import math
import random
import time
import logging
import gc

import torch
from torch.utils.data import random_split

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login

from palm.constants import *
from palm.config import PALMConfig
from palm.model.palm import PALMModel
from palm.data.dataset import load_and_split_dataset
from palm.data.preprocessing import preprocess_function, create_data_loaders
from palm.training.trainer import PALMTrainer
from palm.training.utils import collate_fn_base

from peft import LoraConfig, AdaLoraConfig, get_peft_model

import wandb
from tqdm import tqdm

# Logging configuration
logger = logging.getLogger()
# Level of the logger. Values: DEBUG, INFO, WARNING, ERROR, CRITICAL
logger.setLevel(logging.DEBUG)
# Handler writes log messages to the notebook's output
handler = logging.StreamHandler()
# Set the format for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def maybe_apply_lora(model, use_lora=False, use_qlora=False):
    """
    If use_lora or use_qlora is True, apply the relevant adapter configuration.
    Minimal placeholder, adjust config as needed.
    """
    if use_lora:
        from peft import LoraConfig, get_peft_model
        # Example LoRA config (adjust as needed):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],  # or whichever
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA adapters applied.")

    elif use_qlora:
        from peft import AdaLoraConfig, get_peft_model
        # Example QLoRA/AdaLora config:
        qlora_config = AdaLoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            init_r=8,
            Tinit=200,
            Tfinal=1000,
            deltaT=100
        )
        model = get_peft_model(model, qlora_config)
        logger.info("QLoRA adapters applied.")

    return model

def main():
    # ENVIRONMENT SETUP + CONFIG
    login(token=HF_TOKEN)

    # Enable TF32 for faster matrix multiplies
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"

    wandb.init(project="palm-fine-tuning", name="palm-llama-3.2-3B-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map='auto')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")

    # LOAD AND COMBINE DATASETS
    dataset_1 = load_dataset(DATASET_1_NAME, split="all").remove_columns(["meta"])
    dataset_1 = dataset_1.shuffle(seed=42).select(range(DS1_MAX_SAMPLES))

    dataset_2 = load_dataset(DATASET_2_NAME, split="all").remove_columns(["source"])
    dataset_2 = dataset_2.shuffle(seed=42)#.select(range(DS2_MAX_SAMPLES))

    dataset = concatenate_datasets([dataset_1, dataset_2])
    logger.info(f"Combined dataset size: {len(dataset)}")

    # Shuffle + split
    shuffled_dataset = dataset.shuffle(seed=42)
    dataset_size = len(shuffled_dataset)
    train_size = int(TRAIN_RATIO * dataset_size)
    eval_size = dataset_size - train_size

    # For random splitting
    train_dataset, eval_dataset = random_split(shuffled_dataset, [train_size, eval_size])

    # PREPROCESS
    def _preproc_fn(examples):
        return preprocess_function(examples, tokenizer, MAX_SEQ_LENGTH)

    train_dataset = train_dataset.map(
        _preproc_fn,
        batched=True,
        remove_columns=train_dataset.dataset.column_names
        if hasattr(train_dataset, 'dataset') else train_dataset[0].keys()
    )
    eval_dataset = eval_dataset.map(
        _preproc_fn,
        batched=True,
        remove_columns=eval_dataset.dataset.column_names
        if hasattr(eval_dataset, 'dataset') else eval_dataset[0].keys()
    )
    # Create DataLoaders
    print("Starting train-eval data loading")
    train_dataloader, eval_dataloader = create_data_loaders(
        train_dataset,
        eval_dataset,
        TRAIN_BATCH_SIZE,
        EVAL_BATCH_SIZE,
        collate_fn=collate_fn_base,
        num_workers=12,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )
    print("Train-eval data loading complete")

    # CREATE MODEL + CONFIG
    config = PALMConfig(
        base_model_name=MODEL_NAME,
        hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=ATTN_DROPOUT_PROB,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        hidden_size=HIDDEN_SIZE,
        layer_norm_eps=LAYER_NORM_EPSILON,
        sae_weight=SAE_WEIGHT,
        gradient_checkpointing=GRADIENT_CHECKPOINTING
    )

    model = PALMModel(config)
    logger.info(f"Model pad_token_id: {model.config.pad_token_id}")

    # Optionally apply LoRA or QLoRA
    model = maybe_apply_lora(model, USE_LORA, USE_QLORA)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        torch.cuda.empty_cache()
        gc.collect()
        model = model.to(device, dtype=torch.bfloat16)
        model = torch.nn.DataParallel(model)
    else:
        logger.info("Using single GPU or CPU")
        torch.cuda.empty_cache()
        gc.collect()
        model = model.to(device, dtype=torch.bfloat16)

    # Optional compile on PyTorch 2.0+
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        torch.set_float32_matmul_precision('high')  # optional
        model = torch.compile(model)
        logger.info("Compiled model with torch.compile for speed.")

    # TRAINER
    trainer = PALMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
    # Train the model
    trainer.train()

    # SAVE MODEL + TOKENIZER
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(
        OUTPUT_DIR,
        push_to_hub=PUSH_TO_HUB,
        repo_id=HF_REPO_ID,
        use_auth_token=True
    )
    tokenizer.save_pretrained(
        OUTPUT_DIR,
        push_to_hub=PUSH_TO_HUB,
        repo_id=HF_REPO_ID,
        use_auth_token=True
    )
    logger.info(f"Model and tokenizer saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
