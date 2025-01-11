import torch

from transformers import AutoTokenizer
from datasets import load_dataset

from palm.constants import *
from palm.config import PALMConfig
from palm.model.palm import PALMModel
from palm.data.dataset import load_and_split_dataset
from palm.data.preprocessing import preprocess_function, create_data_loaders
from palm.training.trainer import PALMTrainer
from palm.training.utils import collate_fn

import wandb


def main():
    # Initialize wandb
    wandb.init(project="palm-instruction-tuning", name="palm-llama-3-8b-instruction")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map='auto')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load dataset
    dataset = load_dataset(DATASET_NAME)

    # Create config
    config = PALMConfig(
        base_model_name=MODEL_NAME, 
        hidden_dropout_prob=0.5, 
        attention_probs_dropout_prob=0.5,
        num_hidden_layers=28, 
        num_attention_heads=24, 
        hidden_size=3072, 
        layer_norm_eps=1e-5,
        sae_weight=0.5
    )

    # Create model
    model = PALMModel(config)

    # Preprocess and split dataset
    train_dataset, eval_dataset = load_and_split_dataset(dataset, TRAIN_RATIO)
    
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Create data loaders
    train_dataloader, eval_dataloader = create_data_loaders(
        train_dataset, 
        eval_dataset, 
        TRAIN_BATCH_SIZE,
        EVAL_BATCH_SIZE,
        shuffle=True,
        collate_fn,
        num_workers=12,
        pin_memory=(device.type == "cuda"),
        persistant_workers=True
    )

    # Initialize trainer
    trainer = PALMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config
    )

    # Train the model
    trainer.train()

    # Save the final model
    model.save_pretrained("pallm_llama_3.2-3B_")
    tokenizer.save_pretrained("pallm_llama_3.2-3B_")


if __name__ == "__main__":
    main()
