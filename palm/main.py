import torch

from transformers import AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Create config with training hyperparameters from constants
    config = PALMConfig(
        base_model_name=MODEL_NAME, 
        hidden_dropout_prob=0.3, 
        attention_probs_dropout_prob=0.3,
        num_hidden_layers=10, 
        num_attention_heads=10, 
        hidden_size=750, 
        layer_norm_eps=1e-5,
        # Training hyperparameters from constants
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        train_batch_size=TRAIN_BATCH_SIZE,
        max_length=MAX_SEQ_LENGTH,
        # Sync pad_token_id with tokenizer
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
    )

    # Create model
    model = PALMModel(config)

    # Preprocess and split dataset (pass dataset name, not dataset object)
    train_dataset, eval_dataset = load_and_split_dataset(DATASET_NAME, TRAIN_RATIO)
    
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
        collate_fn
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
    model.save_pretrained("PALM_model")
    tokenizer.save_pretrained("PALM_model")


if __name__ == "__main__":
    main()
