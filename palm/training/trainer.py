import math
import time
import logging
import gc

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
# from torch.optim.lr_scheduler import CosineAnnealingLR

from palm.constants import *
from palm.training.utils import freeze_selected_layers, is_custom_param

# Transformers Schedulers
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


class PALMTrainer:
    '''
    Encapsulates the full training loop (forward, backward, step) + evaluation for
    the PALLM model. Includes selective freezing, dynamic SAE weighting, logging metrics 
    and gradient checkpointing features.
    '''
    def __init__(self, model, train_dataloader, eval_dataloader, tokenizer, config, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine whether to use GPU or CPU
        self.model.to(self.device) # Move model to the selected device

        # Prepare param groups for different LR
        self.pretrained_params = []
        self.custom_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if is_custom_param(name):
                    self.custom_params.append(param)
                else:
                    self.pretrained_params.append(param)
        
        # Initialize optimizer with AdamW, including differential learning rates and weight decay
        optimizer = torch.optim.AdamW([
            {"params": pretrained_params, "lr": PRETRAINED_LR, "weight_decay": 0.01},
            {"params": custom_params, "lr": CUSTOM_LR, "weight_decay": 0.09},
        ])
        # Build LR scheduler
        total_steps = len(self.train_dataloader) * NUM_TRAIN_EPOCHS // GRADIENT_ACCUMULATION_STEPS

        if USE_COSINE_DECAY:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=WARMUP_STEPS,
                num_training_steps=total_steps
            )
        elif USE_POLYNOMIAL_DECAY:
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=WARMUP_STEPS,
                num_training_steps=total_steps,
                lr_end=POLY_LR_END,
                power=1.0
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=WARMUP_STEPS,
                num_training_steps=total_steps
            )
        # AMP GradScaler (only needed if any parameters are fp32)
        has_fp32_params = any(p.dtype == torch.float32 for p in self.model.parameters())
        self.scaler = GradScaler(enabled=(device.type == "cuda" and has_fp32_params))

        # Additional state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.no_improvement_steps = 0
        self.patience = EARLY_STOP_PATIENCE
        self.start_time = time.time()

        # Possibly freeze layers at start
        # If using a chunk-based schedule, that will happen inside train() at the start of each epoch
        freeze_selected_layers(self.model, freeze_embeddings=FREEZE_SCHEDULE_INIT["freeze_embeddings"],
                               freeze_up_to_layer_idx=FREEZE_SCHEDULE_INIT["freeze_up_to_layer_idx"])
        logger.info("Trainer initialized.")
        
    # Define training process
    def train(self):
        """
        Full training loop. Contains selective freezing schedule,
        dynamic SAE weighting (optional), and evaluation + early stopping.
        """
        NUM_LAYERS = real_model.config.num_hidden_layers
        FREEZE_SCHEDULE = [
            # Epoch 0: Only custom components and topmost layers are trainable.
            {"epoch": 0, "freeze_embeddings": True, "freeze_up_to_layer_idx": NUM_LAYERS - 3}, 
            {"epoch": 1, "freeze_embeddings": True, "freeze_up_to_layer_idx": NUM_LAYERS - 6},
            # Epoch 2: Unfreeze the top 9 layers
            {"epoch": 2, "freeze_embeddings": True, "freeze_up_to_layer_idx": NUM_LAYERS - 9},
            {"epoch": 3, "freeze_embeddings": True, "freeze_up_to_layer_idx": NUM_LAYERS - 12},
            {"epoch": 4, "freeze_embeddings": True, "freeze_up_to_layer_idx": NUM_LAYERS - 15},
            # Epoch 5: Unfreeze top 18 layers
            {"epoch": 5, "freeze_embeddings": True, "freeze_up_to_layer_idx": NUM_LAYERS - 18},
            {"epoch": 6, "freeze_embeddings": True, "freeze_up_to_layer_idx": NUM_LAYERS - 21},
            # Epoch 7: Fully unfreeze all layers (including embeddings).
            {"epoch": 7, "freeze_embeddings": False, "freeze_up_to_layer_idx": 0},
        ]
        
        # Initialize with the first schedule stage
        initial_schedule = FREEZE_SCHEDULE[0]
        freeze_selected_layers(model, freeze_embeddings=initial_schedule["freeze_embeddings"],
                               freeze_up_to_layer_idx=initial_schedule["freeze_up_to_layer_idx"])
        # Early stopping variables
        best_eval_loss = float("inf")
        no_improvement_steps = 0
        patience = 3
        
        for epoch in range(NUM_TRAIN_EPOCHS):
            # Check if we should switch freeze settings at this epoch
            for schedule_stage in FREEZE_SCHEDULE:
                if epoch == schedule_stage["epoch"]:
                    freeze_selected_layers(
                        model,
                        freeze_embeddings=schedule_stage["freeze_embeddings"],
                        freeze_up_to_layer_idx=schedule_stage["freeze_up_to_layer_idx"]
                    )
                    logger.info(f"Selective freezing activated at epoch {epoch}. "
                                f"Embeddings frozen: {schedule_stage['freeze_embeddings']}, "
                                f"Layers up to idx {schedule_stage['freeze_up_to_layer_idx']} are frozen.")

            # Possibly adjust dynamic SAE weight
            if USE_DYNAMIC_SAE_WEIGHT:
                fraction_done = epoch / float(NUM_TRAIN_EPOCHS - 1 if NUM_TRAIN_EPOCHS > 1 else 1)
                new_weight = SAE_START_WEIGHT + (SAE_END_WEIGHT - SAE_START_WEIGHT) * fraction_done
                # Update inside model config
                real_model = self.model.module if hasattr(self.model, 'module') else self.model
                real_model.config.sae_weight = new_weight
                logger.info(f"Dynamic SAE weight updated to {new_weight:.4f} at epoch {epoch}")

            # One epoch of training
            self.train_epochs(epoch)

            # Evaluate
            eval_loss = self.evaluate()
            avg_eval_loss = eval_loss / len(self.eval_dataloader)
            perplexity = torch.exp(torch.tensor(avg_eval_loss, device=self.device))

            # Check improvements
            if avg_eval_loss < self.best_eval_loss:
                self.best_eval_loss = avg_eval_loss
                self.no_improvement_steps = 0
            else:
                self.no_improvement_steps += 1
                if self.no_improvement_steps >= self.patience:
                    logger.info("Early stopping triggered.")
                    break

            # Log evaluation results
            wandb.log({
                "eval_loss": avg_eval_loss,
                "perplexity": perplexity.item(),
                "global_step": self.global_step
            })
            logger.info(f"[Epoch {epoch}/{NUM_TRAIN_EPOCHS}] eval_loss={avg_eval_loss:.4f}, ppl={perplexity.item():.4f}")

            # Early Stopping check
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
                if no_improvement_steps >= patience:
                    logger.info("Early stopping triggered.")
                    break

            # Optionally save checkpoint
            if (epoch + 1) % CHECKPOINT_EVERY_EPOCH == 0:
                ckpt_dir = f"{CHECKPOINT_DIR}/epoch_{epoch+1}"
                self.save_checkpoint(ckpt_dir)

        logger.info("Training completed.")

    def train_epoch(self, epoch):
        """
        Trains model across epochs, with gradient accumulation
        and logging. Returns the average train loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_sae_loss = 0.0
        total_steps_in_epoch = len(self.train_dataloader)

        for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")):
            # Move to device
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            source_len = batch["source_len"].to(self.device, non_blocking=True)

            # Forward pass
            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                lm_logits, combined_loss, loss, sae_loss = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels, 
                    source_len=source_len
                )
            # Accumulate loss
            if combined_loss is not None:
                combined_loss = combined_loss / GRADIENT_ACCUMULATION_STEPS
                self.scaler.scale(combined_loss).backward()
                total_loss += (loss.item() if loss is not None else 0.0)
                total_sae_loss += (sae_loss.item() if sae_loss is not None else 0.0)

            # Update
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or step == total_steps_in_epoch - 1:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                # Logging
                if self.global_step % LOG_EVERY_STEPS == 0:
                    samples_per_second = TRAIN_BATCH_SIZE / (time.time() - self.start_time)
                    self.start_time = time.time()

                    wandb.log({
                        "train_loss": loss.item() if loss else 0.0,
                        "sae_loss": sae_loss.item() if sae_loss else 0.0,
                        "combined_loss": combined_loss.item() if combined_loss else 0.0,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "global_step": self.global_step,
                        "epoch": epoch,
                        "samples_per_second": samples_per_second
                    })
                    if torch.cuda.is_available():
                        wandb.log({"gpu_memory": torch.cuda.max_memory_allocated() / 1e9})
                    logger.info(f"Step {self.global_step}, Loss={loss.item() if loss else None:.4f}, "
                                f"SAE={sae_loss.item() if sae_loss else None:.4f}, "
                                f"Comb={combined_loss.item() if combined_loss else None:.4f}")

        avg_loss = total_loss / (total_steps_in_epoch or 1)
        logger.info(f"Epoch {epoch+1} finished. Average training loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self):
        """
        Evaluates the model on the eval_dataloader. Returns total (summed) eval loss.
        """
        self.model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                source_len = batch["source_len"].to(self.device, non_blocking=True)

                outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels, 
                    source_len=source_len
                )
                # outputs = (lm_logits, combined_loss, loss, sae_loss)
                combined_loss = outputs[1]
                total_eval_loss += combined_loss.item() if combined_loss is not None else 0.0

        return total_eval_loss

    def save_checkpoint(self, checkpoint_dir):
        """
        Saves model + tokenizer state to the checkpoint_dir.
        """
        logger.info(f"Saving checkpoint to {checkpoint_dir} ...")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        logger.info("Checkpoint saved.")
