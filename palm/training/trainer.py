import time
import logging

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


class PALMTrainer:
    '''Encapsulates the training and evaluation logic for the PALM model, managing the 
       entire training loop, including optimization, gradient accumulation, and 
       logging of metrics.'''
    
    def __init__(self, model, train_dataloader, eval_dataloader, config):
        self.model = model # Store model
        self.train_dataloader = train_dataloader # Store training data loader
        self.eval_dataloader = eval_dataloader # Store evaluation data loader
        self.config = config # Store configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine whether to use GPU or CPU
        self.model.to(self.device) # Move model to the selected device
        
        # Initialize optimizer with AdamW, including differential learning rates and weight decay
        optimizer = torch.optim.AdamW([
            {"params": pretrained_params, "lr": PRETRAINED_LR, "weight_decay": 0.01},
            {"params": custom_params, "lr": CUSTOM_LR, "weight_decay": 0.09},
        ])
        
        # Set up learning rate scheduler with a linear warm-up and total training steps
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=config.warmup_steps,
        #     num_training_steps=len(self.train_dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
        # )
        total_steps = len(self.train_dataloader) * NUM_TRAIN_EPOCHS // GRADIENT_ACCUMULATION_STEPS
        self.scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps, # Total number of steps for annealing
            eta_min=1e-4 # Minimum learning rate at the end of annealing
            )
        # Initialize global step counter to zero
        start_time = time.time()
        self.global_step = 0
        
    # Define training process over multiple epochs
    def train(self):
        # Loop through each epoch
        for epoch in range(self.config.num_train_epochs):
            # Train model for one epoch
            self.train_epoch(epoch)
            # Evaluate model after each epoch
            self.evaluate()
            
    # Define training process for one epoch
    def train_epoch(self, epoch):
        self.model.train() # Set model to training mode
        total_loss = 0 # Initialize total loss for the epoch
        total_correct = 0  # Initialize the total number of correct predictions
        total_predictions = 0  # Initialize the total number of predictions
        start_time = time.time() # Record start time for the epoch

        # Early stopping variables
        best_eval_loss = float("inf")
        no_improvement_steps = 0
        patience = 3

        # Freeze schedule: Start with all pretrained layers frozen except final layer
        # then unfreeze more layers halfway through
        NUM_LAYERS = real_model.config.num_hidden_layers
        FREEZE_SCHEDULE = [
            {"epoch": 0, "freeze_embeddings": True,  "freeze_up_to_layer_idx": (NUM_LAYERS // 2)},
            {"epoch": 1, "freeze_embeddings": True,  "freeze_up_to_layer_idx": (NUM_LAYERS // 4)},
            {"epoch": 2, "freeze_embeddings": False, "freeze_up_to_layer_idx": 0},
        ]
        # Initialize with the first schedule stage
        initial_schedule = FREEZE_SCHEDULE[0]
        freeze_selected_layers(model, freeze_embeddings=initial_schedule["freeze_embeddings"],
                               freeze_up_to_layer_idx=initial_schedule["freeze_up_to_layer_idx"])

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
            model.train()
            
            # Loop through each batch of data in the training data loader
            for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")):
                print(f"Processing batch {step+1}")
                try:
                    print("Moving batch to device")
                    # Move input IDs, attention mask, labels, and source lengths to the device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    source_len = batch["source_len"].to(self.device)

                    print("Batch moved to device")
                    print("f"Shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")

                    print("Memory usage before forward pass:")
                    for i in range(torch.cuda.device_count()):
                        print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB / {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

                    print("Starting forward pass")
                    device_type_str = "cuda" if device.type == "cuda" else "cpu"
                    with torch.amp.autocast(device_type=device_type_str, enabled=(device.type == "cuda")):
                    
                        # Perform a forward pass to compute logits and losses
                        lm_logits, combined_loss, loss, sae_loss = self.model(
                            input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels, 
                            source_len=source_len
                        )

                        combined_loss = loss + sae_loss if (loss is not None and sae_loss is not None) else None

                    # Compute accuracy
                    preds = lm_logits.argmax(dim=-1)  # Get the index of the highest logit for each token
                    correct = (preds == labels).float() * attention_mask  # Compare predictions to labels
                    total_correct += correct.sum().item()  # Sum the correct predictions
                    total_predictions += attention_mask.sum().item()  # Sum the number of tokens predicted
                    # Calculate accuracy
                    accuracy = total_correct / total_predictions
                    
                    if combined_loss is not None:
                        has_fp32_params = any(p.dtype == torch.float32 for p in model.parameters())
                        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and has_fp32_params))
                        # Scale combined loss for gradient accumulation
                        combined_loss = combined_loss / self.config.gradient_accumulation_steps
                        print("Starting backward pass")
                        # Added for mixed precision
                        scaler.scale(combined_loss).backward()
                        print("Backward pass complete")
                        print("Computing total loss")
                        total_loss += loss.item()
                        total_sae_loss += sae_loss.item() if sae_loss is not None else 0
                        print("Total loss complete:", total_loss)
                        print("SAE loss complete:", total_sae_loss)
                    
                    # Update model parameters if the step is at the accumulation point or the last step
                    if (step + 1) % self.config.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0) # Clip gradients to avoid exploding gradients
                        scaler.unscale_(self.optimizer) # Added for mixed precision
                        scaler.step(self.optimizer) # Update the model parameters
                        scaler.update()
                        self.optimizer.zero_grad() # Zero the gradients for the next step
                        self.scheduler.step() # Update the learning rate
                        self.global_step += 1 # Increment the global step counter
                 
                        # Log metrics at specified intervals
                        if step % self.config.logging_steps == 0:
                            self.log_metrics(accuracy, train_loss, sae_loss, combined_loss, learning_rate, start_time) # Log the metrics
                            start_time = time.time() # Reset the start time for the next logging interval
                
                except Exception as e:
                    logger.error(f"Error in training loop: {str(e)}")
                    logger.error(f"Batch contents: {batch}")
                    raise                                                    
                
    def evaluate(self):
        # Set model to evaluation mode
        self.model.eval()
        eval_loss = 0 # Initialize to zero
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Loop through each batch in the evaluation data loader
            for eval_batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move evaluation input IDs, attention mask, labels, and source lengths to the device
                eval_input_ids = eval_batch["input_ids"].to(self.device)
                eval_attention_mask = eval_batch["attention_mask"].to(self.device)
                eval_labels = eval_batch["labels"].to(self.device)
                eval_source_len = eval_batch["source_len"].to(self.device)
                
                # Perform a forward pass to compute the evaluation outputs
                eval_outputs = self.model(
                    eval_input_ids, 
                    attention_mask=eval_attention_mask,
                    labels=eval_labels, 
                    source_len=eval_source_len
                )
                # Accumulate the evaluation loss
                eval_loss += eval_outputs[1].item()
        
        # Calculate average evaluation loss
        avg_eval_loss = eval_loss / len(self.eval_dataloader)

        # Calculate perplexity from the average evaluation loss
        perplexity = torch.exp(torch.tensor(avg_eval_loss))
        
        # Log evaluation loss and perplexity to Weights & Biases
        wandb.log({
            "eval_loss": avg_eval_loss,
            "perplexity": perplexity,
            "global_step": self.global_step,
        })
        # Log evaluation results to the console
        logger.info(f"Evaluation - Step {self.global_step}, Eval Loss: {avg_eval_loss}, Perplexity: {perplexity}")
        
    def log_metrics(self, loss, sae_loss, combined_loss, start_time, accuracy):
        # Calculate throughput
        samples_per_second = self.config.train_batch_size / (time.time() - start_time)
        
        # Log training metrics to Weights & Biases
        wandb.log({
            "train_loss": loss.item(), # Log training loss
            "sae_loss": sae_loss.item() if sae_loss is not None else 0, # Log SAE loss
            "combined_loss": combined_loss.item(), # Log combined loss
            "learning_rate": self.scheduler.get_last_lr()[0], # Log current learning rate
            "global_step": self.global_step, # Log global step
            "samples_per_second": samples_per_second, # Log throughput
            "accuracy": accuracy,  # Log the accuracy
        })
        
        if torch.cuda.is_available(): # If running on a GPU
            wandb.log({"gpu_memory": torch.cuda.max_memory_allocated() / 1e9}) # Log maximum GPU memory used
        
        # Log metrics to the console for real-time monitoring
        logger.info(f"Step {self.global_step}, Loss: {loss.item()}, SAE Loss: {sae_loss.item()}, Combined Loss: {combined_loss.item()}")
