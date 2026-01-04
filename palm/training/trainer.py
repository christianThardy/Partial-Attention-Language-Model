import time
import logging

import torch
from torch.optim import AdamW

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
        
        # Initialize optimizer with AdamW, including learning rate and weight decay
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.05)
        
        # Set up learning rate scheduler with a linear warm-up and total training steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=len(train_dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
        )
        # Initialize global step counter to zero.
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
        
        # Loop through each batch of data in the training data loader, with a progress bar
        for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")):
            try:
                # Move input IDs, attention mask, labels, and source lengths to the device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                source_len = batch["source_len"].to(self.device)
                
                # Perform a forward pass to compute logits and losses
                # Weird, does not seem like lm_logits is being used downstream
                lm_logits, combined_loss, loss, sae_loss = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels, 
                    source_len=source_len
                )

                # Compute accuracy only on non-masked positions (labels != -100)
                preds = lm_logits.argmax(dim=-1)  # Get the index of the highest logit for each token
                label_mask = (labels != -100)  # Only count positions that contribute to loss
                correct = (preds == labels).float() * label_mask.float()  # Compare predictions to labels
                total_correct += correct.sum().item()  # Sum the correct predictions
                total_predictions += label_mask.sum().item()  # Sum the number of tokens predicted
                
                # Calculate and log accuracy (avoid division by zero)
                accuracy = total_correct / max(total_predictions, 1)  # Compute the accuracy

                # Scale combined loss for gradient accumulation
                combined_loss = combined_loss / self.config.gradient_accumulation_steps
                combined_loss.backward()
                
                # Accumulate total loss for logging
                total_loss += loss.item()
                
                # Update model parameters if the step is at the accumulation point or the last step
                if (step + 1) % self.config.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Clip gradients to avoid exploding gradients
                    self.optimizer.step() # Update the model parameters
                    self.scheduler.step() # Update the learning rate
                    self.optimizer.zero_grad() # Zero the gradients for the next step
                    self.global_step += 1 # Increment the global step counter
                 
                # Log metrics at specified intervals
                if step % self.config.logging_steps == 0:
                    self.log_metrics(loss, sae_loss, combined_loss, start_time, accuracy) # Log the metrics
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
