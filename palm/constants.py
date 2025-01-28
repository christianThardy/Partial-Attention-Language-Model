HF_TOKEN = ""
HF_REPO_ID = ""
OUTPUT_DIR = ""
PUSH_TO_HUB = True

MODEL_NAME = "meta-llama/Llama-3.2-3B"
DATASET_1_NAME = "ola13/small-the_pile"
DS1_MAX_SAMPLES = 
DATASET_2_NAME = ""
DS2_MAX_SAMPLES = 

TRAIN_BATCH_SIZE = 16 # Number of samples processed before the model's internal parameters are updated during training
EVAL_BATCH_SIZE = 16 # Number of samples processed during evaluation to calculate metrics
GRADIENT_ACCUMULATION_STEPS = 8 # Number of steps for which gradients are accumulated before performing a backward/update pass
GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 2.0

PRETRAINED_LEARNING_RATE = 4e-5 # Step size at each iteration while moving toward a minimum of the loss function for the base model
PALLM_LEARNING_RATE = 6.2e-5 # Same, but for the new mechanisms
NUM_TRAIN_EPOCHS = 4 
WARMUP_STEPS = 700 # Number of steps for gradually increasing the learning rate from 0 to the set value
HIDDEN_DROPOUT_PROB = 0.42
ATTN_DROPOUT_PROB = 0.42
NUM_HIDDEN_LAYERS = 28
HIDDEN_SIZE = 3072
NUM_ATTENTION_HEADS = 24
POLY_LR_END = 1e-7
EARLY_STOP_PATIENCE = 3
SAE_WEIGHT = 0.46
SAE_START_WEIGHT = 0.5
SAE_END_WEIGHT = 0.3

MAX_SEQ_LENGTH = 1024 # The maximum sequence length for input data; longer sequences will be truncated
TRAIN_RATIO = 0.8
LOG_EVERY_N_STEPS = 1

USE_FREEZE_IN_CHUNKS = False # Freeze/unfreeze layers chunk-by-chunk over certain steps
USE_DYNAMIC_SAE_WEIGHT = False # Vary the SAE weight across epochs
USE_ALL_LAYERS_SMALL_LR = # Train all layers from the start with smaller LR for base
USE_COSINE_DECAY = True # Learning rate schedule
USE_POLYNOMIAL_DECAY = False # Learning rate schedule
USE_LORA = False # Apply LoRA adapters
USE_QLORA = False # Apply QLoRA (AdaLoRA) adapters
USE_TORCH_COMPILE = True
