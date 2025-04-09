import torch
import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    pipeline, 
    set_seed,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig, 
    get_peft_model
)
from trl import SFTTrainer, SFTConfig
import wandb

# Constants
PROJECT_NAME = "semeval2025"
RUN_NAME = "llama3-fine-tuning"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
HF_TOKEN = "huggingface_token"
OUTPUT_DIR = "llama3-fine-tuning"

# Initialize wandb
wandb.init(
    project=PROJECT_NAME,
    name=RUN_NAME,
)

import warnings
warnings.filterwarnings("ignore")

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TOKEN = HF_TOKEN

# Load data
try:
    dftrain = pd.read_csv("traindata.tsv", sep="\t")
    dftest = pd.read_csv("developmentdata.tsv", sep="\t")
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    raise

# Convert DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(dftrain)
test_dataset = Dataset.from_pandas(dftest)


dataset = DatasetDict({
    'train': train_dataset,
    'dev': test_dataset
})


def format_instruction(article: str, explanation: str, narrative: str, subnarrative: str, intent: str) -> str:
    """Format the instruction prompt for the model.
    
    Args:
        article: The article text
        explanation: The explanation text
        narrative: The narrative category
        subnarrative: The subnarrative category 
        intent: The intent category
        
    Returns:
        Formatted instruction string
    """
    theme = subnarrative if subnarrative != "None" else narrative
    
    return f"""
### Article:
{article}

### Theme:
{theme}

### Intent:
{intent}

### Explanation:
"""


def generate_instruction_dataset(data_point):
    return {
        "article": data_point["text"],
        "highlights": data_point["explanation"],
        "prompt": format_instruction(data_point["text"],data_point["explanation"], data_point["narrative"], data_point["subnarrative"], data_point["intent"])
    }

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_instruction_dataset).remove_columns(['file'])
    )

dataset["train"] = process_dataset(dataset["train"])
dataset["dev"] = process_dataset(dataset["dev"])

# Select 1000 rows from the training split
train_data = dataset['train']
# Select 100 rows from the test and validation splits
dev_data = dataset['dev']


# Initialize model
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto",
        token=HF_TOKEN
    ).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def print_trainable_parameters(model) -> None:
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )



model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
print(model)


lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.00,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, lora_config)
print_trainable_parameters(model)


training_config = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=4e-5,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=4,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="wandb",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
    dataset_text_field="text",
    max_seq_length=2048,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=dev_data,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_config,
)

trainer.train()

wandb.finish()

peft_model_path="PATH_TO_SAVE_MODEL"

trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)