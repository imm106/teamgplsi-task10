import torch
import pandas as pd
import numpy as np
import nltk
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import wandb

# Constants
PROJECT_NAME = "semeval2025"
RUN_NAME = "flan-t5-fine-tuning"
MODEL_NAME = "google/flan-t5-large"
OUTPUT_DIR = "flant5-fine-tuning"

# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH = 2
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 3
NUM_EPOCHS = 5
MAX_INPUT_LENGTH = 2048
MAX_TARGET_LENGTH = 120

# Initialize wandb
wandb.init(
    project=PROJECT_NAME,
    name=RUN_NAME,
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def setup_device():
    """Set up and return the appropriate device."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    return device

def load_model_and_tokenizer(model_name):
    """Initialize and return the model, tokenizer and data collator."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    return model, tokenizer, data_collator

def load_datasets(train_path, dev_path):
    """Load and prepare the datasets."""
    try:
        dftrain = pd.read_csv(train_path, sep="\t")
        dftest = pd.read_csv(dev_path, sep="\t")
        
        return DatasetDict({
            'train': Dataset.from_pandas(dftrain),
            'dev': Dataset.from_pandas(dftest)
        })
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        raise

def preprocess_function(examples, tokenizer):
    """Preprocess the dataset examples."""
    inputs = [
        f"""      
    ### Article:
    {text}

    ### Theme:
    {narrative}

    ### Intent:
    {intent}

    ### Explanation:"""
        for text, narrative, intent in zip(examples['text'], examples['narrative'], examples['intent'])
    ]
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(examples["explanation"], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer):
    """Compute ROUGE metrics for evaluation."""
    metric = evaluate.load("rouge")
    preds, labels = eval_preds
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    return metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

def main():
    # Setup
    device = setup_device()
    model, tokenizer, data_collator = load_model_and_tokenizer(MODEL_NAME)
    nltk.download("punkt", quiet=True)
    
    # Load datasets
    dataset = load_datasets(
        "traindataset.tsv",
        "validationdataset.tsv"
    )
    
    # Preprocess datasets
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        push_to_hub=False
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )
    
    # Train and save
    trainer.train()
    model.save_pretrained(f"{OUTPUT_DIR}/bestmodel")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/bestmodel")

if __name__ == "__main__":
    main()


