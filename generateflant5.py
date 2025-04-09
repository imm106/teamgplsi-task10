import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import warnings

# Constants
MODEL_CHECKPOINT = "flant5-fine-tuning/bestmodel"
TEST_DATA_PATH = "testdata.tsv"
DEV_DATA_PATH = "validationdata.tsv"
OUTPUT_PATH = "outputdata.txt"

# Generation parameters
MAX_LENGTH = 120
NUM_SEQUENCES = 1
TOP_P = 0.7
TEMPERATURE = 0.9

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_device():
    """Set up and return the appropriate device."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    return device

def load_model_and_tokenizer(checkpoint_path, device):
    """Load the model and tokenizer from checkpoint."""
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer

def format_prompt(narrative, text, intent):
    """Format the generation prompt."""
    return f"""
    ### Article:
    {text}

    ### Theme:
    {narrative}

    ### Intent:
    {intent}

    ### Explanation:
    """

def generate_explanations(df, model, tokenizer, device):
    """Generate explanations for the given dataframe."""
    ids = []
    outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        ids.append(row['file'])
        
        prompt = format_prompt(row['narrative'], row['text'], row['intent'])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        output = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_return_sequences=NUM_SEQUENCES,
            top_p=TOP_P,
            temperature=TEMPERATURE
        )
        
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        outputs.append(answer)

    return ids, outputs

def save_results(ids, outputs, output_path):
    """Save the generated results to a file."""
    df = pd.DataFrame({'article_id': ids, 'explanation': outputs})
    df.to_csv(output_path, sep='\t', index=False, header=False)

def main():
    # Setup
    device = setup_device()
    model, tokenizer = load_model_and_tokenizer(MODEL_CHECKPOINT, device)
    
    # Load data
    dftest = pd.read_csv(TEST_DATA_PATH, sep="\t")
    
    # Generate explanations
    ids, outputs = generate_explanations(dftest, model, tokenizer, device)
    
    # Save results
    save_results(ids, outputs, OUTPUT_PATH)

if __name__ == "__main__":
    main()
