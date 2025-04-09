from transformers import TextStreamer, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import AutoPeftModelForCausalLM
import pandas as pd
import warnings
from datasets import Dataset, DatasetDict
import torch
from tqdm import tqdm

# Constants
TOKEN = "huggingface_token"
MODEL_DIR = "modeldir"
TEST_PATH = "validationdataset.tsv"
SUBMISSION_PATH = "testdataset.tsv"
OUTPUT_PATH = "validationoutputs.txt"
SUBMISSION_OUTPUT = "testoutputs.txt"

# Generation parameters
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.6
TOP_K = 70
TOP_P = 0.75

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_device():
    """Set up and return the appropriate device."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    return device

def load_datasets(test_path, dev_path, submission_path):
    """Load and prepare all datasets."""
    dftest = pd.read_csv(test_path, sep="\t")
    dfdev = pd.read_csv(dev_path, sep="\t")
    dfsubmission = pd.read_csv(submission_path, sep="\t")
    
    return {
        'test': Dataset.from_pandas(dftest),
        'submission': Dataset.from_pandas(dfsubmission)
    }

def format_instruction(article: str, narrative: str, subnarrative: str, intent: str) -> str:
    """Format the instruction prompt for the model."""
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

def extract_explanation(text: str) -> str:
    """Extract explanation from generated text."""
    explanation_keyword = "Explanation:"
    start_index = text.find(explanation_keyword)
    
    if start_index == -1:
        return "No explanation found."
    
    return text[start_index + len(explanation_keyword):].strip()

def extract_text_until_first_hashes(text: str) -> str:
    """Extract text until the first ### marker."""
    delimiter = "###"
    end_index = text.find(delimiter)
    return text[:end_index].strip() if end_index != -1 else text.strip()

class StopAtSecondNewDot(StoppingCriteria):
    """Stopping criteria for text generation."""
    def __init__(self, tokenizer, initial_text):
        self.tokenizer = tokenizer
        self.initial_dot_count = initial_text.count(".")
    
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        total_dot_count = text.count(".")
        new_dots = total_dot_count - self.initial_dot_count
        return new_dots >= 2

def generate_explanations(data, model, tokenizer, device):
    """Generate explanations for a dataset."""
    articles = []
    explanations = []

    for item in tqdm(data, total=len(data)):
        articles.append(item["article"])
        prompt = item["prompt"]
        
        input_ids = tokenizer(
            prompt, 
            return_tensors='pt',
            truncation=True, 
            add_special_tokens=False
        ).input_ids.to(device)
        
        stopping_criteria = StoppingCriteriaList([StopAtSecondNewDot(tokenizer, prompt)])
        
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            stopping_criteria=stopping_criteria,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P
        )
        
        output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        explanation = extract_explanation(output)
        explanation = extract_text_until_first_hashes(explanation)
        explanations.append(explanation)

    return articles, explanations

def save_results(articles, explanations, output_path):
    """Save results to file."""
    df = pd.DataFrame({'article_id': articles, 'explanation': explanations})
    df.to_csv(output_path, sep='\t', index=False, header=False)

def main():
    # Setup
    device = setup_device()
    
    # Load datasets
    datasets = load_datasets(TEST_PATH, DEV_PATH, SUBMISSION_PATH)
    
    # Load model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_DIR,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Generate explanations for test data
    articles, explanations = generate_explanations(datasets['test'], model, tokenizer, device)
    save_results(articles, explanations, OUTPUT_PATH)
    
    # Generate explanations for submission data
    sub_articles, sub_explanations = generate_explanations(datasets['submission'], model, tokenizer, device)
    save_results(sub_articles, sub_explanations, SUBMISSION_OUTPUT)

if __name__ == "__main__":
    main()



