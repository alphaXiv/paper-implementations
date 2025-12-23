import os
import argparse
import pandas as pd
from datasets import load_dataset
import sys

# Ensure we can import from the package
try:
    from justrl_reproduction.data.prompt_utils import apply_chat_template
except ImportError:
    # Fallback for running script directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from justrl_reproduction.data.prompt_utils import apply_chat_template

def process_dapo_dataset(dataset_name, output_dir):
    """
    Loads and formats the DAPO-Math-17k dataset for GRPO training.
    
    Args:
        dataset_name (str): HuggingFace dataset identifier.
        output_dir (str): Directory to save the processed parquet file.
    """
    print(f"Downloading/Loading dataset: {dataset_name}")
    
    try:
        # Attempt to load the dataset
        # Note: If DAPO-Math-17k is not public or has a different name, 
        # this will fail. In a real reproduction, ensure the name is correct.
        # For now, we assume standard HF loading works.
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Failed to load {dataset_name} from HuggingFace: {e}")
        print("Please ensure you have access or the dataset exists.")
        return

    data_list = []
    print(f"Processing {len(dataset)} examples...")

    for i, example in enumerate(dataset):
        # Attempt to find question and answer fields
        # DAPO-Math-17k likely uses 'question' and 'answer'
        question = example.get('question') or example.get('problem') or example.get('input')
        ground_truth = example.get('answer') or example.get('solution') or example.get('output')

        if not question or not ground_truth:
            # Skip malformed examples
            continue

        # Apply the specific prompt suffix defined in the recipe
        prompt = apply_chat_template(question)

        # Structure for the Trainer
        # We store 'prompt' for generation and 'ground_truth' for the verifier.
        data_list.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "id": f"dapo_{i}"
        })

    df = pd.DataFrame(data_list)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dapo_math_17k.parquet")
    
    df.to_parquet(output_file, index=False)
    print(f"Successfully saved {len(df)} processed examples to {output_file}")
    print(f"Sample Prompt: {data_list[0]['prompt'][:100]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DAPO-Math-17k dataset")
    parser.add_argument("--dataset_name", type=str, default="DAPO-Math-17k", help="HuggingFace dataset path")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed parquet")
    args = parser.parse_args()

    process_dapo_dataset(args.dataset_name, args.output_dir)
